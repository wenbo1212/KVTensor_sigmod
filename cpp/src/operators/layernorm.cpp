#include "kvtensor/operators.hpp"
#include "kvtensor/operators_impl.hpp"
#include "kvtensor/context.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/profile.hpp"
#include "math/arithmetic.hpp"
#include <dnnl.hpp>
#include <cstring>
#include <stdexcept>

namespace kvtensor {

namespace {
inline std::vector<float> to_float32_buffer(const std::vector<uint8_t>& data, DType dtype) {
    std::vector<float> out(data.size() / (dtype == DType::INT8 ? sizeof(int8_t)
                                    : (dtype == DType::FLOAT16 || dtype == DType::BFLOAT16)
                                        ? sizeof(uint16_t)
                                        : sizeof(float)));
    {
        ProfileScope scope(ProfileKind::Decompress);
        math::convert_buffer_to_float32(data.data(), dtype, out.size(), out.data());
    }
    return out;
}

inline std::vector<float> load_vector(OperatorContext& ctx, const std::string& id, int64_t cols) {
    if (id.empty()) {
        return std::vector<float>();
    }
    std::shared_ptr<InMemoryMatrix> mem = ctx.resolve_in_memory(id);
    if (!mem) {
        auto bm = ctx.resolve_block_matrix(id);
        if (!bm) {
            throw std::runtime_error("LayerNorm weight not found: " + id);
        }
        auto dense = bm->to_dense(ctx);
        mem = ctx.store_in_memory(id + "_dense", bm->shape(), bm->dtype(), std::move(dense));
    }
    auto [r, c] = mem->shape();
    if (r == 1 && c == cols) {
        return to_float32_buffer(mem->data(), mem->dtype());
    }
    if (r == cols && c == 1) {
        return to_float32_buffer(mem->data(), mem->dtype());
    }
    if (r == 1 && c == 1) {
        return std::vector<float>(static_cast<size_t>(cols),
                                  to_float32_buffer(mem->data(), mem->dtype()).front());
    }
    throw std::runtime_error("LayerNorm weight shape mismatch for " + id);
}
} // namespace

class RowColumnLayerNorm : public Operator {
public:
    RowColumnLayerNorm(
        const std::string& matrix_id,
        const std::string& weight_id,
        const std::string& bias_id,
        const std::string& result_id,
        float eps = 1e-5f,
        bool store_in_memory = true
    ) : matrix_id_(matrix_id),
        weight_id_(weight_id),
        bias_id_(bias_id),
        result_id_(result_id),
        eps_(eps),
        store_in_memory_(store_in_memory) {
        name_ = "rowcolumn_layernorm";
    }

    std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) override {
        std::shared_ptr<InMemoryMatrix> matrix = ctx.resolve_in_memory(matrix_id_);
        if (!matrix) {
            auto bm = ctx.resolve_block_matrix(matrix_id_);
            if (!bm) {
                throw std::runtime_error("LayerNorm input not found: " + matrix_id_);
            }
            auto dense = bm->to_dense(ctx);
            matrix = ctx.store_in_memory(matrix_id_ + "_dense", bm->shape(), bm->dtype(), std::move(dense));
        }

        auto [rows, cols] = matrix->shape();
        DType input_dtype = matrix->dtype();
        std::vector<float> input_f32;
        const void* input_ptr = nullptr;
        if (input_dtype != DType::BFLOAT16) {
            input_f32 = to_float32_buffer(matrix->data(), input_dtype);
        }
        std::vector<float> gamma = load_vector(ctx, weight_id_, cols);
        std::vector<float> beta = load_vector(ctx, bias_id_, cols);
        if (gamma.empty()) {
            gamma.assign(static_cast<size_t>(cols), 1.0f);
        }
        if (beta.empty()) {
            beta.assign(static_cast<size_t>(cols), 0.0f);
        }

        DType out_dtype = input_dtype;
        if (!weight_id_.empty()) {
            if (auto wmem = ctx.resolve_in_memory(weight_id_)) {
                out_dtype = wmem->dtype();
            } else if (auto wbm = ctx.resolve_block_matrix(weight_id_)) {
                out_dtype = wbm->dtype();
            }
        }
        std::vector<uint8_t> out_bytes;
        out_bytes.resize(static_cast<size_t>(rows * cols) * (out_dtype == DType::BFLOAT16 ? sizeof(uint16_t) : sizeof(float)));
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        dnnl::stream stream(eng);

        dnnl::memory::dims data_dims = {rows, cols};
        auto data_dt = (out_dtype == DType::BFLOAT16) ? dnnl::memory::data_type::bf16 : dnnl::memory::data_type::f32;
        auto data_md = dnnl::memory::desc(
            data_dims, data_dt, dnnl::memory::format_tag::ab
        );
        auto scale_md = dnnl::memory::desc(
            {cols}, data_dt, dnnl::memory::format_tag::x
        );

        std::vector<uint16_t> input_bf16;
        if (data_dt == dnnl::memory::data_type::bf16) {
            if (input_dtype == DType::BFLOAT16) {
                input_ptr = matrix->data().data();
            } else {
                input_bf16.resize(input_f32.size());
                {
                    ProfileScope scope(ProfileKind::Decompress);
                    math::convert_float32_to_bf16(input_f32.data(), input_f32.size(), input_bf16.data());
                }
                input_ptr = input_bf16.data();
            }
        } else {
            input_ptr = input_f32.data();
        }
        std::vector<uint16_t> gamma_bf16;
        std::vector<uint16_t> beta_bf16;
        void* gamma_ptr = gamma.data();
        void* beta_ptr = beta.data();
        if (out_dtype == DType::BFLOAT16) {
            gamma_bf16.resize(gamma.size());
            beta_bf16.resize(beta.size());
            {
                ProfileScope scope(ProfileKind::Decompress);
                math::convert_float32_to_bf16(gamma.data(), gamma.size(), gamma_bf16.data());
                math::convert_float32_to_bf16(beta.data(), beta.size(), beta_bf16.data());
            }
            gamma_ptr = gamma_bf16.data();
            beta_ptr = beta_bf16.data();
        }

        auto src_mem = dnnl::memory(data_md, eng, const_cast<void*>(input_ptr));
        auto dst_mem = dnnl::memory(data_md, eng, out_bytes.data());
        auto scale_mem = dnnl::memory(scale_md, eng, gamma_ptr);
        auto shift_mem = dnnl::memory(scale_md, eng, beta_ptr);

        auto flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift;
        auto pd = dnnl::layer_normalization_forward::primitive_desc(
            eng,
            dnnl::prop_kind::forward_inference,
            data_md,
            data_md,
            eps_,
            flags
        );
        auto prim = dnnl::layer_normalization_forward(pd);
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            prim.execute(stream, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_SCALE, scale_mem},
                {DNNL_ARG_SHIFT, shift_mem},
                {DNNL_ARG_DST, dst_mem},
            });
            stream.wait();
        }

        return ctx.store_in_memory(result_id_, matrix->shape(), out_dtype, std::move(out_bytes));
    }

private:
    std::string matrix_id_;
    std::string weight_id_;
    std::string bias_id_;
    std::string result_id_;
    float eps_;
    bool store_in_memory_;
};

std::unique_ptr<Operator> create_row_column_layernorm(
    const std::string& matrix_id,
    const std::string& weight_id,
    const std::string& bias_id,
    const std::string& result_id,
    float eps,
    bool store_in_memory
) {
    return std::make_unique<RowColumnLayerNorm>(matrix_id, weight_id, bias_id, result_id, eps, store_in_memory);
}

} // namespace kvtensor

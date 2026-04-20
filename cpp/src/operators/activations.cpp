#include "kvtensor/operators.hpp"
#include "kvtensor/operators_impl.hpp"
#include "kvtensor/context.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/profile.hpp"
#include "math/arithmetic.hpp"
#include <dnnl.hpp>
#include <cstring>
#include <stdexcept>
#include <cmath>

namespace kvtensor {

// Helper functions
template<typename T>
const float* get_float32_ptr(const std::vector<uint8_t>& data) {
    return reinterpret_cast<const float*>(data.data());
}

template<typename T>
float* get_float32_ptr_mut(std::vector<uint8_t>& data) {
    return reinterpret_cast<float*>(data.data());
}

// RowColumnSiLU operator
class RowColumnSiLU : public Operator {
public:
    RowColumnSiLU(
        const std::string& matrix_id,
        const std::string& result_id,
        bool store_in_memory = true
    ) : matrix_id_(matrix_id),
        result_id_(result_id),
        store_in_memory_(store_in_memory) {
        name_ = "rowcolumn_silu";
    }

    std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) override {
        // Resolve input
        auto matrix = ctx.resolve_in_memory(matrix_id_);
        if (!matrix) {
            auto block = ctx.resolve_block_matrix(matrix_id_);
            if (block) {
                auto dense = block->to_dense(ctx);
                matrix = ctx.store_in_memory(
                    matrix_id_ + "_dense",
                    block->shape(),
                    block->dtype(),
                    std::move(dense)
                );
            } else {
                throw std::runtime_error("Matrix not found: " + matrix_id_);
            }
        }
        
        auto [rows, cols] = matrix->shape();
        DType input_dtype = matrix->dtype();
        
        // Convert to float32 for computation
        int64_t size = rows * cols;
        std::vector<float> input_f32(size);
        {
            ProfileScope scope(ProfileKind::Decompress);
            math::convert_buffer_to_float32(matrix->data().data(), input_dtype, size, input_f32.data());
        }
        
        // Allocate result
        std::vector<uint8_t> result_data;
        float* result_ptr = nullptr;
        std::vector<float> result_f32;
        if (input_dtype == DType::BFLOAT16) {
            result_f32.resize(static_cast<size_t>(rows * cols));
            result_ptr = result_f32.data();
        } else {
            size_t result_size = rows * cols * sizeof(float);
            result_data.resize(result_size);
            result_ptr = get_float32_ptr_mut<float>(result_data);
        }
        
        // Compute SiLU: x * sigmoid(x)
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            math::silu(input_f32.data(), size, result_ptr);
        }
        
        if (input_dtype == DType::BFLOAT16) {
            result_data.resize(result_f32.size() * sizeof(uint16_t));
            auto* out_bf16 = reinterpret_cast<uint16_t*>(result_data.data());
            {
                ProfileScope scope(ProfileKind::Decompress);
                math::convert_float32_to_bf16(result_f32.data(), result_f32.size(), out_bf16);
            }
        }
        auto result = ctx.store_in_memory(
            result_id_,
            matrix->shape(),
            input_dtype == DType::BFLOAT16 ? DType::BFLOAT16 : DType::FLOAT32,
            std::move(result_data)
        );
        
        return result;
    }

private:
    std::string matrix_id_;
    std::string result_id_;
    bool store_in_memory_;
};

class RowColumnGEGLU : public Operator {
public:
    RowColumnGEGLU(
        const std::string& matrix_id,
        const std::string& result_id,
        bool store_in_memory = true
    ) : matrix_id_(matrix_id),
        result_id_(result_id),
        store_in_memory_(store_in_memory) {
        name_ = "rowcolumn_geglu";
    }

    std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) override {
        auto matrix = ctx.resolve_in_memory(matrix_id_);
        if (!matrix) {
            auto block = ctx.resolve_block_matrix(matrix_id_);
            if (block) {
                auto dense = block->to_dense(ctx);
                matrix = ctx.store_in_memory(
                    matrix_id_ + "_dense",
                    block->shape(),
                    block->dtype(),
                    std::move(dense)
                );
            } else {
                throw std::runtime_error("Matrix not found: " + matrix_id_);
            }
        }

        auto [rows, cols] = matrix->shape();
        if (cols % 2 != 0) {
            throw std::runtime_error("GEGLU expects even feature dimension");
        }
        int64_t inner = cols / 2;

        int64_t size = rows * cols;
        std::vector<float> input_f32(size);
        {
            ProfileScope scope(ProfileKind::Decompress);
            math::convert_buffer_to_float32(matrix->data().data(), matrix->dtype(), size, input_f32.data());
        }
        DType out_dtype = matrix->dtype();

        std::vector<float> output(static_cast<size_t>(rows * inner));
        // Use oneDNN GELU on the gate half, then elementwise multiply with up half.
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        dnnl::stream s(eng);

        dnnl::memory::dims src_dims = {rows, inner};
        auto src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
        auto dst_md = src_md;

        std::vector<float> gate(static_cast<size_t>(rows * inner));
        std::vector<float> gelu_gate(static_cast<size_t>(rows * inner));
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            for (int64_t r = 0; r < rows; ++r) {
                const float* row_ptr = input_f32.data() + r * cols;
                float* gate_ptr = gate.data() + r * inner;
                std::memcpy(gate_ptr, row_ptr, static_cast<size_t>(inner) * sizeof(float));
            }
        }

        auto src_mem = dnnl::memory(src_md, eng, gate.data());
        auto dst_mem = dnnl::memory(dst_md, eng, gelu_gate.data());

        auto gelu_pd = dnnl::eltwise_forward::primitive_desc(
            eng,
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_gelu_tanh,
            src_md,
            dst_md
        );
        auto gelu_prim = dnnl::eltwise_forward(gelu_pd);
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            gelu_prim.execute(s, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
            s.wait();
        }

        {
            ProfileScope scope(ProfileKind::OtherCompute);
            for (int64_t r = 0; r < rows; ++r) {
                const float* row_ptr = input_f32.data() + r * cols;
                const float* gate_ptr = gelu_gate.data() + r * inner;
                float* out_ptr = output.data() + r * inner;
                for (int64_t c = 0; c < inner; ++c) {
                    out_ptr[c] = gate_ptr[static_cast<size_t>(c)] * row_ptr[c + inner];
                }
            }
        }

        std::vector<uint8_t> out_bytes;
        if (out_dtype == DType::BFLOAT16) {
            out_bytes.resize(output.size() * sizeof(uint16_t));
            auto* out_bf16 = reinterpret_cast<uint16_t*>(out_bytes.data());
            {
                ProfileScope scope(ProfileKind::Decompress);
                math::convert_float32_to_bf16(output.data(), output.size(), out_bf16);
            }
        } else {
            out_bytes.resize(output.size() * sizeof(float));
            std::memcpy(out_bytes.data(), output.data(), out_bytes.size());
        }
        return ctx.store_in_memory(
            result_id_,
            std::make_tuple(rows, inner),
            out_dtype == DType::BFLOAT16 ? DType::BFLOAT16 : DType::FLOAT32,
            std::move(out_bytes)
        );
    }

private:
    std::string matrix_id_;
    std::string result_id_;
    bool store_in_memory_;
};

// Factory function
std::unique_ptr<Operator> create_row_column_silu(
    const std::string& matrix_id,
    const std::string& result_id,
    bool store_in_memory
) {
    return std::make_unique<RowColumnSiLU>(matrix_id, result_id, store_in_memory);
}

std::unique_ptr<Operator> create_row_column_geglu(
    const std::string& matrix_id,
    const std::string& result_id,
    bool store_in_memory
) {
    return std::make_unique<RowColumnGEGLU>(matrix_id, result_id, store_in_memory);
}

} // namespace kvtensor

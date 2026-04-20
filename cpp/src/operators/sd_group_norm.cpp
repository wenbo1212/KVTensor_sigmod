#include "kvtensor/sd_ops.hpp"
#include "kvtensor/context.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/profile.hpp"
#include "math/arithmetic.hpp"
#include <dnnl.hpp>
#include <dnnl_version.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kvtensor {

namespace {
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return sizeof(float);
        case DType::BFLOAT16: return sizeof(uint16_t);
        case DType::FLOAT16: return sizeof(uint16_t);
        case DType::INT8: return sizeof(int8_t);
        default: return sizeof(float);
    }
}

inline std::vector<float> to_float32_buffer(const std::vector<uint8_t>& data, DType dtype) {
    std::vector<float> out(data.size() / dtype_size(dtype));
    {
        ProfileScope scope(ProfileKind::Decompress);
        math::convert_buffer_to_float32(data.data(), dtype, out.size(), out.data());
    }
    return out;
}

struct VectorView {
    const float* ptr = nullptr;
    std::vector<float> owned;
};

// Read 1 x N vector via chunked BlockMatrix when not already in memory
inline VectorView read_vector_view(
    const std::string& id,
    int64_t expected_elems,
    OperatorContext& ctx
) {
    VectorView view;
    if (auto mem = ctx.resolve_in_memory(id)) {
        if (mem->dtype() == DType::FLOAT32) {
            view.ptr = reinterpret_cast<const float*>(mem->data().data());
            int64_t elems = static_cast<int64_t>(mem->data().size() / sizeof(float));
            if (expected_elems > 0 && elems != expected_elems) {
                throw std::runtime_error("Size mismatch for vector " + id);
            }
            return view;
        }
        view.owned = to_float32_buffer(mem->data(), mem->dtype());
        if (expected_elems > 0 && static_cast<int64_t>(view.owned.size()) != expected_elems) {
            throw std::runtime_error("Size mismatch for vector " + id);
        }
        view.ptr = view.owned.data();
        return view;
    }
    auto bm = ctx.resolve_block_matrix(id);
    if (!bm) {
        throw std::runtime_error("Matrix not found: " + id);
    }
    if (bm->split_mode() != SplitMode::COLUMN) {
        throw std::runtime_error("Expected COLUMN split for vector " + id);
    }
    auto [rows, cols] = bm->shape();
    int64_t total = rows * cols;
    if (expected_elems > 0 && total != expected_elems) {
        throw std::runtime_error("Size mismatch for vector " + id);
    }
    view.owned.resize(static_cast<size_t>(total));
    int64_t num_chunks = bm->num_col_chunks();
    int64_t offset = 0;
    for (int64_t j = 0; j < num_chunks; ++j) {
        auto chunk = bm->read_col_chunk(j, ctx);
        auto [chunk_rows, chunk_cols] = bm->col_chunk_shape(j);
        size_t elems = static_cast<size_t>(chunk_rows * chunk_cols);
        math::convert_buffer_to_float32(chunk.data, bm->dtype(), elems, view.owned.data() + offset);
        offset += static_cast<int64_t>(elems);
    }
    view.ptr = view.owned.data();
    return view;
}

void float_to_dtype_buffer(const std::vector<float>& src, DType dtype, std::vector<uint8_t>& dst) {
    size_t elems = src.size();
    switch (dtype) {
        case DType::FLOAT32: {
            dst.resize(elems * sizeof(float));
            std::memcpy(dst.data(), src.data(), dst.size());
            break;
        }
        case DType::BFLOAT16:
        case DType::FLOAT16: {
            dst.resize(elems * sizeof(uint16_t));
            uint16_t* out = reinterpret_cast<uint16_t*>(dst.data());
            for (size_t i = 0; i < elems; ++i) {
                out[i] = math::float_to_bf16(src[i]);
            }
            break;
        }
        case DType::INT8: {
            dst.resize(elems * sizeof(int8_t));
            int8_t* out = reinterpret_cast<int8_t*>(dst.data());
            for (size_t i = 0; i < elems; ++i) {
                out[i] = static_cast<int8_t>(std::nearbyint(src[i]));
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported output dtype");
    }
}

} // namespace

class GroupNormOp : public Operator {
public:
    explicit GroupNormOp(const GroupNormConfig& config) : config_(config) {
        name_ = "group_norm";
        result_id_ = config.result_id;
    }

    std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) override {
        auto input_mem = ctx.resolve_in_memory(config_.input_id);
        std::vector<float> x_f32;
        std::vector<uint16_t> x_bf16;
        const void* x_ptr = nullptr;
        DType input_dtype = DType::FLOAT32;
        Shape in_shape;
        if (input_mem) {
            in_shape = input_mem->shape();
            input_dtype = input_mem->dtype();
            if (input_dtype == DType::BFLOAT16) {
                x_ptr = input_mem->data().data();
            } else {
                x_f32 = to_float32_buffer(input_mem->data(), input_dtype);
            }
        } else {
            auto block = ctx.resolve_block_matrix(config_.input_id);
            if (!block) {
                throw std::runtime_error("Input matrix not found for group norm: " + config_.input_id);
            }
            in_shape = block->shape();
            input_dtype = block->dtype();
            if (input_dtype == DType::BFLOAT16) {
                auto dense = block->to_dense(ctx);
                x_bf16.resize(dense.size() / sizeof(uint16_t));
                std::memcpy(x_bf16.data(), dense.data(), dense.size());
                x_ptr = x_bf16.data();
            } else {
                x_f32 = to_float32_buffer(block->to_dense(ctx), input_dtype);
            }
        }

        int64_t rows = std::get<0>(in_shape);
        int64_t cols = std::get<1>(in_shape);
        int64_t channels = config_.num_channels > 0 ? config_.num_channels : cols;
        if (channels != cols) {
            throw std::runtime_error("GroupNorm expects input cols == num_channels");
        }
        if (channels % config_.num_groups != 0) {
            throw std::runtime_error("num_channels must be divisible by num_groups");
        }
        int64_t group_size = channels / config_.num_groups;

        VectorView gamma_view;
        VectorView beta_view;
        std::vector<float> gamma_fallback;
        std::vector<float> beta_fallback;
        if (!config_.weight_id.empty()) {
            gamma_view = read_vector_view(config_.weight_id, channels, ctx);
        } else {
            gamma_fallback.assign(static_cast<size_t>(channels), 1.0f);
            gamma_view.ptr = gamma_fallback.data();
        }
        if (!config_.bias_id.empty()) {
            beta_view = read_vector_view(config_.bias_id, channels, ctx);
        } else {
            beta_fallback.assign(static_cast<size_t>(channels), 0.0f);
            beta_view.ptr = beta_fallback.data();
        }

        DType weight_dtype = input_dtype;
        if (!config_.weight_id.empty()) {
            if (auto wmem = ctx.resolve_in_memory(config_.weight_id)) {
                weight_dtype = wmem->dtype();
            } else if (auto wbm = ctx.resolve_block_matrix(config_.weight_id)) {
                weight_dtype = wbm->dtype();
            }
        }
        DType out_dtype = (weight_dtype == DType::BFLOAT16) ? DType::BFLOAT16 : DType::FLOAT32;

        std::vector<uint8_t> out_bytes;
        std::vector<float> y_f32;
        std::vector<uint16_t> y_bf16;
        void* y_ptr = nullptr;
        if (out_dtype == DType::BFLOAT16) {
            out_bytes.resize(static_cast<size_t>(rows * channels) * sizeof(uint16_t));
            y_ptr = out_bytes.data();
        } else {
            out_bytes.resize(static_cast<size_t>(rows * channels) * sizeof(float));
            y_ptr = out_bytes.data();
        }

        math::require_onednn("group_norm");
        auto& engine = math::onednn_engine();
        auto& stream = math::onednn_stream();

        dnnl::memory::dims src_dims = {rows, channels, 1, 1};
        auto src_dt = (out_dtype == DType::BFLOAT16) ? dnnl::memory::data_type::bf16 : dnnl::memory::data_type::f32;
        auto src_md = dnnl::memory::desc(src_dims, src_dt, dnnl::memory::format_tag::nchw);
        auto dst_md = dnnl::memory::desc(src_dims, src_dt, dnnl::memory::format_tag::nchw);

        dnnl::normalization_flags flags = dnnl::normalization_flags::none;
#if defined(DNNL_ARG_SCALE_SHIFT)
        flags |= dnnl::normalization_flags::use_scale_shift;
        dnnl::memory::dims scale_shift_dims = {2, channels};
        auto scale_dt = (out_dtype == DType::BFLOAT16) ? dnnl::memory::data_type::bf16 : dnnl::memory::data_type::f32;
        auto scale_shift_md = dnnl::memory::desc(
            scale_shift_dims, scale_dt, dnnl::memory::format_tag::nc
        );
        std::vector<uint16_t> scale_shift_bf16;
        std::vector<float> scale_shift(static_cast<size_t>(2 * channels));
        for (int64_t c = 0; c < channels; ++c) {
            scale_shift[static_cast<size_t>(c)] = gamma_view.ptr[static_cast<size_t>(c)];
            scale_shift[static_cast<size_t>(channels + c)] = beta_view.ptr[static_cast<size_t>(c)];
        }
        if (out_dtype == DType::BFLOAT16) {
            scale_shift_bf16.resize(scale_shift.size());
            {
                ProfileScope scope(ProfileKind::Decompress);
                math::convert_float32_to_bf16(scale_shift.data(), scale_shift.size(), scale_shift_bf16.data());
            }
        }
#else
        flags |= dnnl::normalization_flags::use_scale;
        flags |= dnnl::normalization_flags::use_shift;
        dnnl::memory::dims scale_shift_dims = {channels};
        auto scale_dt = (out_dtype == DType::BFLOAT16) ? dnnl::memory::data_type::bf16 : dnnl::memory::data_type::f32;
        auto scale_shift_md = dnnl::memory::desc(
            scale_shift_dims, scale_dt, dnnl::memory::format_tag::x
        );
        std::vector<float> scale_vec(static_cast<size_t>(channels));
        std::vector<float> shift_vec(static_cast<size_t>(channels));
        std::vector<uint16_t> scale_bf16;
        std::vector<uint16_t> shift_bf16;
        for (int64_t c = 0; c < channels; ++c) {
            scale_vec[static_cast<size_t>(c)] = gamma_view.ptr[static_cast<size_t>(c)];
            shift_vec[static_cast<size_t>(c)] = beta_view.ptr[static_cast<size_t>(c)];
        }
        if (out_dtype == DType::BFLOAT16) {
            scale_bf16.resize(scale_vec.size());
            shift_bf16.resize(shift_vec.size());
            {
                ProfileScope scope(ProfileKind::Decompress);
                math::convert_float32_to_bf16(scale_vec.data(), scale_vec.size(), scale_bf16.data());
                math::convert_float32_to_bf16(shift_vec.data(), shift_vec.size(), shift_bf16.data());
            }
        }
#endif

        dnnl::group_normalization_forward::primitive_desc pd;
#if defined(DNNL_VERSION_MAJOR) && (DNNL_VERSION_MAJOR >= 3)
        pd = dnnl::group_normalization_forward::primitive_desc(
            engine,
            dnnl::prop_kind::forward_inference,
            src_md,
            dst_md,
            config_.num_groups,
            config_.eps,
            flags,
            dnnl::primitive_attr(),
            false
        );
#else
        auto desc = dnnl::group_normalization_forward::desc(
            dnnl::prop_kind::forward_inference,
            src_md,
            dst_md,
            config_.num_groups,
            config_.eps,
            flags
        );
        pd = dnnl::group_normalization_forward::primitive_desc(desc, engine);
#endif
        if (src_dt == dnnl::memory::data_type::bf16) {
            if (x_ptr == nullptr) {
                x_bf16.resize(x_f32.size());
                {
                    ProfileScope scope(ProfileKind::Decompress);
                    math::convert_float32_to_bf16(x_f32.data(), x_f32.size(), x_bf16.data());
                }
                x_ptr = x_bf16.data();
            }
        } else {
            if (x_ptr == nullptr) {
                x_ptr = x_f32.data();
            } else if (input_dtype == DType::BFLOAT16) {
                // Input is bf16 but output is f32, convert.
                size_t elems = static_cast<size_t>(rows * channels);
                x_f32.resize(elems);
                {
                    ProfileScope scope(ProfileKind::Decompress);
                    math::convert_buffer_to_float32(
                        reinterpret_cast<const uint8_t*>(x_ptr),
                        DType::BFLOAT16,
                        elems,
                        x_f32.data()
                    );
                }
                x_ptr = x_f32.data();
            }
        }
        auto src_mem = dnnl::memory(src_md, engine, const_cast<void*>(x_ptr));
        auto dst_mem = dnnl::memory(dst_md, engine, y_ptr);
        dnnl::group_normalization_forward op(pd);
#if defined(DNNL_ARG_SCALE_SHIFT)
        auto ss_ptr = (out_dtype == DType::BFLOAT16) ? static_cast<void*>(scale_shift_bf16.data())
                                                     : static_cast<void*>(scale_shift.data());
        auto ss_mem = dnnl::memory(scale_shift_md, engine, ss_ptr);
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            op.execute(stream, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_DST, dst_mem},
                {DNNL_ARG_SCALE_SHIFT, ss_mem}
            });
            stream.wait();
        }
#else
        void* scale_ptr = (out_dtype == DType::BFLOAT16) ? static_cast<void*>(scale_bf16.data())
                                                         : static_cast<void*>(scale_vec.data());
        void* shift_ptr = (out_dtype == DType::BFLOAT16) ? static_cast<void*>(shift_bf16.data())
                                                         : static_cast<void*>(shift_vec.data());
        auto scale_mem = dnnl::memory(scale_shift_md, engine, scale_ptr);
        auto shift_mem = dnnl::memory(scale_shift_md, engine, shift_ptr);
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            op.execute(stream, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_DST, dst_mem},
                {DNNL_ARG_SCALE, scale_mem},
                {DNNL_ARG_SHIFT, shift_mem}
            });
            stream.wait();
        }
#endif

        Shape out_shape = config_.output_shape;
        if (std::get<0>(out_shape) == 0 || std::get<1>(out_shape) == 0) {
            out_shape = in_shape;
        }

        return ctx.store_in_memory(result_id_, out_shape, out_dtype, std::move(out_bytes));
    }

private:
    GroupNormConfig config_;
};

std::unique_ptr<Operator> create_group_norm(const GroupNormConfig& config) {
    return std::make_unique<GroupNormOp>(config);
}

} // namespace kvtensor

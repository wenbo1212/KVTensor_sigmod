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

} // namespace

class Im2ColConv2DOp : public Operator {
public:
    explicit Im2ColConv2DOp(const Conv2DIm2ColConfig& config) : config_(config) {
        name_ = "onednn_conv2d";
        result_id_ = config.result_id;
    }

    std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) override {
        // Resolve input (activations may be materialized)
        auto input_mem = ctx.resolve_in_memory(config_.input_id);
        std::vector<float> input_f32;
        std::vector<uint16_t> input_bf16;
        const void* input_ptr = nullptr;
        DType input_dtype = DType::FLOAT32;
        Shape input_shape;

        if (input_mem) {
            input_shape = input_mem->shape();
            input_dtype = input_mem->dtype();
            if (input_dtype == DType::BFLOAT16) {
                input_ptr = input_mem->data().data();
            } else {
                input_f32 = to_float32_buffer(input_mem->data(), input_dtype);
            }
        } else {
            auto block = ctx.resolve_block_matrix(config_.input_id);
            if (!block) {
                throw std::runtime_error("Input matrix not found: " + config_.input_id);
            }
            input_shape = block->shape();
            input_dtype = block->dtype();
            if (input_dtype == DType::BFLOAT16) {
                auto dense = block->to_dense(ctx);
                input_bf16.resize(dense.size() / sizeof(uint16_t));
                std::memcpy(input_bf16.data(), dense.data(), dense.size());
                input_ptr = input_bf16.data();
            } else {
                input_f32 = to_float32_buffer(block->to_dense(ctx), input_dtype);
            }
        }

        int64_t in_c = config_.input_channels;
        int64_t in_h = config_.input_height;
        int64_t in_w = config_.input_width;

        if (in_c <= 0 || in_h <= 0 || in_w <= 0) {
            auto [rows, cols] = input_shape;
            if (in_c <= 0) {
                throw std::runtime_error("input_channels must be set for oneDNN conv");
            }
            int64_t spatial = rows * cols / in_c;
            in_h = static_cast<int64_t>(std::sqrt(static_cast<double>(spatial)));
            if (in_h * in_h == spatial) {
                in_w = in_h;
            } else {
                in_w = spatial / in_h;
            }
        }

        // Resolve weights (must already be in memory)
        auto weight_mem = ctx.resolve_in_memory(config_.weight_id);
        if (!weight_mem) {
            throw std::runtime_error("oneDNN conv requires weights in memory: " + config_.weight_id);
        }

        DType weight_dtype = weight_mem->dtype();
        if (weight_dtype != DType::FLOAT32 && weight_dtype != DType::BFLOAT16) {
            throw std::runtime_error("Unsupported weight dtype for oneDNN conv");
        }

        auto [w_rows, w_cols] = weight_mem->shape(); // (OC, IC*KH*KW)
        int64_t out_c = config_.output_channels > 0 ? config_.output_channels : w_rows;
        int64_t k_h = config_.kernel_h;
        int64_t k_w = config_.kernel_w;
        int64_t expected_cols = in_c * k_h * k_w;
        if (w_cols != expected_cols) {
            throw std::runtime_error("Weight shape mismatch for oneDNN conv");
        }

        int64_t dil_h = config_.dilation_h;
        int64_t dil_w = config_.dilation_w;
        if (dil_h <= 0 || dil_w <= 0) {
            throw std::runtime_error("Dilation must be >= 1 (PyTorch-style)");
        }
        int64_t dnnl_dil_h = dil_h - 1;
        int64_t dnnl_dil_w = dil_w - 1;

        int64_t out_h = config_.output_height;
        int64_t out_w = config_.output_width;
        if (out_h <= 0 || out_w <= 0) {
            int64_t eff_kh = (k_h - 1) * dil_h + 1;
            int64_t eff_kw = (k_w - 1) * dil_w + 1;
            out_h = (in_h + 2 * config_.pad_h - eff_kh) / config_.stride_h + 1;
            out_w = (in_w + 2 * config_.pad_w - eff_kw) / config_.stride_w + 1;
        }

        // Bias (optional): accept in-memory or block-matrix and coerce to bf16.
        std::vector<uint8_t> bias_storage;
        std::vector<float> bias_f32;
        std::vector<uint16_t> bias_bf16;
        const void* bias_ptr = nullptr;
        if (!config_.bias_id.empty()) {
            auto bias_mem = ctx.resolve_in_memory(config_.bias_id);
            if (bias_mem) {
                if (bias_mem->dtype() == DType::BFLOAT16) {
                    bias_ptr = bias_mem->data().data();
                } else {
                    bias_f32 = to_float32_buffer(bias_mem->data(), bias_mem->dtype());
                }
            } else {
                auto bias_block = ctx.resolve_block_matrix(config_.bias_id);
                if (!bias_block) {
                    throw std::runtime_error("Bias matrix not found: " + config_.bias_id);
                }
                if (bias_block->dtype() == DType::BFLOAT16) {
                    bias_storage = bias_block->to_dense(ctx);
                    bias_ptr = bias_storage.data();
                } else {
                    bias_f32 = to_float32_buffer(bias_block->to_dense(ctx), bias_block->dtype());
                }
            }
            if (bias_ptr == nullptr && !bias_f32.empty()) {
                bias_bf16.resize(bias_f32.size());
                {
                    ProfileScope scope(ProfileKind::Decompress);
                    math::convert_float32_to_bf16(bias_f32.data(), bias_f32.size(), bias_bf16.data());
                }
                bias_ptr = bias_bf16.data();
            }
        }

        // oneDNN setup (NCHW input, OIHW weights, NHWC output)
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        dnnl::stream s(eng);

        dnnl::memory::dims src_dims = {1, in_c, in_h, in_w};
        dnnl::memory::dims weight_dims = {out_c, in_c, k_h, k_w};
        dnnl::memory::dims dst_dims = {1, out_c, out_h, out_w};
        dnnl::memory::dims bias_dims = {out_c};

        dnnl::memory::dims strides = {config_.stride_h, config_.stride_w};
        dnnl::memory::dims dilations = {dnnl_dil_h, dnnl_dil_w};
        dnnl::memory::dims padding_l = {config_.pad_h, config_.pad_w};
        dnnl::memory::dims padding_r = {config_.pad_h, config_.pad_w};

        auto weight_dt = (weight_dtype == DType::BFLOAT16) ? dnnl::memory::data_type::bf16 : dnnl::memory::data_type::f32;
        DType out_dtype = (weight_dtype == DType::BFLOAT16) ? DType::BFLOAT16 : DType::FLOAT32;
        auto dst_dt = (out_dtype == DType::BFLOAT16) ? dnnl::memory::data_type::bf16 : dnnl::memory::data_type::f32;

        // User layouts (NCHW input, OIHW weights) and final NHWC output.
        auto src_md_user = dnnl::memory::desc(src_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::nchw);
        auto weight_md_user = dnnl::memory::desc(weight_dims, weight_dt, dnnl::memory::format_tag::oihw);
        auto dst_md_user = dnnl::memory::desc(dst_dims, dst_dt, dnnl::memory::format_tag::nhwc);

        // Let oneDNN choose optimal internal layouts.
        auto src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);
        auto weight_md = dnnl::memory::desc(weight_dims, weight_dt, dnnl::memory::format_tag::any);
        auto dst_md = dnnl::memory::desc(dst_dims, dst_dt, dnnl::memory::format_tag::any);

        dnnl::memory::desc bias_md;
        bool use_bias = (bias_ptr != nullptr);
        if (use_bias) {
            bias_md = dnnl::memory::desc(bias_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::x);
        }

        dnnl::convolution_forward::primitive_desc conv_pd;
#if defined(DNNL_VERSION_MAJOR) && (DNNL_VERSION_MAJOR >= 3)
        if (use_bias) {
            conv_pd = dnnl::convolution_forward::primitive_desc(
                eng,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::convolution_direct,
                src_md, weight_md, bias_md, dst_md,
                strides, dilations, padding_l, padding_r,
                dnnl::primitive_attr(),
                false
            );
        } else {
            conv_pd = dnnl::convolution_forward::primitive_desc(
                eng,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::convolution_direct,
                src_md, weight_md, dst_md,
                strides, dilations, padding_l, padding_r,
                dnnl::primitive_attr(),
                false
            );
        }
#else
        dnnl::convolution_forward::desc conv_desc = use_bias
            ? dnnl::convolution_forward::desc(
                  dnnl::prop_kind::forward_inference,
                  dnnl::algorithm::convolution_direct,
                  src_md, weight_md, bias_md, dst_md,
                  strides, dilations, padding_l, padding_r)
            : dnnl::convolution_forward::desc(
                  dnnl::prop_kind::forward_inference,
                  dnnl::algorithm::convolution_direct,
                  src_md, weight_md, dst_md,
                  strides, dilations, padding_l, padding_r);
        conv_pd = dnnl::convolution_forward::primitive_desc(conv_desc, eng);
#endif

        // User memory
        if (input_ptr == nullptr) {
            input_bf16.resize(input_f32.size());
            {
                ProfileScope scope(ProfileKind::Decompress);
                math::convert_float32_to_bf16(input_f32.data(), input_f32.size(), input_bf16.data());
            }
            input_ptr = input_bf16.data();
        }
        auto src_mem_user = dnnl::memory(src_md_user, eng, const_cast<void*>(input_ptr));
        auto weight_mem_user = dnnl::memory(
            weight_md_user, eng, reinterpret_cast<void*>(weight_mem->data().data())
        );
        dnnl::memory bias_mem_dnnl;
        if (use_bias) {
            bias_mem_dnnl = dnnl::memory(bias_md, eng, const_cast<void*>(bias_ptr));
        }

        size_t out_elems = static_cast<size_t>(out_c * out_h * out_w);
        std::vector<uint8_t> out_bytes;
        out_bytes.resize(out_elems * (dst_dt == dnnl::memory::data_type::bf16 ? sizeof(uint16_t) : sizeof(float)));
        auto dst_mem_user = dnnl::memory(dst_md_user, eng, out_bytes.data());

        // Reorders if required
        dnnl::memory src_mem_p = src_mem_user;
        if (conv_pd.src_desc() != src_mem_user.get_desc()) {
            src_mem_p = dnnl::memory(conv_pd.src_desc(), eng);
            {
                ProfileScope scope(ProfileKind::OtherCompute);
                dnnl::reorder(src_mem_user, src_mem_p).execute(s, src_mem_user, src_mem_p);
            }
        }
        dnnl::memory weight_mem_p = weight_mem_user;
        if (conv_pd.weights_desc() != weight_mem_user.get_desc()) {
            weight_mem_p = dnnl::memory(conv_pd.weights_desc(), eng);
            {
                ProfileScope scope(ProfileKind::OtherCompute);
                dnnl::reorder(weight_mem_user, weight_mem_p).execute(s, weight_mem_user, weight_mem_p);
            }
        }
        dnnl::memory dst_mem_p = dnnl::memory(conv_pd.dst_desc(), eng);

        dnnl::convolution_forward conv_prim(conv_pd);
        uint64_t conv_flops = static_cast<uint64_t>(2ULL) *
                              static_cast<uint64_t>(out_c) *
                              static_cast<uint64_t>(out_h) *
                              static_cast<uint64_t>(out_w) *
                              static_cast<uint64_t>(in_c) *
                              static_cast<uint64_t>(k_h) *
                              static_cast<uint64_t>(k_w);
        {
            ProfileScope scope(ProfileKind::Compute);
            add_profile_gemm_flops(conv_flops);
            if (use_bias) {
                conv_prim.execute(s, {
                    {DNNL_ARG_SRC, src_mem_p},
                    {DNNL_ARG_WEIGHTS, weight_mem_p},
                    {DNNL_ARG_BIAS, bias_mem_dnnl},
                    {DNNL_ARG_DST, dst_mem_p}
                });
            } else {
                conv_prim.execute(s, {
                    {DNNL_ARG_SRC, src_mem_p},
                    {DNNL_ARG_WEIGHTS, weight_mem_p},
                    {DNNL_ARG_DST, dst_mem_p}
                });
            }
            s.wait();
        }

        // Reorder to NHWC user layout for downstream ops.
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            dnnl::reorder(dst_mem_p, dst_mem_user).execute(s, dst_mem_p, dst_mem_user);
            s.wait();
        }

        Shape out_shape = config_.output_shape;
        if (std::get<0>(out_shape) == 0 || std::get<1>(out_shape) == 0) {
            out_shape = std::make_tuple(out_h * out_w, out_c);
        }
        return ctx.store_in_memory(result_id_, out_shape, out_dtype, std::move(out_bytes));
    }

private:
    Conv2DIm2ColConfig config_;
};

std::unique_ptr<Operator> create_im2col_conv2d(const Conv2DIm2ColConfig& config) {
    return std::make_unique<Im2ColConv2DOp>(config);
}

} // namespace kvtensor

#include "kvtensor/stable_diffusion.hpp"
#include "kvtensor/sd_ops.hpp"
#include "kvtensor/operators_impl.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/profile.hpp"
#include "math/arithmetic.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace kvtensor {

namespace {

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return sizeof(float);
        case DType::BFLOAT16: return sizeof(uint16_t);
        case DType::FLOAT16: return sizeof(uint16_t);
        case DType::INT8: return sizeof(int8_t);
        default: return sizeof(float);
    }
}

std::vector<float> to_float32_buffer(const std::vector<uint8_t>& data, DType dtype) {
    std::vector<float> out(data.size() / dtype_size(dtype));
    {
        ProfileScope scope(ProfileKind::Decompress);
        math::convert_buffer_to_float32(data.data(), dtype, out.size(), out.data());
    }
    return out;
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
            {
                ProfileScope scope(ProfileKind::Decompress);
                for (size_t i = 0; i < elems; ++i) {
                    out[i] = math::float_to_bf16(src[i]);
                }
            }
            break;
        }
        case DType::INT8: {
            dst.resize(elems * sizeof(int8_t));
            int8_t* out = reinterpret_cast<int8_t*>(dst.data());
            {
                ProfileScope scope(ProfileKind::Decompress);
                for (size_t i = 0; i < elems; ++i) {
                    out[i] = static_cast<int8_t>(std::nearbyint(src[i]));
                }
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported output dtype");
    }
}

bool has_matrix(OperatorContext& ctx, const std::string& id) {
    if (id.empty()) {
        return false;
    }
    if (ctx.resolve_in_memory(id)) {
        return true;
    }
    try {
        if (ctx.resolve_block_matrix(id)) {
            return true;
        }
    } catch (const std::exception&) {
        return false;
    }
    return false;
}

std::string pick_existing(OperatorContext& ctx, const std::string& base, const std::string& suffix) {
    if (base.empty()) {
        return "";
    }
    if (suffix.empty()) {
        return has_matrix(ctx, base) ? base : "";
    }
    std::string candidate = base + suffix;
    if (has_matrix(ctx, candidate)) {
        return candidate;
    }
    return "";
}

std::shared_ptr<InMemoryMatrix> ensure_in_memory(OperatorContext& ctx, const std::string& id) {
    if (auto mem = ctx.resolve_in_memory(id)) {
        return mem;
    }
    if (auto bm = ctx.resolve_block_matrix(id)) {
        auto dense = bm->to_dense(ctx);
        return ctx.store_in_memory(id + "_dense", bm->shape(), bm->dtype(), std::move(dense));
    }
    return nullptr;
}

bool get_matrix_shape(OperatorContext& ctx, const std::string& id, Shape& shape) {
    if (id.empty()) {
        return false;
    }
    if (auto mem = ctx.resolve_in_memory(id)) {
        shape = mem->shape();
        return true;
    }
    try {
        if (auto bm = ctx.resolve_block_matrix(id)) {
            shape = bm->shape();
            return true;
        }
    } catch (const std::exception&) {
        return false;
    }
    return false;
}

struct TensorInfo {
    std::string id;
    Shape shape{0, 0};
    DType dtype{DType::FLOAT32};
    bool found{false};
};

TensorInfo lookup_tensor(OperatorContext& ctx, const std::string& id) {
    TensorInfo info;
    info.id = id;
    if (id.empty()) {
        return info;
    }
    if (auto mem = ctx.resolve_in_memory(id)) {
        info.shape = mem->shape();
        info.dtype = mem->dtype();
        info.found = true;
        return info;
    }
    try {
        if (auto bm = ctx.resolve_block_matrix(id)) {
            info.shape = bm->shape();
            info.dtype = bm->dtype();
            info.found = true;
            return info;
        }
    } catch (const std::exception&) {
        return info;
    }
    return info;
}

int64_t numel(const Shape& shape) {
    auto [r, c] = shape;
    return r * c;
}

std::shared_ptr<InMemoryMatrix> exec_with_log(
    OperatorContext& ctx,
    std::unique_ptr<Operator> op,
    const std::vector<std::string>& input_ids
) {
    std::vector<TensorInfo> inputs;
    inputs.reserve(input_ids.size());
    size_t bytes_in = 0;
    for (const auto& id : input_ids) {
        auto info = lookup_tensor(ctx, id);
        if (info.found) {
            bytes_in += static_cast<size_t>(numel(info.shape)) * dtype_size(info.dtype);
        }
        inputs.push_back(std::move(info));
    }
    auto start = std::chrono::steady_clock::now();
    auto out = op->execute(ctx);
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    size_t bytes_out = 0;
    Shape out_shape{0, 0};
    DType out_dtype = DType::FLOAT32;
    std::string out_id = "<null>";
    if (out) {
        out_shape = out->shape();
        out_dtype = out->dtype();
        out_id = out->matrix_id();
        bytes_out = static_cast<size_t>(numel(out_shape)) * dtype_size(out_dtype);
    }
    if(ms>50){
        std::ostringstream oss;
        oss << "[SDOp] name=" << op->name()
            << " ms=" << ms
            << " bytes_in=" << bytes_in
            << " bytes_out=" << bytes_out
            << " inputs=[";
        bool first = true;
        for (const auto& info : inputs) {
            if (!first) {
                oss << ", ";
            }
            first = false;
            if (!info.found) {
                oss << info.id << ":<missing>";
                continue;
            }
            auto [r, c] = info.shape;
            oss << info.id << ":" << r << "x" << c << ":" << dtype_to_string(info.dtype);
        }
        oss << "] outputs=[";
        if (out) {
            auto [r, c] = out_shape;
            oss << out_id << ":" << r << "x" << c << ":" << dtype_to_string(out_dtype);
        } else {
            oss << "<null>";
        }
        oss << "]";
        std::cout << oss.str() << std::endl;
    }
    return out;
}

int64_t default_num_heads(int64_t hidden) {
    int64_t heads = hidden / 64;
    if (heads < 1) {
        heads = 1;
    }
    while (heads > 1 && (hidden % heads != 0)) {
        --heads;
    }
    return heads;
}

int64_t pick_num_groups(int64_t channels, int64_t max_groups = 32) {
    int64_t groups = std::min<int64_t>(channels, max_groups);
    while (groups > 1 && (channels % groups != 0)) {
        --groups;
    }
    return std::max<int64_t>(groups, 1);
}

std::vector<float> nhwc_to_nchw(const float* nhwc, int64_t height, int64_t width, int64_t channels) {
    std::vector<float> out(static_cast<size_t>(channels * height * width));
    {
        ProfileScope scope(ProfileKind::OtherCompute);
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < width; ++w) {
                    int64_t nhwc_idx = (h * width + w) * channels + c;
                    int64_t nchw_idx = (c * height + h) * width + w;
                    out[static_cast<size_t>(nchw_idx)] = nhwc[static_cast<size_t>(nhwc_idx)];
                }
            }
        }
    }
    return out;
}

std::vector<float> upsample_nearest(
    const float* nhwc,
    int64_t height,
    int64_t width,
    int64_t channels,
    int64_t scale
) {
    if (scale <= 1) {
        return std::vector<float>(nhwc, nhwc + static_cast<size_t>(height * width * channels));
    }
    int64_t out_h = height * scale;
    int64_t out_w = width * scale;
    std::vector<float> out(static_cast<size_t>(out_h * out_w * channels));
    {
        ProfileScope scope(ProfileKind::OtherCompute);
        for (int64_t h = 0; h < out_h; ++h) {
            int64_t src_h = h / scale;
            for (int64_t w = 0; w < out_w; ++w) {
                int64_t src_w = w / scale;
                const float* src = nhwc + (src_h * width + src_w) * channels;
                float* dst = out.data() + (h * out_w + w) * channels;
                std::memcpy(dst, src, static_cast<size_t>(channels) * sizeof(float));
            }
        }
    }
    return out;
}

int64_t infer_chunk_size(OperatorContext& ctx, const std::string& weight_id, int64_t fallback = 512) {
    if (auto bm = ctx.resolve_block_matrix(weight_id)) {
        return bm->chunk_size();
    }
    if (auto mem = ctx.resolve_in_memory(weight_id)) {
        if (mem->packed_gate_up() && !mem->packed_cols().empty()) {
            return mem->packed_cols().front();
        }
    }
    return fallback;
}

std::shared_ptr<InMemoryMatrix> cast_output_dtype(
    OperatorContext& ctx,
    const std::shared_ptr<InMemoryMatrix>& input,
    const std::string& result_id,
    DType dtype
) {
    if (!input) {
        return input;
    }
    if (dtype == input->dtype()) {
        if (input->matrix_id() == result_id) {
            return input;
        }
        return ctx.store_in_memory(result_id, input->shape(), input->dtype(), input->data());
    }
    std::vector<float> f32 = to_float32_buffer(input->data(), input->dtype());
    std::vector<uint8_t> out_bytes;
    float_to_dtype_buffer(f32, dtype, out_bytes);
    return ctx.store_in_memory(result_id, input->shape(), dtype, std::move(out_bytes));
}

} // namespace

std::shared_ptr<InMemoryMatrix> SDTextEncoder::forward(
    const std::vector<int32_t>& input_ids,
    const std::string& result_id,
    OperatorContext& ctx,
    bool profile,
    const std::string& input_embeddings_id
) {
    std::shared_ptr<InMemoryMatrix> current;
    std::string current_id;
    int64_t seq_len = 0;
    int64_t hidden = config_.hidden_size;

    if (!input_embeddings_id.empty()) {
        current = ctx.get_in_memory(input_embeddings_id);
        if (current) {
            auto [rows, cols] = current->shape();
            seq_len = rows;
            hidden = cols;
            current_id = input_embeddings_id;
        }
    }

    if (!current) {
        seq_len = static_cast<int64_t>(input_ids.size());
        if (seq_len == 0) {
            throw std::runtime_error("SDTextEncoder: empty input");
        }

        std::shared_ptr<InMemoryMatrix> embedding_mem = ctx.resolve_in_memory(config_.token_embedding_id);
        std::shared_ptr<BlockMatrix> embedding_block = nullptr;
        if (!embedding_mem) {
            embedding_block = ctx.resolve_block_matrix(config_.token_embedding_id);
        }
        if (!embedding_mem && !embedding_block) {
            throw std::runtime_error("Token embedding not found: " + config_.token_embedding_id);
        }

        Shape emb_shape = embedding_mem ? embedding_mem->shape() : embedding_block->shape();
        auto [emb_rows, emb_cols] = emb_shape;
        hidden = config_.hidden_size > 0 ? config_.hidden_size : emb_cols;
        if (emb_cols != hidden) {
            hidden = emb_cols;
        }

        std::vector<float> embedded_data(static_cast<size_t>(seq_len * hidden));
        DType embedding_dtype = embedding_mem ? embedding_mem->dtype() : embedding_block->dtype();

        if (embedding_mem) {
            const uint8_t* base = embedding_mem->data().data();
            size_t row_bytes = static_cast<size_t>(hidden) * dtype_size(embedding_dtype);
            for (int64_t i = 0; i < seq_len; ++i) {
                int32_t token_id = input_ids[static_cast<size_t>(i)];
                if (token_id < 0 || token_id >= emb_rows) {
                    throw std::runtime_error("Invalid token id: " + std::to_string(token_id));
                }
                const uint8_t* row_ptr = base + static_cast<size_t>(token_id) * row_bytes;
                float* dst = embedded_data.data() + i * hidden;
                if (embedding_dtype == DType::FLOAT32) {
                    std::memcpy(dst, row_ptr, static_cast<size_t>(hidden) * sizeof(float));
                } else {
                    math::convert_buffer_to_float32(row_ptr, embedding_dtype,
                                                    static_cast<size_t>(hidden), dst);
                }
            }
        } else {
            for (int64_t i = 0; i < seq_len; ++i) {
                int32_t token_id = input_ids[static_cast<size_t>(i)];
                if (token_id < 0 || token_id >= emb_rows) {
                    throw std::runtime_error("Invalid token id: " + std::to_string(token_id));
                }
                auto row_bytes = embedding_block->read_row(token_id, ctx);
                float* dst = embedded_data.data() + i * hidden;
                if (embedding_dtype == DType::FLOAT32) {
                    std::memcpy(dst, row_bytes.data(), static_cast<size_t>(hidden) * sizeof(float));
                } else {
                    math::convert_buffer_to_float32(row_bytes.data(), embedding_dtype,
                                                    static_cast<size_t>(hidden), dst);
                }
            }
        }

        std::vector<uint8_t> embedded_bytes(embedded_data.size() * sizeof(float));
        std::memcpy(embedded_bytes.data(), embedded_data.data(), embedded_bytes.size());

        current_id = result_id + "_embed";
        current = ctx.store_in_memory(
            current_id,
            std::make_tuple(seq_len, hidden),
            DType::FLOAT32,
            std::move(embedded_bytes)
        );
    }

    if (profile) {
        std::cout << "[SDTextEncoder] seq_len=" << seq_len
                  << " hidden=" << hidden
                  << " blocks=" << config_.transformer_block_ids.size() << std::endl;
    }

    int64_t num_heads = default_num_heads(hidden);

    for (size_t i = 0; i < config_.transformer_block_ids.size(); ++i) {
        const std::string& prefix = config_.transformer_block_ids[i];
        std::string norm1_weight = pick_existing(ctx, prefix, ".norm1_weight");
        if (norm1_weight.empty()) {
            throw std::runtime_error("Missing norm1 weight for text block: " + prefix);
        }
        std::string norm1_id = result_id + "_text_norm1_" + std::to_string(i);
        auto norm1_op = create_row_column_rmsnorm(current_id, norm1_weight, norm1_id, 1e-5f, true);
        auto norm1 = exec_with_log(ctx, std::move(norm1_op), {current_id, norm1_weight});

        std::string qkv_id = pick_existing(ctx, prefix, ".attn_qkv_proj");
        std::string out_id = pick_existing(ctx, prefix, ".attn_out_proj");
        if (qkv_id.empty() || out_id.empty()) {
            throw std::runtime_error("Missing attention weights for text block: " + prefix);
        }
        AttentionBlockConfig attn_cfg;
        attn_cfg.hidden_states_id = norm1_id;
        attn_cfg.qkv_proj_id = qkv_id;
        attn_cfg.out_proj_id = out_id;
        attn_cfg.result_id = result_id + "_text_attn_" + std::to_string(i);
        attn_cfg.hidden_size = hidden;
        attn_cfg.num_heads = num_heads;
        auto attn_op = create_attention_block(attn_cfg);
        auto attn = exec_with_log(ctx, std::move(attn_op), {norm1_id, qkv_id, out_id});

        std::string attn_res_id = result_id + "_text_attn_res_" + std::to_string(i);
        auto attn_res_op = create_row_column_elementwise_add(
            current_id, attn->matrix_id(), attn_res_id, 1.0f, 1.0f, true
        );
        auto attn_res = exec_with_log(ctx, std::move(attn_res_op), {current_id, attn->matrix_id()});

        ctx.clear_in_memory(norm1_id);
        ctx.clear_in_memory(attn->matrix_id());

        std::string norm2_weight = pick_existing(ctx, prefix, ".norm2_weight");
        if (norm2_weight.empty()) {
            throw std::runtime_error("Missing norm2 weight for text block: " + prefix);
        }
        std::string norm2_id = result_id + "_text_norm2_" + std::to_string(i);
        auto norm2_op = create_row_column_rmsnorm(attn_res_id, norm2_weight, norm2_id, 1e-5f, true);
        auto norm2 = exec_with_log(ctx, std::move(norm2_op), {attn_res_id, norm2_weight});

        std::string gate_up_id = pick_existing(ctx, prefix, ".ffn_gate_up_proj");
        std::string down_id = pick_existing(ctx, prefix, ".ffn_down_proj");
        if (gate_up_id.empty() || down_id.empty()) {
            throw std::runtime_error("Missing feedforward weights for text block: " + prefix);
        }
        int64_t chunk_size = infer_chunk_size(ctx, gate_up_id);
        std::string ffn_id = result_id + "_text_ffn_" + std::to_string(i);
        auto ffn_op = create_row_column_llama_feedforward(norm2_id, gate_up_id, down_id, ffn_id, chunk_size);
        auto ffn = exec_with_log(ctx, std::move(ffn_op), {norm2_id, gate_up_id, down_id});

        std::string block_out_id = result_id + "_text_block_" + std::to_string(i);
        auto out_op = create_row_column_elementwise_add(attn_res_id, ffn_id, block_out_id, 1.0f, 1.0f, true);
        auto block_out = exec_with_log(ctx, std::move(out_op), {attn_res_id, ffn_id});

        ctx.clear_in_memory(attn_res_id);
        ctx.clear_in_memory(norm2_id);
        ctx.clear_in_memory(ffn_id);
        if (current_id != block_out_id) {
            ctx.clear_in_memory(current_id);
        }

        current_id = block_out_id;
        current = block_out;
    }

    std::shared_ptr<InMemoryMatrix> result = current;
    if (!config_.final_layer_norm_id.empty()) {
        auto out_norm_op = create_row_column_rmsnorm(
            current_id, config_.final_layer_norm_id, result_id, 1e-5f, true
        );
        result = exec_with_log(ctx, std::move(out_norm_op), {current_id, config_.final_layer_norm_id});
        if (current_id != result_id) {
            ctx.clear_in_memory(current_id);
        }
    } else if (current_id != result_id) {
        result = ctx.store_in_memory(result_id, current->shape(), current->dtype(), current->data());
    }

    auto casted = cast_output_dtype(ctx, result, result_id, config_.output_dtype);
    if (casted && casted->matrix_id() != result->matrix_id()) {
        ctx.clear_in_memory(result->matrix_id());
    }
    return casted;
}

std::shared_ptr<InMemoryMatrix> SDUNet::forward(
    const std::string& latents_id,
    const std::string& text_embeddings_id,
    const std::string& result_id,
    OperatorContext& ctx,
    int64_t timestep,
    bool profile
) {
    struct ActState {
        std::string id;
        int64_t h = 0;
        int64_t w = 0;
        int64_t c = 0;
    };

    auto ensure_weight = [&](const std::string& id) -> std::shared_ptr<InMemoryMatrix> {
        if (id.empty()) {
            return nullptr;
        }
        if (auto mem = ctx.resolve_in_memory(id)) {
            return mem;
        }
        if (auto bm = ctx.resolve_block_matrix(id)) {
            auto dense = bm->to_dense(ctx);
            return ctx.store_in_memory(id, bm->shape(), bm->dtype(), std::move(dense));
        }
        return nullptr;
    };

    auto latents = ctx.resolve_in_memory(latents_id);
    if (!latents) {
        latents = ensure_in_memory(ctx, latents_id);
    }
    if (!latents) {
        throw std::runtime_error("Latents not found: " + latents_id);
    }

    auto [rows, cols] = latents->shape();
    int64_t height = config_.sample_height;
    int64_t width = config_.sample_width;
    if (height <= 0 || width <= 0 || height * width != rows) {
        int64_t side = static_cast<int64_t>(std::sqrt(static_cast<double>(rows)));
        height = side;
        width = side;
    }
    int64_t channels = cols;

    int64_t base = (config_.model_channels > 0) ? config_.model_channels : 320;
    int64_t time_dim = base;
    int64_t time_embed_dim = base * 4;
    int64_t in_channels = (config_.latent_channels > 0) ? config_.latent_channels : channels;

    if (profile) {
        std::cout << "[SDUNet] rows=" << rows << " cols=" << cols
                  << " H=" << height << " W=" << width << " C=" << channels
                  << " base=" << base << std::endl;
    }

    ActState current{latents->matrix_id(), height, width, channels};

    auto run_silu = [&](const ActState& input, const std::string& tag) -> ActState {
        auto op = create_row_column_silu(input.id, result_id + "_unet_silu_" + tag, true);
        auto out = exec_with_log(ctx, std::move(op), {input.id});
        return ActState{out->matrix_id(), input.h, input.w, input.c};
    };

    auto run_geglu = [&](const ActState& input, const std::string& tag) -> ActState {
        auto op = create_row_column_geglu(input.id, result_id + "_unet_geglu_" + tag, true);
        auto out = exec_with_log(ctx, std::move(op), {input.id});
        return ActState{out->matrix_id(), input.h, input.w, input.c / 2};
    };

    auto run_group_norm = [&](const ActState& input, const std::string& weight_id,
                              const std::string& bias_id, const std::string& tag) -> ActState {
        if (!ensure_weight(weight_id) || !ensure_weight(bias_id)) {
            throw std::runtime_error("GroupNorm weights missing: " + weight_id);
        }
        GroupNormConfig cfg;
        cfg.input_id = input.id;
        cfg.weight_id = weight_id;
        cfg.bias_id = bias_id;
        cfg.result_id = result_id + "_unet_gn_" + tag;
        cfg.num_channels = input.c;
        cfg.num_groups = 32;
        cfg.output_dtype = DType::FLOAT32;
        auto op = create_group_norm(cfg);
        auto out = exec_with_log(ctx, std::move(op), {input.id, weight_id, bias_id});
        return ActState{out->matrix_id(), input.h, input.w, input.c};
    };

    auto run_layer_norm = [&](const ActState& input, const std::string& weight_id,
                              const std::string& bias_id, const std::string& tag) -> ActState {
        if (!ensure_weight(weight_id) || !ensure_weight(bias_id)) {
            throw std::runtime_error("LayerNorm weights missing: " + weight_id);
        }
        auto op = create_row_column_layernorm(input.id, weight_id, bias_id, result_id + "_unet_ln_" + tag, 1e-5f, true);
        auto out = exec_with_log(ctx, std::move(op), {input.id, weight_id, bias_id});
        return ActState{out->matrix_id(), input.h, input.w, input.c};
    };

    auto run_conv = [&](const ActState& input,
                        int64_t out_c,
                        int64_t k,
                        int64_t stride,
                        int64_t pad,
                        const std::string& weight_id,
                        const std::string& bias_id,
                        const std::string& tag) -> ActState {
        if (!ensure_weight(weight_id)) {
            throw std::runtime_error("Conv weight missing: " + weight_id);
        }
        if (!bias_id.empty() && !ensure_weight(bias_id)) {
            throw std::runtime_error("Conv bias missing: " + bias_id);
        }

        auto act_mem = ctx.resolve_in_memory(input.id);
        if (!act_mem) {
            throw std::runtime_error("Activation missing: " + input.id);
        }
        auto act_f32 = to_float32_buffer(act_mem->data(), act_mem->dtype());
        auto nchw = nhwc_to_nchw(act_f32.data(), input.h, input.w, input.c);
        std::vector<uint8_t> nchw_bytes;
        DType nchw_dtype = DType::FLOAT32;
        if (auto wmem = ctx.resolve_in_memory(weight_id)) {
            if (wmem->dtype() == DType::BFLOAT16) {
                nchw_dtype = DType::BFLOAT16;
            }
        }
        if (nchw_dtype == DType::BFLOAT16) {
            nchw_bytes.resize(nchw.size() * sizeof(uint16_t));
            auto* out_bf16 = reinterpret_cast<uint16_t*>(nchw_bytes.data());
            math::convert_float32_to_bf16(nchw.data(), nchw.size(), out_bf16);
        } else {
            nchw_bytes.resize(nchw.size() * sizeof(float));
            std::memcpy(nchw_bytes.data(), nchw.data(), nchw_bytes.size());
        }

        std::string nchw_id = result_id + "_unet_nchw_" + tag;
        ctx.store_in_memory(nchw_id, std::make_tuple(1, input.c * input.h * input.w),
                            nchw_dtype, std::move(nchw_bytes));

        Conv2DIm2ColConfig conv_cfg;
        conv_cfg.input_id = nchw_id;
        conv_cfg.weight_id = weight_id;
        conv_cfg.bias_id = bias_id;
        conv_cfg.result_id = result_id + "_unet_conv_" + tag;
        conv_cfg.input_channels = input.c;
        conv_cfg.input_height = input.h;
        conv_cfg.input_width = input.w;
        conv_cfg.output_channels = out_c;
        conv_cfg.kernel_h = k;
        conv_cfg.kernel_w = k;
        conv_cfg.stride_h = stride;
        conv_cfg.stride_w = stride;
        conv_cfg.pad_h = pad;
        conv_cfg.pad_w = pad;
        conv_cfg.output_dtype = DType::FLOAT32;
        auto conv_op = create_im2col_conv2d(conv_cfg);
        std::vector<std::string> conv_inputs{nchw_id, weight_id};
        if (!bias_id.empty()) {
            conv_inputs.push_back(bias_id);
        }
        auto conv_out = exec_with_log(ctx, std::move(conv_op), conv_inputs);
        
        int64_t out_h = (input.h + 2 * pad - k) / stride + 1;
        int64_t out_w = (input.w + 2 * pad - k) / stride + 1;
        ctx.clear_in_memory(nchw_id);

        return ActState{conv_out->matrix_id(), out_h, out_w, out_c};
    };

    auto add_bias_to_matrix = [&](const std::string& matrix_id, const std::string& bias_id,
                                  int64_t rows_count, int64_t cols_count, const std::string& tag,
                                  DType target_dtype) -> std::string {
        if (bias_id.empty()) {
            return matrix_id;
        }
        auto matrix = ctx.resolve_in_memory(matrix_id);
        auto bias = ctx.resolve_in_memory(bias_id);
        if (!matrix || !bias) {
            throw std::runtime_error("Bias add missing for " + tag);
        }
        DType out_dtype = target_dtype;
        if (out_dtype == DType::BFLOAT16) {
            std::vector<uint16_t> mat_bf16;
            const uint16_t* mat_ptr = nullptr;
            if (matrix->dtype() == DType::BFLOAT16) {
                mat_ptr = reinterpret_cast<const uint16_t*>(matrix->data().data());
            } else {
                size_t elems = static_cast<size_t>(rows_count * cols_count);
                mat_bf16.resize(elems);
                auto mat_f32 = to_float32_buffer(matrix->data(), matrix->dtype());
                math::convert_float32_to_bf16(mat_f32.data(), mat_f32.size(), mat_bf16.data());
                mat_ptr = mat_bf16.data();
            }

            std::vector<uint16_t> bias_bf16;
            const uint16_t* bias_ptr = nullptr;
            if (bias->dtype() == DType::BFLOAT16) {
                bias_ptr = reinterpret_cast<const uint16_t*>(bias->data().data());
            } else {
                auto [b_r, b_c] = bias->shape();
                size_t elems = static_cast<size_t>(b_r * b_c);
                bias_bf16.resize(elems);
                auto bias_f32 = to_float32_buffer(bias->data(), bias->dtype());
                math::convert_float32_to_bf16(bias_f32.data(), bias_f32.size(), bias_bf16.data());
                bias_ptr = bias_bf16.data();
            }
            auto [b_r2, b_c2] = bias->shape();
            if (static_cast<int64_t>(b_r2 * b_c2) != cols_count) {
                throw std::runtime_error("Bias size mismatch for " + bias_id);
            }
            std::vector<uint8_t> out_bytes(static_cast<size_t>(rows_count * cols_count) * sizeof(uint16_t));
            auto* out_bf16 = reinterpret_cast<uint16_t*>(out_bytes.data());
            for (int64_t r = 0; r < rows_count; ++r) {
                for (int64_t c = 0; c < cols_count; ++c) {
                    size_t idx = static_cast<size_t>(r * cols_count + c);
                    float v = math::bf16_to_float(mat_ptr[idx]) + math::bf16_to_float(bias_ptr[c]);
                    out_bf16[idx] = math::float_to_bf16(v);
                }
            }
            std::string out_id = result_id + "_unet_bias_" + tag;
            ctx.store_in_memory(out_id, std::make_tuple(rows_count, cols_count), DType::BFLOAT16, std::move(out_bytes));
            return out_id;
        }

        auto mat_f32 = to_float32_buffer(matrix->data(), matrix->dtype());
        auto bias_f32 = to_float32_buffer(bias->data(), bias->dtype());
        if (static_cast<int64_t>(bias_f32.size()) != cols_count) {
            throw std::runtime_error("Bias size mismatch for " + bias_id);
        }
        for (int64_t r = 0; r < rows_count; ++r) {
            float* row_ptr = mat_f32.data() + r * cols_count;
            for (int64_t c = 0; c < cols_count; ++c) {
                row_ptr[c] += bias_f32[static_cast<size_t>(c)];
            }
        }
        std::vector<uint8_t> out_bytes(mat_f32.size() * sizeof(float));
        std::memcpy(out_bytes.data(), mat_f32.data(), out_bytes.size());
        std::string out_id = result_id + "_unet_bias_" + tag;
        ctx.store_in_memory(out_id, std::make_tuple(rows_count, cols_count), DType::FLOAT32, std::move(out_bytes));
        return out_id;
    };

    auto run_linear = [&](const ActState& input,
                          const std::string& weight_id,
                          const std::string& bias_id,
                          const std::string& tag,
                          int64_t out_cols) -> ActState {
        if (!ensure_weight(weight_id)) {
            throw std::runtime_error("Linear weight missing: " + weight_id);
        }
        if (!bias_id.empty() && !ensure_weight(bias_id)) {
            throw std::runtime_error("Linear bias missing: " + bias_id);
        }
        DType weight_dtype = DType::FLOAT32;
        if (auto wmem = ctx.resolve_in_memory(weight_id)) {
            weight_dtype = wmem->dtype();
        } else if (auto wbm = ctx.resolve_block_matrix(weight_id)) {
            weight_dtype = wbm->dtype();
        }
        auto mm_op = create_row_column_matmul(input.id, weight_id, result_id + "_unet_linear_" + tag, true);
        auto mm_out = exec_with_log(ctx, std::move(mm_op), {input.id, weight_id});
        std::string out_id = mm_out->matrix_id();
        if (!bias_id.empty()) {
            out_id = add_bias_to_matrix(out_id, bias_id, input.h * input.w, out_cols, tag, weight_dtype);
            if (out_id != mm_out->matrix_id()) {
                ctx.clear_in_memory(mm_out->matrix_id());
            }
        }
        return ActState{out_id, input.h, input.w, out_cols};
    };

    auto add_time_embedding = [&](const ActState& input, const std::string& time_id,
                                  const std::string& tag) -> ActState {
        auto act_mem = ctx.resolve_in_memory(input.id);
        auto time_mem = ctx.resolve_in_memory(time_id);
        if (!act_mem || !time_mem) {
            throw std::runtime_error("Time embedding missing for " + tag);
        }
        auto [t_rows, t_cols] = time_mem->shape();
        if (t_rows != 1 || t_cols != input.c) {
            throw std::runtime_error("Time embedding shape mismatch for " + tag);
        }
        auto act_f32 = to_float32_buffer(act_mem->data(), act_mem->dtype());
        auto time_f32 = to_float32_buffer(time_mem->data(), time_mem->dtype());
        size_t rows_local = static_cast<size_t>(input.h * input.w);
        for (size_t r = 0; r < rows_local; ++r) {
            float* row_ptr = act_f32.data() + r * input.c;
            for (int64_t c = 0; c < input.c; ++c) {
                row_ptr[c] += time_f32[static_cast<size_t>(c)];
            }
        }
        std::vector<uint8_t> out_bytes(act_f32.size() * sizeof(float));
        std::memcpy(out_bytes.data(), act_f32.data(), out_bytes.size());
        std::string out_id = result_id + "_unet_time_add_" + tag;
        ctx.store_in_memory(out_id, act_mem->shape(), DType::FLOAT32, std::move(out_bytes));
        return ActState{out_id, input.h, input.w, input.c};
    };

    auto run_add = [&](const ActState& a, const ActState& b, const std::string& tag) -> ActState {
        if (a.h != b.h || a.w != b.w || a.c != b.c) {
            throw std::runtime_error("Residual add shape mismatch: " + tag);
        }
        auto op = create_row_column_elementwise_add(
            a.id, b.id, result_id + "_unet_add_" + tag, 1.0f, 1.0f, true
        );
        auto out = exec_with_log(ctx, std::move(op), {a.id, b.id});
        return ActState{out->matrix_id(), a.h, a.w, a.c};
    };

    auto run_concat = [&](const ActState& a, const ActState& b, const std::string& tag) -> ActState {
        if (a.h != b.h || a.w != b.w) {
            throw std::runtime_error("Concat shape mismatch: " + tag);
        }
        auto a_mem = ctx.resolve_in_memory(a.id);
        auto b_mem = ctx.resolve_in_memory(b.id);
        if (!a_mem || !b_mem) {
            throw std::runtime_error("Concat inputs missing: " + tag);
        }
        auto a_f32 = to_float32_buffer(a_mem->data(), a_mem->dtype());
        auto b_f32 = to_float32_buffer(b_mem->data(), b_mem->dtype());
        size_t rows_local = static_cast<size_t>(a.h * a.w);
        std::vector<float> out(static_cast<size_t>(rows_local * (a.c + b.c)));
        for (size_t r = 0; r < rows_local; ++r) {
            float* dst = out.data() + r * (a.c + b.c);
            const float* src_a = a_f32.data() + r * a.c;
            const float* src_b = b_f32.data() + r * b.c;
            std::memcpy(dst, src_a, static_cast<size_t>(a.c) * sizeof(float));
            std::memcpy(dst + a.c, src_b, static_cast<size_t>(b.c) * sizeof(float));
        }
        std::vector<uint8_t> out_bytes(out.size() * sizeof(float));
        std::memcpy(out_bytes.data(), out.data(), out_bytes.size());
        std::string out_id = result_id + "_unet_cat_" + tag;
        ctx.store_in_memory(out_id, std::make_tuple(a.h * a.w, a.c + b.c), DType::FLOAT32, std::move(out_bytes));
        return ActState{out_id, a.h, a.w, a.c + b.c};
    };

    auto run_attention = [&](const ActState& input,
                             const std::string& prefix,
                             const std::string& encoder_id,
                             const std::string& tag) -> ActState {
        AttentionBlockConfig attn_cfg;
        attn_cfg.hidden_states_id = input.id;
        attn_cfg.encoder_hidden_states_id = encoder_id;
        attn_cfg.q_proj_id = prefix + ".to_q.weight";
        attn_cfg.k_proj_id = prefix + ".to_k.weight";
        attn_cfg.v_proj_id = prefix + ".to_v.weight";
        attn_cfg.out_proj_id = prefix + ".to_out.weight";
        attn_cfg.out_proj_bias_id = prefix + ".to_out.bias";
        attn_cfg.result_id = result_id + "_unet_attn_" + tag;
        attn_cfg.hidden_size = input.c;
        attn_cfg.num_heads = input.c / 8;
        auto attn_op = create_attention_block(attn_cfg);
        std::vector<std::string> attn_inputs{input.id};
        if (!encoder_id.empty()) {
            attn_inputs.push_back(encoder_id);
        }
        attn_inputs.push_back(attn_cfg.q_proj_id);
        attn_inputs.push_back(attn_cfg.k_proj_id);
        attn_inputs.push_back(attn_cfg.v_proj_id);
        attn_inputs.push_back(attn_cfg.out_proj_id);
        if (!attn_cfg.out_proj_bias_id.empty()) {
            attn_inputs.push_back(attn_cfg.out_proj_bias_id);
        }
        auto attn_out = exec_with_log(ctx, std::move(attn_op), attn_inputs);
        return ActState{attn_out->matrix_id(), input.h, input.w, input.c};
    };

    auto run_resnet = [&](const ActState& input,
                          int64_t out_c,
                          bool shortcut,
                          const std::string& prefix,
                          const std::string& tag,
                          const std::string& time_id) -> ActState {
        auto norm1 = run_group_norm(input, prefix + ".norm1.weight", prefix + ".norm1.bias", tag + "_n1");
        auto act1 = run_silu(norm1, tag + "_s1");
        auto conv1 = run_conv(act1, out_c, 3, 1, 1, prefix + ".conv1.weight", prefix + ".conv1.bias", tag + "_c1");
        auto time_proj = run_linear(ActState{time_id, 1, 1, time_embed_dim},
                                    prefix + ".time_emb_proj.weight", prefix + ".time_emb_proj.bias",
                                    tag + "_te", out_c);
        ActState conv1_time = add_time_embedding(conv1, time_proj.id, tag + "_teadd");

        auto norm2 = run_group_norm(conv1_time, prefix + ".norm2.weight", prefix + ".norm2.bias", tag + "_n2");
        auto act2 = run_silu(norm2, tag + "_s2");
        auto conv2 = run_conv(act2, out_c, 3, 1, 1, prefix + ".conv2.weight", prefix + ".conv2.bias", tag + "_c2");

        ActState shortcut_state = input;
        if (shortcut) {
            shortcut_state = run_conv(input, out_c, 1, 1, 0, prefix + ".conv_shortcut.weight",
                                      prefix + ".conv_shortcut.bias", tag + "_cs");
        }

        auto out = run_add(conv2, shortcut_state, tag + "_add");
        if (shortcut_state.id != input.id) {
            ctx.clear_in_memory(shortcut_state.id);
        }
        ctx.clear_in_memory(norm1.id);
        ctx.clear_in_memory(act1.id);
        ctx.clear_in_memory(conv1.id);
        ctx.clear_in_memory(time_proj.id);
        ctx.clear_in_memory(conv1_time.id);
        ctx.clear_in_memory(norm2.id);
        ctx.clear_in_memory(act2.id);
        if (conv2.id != out.id) {
            ctx.clear_in_memory(conv2.id);
        }
        return out;
    };

    auto run_transformer2d = [&](const ActState& input,
                                 const std::string& prefix,
                                 const std::string& encoder_id,
                                 const std::string& tag) -> ActState {
        auto proj_in = run_conv(input, input.c, 1, 1, 0, prefix + ".proj_in.weight", prefix + ".proj_in.bias", tag + "_pin");
        ActState x = proj_in;

        auto n1 = run_layer_norm(x, prefix + ".transformer_blocks.0.norm1.weight",
                                 prefix + ".transformer_blocks.0.norm1.bias", tag + "_ln1");
        auto attn1 = run_attention(n1, prefix + ".transformer_blocks.0.attn1", "", tag + "_attn1");
        auto res1 = run_add(x, attn1, tag + "_res1");
        ctx.clear_in_memory(n1.id);
        ctx.clear_in_memory(attn1.id);
        if (x.id != res1.id) {
            ctx.clear_in_memory(x.id);
        }

        auto n2 = run_layer_norm(res1, prefix + ".transformer_blocks.0.norm2.weight",
                                 prefix + ".transformer_blocks.0.norm2.bias", tag + "_ln2");
        auto attn2 = run_attention(n2, prefix + ".transformer_blocks.0.attn2", encoder_id, tag + "_attn2");
        auto res2 = run_add(res1, attn2, tag + "_res2");
        ctx.clear_in_memory(n2.id);
        ctx.clear_in_memory(attn2.id);
        if (res1.id != res2.id) {
            ctx.clear_in_memory(res1.id);
        }

        auto n3 = run_layer_norm(res2, prefix + ".transformer_blocks.0.norm3.weight",
                                 prefix + ".transformer_blocks.0.norm3.bias", tag + "_ln3");
        auto ff1 = run_linear(n3, prefix + ".transformer_blocks.0.ff.proj.weight",
                              prefix + ".transformer_blocks.0.ff.proj.bias", tag + "_ff1",
                              input.c * 8);
        auto geglu = run_geglu(ff1, tag + "_ffg");
        auto ff2 = run_linear(geglu, prefix + ".transformer_blocks.0.ff.out.weight",
                              prefix + ".transformer_blocks.0.ff.out.bias", tag + "_ff2",
                              input.c);
        auto res3 = run_add(res2, ff2, tag + "_res3");
        ctx.clear_in_memory(n3.id);
        ctx.clear_in_memory(ff1.id);
        ctx.clear_in_memory(geglu.id);
        ctx.clear_in_memory(ff2.id);
        if (res2.id != res3.id) {
            ctx.clear_in_memory(res2.id);
        }

        auto proj_out = run_conv(res3, input.c, 1, 1, 0, prefix + ".proj_out.weight", prefix + ".proj_out.bias", tag + "_pout");
        auto out = run_add(input, proj_out, tag + "_out");
        ctx.clear_in_memory(proj_out.id);
        if (res3.id != out.id) {
            ctx.clear_in_memory(res3.id);
        }
        if (input.id != out.id) {
            ctx.clear_in_memory(input.id);
        }
        return out;
    };

    auto run_upsample = [&](const ActState& input,
                            const std::string& weight_id,
                            const std::string& bias_id,
                            const std::string& tag) -> ActState {
        auto act_mem = ctx.resolve_in_memory(input.id);
        if (!act_mem) {
            throw std::runtime_error("Upsample input missing: " + input.id);
        }
        auto act_f32 = to_float32_buffer(act_mem->data(), act_mem->dtype());
        auto up = upsample_nearest(act_f32.data(), input.h, input.w, input.c, 2);
        int64_t out_h = input.h * 2;
        int64_t out_w = input.w * 2;
        std::vector<uint8_t> up_bytes(up.size() * sizeof(float));
        std::memcpy(up_bytes.data(), up.data(), up_bytes.size());
        std::string up_id = result_id + "_unet_up_" + tag;
        ctx.store_in_memory(up_id, std::make_tuple(out_h * out_w, input.c), DType::FLOAT32, std::move(up_bytes));
        ActState up_state{up_id, out_h, out_w, input.c};
        auto conv = run_conv(up_state, input.c, 3, 1, 1, weight_id, bias_id, tag + "_conv");
        ctx.clear_in_memory(up_state.id);
        return conv;
    };

    // Time embedding (base -> 4*base -> 4*base).
    std::vector<float> emb(static_cast<size_t>(time_dim));
    for (int64_t i = 0; i < time_dim / 2; ++i) {
        float inv = std::exp(-std::log(10000.0f) * static_cast<float>(i) / (time_dim / 2));
        float arg = static_cast<float>(timestep) * inv;
        emb[static_cast<size_t>(2 * i)] = std::sin(arg);
        emb[static_cast<size_t>(2 * i + 1)] = std::cos(arg);
    }
    std::vector<uint8_t> emb_bytes(emb.size() * sizeof(float));
    std::memcpy(emb_bytes.data(), emb.data(), emb_bytes.size());
    std::string time_id = result_id + "_unet_time";
    ctx.store_in_memory(time_id, std::make_tuple(1, time_dim), DType::FLOAT32, std::move(emb_bytes));

    ActState time_state{time_id, 1, 1, time_dim};
    auto t1 = run_linear(time_state, "unet.time_embedding.linear_1.weight",
                         "unet.time_embedding.linear_1.bias", "time1", time_embed_dim);
    auto t1_act = run_silu(t1, "time1");
    auto t2 = run_linear(t1_act, "unet.time_embedding.linear_2.weight",
                         "unet.time_embedding.linear_2.bias", "time2", time_embed_dim);
    ctx.clear_in_memory(time_id);
    ctx.clear_in_memory(t1.id);
    ctx.clear_in_memory(t1_act.id);
    std::string time_embed_id = t2.id;

    // UNet forward.
    current = run_conv(current, base, 3, 1, 1, "unet.conv_in.weight", "unet.conv_in.bias", "conv_in");

    std::vector<ActState> skips;
    skips.push_back(current);

    auto push_skip = [&](const ActState& s) { skips.push_back(s); };
    auto pop_skip = [&]() -> ActState {
        if (skips.empty()) {
            throw std::runtime_error("Skip connection underflow");
        }
        ActState s = skips.back();
        skips.pop_back();
        return s;
    };

    // Down blocks (cross-attn for first 3 blocks).
    auto down_block = [&](int idx, int64_t in_c, int64_t out_c, bool cross_attn, bool downsample) {
        std::string base_id = "unet.down_blocks." + std::to_string(idx);
        current = run_resnet(current, out_c, in_c != out_c, base_id + ".resnets.0",
                             "down" + std::to_string(idx) + "_r0", time_embed_id);
        if (cross_attn) {
            current = run_transformer2d(current, base_id + ".attentions.0", text_embeddings_id,
                                        "down" + std::to_string(idx) + "_a0");
        }
        push_skip(current);
        current = run_resnet(current, out_c, false, base_id + ".resnets.1",
                             "down" + std::to_string(idx) + "_r1", time_embed_id);
        if (cross_attn) {
            current = run_transformer2d(current, base_id + ".attentions.1", text_embeddings_id,
                                        "down" + std::to_string(idx) + "_a1");
        }
        push_skip(current);
        if (downsample) {
            current = run_conv(current, out_c, 3, 2, 1,
                               base_id + ".downsamplers.0.conv.weight",
                               base_id + ".downsamplers.0.conv.bias",
                               "down" + std::to_string(idx) + "_down");
            push_skip(current);
        }
    };

    down_block(0, base, base, true, true);
    down_block(1, base, base * 2, true, true);
    down_block(2, base * 2, base * 4, true, true);
    down_block(3, base * 4, base * 4, false, false);

    // Mid block
    current = run_resnet(current, base * 4, false, "unet.mid_block.resnets.0", "mid_r0", time_embed_id);
    current = run_transformer2d(current, "unet.mid_block.attentions.0", text_embeddings_id, "mid_a0");
    current = run_resnet(current, base * 4, false, "unet.mid_block.resnets.1", "mid_r1", time_embed_id);

    auto up_block = [&](int idx, int64_t out_c, bool cross_attn, bool upsample) {
        std::string base_id = "unet.up_blocks." + std::to_string(idx);
        for (int r = 0; r < 3; ++r) {
            auto skip = pop_skip();
            auto prev_current_id = current.id;
            auto merged = run_concat(current, skip, "up" + std::to_string(idx) + "_cat" + std::to_string(r));
            if (prev_current_id != merged.id && prev_current_id != skip.id) {
                ctx.clear_in_memory(prev_current_id);
            }
            ctx.clear_in_memory(skip.id);
            current = run_resnet(merged, out_c, true,
                                 base_id + ".resnets." + std::to_string(r),
                                 "up" + std::to_string(idx) + "_r" + std::to_string(r),
                                 time_embed_id);
            if (cross_attn) {
                current = run_transformer2d(current,
                                            base_id + ".attentions." + std::to_string(r),
                                            text_embeddings_id,
                                            "up" + std::to_string(idx) + "_a" + std::to_string(r));
            }
            ctx.clear_in_memory(merged.id);
        }
        if (upsample) {
            auto prev_current_id = current.id;
            current = run_upsample(current,
                                   base_id + ".upsamplers.0.conv.weight",
                                   base_id + ".upsamplers.0.conv.bias",
                                   "up" + std::to_string(idx) + "_up");
            if (prev_current_id != current.id) {
                ctx.clear_in_memory(prev_current_id);
            }
        }
    };

    up_block(0, base * 4, false, true);
    up_block(1, base * 4, true, true);
    up_block(2, base * 2, true, true);
    up_block(3, base, true, false);

    ctx.clear_in_memory(time_embed_id);

    auto norm_out = run_group_norm(current, "unet.conv_norm_out.weight", "unet.conv_norm_out.bias", "out_gn");
    auto act_out = run_silu(norm_out, "out_act");
    auto conv_out = run_conv(act_out, in_channels, 3, 1, 1, "unet.conv_out.weight", "unet.conv_out.bias", "conv_out");

    ctx.clear_in_memory(norm_out.id);
    ctx.clear_in_memory(act_out.id);

    auto output_mem = ctx.resolve_in_memory(conv_out.id);
    if (!output_mem) {
        throw std::runtime_error("UNet output missing: " + conv_out.id);
    }
    auto output = cast_output_dtype(ctx, output_mem, result_id, config_.output_dtype);
    if (output && output->matrix_id() != conv_out.id) {
        ctx.clear_in_memory(conv_out.id);
    }
    return output;
}

std::shared_ptr<InMemoryMatrix> SDVAE::decode(
    const std::string& latents_id,
    const std::string& result_id,
    OperatorContext& ctx,
    bool profile
) {
    auto latents = ensure_in_memory(ctx, latents_id);
    if (!latents) {
        throw std::runtime_error("Latents not found for VAE decode: " + latents_id);
    }

    auto [rows, cols] = latents->shape();
    int64_t latent_h = config_.image_size / std::max<int64_t>(config_.scale_factor, 1);
    int64_t latent_w = latent_h;
    if (latent_h * latent_w != rows) {
        int64_t side = static_cast<int64_t>(std::sqrt(static_cast<double>(rows)));
        latent_h = side;
        latent_w = side;
    }

    auto latents_f32 = to_float32_buffer(latents->data(), latents->dtype());
    auto upsampled = upsample_nearest(latents_f32.data(), latent_h, latent_w, cols, config_.scale_factor);

    int64_t height = latent_h * std::max<int64_t>(config_.scale_factor, 1);
    int64_t width = latent_w * std::max<int64_t>(config_.scale_factor, 1);
    int64_t channels = cols;

    std::vector<uint8_t> up_bytes(upsampled.size() * sizeof(float));
    std::memcpy(up_bytes.data(), upsampled.data(), up_bytes.size());
    std::string current_id = result_id + "_vae_up";
    ctx.store_in_memory(current_id, std::make_tuple(height * width, channels), DType::FLOAT32, std::move(up_bytes));

    if (profile) {
        std::cout << "[SDVAE] upsampled to " << height << "x" << width
                  << " channels=" << channels
                  << " convs=" << config_.decoder_weight_ids.size() << std::endl;
    }

    size_t idx = 0;
    for (const auto& prefix : config_.decoder_weight_ids) {
        std::string conv_weight = pick_existing(ctx, prefix, ".conv_weight");
        std::string conv_bias = pick_existing(ctx, prefix, ".conv_bias");
        if (conv_weight.empty()) {
            continue;
        }
        auto conv_weight_mem = ensure_in_memory(ctx, conv_weight);
        if (!conv_weight_mem) {
            throw std::runtime_error("VAE conv weight not found: " + conv_weight);
        }
        auto [w_rows, w_cols] = conv_weight_mem->shape();
        int64_t out_c = w_rows;
        int64_t kernel_elems = w_cols / channels;
        int64_t k = static_cast<int64_t>(std::sqrt(static_cast<double>(kernel_elems)));
        if (k * k != kernel_elems) {
            k = 1;
        }
        int64_t pad = k / 2;

        auto act_mem = ensure_in_memory(ctx, current_id);
        auto act_f32 = to_float32_buffer(act_mem->data(), act_mem->dtype());
        auto nchw = nhwc_to_nchw(act_f32.data(), height, width, channels);
        std::vector<uint8_t> nchw_bytes;
        DType nchw_dtype = DType::FLOAT32;
        if (conv_weight_mem->dtype() == DType::BFLOAT16) {
            nchw_dtype = DType::BFLOAT16;
        }
        if (nchw_dtype == DType::BFLOAT16) {
            nchw_bytes.resize(nchw.size() * sizeof(uint16_t));
            auto* out_bf16 = reinterpret_cast<uint16_t*>(nchw_bytes.data());
            math::convert_float32_to_bf16(nchw.data(), nchw.size(), out_bf16);
        } else {
            nchw_bytes.resize(nchw.size() * sizeof(float));
            std::memcpy(nchw_bytes.data(), nchw.data(), nchw_bytes.size());
        }

        std::string nchw_id = result_id + "_vae_nchw_" + std::to_string(idx);
        ctx.store_in_memory(nchw_id, std::make_tuple(1, channels * height * width), nchw_dtype, std::move(nchw_bytes));

        Conv2DIm2ColConfig conv_cfg;
        conv_cfg.input_id = nchw_id;
        conv_cfg.weight_id = conv_weight_mem->matrix_id();
        conv_cfg.bias_id = conv_bias;
        conv_cfg.result_id = result_id + "_vae_conv_" + std::to_string(idx);
        conv_cfg.input_channels = channels;
        conv_cfg.input_height = height;
        conv_cfg.input_width = width;
        conv_cfg.output_channels = out_c;
        conv_cfg.kernel_h = k;
        conv_cfg.kernel_w = k;
        conv_cfg.stride_h = 1;
        conv_cfg.stride_w = 1;
        conv_cfg.pad_h = pad;
        conv_cfg.pad_w = pad;
        conv_cfg.output_dtype = DType::FLOAT32;
        auto conv_op = create_im2col_conv2d(conv_cfg);
        std::vector<std::string> conv_inputs{nchw_id, conv_weight_mem->matrix_id()};
        if (!conv_bias.empty()) {
            conv_inputs.push_back(conv_bias);
        }
        auto conv_out = exec_with_log(ctx, std::move(conv_op), conv_inputs);

        channels = out_c;
        int64_t out_h = (height + 2 * pad - k) / conv_cfg.stride_h + 1;
        int64_t out_w = (width + 2 * pad - k) / conv_cfg.stride_w + 1;
        height = out_h;
        width = out_w;

        ctx.clear_in_memory(nchw_id);
        if (current_id != conv_out->matrix_id()) {
            ctx.clear_in_memory(current_id);
        }
        current_id = conv_out->matrix_id();
        ++idx;
    }

    auto current = ctx.resolve_in_memory(current_id);
    if (!current) {
        throw std::runtime_error("VAE output missing: " + current_id);
    }

    auto final = cast_output_dtype(ctx, current, result_id, config_.output_dtype);
    if (final->matrix_id() != result_id) {
        final = ctx.store_in_memory(result_id, final->shape(), final->dtype(), final->data());
    }
    if (current_id != final->matrix_id()) {
        ctx.clear_in_memory(current_id);
    }
    return final;
}

StableDiffusionPipeline::StableDiffusionPipeline(const StableDiffusionConfig& config)
    : config_(config),
      text_encoder_(config.text_encoder),
      unet_(config.unet),
      vae_(config.vae) {}

std::shared_ptr<InMemoryMatrix> StableDiffusionPipeline::encode_prompt(
    const std::string& prompt,
    const std::string& result_id,
    OperatorContext& ctx,
    bool profile
) {
    int64_t max_len = config_.text_encoder.max_length;
    int64_t hidden = config_.text_encoder.hidden_size;
    if (max_len <= 0 || hidden <= 0) {
        throw std::runtime_error("SD encode_prompt: invalid text encoder config");
    }

    int64_t seq_len = 0;
    bool in_token = false;
    for (char ch : prompt) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (in_token) {
                ++seq_len;
                in_token = false;
            }
        } else {
            in_token = true;
        }
    }
    if (in_token) {
        ++seq_len;
    }
    if (seq_len <= 0) {
        seq_len = 1;
    }
    if (seq_len > max_len) {
        seq_len = max_len;
    }

    uint64_t seed = config_.scheduler.seed;
    size_t prompt_hash = std::hash<std::string>{}(prompt);
    seed ^= static_cast<uint64_t>(prompt_hash);
    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::normal_distribution<float> dist(0.0f, 0.02f);

    std::vector<float> embedding_data(static_cast<size_t>(seq_len * hidden));
    for (auto& v : embedding_data) {
        v = dist(rng);
    }
    std::vector<uint8_t> embedding_bytes(embedding_data.size() * sizeof(float));
    std::memcpy(embedding_bytes.data(), embedding_data.data(), embedding_bytes.size());

    std::string embed_id = result_id + "_rand_embed";
    ctx.store_in_memory(embed_id, std::make_tuple(seq_len, hidden), DType::FLOAT32, std::move(embedding_bytes));

    std::vector<int32_t> fake_ids(static_cast<size_t>(seq_len), 0);
    auto out = text_encoder_.forward(fake_ids, result_id, ctx, profile, embed_id);
    ctx.clear_in_memory(embed_id);
    return out;
}

std::shared_ptr<InMemoryMatrix> StableDiffusionPipeline::run_denoising_loop(
    const std::string& text_emb_id,
    const std::string& latents_id,
    const std::string& result_id,
    OperatorContext& ctx,
    bool profile
) {
    auto latents = ctx.resolve_in_memory(latents_id);
    if (!latents) {
        int64_t elements = config_.unet.latent_channels * config_.unet.sample_height * config_.unet.sample_width;
        Shape init_shape = {config_.unet.sample_height * config_.unet.sample_width, config_.unet.latent_channels};
        std::vector<float> zeros(static_cast<size_t>(elements), 0.0f);
        std::vector<uint8_t> bytes(zeros.size() * sizeof(float));
        std::memcpy(bytes.data(), zeros.data(), bytes.size());
        latents = ctx.store_in_memory(latents_id, init_shape, DType::FLOAT32, std::move(bytes));
    }

    std::string current_id = latents->matrix_id();
    for (int64_t i = 0; i < config_.scheduler.num_inference_steps; ++i) {
        ctx.set_prefetch_step(i);
        int64_t timestep = static_cast<int64_t>(config_.scheduler.num_inference_steps - i);
        auto step_result_id = result_id + "_step_" + std::to_string(i);
        ProfileStats stats;
        if (profile) {
            reset_profile(&stats);
            set_active_profile(&stats);
        }
        auto t_step_start = std::chrono::steady_clock::now();
        auto step_latents = unet_.forward(current_id, text_emb_id, step_result_id, ctx, timestep, profile);
        auto t_step_end = std::chrono::steady_clock::now();
        if (profile) {
            set_active_profile(nullptr);
        }
        double step_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
        if (profile) {
            double compute_ms = stats.compute_ns.load(std::memory_order_relaxed) / 1e6;
            double other_compute_ms = stats.other_compute_ns.load(std::memory_order_relaxed) / 1e6;
            double kv_ms = stats.kv_read_ns.load(std::memory_order_relaxed) / 1e6;
            double decomp_ms = stats.decompress_ns.load(std::memory_order_relaxed) / 1e6;
            double overhead_ms = step_ms - compute_ms - other_compute_ms - kv_ms - decomp_ms;
            if (overhead_ms < 0) {
                overhead_ms = 0;
            }
            double io_mb = static_cast<double>(stats.bytes_read.load(std::memory_order_relaxed)) / (1024.0 * 1024.0);
            double io_throughput = kv_ms > 0 ? (io_mb / (kv_ms / 1000.0)) : 0.0;
            double gemm_gflops = compute_ms > 0
                ? (static_cast<double>(stats.gemm_flops.load(std::memory_order_relaxed)) / 1e9) / (compute_ms / 1000.0)
                : 0.0;
            std::cout << "[Profile:SDUNet] step=" << i
                      << " timestep=" << timestep
                      << " total_ms=" << step_ms
                      << " compute_ms=" << compute_ms
                      << " other_compute_ms=" << other_compute_ms
                      << " kv_read_ms=" << kv_ms
                      << " decompress_ms=" << decomp_ms
                      << " overhead_ms=" << overhead_ms
                      << std::endl;
            std::cout << "[Profile:SDUNet] bytes_read_mb=" << io_mb
                      << " io_throughput_mb_s=" << io_throughput
                      << " io_lower_bound_ms=" << kv_ms
                      << std::endl;
            std::cout << "[Profile:SDUNet] gemm_flops=" << stats.gemm_flops.load(std::memory_order_relaxed)
                      << " gemm_throughput_gflops=" << gemm_gflops
                      << " compute_lower_bound_ms=" << compute_ms
                      << std::endl;
        } else {
            std::cout << "[SDUNetStep] step=" << i
                      << " timestep=" << timestep
                      << " ms=" << step_ms
                      << std::endl;
        }
        if (step_latents->matrix_id() != current_id) {
            ctx.clear_in_memory(current_id);
        }
        current_id = step_latents->matrix_id();
    }
    ctx.set_prefetch_step(0);

    auto final = ctx.resolve_in_memory(current_id);
    if (!final) {
        throw std::runtime_error("Denoising loop produced no output");
    }
    if (final->matrix_id() != result_id) {
        auto out = ctx.store_in_memory(result_id, final->shape(), final->dtype(), final->data());
        ctx.clear_in_memory(final->matrix_id());
        return out;
    }
    return final;
}

std::shared_ptr<InMemoryMatrix> StableDiffusionPipeline::decode_latents(
    const std::string& latents_id,
    const std::string& result_id,
    OperatorContext& ctx,
    bool profile
) {
    return vae_.decode(latents_id, result_id, ctx, profile);
}

std::shared_ptr<InMemoryMatrix> StableDiffusionPipeline::generate(
    const std::string& prompt,
    const std::string& result_id,
    OperatorContext& ctx,
    bool profile
) {
    auto text_emb = encode_prompt(prompt, result_id + "_text_emb", ctx, profile);
    auto latents = ctx.store_in_memory(
        result_id + "_latents_init",
        {config_.unet.sample_height * config_.unet.sample_width, config_.unet.latent_channels},
        DType::FLOAT32,
        std::vector<uint8_t>(
            static_cast<size_t>(config_.unet.latent_channels *
                                config_.unet.sample_height *
                                config_.unet.sample_width) *
            sizeof(float),
            0
        )
    );
    auto denoised = run_denoising_loop(text_emb->matrix_id(), latents->matrix_id(), result_id + "_denoised", ctx, profile);
    ctx.clear_in_memory(text_emb->matrix_id());
    ctx.clear_in_memory(latents->matrix_id());
    auto decoded = decode_latents(denoised->matrix_id(), result_id, ctx, profile);
    if (denoised->matrix_id() != decoded->matrix_id()) {
        ctx.clear_in_memory(denoised->matrix_id());
    }
    return decoded;
}

} // namespace kvtensor

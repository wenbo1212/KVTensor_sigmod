#include "kvtensor/stable_diffusion_inference.hpp"
#include "kvtensor/preload_utils.hpp"
#include "kvtensor/model_utils.hpp"
#include "kvtensor/bufferpool.hpp"
#include <chrono>
#include <thread>
#include <iostream>

namespace kvtensor {

namespace {
std::vector<std::string> build_prefetch_ids(const StableDiffusionConfig& config) {
    std::vector<std::string> ids;
    auto add = [&](const std::string& id) {
        if (!id.empty()) {
            ids.push_back(id);
        }
    };
    auto add_text_block = [&](const std::string& prefix) {
        if (prefix.empty()) {
            return;
        }
        add(prefix + ".norm1_weight");
        add(prefix + ".attn_qkv_proj");
        add(prefix + ".attn_q_proj");
        add(prefix + ".attn_k_proj");
        add(prefix + ".attn_v_proj");
        add(prefix + ".attn_out_proj");
        add(prefix + ".norm2_weight");
        add(prefix + ".ffn_gate_up_proj");
        add(prefix + ".ffn_down_proj");
    };
    auto add_resnet = [&](const std::string& prefix, bool shortcut, std::vector<std::string>& out) {
        out.push_back(prefix + ".norm1.weight");
        out.push_back(prefix + ".norm1.bias");
        out.push_back(prefix + ".conv1.weight");
        out.push_back(prefix + ".conv1.bias");
        out.push_back(prefix + ".time_emb_proj.weight");
        out.push_back(prefix + ".time_emb_proj.bias");
        out.push_back(prefix + ".norm2.weight");
        out.push_back(prefix + ".norm2.bias");
        out.push_back(prefix + ".conv2.weight");
        out.push_back(prefix + ".conv2.bias");
        if (shortcut) {
            out.push_back(prefix + ".conv_shortcut.weight");
            out.push_back(prefix + ".conv_shortcut.bias");
        }
    };
    auto add_attention = [&](const std::string& prefix, std::vector<std::string>& out) {
        out.push_back(prefix + ".group_norm.weight");
        out.push_back(prefix + ".group_norm.bias");
        out.push_back(prefix + ".attn_qkv_proj");
        out.push_back(prefix + ".attn_out_proj");
    };

    // Text encoder (once).
    for (const auto& id : config.text_encoder.transformer_block_ids) {
        add_text_block(id);
    }
    add(config.text_encoder.final_layer_norm_id);

    // UNet weights are preloaded into memory; prefetch only text + VAE.

    // VAE decoder (once).
    for (const auto& id : config.vae.decoder_weight_ids) {
        add(id + ".conv_weight");
        add(id + ".conv_bias");
    }
    return ids;
}

std::vector<std::string> build_unet_preload_ids() {
    std::vector<std::string> ids;
    auto add = [&](const std::string& id) {
        if (!id.empty()) {
            ids.push_back(id);
        }
    };
    auto add_resnet = [&](const std::string& prefix, bool shortcut) {
        add(prefix + ".norm1.weight");
        add(prefix + ".norm1.bias");
        add(prefix + ".conv1.weight");
        add(prefix + ".conv1.bias");
        add(prefix + ".time_emb_proj.weight");
        add(prefix + ".time_emb_proj.bias");
        add(prefix + ".norm2.weight");
        add(prefix + ".norm2.bias");
        add(prefix + ".conv2.weight");
        add(prefix + ".conv2.bias");
        if (shortcut) {
            add(prefix + ".conv_shortcut.weight");
            add(prefix + ".conv_shortcut.bias");
        }
    };
    auto add_transformer = [&](const std::string& prefix) {
        add(prefix + ".proj_in.weight");
        add(prefix + ".proj_in.bias");
        add(prefix + ".proj_out.weight");
        add(prefix + ".proj_out.bias");

        std::string blk = prefix + ".transformer_blocks.0";
        add(blk + ".norm1.weight");
        add(blk + ".norm1.bias");
        add(blk + ".norm2.weight");
        add(blk + ".norm2.bias");
        add(blk + ".norm3.weight");
        add(blk + ".norm3.bias");

        add(blk + ".attn1.to_q.weight");
        add(blk + ".attn1.to_k.weight");
        add(blk + ".attn1.to_v.weight");
        add(blk + ".attn1.to_out.weight");
        add(blk + ".attn1.to_out.bias");

        add(blk + ".attn2.to_q.weight");
        add(blk + ".attn2.to_k.weight");
        add(blk + ".attn2.to_v.weight");
        add(blk + ".attn2.to_out.weight");
        add(blk + ".attn2.to_out.bias");

        add(blk + ".ff.proj.weight");
        add(blk + ".ff.proj.bias");
        add(blk + ".ff.out.weight");
        add(blk + ".ff.out.bias");
    };

    add("unet.conv_in.weight");
    add("unet.conv_in.bias");
    add("unet.time_embedding.linear_1.weight");
    add("unet.time_embedding.linear_1.bias");
    add("unet.time_embedding.linear_2.weight");
    add("unet.time_embedding.linear_2.bias");

    add_resnet("unet.down_blocks.0.resnets.0", false);
    add_transformer("unet.down_blocks.0.attentions.0");
    add_resnet("unet.down_blocks.0.resnets.1", false);
    add_transformer("unet.down_blocks.0.attentions.1");
    add("unet.down_blocks.0.downsamplers.0.conv.weight");
    add("unet.down_blocks.0.downsamplers.0.conv.bias");

    add_resnet("unet.down_blocks.1.resnets.0", true);
    add_transformer("unet.down_blocks.1.attentions.0");
    add_resnet("unet.down_blocks.1.resnets.1", false);
    add_transformer("unet.down_blocks.1.attentions.1");
    add("unet.down_blocks.1.downsamplers.0.conv.weight");
    add("unet.down_blocks.1.downsamplers.0.conv.bias");

    add_resnet("unet.down_blocks.2.resnets.0", true);
    add_transformer("unet.down_blocks.2.attentions.0");
    add_resnet("unet.down_blocks.2.resnets.1", false);
    add_transformer("unet.down_blocks.2.attentions.1");
    add("unet.down_blocks.2.downsamplers.0.conv.weight");
    add("unet.down_blocks.2.downsamplers.0.conv.bias");

    add_resnet("unet.down_blocks.3.resnets.0", false);
    add_resnet("unet.down_blocks.3.resnets.1", false);

    add_resnet("unet.mid_block.resnets.0", false);
    add_transformer("unet.mid_block.attentions.0");
    add_resnet("unet.mid_block.resnets.1", false);

    for (int i = 0; i < 3; ++i) {
        add_resnet("unet.up_blocks.0.resnets." + std::to_string(i), true);
    }
    add("unet.up_blocks.0.upsamplers.0.conv.weight");
    add("unet.up_blocks.0.upsamplers.0.conv.bias");

    for (int i = 0; i < 3; ++i) {
        add_resnet("unet.up_blocks.1.resnets." + std::to_string(i), true);
        add_transformer("unet.up_blocks.1.attentions." + std::to_string(i));
    }
    add("unet.up_blocks.1.upsamplers.0.conv.weight");
    add("unet.up_blocks.1.upsamplers.0.conv.bias");

    for (int i = 0; i < 3; ++i) {
        add_resnet("unet.up_blocks.2.resnets." + std::to_string(i), true);
        add_transformer("unet.up_blocks.2.attentions." + std::to_string(i));
    }
    add("unet.up_blocks.2.upsamplers.0.conv.weight");
    add("unet.up_blocks.2.upsamplers.0.conv.bias");

    for (int i = 0; i < 3; ++i) {
        add_resnet("unet.up_blocks.3.resnets." + std::to_string(i), true);
        add_transformer("unet.up_blocks.3.attentions." + std::to_string(i));
    }

    add("unet.conv_norm_out.weight");
    add("unet.conv_norm_out.bias");
    add("unet.conv_out.weight");
    add("unet.conv_out.bias");
    return ids;
}
} // namespace

StableDiffusionInference::StableDiffusionInference(const StableDiffusionInferenceConfig& config)
    : pipeline_(config.pipeline),
      prefetch_config_(config.prefetch),
      preload_file_path_(config.preload_file_path),
      in_memory_(config.in_memory),
      prefetch_matrix_ids_(build_prefetch_ids(config.pipeline)) {}

void StableDiffusionInference::ensure_preload_ids() {
    if (preload_initialized_) {
        return;
    }
    preload_ids_ = read_preload_matrix_ids(preload_file_path_);
    preload_initialized_ = true;
}

double StableDiffusionInference::ensure_prefetch_warmup(OperatorContext& ctx) const {
    if (ctx.buffer_pool() == nullptr) {
        return 0.0;
    }

    auto t_fill_start = std::chrono::steady_clock::now();
    BufferPool::Stats stats = ctx.buffer_pool()->get_stats();
    const size_t target = stats.slot_capacity;
    const double memory_fill_ratio = 0.90;
    const auto max_wait = std::chrono::milliseconds(500);
    while (stats.prefetch_active &&
           stats.cached_chunks < target &&
           (stats.memory_total_bytes == 0 ||
            stats.memory_used_bytes < static_cast<size_t>(stats.memory_total_bytes * memory_fill_ratio)) &&
           (std::chrono::steady_clock::now() - t_fill_start) < max_wait) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        stats = ctx.buffer_pool()->get_stats();
    }
    auto t_fill_end = std::chrono::steady_clock::now();
    double fill_ms = std::chrono::duration<double, std::milli>(t_fill_end - t_fill_start).count();
    std::cout << "[Prefetch] Initial fill: cached "
              << stats.cached_chunks << " slots of " << target
              << ", sequence_length=";
    if (stats.sequence_length > 0) {
        std::cout << stats.sequence_length;
    } else {
        std::cout << "unknown";
    }
    std::cout << ", time=" << fill_ms << " ms" << std::endl;
    return fill_ms;
}

void StableDiffusionInference::ensure_in_memory_weights(OperatorContext& ctx) {
    if (in_memory_initialized_) {
        return;
    }
    std::unordered_set<std::string> all_ids;
    if (ctx.registry()) {
        auto ids = ctx.registry()->list_matrix_ids();
        all_ids.insert(ids.begin(), ids.end());
    }
    if (!all_ids.empty()) {
        preload_matrices(all_ids, ctx);
    }
    in_memory_initialized_ = true;
}

std::shared_ptr<InMemoryMatrix> StableDiffusionInference::generate(
    const std::string& prompt,
    const std::string& result_id,
    OperatorContext& ctx,
    bool profile
) {
    ensure_preload_ids();
    last_runtime_trace_ = StableDiffusionRuntimeTrace{};
    last_runtime_trace_.profile_enabled = profile;
    ctx.set_simulated_get_latency_ms(prefetch_config_.simulate_get_latency_ms);
    ctx.set_simulate_get_chunks(prefetch_config_.simulate_prefetch);
    if (profile) {
        reset_bufferpool_profile();
    }

    if (in_memory_) {
        if (ctx.buffer_pool() != nullptr) {
            ctx.stop_weight_prefetch();
            ctx.set_buffer_pool(nullptr);
        }
        auto preload_start = std::chrono::steady_clock::now();
        ensure_in_memory_weights(ctx);
        auto preload_end = std::chrono::steady_clock::now();
        last_runtime_trace_.preload_ms =
            std::chrono::duration<double, std::milli>(preload_end - preload_start).count();
        return pipeline_.generate(prompt, result_id, ctx, profile);
    }

    if (ctx.buffer_pool() != nullptr) {
        ctx.stop_weight_prefetch();
    }

    auto unet_ids = build_unet_preload_ids();
    std::unordered_set<std::string> all_preload_ids = preload_ids_;
    all_preload_ids.insert(unet_ids.begin(), unet_ids.end());

    if (!all_preload_ids.empty()) {
        auto preload_start = std::chrono::steady_clock::now();
        preload_matrices(all_preload_ids, ctx);
        auto preload_end = std::chrono::steady_clock::now();
        last_runtime_trace_.preload_ms =
            std::chrono::duration<double, std::milli>(preload_end - preload_start).count();
    }

    start_or_restart_prefetch(ctx, ctx.storage(), prefetch_config_,
                              prefetch_matrix_ids_, all_preload_ids);

    last_runtime_trace_.prefetch_warmup_ms = ensure_prefetch_warmup(ctx);

    auto result = pipeline_.generate(prompt, result_id, ctx, profile);
    last_runtime_trace_.used_bufferpool = ctx.buffer_pool() != nullptr;
    if (ctx.buffer_pool() != nullptr) {
        auto bp_profile = ctx.buffer_pool()->get_profile_stats();
        auto bp_stats = ctx.buffer_pool()->get_stats();
        last_runtime_trace_.bufferpool_get_chunk_calls = bp_profile.get_chunk_calls;
        last_runtime_trace_.bufferpool_cache_hits = bp_profile.cache_hits;
        last_runtime_trace_.bufferpool_cache_misses = bp_profile.cache_misses;
        last_runtime_trace_.bufferpool_wait_ms =
            static_cast<double>(bp_profile.total_wait_time_ns) / 1e6;
        last_runtime_trace_.bufferpool_max_wait_ms =
            static_cast<double>(bp_profile.max_wait_time_ns) / 1e6;
        last_runtime_trace_.bufferpool_evict_count = bp_profile.evict_count;
        last_runtime_trace_.bufferpool_prefetch_get_calls = bp_profile.prefetch_get_calls;
        last_runtime_trace_.bufferpool_prefetch_get_time_ns = bp_profile.prefetch_get_time_ns;
        last_runtime_trace_.bufferpool_cached_chunks = bp_stats.cached_chunks;
        last_runtime_trace_.bufferpool_slot_capacity = bp_stats.slot_capacity;
        last_runtime_trace_.bufferpool_memory_used_bytes = bp_stats.memory_used_bytes;
        last_runtime_trace_.bufferpool_memory_total_bytes = bp_stats.memory_total_bytes;
        last_runtime_trace_.bufferpool_consumption_position = bp_stats.consumption_position;
        last_runtime_trace_.bufferpool_prefetch_position = bp_stats.prefetch_position;
        last_runtime_trace_.bufferpool_sequence_length = bp_stats.sequence_length;
        if (bp_profile.get_chunk_calls > 0) {
            last_runtime_trace_.bufferpool_hit_rate =
                static_cast<double>(bp_profile.cache_hits) /
                static_cast<double>(bp_profile.get_chunk_calls);
        }
    }

    return result;
}

} // namespace kvtensor

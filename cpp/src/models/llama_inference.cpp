#include "kvtensor/llama_inference.hpp"
#include "kvtensor/preload_utils.hpp"
#include "kvtensor/model_utils.hpp"
#include "kvtensor/bufferpool.hpp"
#include "kvtensor/profile.hpp"
#include <chrono>
#include <algorithm>
#include <iostream>
#include <thread>
#include <unordered_set>

namespace kvtensor {

namespace {
std::string dtype_to_string_local(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return "float32";
        case DType::BFLOAT16: return "bfloat16";
        case DType::FLOAT16: return "float16";
        case DType::INT8: return "int8";
        default: return "unknown";
    }
}

std::string split_mode_to_string_local(SplitMode mode) {
    return mode == SplitMode::COLUMN ? "column" : "row";
}

int preload_priority_rank(const std::string& matrix_id) {
    if (matrix_id == "output.output_proj") {
        return 4;
    }
    if (matrix_id.find(".ffn_gate_up_proj") != std::string::npos) {
        return 0;
    }
    if (matrix_id.find(".ffn_down_proj") != std::string::npos) {
        return 1;
    }
    if (matrix_id.find(".attn_qkv_proj") != std::string::npos) {
        return 2;
    }
    if (matrix_id.find(".attn_o_proj") != std::string::npos) {
        return 3;
    }
    return -1;
}

std::string preload_group_name(const std::string& matrix_id) {
    const int rank = preload_priority_rank(matrix_id);
    switch (rank) {
        case 0: return "ffn_gate_up_proj";
        case 1: return "ffn_down_proj";
        case 2: return "attn_qkv_proj";
        case 3: return "attn_o_proj";
        case 4: return "output_proj";
        default: return "non_preload";
    }
}

std::string matrix_group_name(const std::string& matrix_id) {
    if (matrix_id == "embedding.token_embedding") {
        return "token_embedding";
    }
    if (matrix_id == "output.output_proj") {
        return "output_proj";
    }
    if (matrix_id.find(".attn_qkv_proj") != std::string::npos) {
        return "attn_qkv_proj";
    }
    if (matrix_id.find(".attn_o_proj") != std::string::npos) {
        return "attn_o_proj";
    }
    if (matrix_id.find(".ffn_gate_up_proj") != std::string::npos) {
        return "ffn_gate_up_proj";
    }
    if (matrix_id.find(".ffn_down_proj") != std::string::npos) {
        return "ffn_down_proj";
    }
    if (matrix_id.find("norm_weight") != std::string::npos) {
        return "norm_weight";
    }
    return "other";
}

std::vector<std::string> build_prefetch_ids(const LlamaModelConfig& config) {
    std::unordered_set<std::string> seen_ids;
    std::vector<std::string> ids;

    auto add = [&](const std::string& id) {
        if (!id.empty() && seen_ids.insert(id).second) {
            ids.push_back(id);
        }
    };

    for (const auto& block_config : config.blocks) {
        add(block_config.attn_qkv_proj_id);
        add(block_config.attn_o_proj_id);
        add(block_config.ffn_gate_up_proj_id);
        add(block_config.ffn_down_proj_id);
    }
    add(config.output_proj_id);
    return ids;
}

LlamaPhaseTrace build_phase_trace(
    double total_ms,
    double prefetch_warmup_ms,
    const ProfileStats& stats,
    const BufferPool::ProfileStats* bufferpool_profile,
    const BufferPool::Stats* bufferpool_stats
) {
    LlamaPhaseTrace trace;
    trace.elapsed_ms = total_ms;
    trace.prefetch_warmup_ms = prefetch_warmup_ms;
    trace.compute_ms = stats.compute_ns.load(std::memory_order_relaxed) / 1e6;
    trace.other_compute_ms = stats.other_compute_ns.load(std::memory_order_relaxed) / 1e6;
    trace.kv_read_ms = stats.kv_read_ns.load(std::memory_order_relaxed) / 1e6;
    trace.decompress_ms = stats.decompress_ns.load(std::memory_order_relaxed) / 1e6;
    trace.bytes_read = stats.bytes_read.load(std::memory_order_relaxed);
    trace.gemm_flops = stats.gemm_flops.load(std::memory_order_relaxed);

    trace.overhead_ms = total_ms - trace.compute_ms - trace.other_compute_ms - trace.kv_read_ms - trace.decompress_ms;
    if (trace.overhead_ms < 0.0) {
        trace.overhead_ms = 0.0;
    }

    {
        std::lock_guard<std::mutex> lock(stats.detail_mutex);
        trace.matrix_accesses.reserve(stats.matrix_reads.size());
        for (const auto& [_, entry] : stats.matrix_reads) {
            LlamaPhaseTrace::MatrixAccess access;
            access.matrix_id = entry.matrix_id;
            access.matrix_group = matrix_group_name(entry.matrix_id);
            access.preload_group = preload_group_name(entry.matrix_id);
            access.priority_rank = preload_priority_rank(entry.matrix_id);
            access.rows = std::get<0>(entry.matrix_shape);
            access.cols = std::get<1>(entry.matrix_shape);
            access.chunk_size = entry.chunk_size;
            access.dtype = dtype_to_string_local(entry.dtype);
            access.split_mode = split_mode_to_string_local(entry.split_mode);
            access.bytes_read = entry.bytes_read;
            access.chunk_reads = entry.chunk_reads;
            trace.matrix_accesses.push_back(std::move(access));
        }
        std::sort(trace.matrix_accesses.begin(), trace.matrix_accesses.end(), [](const auto& a, const auto& b) {
            return a.matrix_id < b.matrix_id;
        });

        trace.gemm_buckets.reserve(stats.gemm_buckets.size());
        for (const auto& [_, entry] : stats.gemm_buckets) {
            LlamaPhaseTrace::GemmBucket bucket;
            bucket.operator_class = entry.operator_class;
            bucket.m = entry.m;
            bucket.k = entry.k;
            bucket.n = entry.n;
            bucket.calls = entry.calls;
            bucket.flops = entry.flops;
            trace.gemm_buckets.push_back(std::move(bucket));
        }
        std::sort(trace.gemm_buckets.begin(), trace.gemm_buckets.end(), [](const auto& a, const auto& b) {
            if (a.operator_class != b.operator_class) {
                return a.operator_class < b.operator_class;
            }
            if (a.m != b.m) {
                return a.m < b.m;
            }
            if (a.k != b.k) {
                return a.k < b.k;
            }
            return a.n < b.n;
        });
    }

    if (bufferpool_profile != nullptr) {
        trace.bufferpool_get_chunk_calls = bufferpool_profile->get_chunk_calls;
        trace.bufferpool_cache_misses = bufferpool_profile->cache_misses;
        trace.bufferpool_wait_ms = static_cast<double>(bufferpool_profile->total_wait_time_ns) / 1e6;
    }

    if (bufferpool_stats != nullptr) {
        trace.bufferpool_memory_total_bytes = bufferpool_stats->memory_total_bytes;
    }

    return trace;
}

void print_phase_trace(const LlamaPhaseTrace& trace) {
    std::cout << "[Profile:Llama] total_ms=" << trace.elapsed_ms
              << " prefetch_fill_ms=" << trace.prefetch_warmup_ms
              << " compute_ms=" << trace.compute_ms
              << " other_compute_ms=" << trace.other_compute_ms
              << " kv_read_ms=" << trace.kv_read_ms
              << " decompress_ms=" << trace.decompress_ms
              << " overhead_ms=" << trace.overhead_ms
              << std::endl;
    std::cout << "[Profile:Llama] bytes_read_mb=" << (static_cast<double>(trace.bytes_read) / (1024.0 * 1024.0))
              << " io_lower_bound_ms=" << trace.kv_read_ms
              << std::endl;
    std::cout << "[Profile:Llama] gemm_flops=" << trace.gemm_flops
              << " compute_lower_bound_ms=" << trace.compute_ms
              << std::endl;
    if (!trace.matrix_accesses.empty()) {
        std::cout << "[Profile:Llama] matrix_accesses=" << trace.matrix_accesses.size()
                  << " gemm_buckets=" << trace.gemm_buckets.size()
                  << std::endl;
    }
    if (trace.bufferpool_get_chunk_calls > 0 || trace.bufferpool_cache_misses > 0 || trace.bufferpool_wait_ms > 0.0) {
        std::cout << "[Profile:BufferPool] get_chunk_calls=" << trace.bufferpool_get_chunk_calls
                  << " cache_misses=" << trace.bufferpool_cache_misses
                  << " wait_ms=" << trace.bufferpool_wait_ms
                  << " memory_total_bytes=" << trace.bufferpool_memory_total_bytes
                  << std::endl;
    }
}
} // namespace

LlamaInference::LlamaInference(const LlamaInferenceConfig& config)
    : model_(config.model),
      prefetch_config_(config.prefetch),
      preload_file_path_(config.preload_file_path),
      in_memory_(config.in_memory),
      disable_bufferpool_(config.disable_bufferpool),
      prefetch_matrix_ids_(build_prefetch_ids(config.model)) {}

void LlamaInference::ensure_preload_ids() {
    if (preload_initialized_) {
        return;
    }
    preload_ids_ = read_preload_matrix_ids(preload_file_path_);
    preload_initialized_ = true;
}

double LlamaInference::ensure_prefetch_warmup(OperatorContext& ctx) const {
    if (ctx.buffer_pool() == nullptr) {
        return 0.0;
    }

    auto t_fill_start = std::chrono::steady_clock::now();
    BufferPool::Stats stats = ctx.buffer_pool()->get_stats();
    const size_t target = stats.slot_capacity;
    const double memory_fill_ratio = 0.95;
    const auto max_wait = std::chrono::milliseconds(15000);
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

void LlamaInference::ensure_in_memory_weights(OperatorContext& ctx) {
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

double LlamaInference::preload_static_weights(OperatorContext& ctx) {
    ensure_preload_ids();
    last_prefetch_warmup_ms_ = 0.0;
    last_phase_trace_ = LlamaPhaseTrace{};

    auto t_start = std::chrono::steady_clock::now();
    if (!preload_ids_.empty()) {
        preload_matrices(preload_ids_, ctx);
    }
    auto t_end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t_end - t_start).count();
}

std::shared_ptr<InMemoryMatrix> LlamaInference::forward(
    const std::vector<int32_t>& input_ids,
    const std::string& result_id,
    OperatorContext& ctx,
    const std::string& mask_id,
    const std::vector<std::pair<std::string, std::string>>& kv_cache_ids,
    const std::vector<std::pair<std::string, std::string>>& cache_output_ids,
    const std::string& input_embeddings_id,
    bool profile
) {
    ensure_preload_ids();
    last_prefetch_warmup_ms_ = 0.0;
    last_phase_trace_ = LlamaPhaseTrace{};

    ProfileStats stats;
    auto t_start = std::chrono::steady_clock::now();
    if (profile) {
        reset_profile(&stats);
        set_active_profile(&stats);
        reset_bufferpool_profile();
    }
    ctx.set_simulated_get_latency_ms(prefetch_config_.simulate_get_latency_ms);
    ctx.set_simulate_get_chunks(prefetch_config_.simulate_prefetch);

    if (in_memory_) {
        if (disable_bufferpool_) {
            ctx.set_use_bufferpool(false);
        }
        if (ctx.buffer_pool() != nullptr) {
            ctx.stop_weight_prefetch();
            ctx.set_buffer_pool(nullptr);
        }
        ensure_in_memory_weights(ctx);
        auto out = model_.forward(
            input_ids,
            result_id,
            ctx,
            mask_id,
            kv_cache_ids,
            cache_output_ids,
            input_embeddings_id,
            profile
        );
        if (profile) {
            set_active_profile(nullptr);
        }
        auto t_end = std::chrono::steady_clock::now();
        if (profile) {
            double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            total_ms -= last_prefetch_warmup_ms_;
            if (total_ms < 0) {
                total_ms = 0;
            }
            last_phase_trace_ = build_phase_trace(total_ms, last_prefetch_warmup_ms_, stats, nullptr, nullptr);
            print_phase_trace(last_phase_trace_);
        }
        return out;
    }

    if (disable_bufferpool_) {
        ctx.set_use_bufferpool(false);
        if (ctx.buffer_pool() != nullptr) {
            ctx.stop_weight_prefetch();
            ctx.set_buffer_pool(nullptr);
        }
    } else {
        ctx.set_use_bufferpool(true);
    }

    // Stop any ongoing prefetch before starting a new cycle
    if (!disable_bufferpool_ && ctx.buffer_pool() != nullptr) {
        ctx.stop_weight_prefetch();
    }

    // Preload weights into memory when configured
    if (!preload_ids_.empty()) {
        preload_matrices(preload_ids_, ctx);
    }

    if (!disable_bufferpool_) {
        // Start (or restart) weight prefetching
        start_or_restart_prefetch(ctx, ctx.storage(), prefetch_config_,
                                  prefetch_matrix_ids_, preload_ids_);

        last_prefetch_warmup_ms_ = ensure_prefetch_warmup(ctx);
    }

    auto out = model_.forward(
        input_ids,
        result_id,
        ctx,
        mask_id,
        kv_cache_ids,
        cache_output_ids,
        input_embeddings_id,
        profile
    );
    if (profile) {
        set_active_profile(nullptr);
    }
    auto t_end = std::chrono::steady_clock::now();
    if (profile) {
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        total_ms -= last_prefetch_warmup_ms_;
        if (total_ms < 0) {
            total_ms = 0;
        }
        BufferPool::ProfileStats bp_profile{};
        BufferPool::Stats bp_stats{};
        BufferPool::ProfileStats* bp_profile_ptr = nullptr;
        BufferPool::Stats* bp_stats_ptr = nullptr;
        if (ctx.buffer_pool() != nullptr) {
            bp_profile = ctx.buffer_pool()->get_profile_stats();
            bp_stats = ctx.buffer_pool()->get_stats();
            bp_profile_ptr = &bp_profile;
            bp_stats_ptr = &bp_stats;
        }
        last_phase_trace_ = build_phase_trace(
            total_ms,
            last_prefetch_warmup_ms_,
            stats,
            bp_profile_ptr,
            bp_stats_ptr
        );
        print_phase_trace(last_phase_trace_);
    }
    return out;
}

} // namespace kvtensor

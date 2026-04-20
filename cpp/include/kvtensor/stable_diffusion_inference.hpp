#pragma once

#include "kvtensor/stable_diffusion.hpp"
#include "kvtensor/model_utils.hpp"
#include <unordered_set>
#include <vector>

namespace kvtensor {

struct StableDiffusionInferenceConfig {
    StableDiffusionConfig pipeline;
    PrefetchConfig prefetch;
    std::string preload_file_path;
    bool in_memory = false;
};

struct StableDiffusionRuntimeTrace {
    bool profile_enabled = false;
    bool used_bufferpool = false;
    double preload_ms = 0.0;
    double prefetch_warmup_ms = 0.0;
    double bufferpool_wait_ms = 0.0;
    double bufferpool_max_wait_ms = 0.0;
    double bufferpool_hit_rate = 0.0;
    uint64_t bufferpool_get_chunk_calls = 0;
    uint64_t bufferpool_cache_hits = 0;
    uint64_t bufferpool_cache_misses = 0;
    uint64_t bufferpool_evict_count = 0;
    uint64_t bufferpool_prefetch_get_calls = 0;
    uint64_t bufferpool_prefetch_get_time_ns = 0;
    size_t bufferpool_cached_chunks = 0;
    size_t bufferpool_slot_capacity = 0;
    size_t bufferpool_memory_used_bytes = 0;
    size_t bufferpool_memory_total_bytes = 0;
    size_t bufferpool_consumption_position = 0;
    size_t bufferpool_prefetch_position = 0;
    size_t bufferpool_sequence_length = 0;
};

class StableDiffusionInference {
public:
    explicit StableDiffusionInference(const StableDiffusionInferenceConfig& config);

    std::shared_ptr<InMemoryMatrix> generate(
        const std::string& prompt,
        const std::string& result_id,
        OperatorContext& ctx,
        bool profile = false
    );

    const std::vector<std::string>& prefetch_matrix_ids() const { return prefetch_matrix_ids_; }
    const std::unordered_set<std::string>& preload_ids() const { return preload_ids_; }
    const StableDiffusionRuntimeTrace& last_runtime_trace() const { return last_runtime_trace_; }

private:
    void ensure_preload_ids();
    double ensure_prefetch_warmup(OperatorContext& ctx) const;
    void ensure_in_memory_weights(OperatorContext& ctx);

    StableDiffusionPipeline pipeline_;
    PrefetchConfig prefetch_config_;
    std::string preload_file_path_;
    bool in_memory_ = false;
    bool in_memory_initialized_ = false;

    bool preload_initialized_ = false;
    std::unordered_set<std::string> preload_ids_;
    std::vector<std::string> prefetch_matrix_ids_;
    StableDiffusionRuntimeTrace last_runtime_trace_;
};

} // namespace kvtensor

#pragma once

#include "kvtensor/llama.hpp"
#include "kvtensor/model_utils.hpp"
#include <unordered_set>
#include <vector>

namespace kvtensor {

struct LlamaInferenceConfig {
    LlamaModelConfig model;
    PrefetchConfig prefetch;
    std::string preload_file_path;
    bool in_memory = false;
    bool disable_bufferpool = false;
};

struct LlamaPhaseTrace {
    double elapsed_ms = 0.0;
    double prefetch_warmup_ms = 0.0;
    double compute_ms = 0.0;
    double other_compute_ms = 0.0;
    double kv_read_ms = 0.0;
    double decompress_ms = 0.0;
    double overhead_ms = 0.0;
    double bufferpool_wait_ms = 0.0;
    uint64_t bytes_read = 0;
    uint64_t gemm_flops = 0;
    uint64_t bufferpool_get_chunk_calls = 0;
    uint64_t bufferpool_cache_misses = 0;
    size_t bufferpool_memory_total_bytes = 0;
    struct MatrixAccess {
        std::string matrix_id;
        std::string matrix_group;
        std::string preload_group;
        int priority_rank = -1;
        int64_t rows = 0;
        int64_t cols = 0;
        int64_t chunk_size = 0;
        std::string dtype;
        std::string split_mode;
        uint64_t bytes_read = 0;
        uint64_t chunk_reads = 0;
    };
    struct GemmBucket {
        std::string operator_class;
        int64_t m = 0;
        int64_t k = 0;
        int64_t n = 0;
        uint64_t calls = 0;
        uint64_t flops = 0;
    };
    std::vector<MatrixAccess> matrix_accesses;
    std::vector<GemmBucket> gemm_buckets;
};

class LlamaInference {
public:
    explicit LlamaInference(const LlamaInferenceConfig& config);

    std::shared_ptr<InMemoryMatrix> forward(
        const std::vector<int32_t>& input_ids,
        const std::string& result_id,
        OperatorContext& ctx,
        const std::string& mask_id = "",
        const std::vector<std::pair<std::string, std::string>>& kv_cache_ids = {},
        const std::vector<std::pair<std::string, std::string>>& cache_output_ids = {},
        const std::string& input_embeddings_id = "",
        bool profile = false
    );

    double preload_static_weights(OperatorContext& ctx);

    const std::vector<std::string>& prefetch_matrix_ids() const { return prefetch_matrix_ids_; }
    const std::unordered_set<std::string>& preload_ids() const { return preload_ids_; }
    double last_prefetch_warmup_ms() const { return last_prefetch_warmup_ms_; }
    const LlamaPhaseTrace& last_phase_trace() const { return last_phase_trace_; }

private:
    void ensure_preload_ids();
    double ensure_prefetch_warmup(OperatorContext& ctx) const;
    void ensure_in_memory_weights(OperatorContext& ctx);

    LlamaModel model_;
    PrefetchConfig prefetch_config_;
    std::string preload_file_path_;
    bool in_memory_ = false;
    bool in_memory_initialized_ = false;
    bool disable_bufferpool_ = false;

    bool preload_initialized_ = false;
    std::unordered_set<std::string> preload_ids_;
    std::vector<std::string> prefetch_matrix_ids_;
    double last_prefetch_warmup_ms_ = 0.0;
    LlamaPhaseTrace last_phase_trace_;
};

} // namespace kvtensor

#pragma once

#include "kvtensor/context.hpp"
#include "kvtensor/types.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kvtensor {

// Forward declarations
class OperatorContext;
class InMemoryMatrix;

// Llama transformer block configuration
struct LlamaTransformerBlockConfig {
    std::string attn_qkv_proj_id;
    std::string attn_o_proj_id;
    std::string ffn_gate_up_proj_id;
    std::string ffn_down_proj_id;
    std::string attn_norm_weight_id;
    std::string ffn_norm_weight_id;
    
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t head_dim;
    int64_t hidden_dim;
    int64_t chunk_size = 1024;
    float rms_norm_eps = 1e-6f;
};

// Llama transformer block
class LlamaTransformerBlock {
public:
    explicit LlamaTransformerBlock(const LlamaTransformerBlockConfig& config)
        : config_(config) {}

    std::shared_ptr<InMemoryMatrix> forward(
        const std::string& x_id,
        const std::string& result_id,
        OperatorContext& ctx,
        const std::string& mask_id = "",
        const std::string& kv_cache_k_id = "",
        const std::string& kv_cache_v_id = "",
        const std::string& cache_k_id = "",
        const std::string& cache_v_id = "",
        bool profile = false
    );

private:
    LlamaTransformerBlockConfig config_;
};

// Llama model configuration
struct LlamaModelConfig {
    std::string token_embedding_id;
    std::vector<LlamaTransformerBlockConfig> blocks;
    std::string output_norm_weight_id;
    std::string output_proj_id;
    
    int64_t vocab_size;
    int64_t hidden_dim;
    int64_t chunk_size = 1024;
    size_t arena_size_mb = 512;     // Buffer pool memory limit in MB (backward compatibility: kept name)
    size_t prefetch_window = 128;    // Prefetch window size (larger = more aggressive prefetching)
    std::string preload_file_path = "";  // Path to txt file with matrix IDs to preload (one per line)
};

// Llama model
class LlamaModel {
public:
    explicit LlamaModel(const LlamaModelConfig& config)
        : config_(config) {
        // Initialize transformer blocks
        for (const auto& block_config : config_.blocks) {
            blocks_.emplace_back(block_config);
        }
    }

    // Forward pass
    // If input_embeddings_id is provided and exists in memory, use it directly.
    // Otherwise, perform embedding lookup using input_ids.
    std::shared_ptr<InMemoryMatrix> forward(
        const std::vector<int32_t>& input_ids,
        const std::string& result_id,
        OperatorContext& ctx,
        const std::string& mask_id = "",
        const std::vector<std::pair<std::string, std::string>>& kv_cache_ids = {},
        const std::vector<std::pair<std::string, std::string>>& cache_output_ids = {},
        const std::string& input_embeddings_id = "",  // Optional: use pre-embedded input
        bool profile = false
    );

private:
    LlamaModelConfig config_;
    std::vector<LlamaTransformerBlock> blocks_;
};

} // namespace kvtensor

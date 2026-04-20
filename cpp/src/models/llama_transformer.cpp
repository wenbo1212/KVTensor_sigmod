#include "kvtensor/llama.hpp"
#include "kvtensor/operators_impl.hpp"
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <iomanip>

namespace kvtensor {

std::shared_ptr<InMemoryMatrix> LlamaTransformerBlock::forward(
    const std::string& x_id,
    const std::string& result_id,
    OperatorContext& ctx,
    const std::string& mask_id,
    const std::string& kv_cache_k_id,
    const std::string& kv_cache_v_id,
    const std::string& cache_k_id,
    const std::string& cache_v_id,
    bool profile
) {
    auto block_start = std::chrono::high_resolution_clock::now();
    
    // Pre-attention RMSNorm
    auto attn_norm_start = std::chrono::high_resolution_clock::now();
    std::string attn_norm_id = result_id + "_attn_norm";
    auto attn_norm_op = create_row_column_rmsnorm(
        x_id,
        config_.attn_norm_weight_id,
        attn_norm_id,
        config_.rms_norm_eps,
        true
    );
    auto attn_norm = attn_norm_op->execute(ctx);
    if (profile) {
        auto attn_norm_end = std::chrono::high_resolution_clock::now();
        (void)attn_norm_end;
    }
    
    // Grouped Query Attention
    auto attn_start = std::chrono::high_resolution_clock::now();
    std::string attn_id = result_id + "_attn";
    auto attn_op = create_row_column_grouped_query_attention(
        attn_norm_id,
        config_.attn_qkv_proj_id,
        config_.attn_o_proj_id,
        attn_id,
        config_.num_heads,
        config_.num_kv_heads,
        config_.head_dim,
        mask_id,
        kv_cache_k_id,
        kv_cache_v_id,
        cache_k_id,
        cache_v_id
    );
    auto attn = attn_op->execute(ctx);
    if (profile) {
        auto attn_end = std::chrono::high_resolution_clock::now();
        (void)attn_end;
    }
    
    // Clear attn_norm after attention completes
    ctx.clear_in_memory(attn_norm_id);
    
    // Residual connection: x + attn
    std::string attn_residual_id = result_id + "_attn_residual";
    auto residual_op1 = create_row_column_elementwise_add(
        x_id,
        attn_id,
        attn_residual_id,
        1.0f,
        1.0f,
        true
    );
    auto attn_residual = residual_op1->execute(ctx);
    
    // Clear attn after residual completes
    ctx.clear_in_memory(attn_id);
    
    // Pre-FFN RMSNorm
    auto ffn_norm_start = std::chrono::high_resolution_clock::now();
    std::string ffn_norm_id = result_id + "_ffn_norm";
    auto ffn_norm_op = create_row_column_rmsnorm(
        attn_residual_id,
        config_.ffn_norm_weight_id,
        ffn_norm_id,
        config_.rms_norm_eps,
        true
    );
    auto ffn_norm = ffn_norm_op->execute(ctx);
    if (profile) {
        auto ffn_norm_end = std::chrono::high_resolution_clock::now();
        (void)ffn_norm_end;
    }
    
    // Feed-forward network
    auto ffn_start = std::chrono::high_resolution_clock::now();
    std::string ffn_id = result_id + "_ffn";
    auto ffn_op = create_row_column_llama_feedforward(
        ffn_norm_id,
        config_.ffn_gate_up_proj_id,
        config_.ffn_down_proj_id,
        ffn_id,
        config_.chunk_size
    );
    auto ffn = ffn_op->execute(ctx);
    if (profile) {
        auto ffn_end = std::chrono::high_resolution_clock::now();
        (void)ffn_end;
    }
    
    // Residual connection: attn_residual + ffn
    auto final_residual_op = create_row_column_elementwise_add(
        attn_residual_id,
        ffn_id,
        result_id,
        1.0f,
        1.0f,
        true
    );
    auto result = final_residual_op->execute(ctx);
    
    // Clear ffn_norm, ffn, and attn_residual after final residual completes
    ctx.clear_in_memory(ffn_norm_id);
    ctx.clear_in_memory(ffn_id);
    ctx.clear_in_memory(attn_residual_id);
    
    if (profile) {
        auto block_end = std::chrono::high_resolution_clock::now();
        auto block_duration = std::chrono::duration_cast<std::chrono::microseconds>(block_end - block_start).count() / 1000.0;
        std::cout << "      [Block] " << std::fixed << std::setprecision(2)
                  << block_duration << " ms" << std::endl;
    }
    
    return result;
}

} // namespace kvtensor

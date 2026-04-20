#include "kvtensor/llama.hpp"
#include "kvtensor/operators_impl.hpp"
#include "kvtensor/matrix.hpp"
#include "math/arithmetic.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <iomanip>

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
} // namespace

std::shared_ptr<InMemoryMatrix> LlamaModel::forward(
    const std::vector<int32_t>& input_ids,
    const std::string& result_id,
    OperatorContext& ctx,
    const std::string& mask_id,
    const std::vector<std::pair<std::string, std::string>>& kv_cache_ids,
    const std::vector<std::pair<std::string, std::string>>& cache_output_ids,
    const std::string& input_embeddings_id,
    bool profile
) {
    int64_t seq_len = static_cast<int64_t>(input_ids.size());

    if (profile) {
        std::cout << "[LlamaModel] Starting forward pass: seq_len=" << seq_len 
                  << ", num_layers=" << blocks_.size() << std::endl;
    }
    
    // Start profiling AFTER the log message (excludes setup time)
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::shared_ptr<InMemoryMatrix> x;
    std::string x_id;
    
    // Check if pre-embedded input is provided
    if (!input_embeddings_id.empty()) {
        x = ctx.get_in_memory(input_embeddings_id);
        if (x) {
            // Use pre-embedded input directly
            x_id = input_embeddings_id;
            if (profile) {
                std::cout << "  [LlamaModel] Using pre-embedded input: " << input_embeddings_id << std::endl;
            }
        }
    }
    
    // If no pre-embedded input, perform token embedding lookup
    if (!x) {
        auto embed_start = std::chrono::high_resolution_clock::now();
        if (profile) {
            std::cout << "  [LlamaModel] Token embedding lookup..." << std::endl;
        }
        // Token embedding lookup
        std::shared_ptr<InMemoryMatrix> token_embedding_mem = ctx.resolve_in_memory(config_.token_embedding_id);
        std::shared_ptr<BlockMatrix> token_embedding_block = nullptr;
        if (!token_embedding_mem) {
            token_embedding_block = ctx.resolve_block_matrix(config_.token_embedding_id);
        }
        if (!token_embedding_mem && !token_embedding_block) {
            throw std::runtime_error("Token embedding not found: " + config_.token_embedding_id);
        }

        // Read embedding rows for input_ids (convert to float32)
        std::vector<float> embedded_data(seq_len * config_.hidden_dim);
        DType embedding_dtype = token_embedding_mem ? token_embedding_mem->dtype() : token_embedding_block->dtype();
        int64_t embedding_cols = config_.hidden_dim;

        if (token_embedding_mem) {
            auto [emb_rows, emb_cols] = token_embedding_mem->shape();
            embedding_cols = emb_cols;
            const uint8_t* base = token_embedding_mem->data().data();
            size_t row_bytes = static_cast<size_t>(embedding_cols) * dtype_size(embedding_dtype);

            for (int64_t i = 0; i < seq_len; ++i) {
                int32_t token_id = input_ids[i];
                if (token_id < 0 || token_id >= emb_rows) {
                    throw std::runtime_error("Invalid token ID: " + std::to_string(token_id));
                }
                const uint8_t* row_ptr = base + static_cast<size_t>(token_id) * row_bytes;
                float* dst = embedded_data.data() + i * config_.hidden_dim;
                if (embedding_dtype == DType::FLOAT32) {
                    std::memcpy(dst, row_ptr, static_cast<size_t>(config_.hidden_dim) * sizeof(float));
                } else {
                    math::convert_buffer_to_float32(row_ptr, embedding_dtype,
                                                    static_cast<size_t>(config_.hidden_dim), dst);
                }
            }
        } else {
            for (int64_t i = 0; i < seq_len; ++i) {
                int32_t token_id = input_ids[i];
                if (token_id < 0 || token_id >= config_.vocab_size) {
                    throw std::runtime_error("Invalid token ID: " + std::to_string(token_id));
                }
                auto row_bytes = token_embedding_block->read_row(token_id, ctx);

                // Convert from storage dtype to float32
                if (embedding_dtype == DType::FLOAT32) {
                    std::memcpy(
                        embedded_data.data() + i * config_.hidden_dim,
                        row_bytes.data(),
                        config_.hidden_dim * sizeof(float)
                    );
                } else if (embedding_dtype == DType::BFLOAT16) {
                    const uint16_t* bf16_ptr = reinterpret_cast<const uint16_t*>(row_bytes.data());
                    for (int64_t j = 0; j < config_.hidden_dim; ++j) {
                        embedded_data[i * config_.hidden_dim + j] = math::bf16_to_float(bf16_ptr[j]);
                    }
                } else if (embedding_dtype == DType::INT8) {
                    const int8_t* int8_ptr = reinterpret_cast<const int8_t*>(row_bytes.data());
                    for (int64_t j = 0; j < config_.hidden_dim; ++j) {
                        embedded_data[i * config_.hidden_dim + j] = static_cast<float>(int8_ptr[j]);
                    }
                } else {
                    throw std::runtime_error("Unsupported embedding dtype");
                }
            }
        }
        
        // Store embedded input
        std::vector<uint8_t> embedded_bytes(embedded_data.size() * sizeof(float));
        std::memcpy(embedded_bytes.data(), embedded_data.data(), embedded_bytes.size());
        
        x_id = result_id + "_embedded";
        x = ctx.store_in_memory(
            x_id,
            std::make_tuple(seq_len, config_.hidden_dim),
            DType::FLOAT32,
            std::move(embedded_bytes)
        );
        if (profile) {
            auto embed_end = std::chrono::high_resolution_clock::now();
            auto embed_duration = std::chrono::duration_cast<std::chrono::microseconds>(embed_end - embed_start).count() / 1000.0;
            std::cout << "  [LlamaModel] Token embedding completed: " << std::fixed << std::setprecision(2) 
                      << embed_duration << " ms" << std::endl;
        }
    }
    
    // Pass through transformer blocks
    if (profile) {
        std::cout << "  [LlamaModel] Processing " << blocks_.size() << " transformer blocks..." << std::endl;
        if (ctx.buffer_pool() != nullptr) {
            auto stats = ctx.buffer_pool()->get_stats();
            double used_mb = stats.memory_used_bytes / (1024.0 * 1024.0);
            double total_mb = stats.memory_total_bytes / (1024.0 * 1024.0);
            std::cout << "    [BufferPool Initial] " << std::fixed << std::setprecision(1) << used_mb << "/" << total_mb 
                      << " MB, " << stats.cached_chunks << " chunks prefetched, sequence=" << stats.sequence_length << std::endl;
        }
    }
    std::string current_id = x_id;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        auto block_start = std::chrono::high_resolution_clock::now();
        std::string block_result_id = result_id + "_block_" + std::to_string(i);
        
        // Get KV cache for this layer if provided
        std::string kv_cache_k_id;
        std::string kv_cache_v_id;
        std::string cache_k_id;
        std::string cache_v_id;
        
        if (i < kv_cache_ids.size()) {
            kv_cache_k_id = kv_cache_ids[i].first;
            kv_cache_v_id = kv_cache_ids[i].second;
        }
        if (i < cache_output_ids.size()) {
            cache_k_id = cache_output_ids[i].first;
            cache_v_id = cache_output_ids[i].second;
        }
        
        // Store previous layer's ID before updating
        std::string prev_layer_id = (i > 0) ? current_id : "";
        
        auto result = blocks_[i].forward(
            current_id,
            block_result_id,
            ctx,
            mask_id,
            kv_cache_k_id,
            kv_cache_v_id,
            cache_k_id,
            cache_v_id,
            profile
        );
        
        current_id = block_result_id;
        
        // Clear previous layer's output after current layer has read it
        if (!prev_layer_id.empty()) {
            ctx.clear_in_memory(prev_layer_id);
        }
        
        if (profile) {
            auto block_end = std::chrono::high_resolution_clock::now();
            (void)block_end;
        }
    }
    
    // Output RMSNorm
    auto output_norm_start = std::chrono::high_resolution_clock::now();
    if (profile) {
        std::cout << "  [LlamaModel] Output RMSNorm..." << std::endl;
    }
    std::string output_norm_id = result_id + "_output_norm";
    auto output_norm_op = create_row_column_rmsnorm(
        current_id,
        config_.output_norm_weight_id,
        output_norm_id,
        1e-6f,
        true
    );
    auto output_norm = output_norm_op->execute(ctx);
    if (profile) {
        auto output_norm_end = std::chrono::high_resolution_clock::now();
        auto output_norm_duration = std::chrono::duration_cast<std::chrono::microseconds>(output_norm_end - output_norm_start).count() / 1000.0;
        std::cout << "  [LlamaModel] Output RMSNorm completed: " 
                  << std::fixed << std::setprecision(2) << output_norm_duration << " ms" << std::endl;
    }
    
    // For prefill: only use last token's embedding to predict next token
    // Extract last token from output_norm
    auto output_norm_data = output_norm->data();
    const float* output_norm_ptr = reinterpret_cast<const float*>(output_norm_data.data());
    
    std::vector<float> last_token_embedding(config_.hidden_dim);
    std::memcpy(
        last_token_embedding.data(),
        output_norm_ptr + (seq_len - 1) * config_.hidden_dim,
        config_.hidden_dim * sizeof(float)
    );
    
    // Store last token embedding (keep it for decode phase)
    std::vector<uint8_t> last_token_bytes(last_token_embedding.size() * sizeof(float));
    std::memcpy(last_token_bytes.data(), last_token_embedding.data(), last_token_bytes.size());
    
    std::string last_token_id = result_id + "_last_token";
    auto last_token = ctx.store_in_memory(
        last_token_id,
        std::make_tuple(1, config_.hidden_dim),
        DType::FLOAT32,
        std::move(last_token_bytes)
    );
    
    // Output projection (LM head) - store in memory for final logits
    auto output_proj_start = std::chrono::high_resolution_clock::now();
    if (profile) {
        std::cout << "  [LlamaModel] Output projection (LM head)..." << std::endl;
    }
    auto output_proj_op = create_row_column_matmul(
        last_token_id,
        config_.output_proj_id,
        result_id,
        true  // Store in memory (logits should be in memory, not LevelDB)
    );
    auto result = output_proj_op->execute(ctx);
    if (profile) {
        auto output_proj_end = std::chrono::high_resolution_clock::now();
        auto output_proj_duration = std::chrono::duration_cast<std::chrono::microseconds>(output_proj_end - output_proj_start).count() / 1000.0;
        std::cout << "  [LlamaModel] Output projection completed: " 
                  << std::fixed << std::setprecision(2) << output_proj_duration << " ms" << std::endl;
    }
    
    // Clear intermediate results (but keep last_token_id for decode phase)
    ctx.clear_in_memory(output_norm_id);
    // Note: last_token_id is kept in memory for decode phase to use
    
    if (profile) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
        std::cout << "[LlamaModel] Forward pass completed: " 
                  << std::fixed << std::setprecision(2) << total_duration << " ms" << std::endl;
        
    }
    
    return result;
}

} // namespace kvtensor

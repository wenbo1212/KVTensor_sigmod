#include "kvtensor/operators.hpp"
#include "kvtensor/operators_impl.hpp"
#include "kvtensor/context.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/profile.hpp"
#include "kvtensor/sd_ops.hpp"
#include "math/arithmetic.hpp"
#include <dnnl.hpp>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace kvtensor {

// Helper functions
template<typename T>
const float* get_float32_ptr(const std::vector<uint8_t>& data) {
    return reinterpret_cast<const float*>(data.data());
}

template<typename T>
float* get_float32_ptr_mut(std::vector<uint8_t>& data) {
    return reinterpret_cast<float*>(data.data());
}

inline std::vector<float> to_float32_buffer(const std::vector<uint8_t>& data, DType dtype) {
    std::vector<float> out(data.size() / (dtype == DType::INT8 ? sizeof(int8_t)
                                    : (dtype == DType::FLOAT16 || dtype == DType::BFLOAT16)
                                        ? sizeof(uint16_t)
                                        : sizeof(float)));
    {
        ProfileScope scope(ProfileKind::Decompress);
        math::convert_buffer_to_float32(data.data(), dtype, out.size(), out.data());
    }
    return out;
}

template <typename Fn>
inline void profile_gemm(const std::string& op_class, int64_t m, int64_t k, int64_t n, Fn&& fn) {
    if (m <= 0 || k <= 0 || n <= 0) {
        fn();
        return;
    }
    ProfileScope scope(ProfileKind::Compute);
    add_profile_gemm(op_class, m, k, n);
    fn();
}

inline std::vector<float> load_bias_vector(OperatorContext& ctx, const std::string& id, int64_t cols) {
    if (id.empty()) {
        return {};
    }
    auto mem = ctx.resolve_in_memory(id);
    if (!mem) {
        auto bm = ctx.resolve_block_matrix(id);
        if (!bm) {
            throw std::runtime_error("Bias not found: " + id);
        }
        auto dense = bm->to_dense(ctx);
        mem = ctx.store_in_memory(id + "_dense", bm->shape(), bm->dtype(), std::move(dense));
    }
    auto [r, c] = mem->shape();
    if (r * c == cols) {
        return to_float32_buffer(mem->data(), mem->dtype());
    }
    throw std::runtime_error("Bias size mismatch for " + id);
}

inline void add_bias(std::vector<float>& data, int64_t rows, int64_t cols, const std::vector<float>& bias) {
    if (bias.empty()) {
        return;
    }
    if (static_cast<int64_t>(bias.size()) != cols) {
        throw std::runtime_error("Bias length mismatch");
    }
    for (int64_t r = 0; r < rows; ++r) {
        float* row_ptr = data.data() + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            row_ptr[c] += bias[static_cast<size_t>(c)];
        }
    }
}

inline std::vector<uint16_t> load_bias_vector_bf16(OperatorContext& ctx, const std::string& id, int64_t cols) {
    if (id.empty()) {
        return {};
    }
    auto mem = ctx.resolve_in_memory(id);
    if (!mem) {
        auto bm = ctx.resolve_block_matrix(id);
        if (!bm) {
            throw std::runtime_error("Bias not found: " + id);
        }
        auto dense = bm->to_dense(ctx);
        mem = ctx.store_in_memory(id + "_dense", bm->shape(), bm->dtype(), std::move(dense));
    }
    auto [r, c] = mem->shape();
    if (r * c != cols) {
        throw std::runtime_error("Bias size mismatch for " + id);
    }
    if (mem->dtype() == DType::BFLOAT16) {
        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(mem->data().data());
        return std::vector<uint16_t>(ptr, ptr + cols);
    }
    std::vector<float> tmp = to_float32_buffer(mem->data(), mem->dtype());
    std::vector<uint16_t> out(tmp.size());
    {
        ProfileScope scope(ProfileKind::Decompress);
        math::convert_float32_to_bf16(tmp.data(), tmp.size(), out.data());
    }
    return out;
}

inline void add_bias_bf16(std::vector<uint16_t>& data, int64_t rows, int64_t cols, const std::vector<uint16_t>& bias) {
    if (bias.empty()) {
        return;
    }
    if (static_cast<int64_t>(bias.size()) != cols) {
        throw std::runtime_error("Bias length mismatch");
    }
    for (int64_t r = 0; r < rows; ++r) {
        uint16_t* row_ptr = data.data() + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            float v = math::bf16_to_float(row_ptr[c]) + math::bf16_to_float(bias[static_cast<size_t>(c)]);
            row_ptr[c] = math::float_to_bf16(v);
        }
    }
}

inline void add_bias_bf16(uint16_t* data, int64_t rows, int64_t cols, const uint16_t* bias) {
    if (!bias) {
        return;
    }
    for (int64_t r = 0; r < rows; ++r) {
        uint16_t* row_ptr = data + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            float v = math::bf16_to_float(row_ptr[c]) + math::bf16_to_float(bias[static_cast<size_t>(c)]);
            row_ptr[c] = math::float_to_bf16(v);
        }
    }
}

inline std::vector<float> slice_head(
    const std::vector<float>& src,
    int64_t rows,
    int64_t hidden,
    int64_t head_idx,
    int64_t head_dim
) {
    std::vector<float> out(static_cast<size_t>(rows * head_dim));
    for (int64_t r = 0; r < rows; ++r) {
        const float* row_ptr = src.data() + r * hidden + head_idx * head_dim;
        std::memcpy(out.data() + r * head_dim, row_ptr, static_cast<size_t>(head_dim) * sizeof(float));
    }
    return out;
}

inline void write_head(
    std::vector<float>& dst,
    const std::vector<float>& head,
    int64_t rows,
    int64_t hidden,
    int64_t head_idx,
    int64_t head_dim
) {
    for (int64_t r = 0; r < rows; ++r) {
        float* row_ptr = dst.data() + r * hidden + head_idx * head_dim;
        const float* head_ptr = head.data() + r * head_dim;
        std::memcpy(row_ptr, head_ptr, static_cast<size_t>(head_dim) * sizeof(float));
    }
}

// RowColumnGroupedQueryAttention operator
class RowColumnGroupedQueryAttention : public Operator {
public:
    RowColumnGroupedQueryAttention(
        const std::string& x_id,
        const std::string& qkv_proj_id,
        const std::string& o_proj_id,
        const std::string& result_id,
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t head_dim,
        const std::string& mask_id = "",
        const std::string& kv_cache_k_id = "",
        const std::string& kv_cache_v_id = "",
        const std::string& cache_k_id = "",
        const std::string& cache_v_id = "",
        const std::string& qkv_bias_id = "",
        const std::string& o_proj_bias_id = ""
    ) : x_id_(x_id),
        qkv_proj_id_(qkv_proj_id),
        o_proj_id_(o_proj_id),
        result_id_(result_id),
        num_heads_(num_heads),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        mask_id_(mask_id),
        kv_cache_k_id_(kv_cache_k_id),
        kv_cache_v_id_(kv_cache_v_id),
        cache_k_id_(cache_k_id),
        cache_v_id_(cache_v_id),
        qkv_bias_id_(qkv_bias_id),
        o_proj_bias_id_(o_proj_bias_id) {
        name_ = "rowcolumn_grouped_query_attention";
    }

    std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) override {
        // Resolve inputs
        auto x = ctx.resolve_in_memory(x_id_);
        if (!x) {
            auto block = ctx.resolve_block_matrix(x_id_);
            if (block) {
                auto dense = block->to_dense(ctx);
                x = ctx.store_in_memory(x_id_ + "_dense", block->shape(), block->dtype(), std::move(dense));
            } else {
                throw std::runtime_error("Input matrix not found: " + x_id_);
            }
        }
        
        auto [seq_len, hidden_dim] = x->shape();
        
        // Verify dimensions
        int64_t expected_q_dim = num_heads_ * head_dim_;
        int64_t expected_kv_dim = num_kv_heads_ * head_dim_;
        int64_t expected_qkv_dim = expected_q_dim + 2 * expected_kv_dim;
        
        // Project QKV
        std::string qkv_id = result_id_ + "_qkv";
        auto qkv_op = create_row_column_matmul(x_id_, qkv_proj_id_, qkv_id, true);
        auto qkv_full = qkv_op->execute(ctx);
        std::vector<float> qkv_full_f32 = to_float32_buffer(qkv_full->data(), qkv_full->dtype());
        if (!qkv_bias_id_.empty()) {
            auto bias = load_bias_vector(ctx, qkv_bias_id_, expected_qkv_dim);
            {
                ProfileScope scope(ProfileKind::OtherCompute);
                add_bias(qkv_full_f32, seq_len, expected_qkv_dim, bias);
            }
        }

        // Split QKV (use views when possible to avoid copies)
        const float* qkv_ptr = qkv_full_f32.data();
        
        // Sizes
        int64_t q_size = seq_len * expected_q_dim;
        int64_t kv_size = seq_len * expected_kv_dim;
        
        const float* q_ptr = qkv_ptr;
        const float* k_ptr = qkv_ptr + q_size;
        const float* v_ptr = qkv_ptr + q_size + kv_size;
        
        std::vector<float> k_data;
        std::vector<float> v_data;
        
        // Handle KV cache
        int64_t total_seq_len = seq_len;
        if (!kv_cache_k_id_.empty() && !kv_cache_v_id_.empty()) {
            auto k_cache = ctx.get_in_memory(kv_cache_k_id_);
            auto v_cache = ctx.get_in_memory(kv_cache_v_id_);
            
            if (k_cache && v_cache) {
                const float* k_cache_ptr = get_float32_ptr<float>(k_cache->data());
                const float* v_cache_ptr = get_float32_ptr<float>(v_cache->data());
                auto [cache_rows, cache_cols] = k_cache->shape();
                total_seq_len = cache_rows + seq_len;
                
                // Concatenate
                std::vector<float> k_combined(total_seq_len * expected_kv_dim);
                std::vector<float> v_combined(total_seq_len * expected_kv_dim);
                {
                    ProfileScope scope(ProfileKind::OtherCompute);
                    std::memcpy(k_combined.data(), k_cache_ptr, cache_rows * expected_kv_dim * sizeof(float));
                    std::memcpy(k_combined.data() + cache_rows * expected_kv_dim, k_ptr, kv_size * sizeof(float));

                    std::memcpy(v_combined.data(), v_cache_ptr, cache_rows * expected_kv_dim * sizeof(float));
                    std::memcpy(v_combined.data() + cache_rows * expected_kv_dim, v_ptr, kv_size * sizeof(float));
                }
                
                k_data = std::move(k_combined);
                v_data = std::move(v_combined);
                k_ptr = k_data.data();
                v_ptr = v_data.data();
            }
        }
        
        // Store updated cache if requested
        if (!cache_k_id_.empty() && !cache_v_id_.empty()) {
            size_t cache_elems = static_cast<size_t>(total_seq_len) * static_cast<size_t>(expected_kv_dim);
            std::vector<uint8_t> k_cache_bytes(cache_elems * sizeof(float));
            std::vector<uint8_t> v_cache_bytes(cache_elems * sizeof(float));
            std::memcpy(k_cache_bytes.data(), k_ptr, cache_elems * sizeof(float));
            std::memcpy(v_cache_bytes.data(), v_ptr, cache_elems * sizeof(float));
            
            ctx.store_in_memory(
                cache_k_id_,
                std::make_tuple(total_seq_len, expected_kv_dim),
                DType::FLOAT32,
                std::move(k_cache_bytes)
            );
            ctx.store_in_memory(
                cache_v_id_,
                std::make_tuple(total_seq_len, expected_kv_dim),
                DType::FLOAT32,
                std::move(v_cache_bytes)
            );
        }
        
        // Reshape to separate heads
        // Q: (seq_len, num_heads, head_dim)
        // K, V: (total_seq_len, num_kv_heads, head_dim)
        
        // Compute attention per query head
        int64_t heads_per_kv = num_heads_ / num_kv_heads_;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        
        // Get mask if provided
        const float* mask_ptr = nullptr;
        if (!mask_id_.empty()) {
            auto mask = ctx.get_causal_mask(mask_id_);
            if (mask) {
                mask_ptr = get_float32_ptr<float>(mask->data());
            }
        }
        
        // Store concatenated output directly to avoid extra copies
        std::vector<uint8_t> concat_bytes(seq_len * num_heads_ * head_dim_ * sizeof(float));
        float* concat_ptr = reinterpret_cast<float*>(concat_bytes.data());
        
        for (int64_t q_head_idx = 0; q_head_idx < num_heads_; ++q_head_idx) {
            int64_t kv_head_idx = q_head_idx / heads_per_kv;
            
            // Extract head data
            std::vector<float> q_head(seq_len * head_dim_);
            std::vector<float> k_head(total_seq_len * head_dim_);
            std::vector<float> v_head(total_seq_len * head_dim_);
            {
                ProfileScope scope(ProfileKind::OtherCompute);
                for (int64_t i = 0; i < seq_len; ++i) {
                    std::memcpy(
                        q_head.data() + i * head_dim_,
                        q_ptr + i * expected_q_dim + q_head_idx * head_dim_,
                        head_dim_ * sizeof(float)
                    );
                }

                for (int64_t i = 0; i < total_seq_len; ++i) {
                    std::memcpy(
                        k_head.data() + i * head_dim_,
                        k_ptr + i * expected_kv_dim + kv_head_idx * head_dim_,
                        head_dim_ * sizeof(float)
                    );
                    std::memcpy(
                        v_head.data() + i * head_dim_,
                        v_ptr + i * expected_kv_dim + kv_head_idx * head_dim_,
                        head_dim_ * sizeof(float)
                    );
                }
            }
            
            // Compute attention scores: Q @ K^T using optimized matmul
            // Q: (seq_len, head_dim), K: (total_seq_len, head_dim)
            // Result: (seq_len, total_seq_len)
            std::vector<float> scores(seq_len * total_seq_len);
            profile_gemm("attention_scores", seq_len, head_dim_, total_seq_len, [&]() {
                math::matmul_ex(
                    q_head.data(), seq_len, head_dim_,
                    k_head.data(), total_seq_len,
                    scores.data(),
                    math::Transpose::No,
                    math::Transpose::Yes
                );
            });
            
            // Apply scale, mask, and softmax (oneDNN-backed via math::softmax)
            std::vector<float> softmax_scores(seq_len * total_seq_len);
            {
                ProfileScope scope(ProfileKind::OtherCompute);
                // Apply scale
                for (int64_t i = 0; i < seq_len * total_seq_len; ++i) {
                    scores[i] *= scale;
                }

                // Apply mask if provided
                if (mask_ptr != nullptr) {
                    for (int64_t i = 0; i < seq_len * total_seq_len; ++i) {
                        scores[i] += mask_ptr[i];
                    }
                }

                math::softmax(scores.data(), seq_len, total_seq_len, 1, softmax_scores.data());
            }
            
            // Attention output: softmax_scores @ V using optimized matmul
            // softmax_scores: (seq_len, total_seq_len), V: (total_seq_len, head_dim)
            // Result: (seq_len, head_dim)
            std::vector<float> head_out(seq_len * head_dim_);
            profile_gemm("attention_values", seq_len, total_seq_len, head_dim_, [&]() {
                math::matmul(
                    softmax_scores.data(), seq_len, total_seq_len,
                    v_head.data(), head_dim_,
                    head_out.data()
                );
            });
            
            // Store in concatenated output (seq_len, num_heads * head_dim)
            {
                ProfileScope scope(ProfileKind::OtherCompute);
                for (int64_t i = 0; i < seq_len; ++i) {
                    std::memcpy(
                        concat_ptr + i * (num_heads_ * head_dim_) + q_head_idx * head_dim_,
                        head_out.data() + i * head_dim_,
                        head_dim_ * sizeof(float)
                    );
                }
            }
        }
        
        // Clear qkv_full after attention computation to avoid keeping large intermediates in memory
        ctx.clear_in_memory(qkv_id);
        
        std::string concat_id = result_id_ + "_concat";
        auto concat = ctx.store_in_memory(
            concat_id,
            std::make_tuple(seq_len, num_heads_ * head_dim_),
            DType::FLOAT32,
            std::move(concat_bytes)
        );
        
        // Output projection
        auto o_op = create_row_column_matmul(concat_id, o_proj_id_, result_id_, true);
        auto result = o_op->execute(ctx);
        if (!o_proj_bias_id_.empty()) {
            auto bias = load_bias_vector(ctx, o_proj_bias_id_, hidden_dim);
            auto out_f32 = to_float32_buffer(result->data(), result->dtype());
            {
                ProfileScope scope(ProfileKind::OtherCompute);
                add_bias(out_f32, seq_len, hidden_dim, bias);
            }
            std::vector<uint8_t> out_bytes(out_f32.size() * sizeof(float));
            std::memcpy(out_bytes.data(), out_f32.data(), out_bytes.size());
            result = ctx.store_in_memory(result_id_, result->shape(), DType::FLOAT32, std::move(out_bytes));
        }

        // Clear concat after output projection
        ctx.clear_in_memory(concat_id);
        return result;
    }

private:
    std::string x_id_;
    std::string qkv_proj_id_;
    std::string o_proj_id_;
    std::string result_id_;
    int64_t num_heads_;
    int64_t num_kv_heads_;
    int64_t head_dim_;
    std::string mask_id_;
    std::string kv_cache_k_id_;
    std::string kv_cache_v_id_;
    std::string cache_k_id_;
    std::string cache_v_id_;
    std::string qkv_bias_id_;
    std::string o_proj_bias_id_;
};

class SDAttentionBlockOp : public Operator {
public:
    explicit SDAttentionBlockOp(const AttentionBlockConfig& config) : config_(config) {
        name_ = "sd_attention";
        result_id_ = config.result_id;
    }

    std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) override {
        auto hidden_mem = ctx.resolve_in_memory(config_.hidden_states_id);
        if (!hidden_mem) {
            auto bm = ctx.resolve_block_matrix(config_.hidden_states_id);
            if (!bm) {
                throw std::runtime_error("Hidden states not found: " + config_.hidden_states_id);
            }
            hidden_mem = ctx.store_in_memory(
                config_.hidden_states_id + "_dense",
                bm->shape(),
                bm->dtype(),
                bm->to_dense(ctx)
            );
        }
        auto [seq_len, hidden] = hidden_mem->shape();
        if (hidden != config_.hidden_size) {
            config_.hidden_size = hidden;
        }
        int64_t head_dim = config_.hidden_size / config_.num_heads;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::shared_ptr<InMemoryMatrix> enc_mem;
        int64_t enc_seq = seq_len;
        if (!config_.encoder_hidden_states_id.empty()) {
            enc_mem = ctx.resolve_in_memory(config_.encoder_hidden_states_id);
            if (!enc_mem) {
                auto bm = ctx.resolve_block_matrix(config_.encoder_hidden_states_id);
                if (!bm) {
                    throw std::runtime_error("Encoder hidden states not found");
                }
                enc_mem = ctx.store_in_memory(
                    config_.encoder_hidden_states_id + "_dense",
                    bm->shape(),
                    bm->dtype(),
                    bm->to_dense(ctx)
                );
            }
            enc_seq = std::get<0>(enc_mem->shape());
        }

        const int64_t heads = config_.num_heads;
        size_t q_elems = static_cast<size_t>(seq_len * config_.hidden_size);
        size_t kv_elems = static_cast<size_t>(enc_seq * config_.hidden_size);
        size_t qbatch_elems = static_cast<size_t>(heads * seq_len * head_dim);
        size_t kbatch_elems = static_cast<size_t>(heads * enc_seq * head_dim);
        size_t kt_elems = static_cast<size_t>(heads * head_dim * enc_seq);
        size_t scores_elems = static_cast<size_t>(heads * seq_len * enc_seq);
        size_t ctx_batched_elems = qbatch_elems;
        size_t context_elems = static_cast<size_t>(seq_len * config_.hidden_size);
        size_t total_u16 = q_elems + kv_elems + kv_elems + qbatch_elems + kbatch_elems + kbatch_elems +
            kt_elems + scores_elems + ctx_batched_elems + context_elems;
        size_t total_bytes = total_u16 * sizeof(uint16_t) + 64 * 16;
        auto run_matmul = [&](const std::string& lhs_id, const std::string& weight_id, const std::string& out_id) {
            auto mm = create_row_column_matmul(lhs_id, weight_id, out_id, true);
            return mm->execute(ctx);
        };

        uint8_t* arena_base = ctx.attention_scratch(total_bytes);
        size_t arena_head = 0;
        auto alloc_bytes = [&](size_t bytes) -> uint8_t* {
            const size_t align = 64;
            size_t aligned = (arena_head + (align - 1)) & ~(align - 1);
            if (aligned + bytes > total_bytes) {
                throw std::runtime_error("attention arena overflow");
            }
            uint8_t* ptr = arena_base + aligned;
            arena_head = aligned + bytes;
            return ptr;
        };
        auto alloc_u16 = [&](size_t count) -> uint16_t* {
            return reinterpret_cast<uint16_t*>(alloc_bytes(count * sizeof(uint16_t)));
        };

        uint16_t* Qb16 = alloc_u16(q_elems);
        uint16_t* Kb16 = alloc_u16(kv_elems);
        uint16_t* Vb16 = alloc_u16(kv_elems);
        bool qkv_bf16 = false;
        if (config_.encoder_hidden_states_id.empty() && !config_.qkv_proj_id.empty()) {
            auto qkv_mem = run_matmul(hidden_mem->matrix_id(), config_.qkv_proj_id, result_id_ + "_qkv");
            qkv_bf16 = (qkv_mem->dtype() == DType::BFLOAT16);
            int64_t q_size = seq_len * config_.hidden_size;
            int64_t kv_size = q_size;
            if (qkv_bf16) {
                const uint16_t* qkv_ptr = reinterpret_cast<const uint16_t*>(qkv_mem->data().data());
                {
                    ProfileScope scope(ProfileKind::OtherCompute);
                    std::memcpy(Qb16, qkv_ptr, static_cast<size_t>(q_size) * sizeof(uint16_t));
                    std::memcpy(Kb16, qkv_ptr + q_size, static_cast<size_t>(kv_size) * sizeof(uint16_t));
                    std::memcpy(Vb16, qkv_ptr + q_size + kv_size, static_cast<size_t>(kv_size) * sizeof(uint16_t));
                    if (!config_.qkv_proj_bias_id.empty()) {
                        auto bias = load_bias_vector_bf16(ctx, config_.qkv_proj_bias_id, 3 * config_.hidden_size);
                        int64_t h = config_.hidden_size;
                        const uint16_t* bias_q = bias.data();
                        const uint16_t* bias_k = bias.data() + h;
                        const uint16_t* bias_v = bias.data() + 2 * h;
                        add_bias_bf16(Qb16, seq_len, h, bias_q);
                        add_bias_bf16(Kb16, seq_len, h, bias_k);
                        add_bias_bf16(Vb16, seq_len, h, bias_v);
                    }
                }
            } else {
                std::vector<float> qkv_f32 = to_float32_buffer(qkv_mem->data(), qkv_mem->dtype());
                if (!config_.qkv_proj_bias_id.empty()) {
                    auto bias = load_bias_vector(ctx, config_.qkv_proj_bias_id, 3 * config_.hidden_size);
                    {
                        ProfileScope scope(ProfileKind::OtherCompute);
                        add_bias(qkv_f32, seq_len, 3 * config_.hidden_size, bias);
                    }
                }
                if (static_cast<int64_t>(qkv_f32.size()) != q_size + kv_size + kv_size) {
                    throw std::runtime_error("QKV fused output size mismatch");
                }
                {
                    ProfileScope scope(ProfileKind::Decompress);
                    math::convert_float32_to_bf16(qkv_f32.data(), static_cast<size_t>(q_size), Qb16);
                    math::convert_float32_to_bf16(qkv_f32.data() + q_size, static_cast<size_t>(kv_size), Kb16);
                    math::convert_float32_to_bf16(qkv_f32.data() + q_size + kv_size, static_cast<size_t>(kv_size), Vb16);
                }
            }
            ctx.clear_in_memory(qkv_mem->matrix_id());
            enc_seq = seq_len;
        } else {
            auto q_mem = run_matmul(hidden_mem->matrix_id(), config_.q_proj_id, result_id_ + "_q");
            auto k_mem = run_matmul(enc_mem ? enc_mem->matrix_id() : hidden_mem->matrix_id(),
                                    config_.k_proj_id, result_id_ + "_k");
            auto v_mem = run_matmul(enc_mem ? enc_mem->matrix_id() : hidden_mem->matrix_id(),
                                    config_.v_proj_id, result_id_ + "_v");
            qkv_bf16 = (q_mem->dtype() == DType::BFLOAT16);
            if (qkv_bf16) {
                const uint16_t* qp = reinterpret_cast<const uint16_t*>(q_mem->data().data());
                const uint16_t* kp = reinterpret_cast<const uint16_t*>(k_mem->data().data());
                const uint16_t* vp = reinterpret_cast<const uint16_t*>(v_mem->data().data());
                {
                    ProfileScope scope(ProfileKind::OtherCompute);
                    std::memcpy(Qb16, qp, static_cast<size_t>(seq_len * config_.hidden_size) * sizeof(uint16_t));
                    std::memcpy(Kb16, kp, static_cast<size_t>(enc_seq * config_.hidden_size) * sizeof(uint16_t));
                    std::memcpy(Vb16, vp, static_cast<size_t>(enc_seq * config_.hidden_size) * sizeof(uint16_t));
                    if (!config_.q_proj_bias_id.empty()) {
                        auto bias = load_bias_vector_bf16(ctx, config_.q_proj_bias_id, config_.hidden_size);
                        add_bias_bf16(Qb16, seq_len, config_.hidden_size, bias.data());
                    }
                    if (!config_.k_proj_bias_id.empty()) {
                        auto bias = load_bias_vector_bf16(ctx, config_.k_proj_bias_id, config_.hidden_size);
                        add_bias_bf16(Kb16, enc_seq, config_.hidden_size, bias.data());
                    }
                    if (!config_.v_proj_bias_id.empty()) {
                        auto bias = load_bias_vector_bf16(ctx, config_.v_proj_bias_id, config_.hidden_size);
                        add_bias_bf16(Vb16, enc_seq, config_.hidden_size, bias.data());
                    }
                }
            } else {
                std::vector<float> qf = to_float32_buffer(q_mem->data(), q_mem->dtype());
                std::vector<float> kf = to_float32_buffer(k_mem->data(), k_mem->dtype());
                std::vector<float> vf = to_float32_buffer(v_mem->data(), v_mem->dtype());
                if (!config_.q_proj_bias_id.empty()) {
                    auto bias = load_bias_vector(ctx, config_.q_proj_bias_id, config_.hidden_size);
                    {
                        ProfileScope scope(ProfileKind::OtherCompute);
                        add_bias(qf, seq_len, config_.hidden_size, bias);
                    }
                }
                if (!config_.k_proj_bias_id.empty()) {
                    auto bias = load_bias_vector(ctx, config_.k_proj_bias_id, config_.hidden_size);
                    {
                        ProfileScope scope(ProfileKind::OtherCompute);
                        add_bias(kf, enc_seq, config_.hidden_size, bias);
                    }
                }
                if (!config_.v_proj_bias_id.empty()) {
                    auto bias = load_bias_vector(ctx, config_.v_proj_bias_id, config_.hidden_size);
                    {
                        ProfileScope scope(ProfileKind::OtherCompute);
                        add_bias(vf, enc_seq, config_.hidden_size, bias);
                    }
                }
                {
                    ProfileScope scope(ProfileKind::Decompress);
                    math::convert_float32_to_bf16(qf.data(), qf.size(), Qb16);
                    math::convert_float32_to_bf16(kf.data(), kf.size(), Kb16);
                    math::convert_float32_to_bf16(vf.data(), vf.size(), Vb16);
                }
            }
            ctx.clear_in_memory(q_mem->matrix_id());
            ctx.clear_in_memory(k_mem->matrix_id());
            ctx.clear_in_memory(v_mem->matrix_id());
        }
        auto t_attn_start = std::chrono::steady_clock::now();
        uint16_t* Qbatch = alloc_u16(static_cast<size_t>(heads * seq_len * head_dim));
        uint16_t* Kbatch = alloc_u16(static_cast<size_t>(heads * enc_seq * head_dim));
        uint16_t* Vbatch = alloc_u16(static_cast<size_t>(heads * enc_seq * head_dim));
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            for (int64_t s = 0; s < seq_len; ++s) {
                const uint16_t* src = Qb16 + s * config_.hidden_size;
                for (int64_t h = 0; h < heads; ++h) {
                    uint16_t* dst = Qbatch + (h * seq_len + s) * head_dim;
                    std::memcpy(dst, src + h * head_dim, static_cast<size_t>(head_dim) * sizeof(uint16_t));
                }
            }
            for (int64_t s = 0; s < enc_seq; ++s) {
                const uint16_t* src_k = Kb16 + s * config_.hidden_size;
                const uint16_t* src_v = Vb16 + s * config_.hidden_size;
                for (int64_t h = 0; h < heads; ++h) {
                    uint16_t* dst_k = Kbatch + (h * enc_seq + s) * head_dim;
                    uint16_t* dst_v = Vbatch + (h * enc_seq + s) * head_dim;
                    std::memcpy(dst_k, src_k + h * head_dim, static_cast<size_t>(head_dim) * sizeof(uint16_t));
                    std::memcpy(dst_v, src_v + h * head_dim, static_cast<size_t>(head_dim) * sizeof(uint16_t));
                }
            }
        }

        uint16_t* Kt = alloc_u16(static_cast<size_t>(heads * head_dim * enc_seq));
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            for (int64_t h = 0; h < heads; ++h) {
                for (int64_t e = 0; e < enc_seq; ++e) {
                    const uint16_t* src = Kbatch + (h * enc_seq + e) * head_dim;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        Kt[(h * head_dim + d) * enc_seq + e] = src[d];
                    }
                }
            }
        }
        auto t_prep_end = std::chrono::steady_clock::now();

        uint16_t* scores_bf16 = alloc_u16(static_cast<size_t>(heads * seq_len * enc_seq));
        auto& engine = math::onednn_engine();
        auto& stream = math::onednn_stream();
        dnnl::memory::dims q_dims = {heads, seq_len, head_dim};
        dnnl::memory::dims k_dims = {heads, head_dim, enc_seq};
        dnnl::memory::dims s_dims = {heads, seq_len, enc_seq};
        auto q_md = dnnl::memory::desc(q_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc);
        auto k_md = dnnl::memory::desc(k_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc);
        auto s_md = dnnl::memory::desc(s_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc);
        auto q_mem = dnnl::memory(q_md, engine, Qbatch);
        auto k_mem = dnnl::memory(k_md, engine, Kt);
        auto s_mem = dnnl::memory(s_md, engine, scores_bf16);
        auto mm_pd = dnnl::matmul::primitive_desc(engine, q_md, k_md, s_md);
        profile_gemm("cross_attention_scores", heads * seq_len, head_dim, enc_seq, [&]() {
            dnnl::matmul(mm_pd).execute(stream, {
                {DNNL_ARG_SRC, q_mem},
                {DNNL_ARG_WEIGHTS, k_mem},
                {DNNL_ARG_DST, s_mem}
            });
            stream.wait();
        });
        auto t_mm1_end = std::chrono::steady_clock::now();

        uint16_t scale_bf16 = math::float_to_bf16(scale);
        dnnl::memory::dims scale_dims = {1};
        auto scale_md = dnnl::memory::desc(scale_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::x);
        auto scale_mem = dnnl::memory(scale_md, engine, &scale_bf16);
        dnnl::memory::dims scores_dims = {static_cast<int64_t>(scores_elems)};
        auto scores_md = dnnl::memory::desc(scores_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::x);
        auto scores_mem = dnnl::memory(scores_md, engine, scores_bf16);
        auto mul_pd = dnnl::binary::primitive_desc(
            engine,
            dnnl::algorithm::binary_mul,
            scores_md,
            scale_md,
            scores_md
        );
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            dnnl::binary(mul_pd).execute(stream, {
                {DNNL_ARG_SRC_0, scores_mem},
                {DNNL_ARG_SRC_1, scale_mem},
                {DNNL_ARG_DST, scores_mem}
            });
            stream.wait();
        }
        auto softmax_md = dnnl::memory::desc(s_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc);
        auto softmax_src = dnnl::memory(softmax_md, engine, scores_bf16);
        auto softmax_dst = softmax_src; // in-place softmax
        auto softmax_pd = dnnl::softmax_forward::primitive_desc(
            engine,
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::softmax_accurate,
            softmax_md,
            softmax_md,
            2,
            dnnl::primitive_attr(),
            false
        );
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            dnnl::softmax_forward(softmax_pd).execute(stream, {
                {DNNL_ARG_SRC, softmax_src},
                {DNNL_ARG_DST, softmax_dst}
            });
            stream.wait();
        }
        auto t_post1_end = std::chrono::steady_clock::now();

        uint16_t* ctx_batched = alloc_u16(static_cast<size_t>(heads * seq_len * head_dim));
        dnnl::memory::dims v_dims = {heads, enc_seq, head_dim};
        dnnl::memory::dims c_dims = {heads, seq_len, head_dim};
        auto v_md = dnnl::memory::desc(v_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc);
        auto c_md = dnnl::memory::desc(c_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc);
        auto s2_md = dnnl::memory::desc(s_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc);
        auto s2_mem = dnnl::memory(s2_md, engine, scores_bf16);
        auto v_mem = dnnl::memory(v_md, engine, Vbatch);
        auto c_mem = dnnl::memory(c_md, engine, ctx_batched);
        auto mm2_pd = dnnl::matmul::primitive_desc(engine, s2_md, v_md, c_md);
        profile_gemm("cross_attention_values", heads * seq_len, enc_seq, head_dim, [&]() {
            dnnl::matmul(mm2_pd).execute(stream, {
                {DNNL_ARG_SRC, s2_mem},
                {DNNL_ARG_WEIGHTS, v_mem},
                {DNNL_ARG_DST, c_mem}
            });
            stream.wait();
        });
        auto t_mm2_end = std::chrono::steady_clock::now();

        uint16_t* context = alloc_u16(context_elems);
        {
            ProfileScope scope(ProfileKind::OtherCompute);
            for (int64_t h = 0; h < heads; ++h) {
                for (int64_t s = 0; s < seq_len; ++s) {
                    const uint16_t* src = ctx_batched + (h * seq_len + s) * head_dim;
                    uint16_t* dst = context + s * config_.hidden_size + h * head_dim;
                    std::memcpy(dst, src, static_cast<size_t>(head_dim) * sizeof(uint16_t));
                }
            }
        }


        std::vector<uint8_t> ctx_bytes(context_elems * sizeof(uint16_t));
        std::memcpy(ctx_bytes.data(), context, ctx_bytes.size());
        std::string ctx_id = result_id_ + "_context";
        ctx.store_in_memory(ctx_id, std::make_tuple(seq_len, config_.hidden_size), DType::BFLOAT16, std::move(ctx_bytes));

        auto out_mem = run_matmul(ctx_id, config_.out_proj_id, result_id_);
        DType out_dtype = DType::FLOAT32;
        if (auto wmem = ctx.resolve_in_memory(config_.out_proj_id)) {
            out_dtype = wmem->dtype();
        } else if (auto wbm = ctx.resolve_block_matrix(config_.out_proj_id)) {
            out_dtype = wbm->dtype();
        }
        if (!config_.out_proj_bias_id.empty()) {
            auto out_f32 = to_float32_buffer(out_mem->data(), out_mem->dtype());
            auto bias = load_bias_vector(ctx, config_.out_proj_bias_id, config_.hidden_size);
            {
                ProfileScope scope(ProfileKind::OtherCompute);
                add_bias(out_f32, seq_len, config_.hidden_size, bias);
            }
            if (out_dtype == DType::BFLOAT16) {
                std::vector<uint8_t> out_bytes(out_f32.size() * sizeof(uint16_t));
                auto* out_bf16 = reinterpret_cast<uint16_t*>(out_bytes.data());
                {
                    ProfileScope scope(ProfileKind::Decompress);
                    math::convert_float32_to_bf16(out_f32.data(), out_f32.size(), out_bf16);
                }
                out_mem = ctx.store_in_memory(result_id_, out_mem->shape(), DType::BFLOAT16, std::move(out_bytes));
            } else {
                std::vector<uint8_t> out_bytes(out_f32.size() * sizeof(float));
                std::memcpy(out_bytes.data(), out_f32.data(), out_bytes.size());
                out_mem = ctx.store_in_memory(result_id_, out_mem->shape(), DType::FLOAT32, std::move(out_bytes));
            }
        }
        if (out_dtype == DType::BFLOAT16 && out_mem->dtype() != DType::BFLOAT16) {
            auto out_f32 = to_float32_buffer(out_mem->data(), out_mem->dtype());
            std::vector<uint8_t> out_bytes(out_f32.size() * sizeof(uint16_t));
            auto* out_bf16 = reinterpret_cast<uint16_t*>(out_bytes.data());
            {
                ProfileScope scope(ProfileKind::Decompress);
                math::convert_float32_to_bf16(out_f32.data(), out_f32.size(), out_bf16);
            }
            out_mem = ctx.store_in_memory(result_id_, out_mem->shape(), DType::BFLOAT16, std::move(out_bytes));
        }
        ctx.clear_in_memory(ctx_id);
        return out_mem;
    }

private:
    AttentionBlockConfig config_;
};

// Factory function
std::unique_ptr<Operator> create_row_column_grouped_query_attention(
    const std::string& x_id,
    const std::string& qkv_proj_id,
    const std::string& o_proj_id,
    const std::string& result_id,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    const std::string& mask_id,
    const std::string& kv_cache_k_id,
    const std::string& kv_cache_v_id,
    const std::string& cache_k_id,
    const std::string& cache_v_id,
    const std::string& qkv_bias_id,
    const std::string& o_proj_bias_id
) {
    return std::make_unique<RowColumnGroupedQueryAttention>(
        x_id, qkv_proj_id, o_proj_id, result_id,
        num_heads, num_kv_heads, head_dim,
        mask_id, kv_cache_k_id, kv_cache_v_id, cache_k_id, cache_v_id,
        qkv_bias_id, o_proj_bias_id
    );
}

std::unique_ptr<Operator> create_attention_block(const AttentionBlockConfig& config) {
    return std::make_unique<SDAttentionBlockOp>(config);
}

} // namespace kvtensor

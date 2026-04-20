#pragma once

#include "kvtensor/operators.hpp"
#include <memory>
#include <string>

namespace kvtensor {

// Forward declarations
class OperatorContext;

// Operator factory functions
std::unique_ptr<Operator> create_row_column_matmul(
    const std::string& lhs_id,
    const std::string& rhs_id,
    const std::string& result_id,
    bool store_in_memory = true
);

std::unique_ptr<Operator> create_row_column_rmsnorm(
    const std::string& matrix_id,
    const std::string& weight_id,
    const std::string& result_id,
    float eps = 1e-6f,
    bool store_in_memory = true
);

std::unique_ptr<Operator> create_row_column_layernorm(
    const std::string& matrix_id,
    const std::string& weight_id,
    const std::string& bias_id,
    const std::string& result_id,
    float eps = 1e-5f,
    bool store_in_memory = true
);

std::unique_ptr<Operator> create_row_column_elementwise_add(
    const std::string& lhs_id,
    const std::string& rhs_id,
    const std::string& result_id,
    float alpha = 1.0f,
    float beta = 1.0f,
    bool store_in_memory = true
);

std::unique_ptr<Operator> create_row_column_elementwise_multiply(
    const std::string& lhs_id,
    const std::string& rhs_id,
    const std::string& result_id,
    bool store_in_memory = true
);

std::unique_ptr<Operator> create_row_column_silu(
    const std::string& matrix_id,
    const std::string& result_id,
    bool store_in_memory = true
);

std::unique_ptr<Operator> create_row_column_geglu(
    const std::string& matrix_id,
    const std::string& result_id,
    bool store_in_memory = true
);

std::unique_ptr<Operator> create_row_column_grouped_query_attention(
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
);

std::unique_ptr<Operator> create_row_column_llama_feedforward(
    const std::string& x_id,
    const std::string& gate_up_proj_id,
    const std::string& down_proj_id,
    const std::string& result_id,
    int64_t chunk_size
);

} // namespace kvtensor

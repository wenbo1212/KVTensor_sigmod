#pragma once

#include "kvtensor/operators.hpp"
#include "kvtensor/types.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kvtensor {

// Configuration for im2col + matmul convolution
struct Conv2DIm2ColConfig {
    std::string input_id;
    std::string weight_id;
    std::string bias_id;
    std::string result_id;
    // Input / output geometry (NCHW). Batch is assumed 1 for SD.
    int64_t input_channels{0};
    int64_t input_height{0};
    int64_t input_width{0};
    int64_t output_channels{0}; // optional; inferred from weight rows when 0
    int64_t output_height{0};
    int64_t output_width{0};
    // Output stored as (rows, cols) = (out_height * out_width, out_channels)
    Shape output_shape{0, 0};
    DType output_dtype{DType::FLOAT32};
    int64_t kernel_h{3};
    int64_t kernel_w{3};
    int64_t stride_h{1};
    int64_t stride_w{1};
    int64_t pad_h{1};
    int64_t pad_w{1};
    int64_t dilation_h{1};
    int64_t dilation_w{1};
    int64_t groups{1};
    bool store_in_memory{true};
};

// GroupNorm placeholder configuration
struct GroupNormConfig {
    std::string input_id;
    std::string weight_id;
    std::string bias_id;
    std::string result_id;
    int64_t num_groups{32};
    int64_t num_channels{0}; // required
    int64_t spatial_size{0}; // optional; if 0, derived from input shape rows
    float eps{1e-5f};
    Shape output_shape{0, 0};
    DType output_dtype{DType::FLOAT32};
    bool store_in_memory{true};
};

// Attention block placeholder (for UNet cross/self-attention)
struct AttentionBlockConfig {
    std::string hidden_states_id;
    std::string encoder_hidden_states_id;  // empty = self-attention
    std::string q_proj_id;
    std::string k_proj_id;
    std::string v_proj_id;
    std::string qkv_proj_id;              // optional fused QKV (used when encoder_hidden_states_id is empty)
    std::string q_proj_bias_id;
    std::string k_proj_bias_id;
    std::string v_proj_bias_id;
    std::string qkv_proj_bias_id;
    std::string out_proj_id;
    std::string out_proj_bias_id;
    std::string result_id;
    int64_t hidden_size{768};
    int64_t num_heads{12};
    Shape output_shape{0, 0};
    DType output_dtype{DType::FLOAT32};
    bool store_in_memory{true};
};

// Factory helpers (placeholders; implemented in src/operators/sd_ops.cpp)
std::unique_ptr<Operator> create_im2col_conv2d(const Conv2DIm2ColConfig& config);
std::unique_ptr<Operator> create_group_norm(const GroupNormConfig& config);
std::unique_ptr<Operator> create_attention_block(const AttentionBlockConfig& config);

} // namespace kvtensor

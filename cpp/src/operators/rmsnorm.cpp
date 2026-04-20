#include "kvtensor/operators.hpp"
#include "kvtensor/operators_impl.hpp"
#include "kvtensor/context.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/profile.hpp"
#include "math/arithmetic.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numeric>

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

// RowColumnRMSNorm operator
class RowColumnRMSNorm : public Operator {
public:
    RowColumnRMSNorm(
        const std::string& matrix_id,
        const std::string& weight_id,
        const std::string& result_id,
        float eps = 1e-6f,
        bool store_in_memory = true
    ) : matrix_id_(matrix_id),
        weight_id_(weight_id),
        result_id_(result_id),
        eps_(eps),
        store_in_memory_(store_in_memory) {
        name_ = "rowcolumn_rmsnorm";
    }

    std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) override {
        // Resolve input matrix - check in-memory first
        auto matrix = ctx.resolve_in_memory(matrix_id_);
        std::shared_ptr<BlockMatrix> block_matrix = nullptr;
        bool is_block_matrix = false;
        
        if (!matrix) {
            block_matrix = ctx.resolve_block_matrix(matrix_id_);
            if (!block_matrix) {
                throw std::runtime_error(
                    "[" + name_ + "] Matrix not found: " + matrix_id_
                );
            }
            is_block_matrix = true;
        }
        
        // Get normalization weight
        auto weight = ctx.get_norm_weight(weight_id_);
        if (!weight) {
            if (auto weight_mem = ctx.resolve_in_memory(weight_id_)) {
                weight = weight_mem;
            } else if (auto weight_block = ctx.resolve_block_matrix(weight_id_)) {
                auto weight_dense = weight_block->to_dense(ctx);
                weight = ctx.store_in_memory(
                    weight_id_ + "_dense",
                    weight_block->shape(),
                    weight_block->dtype(),
                    std::move(weight_dense)
                );
                ctx.store_norm_weight(weight_id_, weight->shape(), weight->dtype(), weight->data());
            } else {
                throw std::runtime_error(
                    "[" + name_ + "] Weight not found: " + weight_id_ + 
                    " (matrix_id: " + matrix_id_ + ")"
                );
            }
        }
        
        // Get shape from either matrix or block_matrix
        int64_t rows, cols;
        if (is_block_matrix) {
            auto shape = block_matrix->shape();
            rows = std::get<0>(shape);
            cols = std::get<1>(shape);
        } else {
            auto shape = matrix->shape();
            rows = std::get<0>(shape);
            cols = std::get<1>(shape);
        }
        
        auto [weight_rows, weight_cols] = weight->shape();
        
        // Check if weight shape is valid (not zero)
        if (weight_rows == 0 || weight_cols == 0) {
            throw std::runtime_error(
                "[" + name_ + "] Weight has invalid shape (0, 0): weight_id=" + weight_id_ + 
                ", matrix_id=" + matrix_id_ + 
                ", matrix_shape=(" + std::to_string(rows) + ", " + std::to_string(cols) + ")"
            );
        }
        
        // Validate weight shape: weight should be (1, cols), (cols, 1), or (1, 1) for broadcasting
        // The weight must match the column dimension of the matrix
        bool valid_shape = false;
        if (weight_rows == 1 && weight_cols == cols) {
            // Normal case: (1, hidden_dim)
            valid_shape = true;
        } else if (weight_rows == cols && weight_cols == 1) {
            // Transposed case: (hidden_dim, 1)
            valid_shape = true;
        } else if (weight_rows == 1 && weight_cols == 1) {
            // Broadcast case: (1, 1)
            valid_shape = true;
        }
        
        if (!valid_shape) {
            throw std::runtime_error(
                "[" + name_ + "] Weight shape incompatible with matrix: " +
                "matrix_id=" + matrix_id_ + ", matrix_shape=(" + 
                std::to_string(rows) + ", " + std::to_string(cols) + 
                "), weight_id=" + weight_id_ + ", weight_shape=(" + 
                std::to_string(weight_rows) + ", " + std::to_string(weight_cols) + ")"
            );
        }
        
        // Get dtypes
        DType input_dtype = is_block_matrix ? block_matrix->dtype() : matrix->dtype();
        DType weight_dtype = weight->dtype();
        
        // Convert weight to float32
        const float* weight_ptr = get_float32_ptr<float>(weight->data());
        std::vector<float> weight_vec(cols);
        if (weight_rows == 1 && weight_cols == cols) {
            // Normal case: (1, cols)
            std::memcpy(weight_vec.data(), weight_ptr, cols * sizeof(float));
        } else if (weight_rows == cols && weight_cols == 1) {
            // Transposed case: (cols, 1)
            std::memcpy(weight_vec.data(), weight_ptr, cols * sizeof(float));
        } else if (weight_rows == 1 && weight_cols == 1) {
            // Broadcast case: (1, 1)
            float val = weight_ptr[0];
            std::fill(weight_vec.begin(), weight_vec.end(), val);
        } else {
            throw std::runtime_error("Invalid weight shape for RMSNorm");
        }
        
        // Allocate result
        size_t result_size = rows * cols * sizeof(float);
        std::vector<uint8_t> result_data(result_size);
        float* result_ptr = get_float32_ptr_mut<float>(result_data);
        
        // Compute RMSNorm: x_norm = x / sqrt(mean(x^2) + eps) * weight
        // Process row-by-row to avoid materializing entire matrix
        if (is_block_matrix) {
            // Process BlockMatrix row-by-row
            for (int64_t i = 0; i < rows; ++i) {
                // Read single row from BlockMatrix
                auto row_bytes = block_matrix->read_row(i, ctx);
                
                // Convert row to float32
                std::vector<float> row_f32(cols);
                {
                    ProfileScope scope(ProfileKind::Decompress);
                    math::convert_buffer_to_float32(row_bytes.data(), input_dtype, cols, row_f32.data());
                }
                
                // Compute mean of squares for this row
                {
                    ProfileScope scope(ProfileKind::OtherCompute);
                    float sum_sq = 0.0f;
                    for (int64_t j = 0; j < cols; ++j) {
                        float val = row_f32[j];
                        sum_sq += val * val;
                    }
                    float mean_sq = sum_sq / static_cast<float>(cols);
                    float rms = std::sqrt(mean_sq + eps_);
                    
                    // Normalize and scale by weight
                    for (int64_t j = 0; j < cols; ++j) {
                        result_ptr[i * cols + j] = (row_f32[j] / rms) * weight_vec[j];
                    }
                }
            }
        } else {
            // Process InMemoryMatrix (already dense)
            // Convert input to float32
            std::vector<float> input_f32(rows * cols);
            {
                ProfileScope scope(ProfileKind::Decompress);
                math::convert_buffer_to_float32(matrix->data().data(), input_dtype, rows * cols, input_f32.data());
            }
            
            {
                ProfileScope scope(ProfileKind::OtherCompute);
                for (int64_t i = 0; i < rows; ++i) {
                    // Compute mean of squares for this row
                    float sum_sq = 0.0f;
                    for (int64_t j = 0; j < cols; ++j) {
                        float val = input_f32[i * cols + j];
                        sum_sq += val * val;
                    }
                    float mean_sq = sum_sq / static_cast<float>(cols);
                    float rms = std::sqrt(mean_sq + eps_);
                    
                    // Normalize and scale by weight
                    for (int64_t j = 0; j < cols; ++j) {
                        result_ptr[i * cols + j] = (input_f32[i * cols + j] / rms) * weight_vec[j];
                    }
                }
            }
        }
        
        // Store result
        Shape result_shape = std::make_tuple(rows, cols);
        auto result = ctx.store_in_memory(
            result_id_,
            result_shape,
            DType::FLOAT32,
            std::move(result_data)
        );
        
        return result;
    }

private:
    std::string matrix_id_;
    std::string weight_id_;
    std::string result_id_;
    float eps_;
    bool store_in_memory_;
};

// Factory function
std::unique_ptr<Operator> create_row_column_rmsnorm(
    const std::string& matrix_id,
    const std::string& weight_id,
    const std::string& result_id,
    float eps,
    bool store_in_memory
) {
    return std::make_unique<RowColumnRMSNorm>(matrix_id, weight_id, result_id, eps, store_in_memory);
}

} // namespace kvtensor

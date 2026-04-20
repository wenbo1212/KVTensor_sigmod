#include "kvtensor/operators.hpp"
#include "kvtensor/operators_impl.hpp"
#include "kvtensor/context.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/profile.hpp"
#include "math/arithmetic.hpp"
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <sstream>


namespace kvtensor {

// Helper to get dtype size in bytes
inline size_t get_dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return sizeof(float);
        case DType::BFLOAT16: return sizeof(uint16_t);
        case DType::FLOAT16: return sizeof(uint16_t);
        case DType::INT8: return sizeof(int8_t);
        default: return sizeof(float);
    }
}

// RowColumnLlamaFeedForward operator
class RowColumnLlamaFeedForward : public Operator {
public:
    RowColumnLlamaFeedForward(
        const std::string& x_id,
        const std::string& gate_up_proj_id,
        const std::string& down_proj_id,
        const std::string& result_id,
        int64_t chunk_size
    ) : x_id_(x_id),
        gate_up_proj_id_(gate_up_proj_id),
        down_proj_id_(down_proj_id),
        result_id_(result_id),
        storage_chunk_size_(chunk_size) {
        name_ = "rowcolumn_llama_feedforward";
    }

    std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) override {
        // Resolve input
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
        
        // Get input shape
        auto [rows, hidden_dim] = x->shape();
        
        // Check if gate_up_proj is preloaded (in memory) or in DB (BlockMatrix)
        auto gate_up_in_memory = ctx.resolve_in_memory(gate_up_proj_id_);
        std::shared_ptr<BlockMatrix> gate_up_proj = nullptr;
        int64_t ffn_dim = 0;
        int64_t num_chunks = 0;
        DType gate_up_dtype = DType::FLOAT32;
        
        if (gate_up_in_memory) {
            // Preloaded matrix: use chunk_size from constructor parameter
            auto [gate_up_rows, gate_up_cols] = gate_up_in_memory->shape();
            ffn_dim = gate_up_cols / 2;
            gate_up_dtype = gate_up_in_memory->dtype();
            
            // Calculate number of chunks based on storage format
            // Stored chunks have width storage_chunk_size_, so num_chunks = (2*ffn_dim) / storage_chunk_size_
            if (gate_up_in_memory->packed_gate_up()) {
                num_chunks = static_cast<int64_t>(gate_up_in_memory->packed_cols().size());
            } else {
                num_chunks = (2 * ffn_dim + storage_chunk_size_ - 1) / storage_chunk_size_;
            }
        } else {
            // Matrix in DB: resolve as BlockMatrix
            gate_up_proj = ctx.resolve_block_matrix(gate_up_proj_id_);
            if (!gate_up_proj) {
                throw std::runtime_error("gate_up_proj matrix not found: " + gate_up_proj_id_);
            }
            
            auto [gate_up_rows, gate_up_cols] = gate_up_proj->shape();
            ffn_dim = gate_up_cols / 2;
            num_chunks = gate_up_proj->num_col_chunks();
            gate_up_dtype = gate_up_proj->dtype();
        }

        int64_t output_chunk_size = storage_chunk_size_ / 2;  // Output dimensions per chunk (e.g., 128)
        
        // Principle: Matmul follows weight matrix type
        // Use dtype-specific matmul functions based on weight dtype
        DType x_dtype = x->dtype();
        
        // Allocate result for activated output (rows, ffn_dim) - always in float32
        std::vector<float> activated_data(rows * ffn_dim, 0.0f);
        
        // Convert x to float32 (will be converted to weight dtype in matmul if needed)
        std::vector<float> x_f32;
        const float* x_ptr = nullptr;
        if (x_dtype == DType::FLOAT32) {
            x_ptr = reinterpret_cast<const float*>(x->data().data());
        } else {
            x_f32.resize(rows * hidden_dim);
            {
                ProfileScope scope(ProfileKind::Decompress);
                math::convert_buffer_to_float32(x->data().data(), x_dtype, rows * hidden_dim, x_f32.data());
            }
            x_ptr = x_f32.data();
        }
        
        // Keep weight matrix in its original dtype - don't convert to float32
        const uint8_t* gate_up_dense_ptr_raw = nullptr;
        if (gate_up_in_memory) {
            gate_up_dense_ptr_raw = gate_up_in_memory->data().data();
        }
        
        // Reusable buffer for combined gate+up output (always float32)
        std::vector<float> combined_chunk_buffer;
        
        // Reusable chunk buffers for different dtypes (kept alive across iterations)
        std::vector<uint8_t> extracted_chunk_bytes;  // Raw bytes in weight dtype
        std::vector<float> extracted_chunk_f32;       // For FLOAT32 weights
        std::vector<uint16_t> extracted_chunk_bf16;  // For BFLOAT16 weights
        std::vector<int8_t> extracted_chunk_int8;     // For INT8 weights
        std::vector<uint16_t> x_bf16;                 // x converted to bf16 if needed
        std::vector<int8_t> x_int8;                   // x converted to int8 if needed
        
        // Process gate_up_proj chunk by chunk
        uint64_t total_data_ns = 0;
        uint64_t total_compute_ns = 0;
        uint64_t total_activation_ns = 0;
        auto profile_gemm = [&](const std::string& op_class, int64_t m, int64_t k, int64_t n, const auto& fn) {
            if (m <= 0 || k <= 0 || n <= 0) {
                fn();
                return;
            }
            ProfileScope scope(ProfileKind::Compute);
            add_profile_gemm(op_class, m, k, n);
            fn();
        };
        for (int64_t j = 0; j < num_chunks; ++j) {
            auto data_start = std::chrono::steady_clock::now();
            int64_t chunk_rows = hidden_dim;
            int64_t chunk_cols = 0;
            int64_t actual_output_size = 0;
            const uint8_t* weight_chunk_ptr = nullptr;
            bool weight_chunk_strided = false;
            int64_t weight_stride = 0;
            
            if (gate_up_in_memory) {
                // Preloaded matrix: extract chunk slice in original dtype
                if (gate_up_in_memory->packed_gate_up()) {
                    const auto& offsets = gate_up_in_memory->packed_offsets();
                    const auto& cols = gate_up_in_memory->packed_cols();
                    if (static_cast<size_t>(j) >= offsets.size()) {
                        break;
                    }
                    chunk_cols = cols[static_cast<size_t>(j)];
                    actual_output_size = chunk_cols / 2;
                    weight_chunk_ptr = gate_up_dense_ptr_raw + offsets[static_cast<size_t>(j)];
                } else {
                    int64_t col_start = j * storage_chunk_size_;
                    int64_t col_end = std::min(col_start + storage_chunk_size_, 2 * ffn_dim);
                    chunk_cols = col_end - col_start;
                    actual_output_size = chunk_cols / 2;
                
                    // Use a strided view for float32 to avoid copying; fallback to copy for other dtypes.
                    if (gate_up_dtype == DType::FLOAT32) {
                        size_t dtype_size = get_dtype_size(gate_up_dtype);
                        weight_chunk_ptr = gate_up_dense_ptr_raw + col_start * dtype_size;
                        weight_chunk_strided = true;
                        weight_stride = 2 * ffn_dim;
                    } else {
                        size_t dtype_size = get_dtype_size(gate_up_dtype);
                        size_t chunk_bytes_size = chunk_cols * hidden_dim * dtype_size;
                        extracted_chunk_bytes.resize(chunk_bytes_size);

                        for (int64_t r = 0; r < hidden_dim; ++r) {
                            const uint8_t* row_start = gate_up_dense_ptr_raw +
                                r * (2 * ffn_dim) * dtype_size +
                                col_start * dtype_size;
                            std::memcpy(
                                extracted_chunk_bytes.data() + r * chunk_cols * dtype_size,
                                row_start,
                                chunk_cols * dtype_size
                            );
                        }
                        weight_chunk_ptr = extracted_chunk_bytes.data();
                    }
                }
            } else {
                // Matrix in DB: read chunk from BlockMatrix (already in original dtype)
                auto chunk = gate_up_proj->read_col_chunk(j, ctx);
                auto chunk_shape = gate_up_proj->col_chunk_shape(j);  // authoritative shape
                chunk_rows = std::get<0>(chunk_shape);
                chunk_cols = std::get<1>(chunk_shape);
                actual_output_size = chunk_cols / 2;
                
                // Copy chunk into a contiguous, aligned buffer in original dtype
                size_t chunk_bytes_size = chunk.size;
                if (extracted_chunk_bytes.size() < chunk_bytes_size) {
                    extracted_chunk_bytes.resize(chunk_bytes_size);
                }
                std::memcpy(extracted_chunk_bytes.data(), chunk.data, chunk_bytes_size);
                weight_chunk_ptr = extracted_chunk_bytes.data();
            }
            auto data_end = std::chrono::steady_clock::now();
            total_data_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(data_end - data_start).count()
            );
            
            // Calculate output range for this chunk
            // Chunk j contains gate+up for output dimensions [j * output_chunk_size : (j+1) * output_chunk_size]
            // But actual_output_size may be less for the last chunk
            int64_t output_start = j * output_chunk_size;
            int64_t output_end = std::min(output_start + actual_output_size, ffn_dim);
            int64_t output_size = output_end - output_start;
            
            // Ensure we don't exceed ffn_dim
            if (output_start >= ffn_dim) {
                break;  // No more output to process
            }
            if (output_size <= 0) {
                continue;  // Skip empty chunks
            }
            
            // Gate portion and up portion pointers are already set above
            // Verify we have enough data
            if (actual_output_size == 0 || chunk_cols < 2 * actual_output_size) {
                throw std::runtime_error(
                    "Invalid chunk size in feedforward: chunk_idx=" + std::to_string(j) +
                    ", chunk_cols=" + std::to_string(chunk_cols) +
                    ", actual_output_size=" + std::to_string(actual_output_size)
                );
            }
            
            // output_size should be <= actual_output_size (we can't process more than what's in the chunk)
            if (output_size > actual_output_size) {
                output_size = actual_output_size;  // Clamp to available data
            }
            
            // Compute combined gate+up matmul once: (rows, hidden_dim) @ (hidden_dim, chunk_cols)
            // Use dtype-specific matmul based on weight dtype
            if (static_cast<int64_t>(combined_chunk_buffer.size()) < rows * chunk_cols) {
                combined_chunk_buffer.resize(rows * chunk_cols);
            }
            auto compute_start = std::chrono::steady_clock::now();
            profile_gemm("ffn_gate_up_proj", rows, hidden_dim, chunk_cols, [&]() {
                if (gate_up_dtype == DType::FLOAT32) {
                    // FLOAT32 matmul
                    const float* weight_ptr = reinterpret_cast<const float*>(weight_chunk_ptr);
                    if (weight_chunk_strided) {
                        math::matmul_strided(x_ptr, rows, hidden_dim, weight_ptr, chunk_cols, weight_stride, combined_chunk_buffer.data());
                    } else {
                        math::matmul(x_ptr, rows, hidden_dim, weight_ptr, chunk_cols, combined_chunk_buffer.data());
                    }
                } else if (gate_up_dtype == DType::BFLOAT16) {
                    // BFLOAT16 matmul: convert x to bf16 if needed, then use matmul_bf16bf16f32
                    const uint16_t* weight_bf16_ptr = reinterpret_cast<const uint16_t*>(weight_chunk_ptr);

                    // Convert x to bf16 if it's not already
                    const uint16_t* x_bf16_ptr = nullptr;
                    if (x_dtype == DType::BFLOAT16) {
                        x_bf16_ptr = reinterpret_cast<const uint16_t*>(x->data().data());
                    } else {
                        // Convert x from float32 to bf16
                        x_bf16.resize(rows * hidden_dim);
                        {
                            ProfileScope scope(ProfileKind::Decompress);
                            math::convert_float32_to_bf16(x_ptr, rows * hidden_dim, x_bf16.data());
                        }
                        x_bf16_ptr = x_bf16.data();
                    }

                    math::matmul_bf16bf16f32(
                        x_bf16_ptr, rows, hidden_dim,
                        weight_bf16_ptr, chunk_cols,
                        combined_chunk_buffer.data()
                    );
                } else if (gate_up_dtype == DType::INT8) {
                    // INT8 matmul: convert x to int8 if needed, then use matmul_int8_int8_f32
                    const uint8_t* weight_uint8_ptr = weight_chunk_ptr;  // INT8 weights stored as uint8 for matmul

                    // Convert x to int8 if needed
                    const int8_t* x_int8_ptr = nullptr;
                    float scale_x = 1.0f;  // Default scale (should come from metadata)
                    if (x_dtype == DType::INT8) {
                        x_int8_ptr = reinterpret_cast<const int8_t*>(x->data().data());
                    } else {
                        // Convert x from float32 to int8
                        {
                            ProfileScope scope(ProfileKind::Decompress);
                            scale_x = math::compute_quantization_scale(x_ptr, rows * hidden_dim);
                            x_int8.resize(rows * hidden_dim);
                            math::convert_float32_to_int8(x_ptr, rows * hidden_dim, x_int8.data(), scale_x);
                        }
                        x_int8_ptr = x_int8.data();
                    }

                    float scale_w = 1.0f;  // Default scale (should come from weight metadata)
                    math::matmul_int8_int8_f32(
                        x_int8_ptr, rows, hidden_dim,
                        weight_uint8_ptr, chunk_cols,
                        combined_chunk_buffer.data(),
                        scale_x, scale_w, 1.0f
                    );
                } else {
                    // Fallback: convert to float32 and use standard matmul
                    extracted_chunk_f32.resize(chunk_rows * chunk_cols);
                    {
                        ProfileScope scope(ProfileKind::Decompress);
                        math::convert_buffer_to_float32(weight_chunk_ptr, gate_up_dtype, chunk_rows * chunk_cols, extracted_chunk_f32.data());
                    }
                    math::matmul(x_ptr, rows, hidden_dim, extracted_chunk_f32.data(), chunk_cols, combined_chunk_buffer.data());
                }
            });
            
            // Apply SiLU on gate half and multiply with up half
            auto activation_start = std::chrono::steady_clock::now();
            {
                ProfileScope scope(ProfileKind::OtherCompute);
                for (int64_t r = 0; r < rows; ++r) {
                    const float* row_ptr = combined_chunk_buffer.data() + r * chunk_cols;
                    const float* gate_ptr = row_ptr;
                    const float* up_ptr = row_ptr + actual_output_size;
                    float* out_ptr = activated_data.data() + r * ffn_dim + output_start;
                    math::silu(gate_ptr, output_size, out_ptr);
                    for (int64_t c = 0; c < output_size; ++c) {
                        out_ptr[c] *= up_ptr[c];
                    }
                }
            }
            auto activation_end = std::chrono::steady_clock::now();
            total_activation_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(activation_end - activation_start).count()
            );
            auto compute_end = std::chrono::steady_clock::now();
            total_compute_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(compute_end - compute_start).count()
            );
        }
        
        // Store activated result in float32 (no conversion - following operations will convert if needed)
        std::string activated_id = result_id_ + "_activated";
        std::vector<uint8_t> activated_bytes(activated_data.size() * sizeof(float));
        std::memcpy(activated_bytes.data(), activated_data.data(), activated_bytes.size());
        
        auto activated = ctx.store_in_memory(
            activated_id,
            std::make_tuple(rows, ffn_dim),
            DType::FLOAT32,  // Always store in float32
            std::move(activated_bytes)
        );
        
        // down = down_proj(activated)
        // down_proj is now COLUMN split (ffn_dim, hidden_dim), so normal matmul
        auto down_start = std::chrono::steady_clock::now();
        auto down_op = create_row_column_matmul(activated_id, down_proj_id_, result_id_, true);
        auto result = down_op->execute(ctx);
        auto down_end = std::chrono::steady_clock::now();
        double data_ms = total_data_ns / 1000000.0;
        double compute_ms = total_compute_ns / 1000000.0;
        double activation_ms = total_activation_ns / 1000000.0;
        double down_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(down_end - down_start).count() / 1000000.0;
        // Clear activated after down_proj
        ctx.clear_in_memory(activated_id);
        return result;
    }

private:
    std::string x_id_;
    std::string gate_up_proj_id_;
    std::string down_proj_id_;
    std::string result_id_;
    int64_t storage_chunk_size_;
};

// Factory function
std::unique_ptr<Operator> create_row_column_llama_feedforward(
    const std::string& x_id,
    const std::string& gate_up_proj_id,
    const std::string& down_proj_id,
    const std::string& result_id,
    int64_t chunk_size
) {
    return std::make_unique<RowColumnLlamaFeedForward>(x_id, gate_up_proj_id, down_proj_id, result_id, chunk_size);
}

} // namespace kvtensor

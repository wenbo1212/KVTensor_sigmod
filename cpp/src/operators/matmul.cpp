#include "kvtensor/operators.hpp"
#include "kvtensor/context.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/profile.hpp"
#include "math/arithmetic.hpp"
#include "math/matmul_strided_ext.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <numeric>

#include "kvtensor/operators_impl.hpp"

namespace kvtensor {

// Helper to get float32 data pointer
template<typename T>
const float* get_float32_ptr(const std::vector<uint8_t>& data, size_t offset = 0) {
    return reinterpret_cast<const float*>(data.data() + offset);
}

template<typename T>
float* get_float32_ptr_mut(std::vector<uint8_t>& data, size_t offset = 0) {
    return reinterpret_cast<float*>(data.data() + offset);
}

// Helper to convert buffer to float32 based on dtype
inline void convert_to_float32(
    const uint8_t* src,
    DType src_dtype,
    size_t num_elements,
    float* dst
) {
    math::convert_buffer_to_float32(src, src_dtype, num_elements, dst);
}

// Helper to get bfloat16 pointer
inline const uint16_t* get_bf16_ptr(const std::vector<uint8_t>& data) {
    return reinterpret_cast<const uint16_t*>(data.data());
}

// Helper to get int8 pointer
inline const int8_t* get_int8_ptr(const std::vector<uint8_t>& data) {
    return reinterpret_cast<const int8_t*>(data.data());
}

// Helper to get uint8 pointer (for matrix B in int8 GEMM)
inline const uint8_t* get_uint8_ptr(const std::vector<uint8_t>& data) {
    return data.data();
}

// RowColumnMatMul operator
class RowColumnMatMul : public Operator {
public:
    RowColumnMatMul(
        const std::string& lhs_id,
        const std::string& rhs_id,
        const std::string& result_id,
        bool store_in_memory = true
    ) : lhs_id_(lhs_id),
        rhs_id_(rhs_id),
        result_id_(result_id),
        store_in_memory_(store_in_memory) {
        name_ = "row_column_matmul";
    }

    std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) override {
        // Resolve inputs - check in-memory first, then BlockMatrix (don't auto-densify)
        std::shared_ptr<InMemoryMatrix> lhs_in_memory = ctx.resolve_in_memory(lhs_id_);
        std::shared_ptr<BlockMatrix> lhs_block = nullptr;
        bool lhs_is_in_memory = (lhs_in_memory != nullptr);
        
        if (!lhs_is_in_memory) {
            try {
                lhs_block = ctx.resolve_block_matrix(lhs_id_);
            } catch (const std::exception&) {
                lhs_block = nullptr;
            }
        }
        
        // Try to resolve RHS as in-memory first, then as BlockMatrix
        std::shared_ptr<InMemoryMatrix> rhs_in_memory = ctx.resolve_in_memory(rhs_id_);
        std::shared_ptr<BlockMatrix> rhs_block = nullptr;
        bool rhs_is_in_memory = (rhs_in_memory != nullptr);
        
        if (!rhs_is_in_memory) {
            rhs_block = ctx.resolve_block_matrix(rhs_id_);
        }

        if (!lhs_is_in_memory && !lhs_block) {
            throw std::runtime_error(
                "[" + name_ + "] LHS matrix not found: " + lhs_id_
            );
        }

        if (!rhs_is_in_memory && !rhs_block) {
            throw std::runtime_error(
                "[" + name_ + "] RHS matrix not found: " + rhs_id_
            );
        }
        
        // Get shapes
        int64_t lhs_rows, lhs_cols;
        if (lhs_is_in_memory) {
            auto [r, c] = lhs_in_memory->shape();
            lhs_rows = r;
            lhs_cols = c;
        } else {
            auto [r, c] = lhs_block->shape();
            lhs_rows = r;
            lhs_cols = c;
        }
        int64_t rhs_rows, rhs_cols;
        bool rhs_transposed = false;

        auto classify_op = [&]() -> std::string {
            if (rhs_id_.find("attn_qkv_proj") != std::string::npos) {
                return "attn_qkv_proj";
            }
            if (rhs_id_.find("attn_o_proj") != std::string::npos) {
                return "attn_o_proj";
            }
            if (rhs_id_.find("ffn_down_proj") != std::string::npos) {
                return "ffn_down_proj";
            }
            if (rhs_id_.find("output.output_proj") != std::string::npos) {
                return "output_proj";
            }
            return "row_column_matmul";
        };

        const std::string op_class = classify_op();

        auto profile_gemm = [&](int64_t m, int64_t k, int64_t n, const auto& fn) {
            if (m <= 0 || k <= 0 || n <= 0) {
                fn();
                return;
            }
            ProfileScope scope(ProfileKind::Compute);
            add_profile_gemm(op_class, m, k, n);
            fn();
        };

        auto profile_decompress = [&](const auto& fn) {
            ProfileScope scope(ProfileKind::Decompress);
            fn();
        };

        if (rhs_is_in_memory) {
            // RHS is in-memory (dense matrix, not transposed)
            auto [r, c] = rhs_in_memory->shape();
            rhs_rows = r;
            rhs_cols = c;
            rhs_transposed = false;  // In-memory matrices are always stored normally
        } else {
            // RHS is BlockMatrix
            auto [r, c] = rhs_block->shape();
            rhs_rows = r;
            rhs_cols = c;
            rhs_transposed = (rhs_block->split_mode() == SplitMode::ROW);
        }
        
        // Validate split mode for known weight matrices (only for BlockMatrix)
        if (!rhs_is_in_memory) {
            std::string actual_split_mode_str = (rhs_block->split_mode() == SplitMode::ROW) ? "ROW" : "COLUMN";
            // All weight matrices should be COLUMN split now (including down_proj)
            // qkv_proj, o_proj, gate_up_proj, down_proj should all be COLUMN split
            if (rhs_id_.find("qkv_proj") != std::string::npos || 
                rhs_id_.find("o_proj") != std::string::npos ||
                rhs_id_.find("gate_up_proj") != std::string::npos ||
                rhs_id_.find("gate_proj") != std::string::npos ||
                rhs_id_.find("up_proj") != std::string::npos ||
                rhs_id_.find("down_proj") != std::string::npos) {
                if (rhs_block->split_mode() == SplitMode::ROW) {
                    // Read raw metadata to print it
                    std::string metadata_json = "N/A";
                    if (auto meta = ctx.registry()->get_metadata_json(rhs_id_)) {
                        metadata_json = *meta;
                    }
                    
                    throw std::runtime_error(
                        "[" + name_ + "] Invalid split_mode for weight matrix: " +
                        "rhs_id=" + rhs_id_ + " should be COLUMN split, but BlockMatrix reports split_mode=" + actual_split_mode_str + ". " +
                        "Matrix metadata from database: " + metadata_json + ". " +
                        "This suggests the metadata was parsed incorrectly. Please check the weight database metadata parsing."
                    );
                }
            }
        }
        
        // For ROW split B (stored transposed), check b.shape[1] instead of b.shape[0]
        // For COLUMN split B (normal), check b.shape[0]
        // This matches Python implementation: line 90-93 in matmul.py
        int64_t expected_rhs_dim;
        if (rhs_transposed) {
            // B is stored transposed: check a.shape[1] == b.shape[1]
            expected_rhs_dim = rhs_cols;
        } else {
            // B is COLUMN split (normal): check a.shape[1] == b.shape[0]
            expected_rhs_dim = rhs_rows;
        }
        
        // Validate dimensions (matching Python logic)
        if (lhs_cols != expected_rhs_dim) {
            std::string rhs_type_str = rhs_is_in_memory ? "InMemory" : 
                                       (rhs_block->split_mode() == SplitMode::ROW ? "ROW" : "COLUMN");
            throw std::runtime_error(
                "[" + name_ + "] Matrix dimension mismatch for matmul: " +
                "lhs_id=" + lhs_id_ + ", lhs_shape=(" + 
                std::to_string(lhs_rows) + ", " + std::to_string(lhs_cols) + "), " +
                "rhs_id=" + rhs_id_ + ", rhs_shape=(" + 
                std::to_string(rhs_rows) + ", " + std::to_string(rhs_cols) + "), " +
                "rhs_type=" + rhs_type_str + ", " +
                "expected lhs_cols=" + std::to_string(lhs_cols) + 
                " to match " + (rhs_transposed ? "rhs.shape[1]" : "rhs.shape[0]") + 
                "=" + std::to_string(expected_rhs_dim) + ". " +
                "NOTE: If rhs_id contains 'qkv_proj', 'o_proj', 'gate_proj', or 'up_proj', " +
                "it should be COLUMN split, not ROW split. Check the weight database metadata."
            );
        }
        
        // Result shape: for ROW split B (transposed), result columns = b.shape[0]
        // For COLUMN split B (normal), result columns = b.shape[1]
        // This matches Python implementation: line 184-186 in matmul.py
        int64_t result_cols = rhs_transposed ? rhs_rows : rhs_cols;
        Shape result_shape = std::make_tuple(lhs_rows, result_cols);
        
        // Allocate result (size depends on output dtype)
        std::vector<uint8_t> result_data;
        
        // Perform multiplication
        DType lhs_dtype = lhs_is_in_memory ? lhs_in_memory->dtype() : lhs_block->dtype();
        DType rhs_dtype = rhs_is_in_memory ? rhs_in_memory->dtype() : rhs_block->dtype();
        
        // Principle: Follow the weight precision (RHS dtype)
        // If weights are BFLOAT16, quantize activations to BFLOAT16
        // If weights are INT8, quantize activations to INT8
        // If weights are FLOAT32, use FLOAT32
        
        // Determine target precision based on weight (RHS) dtype
        DType target_dtype = rhs_dtype;  // Follow weight precision
        bool need_quantize_lhs = (lhs_dtype == DType::FLOAT32 && rhs_dtype != DType::FLOAT32);
        
        // Handle different dtype combinations
        bool use_bf16_gemm = (target_dtype == DType::BFLOAT16);
        bool use_int8_gemm = (target_dtype == DType::INT8);
        bool use_float32 = (target_dtype == DType::FLOAT32);
        
        if (use_float32) {
            // FLOAT32 GEMM: weights are FLOAT32, convert LHS to FLOAT32 if needed
            size_t result_size = static_cast<size_t>(lhs_rows * result_cols) * sizeof(float);
            result_data.resize(result_size, 0);
            float* result_ptr = get_float32_ptr_mut<float>(result_data);
            
            // Convert LHS to float32 if needed
            std::vector<float> lhs_f32_quantized;
            const float* lhs_ptr;
            bool lhs_needs_conversion = (lhs_dtype != DType::FLOAT32);
            
            if (lhs_is_in_memory && rhs_is_in_memory) {
                // Both in-memory - direct matmul
                if (lhs_needs_conversion) {
                    lhs_f32_quantized.resize(lhs_rows * lhs_cols);
                    profile_decompress([&]() {
                        convert_to_float32(lhs_in_memory->data().data(), lhs_dtype, lhs_rows * lhs_cols,
                                           lhs_f32_quantized.data());
                    });
                    lhs_ptr = lhs_f32_quantized.data();
                } else {
                    lhs_ptr = get_float32_ptr<float>(lhs_in_memory->data());
                }
                const float* rhs_ptr = get_float32_ptr<float>(rhs_in_memory->data());
                profile_gemm(lhs_rows, lhs_cols, rhs_cols, [&]() {
                    profile_gemm(lhs_rows, lhs_cols, rhs_cols, [&]() {
                        math::matmul(lhs_ptr, lhs_rows, lhs_cols, rhs_ptr, rhs_cols, result_ptr);
                    });
                });
            } else if (lhs_is_in_memory && !rhs_is_in_memory) {
                // LHS in-memory, RHS BlockMatrix - process RHS chunks
                // Convert LHS to float32 if needed
                if (lhs_needs_conversion) {
                    lhs_f32_quantized.resize(lhs_rows * lhs_cols);
                    profile_decompress([&]() {
                        convert_to_float32(lhs_in_memory->data().data(), lhs_dtype, lhs_rows * lhs_cols,
                                           lhs_f32_quantized.data());
                    });
                    lhs_ptr = lhs_f32_quantized.data();
                } else {
                    lhs_ptr = get_float32_ptr<float>(lhs_in_memory->data());
                }
                
                if (!rhs_transposed && rhs_block->split_mode() == SplitMode::COLUMN) {
                    // RHS stored as COLUMN split (normal)
                    int64_t col_offset = 0;
                    for (int64_t j = 0; j < rhs_block->num_col_chunks(); ++j) {
                        auto chunk_shape = rhs_block->col_chunk_shape(j);
                        int64_t chunk_rows = std::get<0>(chunk_shape);
                        int64_t chunk_cols = std::get<1>(chunk_shape);

                        auto chunk = rhs_block->read_col_chunk(j, ctx);
                        const float* rhs_chunk_ptr = reinterpret_cast<const float*>(chunk.data);

                        if (chunk_rows != lhs_cols) {
                            throw std::runtime_error(
                                "[" + name_ + "] RHS chunk row mismatch for matmul: " +
                                "lhs_id=" + lhs_id_ + ", lhs_cols=" + std::to_string(lhs_cols) + ", " +
                                "rhs_id=" + rhs_id_ + ", chunk_idx=" + std::to_string(j) + ", " +
                                "chunk_rows=" + std::to_string(chunk_rows)
                            );
                        }

                        std::vector<float> partial(lhs_rows * chunk_cols);
                        profile_gemm(lhs_rows, lhs_cols, chunk_cols, [&]() {
                            profile_gemm(lhs_rows, lhs_cols, chunk_cols, [&]() {
                                math::matmul(lhs_ptr, lhs_rows, lhs_cols, rhs_chunk_ptr, chunk_cols, partial.data());
                            });
                        });

                        for (int64_t i = 0; i < lhs_rows; ++i) {
                            std::memcpy(
                                result_ptr + i * result_cols + col_offset,
                                partial.data() + i * chunk_cols,
                                chunk_cols * sizeof(float)
                            );
                        }

                        col_offset += chunk_cols;
                    }
                } else if (rhs_transposed && rhs_block->split_mode() == SplitMode::ROW) {
                    // RHS stored transposed as ROW split (shape: actual_cols x actual_rows)
                    int64_t col_offset = 0;
                    for (int64_t j = 0; j < rhs_block->num_row_chunks(); ++j) {
                        auto chunk_shape = rhs_block->row_chunk_shape(j);
                        int64_t chunk_rows = std::get<0>(chunk_shape); // rows in stored (transposed) matrix
                        int64_t chunk_cols = std::get<1>(chunk_shape); // should equal lhs_cols

                        auto chunk = rhs_block->read_row_chunk(j, ctx);
                        const float* rhs_chunk_ptr = reinterpret_cast<const float*>(chunk.data);

                        if (chunk_cols != lhs_cols) {
                            throw std::runtime_error(
                                "[" + name_ + "] RHS chunk col mismatch for matmul: " +
                                "lhs_id=" + lhs_id_ + ", lhs_cols=" + std::to_string(lhs_cols) + ", " +
                                "rhs_id=" + rhs_id_ + ", chunk_idx=" + std::to_string(j) + ", " +
                                "chunk_cols=" + std::to_string(chunk_cols)
                            );
                        }

                        // Use GEMM with transposed B (no explicit transpose needed)
                        std::vector<float> partial(lhs_rows * chunk_rows);
                        profile_gemm(lhs_rows, lhs_cols, chunk_rows, [&]() {
                            math::matmul_ex(
                                lhs_ptr, lhs_rows, lhs_cols,
                                rhs_chunk_ptr, chunk_rows,
                                partial.data(),
                                math::Transpose::No,
                                math::Transpose::Yes
                            );
                        });

                        for (int64_t i = 0; i < lhs_rows; ++i) {
                            std::memcpy(
                                result_ptr + i * result_cols + col_offset,
                                partial.data() + i * chunk_rows,
                                chunk_rows * sizeof(float)
                            );
                        }

                        col_offset += chunk_rows;
                    }
                } else {
                    throw std::runtime_error(
                        "[" + name_ + "] Unsupported RHS split mode for matmul: " +
                        "rhs_id=" + rhs_id_ + ", split_mode=" + 
                        (rhs_block->split_mode() == SplitMode::ROW ? "ROW" : "COLUMN")
                    );
                }
            } else {
                // LHS is BlockMatrix - densify and convert to float32 if needed
                std::vector<uint8_t> lhs_dense = lhs_block->to_dense(ctx);
                if (lhs_needs_conversion) {
                    lhs_f32_quantized.resize(lhs_rows * lhs_cols);
                    profile_decompress([&]() {
                        convert_to_float32(lhs_dense.data(), lhs_dtype, lhs_rows * lhs_cols,
                                           lhs_f32_quantized.data());
                    });
                    lhs_ptr = lhs_f32_quantized.data();
                } else {
                    lhs_ptr = reinterpret_cast<const float*>(lhs_dense.data());
                }
                
                if (rhs_is_in_memory) {
                    // RHS is in-memory - direct matmul
                    const float* rhs_ptr = get_float32_ptr<float>(rhs_in_memory->data());
                    profile_gemm(lhs_rows, lhs_cols, rhs_cols, [&]() {
                        math::matmul(lhs_ptr, lhs_rows, lhs_cols, rhs_ptr, rhs_cols, result_ptr);
                    });
                } else {
                    // Both are BlockMatrix - process RHS chunks
                    if (!rhs_transposed && rhs_block->split_mode() == SplitMode::COLUMN) {
                        int64_t col_offset = 0;
                        for (int64_t j = 0; j < rhs_block->num_col_chunks(); ++j) {
                            auto chunk_shape = rhs_block->col_chunk_shape(j);
                            int64_t chunk_rows = std::get<0>(chunk_shape);
                            int64_t chunk_cols = std::get<1>(chunk_shape);

                            auto chunk = rhs_block->read_col_chunk(j, ctx);
                            const float* rhs_chunk_ptr = reinterpret_cast<const float*>(chunk.data);

                            std::vector<float> partial(lhs_rows * chunk_cols);
                            profile_gemm(lhs_rows, lhs_cols, chunk_cols, [&]() {
                                math::matmul(lhs_ptr, lhs_rows, lhs_cols, rhs_chunk_ptr, chunk_cols, partial.data());
                            });

                            for (int64_t i = 0; i < lhs_rows; ++i) {
                                std::memcpy(
                                    result_ptr + i * result_cols + col_offset,
                                    partial.data() + i * chunk_cols,
                                    chunk_cols * sizeof(float)
                                );
                            }

                            col_offset += chunk_cols;
                        }
                    } else {
                        throw std::runtime_error(
                            "[" + name_ + "] BlockMatrix LHS with ROW-split RHS not yet implemented"
                        );
                    }
                }
            }
        } else if (use_bf16_gemm) {
            // BFloat16 GEMM: keep inputs/outputs in bf16 (no fp32 round-trip)
            result_data.resize(static_cast<size_t>(lhs_rows * result_cols) * sizeof(uint16_t));
            auto* result_ptr = reinterpret_cast<uint16_t*>(result_data.data());
            
            if (lhs_is_in_memory && rhs_is_in_memory) {
                // Both in-memory - direct bf16 GEMM
                const uint16_t* lhs_ptr;
                const uint16_t* rhs_ptr = get_bf16_ptr(rhs_in_memory->data());
                
                // Quantize LHS if needed
                std::vector<uint16_t> lhs_bf16_quantized;
                if (need_quantize_lhs) {
                    const float* lhs_f32 = get_float32_ptr<float>(lhs_in_memory->data());
                    lhs_bf16_quantized.resize(lhs_rows * lhs_cols);
                    profile_decompress([&]() {
                        math::convert_float32_to_bf16(lhs_f32, lhs_rows * lhs_cols, lhs_bf16_quantized.data());
                    });
                    lhs_ptr = lhs_bf16_quantized.data();
                } else {
                    lhs_ptr = get_bf16_ptr(lhs_in_memory->data());
                }
                
                profile_gemm(lhs_rows, lhs_cols, rhs_cols, [&]() {
                    math::matmul_bf16bf16bf16(lhs_ptr, lhs_rows, lhs_cols, rhs_ptr, rhs_cols, result_ptr);
                });
            } else if (lhs_is_in_memory && !rhs_is_in_memory) {
                // LHS in-memory, RHS BlockMatrix - process RHS chunks
                // Quantize LHS if needed
                std::vector<uint16_t> lhs_bf16_quantized;
                const uint16_t* lhs_ptr;
                if (need_quantize_lhs) {
                    const float* lhs_f32 = get_float32_ptr<float>(lhs_in_memory->data());
                    lhs_bf16_quantized.resize(lhs_rows * lhs_cols);
                    profile_decompress([&]() {
                        math::convert_float32_to_bf16(lhs_f32, lhs_rows * lhs_cols, lhs_bf16_quantized.data());
                    });
                    lhs_ptr = lhs_bf16_quantized.data();
                } else {
                    lhs_ptr = get_bf16_ptr(lhs_in_memory->data());
                }
                if (!rhs_transposed && rhs_block->split_mode() == SplitMode::COLUMN) {
                    int64_t col_offset = 0;
                    for (int64_t j = 0; j < rhs_block->num_col_chunks(); ++j) {
                        auto chunk_shape = rhs_block->col_chunk_shape(j);
                        int64_t chunk_rows = std::get<0>(chunk_shape);
                        int64_t chunk_cols = std::get<1>(chunk_shape);
                        
                        if (chunk_rows != lhs_cols) {
                            throw std::runtime_error(
                                "[" + name_ + "] RHS chunk row mismatch for bf16 matmul"
                            );
                        }
                        
                        auto chunk = rhs_block->read_col_chunk(j, ctx);
                        const uint16_t* rhs_chunk_ptr = reinterpret_cast<const uint16_t*>(chunk.data);
                        
                        profile_gemm(lhs_rows, lhs_cols, chunk_cols, [&]() {
                            math::matmul_bf16bf16bf16_out_strided(
                                lhs_ptr, lhs_rows, lhs_cols,
                                rhs_chunk_ptr, chunk_cols,
                                result_ptr + col_offset, result_cols
                            );
                        });
                        col_offset += chunk_cols;
                    }
                } else if (rhs_transposed && rhs_block->split_mode() == SplitMode::ROW) {
                    int64_t col_offset = 0;
                    for (int64_t j = 0; j < rhs_block->num_row_chunks(); ++j) {
                        auto chunk_shape = rhs_block->row_chunk_shape(j);
                        int64_t chunk_rows = std::get<0>(chunk_shape);
                        int64_t chunk_cols = std::get<1>(chunk_shape);
                        
                        if (chunk_cols != lhs_cols) {
                            throw std::runtime_error(
                                "[" + name_ + "] RHS chunk col mismatch for bf16 matmul"
                            );
                        }
                        
                        auto chunk = rhs_block->read_row_chunk(j, ctx);
                        const uint16_t* rhs_chunk_ptr = reinterpret_cast<const uint16_t*>(chunk.data);
                        
                        profile_gemm(lhs_rows, lhs_cols, chunk_rows, [&]() {
                            math::matmul_ex_bf16bf16bf16_out_strided(
                                lhs_ptr, lhs_rows, lhs_cols,
                                rhs_chunk_ptr, chunk_rows,
                                result_ptr + col_offset, result_cols,
                                math::Transpose::No,
                                math::Transpose::Yes
                            );
                        });
                        col_offset += chunk_rows;
                    }
                } else {
                    throw std::runtime_error(
                        "[" + name_ + "] Unsupported RHS split mode for bf16 matmul"
                    );
                }
            } else {
                // LHS is BlockMatrix - densify and quantize if needed
                std::vector<uint8_t> lhs_dense = lhs_block->to_dense(ctx);
                std::vector<uint16_t> lhs_bf16_quantized;
                const uint16_t* lhs_ptr;
                
                if (need_quantize_lhs) {
                    // Convert from current dtype to bfloat16
                    const float* lhs_f32 = reinterpret_cast<const float*>(lhs_dense.data());
                    lhs_bf16_quantized.resize(lhs_rows * lhs_cols);
                    profile_decompress([&]() {
                        math::convert_float32_to_bf16(lhs_f32, lhs_rows * lhs_cols, lhs_bf16_quantized.data());
                    });
                    lhs_ptr = lhs_bf16_quantized.data();
                } else {
                    lhs_ptr = reinterpret_cast<const uint16_t*>(lhs_dense.data());
                }
                
                if (rhs_is_in_memory) {
                    const uint16_t* rhs_ptr = get_bf16_ptr(rhs_in_memory->data());
                    profile_gemm(lhs_rows, lhs_cols, rhs_cols, [&]() {
                        math::matmul_bf16bf16bf16(lhs_ptr, lhs_rows, lhs_cols, rhs_ptr, rhs_cols, result_ptr);
                    });
                } else {
                    // Both are BlockMatrix - process RHS chunks
                    if (!rhs_transposed && rhs_block->split_mode() == SplitMode::COLUMN) {
                        int64_t col_offset = 0;
                        for (int64_t j = 0; j < rhs_block->num_col_chunks(); ++j) {
                            auto chunk_shape = rhs_block->col_chunk_shape(j);
                            int64_t chunk_rows = std::get<0>(chunk_shape);
                            int64_t chunk_cols = std::get<1>(chunk_shape);
                            
                            auto chunk = rhs_block->read_col_chunk(j, ctx);
                            const uint16_t* rhs_chunk_ptr = reinterpret_cast<const uint16_t*>(chunk.data);
                            
                            profile_gemm(lhs_rows, lhs_cols, chunk_cols, [&]() {
                                math::matmul_bf16bf16bf16_out_strided(
                                    lhs_ptr, lhs_rows, lhs_cols,
                                    rhs_chunk_ptr, chunk_cols,
                                    result_ptr + col_offset, result_cols
                                );
                            });
                            col_offset += chunk_cols;
                        }
                    } else {
                        throw std::runtime_error(
                            "[" + name_ + "] BlockMatrix LHS with ROW-split RHS not yet implemented for bf16"
                        );
                    }
                }
            }
        } else if (use_int8_gemm) {
            // INT8 GEMM: C = A @ B (quantize to int8 if needed, output float32)
            size_t result_size = static_cast<size_t>(lhs_rows * result_cols) * sizeof(float);
            result_data.resize(result_size, 0);
            float* result_ptr = get_float32_ptr_mut<float>(result_data);
            
            // Default quantization scales (should come from metadata in production)
            float scale_a = 1.0f;  // Scale for LHS (activations)
            float scale_b = 1.0f;  // Scale for RHS (weights) - should come from weight metadata
            float scale_c = 1.0f;  // Output scale
            
            if (lhs_is_in_memory && rhs_is_in_memory) {
                // Both in-memory - direct int8 GEMM
                const int8_t* lhs_ptr;
                const uint8_t* rhs_ptr = get_uint8_ptr(rhs_in_memory->data());
                
                // Quantize LHS if needed
                std::vector<int8_t> lhs_int8_quantized;
                if (need_quantize_lhs) {
                    const float* lhs_f32 = get_float32_ptr<float>(lhs_in_memory->data());
                    // Compute quantization scale for LHS
                    profile_decompress([&]() {
                        scale_a = math::compute_quantization_scale(lhs_f32, lhs_rows * lhs_cols);
                        lhs_int8_quantized.resize(lhs_rows * lhs_cols);
                        math::convert_float32_to_int8(lhs_f32, lhs_rows * lhs_cols, lhs_int8_quantized.data(), scale_a);
                    });
                    lhs_ptr = lhs_int8_quantized.data();
                } else {
                    lhs_ptr = get_int8_ptr(lhs_in_memory->data());
                }
                
                profile_gemm(lhs_rows, lhs_cols, rhs_cols, [&]() {
                    math::matmul_int8_int8_f32(lhs_ptr, lhs_rows, lhs_cols, rhs_ptr, rhs_cols, result_ptr, scale_a, scale_b, scale_c);
                });
            } else if (lhs_is_in_memory && !rhs_is_in_memory) {
                // LHS in-memory, RHS BlockMatrix - process RHS chunks
                // Quantize LHS if needed
                std::vector<int8_t> lhs_int8_quantized;
                const int8_t* lhs_ptr;
                if (need_quantize_lhs) {
                    const float* lhs_f32 = get_float32_ptr<float>(lhs_in_memory->data());
                    // Compute quantization scale for LHS
                    profile_decompress([&]() {
                        scale_a = math::compute_quantization_scale(lhs_f32, lhs_rows * lhs_cols);
                        lhs_int8_quantized.resize(lhs_rows * lhs_cols);
                        math::convert_float32_to_int8(lhs_f32, lhs_rows * lhs_cols, lhs_int8_quantized.data(), scale_a);
                    });
                    lhs_ptr = lhs_int8_quantized.data();
                } else {
                    lhs_ptr = get_int8_ptr(lhs_in_memory->data());
                }
                
                if (!rhs_transposed && rhs_block->split_mode() == SplitMode::COLUMN) {
                    int64_t col_offset = 0;
                    for (int64_t j = 0; j < rhs_block->num_col_chunks(); ++j) {
                        auto chunk_shape = rhs_block->col_chunk_shape(j);
                        int64_t chunk_rows = std::get<0>(chunk_shape);
                        int64_t chunk_cols = std::get<1>(chunk_shape);
                        
                        if (chunk_rows != lhs_cols) {
                            throw std::runtime_error(
                                "[" + name_ + "] RHS chunk row mismatch for int8 matmul"
                            );
                        }
                        
                        auto chunk = rhs_block->read_col_chunk(j, ctx);
                        // Convert int8 to uint8 for B matrix (add 128 offset)
                        std::vector<uint8_t> rhs_chunk_uint8(chunk.size);
                        const int8_t* chunk_int8 = reinterpret_cast<const int8_t*>(chunk.data);
                        profile_decompress([&]() {
                            math::convert_int8_to_uint8(chunk_int8, rhs_chunk_uint8.size(), rhs_chunk_uint8.data());
                        });
                        const uint8_t* rhs_chunk_ptr = rhs_chunk_uint8.data();
                        
                        profile_gemm(lhs_rows, lhs_cols, chunk_cols, [&]() {
                            math::matmul_int8_int8_f32_out_strided(
                                lhs_ptr, lhs_rows, lhs_cols,
                                rhs_chunk_ptr, chunk_cols,
                                result_ptr + col_offset, result_cols,
                                scale_a, scale_b, scale_c
                            );
                        });
                        col_offset += chunk_cols;
                    }
                } else if (rhs_transposed && rhs_block->split_mode() == SplitMode::ROW) {
                    int64_t col_offset = 0;
                    for (int64_t j = 0; j < rhs_block->num_row_chunks(); ++j) {
                        auto chunk_shape = rhs_block->row_chunk_shape(j);
                        int64_t chunk_rows = std::get<0>(chunk_shape);
                        int64_t chunk_cols = std::get<1>(chunk_shape);
                        
                        if (chunk_cols != lhs_cols) {
                            throw std::runtime_error(
                                "[" + name_ + "] RHS chunk col mismatch for int8 matmul"
                            );
                        }
                        
                        auto chunk = rhs_block->read_row_chunk(j, ctx);
                        // Convert int8 to uint8 for B matrix
                        std::vector<uint8_t> rhs_chunk_uint8(chunk.size);
                        const int8_t* chunk_int8 = reinterpret_cast<const int8_t*>(chunk.data);
                        profile_decompress([&]() {
                            math::convert_int8_to_uint8(chunk_int8, rhs_chunk_uint8.size(), rhs_chunk_uint8.data());
                        });
                        const uint8_t* rhs_chunk_ptr = rhs_chunk_uint8.data();
                        
                        profile_gemm(lhs_rows, lhs_cols, chunk_rows, [&]() {
                            math::matmul_ex_int8_int8_f32_out_strided(
                                lhs_ptr, lhs_rows, lhs_cols,
                                rhs_chunk_ptr, chunk_rows,
                                result_ptr + col_offset, result_cols,
                                math::Transpose::No,
                                math::Transpose::Yes,
                                scale_a, scale_b, scale_c
                            );
                        });
                        col_offset += chunk_rows;
                    }
                } else {
                    throw std::runtime_error(
                        "[" + name_ + "] Unsupported RHS split mode for int8 matmul"
                    );
                }
            } else {
                // LHS is BlockMatrix - densify and quantize if needed
                std::vector<uint8_t> lhs_dense = lhs_block->to_dense(ctx);
                std::vector<int8_t> lhs_int8_quantized;
                const int8_t* lhs_ptr;
                
                if (need_quantize_lhs) {
                    // Convert from current dtype to int8
                    const float* lhs_f32 = reinterpret_cast<const float*>(lhs_dense.data());
                    // Compute quantization scale for LHS
                    profile_decompress([&]() {
                        scale_a = math::compute_quantization_scale(lhs_f32, lhs_rows * lhs_cols);
                        lhs_int8_quantized.resize(lhs_rows * lhs_cols);
                        math::convert_float32_to_int8(lhs_f32, lhs_rows * lhs_cols, lhs_int8_quantized.data(), scale_a);
                    });
                    lhs_ptr = lhs_int8_quantized.data();
                } else {
                    lhs_ptr = reinterpret_cast<const int8_t*>(lhs_dense.data());
                }
                
                if (rhs_is_in_memory) {
                    const uint8_t* rhs_ptr = get_uint8_ptr(rhs_in_memory->data());
                    profile_gemm(lhs_rows, lhs_cols, rhs_cols, [&]() {
                        math::matmul_int8_int8_f32(lhs_ptr, lhs_rows, lhs_cols, rhs_ptr, rhs_cols, result_ptr, scale_a, scale_b, scale_c);
                    });
                } else {
                    // Both are BlockMatrix - process RHS chunks
                    if (!rhs_transposed && rhs_block->split_mode() == SplitMode::COLUMN) {
                        int64_t col_offset = 0;
                        for (int64_t j = 0; j < rhs_block->num_col_chunks(); ++j) {
                            auto chunk_shape = rhs_block->col_chunk_shape(j);
                            int64_t chunk_rows = std::get<0>(chunk_shape);
                            int64_t chunk_cols = std::get<1>(chunk_shape);
                            
                            auto chunk = rhs_block->read_col_chunk(j, ctx);
                            // Convert int8 to uint8 for B matrix (add 128 offset)
                            std::vector<uint8_t> rhs_chunk_uint8(chunk.size);
                            const int8_t* chunk_int8 = reinterpret_cast<const int8_t*>(chunk.data);
                            profile_decompress([&]() {
                                math::convert_int8_to_uint8(chunk_int8, rhs_chunk_uint8.size(), rhs_chunk_uint8.data());
                            });
                            const uint8_t* rhs_chunk_ptr = rhs_chunk_uint8.data();
                            
                            profile_gemm(lhs_rows, lhs_cols, chunk_cols, [&]() {
                                math::matmul_int8_int8_f32_out_strided(
                                    lhs_ptr, lhs_rows, lhs_cols,
                                    rhs_chunk_ptr, chunk_cols,
                                    result_ptr + col_offset, result_cols,
                                    scale_a, scale_b, scale_c
                                );
                            });
                            col_offset += chunk_cols;
                        }
                    } else {
                        throw std::runtime_error(
                            "[" + name_ + "] BlockMatrix LHS with ROW-split RHS not yet implemented for int8"
                        );
                    }
                }
            }
        } else {
            // Mixed dtypes or unsupported - convert to float32 and use standard matmul
            size_t result_size = static_cast<size_t>(lhs_rows * result_cols) * sizeof(float);
            result_data.resize(result_size, 0);
            float* result_ptr = get_float32_ptr_mut<float>(result_data);
            
            // Convert LHS to float32
            std::vector<float> lhs_f32(lhs_rows * lhs_cols);
            if (lhs_is_in_memory) {
                profile_decompress([&]() {
                    convert_to_float32(lhs_in_memory->data().data(), lhs_dtype, lhs_rows * lhs_cols, lhs_f32.data());
                });
            } else {
                std::vector<uint8_t> lhs_dense = lhs_block->to_dense(ctx);
                profile_decompress([&]() {
                    convert_to_float32(lhs_dense.data(), lhs_dtype, lhs_rows * lhs_cols, lhs_f32.data());
                });
            }
            
            // Convert RHS to float32 and perform matmul
            if (lhs_is_in_memory && rhs_is_in_memory) {
                std::vector<float> rhs_f32(rhs_rows * rhs_cols);
                profile_decompress([&]() {
                    convert_to_float32(rhs_in_memory->data().data(), rhs_dtype, rhs_rows * rhs_cols, rhs_f32.data());
                });
                profile_gemm(lhs_rows, lhs_cols, rhs_cols, [&]() {
                    math::matmul(lhs_f32.data(), lhs_rows, lhs_cols, rhs_f32.data(), rhs_cols, result_ptr);
                });
            } else if (lhs_is_in_memory && !rhs_is_in_memory) {
                // Process RHS chunks
                if (!rhs_transposed && rhs_block->split_mode() == SplitMode::COLUMN) {
                    int64_t col_offset = 0;
                    for (int64_t j = 0; j < rhs_block->num_col_chunks(); ++j) {
                        auto chunk_shape = rhs_block->col_chunk_shape(j);
                        int64_t chunk_rows = std::get<0>(chunk_shape);
                        int64_t chunk_cols = std::get<1>(chunk_shape);
                        
                        auto chunk = rhs_block->read_col_chunk(j, ctx);
                        std::vector<float> rhs_chunk_f32(chunk_rows * chunk_cols);
                        profile_decompress([&]() {
                            convert_to_float32(chunk.data, rhs_dtype, chunk_rows * chunk_cols, rhs_chunk_f32.data());
                        });
                        
                        std::vector<float> partial(lhs_rows * chunk_cols);
                        profile_gemm(lhs_rows, lhs_cols, chunk_cols, [&]() {
                            math::matmul(lhs_f32.data(), lhs_rows, lhs_cols, rhs_chunk_f32.data(), chunk_cols, partial.data());
                        });
                        
                        for (int64_t i = 0; i < lhs_rows; ++i) {
                            std::memcpy(
                                result_ptr + i * result_cols + col_offset,
                                partial.data() + i * chunk_cols,
                                chunk_cols * sizeof(float)
                            );
                        }
                        col_offset += chunk_cols;
                    }
                } else {
                    throw std::runtime_error(
                        "[" + name_ + "] Unsupported RHS split mode for mixed dtype matmul"
                    );
                }
            } else {
                throw std::runtime_error(
                    "[" + name_ + "] BlockMatrix LHS with mixed dtype not yet fully implemented"
                );
            }
        }

        // Store result in memory (intermediate results always stay in memory, never write to DB)
        // If LHS is in-memory, result must be in-memory (intermediate result from previous operations)
        // Only large weight matrices are chunked and stored in DB, not intermediate results
        // The store_in_memory_ flag is kept for API compatibility but intermediate results always use memory
        DType out_dtype = use_bf16_gemm ? DType::BFLOAT16 : DType::FLOAT32;
        auto result = ctx.store_in_memory(
            result_id_,
            result_shape,
            out_dtype,
            std::move(result_data)
        );
        return result;
    }

private:
    std::string lhs_id_;
    std::string rhs_id_;
    std::string result_id_;
    bool store_in_memory_;
};

// Factory function
std::unique_ptr<Operator> create_row_column_matmul(
    const std::string& lhs_id,
    const std::string& rhs_id,
    const std::string& result_id,
    bool store_in_memory
) {
    return std::make_unique<RowColumnMatMul>(lhs_id, rhs_id, result_id, store_in_memory);
}

} // namespace kvtensor

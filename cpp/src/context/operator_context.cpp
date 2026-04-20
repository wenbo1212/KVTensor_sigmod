#include "kvtensor/context.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/storage.hpp"
#include "kvtensor/types.hpp"
#include "kvtensor/bufferpool.hpp"
#include "kvtensor/profile.hpp"
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <new>
#include <thread>

namespace kvtensor {

// OperatorContext implementation
std::shared_ptr<BlockMatrix> OperatorContext::resolve_block_matrix(const std::string& matrix_id) {
    return registry_->get_matrix(matrix_id);
}

std::shared_ptr<InMemoryMatrix> OperatorContext::resolve_in_memory(const std::string& matrix_id) {
    return get_in_memory(matrix_id);
}

std::shared_ptr<InMemoryMatrix> OperatorContext::store_in_memory(
    const std::string& matrix_id,
    const Shape& shape,
    DType dtype,
    std::vector<uint8_t> data
) {
    auto matrix = std::make_shared<InMemoryMatrix>(matrix_id, shape, dtype, std::move(data));
    in_memory_cache_[matrix_id] = matrix;
    return matrix;
}

std::shared_ptr<InMemoryMatrix> OperatorContext::store_in_memory_packed_gate_up(
    const std::string& matrix_id,
    const Shape& shape,
    DType dtype,
    std::vector<uint8_t> data,
    std::vector<size_t> offsets,
    std::vector<int64_t> cols
) {
    auto matrix = std::make_shared<InMemoryMatrix>(matrix_id, shape, dtype, std::move(data));
    matrix->set_packed_gate_up(std::move(offsets), std::move(cols));
    in_memory_cache_[matrix_id] = matrix;
    return matrix;
}

std::shared_ptr<InMemoryMatrix> OperatorContext::get_in_memory(const std::string& matrix_id) {
    auto it = in_memory_cache_.find(matrix_id);
    if (it != in_memory_cache_.end()) {
        return it->second;
    }
    return nullptr;
}

void OperatorContext::clear_in_memory(const std::string& matrix_id) {
    in_memory_cache_.erase(matrix_id);
}

void OperatorContext::clear_all_in_memory() {
    in_memory_cache_.clear();
}

void OperatorContext::store_norm_weight(
    const std::string& weight_id,
    const Shape& shape,
    DType dtype,
    std::vector<uint8_t> data
) {
    auto weight = std::make_shared<InMemoryMatrix>(weight_id, shape, dtype, std::move(data));
    norm_weights_[weight_id] = weight;
}

std::shared_ptr<InMemoryMatrix> OperatorContext::get_norm_weight(const std::string& weight_id) {
    auto it = norm_weights_.find(weight_id);
    if (it != norm_weights_.end()) {
        return it->second;
    }
    return nullptr;
}

void OperatorContext::store_causal_mask(
    const std::string& mask_id,
    const Shape& shape,
    DType dtype,
    std::vector<uint8_t> data
) {
    auto mask = std::make_shared<InMemoryMatrix>(mask_id, shape, dtype, std::move(data));
    causal_masks_[mask_id] = mask;
}

std::shared_ptr<InMemoryMatrix> OperatorContext::get_causal_mask(const std::string& mask_id) {
    auto it = causal_masks_.find(mask_id);
    if (it != causal_masks_.end()) {
        return it->second;
    }
    return nullptr;
}

std::shared_ptr<InMemoryMatrix> OperatorContext::resolve(const std::string& matrix_id) {
    // Try in-memory first
    auto in_mem = get_in_memory(matrix_id);
    if (in_mem) {
        return in_mem;
    }
    
    // DO NOT auto-densify BlockMatrix - this causes excessive memory usage
    // Operators should use resolve_in_memory() or resolve_block_matrix() directly
    // and handle chunks appropriately
    // If you need dense data, explicitly call to_dense() in the operator
    
    return nullptr;
}

void OperatorContext::set_buffer_pool(std::unique_ptr<BufferPool> pool) {
    buffer_pool_ = std::move(pool);
}

PinnedChunk OperatorContext::get_weight_chunk(
    const std::string& matrix_id,
    int64_t chunk_idx,
    const BlockMatrix& matrix,
    bool use_bufferpool
) {
    if (!use_bufferpool_ || !use_bufferpool || buffer_pool_ == nullptr) {
        if (buffer_pool_ != nullptr) {
            buffer_pool_->stop_prefetch();
        }
        auto storage = storage_;
        if (!storage) {
            throw std::runtime_error("OperatorContext: storage is null for direct chunk read.");
        }
        Shape shape;
        if (matrix.split_mode() == SplitMode::ROW) {
            shape = matrix.row_chunk_shape(chunk_idx);
        } else if (matrix.split_mode() == SplitMode::COLUMN) {
            shape = matrix.col_chunk_shape(chunk_idx);
        } else {
            throw std::runtime_error("OperatorContext: unsupported split mode for direct chunk read.");
        }

        auto [rows, cols] = shape;
        size_t dtype_size = 4;
        switch (matrix.dtype()) {
            case DType::FLOAT32: dtype_size = 4; break;
            case DType::BFLOAT16:
            case DType::FLOAT16: dtype_size = 2; break;
            case DType::INT8: dtype_size = 1; break;
            default: dtype_size = 4; break;
        }
        const size_t expected_size = static_cast<size_t>(rows * cols) * dtype_size;

        if (simulate_get_chunks_) {
            if (simulated_get_latency_ms_ > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(simulated_get_latency_ms_));
            }
            void* aligned_ptr = nullptr;
            constexpr size_t kAlign = 4096;
            if (expected_size > 0) {
                if (posix_memalign(&aligned_ptr, kAlign, expected_size) != 0) {
                    throw std::bad_alloc();
                }
                std::memset(aligned_ptr, 0, expected_size);
            }
            auto owned = std::shared_ptr<void>(aligned_ptr, [](void* p) { free(p); });
            const auto* owned_ptr = reinterpret_cast<const uint8_t*>(aligned_ptr);
            return PinnedChunk(owned_ptr,
                               expected_size,
                               shape,
                               matrix.dtype(),
                               ChunkKey(matrix_id.empty() ? matrix.matrix_id() : matrix_id,
                                        chunk_idx, matrix.split_mode(), prefetch_step_),
                               0,
                               nullptr,
                               std::move(owned));
        }

        auto read_start = std::chrono::steady_clock::now();
        std::optional<std::vector<uint8_t>> data_opt;
        if (matrix.split_mode() == SplitMode::ROW) {
            data_opt = storage->get_row_chunk(matrix_id.empty() ? matrix.matrix_id() : matrix_id, chunk_idx);
        } else {
            data_opt = storage->get_col_chunk(matrix_id.empty() ? matrix.matrix_id() : matrix_id, chunk_idx);
        }
        auto read_end = std::chrono::steady_clock::now();
        if (!data_opt) {
            throw std::runtime_error("OperatorContext: chunk not found in storage.");
        }
        const size_t data_size = data_opt->size();
        uint64_t read_elapsed = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(read_end - read_start).count()
        );
        add_profile_kv_read_ns(read_elapsed);
        add_profile_bytes(static_cast<uint64_t>(data_size));
        add_profile_matrix_read(
            matrix_id.empty() ? matrix.matrix_id() : matrix_id,
            matrix.shape(),
            matrix.dtype(),
            matrix.split_mode(),
            matrix.chunk_size(),
            static_cast<uint64_t>(data_size)
        );
        void* aligned_ptr = nullptr;
        constexpr size_t kAlign = 4096;
        if (data_size > 0) {
            if (posix_memalign(&aligned_ptr, kAlign, data_size) != 0) {
                throw std::bad_alloc();
            }
            std::memcpy(aligned_ptr, data_opt->data(), data_size);
        }
        auto owned = std::shared_ptr<void>(aligned_ptr, [](void* p) { free(p); });
        const auto* owned_ptr = reinterpret_cast<const uint8_t*>(aligned_ptr);
        return PinnedChunk(owned_ptr,
                           data_size,
                           shape,
                           matrix.dtype(),
                           ChunkKey(matrix_id.empty() ? matrix.matrix_id() : matrix_id,
                                    chunk_idx, matrix.split_mode(), prefetch_step_),
                           0,
                           nullptr,
                           std::move(owned));
    }

    ChunkKey key(matrix_id.empty() ? matrix.matrix_id() : matrix_id,
                 chunk_idx, matrix.split_mode(), prefetch_step_);

    // Prefer non-blocking path; fall back to blocking if not yet loaded.
    auto t_start = std::chrono::steady_clock::now();
    if (auto opt = buffer_pool_->try_get_chunk(key)) {
        auto t_end = std::chrono::steady_clock::now();
        uint64_t elapsed = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count()
        );
        add_profile_kv_read_ns(elapsed);
        add_profile_bytes(static_cast<uint64_t>(opt->size));
        add_profile_matrix_read(
            matrix_id.empty() ? matrix.matrix_id() : matrix_id,
            matrix.shape(),
            matrix.dtype(),
            matrix.split_mode(),
            matrix.chunk_size(),
            static_cast<uint64_t>(opt->size)
        );
        return std::move(*opt);
    }

    auto chunk = buffer_pool_->get_chunk_pinned(matrix_id, chunk_idx, matrix, prefetch_step_);
    auto t_end = std::chrono::steady_clock::now();
    uint64_t elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count()
    );
    add_profile_kv_read_ns(elapsed);
    add_profile_bytes(static_cast<uint64_t>(chunk.size));
    add_profile_matrix_read(
        matrix_id.empty() ? matrix.matrix_id() : matrix_id,
        matrix.shape(),
        matrix.dtype(),
        matrix.split_mode(),
        matrix.chunk_size(),
        static_cast<uint64_t>(chunk.size)
    );
    return chunk;
}

void OperatorContext::preload_initial_chunks(
    const std::vector<ChunkKey>& sequence,
    const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices
) {
    if (buffer_pool_ == nullptr) {
        return;
    }
    buffer_pool_->preload_initial_chunks(sequence, matrices);
}

void OperatorContext::start_sequence_prefetch(
    const std::vector<ChunkKey>& sequence,
    const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices
) {
    if (buffer_pool_ == nullptr) {
        return;
    }
    buffer_pool_->start_sequence_prefetch(sequence, matrices);
}

void OperatorContext::stop_weight_prefetch() {
    if (buffer_pool_ == nullptr) {
        return;
    }
    buffer_pool_->stop_prefetch();
}

uint8_t* OperatorContext::attention_scratch(size_t bytes) {
    if (attention_scratch_.size() < bytes) {
        attention_scratch_.resize(bytes);
    }
    return attention_scratch_.data();
}

} // namespace kvtensor

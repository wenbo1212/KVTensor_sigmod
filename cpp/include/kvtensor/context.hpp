#pragma once

#include "kvtensor/matrix.hpp"
#include "kvtensor/storage.hpp"
#include "kvtensor/types.hpp"
#include "kvtensor/bufferpool.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace kvtensor {

// Forward declarations
class MatrixRegistry;
class BufferPool;

// ChunkKey forward declaration (defined in bufferpool.hpp)
struct ChunkKey;

// In-memory matrix (for intermediate results)
class InMemoryMatrix {
public:
    InMemoryMatrix(
        const std::string& matrix_id,
        const Shape& shape,
        DType dtype,
        std::vector<uint8_t> data
    ) : matrix_id_(matrix_id),
        shape_(shape),
        dtype_(dtype),
        data_(std::move(data)) {}

    const std::string& matrix_id() const { return matrix_id_; }
    Shape shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    const std::vector<uint8_t>& data() const { return data_; }
    std::vector<uint8_t>& data() { return data_; }

    void set_packed_gate_up(std::vector<size_t> offsets, std::vector<int64_t> cols) {
        packed_gate_up_ = true;
        packed_offsets_ = std::move(offsets);
        packed_cols_ = std::move(cols);
    }

    bool packed_gate_up() const { return packed_gate_up_; }
    const std::vector<size_t>& packed_offsets() const { return packed_offsets_; }
    const std::vector<int64_t>& packed_cols() const { return packed_cols_; }

private:
    std::string matrix_id_;
    Shape shape_;
    DType dtype_;
    std::vector<uint8_t> data_;
    bool packed_gate_up_ = false;
    std::vector<size_t> packed_offsets_;
    std::vector<int64_t> packed_cols_;
};

// Operator context: manages storage, registry, and in-memory cache
class OperatorContext {
public:
    OperatorContext(
        SimpleDBStorage* storage,
        MatrixRegistry* registry
    ) : storage_(storage), registry_(registry), buffer_pool_(nullptr) {}

    // Storage access
    SimpleDBStorage* storage() const { return storage_; }
    MatrixRegistry* registry() const { return registry_; }

    // Resolve matrix ID to BlockMatrix or InMemoryMatrix
    std::shared_ptr<BlockMatrix> resolve_block_matrix(const std::string& matrix_id);
    std::shared_ptr<InMemoryMatrix> resolve_in_memory(const std::string& matrix_id);
    
    // Helper to resolve either type
    std::shared_ptr<InMemoryMatrix> resolve(const std::string& matrix_id);

    // Store in-memory matrix
    std::shared_ptr<InMemoryMatrix> store_in_memory(
        const std::string& matrix_id,
        const Shape& shape,
        DType dtype,
        std::vector<uint8_t> data
    );

    std::shared_ptr<InMemoryMatrix> store_in_memory_packed_gate_up(
        const std::string& matrix_id,
        const Shape& shape,
        DType dtype,
        std::vector<uint8_t> data,
        std::vector<size_t> offsets,
        std::vector<int64_t> cols
    );

    // Get in-memory matrix
    std::shared_ptr<InMemoryMatrix> get_in_memory(const std::string& matrix_id);

    // Clear in-memory cache
    void clear_in_memory(const std::string& matrix_id);
    void clear_all_in_memory();

    // Store normalization weight (small vectors, kept resident)
    void store_norm_weight(
        const std::string& weight_id,
        const Shape& shape,
        DType dtype,
        std::vector<uint8_t> data
    );

    // Get normalization weight
    std::shared_ptr<InMemoryMatrix> get_norm_weight(const std::string& weight_id);

    // Store causal mask
    void store_causal_mask(
        const std::string& mask_id,
        const Shape& shape,
        DType dtype,
        std::vector<uint8_t> data
    );

    // Get causal mask
    std::shared_ptr<InMemoryMatrix> get_causal_mask(const std::string& mask_id);

    // BufferPool support
    void set_buffer_pool(std::unique_ptr<BufferPool> pool);
    BufferPool* buffer_pool() const { return buffer_pool_.get(); }
    void set_use_bufferpool(bool enabled) { use_bufferpool_ = enabled; }
    bool use_bufferpool() const { return use_bufferpool_; }
    void set_simulate_get_chunks(bool enabled) { simulate_get_chunks_ = enabled; }
    bool simulate_get_chunks() const { return simulate_get_chunks_; }
    void set_simulated_get_latency_ms(uint64_t ms) { simulated_get_latency_ms_ = ms; }
    uint64_t simulated_get_latency_ms() const { return simulated_get_latency_ms_; }
    void set_prefetch_step(int64_t step) { prefetch_step_ = step; }
    int64_t prefetch_step() const { return prefetch_step_; }

    // Shared attention scratch arena (reused across attention ops)
    uint8_t* attention_scratch(size_t bytes);
    
    // Get weight chunk through bufferpool
    PinnedChunk get_weight_chunk(
        const std::string& matrix_id,
        int64_t chunk_idx,
        const BlockMatrix& matrix,
        bool use_bufferpool = true
    );
    
    // Prefetch control methods
    void preload_initial_chunks(
        const std::vector<ChunkKey>& sequence,
        const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices
    );
    
    void start_sequence_prefetch(
        const std::vector<ChunkKey>& sequence,
        const std::unordered_map<std::string, std::shared_ptr<BlockMatrix>>& matrices
    );
    
    void stop_weight_prefetch();

private:
    SimpleDBStorage* storage_;
    MatrixRegistry* registry_;
    std::unique_ptr<BufferPool> buffer_pool_;
    std::unordered_map<std::string, std::shared_ptr<InMemoryMatrix>> in_memory_cache_;
    std::unordered_map<std::string, std::shared_ptr<InMemoryMatrix>> norm_weights_;
    std::unordered_map<std::string, std::shared_ptr<InMemoryMatrix>> causal_masks_;
    int64_t prefetch_step_ = 0;
    std::vector<uint8_t> attention_scratch_;
    bool use_bufferpool_ = true;
    bool simulate_get_chunks_ = false;
    uint64_t simulated_get_latency_ms_ = 0;
};

// Matrix registry (simplified interface)
class MatrixRegistry {
public:
    struct MatrixMetadata {
        std::string matrix_id;
        Shape shape;
        DType dtype;
        SplitMode split_mode;
        int64_t chunk_size;
    };

    explicit MatrixRegistry(SimpleDBStorage* storage) : storage_(storage) {}

    void save_metadata(
        const std::string& matrix_id,
        const Shape& shape,
        DType dtype,
        SplitMode split_mode,
        int64_t chunk_size
    );

    // List all matrix IDs known from metadata.jsonl (loads metadata if needed).
    std::vector<std::string> list_matrix_ids();

    std::shared_ptr<BlockMatrix> get_matrix(const std::string& matrix_id);
    std::optional<std::string> get_metadata_json(const std::string& matrix_id);

private:
    void load_metadata_file();

    SimpleDBStorage* storage_;
    std::unordered_map<std::string, std::shared_ptr<BlockMatrix>> cache_;
    std::unordered_map<std::string, MatrixMetadata> metadata_;
    std::unordered_map<std::string, std::string> metadata_json_;
    bool metadata_loaded_ = false;
};

} // namespace kvtensor

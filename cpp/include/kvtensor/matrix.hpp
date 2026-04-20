#pragma once

#include "kvtensor/storage.hpp"
#include "kvtensor/types.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace kvtensor {

// Forward declarations
class MatrixRegistry;
class OperatorContext;
struct PinnedChunk;

// BlockMatrix: Represents chunked matrices stored in SimpleKVStore
class BlockMatrix {
public:
    BlockMatrix(
        const std::string& matrix_id,
        const Shape& shape,
        SimpleDBStorage* storage,
        DType dtype,
        SplitMode split_mode = SplitMode::ROW,
        int64_t chunk_size = 1024
    );

    // Getters
    const std::string& matrix_id() const { return matrix_id_; }
    Shape shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    SplitMode split_mode() const { return split_mode_; }
    int64_t chunk_size() const { return chunk_size_; }

    // Number of chunks
    int64_t num_row_chunks() const;
    int64_t num_col_chunks() const;

    // Chunk shapes
    Shape row_chunk_shape(int64_t chunk_idx) const;
    Shape col_chunk_shape(int64_t chunk_idx) const;

    // Read chunks (require OperatorContext - all reads go through bufferpool)
    PinnedChunk read_row_chunk(int64_t chunk_idx, OperatorContext& ctx, bool use_bufferpool = true) const;
    PinnedChunk read_col_chunk(int64_t chunk_idx, OperatorContext& ctx, bool use_bufferpool = true) const;
    std::vector<uint8_t> read_row(int64_t row_idx, OperatorContext& ctx) const;

    // Write chunks
    void write_row_chunk(int64_t chunk_idx, const uint8_t* data, size_t data_size);
    void write_col_chunk(int64_t chunk_idx, const uint8_t* data, size_t data_size);

    // Get cursor for sequential iteration
    std::unique_ptr<SimpleDBCursor> get_row_chunk_cursor() const;
    std::unique_ptr<SimpleDBCursor> get_col_chunk_cursor() const;

    // Materialize full matrix (for debugging/testing)
    std::vector<uint8_t> to_dense(OperatorContext& ctx) const;

    // Save metadata to registry
    void save_metadata();

    // Create from dense data
    static BlockMatrix from_dense(
        const std::string& matrix_id,
        const uint8_t* data,
        size_t data_size,
        const Shape& shape,
        SimpleDBStorage* storage,
        DType dtype,
        SplitMode split_mode = SplitMode::ROW,
        int64_t chunk_size = 1024
    );

private:
    std::string matrix_id_;
    Shape shape_;
    SimpleDBStorage* storage_;
    DType dtype_;
    SplitMode split_mode_;
    int64_t chunk_size_;

    size_t dtype_size() const;
};

} // namespace kvtensor

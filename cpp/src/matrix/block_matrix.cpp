#include "kvtensor/matrix.hpp"
#include "kvtensor/storage.hpp"
#include "kvtensor/context.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace kvtensor {

BlockMatrix::BlockMatrix(
    const std::string& matrix_id,
    const Shape& shape,
    SimpleDBStorage* storage,
    DType dtype,
    SplitMode split_mode,
    int64_t chunk_size
) : matrix_id_(matrix_id),
    shape_(shape),
    storage_(storage),
    dtype_(dtype),
    split_mode_(split_mode),
    chunk_size_(chunk_size) {
}

size_t BlockMatrix::dtype_size() const {
    switch (dtype_) {
        case DType::FLOAT32: return 4;
        case DType::FLOAT16: return 2;
        case DType::BFLOAT16: return 2;
        case DType::INT8: return 1;
        default: return 4;
    }
}

int64_t BlockMatrix::num_row_chunks() const {
    if (split_mode_ != SplitMode::ROW) {
        throw std::runtime_error("num_row_chunks only valid for ROW split_mode");
    }
    int64_t rows = std::get<0>(shape_);
    return (rows + chunk_size_ - 1) / chunk_size_; // Ceiling division
}

int64_t BlockMatrix::num_col_chunks() const {
    if (split_mode_ != SplitMode::COLUMN) {
        throw std::runtime_error("num_col_chunks only valid for COLUMN split_mode");
    }
    int64_t cols = std::get<1>(shape_);
    return (cols + chunk_size_ - 1) / chunk_size_; // Ceiling division
}

Shape BlockMatrix::row_chunk_shape(int64_t chunk_idx) const {
    if (split_mode_ != SplitMode::ROW) {
        throw std::runtime_error("row_chunk_shape only valid for ROW split_mode");
    }
    int64_t rows = std::get<0>(shape_);
    int64_t cols = std::get<1>(shape_);
    int64_t chunk_rows = (chunk_idx < num_row_chunks() - 1) 
        ? chunk_size_ 
        : (rows - chunk_idx * chunk_size_);
    return std::make_tuple(chunk_rows, cols);
}

Shape BlockMatrix::col_chunk_shape(int64_t chunk_idx) const {
    if (split_mode_ != SplitMode::COLUMN) {
        throw std::runtime_error("col_chunk_shape only valid for COLUMN split_mode");
    }
    int64_t rows = std::get<0>(shape_);
    int64_t cols = std::get<1>(shape_);
    int64_t chunk_cols = (chunk_idx < num_col_chunks() - 1)
        ? chunk_size_
        : (cols - chunk_idx * chunk_size_);
    return std::make_tuple(rows, chunk_cols);
}

PinnedChunk BlockMatrix::read_row_chunk(int64_t chunk_idx, OperatorContext& ctx, bool use_bufferpool) const {
    if (split_mode_ != SplitMode::ROW) {
        throw std::runtime_error("read_row_chunk only valid for ROW split_mode");
    }
    
    // All chunk reads must go through bufferpool
    return ctx.get_weight_chunk(matrix_id_, chunk_idx, *this, use_bufferpool);
}

PinnedChunk BlockMatrix::read_col_chunk(int64_t chunk_idx, OperatorContext& ctx, bool use_bufferpool) const {
    if (split_mode_ != SplitMode::COLUMN) {
        throw std::runtime_error("read_col_chunk only valid for COLUMN split_mode");
    }
    
    // All chunk reads must go through bufferpool
    return ctx.get_weight_chunk(matrix_id_, chunk_idx, *this, use_bufferpool);
}

std::vector<uint8_t> BlockMatrix::read_row(int64_t row_idx, OperatorContext& ctx) const {
    int64_t rows = std::get<0>(shape_);
    int64_t cols = std::get<1>(shape_);

    if (row_idx < 0 || row_idx >= rows) {
        throw std::runtime_error("row index out of bounds");
    }

    size_t element_size = dtype_size();
    std::vector<uint8_t> row_data(cols * element_size, 0);

    if (split_mode_ == SplitMode::ROW) {
        int64_t chunk_idx = row_idx / chunk_size_;
        int64_t in_chunk_row = row_idx % chunk_size_;
        auto chunk = read_row_chunk(chunk_idx, ctx);
        auto [chunk_rows, chunk_cols] = row_chunk_shape(chunk_idx);
        if (in_chunk_row < chunk_rows) {
            size_t src_offset = in_chunk_row * chunk_cols * element_size;
            std::memcpy(row_data.data(), chunk.data + src_offset, chunk_cols * element_size);
        }
        return row_data;
    }

    if (split_mode_ == SplitMode::COLUMN) {
        int64_t col_offset = 0;
        for (int64_t j = 0; j < num_col_chunks(); ++j) {
            auto chunk = read_col_chunk(j, ctx);
            auto [chunk_rows, chunk_cols] = col_chunk_shape(j);
            if (row_idx < chunk_rows) {
                size_t src_offset = row_idx * chunk_cols * element_size;
                size_t dst_offset = col_offset * element_size;
                std::memcpy(row_data.data() + dst_offset, chunk.data + src_offset, chunk_cols * element_size);
            }
            col_offset += chunk_cols;
        }
        return row_data;
    }

    throw std::runtime_error("read_row not supported for split_mode");
}

void BlockMatrix::write_row_chunk(int64_t chunk_idx, const uint8_t* data, size_t data_size) {
    if (split_mode_ != SplitMode::ROW) {
        throw std::runtime_error("write_row_chunk only valid for ROW split_mode");
    }
    storage_->put_row_chunk(matrix_id_, chunk_idx, data, data_size);
}

void BlockMatrix::write_col_chunk(int64_t chunk_idx, const uint8_t* data, size_t data_size) {
    if (split_mode_ != SplitMode::COLUMN) {
        throw std::runtime_error("write_col_chunk only valid for COLUMN split_mode");
    }
    storage_->put_col_chunk(matrix_id_, chunk_idx, data, data_size);
}

std::unique_ptr<SimpleDBCursor> BlockMatrix::get_row_chunk_cursor() const {
    if (split_mode_ != SplitMode::ROW) {
        throw std::runtime_error("get_row_chunk_cursor only valid for ROW split_mode");
    }
    return storage_->get_row_chunk_cursor(matrix_id_);
}

std::unique_ptr<SimpleDBCursor> BlockMatrix::get_col_chunk_cursor() const {
    if (split_mode_ != SplitMode::COLUMN) {
        throw std::runtime_error("get_col_chunk_cursor only valid for COLUMN split_mode");
    }
    return storage_->get_col_chunk_cursor(matrix_id_);
}

std::vector<uint8_t> BlockMatrix::to_dense(OperatorContext& ctx) const {
    int64_t rows = std::get<0>(shape_);
    int64_t cols = std::get<1>(shape_);
    size_t element_size = dtype_size();
    size_t total_size = rows * cols * element_size;
    std::vector<uint8_t> result(total_size, 0);
    
    if (split_mode_ == SplitMode::ROW) {
        int64_t num_chunks = num_row_chunks();
        for (int64_t i = 0; i < num_chunks; ++i) {
            auto chunk_data = read_row_chunk(i, ctx);
            auto [chunk_rows, chunk_cols] = row_chunk_shape(i);
            
            int64_t row_offset = i * chunk_size_;
            for (int64_t r = 0; r < chunk_rows; ++r) {
                size_t src_offset = r * chunk_cols * element_size;
                size_t dst_offset = (row_offset + r) * cols * element_size;
                std::memcpy(
                    result.data() + dst_offset,
                    chunk_data.data + src_offset,
                    chunk_cols * element_size
                );
            }
        }
    } else if (split_mode_ == SplitMode::COLUMN) {
        int64_t num_chunks = num_col_chunks();
        for (int64_t j = 0; j < num_chunks; ++j) {
            auto chunk_data = read_col_chunk(j, ctx);
            auto [chunk_rows, chunk_cols] = col_chunk_shape(j);
            
            int64_t col_offset = j * chunk_size_;
            for (int64_t c = 0; c < chunk_cols; ++c) {
                for (int64_t r = 0; r < chunk_rows; ++r) {
                    size_t src_offset = (r * chunk_cols + c) * element_size;
                    size_t dst_offset = (r * cols + col_offset + c) * element_size;
                    std::memcpy(
                        result.data() + dst_offset,
                        chunk_data.data + src_offset,
                        element_size
                    );
                }
            }
        }
    }
    
    return result;
}

void BlockMatrix::save_metadata() {
    // This will be implemented when we integrate MatrixRegistry properly
    // For now, we'll add it to the storage interface
}

BlockMatrix BlockMatrix::from_dense(
    const std::string& matrix_id,
    const uint8_t* data,
    size_t data_size,
    const Shape& shape,
    SimpleDBStorage* storage,
    DType dtype,
    SplitMode split_mode,
    int64_t chunk_size
) {
    int64_t rows = std::get<0>(shape);
    int64_t cols = std::get<1>(shape);
    size_t element_size = (dtype == DType::FLOAT32) ? 4 : 
                         (dtype == DType::FLOAT16 || dtype == DType::BFLOAT16) ? 2 : 1;
    
    BlockMatrix matrix(matrix_id, shape, storage, dtype, split_mode, chunk_size);
    
    if (split_mode == SplitMode::ROW) {
        int64_t num_chunks = matrix.num_row_chunks();
        for (int64_t i = 0; i < num_chunks; ++i) {
            auto [chunk_rows, chunk_cols] = matrix.row_chunk_shape(i);
            size_t chunk_size_bytes = chunk_rows * chunk_cols * element_size;
            
            int64_t row_offset = i * chunk_size;
            const uint8_t* chunk_data = data + row_offset * cols * element_size;
            
            matrix.write_row_chunk(i, chunk_data, chunk_size_bytes);
        }
    } else if (split_mode == SplitMode::COLUMN) {
        int64_t num_chunks = matrix.num_col_chunks();
        for (int64_t j = 0; j < num_chunks; ++j) {
            auto [chunk_rows, chunk_cols] = matrix.col_chunk_shape(j);
            size_t chunk_size_bytes = chunk_rows * chunk_cols * element_size;
            
            std::vector<uint8_t> chunk_data(chunk_size_bytes);
            int64_t col_offset = j * chunk_size;
            
            for (int64_t c = 0; c < chunk_cols; ++c) {
                for (int64_t r = 0; r < chunk_rows; ++r) {
                    size_t src_offset = (r * cols + col_offset + c) * element_size;
                    size_t dst_offset = (r * chunk_cols + c) * element_size;
                    std::memcpy(
                        chunk_data.data() + dst_offset,
                        data + src_offset,
                        element_size
                    );
                }
            }
            
            matrix.write_col_chunk(j, chunk_data.data(), chunk_size_bytes);
        }
    }
    
    return matrix;
}

} // namespace kvtensor

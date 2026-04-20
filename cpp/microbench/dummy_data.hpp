#pragma once

#include "kvtensor/simpledb.hpp"
#include "kvtensor/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace kvtensor {
namespace microbench {

struct MatrixSpec {
    std::string db_path;
    std::string matrix_id = "bench.matrix";
    int64_t rows = 4096;
    int64_t cols = 16384;
    SplitMode split_mode = SplitMode::COLUMN;
    int64_t chunk_size = 256;
    DType dtype = DType::FLOAT32;
    uint32_t seed = 42;
    bool truncate = true;
};

struct DatasetInfo {
    std::string prefix;
    std::vector<std::string> keys;
    uint64_t total_bytes = 0;
    int64_t chunk_count = 0;
    size_t first_chunk_bytes = 0;
};

size_t dtype_size_bytes(DType dtype);

std::string format_chunk_key(const std::string& matrix_id, SplitMode split_mode, int64_t chunk_idx);

DatasetInfo create_dummy_matrix_db(const MatrixSpec& spec);

std::vector<std::string> collect_keys_with_prefix(
    const std::string& db_path,
    const std::string& prefix,
    size_t limit = 0
);

} // namespace microbench
} // namespace kvtensor

#include "dummy_data.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

namespace kvtensor {
namespace microbench {

size_t dtype_size_bytes(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return sizeof(float);
        case DType::BFLOAT16: return sizeof(uint16_t);
        case DType::INT8: return sizeof(int8_t);
        default: return 0;
    }
}

std::string format_chunk_key(const std::string& matrix_id, SplitMode split_mode, int64_t chunk_idx) {
    std::ostringstream oss;
    oss << matrix_id
        << ":"
        << (split_mode == SplitMode::COLUMN ? "col" : "row")
        << ":"
        << std::setfill('0')
        << std::setw(6)
        << chunk_idx;
    return oss.str();
}

namespace {

uint16_t f32_to_bf16(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

std::vector<uint8_t> make_random_chunk_bytes(size_t num_elements, DType dtype, std::mt19937& rng) {
    const size_t elem_bytes = dtype_size_bytes(dtype);
    if (elem_bytes == 0) {
        throw std::runtime_error("Unsupported dtype for microbench dummy data: " + dtype_to_string(dtype));
    }
    std::uniform_real_distribution<float> fdist(-0.2f, 0.2f);
    std::vector<uint8_t> out(num_elements * elem_bytes, 0);

    if (dtype == DType::FLOAT32) {
        auto* ptr = reinterpret_cast<float*>(out.data());
        for (size_t i = 0; i < num_elements; ++i) {
            ptr[i] = fdist(rng);
        }
        return out;
    }

    if (dtype == DType::BFLOAT16) {
        std::vector<float> f32(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            f32[i] = fdist(rng);
        }
        for (size_t i = 0; i < num_elements; ++i) {
            const uint16_t bf16 = f32_to_bf16(f32[i]);
            std::memcpy(out.data() + i * sizeof(uint16_t), &bf16, sizeof(uint16_t));
        }
        return out;
    }

    std::uniform_int_distribution<int> idist(-64, 64);
    auto* ptr = reinterpret_cast<int8_t*>(out.data());
    for (size_t i = 0; i < num_elements; ++i) {
        ptr[i] = static_cast<int8_t>(idist(rng));
    }
    return out;
}

void write_metadata_jsonl(const MatrixSpec& spec) {
    const std::filesystem::path meta_path = std::filesystem::path(spec.db_path) / "metadata.jsonl";
    std::ofstream out(meta_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open metadata.jsonl for writing: " + meta_path.string());
    }

    out << "{"
        << "\"matrix_id\":\"" << spec.matrix_id << "\","
        << "\"shape\":[" << spec.rows << "," << spec.cols << "],"
        << "\"dtype\":\"" << dtype_to_string(spec.dtype) << "\","
        << "\"split_mode\":\"" << split_mode_to_string(spec.split_mode) << "\","
        << "\"chunk_size\":" << spec.chunk_size
        << "}\n";
}

} // namespace

DatasetInfo create_dummy_matrix_db(const MatrixSpec& spec) {
    if (spec.db_path.empty()) {
        throw std::runtime_error("MatrixSpec.db_path is required");
    }
    if (spec.rows <= 0 || spec.cols <= 0 || spec.chunk_size <= 0) {
        throw std::runtime_error("MatrixSpec rows/cols/chunk_size must be > 0");
    }
    if (dtype_size_bytes(spec.dtype) == 0) {
        throw std::runtime_error(
            "Unsupported dtype for microbench (supported: float32, bfloat16, int8): " +
            dtype_to_string(spec.dtype)
        );
    }

    std::filesystem::create_directories(spec.db_path);

    SimpleKVOptions options;
    options.create_if_missing = true;
    options.truncate = spec.truncate;
    options.read_only = false;
    SimpleKVStore db(spec.db_path, options);

    DatasetInfo info;
    info.prefix = spec.matrix_id + ":" + (spec.split_mode == SplitMode::COLUMN ? "col" : "row") + ":";

    std::mt19937 rng(spec.seed);
    const int64_t axis_len = (spec.split_mode == SplitMode::COLUMN) ? spec.cols : spec.rows;
    const int64_t chunks = (axis_len + spec.chunk_size - 1) / spec.chunk_size;

    for (int64_t chunk_idx = 0; chunk_idx < chunks; ++chunk_idx) {
        const int64_t begin = chunk_idx * spec.chunk_size;
        const int64_t end = std::min<int64_t>(axis_len, begin + spec.chunk_size);
        const int64_t axis_chunk = end - begin;
        if (axis_chunk <= 0) {
            continue;
        }

        const int64_t rows = (spec.split_mode == SplitMode::COLUMN) ? spec.rows : axis_chunk;
        const int64_t cols = (spec.split_mode == SplitMode::COLUMN) ? axis_chunk : spec.cols;
        const size_t num_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        std::vector<uint8_t> bytes = make_random_chunk_bytes(num_elements, spec.dtype, rng);

        const std::string key = format_chunk_key(spec.matrix_id, spec.split_mode, chunk_idx);
        if (!db.put(key, bytes.data(), bytes.size())) {
            throw std::runtime_error("SimpleKVStore::put failed for key: " + key);
        }

        info.keys.push_back(key);
        info.total_bytes += bytes.size();
        info.chunk_count += 1;
        if (info.first_chunk_bytes == 0) {
            info.first_chunk_bytes = bytes.size();
        }
    }

    db.flush_index();
    write_metadata_jsonl(spec);
    return info;
}

std::vector<std::string> collect_keys_with_prefix(
    const std::string& db_path,
    const std::string& prefix,
    size_t limit
) {
    SimpleKVStore db = SimpleKVStore::OpenReadOnly(db_path);
    std::vector<std::string> keys;
    db.iterate_prefix(prefix, [&](const std::string& key, const uint8_t*, size_t) {
        keys.push_back(key);
        if (limit > 0 && keys.size() >= limit) {
            return false;
        }
        return true;
    });
    std::sort(keys.begin(), keys.end());
    return keys;
}

} // namespace microbench
} // namespace kvtensor

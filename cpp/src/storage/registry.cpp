#include "kvtensor/context.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/storage.hpp"
#include "kvtensor/types.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace kvtensor {

namespace {
class SimpleJSON {
public:
    static std::string encode(const MatrixRegistry::MatrixMetadata& meta) {
        std::ostringstream oss;
        oss << "{"
            << "\"matrix_id\":\"" << meta.matrix_id << "\","
            << "\"shape\":[" << std::get<0>(meta.shape) << "," << std::get<1>(meta.shape) << "],"
            << "\"dtype\":\"" << dtype_to_string(meta.dtype) << "\","
            << "\"split_mode\":\"" << split_mode_to_string(meta.split_mode) << "\","
            << "\"chunk_size\":" << meta.chunk_size
            << "}";
        return oss.str();
    }

    static MatrixRegistry::MatrixMetadata decode(const std::string& json_str) {
        MatrixRegistry::MatrixMetadata meta{};
        meta.shape = std::make_tuple(0, 0);  // Initialize to invalid shape
        meta.split_mode = SplitMode::ROW;  // Initialize to default (will be overwritten if found in JSON)
        meta.chunk_size = 0;

        // Parse matrix_id - handle both "matrix_id":"value" and "matrix_id": "value" (with space)
        size_t pos = json_str.find("\"matrix_id\":");
        if (pos != std::string::npos) {
            pos += 12;  // Length of "\"matrix_id\":"
            // Skip whitespace after colon
            while (pos < json_str.length() && (json_str[pos] == ' ' || json_str[pos] == '\t')) {
                pos++;
            }
            // Find opening quote
            if (pos < json_str.length() && json_str[pos] == '"') {
                pos++;  // Skip opening quote
                size_t end = json_str.find("\"", pos);
                if (end != std::string::npos && end > pos) {
                    meta.matrix_id = json_str.substr(pos, end - pos);
                }
            }
        }

        pos = json_str.find("\"shape\":");
        if (pos != std::string::npos) {
            // Find the opening bracket (may have whitespace)
            size_t bracket_start = json_str.find("[", pos);
            if (bracket_start != std::string::npos) {
                bracket_start += 1;
                // Skip whitespace
                while (bracket_start < json_str.length() && 
                       (json_str[bracket_start] == ' ' || json_str[bracket_start] == '\t')) {
                    bracket_start++;
                }
                size_t comma = json_str.find(",", bracket_start);
                size_t end = json_str.find("]", bracket_start);
                if (comma != std::string::npos && end != std::string::npos) {
                    try {
                        // Extract and trim rows
                        std::string rows_str = json_str.substr(bracket_start, comma - bracket_start);
                        // Trim whitespace
                        rows_str.erase(0, rows_str.find_first_not_of(" \t"));
                        rows_str.erase(rows_str.find_last_not_of(" \t") + 1);
                        
                        // Extract and trim cols
                        std::string cols_str = json_str.substr(comma + 1, end - comma - 1);
                        cols_str.erase(0, cols_str.find_first_not_of(" \t"));
                        cols_str.erase(cols_str.find_last_not_of(" \t") + 1);
                        
                        int64_t rows = std::stoll(rows_str);
                        int64_t cols = std::stoll(cols_str);
                        meta.shape = std::make_tuple(rows, cols);
                    } catch (const std::exception& e) {
                        throw std::runtime_error(
                            "Failed to decode shape from JSON: " + json_str.substr(bracket_start, end - bracket_start + 1) + 
                            " (error: " + e.what() + ")"
                        );
                    }
                }
            }
        }

        // block_shape is deprecated - skip it if present in JSON (for backward compatibility)
        // No need to parse it

        // Parse dtype - handle both "dtype":"value" and "dtype": "value" (with space)
        pos = json_str.find("\"dtype\":");
        if (pos != std::string::npos) {
            pos += 8;  // Length of "\"dtype\":"
            // Skip whitespace after colon
            while (pos < json_str.length() && (json_str[pos] == ' ' || json_str[pos] == '\t')) {
                pos++;
            }
            // Find opening quote
            if (pos < json_str.length() && json_str[pos] == '"') {
                pos++;  // Skip opening quote
                size_t end = json_str.find("\"", pos);
                if (end != std::string::npos && end > pos) {
                    std::string dtype_str = json_str.substr(pos, end - pos);
                    meta.dtype = string_to_dtype(dtype_str);
                }
            }
        }

        // Find split_mode field - handle both "split_mode":"value" and "split_mode": "value" (with space)
        pos = json_str.find("\"split_mode\":");
        if (pos != std::string::npos) {
            pos += 13;  // Length of "\"split_mode\":"
            // Skip whitespace after colon
            while (pos < json_str.length() && (json_str[pos] == ' ' || json_str[pos] == '\t')) {
                pos++;
            }
            // Find opening quote
            if (pos < json_str.length() && json_str[pos] == '"') {
                pos++;  // Skip opening quote
                size_t end = json_str.find("\"", pos);
                if (end != std::string::npos && end > pos) {
                    std::string mode_str = json_str.substr(pos, end - pos);
                    // Trim any whitespace (shouldn't be any, but be safe)
                    mode_str.erase(0, mode_str.find_first_not_of(" \t"));
                    mode_str.erase(mode_str.find_last_not_of(" \t") + 1);
                    SplitMode parsed_mode = string_to_split_mode(mode_str);
                    meta.split_mode = parsed_mode;
                }
            }
            // If format is wrong, keep default (ROW)
        }
        // If split_mode not found in JSON, keep default (ROW)

        pos = json_str.find("\"chunk_size\":");
        if (pos != std::string::npos) {
            pos += 13;
            size_t end = json_str.find_first_of(",}", pos);
            meta.chunk_size = std::stoll(json_str.substr(pos, end - pos));
        }

        return meta;
    }
};

} // namespace

void MatrixRegistry::save_metadata(
    const std::string& matrix_id,
    const Shape& shape,
    DType dtype,
    SplitMode split_mode,
    int64_t chunk_size
) {
    MatrixMetadata meta;
    meta.matrix_id = matrix_id;
    meta.shape = shape;
    meta.dtype = dtype;
    meta.split_mode = split_mode;
    meta.chunk_size = chunk_size;
    std::string json = SimpleJSON::encode(meta);

    std::filesystem::path meta_path = std::filesystem::path(storage_->path()) / "metadata.jsonl";
    std::ofstream out(meta_path, std::ios::binary | std::ios::app);
    if (!out) {
        throw std::runtime_error("Failed to open metadata file: " + meta_path.string());
    }
    out << json << "\n";
    if (!out) {
        throw std::runtime_error("Failed to write metadata for matrix_id: " + matrix_id);
    }

    metadata_[matrix_id] = meta;
    metadata_json_[matrix_id] = json;
}

std::vector<std::string> MatrixRegistry::list_matrix_ids() {
    load_metadata_file();
    std::vector<std::string> ids;
    ids.reserve(metadata_.size());
    for (const auto& kv : metadata_) {
        ids.push_back(kv.first);
    }
    return ids;
}

std::shared_ptr<BlockMatrix> MatrixRegistry::get_matrix(const std::string& matrix_id) {
    auto it = cache_.find(matrix_id);
    if (it != cache_.end()) {
        return it->second;
    }
    load_metadata_file();
    auto meta_it = metadata_.find(matrix_id);
    if (meta_it == metadata_.end()) {
        throw std::runtime_error("Matrix not found: " + matrix_id);
    }
    const auto& meta = meta_it->second;
    std::string json_str;
    auto json_it = metadata_json_.find(matrix_id);
    if (json_it != metadata_json_.end()) {
        json_str = json_it->second;
    } else {
        json_str = SimpleJSON::encode(meta);
    }
    
    // Debug: print JSON if it looks suspicious
    if (json_str.empty() || json_str.length() < 10) {
        throw std::runtime_error(
            "Matrix metadata JSON is too short or empty for matrix_id: " + matrix_id +
            ". JSON length: " + std::to_string(json_str.length())
        );
    }
    
    // Validate decoded metadata
    auto [rows, cols] = meta.shape;
    if (rows == 0 && cols == 0) {
        throw std::runtime_error(
            "Matrix metadata has invalid shape (0, 0) for matrix_id: " + matrix_id +
            ". Decoded JSON: " + json_str + 
            ". Please check if the metadata was written correctly."
        );
    }

    if (meta.chunk_size == 0) {
        throw std::runtime_error(
            "Matrix metadata has invalid chunk_size (0) for matrix_id: " + matrix_id +
            ". JSON: " + json_str
        );
    }

    // Debug: Log what split_mode was parsed
    std::string parsed_split_str = (meta.split_mode == SplitMode::ROW) ? "ROW" : "COLUMN";
    
    const std::string& resolved_id = meta.matrix_id.empty() ? matrix_id : meta.matrix_id;
    auto matrix = std::make_shared<BlockMatrix>(
        resolved_id, meta.shape, storage_,
        meta.dtype, meta.split_mode, meta.chunk_size
    );
    
    // Verify the matrix was created with the correct split_mode
    std::string matrix_split_str = (matrix->split_mode() == SplitMode::ROW) ? "ROW" : "COLUMN";
    if (matrix->split_mode() != meta.split_mode) {
        throw std::runtime_error(
            "Matrix split_mode mismatch: parsed=" + parsed_split_str + 
            ", matrix=" + matrix_split_str + " for matrix_id: " + matrix_id +
            ". This indicates a bug in BlockMatrix constructor."
        );
    }

    cache_[matrix_id] = matrix;
    return matrix;
}

std::optional<std::string> MatrixRegistry::get_metadata_json(const std::string& matrix_id) {
    load_metadata_file();
    auto it = metadata_json_.find(matrix_id);
    if (it == metadata_json_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void MatrixRegistry::load_metadata_file() {
    if (metadata_loaded_) {
        return;
    }
    metadata_loaded_ = true;
    std::filesystem::path meta_path = std::filesystem::path(storage_->path()) / "metadata.jsonl";
    std::ifstream in(meta_path, std::ios::binary);
    if (!in) {
        return;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        try {
            MatrixMetadata meta = SimpleJSON::decode(line);
            if (meta.matrix_id.empty()) {
                continue;
            }
            metadata_[meta.matrix_id] = meta;
            metadata_json_[meta.matrix_id] = line;
        } catch (const std::exception&) {
            continue;
        }
    }
}

} // namespace kvtensor

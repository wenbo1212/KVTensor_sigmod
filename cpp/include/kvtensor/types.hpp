#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <algorithm>
#include <cctype>
#include <iostream>

namespace kvtensor {

// Matrix split mode
enum class SplitMode {
    ROW,      // Row chunk-based splitting
    COLUMN    // Column chunk-based splitting
};

// Data type enumeration
enum class DType {
    FLOAT32,
    FLOAT16,
    BFLOAT16,
    INT8
};

// Convert DType to string
inline std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return "float32";
        case DType::FLOAT16: return "float16";
        case DType::BFLOAT16: return "bfloat16";
        case DType::INT8: return "int8";
        default: return "unknown";
    }
}

// Convert string to DType (case-insensitive)
inline DType string_to_dtype(const std::string& str) {
    // Convert to lowercase for case-insensitive matching
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
    
    if (lower_str == "float32" || lower_str == "fp32") return DType::FLOAT32;
    if (lower_str == "float16" || lower_str == "fp16") return DType::FLOAT16;
    if (lower_str == "bfloat16" || lower_str == "bf16" || lower_str == "bfloat") return DType::BFLOAT16;
    if (lower_str == "int8" || lower_str == "i8") return DType::INT8;
    
    return DType::FLOAT32; // Default
}

// Convert SplitMode to string
inline std::string split_mode_to_string(SplitMode mode) {
    switch (mode) {
        case SplitMode::ROW: return "row";
        case SplitMode::COLUMN: return "column";
        default: return "unknown";
    }
}

// Convert string to SplitMode
inline SplitMode string_to_split_mode(const std::string& str) {
    if (str == "row") return SplitMode::ROW;
    if (str == "column") return SplitMode::COLUMN;
    return SplitMode::ROW; // Default
}

// Matrix shape: (rows, cols)
using Shape = std::tuple<int64_t, int64_t>;

} // namespace kvtensor

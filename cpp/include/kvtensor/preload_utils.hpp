#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace kvtensor {

// Expand range patterns like [0-15] to individual values
// Example: "transformer.[0-15].attn_qkv_proj" expands to:
//   transformer.0.attn_qkv_proj, transformer.1.attn_qkv_proj, ..., transformer.15.attn_qkv_proj
// Supports multiple ranges in the same pattern
std::vector<std::string> expand_ranges(const std::string& pattern);

// Read preload matrix IDs from a text file
// Each line can be:
//   - A plain matrix ID: "transformer.0.attn_qkv_proj"
//   - A pattern with ranges: "transformer.[0-15].attn_qkv_proj"
//   - A comment (starts with #): "# This is a comment"
//   - Empty lines are skipped
// Returns a set of expanded matrix IDs
std::unordered_set<std::string> read_preload_matrix_ids(const std::string& file_path);

} // namespace kvtensor

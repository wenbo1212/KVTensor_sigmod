#include "kvtensor/preload_utils.hpp"
#include <fstream>
#include <regex>
#include <algorithm>
#include <functional>
#include <iostream>

namespace kvtensor {

std::vector<std::string> expand_ranges(const std::string& pattern) {
    std::vector<std::string> results;
    
    // Find all range patterns [N-M] where N and M are integers
    std::regex range_regex(R"(\[(\d+)-(\d+)\])");
    std::smatch match;
    
    std::string remaining = pattern;
    std::vector<std::pair<size_t, std::pair<int, int>>> ranges; // position, start, end
    
    // Find all ranges and their positions
    std::string::const_iterator search_start = remaining.cbegin();
    while (std::regex_search(search_start, remaining.cend(), match, range_regex)) {
        int start = std::stoi(match[1].str());
        int end = std::stoi(match[2].str());
        size_t pos = match.position(0) + (search_start - remaining.cbegin());
        
        if (start > end) {
            // Invalid range, skip it
            search_start = match[0].second;
            continue;
        }
        
        ranges.emplace_back(pos, std::make_pair(start, end));
        search_start = match[0].second;
    }
    
    if (ranges.empty()) {
        // No ranges found, return the pattern as-is
        results.push_back(pattern);
        return results;
    }
    
    // Sort ranges by position (right to left to avoid position shifts when replacing)
    std::sort(ranges.begin(), ranges.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Expand ranges recursively (handle multiple ranges)
    std::function<void(const std::string&, size_t, std::vector<std::string>&)> expand_recursive =
        [&](const std::string& current, size_t range_idx, std::vector<std::string>& acc) {
            if (range_idx >= ranges.size()) {
                acc.push_back(current);
                return;
            }
            
            const auto& range_info = ranges[range_idx];
            size_t pos = range_info.first;
            int start = range_info.second.first;
            int end = range_info.second.second;
            
            // Replace this range with each value
            for (int val = start; val <= end; ++val) {
                std::string new_str = current;
                std::string range_str = "[" + std::to_string(start) + "-" + std::to_string(end) + "]";
                std::string replacement = std::to_string(val);
                new_str.replace(pos, range_str.length(), replacement);
                expand_recursive(new_str, range_idx + 1, acc);
            }
        };
    
    expand_recursive(pattern, 0, results);
    return results;
}

std::unordered_set<std::string> read_preload_matrix_ids(const std::string& file_path) {
    std::unordered_set<std::string> preload_ids;
    
    if (file_path.empty()) {
        return preload_ids;
    }
    
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open preload file: " << file_path << std::endl;
        return preload_ids;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Expand ranges (e.g., transformer.[0-15].attn_qkv_proj)
        std::vector<std::string> expanded = expand_ranges(line);
        for (const auto& matrix_id : expanded) {
            preload_ids.insert(matrix_id);
        }
    }
    
    return preload_ids;
}

} // namespace kvtensor

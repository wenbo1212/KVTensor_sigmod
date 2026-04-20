#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace kvtensor {
namespace microbench {

struct LatencyStats {
    size_t count = 0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double avg_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double total_ms = 0.0;
};

uint64_t now_ns();

std::vector<std::string> split(const std::string& s, char delim);
std::string trim(const std::string& s);
std::string to_lower(std::string s);

std::vector<int64_t> parse_i64_list(const std::string& s, char delim = ',');
std::vector<int> parse_int_list(const std::string& s, char delim = ',');

LatencyStats summarize_latencies_ms(const std::vector<double>& samples_ms);
double bytes_to_mb(double bytes);
std::string format_double(double value, int precision = 6);

void configure_threads(int threads);

std::string csv_escape(const std::string& raw);
void write_csv_header(std::ofstream& out, const std::vector<std::string>& columns);
void write_csv_row(
    std::ofstream& out,
    const std::vector<std::string>& columns,
    const std::unordered_map<std::string, std::string>& values
);

} // namespace microbench
} // namespace kvtensor

#include "common.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kvtensor {
namespace microbench {

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        parts.push_back(item);
    }
    return parts;
}

std::string trim(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin])) != 0) {
        ++begin;
    }
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        --end;
    }
    return s.substr(begin, end - begin);
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::vector<int64_t> parse_i64_list(const std::string& s, char delim) {
    std::vector<int64_t> out;
    for (const auto& token : split(s, delim)) {
        const std::string t = trim(token);
        if (t.empty()) {
            continue;
        }
        out.push_back(std::stoll(t));
    }
    return out;
}

std::vector<int> parse_int_list(const std::string& s, char delim) {
    std::vector<int> out;
    for (const auto& token : split(s, delim)) {
        const std::string t = trim(token);
        if (t.empty()) {
            continue;
        }
        out.push_back(std::stoi(t));
    }
    return out;
}

namespace {

double percentile_sorted(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) {
        return 0.0;
    }
    if (sorted.size() == 1) {
        return sorted[0];
    }
    const double idx = (p / 100.0) * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) {
        return sorted[lo];
    }
    const double alpha = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - alpha) + sorted[hi] * alpha;
}

} // namespace

LatencyStats summarize_latencies_ms(const std::vector<double>& samples_ms) {
    LatencyStats stats;
    if (samples_ms.empty()) {
        return stats;
    }
    std::vector<double> sorted = samples_ms;
    std::sort(sorted.begin(), sorted.end());

    stats.count = sorted.size();
    stats.min_ms = sorted.front();
    stats.max_ms = sorted.back();
    stats.total_ms = 0.0;
    for (double v : sorted) {
        stats.total_ms += v;
    }
    stats.avg_ms = stats.total_ms / static_cast<double>(stats.count);
    stats.p50_ms = percentile_sorted(sorted, 50.0);
    stats.p95_ms = percentile_sorted(sorted, 95.0);
    stats.p99_ms = percentile_sorted(sorted, 99.0);
    return stats;
}

double bytes_to_mb(double bytes) {
    return bytes / (1024.0 * 1024.0);
}

std::string format_double(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

void configure_threads(int threads) {
    if (threads <= 0) {
        return;
    }
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    const std::string value = std::to_string(threads);
    setenv("OMP_NUM_THREADS", value.c_str(), 1);
}

std::string csv_escape(const std::string& raw) {
    if (raw.find_first_of(",\"\n") == std::string::npos) {
        return raw;
    }
    std::string escaped;
    escaped.reserve(raw.size() + 2);
    escaped.push_back('"');
    for (char c : raw) {
        if (c == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}

void write_csv_header(std::ofstream& out, const std::vector<std::string>& columns) {
    for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << csv_escape(columns[i]);
    }
    out << "\n";
}

void write_csv_row(
    std::ofstream& out,
    const std::vector<std::string>& columns,
    const std::unordered_map<std::string, std::string>& values
) {
    for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        auto it = values.find(columns[i]);
        if (it == values.end()) {
            out << "";
        } else {
            out << csv_escape(it->second);
        }
    }
    out << "\n";
}

} // namespace microbench
} // namespace kvtensor

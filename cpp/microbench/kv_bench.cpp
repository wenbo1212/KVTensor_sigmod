#include "common.hpp"
#include "dummy_data.hpp"

#include "kvtensor/simpledb.hpp"
#include "kvtensor/types.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace kvtensor;
using namespace kvtensor::microbench;

bool is_supported_dtype(DType dtype) {
    return dtype == DType::FLOAT32 || dtype == DType::BFLOAT16 || dtype == DType::INT8;
}

DType parse_supported_dtype(const std::string& raw) {
    const std::string norm = to_lower(trim(raw));
    if (norm == "float32" || norm == "fp32") {
        return DType::FLOAT32;
    }
    if (norm == "bfloat16" || norm == "bf16" || norm == "bfloat") {
        return DType::BFLOAT16;
    }
    if (norm == "int8" || norm == "i8") {
        return DType::INT8;
    }
    throw std::runtime_error("Unsupported dtype for kv bench. Use float32, bfloat16, or int8.");
}

struct Config {
    std::string db_root = "/tmp/kvtensor_microbench/kv";
    std::string matrix_id = "bench.matrix";
    int64_t rows = 4096;
    int64_t cols = 16384;
    SplitMode split_mode = SplitMode::COLUMN;
    DType dtype = DType::FLOAT32;
    std::vector<int64_t> chunk_size_list = {64, 128, 256, 512};
    std::vector<int> thread_list = {1};
    std::vector<int> window_list = {1, 2, 4, 8};
    size_t scan_passes = 2;
    size_t max_keys = 1024;
    std::string csv_path;
};

struct WorkloadResult {
    LatencyStats latency;
    uint64_t bytes = 0;
    size_t requests = 0;
};

void print_help(const char* exe) {
    std::cout << "Usage: " << exe << " [options]\n"
              << "  --db-root PATH               Root path for generated benchmark DBs\n"
              << "  --matrix-id ID               Matrix id prefix\n"
              << "  --rows N                     Matrix rows\n"
              << "  --cols N                     Matrix cols\n"
              << "  --split row|column           Split mode\n"
              << "  --dtype float32|bfloat16|int8\n"
              << "  --chunk-size-list a,b,c      Chunk size parameter (element count)\n"
              << "  --window-list a,b,c          Number of chunks per sequential window (w)\n"
              << "  --threads a,b,c              Thread counts\n"
              << "  --scan-passes N              Number of full scans per config\n"
              << "  --max-keys N                 Limit keys per scan (0 = all)\n"
              << "  --csv PATH                   Optional CSV output\n"
              << "  --help\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto read_string = [&](std::string& out) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            out = argv[++i];
        };
        auto read_i64 = [&](int64_t& out) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            out = std::stoll(argv[++i]);
        };
        auto read_size = [&](size_t& out) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            out = static_cast<size_t>(std::stoull(argv[++i]));
        };

        if (arg == "--db-root") {
            read_string(cfg.db_root);
        } else if (arg == "--matrix-id") {
            read_string(cfg.matrix_id);
        } else if (arg == "--rows") {
            read_i64(cfg.rows);
        } else if (arg == "--cols") {
            read_i64(cfg.cols);
        } else if (arg == "--split") {
            std::string mode;
            read_string(mode);
            cfg.split_mode = string_to_split_mode(to_lower(mode));
        } else if (arg == "--dtype") {
            std::string dtype;
            read_string(dtype);
            cfg.dtype = parse_supported_dtype(dtype);
        } else if (arg == "--chunk-size-list") {
            std::string list;
            read_string(list);
            cfg.chunk_size_list = parse_i64_list(list);
        } else if (arg == "--window-list") {
            std::string list;
            read_string(list);
            cfg.window_list = parse_int_list(list);
        } else if (arg == "--threads") {
            std::string list;
            read_string(list);
            cfg.thread_list = parse_int_list(list);
        } else if (arg == "--scan-passes") {
            read_size(cfg.scan_passes);
        } else if (arg == "--max-keys") {
            read_size(cfg.max_keys);
        } else if (arg == "--csv") {
            read_string(cfg.csv_path);
        } else if (arg == "--help") {
            print_help(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.chunk_size_list.empty()) {
        throw std::runtime_error("chunk-size-list cannot be empty");
    }
    if (cfg.window_list.empty()) {
        throw std::runtime_error("window-list cannot be empty");
    }
    if (cfg.thread_list.empty()) {
        throw std::runtime_error("threads cannot be empty");
    }
    if (!is_supported_dtype(cfg.dtype)) {
        throw std::runtime_error("Unsupported dtype for kv bench. Use float32, bfloat16, or int8.");
    }
    return cfg;
}

WorkloadResult bench_sequential_window(
    SimpleKVStore& db,
    const std::vector<std::string>& keys,
    int window_chunks,
    size_t scan_passes
) {
    WorkloadResult out;
    if (keys.empty() || window_chunks <= 0 || scan_passes == 0) {
        return out;
    }

    std::vector<double> samples_ms;
    samples_ms.reserve((keys.size() * scan_passes) / static_cast<size_t>(window_chunks) + 1);
    AlignedString value;

    for (size_t pass = 0; pass < scan_passes; ++pass) {
        size_t i = 0;
        while (i < keys.size()) {
            const uint64_t t0 = now_ns();
            int fetched = 0;
            for (; fetched < window_chunks && i < keys.size(); ++fetched, ++i) {
                if (!db.get_into(keys[i], value)) {
                    throw std::runtime_error("get_into failed for key: " + keys[i]);
                }
                out.bytes += value.size();
            }
            const uint64_t t1 = now_ns();
            samples_ms.push_back(static_cast<double>(t1 - t0) / 1e6);
            out.requests += 1;
        }
    }

    out.latency = summarize_latencies_ms(samples_ms);
    return out;
}

std::unordered_map<std::string, std::string> make_row(
    const std::string& config_id,
    DType dtype,
    int chunk_size_param,
    int chunk_size_bytes,
    int window_chunks,
    int thread_count,
    const WorkloadResult& r
) {
    const double mb = bytes_to_mb(static_cast<double>(r.bytes));
    const double total_ms = r.latency.total_ms;
    const double mb_s = total_ms > 0.0 ? (mb * 1000.0 / total_ms) : 0.0;
    const double req_s = total_ms > 0.0 ? (static_cast<double>(r.requests) * 1000.0 / total_ms) : 0.0;
    const int read_size_bytes = chunk_size_bytes * window_chunks;

    return {
        {"benchmark", "kv"},
        {"config_id", config_id},
        {"workload", "window_seq"},
        {"dtype", dtype_to_string(dtype)},
        {"thread_count", std::to_string(thread_count)},
        {"chunk_size_param", std::to_string(chunk_size_param)},
        {"chunk_size_bytes", std::to_string(chunk_size_bytes)},
        {"window_chunks", std::to_string(window_chunks)},
        {"read_size_bytes", std::to_string(read_size_bytes)},
        {"request_count", std::to_string(r.requests)},
        {"bytes_transferred", std::to_string(r.bytes)},
        {"avg_ms", format_double(r.latency.avg_ms)},
        {"p50_ms", format_double(r.latency.p50_ms)},
        {"p95_ms", format_double(r.latency.p95_ms)},
        {"p99_ms", format_double(r.latency.p99_ms)},
        {"total_ms", format_double(total_ms)},
        {"throughput_mb_s", format_double(mb_s)},
        {"requests_per_s", format_double(req_s)}
    };
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        const std::vector<std::string> csv_columns = {
            "benchmark",
            "config_id",
            "workload",
            "dtype",
            "thread_count",
            "chunk_size_param",
            "chunk_size_bytes",
            "window_chunks",
            "read_size_bytes",
            "request_count",
            "bytes_transferred",
            "avg_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "total_ms",
            "throughput_mb_s",
            "requests_per_s"
        };

        std::ofstream csv_out;
        if (!cfg.csv_path.empty()) {
            csv_out.open(cfg.csv_path, std::ios::trunc);
            if (!csv_out) {
                throw std::runtime_error("Failed to open CSV output: " + cfg.csv_path);
            }
            write_csv_header(csv_out, csv_columns);
        }

        std::cout << "=== KV Sequential Window Microbenchmark ===\n"
                  << "dtype=" << dtype_to_string(cfg.dtype)
                  << " split=" << split_mode_to_string(cfg.split_mode)
                  << " rows=" << cfg.rows
                  << " cols=" << cfg.cols
                  << " chunk_size_mode=element_count"
                  << " scan_passes=" << cfg.scan_passes
                  << " max_keys=" << cfg.max_keys
                  << std::endl;

        for (const int thread_count : cfg.thread_list) {
            configure_threads(thread_count);

            for (const int64_t chunk_size_param : cfg.chunk_size_list) {
                if (chunk_size_param <= 0) {
                    throw std::runtime_error("chunk-size-list values must be > 0");
                }
                MatrixSpec spec;
                spec.db_path = cfg.db_root + "/s" + std::to_string(chunk_size_param) + "_t" + std::to_string(thread_count);
                spec.matrix_id = cfg.matrix_id;
                spec.rows = cfg.rows;
                spec.cols = cfg.cols;
                spec.split_mode = cfg.split_mode;
                spec.chunk_size = chunk_size_param;
                spec.dtype = cfg.dtype;
                spec.seed = 42u + static_cast<uint32_t>(chunk_size_param * 31 + thread_count);
                spec.truncate = true;

                DatasetInfo data_info = create_dummy_matrix_db(spec);
                std::vector<std::string> keys = data_info.keys;
                if (cfg.max_keys > 0 && keys.size() > cfg.max_keys) {
                    keys.resize(cfg.max_keys);
                }
                if (keys.empty()) {
                    continue;
                }

                SimpleKVStore db = SimpleKVStore::OpenReadOnly(spec.db_path);
                AlignedString first_value;
                if (!db.get_into(keys.front(), first_value)) {
                    throw std::runtime_error("Failed to read first key for chunk size setup");
                }
                const int chunk_size_bytes = static_cast<int>(first_value.size());

                for (const int window_chunks : cfg.window_list) {
                    if (window_chunks <= 0) {
                        throw std::runtime_error("window-list values must be > 0");
                    }

                    const std::string config_id =
                        "kv_window:s" + std::to_string(chunk_size_param) +
                        ":b" + std::to_string(chunk_size_bytes) +
                        ":w" + std::to_string(window_chunks) +
                        ":t" + std::to_string(thread_count) +
                        ":d=" + dtype_to_string(cfg.dtype);

                    WorkloadResult r = bench_sequential_window(db, keys, window_chunks, cfg.scan_passes);
                    auto row = make_row(
                        config_id,
                        cfg.dtype,
                        static_cast<int>(chunk_size_param),
                        chunk_size_bytes,
                        window_chunks,
                        thread_count,
                        r
                    );

                    std::cout << row.at("config_id")
                              << " avg_ms=" << row.at("avg_ms")
                              << " throughput_mb_s=" << row.at("throughput_mb_s")
                              << std::endl;

                    if (csv_out) {
                        write_csv_row(csv_out, csv_columns, row);
                    }
                }
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

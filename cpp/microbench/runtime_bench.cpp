#include "common.hpp"
#include "dummy_data.hpp"

#include "kvtensor/simpledb.hpp"
#include "kvtensor/types.hpp"
#include "math/arithmetic.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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
    throw std::runtime_error("Unsupported dtype for runtime bench. Use float32, bfloat16, or int8.");
}

struct Config {
    std::string db_root = "/tmp/kvtensor_microbench/runtime";
    std::string matrix_id = "bench.matrix";
    int64_t rows = 4096;
    int64_t cols = 16384;
    DType dtype = DType::FLOAT32;
    SplitMode split_mode = SplitMode::COLUMN;
    std::vector<int64_t> chunk_size_list = {64, 128, 256, 512};
    std::vector<int> thread_list = {1};
    std::vector<int> prefetch_depth_list = {0, 1};
    size_t max_keys = 128;
    size_t iters = 5;
    std::string csv_path;
};

struct FetchResult {
    AlignedString value;
    double fetch_ms = 0.0;
    bool ok = false;
};

struct IterationMetrics {
    double total_ms = 0.0;
    double io_total_ms = 0.0;
    double io_exposed_ms = 0.0;
    double compute_ms = 0.0;
    double merge_ms = 0.0;
    uint64_t bytes = 0;
    size_t requests = 0;
};

struct LoopResult {
    LatencyStats total_latency;
    double io_total_ms_avg = 0.0;
    double io_exposed_ms_avg = 0.0;
    double compute_ms_avg = 0.0;
    double merge_ms_avg = 0.0;
    double runtime_overhead_ms_avg = 0.0;
    uint64_t bytes = 0;
    size_t requests = 0;
};

void print_help(const char* exe) {
    std::cout << "Usage: " << exe << " [options]\n"
              << "  --db-root PATH                 Root path for generated benchmark DBs\n"
              << "  --matrix-id ID                 Matrix id\n"
              << "  --rows N                       Matrix rows\n"
              << "  --cols N                       Matrix cols\n"
              << "  --dtype float32|bfloat16|int8  Matrix dtype\n"
              << "  --chunk-size-list a,b,c        Chunk size parameter (element count)\n"
              << "  --threads a,b,c                Thread counts\n"
              << "  --prefetch-depth a,b,c         0=no overlap, 1=double-buffer overlap\n"
              << "  --max-keys N                   Number of chunks consumed per iteration\n"
              << "  --iters N                      Full-loop iterations per config\n"
              << "  --csv PATH                     Optional CSV output\n"
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
        } else if (arg == "--dtype") {
            std::string dtype;
            read_string(dtype);
            cfg.dtype = parse_supported_dtype(dtype);
        } else if (arg == "--chunk-size-list") {
            std::string list;
            read_string(list);
            cfg.chunk_size_list = parse_i64_list(list);
        } else if (arg == "--threads") {
            std::string list;
            read_string(list);
            cfg.thread_list = parse_int_list(list);
        } else if (arg == "--prefetch-depth") {
            std::string list;
            read_string(list);
            cfg.prefetch_depth_list = parse_int_list(list);
        } else if (arg == "--max-keys") {
            read_size(cfg.max_keys);
        } else if (arg == "--iters") {
            read_size(cfg.iters);
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
    if (cfg.thread_list.empty()) {
        throw std::runtime_error("threads cannot be empty");
    }
    if (cfg.prefetch_depth_list.empty()) {
        throw std::runtime_error("prefetch-depth cannot be empty");
    }
    if (!is_supported_dtype(cfg.dtype)) {
        throw std::runtime_error("Unsupported dtype for runtime bench. Use float32, bfloat16, or int8.");
    }
    return cfg;
}

FetchResult fetch_key(SimpleKVStore* db, const std::string& key) {
    FetchResult res;
    const uint64_t t0 = now_ns();
    res.ok = db->get_into(key, res.value);
    const uint64_t t1 = now_ns();
    res.fetch_ms = static_cast<double>(t1 - t0) / 1e6;
    return res;
}

int64_t process_chunk(
    const AlignedString& chunk,
    DType dtype,
    int64_t rows,
    int64_t out_offset,
    const std::vector<float>& activation,
    std::vector<float>& output,
    double* compute_ms,
    double* merge_ms
) {
    if (chunk.empty()) {
        return 0;
    }

    const size_t dsize = dtype_size_bytes(dtype);
    if (dsize == 0) {
        throw std::runtime_error("Invalid dtype size");
    }
    const size_t num_elements = chunk.size() / dsize;
    if (num_elements % static_cast<size_t>(rows) != 0) {
        throw std::runtime_error("Chunk element count is not divisible by rows");
    }
    const int64_t chunk_cols = static_cast<int64_t>(num_elements / static_cast<size_t>(rows));

    std::vector<float> weights_f32(num_elements);
    if (dtype == DType::FLOAT32) {
        const float* src = reinterpret_cast<const float*>(chunk.data());
        std::copy(src, src + num_elements, weights_f32.begin());
    } else {
        math::convert_buffer_to_float32(
            reinterpret_cast<const uint8_t*>(chunk.data()),
            dtype,
            num_elements,
            weights_f32.data()
        );
    }

    std::vector<float> partial(static_cast<size_t>(chunk_cols), 0.0f);
    const uint64_t t_compute0 = now_ns();
    for (int64_t c = 0; c < chunk_cols; ++c) {
        float acc = 0.0f;
        for (int64_t r = 0; r < rows; ++r) {
            acc += activation[static_cast<size_t>(r)] *
                weights_f32[static_cast<size_t>(r * chunk_cols + c)];
        }
        partial[static_cast<size_t>(c)] = acc;
    }
    const uint64_t t_compute1 = now_ns();
    *compute_ms += static_cast<double>(t_compute1 - t_compute0) / 1e6;

    if (output.size() < static_cast<size_t>(out_offset + chunk_cols)) {
        output.resize(static_cast<size_t>(out_offset + chunk_cols), 0.0f);
    }
    const uint64_t t_merge0 = now_ns();
    for (int64_t c = 0; c < chunk_cols; ++c) {
        output[static_cast<size_t>(out_offset + c)] += partial[static_cast<size_t>(c)];
    }
    const uint64_t t_merge1 = now_ns();
    *merge_ms += static_cast<double>(t_merge1 - t_merge0) / 1e6;

    return chunk_cols;
}

LoopResult run_loop(
    const std::string& db_path,
    const std::vector<std::string>& keys,
    const std::vector<float>& activation,
    DType dtype,
    int64_t rows,
    int64_t initial_output_size,
    int prefetch_depth,
    size_t iters
) {
    if (keys.empty()) {
        throw std::runtime_error("run_loop requires non-empty keys");
    }
    const bool overlap = prefetch_depth > 0;

    std::vector<double> total_samples;
    total_samples.reserve(iters);
    double io_total_sum = 0.0;
    double io_exposed_sum = 0.0;
    double compute_sum = 0.0;
    double merge_sum = 0.0;
    uint64_t bytes_total = 0;
    size_t requests_total = 0;

    for (size_t iter = 0; iter < iters; ++iter) {
        IterationMetrics m;
        std::vector<float> output(static_cast<size_t>(initial_output_size), 0.0f);
        int64_t out_offset = 0;
        const uint64_t t_iter0 = now_ns();

        if (!overlap) {
            SimpleKVStore db = SimpleKVStore::OpenReadOnly(db_path);
            for (const auto& key : keys) {
                FetchResult fr = fetch_key(&db, key);
                if (!fr.ok) {
                    throw std::runtime_error("Failed to fetch key: " + key);
                }
                m.io_total_ms += fr.fetch_ms;
                m.io_exposed_ms += fr.fetch_ms;
                m.bytes += fr.value.size();
                m.requests += 1;
                out_offset += process_chunk(
                    fr.value, dtype, rows, out_offset, activation, output, &m.compute_ms, &m.merge_ms);
            }
        } else {
            SimpleKVStore db_main = SimpleKVStore::OpenReadOnly(db_path);
            SimpleKVStore db_prefetch = SimpleKVStore::OpenReadOnly(db_path);

            FetchResult current = fetch_key(&db_main, keys.front());
            if (!current.ok) {
                throw std::runtime_error("Failed to fetch first key");
            }
            m.io_total_ms += current.fetch_ms;
            m.io_exposed_ms += current.fetch_ms;
            m.bytes += current.value.size();
            m.requests += 1;

            std::future<FetchResult> next_future;
            bool future_valid = false;

            for (size_t idx = 0; idx < keys.size(); ++idx) {
                if (idx + 1 < keys.size()) {
                    const std::string next_key = keys[idx + 1];
                    next_future = std::async(std::launch::async, [&db_prefetch, next_key]() {
                        return fetch_key(&db_prefetch, next_key);
                    });
                    future_valid = true;
                } else {
                    future_valid = false;
                }

                out_offset += process_chunk(
                    current.value, dtype, rows, out_offset, activation, output, &m.compute_ms, &m.merge_ms);

                if (future_valid) {
                    const uint64_t t_wait0 = now_ns();
                    FetchResult next = next_future.get();
                    const uint64_t t_wait1 = now_ns();
                    const double wait_ms = static_cast<double>(t_wait1 - t_wait0) / 1e6;
                    m.io_exposed_ms += wait_ms;
                    m.io_total_ms += next.fetch_ms;
                    if (!next.ok) {
                        throw std::runtime_error("Failed to fetch prefetched key");
                    }
                    m.bytes += next.value.size();
                    m.requests += 1;
                    current = std::move(next);
                }
            }
        }

        const uint64_t t_iter1 = now_ns();
        m.total_ms = static_cast<double>(t_iter1 - t_iter0) / 1e6;

        total_samples.push_back(m.total_ms);
        io_total_sum += m.io_total_ms;
        io_exposed_sum += m.io_exposed_ms;
        compute_sum += m.compute_ms;
        merge_sum += m.merge_ms;
        bytes_total += m.bytes;
        requests_total += m.requests;
    }

    LoopResult out;
    out.total_latency = summarize_latencies_ms(total_samples);
    const double denom = (iters > 0) ? static_cast<double>(iters) : 1.0;
    out.io_total_ms_avg = io_total_sum / denom;
    out.io_exposed_ms_avg = io_exposed_sum / denom;
    out.compute_ms_avg = compute_sum / denom;
    out.merge_ms_avg = merge_sum / denom;
    out.runtime_overhead_ms_avg =
        std::max(0.0, out.total_latency.avg_ms - out.io_exposed_ms_avg - out.compute_ms_avg - out.merge_ms_avg);
    out.bytes = bytes_total;
    out.requests = requests_total;
    return out;
}

std::unordered_map<std::string, std::string> make_runtime_row(
    const std::string& config_id,
    DType dtype,
    int threads,
    int chunk_size_bytes,
    int chunk_size_param,
    int prefetch_depth,
    const LoopResult& r
) {
    const double mb = bytes_to_mb(static_cast<double>(r.bytes));
    const double throughput_mb_s = (r.total_latency.total_ms > 0.0)
        ? (mb / (r.total_latency.total_ms / 1000.0))
        : 0.0;
    const double denom = std::max(1.0, std::min(r.io_exposed_ms_avg, r.compute_ms_avg));
    const double omega_hint = std::max(0.0, std::min(1.0, (r.io_total_ms_avg - r.io_exposed_ms_avg) / denom));

    return {
        {"benchmark", "runtime"},
        {"config_id", config_id},
        {"workload", "fetch_compute_merge"},
        {"dtype", dtype_to_string(dtype)},
        {"thread_count", std::to_string(threads)},
        {"chunk_size_bytes", std::to_string(chunk_size_bytes)},
        {"chunk_size_param", std::to_string(chunk_size_param)},
        {"prefetch_depth", std::to_string(prefetch_depth)},
        {"request_count", std::to_string(r.requests)},
        {"bytes_transferred", std::to_string(r.bytes)},
        {"avg_ms", format_double(r.total_latency.avg_ms)},
        {"p50_ms", format_double(r.total_latency.p50_ms)},
        {"p95_ms", format_double(r.total_latency.p95_ms)},
        {"p99_ms", format_double(r.total_latency.p99_ms)},
        {"total_ms", format_double(r.total_latency.total_ms)},
        {"io_total_ms", format_double(r.io_total_ms_avg)},
        {"io_exposed_ms", format_double(r.io_exposed_ms_avg)},
        {"compute_ms", format_double(r.compute_ms_avg)},
        {"merge_ms", format_double(r.merge_ms_avg)},
        {"runtime_overhead_ms", format_double(r.runtime_overhead_ms_avg)},
        {"throughput_mb_s", format_double(throughput_mb_s)},
        {"omega_hint", format_double(omega_hint)}
    };
}

} // namespace

int main(int argc, char** argv) {
    try {
        math::set_backend(math::Backend::OneDNN);
        Config cfg = parse_args(argc, argv);
        const std::vector<std::string> csv_columns = {
            "benchmark",
            "config_id",
            "workload",
            "dtype",
            "thread_count",
            "chunk_size_bytes",
            "chunk_size_param",
            "prefetch_depth",
            "request_count",
            "bytes_transferred",
            "avg_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "total_ms",
            "io_total_ms",
            "io_exposed_ms",
            "compute_ms",
            "merge_ms",
            "runtime_overhead_ms",
            "throughput_mb_s",
            "omega_hint"
        };

        std::ofstream csv_out;
        if (!cfg.csv_path.empty()) {
            csv_out.open(cfg.csv_path, std::ios::trunc);
            if (!csv_out) {
                throw std::runtime_error("Failed to open CSV output: " + cfg.csv_path);
            }
            write_csv_header(csv_out, csv_columns);
        }

        std::vector<float> activation(static_cast<size_t>(cfg.rows), 0.0f);
        {
            std::mt19937 rng(777);
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            for (float& v : activation) {
                v = dist(rng);
            }
        }

        std::cout << "=== Runtime Microbenchmark ===\n"
                  << "backend=onednn\n"
                  << "dtype=" << dtype_to_string(cfg.dtype)
                  << " rows=" << cfg.rows
                  << " cols=" << cfg.cols
                  << " chunk_size_mode=element_count"
                  << " iters=" << cfg.iters
                  << " max_keys=" << cfg.max_keys
                  << std::endl;

        for (const int threads : cfg.thread_list) {
            configure_threads(threads);

            for (const int64_t chunk_size_param : cfg.chunk_size_list) {
                if (chunk_size_param <= 0) {
                    throw std::runtime_error("chunk-size-list values must be > 0");
                }

                MatrixSpec spec;
                spec.db_path = cfg.db_root + "/s" + std::to_string(chunk_size_param) + "_t" + std::to_string(threads);
                spec.matrix_id = cfg.matrix_id;
                spec.rows = cfg.rows;
                spec.cols = cfg.cols;
                spec.split_mode = cfg.split_mode;
                spec.chunk_size = chunk_size_param;
                spec.dtype = cfg.dtype;
                spec.seed = 9001u + static_cast<uint32_t>(chunk_size_param * 11 + threads);
                spec.truncate = true;

                DatasetInfo info = create_dummy_matrix_db(spec);
                std::vector<std::string> keys = info.keys;
                if (cfg.max_keys > 0 && keys.size() > cfg.max_keys) {
                    keys.resize(cfg.max_keys);
                }
                if (keys.empty()) {
                    continue;
                }

                SimpleKVStore inspect_db = SimpleKVStore::OpenReadOnly(spec.db_path);
                AlignedString first;
                if (!inspect_db.get_into(keys.front(), first)) {
                    throw std::runtime_error("Failed to read first key for runtime benchmark");
                }
                const int chunk_size_bytes = static_cast<int>(first.size());

                for (const int prefetch_depth : cfg.prefetch_depth_list) {
                    LoopResult lr = run_loop(
                        spec.db_path,
                        keys,
                        activation,
                        cfg.dtype,
                        cfg.rows,
                        cfg.cols,
                        prefetch_depth,
                        cfg.iters
                    );

                    const std::string cfg_id =
                        "runtime:s" + std::to_string(chunk_size_param) +
                        ":b" + std::to_string(chunk_size_bytes) +
                        ":d=" + dtype_to_string(cfg.dtype) +
                        ":t" + std::to_string(threads) +
                        ":p" + std::to_string(prefetch_depth);
                    auto row = make_runtime_row(
                        cfg_id,
                        cfg.dtype,
                        threads,
                        chunk_size_bytes,
                        static_cast<int>(chunk_size_param),
                        prefetch_depth,
                        lr
                    );

                    std::cout << row.at("config_id")
                              << " avg_ms=" << row.at("avg_ms")
                              << " io_exposed_ms=" << row.at("io_exposed_ms")
                              << " compute_ms=" << row.at("compute_ms")
                              << " runtime_overhead_ms=" << row.at("runtime_overhead_ms")
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

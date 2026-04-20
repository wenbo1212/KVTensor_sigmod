#include "common.hpp"

#include "kvtensor/types.hpp"
#include "math/arithmetic.hpp"

#include <algorithm>
#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
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
    throw std::runtime_error("Unsupported compute dtype. Use float32, bfloat16, or int8.");
}

struct Shape3D {
    int64_t m = 1;
    int64_t k = 1;
    int64_t n = 1;
};

std::vector<Shape3D> default_gemv_shapes() {
    return {
        {1, 4096, 64},
        {1, 4096, 128},
        {1, 4096, 256},
        {1, 4096, 512},
        {1, 4096, 1024},
        {1, 4096, 4096},
        {1, 4096, 6144},
        {1, 4096, 11008},
        {1, 4096, 28672},
        {1, 4096, 32000},
        {1, 14336, 64},
        {1, 14336, 128},
        {1, 14336, 256},
        {1, 14336, 512},
        {1, 14336, 4096},
    };
}

std::vector<Shape3D> default_gemm_shapes() {
    std::vector<Shape3D> out;
    const std::vector<int64_t> ms = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    const std::vector<int64_t> k4096_ns = {64, 128, 256, 512, 4096, 6144, 11008, 28672, 32000};
    const std::vector<int64_t> k14336_ns = {64, 128, 256, 512, 4096};
    for (int64_t m : ms) {
        for (int64_t n : k4096_ns) {
            out.push_back({m, 4096, n});
        }
        for (int64_t n : k14336_ns) {
            out.push_back({m, 14336, n});
        }
    }
    return out;
}

struct Config {
    std::vector<int> thread_list = {1, 4};
    std::vector<DType> dtypes = {DType::FLOAT32, DType::BFLOAT16};
    std::vector<Shape3D> gemv_shapes = default_gemv_shapes();
    std::vector<Shape3D> gemm_shapes = default_gemm_shapes();
    std::vector<int64_t> reduction_sizes = {4096, 11008, 16384};
    std::vector<int64_t> reduction_partials = {4, 8, 16};
    size_t iters = 20;
    size_t warmup = 4;
    std::string csv_path;
};

void print_help(const char* exe) {
    std::cout << "Usage: " << exe << " [options]\n"
              << "  --threads a,b,c                   Thread counts\n"
              << "  --dtypes float32,bfloat16,int8    DTypes to benchmark\n"
              << "  --gemv-shapes m1xk1xn1;...        GEMV shape list\n"
              << "  --gemm-shapes m1xk1xn1;...        GEMM shape list\n"
              << "  --reduction-sizes a,b,c           Output vector sizes\n"
              << "  --reduction-partials a,b,c        Number of partial vectors\n"
              << "  --iters N                         Timed iterations\n"
              << "  --warmup N                        Warmup iterations\n"
              << "  --csv PATH                        Optional CSV output\n"
              << "  --help\n";
}

Shape3D parse_shape_token(const std::string& token) {
    std::vector<int64_t> vals;
    std::string cur;
    for (char c : token) {
        if (c == 'x' || c == 'X') {
            if (!cur.empty()) {
                vals.push_back(std::stoll(cur));
                cur.clear();
            }
        } else if (!std::isspace(static_cast<unsigned char>(c))) {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        vals.push_back(std::stoll(cur));
    }
    if (vals.size() != 3) {
        throw std::runtime_error("Invalid shape token: " + token + " (expected m x k x n)");
    }
    return Shape3D{vals[0], vals[1], vals[2]};
}

std::vector<Shape3D> parse_shape_list(const std::string& raw) {
    std::vector<Shape3D> out;
    for (const auto& token : split(raw, ';')) {
        const std::string t = trim(token);
        if (t.empty()) {
            continue;
        }
        out.push_back(parse_shape_token(t));
    }
    return out;
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
        auto read_size = [&](size_t& out) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            out = static_cast<size_t>(std::stoull(argv[++i]));
        };

        if (arg == "--threads") {
            std::string raw;
            read_string(raw);
            cfg.thread_list = parse_int_list(raw);
        } else if (arg == "--dtypes") {
            std::string raw;
            read_string(raw);
            cfg.dtypes.clear();
            for (const auto& token : split(raw, ',')) {
                const std::string t = trim(token);
                if (t.empty()) {
                    continue;
                }
                const DType dtype = parse_supported_dtype(t);
                if (!is_supported_dtype(dtype)) {
                    throw std::runtime_error("Unsupported compute dtype. Use float32, bfloat16, or int8.");
                }
                cfg.dtypes.push_back(dtype);
            }
        } else if (arg == "--gemv-shapes") {
            std::string raw;
            read_string(raw);
            cfg.gemv_shapes = parse_shape_list(raw);
        } else if (arg == "--gemm-shapes") {
            std::string raw;
            read_string(raw);
            cfg.gemm_shapes = parse_shape_list(raw);
        } else if (arg == "--reduction-sizes") {
            std::string raw;
            read_string(raw);
            cfg.reduction_sizes = parse_i64_list(raw);
        } else if (arg == "--reduction-partials") {
            std::string raw;
            read_string(raw);
            cfg.reduction_partials = parse_i64_list(raw);
        } else if (arg == "--iters") {
            read_size(cfg.iters);
        } else if (arg == "--warmup") {
            read_size(cfg.warmup);
        } else if (arg == "--csv") {
            read_string(cfg.csv_path);
        } else if (arg == "--help") {
            print_help(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    for (DType dtype : cfg.dtypes) {
        if (!is_supported_dtype(dtype)) {
            throw std::runtime_error("Unsupported compute dtype. Use float32, bfloat16, or int8.");
        }
    }
    return cfg;
}

std::vector<float> make_random_f32(size_t n, uint32_t seed) {
    std::vector<float> out(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < n; ++i) {
        out[i] = dist(rng);
    }
    return out;
}

LatencyStats run_matmul_bench(
    DType dtype,
    int64_t m,
    int64_t k,
    int64_t n,
    size_t warmup,
    size_t iters
) {
    std::vector<float> A = make_random_f32(static_cast<size_t>(m * k), 1001);
    std::vector<float> B = make_random_f32(static_cast<size_t>(k * n), 2003);
    std::vector<float> C(static_cast<size_t>(m * n), 0.0f);
    std::vector<double> samples_ms;
    samples_ms.reserve(iters);

    if (dtype == DType::FLOAT32) {
        for (size_t i = 0; i < warmup; ++i) {
            math::matmul_ex<float>(A.data(), m, k, B.data(), n, C.data(), math::Transpose::No, math::Transpose::No);
        }
        for (size_t i = 0; i < iters; ++i) {
            const uint64_t t0 = now_ns();
            math::matmul_ex<float>(A.data(), m, k, B.data(), n, C.data(), math::Transpose::No, math::Transpose::No);
            const uint64_t t1 = now_ns();
            samples_ms.push_back(static_cast<double>(t1 - t0) / 1e6);
        }
        return summarize_latencies_ms(samples_ms);
    }

    if (dtype == DType::BFLOAT16) {
        std::vector<uint16_t> A_bf16(static_cast<size_t>(m * k), 0);
        std::vector<uint16_t> B_bf16(static_cast<size_t>(k * n), 0);
        math::convert_float32_to_bf16(A.data(), A.size(), A_bf16.data());
        math::convert_float32_to_bf16(B.data(), B.size(), B_bf16.data());

        for (size_t i = 0; i < warmup; ++i) {
            math::matmul_ex_bf16bf16f32(
                A_bf16.data(), m, k,
                B_bf16.data(), n,
                C.data(),
                math::Transpose::No,
                math::Transpose::No
            );
        }
        for (size_t i = 0; i < iters; ++i) {
            const uint64_t t0 = now_ns();
            math::matmul_ex_bf16bf16f32(
                A_bf16.data(), m, k,
                B_bf16.data(), n,
                C.data(),
                math::Transpose::No,
                math::Transpose::No
            );
            const uint64_t t1 = now_ns();
            samples_ms.push_back(static_cast<double>(t1 - t0) / 1e6);
        }
        return summarize_latencies_ms(samples_ms);
    }

    if (dtype == DType::INT8) {
        std::vector<float> C_f32(static_cast<size_t>(m * n), 0.0f);
        std::vector<int8_t> A_i8(static_cast<size_t>(m * k), 0);
        std::vector<int8_t> B_i8(static_cast<size_t>(k * n), 0);
        std::vector<uint8_t> B_u8(static_cast<size_t>(k * n), 0);

        const float scale_a = math::compute_quantization_scale(A.data(), A.size());
        const float scale_b = math::compute_quantization_scale(B.data(), B.size());
        math::convert_float32_to_int8(A.data(), A.size(), A_i8.data(), scale_a);
        math::convert_float32_to_int8(B.data(), B.size(), B_i8.data(), scale_b);
        math::convert_int8_to_uint8(B_i8.data(), B_i8.size(), B_u8.data());

        for (size_t i = 0; i < warmup; ++i) {
            math::matmul_ex_int8_int8_f32(
                A_i8.data(), m, k,
                B_u8.data(), n,
                C_f32.data(),
                math::Transpose::No,
                math::Transpose::No,
                scale_a,
                scale_b,
                1.0f
            );
        }
        for (size_t i = 0; i < iters; ++i) {
            const uint64_t t0 = now_ns();
            math::matmul_ex_int8_int8_f32(
                A_i8.data(), m, k,
                B_u8.data(), n,
                C_f32.data(),
                math::Transpose::No,
                math::Transpose::No,
                scale_a,
                scale_b,
                1.0f
            );
            const uint64_t t1 = now_ns();
            samples_ms.push_back(static_cast<double>(t1 - t0) / 1e6);
        }
        return summarize_latencies_ms(samples_ms);
    }

    throw std::runtime_error("Unsupported compute benchmark dtype: " + dtype_to_string(dtype));
}

LatencyStats run_reduction_merge_bench(
    int64_t output_elements,
    int64_t partials,
    size_t warmup,
    size_t iters
) {
    std::vector<std::vector<float>> partial_data;
    partial_data.reserve(static_cast<size_t>(partials));
    for (int64_t p = 0; p < partials; ++p) {
        partial_data.push_back(make_random_f32(static_cast<size_t>(output_elements), 3007u + static_cast<uint32_t>(p)));
    }
    std::vector<float> out(static_cast<size_t>(output_elements), 0.0f);

    auto run_once = [&]() {
        std::fill(out.begin(), out.end(), 0.0f);
        for (int64_t p = 0; p < partials; ++p) {
            const auto& src = partial_data[static_cast<size_t>(p)];
            for (int64_t i = 0; i < output_elements; ++i) {
                out[static_cast<size_t>(i)] += src[static_cast<size_t>(i)];
            }
        }
    };

    for (size_t i = 0; i < warmup; ++i) {
        run_once();
    }
    std::vector<double> samples_ms;
    samples_ms.reserve(iters);
    for (size_t i = 0; i < iters; ++i) {
        const uint64_t t0 = now_ns();
        run_once();
        const uint64_t t1 = now_ns();
        samples_ms.push_back(static_cast<double>(t1 - t0) / 1e6);
    }
    return summarize_latencies_ms(samples_ms);
}

std::unordered_map<std::string, std::string> make_compute_row(
    const std::string& config_id,
    const std::string& workload,
    DType dtype,
    int threads,
    const Shape3D& shape,
    const LatencyStats& stats,
    size_t iters
) {
    const double flops_per_iter =
        2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.k) * static_cast<double>(shape.n);
    const double total_flops = flops_per_iter * static_cast<double>(iters);
    const double gflops = (stats.total_ms > 0.0) ? (total_flops / 1e9) / (stats.total_ms / 1000.0) : 0.0;

    return {
        {"benchmark", "compute"},
        {"config_id", config_id},
        {"workload", workload},
        {"dtype", dtype_to_string(dtype)},
        {"thread_count", std::to_string(threads)},
        {"m", std::to_string(shape.m)},
        {"k", std::to_string(shape.k)},
        {"n", std::to_string(shape.n)},
        {"request_count", std::to_string(iters)},
        {"avg_ms", format_double(stats.avg_ms)},
        {"p50_ms", format_double(stats.p50_ms)},
        {"p95_ms", format_double(stats.p95_ms)},
        {"p99_ms", format_double(stats.p99_ms)},
        {"total_ms", format_double(stats.total_ms)},
        {"throughput_gflops", format_double(gflops)},
        {"output_elements", "0"},
        {"partials", "0"}
    };
}

std::unordered_map<std::string, std::string> make_reduction_row(
    const std::string& config_id,
    int threads,
    int64_t output_elements,
    int64_t partials,
    const LatencyStats& stats,
    size_t iters
) {
    const double bytes_per_iter = static_cast<double>(output_elements * (partials + 1)) * sizeof(float);
    const double throughput_mb_s = (stats.total_ms > 0.0)
        ? (bytes_to_mb(bytes_per_iter * static_cast<double>(iters)) / (stats.total_ms / 1000.0))
        : 0.0;

    return {
        {"benchmark", "compute"},
        {"config_id", config_id},
        {"workload", "reduction_merge"},
        {"dtype", "float32"},
        {"thread_count", std::to_string(threads)},
        {"m", "0"},
        {"k", "0"},
        {"n", "0"},
        {"request_count", std::to_string(iters)},
        {"avg_ms", format_double(stats.avg_ms)},
        {"p50_ms", format_double(stats.p50_ms)},
        {"p95_ms", format_double(stats.p95_ms)},
        {"p99_ms", format_double(stats.p99_ms)},
        {"total_ms", format_double(stats.total_ms)},
        {"throughput_gflops", "0.000000"},
        {"output_elements", std::to_string(output_elements)},
        {"partials", std::to_string(partials)},
        {"throughput_mb_s", format_double(throughput_mb_s)}
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
            "m",
            "k",
            "n",
            "request_count",
            "avg_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "total_ms",
            "throughput_gflops",
            "output_elements",
            "partials",
            "throughput_mb_s"
        };

        std::ofstream csv_out;
        if (!cfg.csv_path.empty()) {
            csv_out.open(cfg.csv_path, std::ios::trunc);
            if (!csv_out) {
                throw std::runtime_error("Failed to open CSV output: " + cfg.csv_path);
            }
            write_csv_header(csv_out, csv_columns);
        }

        std::cout << "=== Compute Microbenchmark ===\n"
                  << "backend=onednn\n"
                  << "iters=" << cfg.iters
                  << " warmup=" << cfg.warmup
                  << std::endl;

        for (const int threads : cfg.thread_list) {
            configure_threads(threads);

            for (const DType dtype : cfg.dtypes) {
                for (const auto& shape : cfg.gemv_shapes) {
                    LatencyStats stats = run_matmul_bench(dtype, shape.m, shape.k, shape.n, cfg.warmup, cfg.iters);
                    const std::string cfg_id =
                        "gemv:d=" + dtype_to_string(dtype) +
                        ":m" + std::to_string(shape.m) +
                        ":k" + std::to_string(shape.k) +
                        ":n" + std::to_string(shape.n) +
                        ":t" + std::to_string(threads);
                    auto row = make_compute_row(cfg_id, "gemv", dtype, threads, shape, stats, cfg.iters);
                    std::cout << row.at("config_id")
                              << " avg_ms=" << row.at("avg_ms")
                              << " gflops=" << row.at("throughput_gflops")
                              << std::endl;
                    if (csv_out) {
                        write_csv_row(csv_out, csv_columns, row);
                    }
                }

                for (const auto& shape : cfg.gemm_shapes) {
                    LatencyStats stats = run_matmul_bench(dtype, shape.m, shape.k, shape.n, cfg.warmup, cfg.iters);
                    const std::string cfg_id =
                        "gemm:d=" + dtype_to_string(dtype) +
                        ":m" + std::to_string(shape.m) +
                        ":k" + std::to_string(shape.k) +
                        ":n" + std::to_string(shape.n) +
                        ":t" + std::to_string(threads);
                    auto row = make_compute_row(cfg_id, "gemm", dtype, threads, shape, stats, cfg.iters);
                    std::cout << row.at("config_id")
                              << " avg_ms=" << row.at("avg_ms")
                              << " gflops=" << row.at("throughput_gflops")
                              << std::endl;
                    if (csv_out) {
                        write_csv_row(csv_out, csv_columns, row);
                    }
                }
            }

            for (const int64_t output_elements : cfg.reduction_sizes) {
                for (const int64_t partials : cfg.reduction_partials) {
                    LatencyStats stats = run_reduction_merge_bench(output_elements, partials, cfg.warmup, cfg.iters);
                    const std::string cfg_id =
                        "merge:out" + std::to_string(output_elements) +
                        ":p" + std::to_string(partials) +
                        ":t" + std::to_string(threads);
                    auto row = make_reduction_row(cfg_id, threads, output_elements, partials, stats, cfg.iters);
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

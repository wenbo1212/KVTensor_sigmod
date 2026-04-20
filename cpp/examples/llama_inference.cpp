#include "kvtensor/llama_inference.hpp"
#include "kvtensor/storage.hpp"
#include "kvtensor/context.hpp"
#include "kvtensor/matrix.hpp"
#include "kvtensor/preload_utils.hpp"
#include "math/arithmetic.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstring>

static std::vector<float> to_float32(const std::shared_ptr<kvtensor::InMemoryMatrix>& matrix) {
    auto [rows, cols] = matrix->shape();
    size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    std::vector<float> out(count);
    kvtensor::math::convert_buffer_to_float32(
        matrix->data().data(),
        matrix->dtype(),
        count,
        out.data()
    );
    return out;
}

static std::string json_escape(const std::string& value) {
    std::ostringstream oss;
    for (char ch : value) {
        switch (ch) {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    oss << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec << std::setfill(' ');
                } else {
                    oss << ch;
                }
                break;
        }
    }
    return oss.str();
}

static int64_t detect_thread_count() {
    if (const char* omp_threads = std::getenv("OMP_NUM_THREADS")) {
        try {
            return std::stoll(omp_threads);
        } catch (const std::exception&) {
        }
    }
    unsigned int hw_threads = std::thread::hardware_concurrency();
    return hw_threads > 0 ? static_cast<int64_t>(hw_threads) : 1;
}

static const char* dtype_name(kvtensor::DType dtype) {
    using kvtensor::DType;
    switch (dtype) {
        case DType::FLOAT32: return "float32";
        case DType::BFLOAT16: return "bfloat16";
        case DType::FLOAT16: return "float16";
        case DType::INT8: return "int8";
        default: return "unknown";
    }
}

static size_t dtype_size_bytes(kvtensor::DType dtype) {
    using kvtensor::DType;
    switch (dtype) {
        case DType::FLOAT32: return 4;
        case DType::BFLOAT16:
        case DType::FLOAT16: return 2;
        case DType::INT8: return 1;
        default: return 0;
    }
}

struct ReportMatrixInfo {
    std::string matrix_id;
    std::string matrix_group;
    std::string preload_group;
    std::string resident_execution;
    std::string streamed_execution;
    int priority_rank = -1;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t chunk_size = 0;
    std::string dtype;
    std::string split_mode;
    uint64_t size_bytes = 0;
};

static int preload_priority_rank(const std::string& matrix_id) {
    if (matrix_id == "output.output_proj") {
        return 4;
    }
    if (matrix_id.find(".ffn_gate_up_proj") != std::string::npos) {
        return 0;
    }
    if (matrix_id.find(".ffn_down_proj") != std::string::npos) {
        return 1;
    }
    if (matrix_id.find(".attn_qkv_proj") != std::string::npos) {
        return 2;
    }
    if (matrix_id.find(".attn_o_proj") != std::string::npos) {
        return 3;
    }
    return -1;
}

static std::string preload_group_name(const std::string& matrix_id) {
    switch (preload_priority_rank(matrix_id)) {
        case 0: return "ffn_gate_up_proj";
        case 1: return "ffn_down_proj";
        case 2: return "attn_qkv_proj";
        case 3: return "attn_o_proj";
        case 4: return "output_proj";
        default: return "non_preload";
    }
}

static std::string matrix_group_name(const std::string& matrix_id) {
    if (matrix_id == "embedding.token_embedding") {
        return "token_embedding";
    }
    if (matrix_id == "output.output_proj") {
        return "output_proj";
    }
    if (matrix_id.find(".attn_qkv_proj") != std::string::npos) {
        return "attn_qkv_proj";
    }
    if (matrix_id.find(".attn_o_proj") != std::string::npos) {
        return "attn_o_proj";
    }
    if (matrix_id.find(".ffn_gate_up_proj") != std::string::npos) {
        return "ffn_gate_up_proj";
    }
    if (matrix_id.find(".ffn_down_proj") != std::string::npos) {
        return "ffn_down_proj";
    }
    if (matrix_id.find("norm_weight") != std::string::npos) {
        return "norm_weight";
    }
    return "other";
}

static std::string resident_execution_mode(const std::string& matrix_id) {
    if (matrix_id.find(".ffn_gate_up_proj") != std::string::npos) {
        return "chunked";
    }
    return "dense";
}

static std::string streamed_execution_mode(const std::string& matrix_id) {
    if (preload_priority_rank(matrix_id) >= 0) {
        return "chunked";
    }
    return "none";
}

static void write_phase_trace_json(
    std::ostream& out,
    const kvtensor::LlamaPhaseTrace& trace,
    const std::string& indent
) {
    out << indent << "{\n";
    out << indent << "  \"elapsed_ms\": " << trace.elapsed_ms << ",\n";
    out << indent << "  \"prefetch_warmup_ms\": " << trace.prefetch_warmup_ms << ",\n";
    out << indent << "  \"compute_ms\": " << trace.compute_ms << ",\n";
    out << indent << "  \"other_compute_ms\": " << trace.other_compute_ms << ",\n";
    out << indent << "  \"kv_read_ms\": " << trace.kv_read_ms << ",\n";
    out << indent << "  \"decompress_ms\": " << trace.decompress_ms << ",\n";
    out << indent << "  \"overhead_ms\": " << trace.overhead_ms << ",\n";
    out << indent << "  \"bytes_read\": " << trace.bytes_read << ",\n";
    out << indent << "  \"gemm_flops\": " << trace.gemm_flops << ",\n";
    out << indent << "  \"bufferpool\": {\n";
    out << indent << "    \"get_chunk_calls\": " << trace.bufferpool_get_chunk_calls << ",\n";
    out << indent << "    \"cache_misses\": " << trace.bufferpool_cache_misses << ",\n";
    out << indent << "    \"wait_ms\": " << trace.bufferpool_wait_ms << ",\n";
    out << indent << "    \"memory_total_bytes\": " << trace.bufferpool_memory_total_bytes << "\n";
    out << indent << "  },\n";
    out << indent << "  \"matrix_accesses\": [\n";
    for (size_t i = 0; i < trace.matrix_accesses.size(); ++i) {
        const auto& access = trace.matrix_accesses[i];
        out << indent << "    {\n";
        out << indent << "      \"matrix_id\": \"" << json_escape(access.matrix_id) << "\",\n";
        out << indent << "      \"matrix_group\": \"" << json_escape(access.matrix_group) << "\",\n";
        out << indent << "      \"preload_group\": \"" << json_escape(access.preload_group) << "\",\n";
        out << indent << "      \"priority_rank\": " << access.priority_rank << ",\n";
        out << indent << "      \"rows\": " << access.rows << ",\n";
        out << indent << "      \"cols\": " << access.cols << ",\n";
        out << indent << "      \"chunk_size\": " << access.chunk_size << ",\n";
        out << indent << "      \"dtype\": \"" << json_escape(access.dtype) << "\",\n";
        out << indent << "      \"split_mode\": \"" << json_escape(access.split_mode) << "\",\n";
        out << indent << "      \"bytes_read\": " << access.bytes_read << ",\n";
        out << indent << "      \"chunk_reads\": " << access.chunk_reads << "\n";
        out << indent << "    }";
        if (i + 1 < trace.matrix_accesses.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << indent << "  ],\n";
    out << indent << "  \"gemm_buckets\": [\n";
    for (size_t i = 0; i < trace.gemm_buckets.size(); ++i) {
        const auto& bucket = trace.gemm_buckets[i];
        out << indent << "    {\n";
        out << indent << "      \"operator_class\": \"" << json_escape(bucket.operator_class) << "\",\n";
        out << indent << "      \"m\": " << bucket.m << ",\n";
        out << indent << "      \"k\": " << bucket.k << ",\n";
        out << indent << "      \"n\": " << bucket.n << ",\n";
        out << indent << "      \"calls\": " << bucket.calls << ",\n";
        out << indent << "      \"flops\": " << bucket.flops << "\n";
        out << indent << "    }";
        if (i + 1 < trace.gemm_buckets.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << indent << "  ]\n";
    out << indent << "}";
}

static void write_report_json(
    const std::string& report_path,
    const std::string& db_path,
    int64_t hidden_dim,
    int64_t num_layers,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t chunk_size,
    int64_t arena_size_mb,
    int64_t seq_len,
    int64_t decode_steps,
    int64_t thread_count,
    int64_t vocab_size,
    int64_t seed,
    const std::string& model_dtype,
    uint64_t streamable_weight_bytes,
    size_t streamable_matrix_count,
    int64_t prefetch_window,
    int64_t prefetch_latency_ms,
    bool prefetch_simulate,
    bool in_memory,
    bool disable_bufferpool,
    const std::string& preload_file,
    const std::vector<ReportMatrixInfo>& matrix_catalog,
    const std::vector<std::string>& preload_selected_matrix_ids,
    uint64_t preload_selected_bytes,
    size_t preload_selected_matrix_count,
    double preload_elapsed_s,
    double prefill_elapsed_s,
    const kvtensor::LlamaPhaseTrace& prefill_trace,
    const std::vector<double>& decode_times_s,
    const std::vector<kvtensor::LlamaPhaseTrace>& decode_traces,
    bool dump_first_token_logits,
    const std::vector<float>& first_token_logits,
    int64_t first_token_predicted_id,
    double first_token_max_logit
) {
    if (report_path.empty()) {
        return;
    }

    std::filesystem::path output_path(report_path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    double avg_decode_time_s = 0.0;
    for (double value : decode_times_s) {
        avg_decode_time_s += value;
    }
    if (!decode_times_s.empty()) {
        avg_decode_time_s /= static_cast<double>(decode_times_s.size());
    }
    double decode_ms_per_token = avg_decode_time_s * 1000.0;
    double decode_tokens_per_s = avg_decode_time_s > 0.0 ? (1.0 / avg_decode_time_s) : 0.0;

    std::ofstream out(report_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open report JSON path: " + report_path);
    }

    out << std::fixed << std::setprecision(9);
    out << "{\n";
    out << "  \"system\": {\n";
    out << "    \"db_path\": \"" << json_escape(db_path) << "\",\n";
    out << "    \"hidden_dim\": " << hidden_dim << ",\n";
    out << "    \"num_layers\": " << num_layers << ",\n";
    out << "    \"num_heads\": " << num_heads << ",\n";
    out << "    \"num_kv_heads\": " << num_kv_heads << ",\n";
    out << "    \"chunk_size\": " << chunk_size << ",\n";
    out << "    \"arena_size_mb\": " << arena_size_mb << ",\n";
    out << "    \"seq_len\": " << seq_len << ",\n";
    out << "    \"decode_steps\": " << decode_steps << ",\n";
    out << "    \"thread_count\": " << thread_count << ",\n";
    out << "    \"vocab_size\": " << vocab_size << ",\n";
    out << "    \"seed\": " << seed << ",\n";
    out << "    \"dtype\": \"" << json_escape(model_dtype) << "\",\n";
    out << "    \"streamable_weight_bytes\": " << streamable_weight_bytes << ",\n";
    out << "    \"streamable_matrix_count\": " << streamable_matrix_count << ",\n";
    out << "    \"prefetch_window\": " << prefetch_window << ",\n";
    out << "    \"prefetch_simulate\": " << (prefetch_simulate ? "true" : "false") << ",\n";
    out << "    \"prefetch_latency_ms\": " << prefetch_latency_ms << ",\n";
    out << "    \"in_memory\": " << (in_memory ? "true" : "false") << ",\n";
    out << "    \"disable_bufferpool\": " << (disable_bufferpool ? "true" : "false") << ",\n";
    out << "    \"preload_file\": \"" << json_escape(preload_file) << "\",\n";
    out << "    \"matrix_catalog\": [\n";
    for (size_t i = 0; i < matrix_catalog.size(); ++i) {
        const auto& matrix = matrix_catalog[i];
        out << "      {\n";
        out << "        \"matrix_id\": \"" << json_escape(matrix.matrix_id) << "\",\n";
        out << "        \"matrix_group\": \"" << json_escape(matrix.matrix_group) << "\",\n";
        out << "        \"preload_group\": \"" << json_escape(matrix.preload_group) << "\",\n";
        out << "        \"resident_execution\": \"" << json_escape(matrix.resident_execution) << "\",\n";
        out << "        \"streamed_execution\": \"" << json_escape(matrix.streamed_execution) << "\",\n";
        out << "        \"priority_rank\": " << matrix.priority_rank << ",\n";
        out << "        \"rows\": " << matrix.rows << ",\n";
        out << "        \"cols\": " << matrix.cols << ",\n";
        out << "        \"chunk_size\": " << matrix.chunk_size << ",\n";
        out << "        \"dtype\": \"" << json_escape(matrix.dtype) << "\",\n";
        out << "        \"split_mode\": \"" << json_escape(matrix.split_mode) << "\",\n";
        out << "        \"size_bytes\": " << matrix.size_bytes << "\n";
        out << "      }";
        if (i + 1 < matrix_catalog.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "    ]\n";
    out << "  },\n";
    out << "  \"preload\": {\n";
    out << "    \"selected_bytes\": " << preload_selected_bytes << ",\n";
    out << "    \"selected_matrix_count\": " << preload_selected_matrix_count << ",\n";
    out << "    \"selected_matrix_ids\": [";
    for (size_t i = 0; i < preload_selected_matrix_ids.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << "\"" << json_escape(preload_selected_matrix_ids[i]) << "\"";
    }
    out << "],\n";
    out << "    \"elapsed_s\": " << preload_elapsed_s << ",\n";
    out << "    \"throughput_mb_s\": "
        << ((preload_elapsed_s > 0.0 && preload_selected_bytes > 0)
            ? (static_cast<double>(preload_selected_bytes) / (1024.0 * 1024.0)) / preload_elapsed_s
            : 0.0)
        << "\n";
    out << "  },\n";
    out << "  \"prefill\": {\n";
    out << "    \"elapsed_s\": " << prefill_elapsed_s << ",\n";
    out << "    \"trace\": ";
    write_phase_trace_json(out, prefill_trace, "    ");
    out << "\n";
    out << "  },\n";
    out << "  \"decode\": {\n";
    out << "    \"token_times_s\": [";
    for (size_t i = 0; i < decode_times_s.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << decode_times_s[i];
    }
    out << "],\n";
    out << "    \"avg_time_s\": " << avg_decode_time_s << ",\n";
    out << "    \"ms_per_token\": " << decode_ms_per_token << ",\n";
    out << "    \"tokens_per_s\": " << decode_tokens_per_s << ",\n";
    out << "    \"first_generated_token\": {\n";
    out << "      \"logits_dumped\": " << (dump_first_token_logits ? "true" : "false") << ",\n";
    out << "      \"predicted_token_id\": " << first_token_predicted_id << ",\n";
    out << "      \"max_logit\": " << first_token_max_logit;
    if (dump_first_token_logits) {
        out << ",\n";
        out << "      \"logits\": [";
        for (size_t i = 0; i < first_token_logits.size(); ++i) {
            if (i > 0) {
                out << ", ";
            }
            out << first_token_logits[i];
        }
        out << "]\n";
    } else {
        out << "\n";
    }
    out << "    },\n";
    out << "    \"token_traces\": [\n";
    for (size_t i = 0; i < decode_traces.size(); ++i) {
        out << "      {\n";
        out << "        \"step\": " << i << ",\n";
        out << "        \"trace\": ";
        write_phase_trace_json(out, decode_traces[i], "        ");
        out << "\n";
        out << "      }";
        if (i + 1 < decode_traces.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "    ]\n";
    out << "  }\n";
    out << "}\n";

    if (!out) {
        throw std::runtime_error("Failed to write report JSON: " + report_path);
    }
}

// Simple argument parser
struct Args {
    std::string db_path = "./data/llama_db";
    int64_t hidden_dim = 2048;
    int64_t num_layers = 20;
    int64_t num_heads = 32;
    int64_t num_kv_heads = 0;  // 0 means auto: num_heads // 8
    int64_t seq_len = 64;
    int64_t chunk_size = 1024;
    int64_t decode_steps = 5;
    int64_t seed = 42;
    bool profile = false;
    int64_t vocab_size = 32000;
    int64_t arena_size_mb = 512;
    int64_t prefetch_window = 32;
    bool prefetch_simulate = false;
    int64_t prefetch_latency_ms = 0;
    std::string preload_file_path = "";
    std::string report_json_path = "";
    bool preload_only = false;
    bool dump_first_token_logits = false;
    bool in_memory = false;
    bool disable_bufferpool = false;
    
    static Args parse(int argc, char* argv[]) {
        Args args;
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "--db-path" && i + 1 < argc) {
                args.db_path = argv[++i];
            } else if (arg == "--hidden-dim" && i + 1 < argc) {
                args.hidden_dim = std::stoll(argv[++i]);
            } else if (arg == "--num-layers" && i + 1 < argc) {
                args.num_layers = std::stoll(argv[++i]);
            } else if (arg == "--num-heads" && i + 1 < argc) {
                args.num_heads = std::stoll(argv[++i]);
            } else if (arg == "--num-kv-heads" && i + 1 < argc) {
                args.num_kv_heads = std::stoll(argv[++i]);
            } else if (arg == "--seq-len" && i + 1 < argc) {
                args.seq_len = std::stoll(argv[++i]);
            } else if (arg == "--chunk-size" && i + 1 < argc) {
                args.chunk_size = std::stoll(argv[++i]);
            } else if (arg == "--decode-steps" && i + 1 < argc) {
                args.decode_steps = std::stoll(argv[++i]);
            } else if (arg == "--seed" && i + 1 < argc) {
                args.seed = std::stoll(argv[++i]);
            } else if (arg == "--profile") {
                args.profile = true;
            } else if (arg == "--vocab-size" && i + 1 < argc) {
                args.vocab_size = std::stoll(argv[++i]);
            } else if (arg == "--arena-size-mb" && i + 1 < argc) {
                args.arena_size_mb = std::stoll(argv[++i]);
            } else if (arg == "--prefetch-window" && i + 1 < argc) {
                args.prefetch_window = std::stoll(argv[++i]);
            } else if (arg == "--prefetch-simulate") {
                args.prefetch_simulate = true;
            } else if (arg == "--prefetch-latency-ms" && i + 1 < argc) {
                args.prefetch_latency_ms = std::stoll(argv[++i]);
            } else if (arg == "--preload-file" && i + 1 < argc) {
                args.preload_file_path = argv[++i];
            } else if (arg == "--report-json" && i + 1 < argc) {
                args.report_json_path = argv[++i];
            } else if (arg == "--preload-only") {
                args.preload_only = true;
            } else if (arg == "--dump-first-token-logits") {
                args.dump_first_token_logits = true;
            } else if (arg == "--in-memory") {
                args.in_memory = true;
            } else if (arg == "--disable-bufferpool") {
                args.disable_bufferpool = true;
            } else if (arg == "--help" || arg == "-h") {
                print_help();
                std::exit(0);
            } else if (i == 1 && arg[0] != '-') {
                // First positional argument is db_path
                args.db_path = arg;
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                print_help();
                std::exit(1);
            }
        }
        
        // Auto-set num_kv_heads if not specified
        if (args.num_kv_heads == 0) {
            args.num_kv_heads = args.num_heads / 8;
            if (args.num_kv_heads == 0) {
                args.num_kv_heads = 1;
            }
        }
        
        return args;
    }
    
    static void print_help() {
        std::cout << "Usage: llama_inference_example [OPTIONS] [DB_PATH]\n\n";
        std::cout << "Options:\n";
        std::cout << "  --db-path PATH          LevelDB database path (default: ./data/llama_db)\n";
        std::cout << "  --hidden-dim N          Hidden dimension (default: 2048)\n";
        std::cout << "  --num-layers N          Number of transformer layers (default: 20)\n";
        std::cout << "  --num-heads N           Number of query heads (default: 32)\n";
        std::cout << "  --num-kv-heads N        Number of KV heads (default: num_heads // 8)\n";
        std::cout << "  --seq-len N             Sequence length (default: 64)\n";
        std::cout << "  --chunk-size N          Chunk size for row/column splits (default: 1024)\n";
        std::cout << "  --decode-steps N        Number of decode steps (default: 5)\n";
        std::cout << "  --seed N                Random seed (default: 42)\n";
        std::cout << "  --vocab-size N          Vocabulary size (default: 32000)\n";
        std::cout << "  --arena-size-mb N        Buffer pool memory limit in MB (default: 512)\n";
        std::cout << "  --prefetch-window N      Prefetch window size (default: 32)\n";
        std::cout << "  --prefetch-simulate      Enable simulated prefetch (skip DB reads)\n";
        std::cout << "  --prefetch-latency-ms N  Simulated get_into latency in ms (default: 0)\n";
        std::cout << "  --preload-file PATH      Path to txt file with matrix IDs to preload (one per line)\n";
        std::cout << "  --report-json PATH       Write machine-readable timing report to PATH\n";
        std::cout << "  --preload-only           Only measure preload-file materialization and exit\n";
        std::cout << "  --dump-first-token-logits  Include full logits for decode step 0 in report JSON\n";
        std::cout << "  --in-memory             Densify all weights into memory and disable bufferpool\n";
        std::cout << "  --disable-bufferpool    Read chunks directly from DB and disable prefetching\n";
        std::cout << "  --profile               Enable detailed profiling\n";
        std::cout << "  --help, -h              Show this help message\n";
    }
};

int main(int argc, char* argv[]) {
    try {
        using namespace kvtensor;

        // Parse arguments
        Args args = Args::parse(argc, argv);
        
        // Validate arguments
        if (args.hidden_dim % args.num_heads != 0) {
            std::cerr << "Error: hidden_dim (" << args.hidden_dim 
                      << ") must be divisible by num_heads (" << args.num_heads << ")" << std::endl;
            return 1;
        }
        
        if (args.num_heads % args.num_kv_heads != 0) {
            std::cerr << "Error: num_heads (" << args.num_heads 
                      << ") must be divisible by num_kv_heads (" << args.num_kv_heads << ")" << std::endl;
            return 1;
        }
        
        int64_t head_dim = args.hidden_dim / args.num_heads;
        
        // Print configuration
        std::cout << "=" << std::string(60, '=') << std::endl;
        std::cout << "Llama 3.1 Forward Pass (C++ Implementation)" << std::endl;
        std::cout << "=" << std::string(60, '=') << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Database path: " << args.db_path << std::endl;
        std::cout << "  Hidden dim: " << args.hidden_dim << std::endl;
        std::cout << "  Num layers: " << args.num_layers << std::endl;
        std::cout << "  Num query heads: " << args.num_heads << std::endl;
        std::cout << "  Num KV heads: " << args.num_kv_heads << std::endl;
        std::cout << "  Head dim: " << head_dim << std::endl;
        std::cout << "  Sequence length: " << args.seq_len << std::endl;
        std::cout << "  Chunk size: " << args.chunk_size << std::endl;
        std::cout << "  Vocab size: " << args.vocab_size << std::endl;
        std::cout << "  Decode steps: " << args.decode_steps << std::endl;
        std::cout << "  Buffer pool memory limit (MB): " << args.arena_size_mb << std::endl;
        std::cout << "  Prefetch window: " << args.prefetch_window << std::endl;
        std::cout << "  Prefetch simulate: " << (args.prefetch_simulate ? "true" : "false") << std::endl;
        if (args.prefetch_simulate) {
            std::cout << "  Prefetch latency (ms): " << args.prefetch_latency_ms << std::endl;
        }
        if (!args.preload_file_path.empty()) {
            std::cout << "  Preload file: " << args.preload_file_path << std::endl;
        }
        if (!args.report_json_path.empty()) {
            std::cout << "  Report JSON: " << args.report_json_path << std::endl;
        }
        std::cout << "  Preload only: " << (args.preload_only ? "true" : "false") << std::endl;
        std::cout << "  Dump first-token logits: " << (args.dump_first_token_logits ? "true" : "false") << std::endl;
        std::cout << "  In-memory mode: " << (args.in_memory ? "true" : "false") << std::endl;
        std::cout << "  Disable bufferpool: " << (args.disable_bufferpool ? "true" : "false") << std::endl;
        std::cout << std::endl;
        
        // Initialize storage
        auto storage = std::make_unique<SimpleDBStorage>(args.db_path, true /*read_only*/);
        auto registry = std::make_unique<MatrixRegistry>(storage.get());
        auto ctx = std::make_unique<OperatorContext>(storage.get(), registry.get());

        auto resolve_matrix_bytes = [&](const std::string& matrix_id) -> uint64_t {
            auto matrix = registry->get_matrix(matrix_id);
            auto [rows, cols] = matrix->shape();
            return static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols) * dtype_size_bytes(matrix->dtype());
        };
        auto make_matrix_info = [&](const std::string& matrix_id) -> ReportMatrixInfo {
            auto matrix = registry->get_matrix(matrix_id);
            auto [rows, cols] = matrix->shape();
            ReportMatrixInfo info;
            info.matrix_id = matrix_id;
            info.matrix_group = matrix_group_name(matrix_id);
            info.preload_group = preload_group_name(matrix_id);
            info.resident_execution = resident_execution_mode(matrix_id);
            info.streamed_execution = streamed_execution_mode(matrix_id);
            info.priority_rank = preload_priority_rank(matrix_id);
            info.rows = rows;
            info.cols = cols;
            info.chunk_size = matrix->chunk_size();
            info.dtype = dtype_name(matrix->dtype());
            info.split_mode = matrix->split_mode() == SplitMode::COLUMN ? "column" : "row";
            info.size_bytes = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols) * dtype_size_bytes(matrix->dtype());
            return info;
        };
        
        // Check if token embedding exists, if not generate random input embeddings
        std::string token_embedding_id = "embedding.token_embedding";
        std::string input_embeddings_id = "input_embeddings";
        bool use_random_embeddings = false;
        
        try {
            auto test_embedding = registry->get_matrix(token_embedding_id);
            std::cout << "  Token embedding found in database" << std::endl;
        } catch (const std::exception&) {
            // Token embedding not found - generate random input embeddings directly
            std::cout << "  Token embedding not found, generating random input embeddings..." << std::endl;
            
            std::mt19937 rng(static_cast<unsigned int>(args.seed));
            std::normal_distribution<float> dist(0.0f, 0.02f);
            
            // Generate random embeddings: (seq_len, hidden_dim) - this is the input for inference
            size_t embedding_size = args.seq_len * args.hidden_dim;
            std::vector<float> embedding_data(embedding_size);
            for (size_t i = 0; i < embedding_size; ++i) {
                embedding_data[i] = dist(rng);
            }
            
            // Convert to bytes and store in memory
            std::vector<uint8_t> embedding_bytes(embedding_size * sizeof(float));
            std::memcpy(embedding_bytes.data(), embedding_data.data(), embedding_bytes.size());
            
            ctx->store_in_memory(
                input_embeddings_id,
                std::make_tuple(args.seq_len, args.hidden_dim),
                DType::FLOAT32,
                std::move(embedding_bytes)
            );
            
            use_random_embeddings = true;
            std::cout << "  Generated random input embeddings: shape (" << args.seq_len 
                      << ", " << args.hidden_dim << ")" << std::endl;
        }
        
        // Preload normalization weights into memory (generate if missing)
        std::cout << "\nPreloading normalization weights..." << std::endl;
        std::vector<std::string> norm_weight_ids;
        for (int64_t i = 0; i < args.num_layers; ++i) {
            norm_weight_ids.push_back("transformer." + std::to_string(i) + ".attn_norm_weight");
            norm_weight_ids.push_back("transformer." + std::to_string(i) + ".ffn_norm_weight");
        }
        norm_weight_ids.push_back("output.output_norm_weight");
        
        std::mt19937 norm_rng(static_cast<unsigned int>(args.seed + 1000));
        std::normal_distribution<float> norm_dist(1.0f, 0.01f);  // Around 1.0 like Python example
        
        for (const auto& weight_id : norm_weight_ids) {
            try {
                // Try to get from database first
                auto weight_block = registry->get_matrix(weight_id);
                
                // Load from database
                auto weight_dense = weight_block->to_dense(*ctx);
                auto [weight_rows, weight_cols] = weight_block->shape();
                
                // Normalize to 1D vector (handle different shapes)
                std::vector<float> weight_vec;
                if (weight_rows == 1 && weight_cols == args.hidden_dim) {
                    const float* ptr = reinterpret_cast<const float*>(weight_dense.data());
                    weight_vec.assign(ptr, ptr + args.hidden_dim);
                } else if (weight_rows == args.hidden_dim && weight_cols == 1) {
                    const float* ptr = reinterpret_cast<const float*>(weight_dense.data());
                    weight_vec.assign(ptr, ptr + args.hidden_dim);
                } else {
                    // Unexpected shape, generate random
                    std::cout << "    Warning: Unexpected shape for " << weight_id 
                              << " (" << weight_rows << ", " << weight_cols 
                              << "), generating random weight" << std::endl;
                    weight_vec.resize(args.hidden_dim);
                    for (int64_t i = 0; i < args.hidden_dim; ++i) {
                        weight_vec[i] = norm_dist(norm_rng);
                    }
                }
                
                // Check if weight is all zeros (invalid weight)
                float weight_sum = 0.0f;
                for (float w : weight_vec) {
                    weight_sum += std::abs(w);
                }
                if (weight_sum < 1e-6f) {
                    std::cout << "    Warning: Weight " << weight_id << " is all zeros, generating random weight" << std::endl;
                    weight_vec.resize(args.hidden_dim);
                    for (int64_t i = 0; i < args.hidden_dim; ++i) {
                        weight_vec[i] = norm_dist(norm_rng);
                    }
                }
                
                // Store in memory
                std::vector<uint8_t> weight_bytes(weight_vec.size() * sizeof(float));
                std::memcpy(weight_bytes.data(), weight_vec.data(), weight_bytes.size());
                auto weight_matrix = ctx->store_in_memory(
                    weight_id + "_norm",
                    std::make_tuple(1, args.hidden_dim),
                    DType::FLOAT32,
                    std::move(weight_bytes)
                );
                ctx->store_norm_weight(weight_id, weight_matrix->shape(), weight_matrix->dtype(), weight_matrix->data());
            } catch (const std::exception&) {
                // Generate random normalization weight if not found in database
                std::cout << "    Generating random normalization weight: " << weight_id << std::endl;
                std::vector<float> weight_vec(args.hidden_dim);
                for (int64_t i = 0; i < args.hidden_dim; ++i) {
                    weight_vec[i] = norm_dist(norm_rng);
                }
                
                // Store in memory
                std::vector<uint8_t> weight_bytes(weight_vec.size() * sizeof(float));
                std::memcpy(weight_bytes.data(), weight_vec.data(), weight_bytes.size());
                auto weight_matrix = ctx->store_in_memory(
                    weight_id + "_norm",
                    std::make_tuple(1, args.hidden_dim),
                    DType::FLOAT32,
                    std::move(weight_bytes)
                );
                ctx->store_norm_weight(weight_id, weight_matrix->shape(), weight_matrix->dtype(), weight_matrix->data());
            }
        }
        std::cout << "  Preloaded " << norm_weight_ids.size() << " normalization weights" << std::endl;
        
        // Output projection is a large weight matrix; keep it chunked in DB (avoid full materialization).
        std::string output_proj_id = "output.output_proj";
        auto existing_output_proj = ctx->get_in_memory(output_proj_id);
        bool output_proj_in_db = false;
        if (!existing_output_proj) {
            try {
                auto existing_block = ctx->resolve_block_matrix(output_proj_id);
                (void)existing_block;
                output_proj_in_db = true;
            } catch (const std::exception&) {
                output_proj_in_db = false;
            }
        }

        if (existing_output_proj) {
            std::cout << "  Output projection matrix already in memory" << std::endl;
        } else if (output_proj_in_db) {
            std::cout << "  Output projection matrix already in DB (chunked)" << std::endl;
        } else {
            std::cout << "  Generating output projection matrix in DB (chunked)..." << std::endl;

            std::mt19937 proj_rng(static_cast<unsigned int>(args.seed + 2000));
            std::normal_distribution<float> proj_dist(0.0f, 0.02f);  // std=0.02 like Python

            Shape proj_shape = std::make_tuple(args.hidden_dim, args.vocab_size);
            BlockMatrix output_proj(
                output_proj_id,
                proj_shape,
                ctx->storage(),
                DType::FLOAT32,
                SplitMode::COLUMN,
                args.chunk_size
            );

            int64_t num_chunks = output_proj.num_col_chunks();
            int64_t rows = args.hidden_dim;
            for (int64_t j = 0; j < num_chunks; ++j) {
                auto [chunk_rows, chunk_cols] = output_proj.col_chunk_shape(j);
                (void)chunk_rows; // chunk_rows == rows

                std::vector<float> chunk_data(rows * chunk_cols);
                for (int64_t r = 0; r < rows; ++r) {
                    for (int64_t c = 0; c < chunk_cols; ++c) {
                        chunk_data[r * chunk_cols + c] = proj_dist(proj_rng);
                    }
                }

                std::vector<uint8_t> chunk_bytes(chunk_data.size() * sizeof(float));
                std::memcpy(chunk_bytes.data(), chunk_data.data(), chunk_bytes.size());
                output_proj.write_col_chunk(j, chunk_bytes.data(), chunk_bytes.size());
            }

            ctx->registry()->save_metadata(
                output_proj_id,
                proj_shape,
                DType::FLOAT32,
                SplitMode::COLUMN,
                args.chunk_size
            );

            std::cout << "  Generated output projection matrix in DB: shape (" << args.hidden_dim
                      << ", " << args.vocab_size << "), chunk_size=" << args.chunk_size << std::endl;
        }
        
        // Build model configuration
        LlamaModelConfig model_config;
        model_config.vocab_size = args.vocab_size;
        model_config.hidden_dim = args.hidden_dim;
        model_config.chunk_size = args.chunk_size;
        
        // Token embedding
        model_config.token_embedding_id = "embedding.token_embedding";
        
        // Output layer
        model_config.output_norm_weight_id = "output.output_norm_weight";
        model_config.output_proj_id = "output.output_proj";
        
        // Transformer blocks
        for (int64_t i = 0; i < args.num_layers; ++i) {
            LlamaTransformerBlockConfig block_config;
            std::string prefix = "transformer." + std::to_string(i);
            
            block_config.attn_qkv_proj_id = prefix + ".attn_qkv_proj";
            block_config.attn_o_proj_id = prefix + ".attn_o_proj";
            block_config.ffn_gate_up_proj_id = prefix + ".ffn_gate_up_proj";
            block_config.ffn_down_proj_id = prefix + ".ffn_down_proj";
            block_config.attn_norm_weight_id = prefix + ".attn_norm_weight";
            block_config.ffn_norm_weight_id = prefix + ".ffn_norm_weight";
            block_config.num_heads = args.num_heads;
            block_config.num_kv_heads = args.num_kv_heads;
            block_config.head_dim = head_dim;
            block_config.hidden_dim = args.hidden_dim;
            block_config.chunk_size = args.chunk_size;
            block_config.rms_norm_eps = 1e-6f;
            
            model_config.blocks.push_back(block_config);
        }
        
        // Build inference wrapper (handles preload + prefetch orchestration)
        LlamaInferenceConfig inference_config;
        inference_config.model = model_config;
        inference_config.prefetch.arena_size_mb = static_cast<size_t>(args.arena_size_mb);
        inference_config.prefetch.prefetch_window = static_cast<size_t>(args.prefetch_window);
        inference_config.prefetch.ring = false;
        inference_config.prefetch.simulate_prefetch = args.prefetch_simulate;
        inference_config.prefetch.simulate_get_latency_ms = static_cast<uint64_t>(args.prefetch_latency_ms);
        inference_config.preload_file_path = args.preload_file_path;
        inference_config.in_memory = args.in_memory;
        inference_config.disable_bufferpool = args.disable_bufferpool;

        LlamaInference inference(inference_config);

        std::string model_dtype = "unknown";
        uint64_t streamable_weight_bytes = 0;
        size_t streamable_matrix_count = 0;
        for (const auto& matrix_id : inference.prefetch_matrix_ids()) {
            try {
                auto matrix = registry->get_matrix(matrix_id);
                if (model_dtype == "unknown") {
                    model_dtype = dtype_name(matrix->dtype());
                }
                streamable_weight_bytes += resolve_matrix_bytes(matrix_id);
                ++streamable_matrix_count;
            } catch (const std::exception&) {
            }
        }
        uint64_t preload_selected_bytes = 0;
        size_t preload_selected_matrix_count = 0;
        std::vector<std::string> preload_selected_matrix_ids;
        if (!args.preload_file_path.empty()) {
            auto preload_ids = read_preload_matrix_ids(args.preload_file_path);
            preload_selected_matrix_count = preload_ids.size();
            preload_selected_matrix_ids.assign(preload_ids.begin(), preload_ids.end());
            std::sort(preload_selected_matrix_ids.begin(), preload_selected_matrix_ids.end());
            for (const auto& matrix_id : preload_ids) {
                try {
                    preload_selected_bytes += resolve_matrix_bytes(matrix_id);
                } catch (const std::exception&) {
                }
            }
        }
        auto build_matrix_catalog = [&](const kvtensor::LlamaPhaseTrace& prefill_trace,
                                        const std::vector<kvtensor::LlamaPhaseTrace>& decode_traces) {
            std::unordered_set<std::string> ids;
            for (const auto& matrix_id : inference.prefetch_matrix_ids()) {
                ids.insert(matrix_id);
            }
            for (const auto& matrix_id : preload_selected_matrix_ids) {
                ids.insert(matrix_id);
            }
            for (const auto& access : prefill_trace.matrix_accesses) {
                ids.insert(access.matrix_id);
            }
            for (const auto& trace : decode_traces) {
                for (const auto& access : trace.matrix_accesses) {
                    ids.insert(access.matrix_id);
                }
            }

            std::vector<ReportMatrixInfo> catalog;
            catalog.reserve(ids.size());
            for (const auto& matrix_id : ids) {
                try {
                    catalog.push_back(make_matrix_info(matrix_id));
                } catch (const std::exception&) {
                }
            }
            std::sort(catalog.begin(), catalog.end(), [](const auto& a, const auto& b) {
                if (a.priority_rank != b.priority_rank) {
                    return a.priority_rank < b.priority_rank;
                }
                return a.matrix_id < b.matrix_id;
            });
            return catalog;
        };

        double preload_elapsed = 0.0;
        if (args.preload_only) {
            std::cout << "=" << std::string(60, '=') << std::endl;
            std::cout << "Preload Only Mode" << std::endl;
            std::cout << "=" << std::string(60, '=') << std::endl;
            preload_elapsed = inference.preload_static_weights(*ctx);
            std::cout << "  Preload time: " << preload_elapsed << "s" << std::endl;
            const auto matrix_catalog = build_matrix_catalog(kvtensor::LlamaPhaseTrace{}, {});

            write_report_json(
                args.report_json_path,
                args.db_path,
                args.hidden_dim,
                args.num_layers,
                args.num_heads,
                args.num_kv_heads,
                args.chunk_size,
                args.arena_size_mb,
                args.seq_len,
                args.decode_steps,
                detect_thread_count(),
                args.vocab_size,
                args.seed,
                model_dtype,
                streamable_weight_bytes,
                streamable_matrix_count,
                args.prefetch_window,
                args.prefetch_latency_ms,
                args.prefetch_simulate,
                args.in_memory,
                args.disable_bufferpool,
                args.preload_file_path,
                matrix_catalog,
                preload_selected_matrix_ids,
                preload_selected_bytes,
                preload_selected_matrix_count,
                preload_elapsed,
                0.0,
                kvtensor::LlamaPhaseTrace{},
                {},
                {},
                args.dump_first_token_logits,
                {},
                -1,
                0.0
            );
            return 0;
        }
        
        // Prefill phase: process all tokens
        std::cout << "=" << std::string(60, '=') << std::endl;
        std::cout << "Prefill Phase (processing " << args.seq_len << " tokens)" << std::endl;
        std::cout << "=" << std::string(60, '=') << std::endl;
        
        // Generate input token IDs (for demonstration)
        std::vector<int32_t> input_ids;
        input_ids.reserve(args.seq_len);
        // Simple deterministic sequence based on seed
        for (int64_t i = 0; i < args.seq_len; ++i) {
            input_ids.push_back((i + args.seed) % args.vocab_size);
        }
        
        // Prepare KV cache output IDs for prefill
        std::vector<std::pair<std::string, std::string>> prefill_cache_output_ids;
        for (int64_t i = 0; i < args.num_layers; ++i) {
            std::string k_id = "kv_cache_layer_" + std::to_string(i) + "_k_prefill";
            std::string v_id = "kv_cache_layer_" + std::to_string(i) + "_v_prefill";
            prefill_cache_output_ids.emplace_back(k_id, v_id);
        }
        
        auto prefill_start = std::chrono::high_resolution_clock::now();
        std::string prefill_result_id = "prefill_output";
        auto prefill_result = inference.forward(
            input_ids,
            prefill_result_id,
            *ctx,
            "",  // mask_id (empty for now, can add causal mask later)
            {},  // kv_cache_ids (empty for prefill)
            prefill_cache_output_ids,
            use_random_embeddings ? input_embeddings_id : "",  // Use pre-generated embeddings if available
            args.profile  // Enable profiling if requested
        );
        auto prefill_end = std::chrono::high_resolution_clock::now();
        double prefill_elapsed = std::chrono::duration<double>(prefill_end - prefill_start).count();
        prefill_elapsed -= inference.last_prefetch_warmup_ms() / 1000.0;
        if (prefill_elapsed < 0.0) {
            prefill_elapsed = 0.0;
        }
        auto prefill_trace = inference.last_phase_trace();
        
        auto [prefill_rows, prefill_cols] = prefill_result->shape();
        std::cout << "\nPrefill complete!" << std::endl;
        std::cout << "  Output shape: (" << prefill_rows << ", " << prefill_cols << ")" << std::endl;
        std::cout << "  Time: " << prefill_elapsed << "s" << std::endl;
        
        // Get predicted token (argmax of last token's logits)
        // Output should be (1, vocab_size) - only last token's logits
        std::vector<float> prefill_f32 = to_float32(prefill_result);
        const float* prefill_ptr = prefill_f32.data();
        
        // Find argmax - iterate over all vocab_size elements
        int32_t predicted_token = 0;
        float max_logit = prefill_ptr[0];
        for (int64_t i = 1; i < prefill_cols; ++i) {
            if (prefill_ptr[i] > max_logit) {
                max_logit = prefill_ptr[i];
                predicted_token = static_cast<int32_t>(i);
            }
        }
        std::cout << "  Predicted next token ID: " << predicted_token 
                  << " (max_logit: " << max_logit << ")" << std::endl;
        
        // Decode phase: generate tokens one at a time
        std::cout << "\n" << "=" << std::string(60, '=') << std::endl;
        std::cout << "Decode Phase (generating " << args.decode_steps << " tokens)" << std::endl;
        std::cout << "=" << std::string(60, '=') << std::endl;
        
        // Use prefill cache as input for decode
        std::vector<std::pair<std::string, std::string>> kv_cache_ids = prefill_cache_output_ids;
        
        // The last token's hidden state from prefill is stored as "prefill_output_last_token"
        // This is the output_norm result (before final projection) - use it for first decode step
        std::string prefill_last_token_id = prefill_result_id + "_last_token";
        
        std::vector<double> decode_times;
        std::vector<kvtensor::LlamaPhaseTrace> decode_traces;
        std::vector<float> first_token_logits;
        int64_t first_token_predicted_id = -1;
        double first_token_max_logit = 0.0;
        for (int64_t step = 0; step < args.decode_steps; ++step) {
            // Prepare cache output IDs for this step
            std::vector<std::pair<std::string, std::string>> cache_output_ids;
            for (int64_t i = 0; i < args.num_layers; ++i) {
                std::string k_id = "kv_cache_layer_" + std::to_string(i) + "_k_step" + std::to_string(step);
                std::string v_id = "kv_cache_layer_" + std::to_string(i) + "_v_step" + std::to_string(step);
                cache_output_ids.emplace_back(k_id, v_id);
            }
            
            
            // For decode step 0: use last token's hidden state from prefill
            // For decode step 1+: use predicted token's embedding
            std::string decode_embeddings_id = "";
            std::vector<int32_t> decode_input;
            
            if (step == 0) {
                // First decode step: use prefill's last token hidden state
                auto prefill_last_token = ctx->get_in_memory(prefill_last_token_id);
                if (prefill_last_token) {
                    decode_embeddings_id = prefill_last_token_id;
                    decode_input = {predicted_token};  // Still need input_ids for model signature, but won't be used
                } else {
                    // Fallback: generate random embedding
                    std::mt19937 rng(static_cast<unsigned int>(args.seed + step + 1000));
                    std::normal_distribution<float> dist(0.0f, 0.02f);
                    
                    std::vector<float> decode_embedding_data(args.hidden_dim);
                    for (int64_t j = 0; j < args.hidden_dim; ++j) {
                        decode_embedding_data[j] = dist(rng);
                    }
                    
                    std::vector<uint8_t> decode_embedding_bytes(args.hidden_dim * sizeof(float));
                    std::memcpy(decode_embedding_bytes.data(), decode_embedding_data.data(), decode_embedding_bytes.size());
                    
                    decode_embeddings_id = "decode_embeddings_" + std::to_string(step);
                    ctx->store_in_memory(
                        decode_embeddings_id,
                        std::make_tuple(1, args.hidden_dim),
                        DType::FLOAT32,
                        std::move(decode_embedding_bytes)
                    );
                    decode_input = {predicted_token};
                }
            } else {
                // Subsequent decode steps: use previous decode step's last token hidden state
                std::string prev_decode_last_token_id = "decode_output_" + std::to_string(step - 1) + "_last_token";
                auto prev_decode_last_token = ctx->get_in_memory(prev_decode_last_token_id);
                
                if (prev_decode_last_token) {
                    // Use previous decode step's last token hidden state (output_norm before projection)
                    decode_embeddings_id = prev_decode_last_token_id;
                    decode_input = {predicted_token};  // Still need input_ids for model signature, but won't be used
                } else {
                    // Fallback: use predicted token's embedding
                    decode_input = {predicted_token};
                    
                    if (use_random_embeddings) {
                        // Generate random embedding for the predicted token
                        std::mt19937 rng(static_cast<unsigned int>(args.seed + step + 1000));
                        std::normal_distribution<float> dist(0.0f, 0.02f);
                        
                        std::vector<float> decode_embedding_data(args.hidden_dim);
                        for (int64_t j = 0; j < args.hidden_dim; ++j) {
                            decode_embedding_data[j] = dist(rng);
                        }
                        
                        std::vector<uint8_t> decode_embedding_bytes(args.hidden_dim * sizeof(float));
                        std::memcpy(decode_embedding_bytes.data(), decode_embedding_data.data(), decode_embedding_bytes.size());
                        
                        decode_embeddings_id = "decode_embeddings_" + std::to_string(step);
                        ctx->store_in_memory(
                            decode_embeddings_id,
                            std::make_tuple(1, args.hidden_dim),
                            DType::FLOAT32,
                            std::move(decode_embedding_bytes)
                        );
                    }
                    // If not use_random_embeddings, model will do normal embedding lookup
                }
            }
            
            auto decode_start = std::chrono::high_resolution_clock::now();
            auto decode_result = inference.forward(
                decode_input,
                "decode_output_" + std::to_string(step),
                *ctx,
                "",  // mask_id (empty for single token)
                kv_cache_ids,
                cache_output_ids,
                decode_embeddings_id,  // Use pre-generated embedding if available
                args.profile  // Enable profiling if requested
            );
            auto decode_end = std::chrono::high_resolution_clock::now();
            double decode_elapsed = std::chrono::duration<double>(decode_end - decode_start).count();
            decode_elapsed -= inference.last_prefetch_warmup_ms() / 1000.0;
            if (decode_elapsed < 0.0) {
                decode_elapsed = 0.0;
            }
            decode_times.push_back(decode_elapsed);
            decode_traces.push_back(inference.last_phase_trace());
            
            // Get predicted token
            // Decode output should be (1, vocab_size) - single token's logits
            auto [decode_rows, decode_cols] = decode_result->shape();
            std::vector<float> decode_f32 = to_float32(decode_result);
            const float* decode_ptr = decode_f32.data();
            
            // Use actual shape instead of args.vocab_size
            int64_t vocab_size = decode_cols;
            if (vocab_size == 0 || vocab_size > args.vocab_size * 2) {
                // Fallback to args.vocab_size if shape seems wrong
                vocab_size = args.vocab_size;
            }
            
            // Find argmax
            predicted_token = 0;
            max_logit = decode_ptr[0];
            for (int64_t i = 1; i < vocab_size; ++i) {
                if (decode_ptr[i] > max_logit) {
                    max_logit = decode_ptr[i];
                    predicted_token = static_cast<int32_t>(i);
                }
            }

            if (step == 0) {
                first_token_predicted_id = predicted_token;
                first_token_max_logit = static_cast<double>(max_logit);
                if (args.dump_first_token_logits) {
                    first_token_logits = decode_f32;
                }
            }
            
            // Update cache IDs for next step
            kv_cache_ids = cache_output_ids;
            
            std::cout << "  Step " << (step + 1) << ": predicted=" << predicted_token 
                      << ", time=" << decode_elapsed << "s" << std::endl;

            // Clear bufferpool after each token to restart prefetch next step.
            ctx->stop_weight_prefetch();
        }
        
        double avg_decode_time = 0.0;
        if (!decode_times.empty()) {
            for (double t : decode_times) {
                avg_decode_time += t;
            }
            avg_decode_time /= decode_times.size();
        }
        
        std::cout << "\nDecode complete!" << std::endl;
        std::cout << "  Average decode time per token: " << avg_decode_time << "s" << std::endl;
        const auto matrix_catalog = build_matrix_catalog(prefill_trace, decode_traces);

        write_report_json(
            args.report_json_path,
            args.db_path,
            args.hidden_dim,
            args.num_layers,
            args.num_heads,
            args.num_kv_heads,
            args.chunk_size,
            args.arena_size_mb,
            args.seq_len,
            args.decode_steps,
            detect_thread_count(),
            args.vocab_size,
            args.seed,
            model_dtype,
            streamable_weight_bytes,
            streamable_matrix_count,
            args.prefetch_window,
            args.prefetch_latency_ms,
            args.prefetch_simulate,
            args.in_memory,
            args.disable_bufferpool,
            args.preload_file_path,
            matrix_catalog,
            preload_selected_matrix_ids,
            preload_selected_bytes,
            preload_selected_matrix_count,
            0.0,
            prefill_elapsed,
            prefill_trace,
            decode_times,
            decode_traces,
            args.dump_first_token_logits,
            first_token_logits,
            first_token_predicted_id,
            first_token_max_logit
        );

        // Print output statistics
        std::cout << "\nOutput statistics:" << std::endl;
        const float* result_ptr = prefill_f32.data();
        float min_val = result_ptr[0];
        float max_val = result_ptr[0];
        double sum_val = 0.0;
        int64_t count = prefill_rows * prefill_cols;
        
        for (int64_t i = 0; i < count; ++i) {
            float val = result_ptr[i];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum_val += val;
        }
        double mean_val = sum_val / count;
        
        double sum_sq_diff = 0.0;
        for (int64_t i = 0; i < count; ++i) {
            double diff = result_ptr[i] - mean_val;
            sum_sq_diff += diff * diff;
        }
        double std_val = std::sqrt(sum_sq_diff / count);
        
        std::cout << "  Min: " << min_val << std::endl;
        std::cout << "  Max: " << max_val << std::endl;
        std::cout << "  Mean: " << mean_val << std::endl;
        std::cout << "  Std: " << std_val << std::endl;
        
        std::cout << "\n" << "=" << std::string(60, '=') << std::endl;
        std::cout << "✓ Example complete!" << std::endl;
        std::cout << "=" << std::string(60, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

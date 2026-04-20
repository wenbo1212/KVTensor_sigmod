#include "kvtensor/stable_diffusion_inference.hpp"
#include "kvtensor/storage.hpp"
#include "kvtensor/context.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string json_escape(const std::string& value) {
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

int64_t detect_thread_count() {
    if (const char* omp_threads = std::getenv("OMP_NUM_THREADS")) {
        try {
            return std::stoll(omp_threads);
        } catch (const std::exception&) {
        }
    }
    unsigned int hw_threads = std::thread::hardware_concurrency();
    return hw_threads > 0 ? static_cast<int64_t>(hw_threads) : 1;
}

} // namespace

struct Args {
    std::string db_path = "./data/sd_db";
    std::string prompt = "a photo";
    int64_t text_hidden = 768;
    int64_t text_layers = 12;
    int64_t text_vocab_size = 49408;
    int64_t text_max_length = 77;
    int64_t unet_hidden = 320;
    int64_t unet_down_blocks = 4;
    int64_t unet_mid_blocks = 1;
    int64_t unet_up_blocks = 4;
    int64_t latent_channels = 4;
    int64_t sample_height = 64;
    int64_t sample_width = 64;
    int64_t image_size = 512;
    int64_t image_channels = 3;
    int64_t vae_conv_layers = 4;
    int64_t num_steps = 20;
    float guidance_scale = 7.5f;
    int64_t seed = 42;
    int64_t chunk_size = 512;
    int64_t arena_size_mb = 512;
    int64_t prefetch_window = 1;
    bool prefetch_simulate = false;
    int64_t prefetch_latency_ms = 0;
    std::string prefetch_graph_path = "";
    int64_t prefetch_graph_max_nodes = 1000000;
    std::string preload_file_path = "";
    bool in_memory = false;
    bool profile = false;
    std::string report_json_path = "";

    // ./stable_diffusion_inference_example --db-path /home/ubuntu/sd --arena-size-mb 1024 --prefetch-window 1 --profile 

    static Args parse(int argc, char* argv[]) {
        Args args;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--db-path" && i + 1 < argc) {
                args.db_path = argv[++i];
            } else if (arg == "--prompt" && i + 1 < argc) {
                args.prompt = argv[++i];
            } else if (arg == "--text-hidden" && i + 1 < argc) {
                args.text_hidden = std::stoll(argv[++i]);
            } else if (arg == "--text-layers" && i + 1 < argc) {
                args.text_layers = std::stoll(argv[++i]);
            } else if (arg == "--text-vocab-size" && i + 1 < argc) {
                args.text_vocab_size = std::stoll(argv[++i]);
            } else if (arg == "--text-max-length" && i + 1 < argc) {
                args.text_max_length = std::stoll(argv[++i]);
            } else if (arg == "--unet-hidden" && i + 1 < argc) {
                args.unet_hidden = std::stoll(argv[++i]);
            } else if (arg == "--unet-down-blocks" && i + 1 < argc) {
                args.unet_down_blocks = std::stoll(argv[++i]);
            } else if (arg == "--unet-mid-blocks" && i + 1 < argc) {
                args.unet_mid_blocks = std::stoll(argv[++i]);
            } else if (arg == "--unet-up-blocks" && i + 1 < argc) {
                args.unet_up_blocks = std::stoll(argv[++i]);
            } else if (arg == "--latent-channels" && i + 1 < argc) {
                args.latent_channels = std::stoll(argv[++i]);
            } else if (arg == "--sample-height" && i + 1 < argc) {
                args.sample_height = std::stoll(argv[++i]);
            } else if (arg == "--sample-width" && i + 1 < argc) {
                args.sample_width = std::stoll(argv[++i]);
            } else if (arg == "--image-size" && i + 1 < argc) {
                args.image_size = std::stoll(argv[++i]);
            } else if (arg == "--image-channels" && i + 1 < argc) {
                args.image_channels = std::stoll(argv[++i]);
            } else if (arg == "--vae-conv-layers" && i + 1 < argc) {
                args.vae_conv_layers = std::stoll(argv[++i]);
            } else if (arg == "--steps" && i + 1 < argc) {
                args.num_steps = std::stoll(argv[++i]);
            } else if (arg == "--guidance-scale" && i + 1 < argc) {
                args.guidance_scale = std::stof(argv[++i]);
            } else if (arg == "--seed" && i + 1 < argc) {
                args.seed = std::stoll(argv[++i]);
            } else if (arg == "--chunk-size" && i + 1 < argc) {
                args.chunk_size = std::stoll(argv[++i]);
            } else if (arg == "--arena-size-mb" && i + 1 < argc) {
                args.arena_size_mb = std::stoll(argv[++i]);
            } else if (arg == "--prefetch-window" && i + 1 < argc) {
                args.prefetch_window = std::stoll(argv[++i]);
            } else if (arg == "--prefetch-simulate") {
                args.prefetch_simulate = true;
            } else if (arg == "--prefetch-latency-ms" && i + 1 < argc) {
                args.prefetch_latency_ms = std::stoll(argv[++i]);
            } else if (arg == "--prefetch-graph" && i + 1 < argc) {
                args.prefetch_graph_path = argv[++i];
            } else if (arg == "--prefetch-graph-max-nodes" && i + 1 < argc) {
                args.prefetch_graph_max_nodes = std::stoll(argv[++i]);
            } else if (arg == "--preload-file" && i + 1 < argc) {
                args.preload_file_path = argv[++i];
            } else if (arg == "--in-memory") {
                args.in_memory = true;
            } else if (arg == "--profile") {
                args.profile = true;
            } else if (arg == "--report-json" && i + 1 < argc) {
                args.report_json_path = argv[++i];
            } else if (arg == "--help" || arg == "-h") {
                print_help();
                std::exit(0);
            } else if (i == 1 && arg[0] != '-') {
                args.db_path = arg;
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                print_help();
                std::exit(1);
            }
        }
        return args;
    }

    static void print_help() {
        std::cout << "Usage: stable_diffusion_inference_example [OPTIONS] [DB_PATH]\n\n";
        std::cout << "Options:\n";
        std::cout << "  --db-path PATH          SimpleDB path (default: ./data/sd_db)\n";
        std::cout << "  --prompt TEXT           Prompt string (default: a photo)\n";
        std::cout << "  --text-hidden N          Text hidden size (default: 768)\n";
        std::cout << "  --text-layers N          Text encoder layers (default: 12)\n";
        std::cout << "  --text-vocab-size N      Text vocab size (default: 49408)\n";
        std::cout << "  --text-max-length N      Text max length (default: 77)\n";
        std::cout << "  --unet-hidden N          UNet base channels (default: 320)\n";
        std::cout << "  --unet-down-blocks N     UNet down blocks (default: 4)\n";
        std::cout << "  --unet-mid-blocks N      UNet mid blocks (default: 1)\n";
        std::cout << "  --unet-up-blocks N       UNet up blocks (default: 4)\n";
        std::cout << "  --latent-channels N      Input channels (default: 4)\n";
        std::cout << "  --sample-height N        Input height (default: 64)\n";
        std::cout << "  --sample-width N         Input width (default: 64)\n";
        std::cout << "  --image-size N           Image size (default: 512)\n";
        std::cout << "  --image-channels N       Image channels (default: 3)\n";
        std::cout << "  --vae-conv-layers N      VAE conv layers (default: 4)\n";
        std::cout << "  --steps N                Denoising steps (default: 20)\n";
        std::cout << "  --guidance-scale X       Guidance scale (default: 7.5)\n";
        std::cout << "  --seed N                 Random seed (default: 42)\n";
        std::cout << "  --chunk-size N           Chunk size for weight splits (default: 512)\n";
        std::cout << "  --arena-size-mb N        Buffer pool memory limit in MB (default: 512)\n";
        std::cout << "  --prefetch-window N      Prefetch window size (default: 1)\n";
        std::cout << "  --prefetch-simulate      Enable simulated prefetch (skip DB reads)\n";
        std::cout << "  --prefetch-latency-ms N  Simulated get_into latency in ms (default: 0)\n";
        std::cout << "  --prefetch-graph PATH    Prefetch graph file path (default: empty)\n";
        std::cout << "  --prefetch-graph-max-nodes N  Max nodes to traverse in graph (default: 1000000)\n";
        std::cout << "  --preload-file PATH      Path to txt file with matrix IDs to preload\n";
        std::cout << "  --in-memory             Densify all weights into memory and disable bufferpool\n";
        std::cout << "  --profile               Enable detailed profiling\n";
        std::cout << "  --report-json PATH      Write machine-readable timing report to PATH\n";
        std::cout << "  --help, -h              Show this help message\n";
    }
};

void write_runtime_report(
    const std::string& path,
    const Args& args,
    const kvtensor::StableDiffusionRuntimeTrace& trace,
    double total_wall_ms,
    int64_t output_rows,
    int64_t output_cols
) {
    const double execution_ms_excluding_preload =
        std::max(0.0, total_wall_ms - trace.preload_ms);
    const double pipeline_ms =
        std::max(0.0, total_wall_ms - trace.preload_ms - trace.prefetch_warmup_ms);
    const double prefetch_get_time_ms =
        static_cast<double>(trace.bufferpool_prefetch_get_time_ns) / 1e6;
    const double io_hidden_est_ms =
        std::max(0.0, prefetch_get_time_ms - std::min(prefetch_get_time_ms, trace.bufferpool_wait_ms));
    const double io_hidden_est_ratio =
        prefetch_get_time_ms > 0.0 ? (io_hidden_est_ms / prefetch_get_time_ms) : 0.0;

    std::filesystem::path report_path(path);
    if (report_path.has_parent_path()) {
        std::filesystem::create_directories(report_path.parent_path());
    }

    std::ofstream out(report_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open report path: " + path);
    }

    out << "{\n";
    out << "  \"system\": {\n";
    out << "    \"db_path\": \"" << json_escape(args.db_path) << "\",\n";
    out << "    \"prompt\": \"" << json_escape(args.prompt) << "\",\n";
    out << "    \"chunk_size\": " << args.chunk_size << ",\n";
    out << "    \"steps\": " << args.num_steps << ",\n";
    out << "    \"guidance_scale\": " << args.guidance_scale << ",\n";
    out << "    \"arena_size_mb\": " << args.arena_size_mb << ",\n";
    out << "    \"prefetch_window\": " << args.prefetch_window << ",\n";
    out << "    \"prefetch_graph\": \"" << json_escape(args.prefetch_graph_path) << "\",\n";
    out << "    \"preload_file\": \"" << json_escape(args.preload_file_path) << "\",\n";
    out << "    \"in_memory\": " << (args.in_memory ? "true" : "false") << ",\n";
    out << "    \"profile\": " << (args.profile ? "true" : "false") << ",\n";
    out << "    \"thread_count\": " << detect_thread_count() << "\n";
    out << "  },\n";
    out << "  \"timing\": {\n";
    out << "    \"total_wall_ms\": " << total_wall_ms << ",\n";
    out << "    \"preload_ms\": " << trace.preload_ms << ",\n";
    out << "    \"prefetch_warmup_ms\": " << trace.prefetch_warmup_ms << ",\n";
    out << "    \"execution_ms_excluding_preload\": " << execution_ms_excluding_preload << ",\n";
    out << "    \"pipeline_ms_excluding_preload_and_prefetch_warmup\": " << pipeline_ms << "\n";
    out << "  },\n";
    out << "  \"bufferpool\": {\n";
    out << "    \"used_bufferpool\": " << (trace.used_bufferpool ? "true" : "false") << ",\n";
    out << "    \"get_chunk_calls\": " << trace.bufferpool_get_chunk_calls << ",\n";
    out << "    \"cache_hits\": " << trace.bufferpool_cache_hits << ",\n";
    out << "    \"cache_misses\": " << trace.bufferpool_cache_misses << ",\n";
    out << "    \"hit_rate\": " << trace.bufferpool_hit_rate << ",\n";
    out << "    \"wait_ms\": " << trace.bufferpool_wait_ms << ",\n";
    out << "    \"max_wait_ms\": " << trace.bufferpool_max_wait_ms << ",\n";
    out << "    \"evict_count\": " << trace.bufferpool_evict_count << ",\n";
    out << "    \"prefetch_get_calls\": " << trace.bufferpool_prefetch_get_calls << ",\n";
    out << "    \"prefetch_get_time_ns\": " << trace.bufferpool_prefetch_get_time_ns << ",\n";
    out << "    \"cached_chunks\": " << trace.bufferpool_cached_chunks << ",\n";
    out << "    \"slot_capacity\": " << trace.bufferpool_slot_capacity << ",\n";
    out << "    \"memory_used_bytes\": " << trace.bufferpool_memory_used_bytes << ",\n";
    out << "    \"memory_total_bytes\": " << trace.bufferpool_memory_total_bytes << ",\n";
    out << "    \"consumption_position\": " << trace.bufferpool_consumption_position << ",\n";
    out << "    \"prefetch_position\": " << trace.bufferpool_prefetch_position << ",\n";
    out << "    \"sequence_length\": " << trace.bufferpool_sequence_length << "\n";
    out << "  },\n";
    out << "  \"io_overlap_estimate\": {\n";
    out << "    \"prefetch_get_time_ms\": " << prefetch_get_time_ms << ",\n";
    out << "    \"exposed_wait_ms\": " << trace.bufferpool_wait_ms << ",\n";
    out << "    \"hidden_io_ms\": " << io_hidden_est_ms << ",\n";
    out << "    \"hidden_io_ratio\": " << io_hidden_est_ratio << "\n";
    out << "  },\n";
    out << "  \"output\": {\n";
    out << "    \"rows\": " << output_rows << ",\n";
    out << "    \"cols\": " << output_cols << "\n";
    out << "  }\n";
    out << "}\n";
}

int main(int argc, char* argv[]) {
    try {
        using namespace kvtensor;
        Args args = Args::parse(argc, argv);
        if (args.unet_down_blocks != 4 || args.unet_mid_blocks != 1 || args.unet_up_blocks != 4) {
            std::cerr << "Warning: UNet blocks are fixed to down=4, mid=1, up=4 per UNET_config.txt."
                      << " Provided values will be ignored." << std::endl;
        }

        std::cout << "=" << std::string(60, '=') << std::endl;
        std::cout << "Stable Diffusion Forward Pass (C++ Implementation)" << std::endl;
        std::cout << "=" << std::string(60, '=') << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Database path: " << args.db_path << std::endl;
        std::cout << "  Prompt: " << args.prompt << std::endl;
        std::cout << "  Text hidden: " << args.text_hidden << std::endl;
        std::cout << "  Text layers: " << args.text_layers << std::endl;
        std::cout << "  Text vocab size: " << args.text_vocab_size << std::endl;
        std::cout << "  Text max length: " << args.text_max_length << std::endl;
        std::cout << "  UNet hidden: " << args.unet_hidden << std::endl;
        std::cout << "  UNet blocks: down=" << args.unet_down_blocks
                  << " mid=" << args.unet_mid_blocks
                  << " up=" << args.unet_up_blocks << std::endl;
        std::cout << "  Latent size: " << args.sample_height << "x" << args.sample_width
                  << " (channels=" << args.latent_channels << ")" << std::endl;
        std::cout << "  Image size: " << args.image_size
                  << " (channels=" << args.image_channels << ")" << std::endl;
        std::cout << "  Steps: " << args.num_steps << std::endl;
        std::cout << "  Guidance scale: " << args.guidance_scale << std::endl;
        std::cout << "  Chunk size: " << args.chunk_size << std::endl;
        std::cout << "  Buffer pool memory limit (MB): " << args.arena_size_mb << std::endl;
        std::cout << "  Prefetch window: " << args.prefetch_window << std::endl;
        std::cout << "  Prefetch ring: true" << std::endl;
        std::cout << "  Prefetch simulate: " << (args.prefetch_simulate ? "true" : "false") << std::endl;
        if (args.prefetch_simulate) {
            std::cout << "  Prefetch latency (ms): " << args.prefetch_latency_ms << std::endl;
        }
        if (!args.prefetch_graph_path.empty()) {
            std::cout << "  Prefetch graph: " << args.prefetch_graph_path << std::endl;
            std::cout << "  Prefetch graph max nodes: " << args.prefetch_graph_max_nodes << std::endl;
        }
        if (!args.preload_file_path.empty()) {
            std::cout << "  Preload file: " << args.preload_file_path << std::endl;
        }
        std::cout << "  In-memory mode: " << (args.in_memory ? "true" : "false") << std::endl;
        std::cout << std::endl;

        auto storage = std::make_unique<SimpleDBStorage>(args.db_path, true /*read_only*/);
        auto registry = std::make_unique<MatrixRegistry>(storage.get());
        auto ctx = std::make_unique<OperatorContext>(storage.get(), registry.get());

        StableDiffusionConfig cfg;
        cfg.text_encoder.token_embedding_id = "text.token_embedding";
        cfg.text_encoder.final_layer_norm_id = "text.final_norm_weight";
        cfg.text_encoder.hidden_size = args.text_hidden;
        cfg.text_encoder.max_length = args.text_max_length;
        cfg.text_encoder.vocab_size = args.text_vocab_size;
        for (int64_t i = 0; i < args.text_layers; ++i) {
            cfg.text_encoder.transformer_block_ids.push_back("text.block." + std::to_string(i));
        }

        cfg.unet.latent_channels = args.latent_channels;
        cfg.unet.model_channels = args.unet_hidden;
        cfg.unet.sample_height = args.sample_height;
        cfg.unet.sample_width = args.sample_width;
        cfg.unet.input_conv_weight_id = "unet.conv_in.weight";
        cfg.unet.input_conv_bias_id = "unet.conv_in.bias";
        cfg.unet.output_conv_weight_id = "unet.conv_out.weight";
        cfg.unet.output_conv_bias_id = "unet.conv_out.bias";

        cfg.vae.latent_channels = args.latent_channels;
        cfg.vae.image_channels = args.image_channels;
        cfg.vae.image_size = args.image_size;
        cfg.vae.scale_factor = args.image_size / std::max<int64_t>(args.sample_height, 1);
        for (int64_t i = 0; i < args.vae_conv_layers; ++i) {
            cfg.vae.decoder_weight_ids.push_back("vae.decoder.conv" + std::to_string(i));
        }

        cfg.scheduler.num_inference_steps = args.num_steps;
        cfg.scheduler.guidance_scale = args.guidance_scale;
        cfg.scheduler.seed = static_cast<uint64_t>(args.seed);

        cfg.prefetch_window = static_cast<size_t>(args.prefetch_window);
        cfg.arena_size_mb = static_cast<size_t>(args.arena_size_mb);
        cfg.preload_file_path = args.preload_file_path;

        StableDiffusionInferenceConfig infer_cfg;
        infer_cfg.pipeline = cfg;
        infer_cfg.prefetch.arena_size_mb = static_cast<size_t>(args.arena_size_mb);
        infer_cfg.prefetch.prefetch_window = static_cast<size_t>(args.prefetch_window);
        infer_cfg.prefetch.ring = false;
        infer_cfg.prefetch.simulate_prefetch = args.prefetch_simulate;
        infer_cfg.prefetch.simulate_get_latency_ms = static_cast<uint64_t>(args.prefetch_latency_ms);
        infer_cfg.prefetch.graph_path = args.prefetch_graph_path;
        infer_cfg.prefetch.graph_max_nodes = static_cast<size_t>(args.prefetch_graph_max_nodes);
        infer_cfg.preload_file_path = args.preload_file_path;
        infer_cfg.in_memory = args.in_memory;

        StableDiffusionInference inference(infer_cfg);

        auto start = std::chrono::high_resolution_clock::now();
        auto result = inference.generate(args.prompt, "sd_output", *ctx, args.profile);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        auto [rows, cols] = result->shape();
        if (!args.report_json_path.empty()) {
            write_runtime_report(
                args.report_json_path,
                args,
                inference.last_runtime_trace(),
                elapsed_ms,
                rows,
                cols
            );
        }
        std::cout << "\nGeneration complete." << std::endl;
        std::cout << "  Output shape: (" << rows << ", " << cols << ")" << std::endl;
        std::cout << "  Time: " << (elapsed_ms / 1000.0) << "s" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

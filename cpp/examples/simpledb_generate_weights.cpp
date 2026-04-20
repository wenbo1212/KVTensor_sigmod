#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

enum class Precision {
    Float32,
    Int8,
    BFloat16,
};

struct Config {
    std::string db_path = "./simpledb_weights";
    int hidden_dim = 4096;
    int num_layers = 32;
    int num_heads = 32;
    int num_kv_heads = 8;
    int chunk_size = 512;
    int vocab_size = 32000;
    int seed = 42;
    Precision precision = Precision::Float32;
};

struct IndexHeader {
    uint32_t magic = 0x31494B53;  // "SKI1"
    uint32_t version = 1;
    uint32_t alignment = 4096;
    uint64_t count = 0;
};

struct IndexEntry {
    std::string key;
    uint64_t offset = 0;
    uint32_t value_len = 0;
};

static inline uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

class MetadataWriter {
public:
    explicit MetadataWriter(const std::string& path) {
        std::filesystem::create_directories(path);
        meta_path_ = (std::filesystem::path(path) / "metadata.jsonl").string();
        out_.open(meta_path_, std::ios::binary | std::ios::trunc);
        if (!out_) {
            throw std::runtime_error("Failed to open metadata file: " + meta_path_);
        }
    }

    void write(const std::string& json_line) {
        out_ << json_line << "\n";
        if (!out_) {
            throw std::runtime_error("Failed to write metadata line");
        }
    }

private:
    std::string meta_path_;
    std::ofstream out_;
};

class SimpleDBWriter {
public:
    explicit SimpleDBWriter(const std::string& path) {
        std::filesystem::create_directories(path);
        data_path_ = (std::filesystem::path(path) / "data.kv").string();
        index_path_ = (std::filesystem::path(path) / "index.kv").string();

        data_.open(data_path_, std::ios::binary | std::ios::trunc);
        if (!data_) {
            throw std::runtime_error("Failed to open data file: " + data_path_);
        }
    }

    void write(const std::string& key, const uint8_t* data, size_t len) {
        uint64_t current = static_cast<uint64_t>(data_.tellp());
        uint64_t aligned = align_up(current, kAlignment);
        if (aligned > current) {
            std::vector<char> zeros(static_cast<size_t>(aligned - current), 0);
            data_.write(zeros.data(), zeros.size());
        }

        uint64_t offset = static_cast<uint64_t>(data_.tellp());
        data_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(len));

        if (!data_) {
            throw std::runtime_error("Failed to write record for key: " + key);
        }

        entries_.push_back(IndexEntry{key, offset, static_cast<uint32_t>(len)});
    }

    void flush_index() {
        std::ofstream index(index_path_, std::ios::binary | std::ios::trunc);
        if (!index) {
            throw std::runtime_error("Failed to open index file: " + index_path_);
        }

        IndexHeader header;
        header.count = entries_.size();
        header.alignment = kAlignment;
        index.write(reinterpret_cast<const char*>(&header), sizeof(header));

        for (const auto& entry : entries_) {
            uint32_t key_len = static_cast<uint32_t>(entry.key.size());
            index.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
            index.write(entry.key.data(), static_cast<std::streamsize>(entry.key.size()));
            index.write(reinterpret_cast<const char*>(&entry.offset), sizeof(entry.offset));
            index.write(reinterpret_cast<const char*>(&entry.value_len), sizeof(entry.value_len));
        }

        if (!index) {
            throw std::runtime_error("Failed to write index file");
        }
    }

private:
    std::string data_path_;
    std::string index_path_;
    std::ofstream data_;
    std::vector<IndexEntry> entries_;
    static constexpr uint32_t kAlignment = 4096;
};

std::string format_chunk_key(const std::string& matrix_id, const std::string& split, int64_t chunk_idx) {
    std::ostringstream oss;
    oss << matrix_id << ":" << split << ":" << std::setfill('0') << std::setw(6) << chunk_idx;
    return oss.str();
}

std::vector<float> make_normal_chunk(size_t count, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

std::vector<int8_t> quantize_int8(const std::vector<float>& data) {
    auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    float range = *max_it - *min_it;
    float scale = (range == 0.0f) ? 1.0f : (range / 255.0f);
    std::vector<int8_t> out(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        float scaled = data[i] / scale;
        int32_t q = static_cast<int32_t>(std::nearbyint(scaled));
        q = std::max(-128, std::min(127, q));
        out[i] = static_cast<int8_t>(q);
    }
    return out;
}

std::vector<uint16_t> quantize_bf16(const std::vector<float>& data) {
    std::vector<uint16_t> out(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &data[i], sizeof(bits));
        out[i] = static_cast<uint16_t>(bits >> 16);
    }
    return out;
}

std::string encode_metadata_json(const std::string& matrix_id,
                                 int64_t rows,
                                 int64_t cols,
                                 const std::string& dtype,
                                 const std::string& split_mode,
                                 int64_t chunk_size) {
    std::ostringstream oss;
    oss << "{"
        << "\"matrix_id\":\"" << matrix_id << "\","
        << "\"shape\":[" << rows << "," << cols << "],"
        << "\"dtype\":\"" << dtype << "\","
        << "\"split_mode\":\"" << split_mode << "\","
        << "\"chunk_size\":" << chunk_size
        << "}";
    return oss.str();
}

std::string dtype_string(Precision precision) {
    switch (precision) {
        case Precision::Float32: return "float32";
        case Precision::Int8: return "int8";
        case Precision::BFloat16: return "bfloat16";
        default: return "float32";
    }
}

bool write_chunk(SimpleDBWriter& db,
                 const std::string& key,
                 const std::vector<float>& data,
                 Precision precision) {
    if (precision == Precision::Float32) {
        db.write(key, reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(float));
        return true;
    }
    if (precision == Precision::Int8) {
        auto q = quantize_int8(data);
        db.write(key, reinterpret_cast<const uint8_t*>(q.data()), q.size() * sizeof(int8_t));
        return true;
    }
    auto q = quantize_bf16(data);
    db.write(key, reinterpret_cast<const uint8_t*>(q.data()), q.size() * sizeof(uint16_t));
    return true;
}

bool write_float32_chunk(SimpleDBWriter& db, const std::string& key, const std::vector<float>& data) {
    db.write(key, reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(float));
    return true;
}

void generate_weights(const Config& cfg) {
    SimpleDBWriter db(cfg.db_path);
    MetadataWriter meta(cfg.db_path);

    std::mt19937 rng(cfg.seed);
    int ffn_dim = cfg.hidden_dim * 7 / 2;

    std::cout << "Creating weights in SimpleKVStore..." << std::endl;
    std::cout << "  Path: " << cfg.db_path << std::endl;
    std::cout << "  Hidden dim: " << cfg.hidden_dim << ", layers: " << cfg.num_layers
              << ", chunk size: " << cfg.chunk_size << std::endl;

    for (int layer_idx = 0; layer_idx < cfg.num_layers; ++layer_idx) {
        std::string prefix = "transformer." + std::to_string(layer_idx);

        int q_dim = cfg.num_heads * (cfg.hidden_dim / cfg.num_heads);
        int kv_dim = cfg.num_kv_heads * (cfg.hidden_dim / cfg.num_heads);
        int qkv_dim = q_dim + 2 * kv_dim;
        meta.write(encode_metadata_json(prefix + ".attn_qkv_proj",
                                        cfg.hidden_dim, qkv_dim,
                                        dtype_string(cfg.precision),
                                        "column",
                                        cfg.chunk_size));

        // QKV projection
        int qkv_chunks = (qkv_dim + cfg.chunk_size - 1) / cfg.chunk_size;
        for (int j = 0; j < qkv_chunks; ++j) {
            int c0 = j * cfg.chunk_size;
            int c1 = std::min(c0 + cfg.chunk_size, qkv_dim);
            size_t count = static_cast<size_t>(cfg.hidden_dim) * static_cast<size_t>(c1 - c0);
            auto chunk = make_normal_chunk(count, rng);
            std::string key = format_chunk_key(prefix + ".attn_qkv_proj", "col", j);
            if (!write_chunk(db, key, chunk, cfg.precision)) {
                throw std::runtime_error("Failed to write qkv chunk");
            }
        }

        // O projection
        meta.write(encode_metadata_json(prefix + ".attn_o_proj",
                                        cfg.hidden_dim, cfg.hidden_dim,
                                        dtype_string(cfg.precision),
                                        "column",
                                        cfg.chunk_size));
        int o_chunks = (cfg.hidden_dim + cfg.chunk_size - 1) / cfg.chunk_size;
        for (int j = 0; j < o_chunks; ++j) {
            int c0 = j * cfg.chunk_size;
            int c1 = std::min(c0 + cfg.chunk_size, cfg.hidden_dim);
            size_t count = static_cast<size_t>(cfg.hidden_dim) * static_cast<size_t>(c1 - c0);
            auto chunk = make_normal_chunk(count, rng);
            std::string key = format_chunk_key(prefix + ".attn_o_proj", "col", j);
            if (!write_chunk(db, key, chunk, cfg.precision)) {
                throw std::runtime_error("Failed to write o_proj chunk");
            }
        }

        // Gate+Up projection (interleaved)
        meta.write(encode_metadata_json(prefix + ".ffn_gate_up_proj",
                                        cfg.hidden_dim, 2 * ffn_dim,
                                        dtype_string(cfg.precision),
                                        "column",
                                        cfg.chunk_size));
        int output_chunk_size = cfg.chunk_size / 2;
        int gate_up_chunks = (2 * ffn_dim + cfg.chunk_size - 1) / cfg.chunk_size;
        for (int j = 0; j < gate_up_chunks; ++j) {
            int gate_start = j * output_chunk_size;
            int gate_end = std::min(gate_start + output_chunk_size, ffn_dim);
            int width = gate_end - gate_start;
            size_t count = static_cast<size_t>(cfg.hidden_dim) * static_cast<size_t>(width);
            auto gate_chunk = make_normal_chunk(count, rng);
            auto up_chunk = make_normal_chunk(count, rng);

            std::vector<float> combined;
            combined.resize(static_cast<size_t>(cfg.hidden_dim) * static_cast<size_t>(width) * 2);
            for (int r = 0; r < cfg.hidden_dim; ++r) {
                size_t row_offset = static_cast<size_t>(r) * static_cast<size_t>(width);
                size_t combined_offset = static_cast<size_t>(r) * static_cast<size_t>(width) * 2;
                std::memcpy(combined.data() + combined_offset,
                            gate_chunk.data() + row_offset,
                            sizeof(float) * static_cast<size_t>(width));
                std::memcpy(combined.data() + combined_offset + static_cast<size_t>(width),
                            up_chunk.data() + row_offset,
                            sizeof(float) * static_cast<size_t>(width));
            }

            std::string key = format_chunk_key(prefix + ".ffn_gate_up_proj", "col", j);
            if (!write_chunk(db, key, combined, cfg.precision)) {
                throw std::runtime_error("Failed to write gate_up chunk");
            }
        }

        // Down projection (half chunk size)
        int down_chunk_size = cfg.chunk_size / 2;
        meta.write(encode_metadata_json(prefix + ".ffn_down_proj",
                                        ffn_dim, cfg.hidden_dim,
                                        dtype_string(cfg.precision),
                                        "column",
                                        down_chunk_size));
        int down_chunks = (cfg.hidden_dim + down_chunk_size - 1) / down_chunk_size;
        for (int j = 0; j < down_chunks; ++j) {
            int c0 = j * down_chunk_size;
            int c1 = std::min(c0 + down_chunk_size, cfg.hidden_dim);
            size_t count = static_cast<size_t>(ffn_dim) * static_cast<size_t>(c1 - c0);
            auto chunk = make_normal_chunk(count, rng);
            std::string key = format_chunk_key(prefix + ".ffn_down_proj", "col", j);
            if (!write_chunk(db, key, chunk, cfg.precision)) {
                throw std::runtime_error("Failed to write down_proj chunk");
            }
        }
    }

    // Normalization weights (float32)
    for (int layer_idx = 0; layer_idx < cfg.num_layers; ++layer_idx) {
        std::string prefix = "transformer." + std::to_string(layer_idx);
        meta.write(encode_metadata_json(prefix + ".attn_norm_weight",
                                        1, cfg.hidden_dim,
                                        "float32",
                                        "column",
                                        cfg.chunk_size));
        meta.write(encode_metadata_json(prefix + ".ffn_norm_weight",
                                        1, cfg.hidden_dim,
                                        "float32",
                                        "column",
                                        cfg.chunk_size));
        std::vector<float> ones(cfg.hidden_dim, 1.0f);
        std::string attn_key = format_chunk_key(prefix + ".attn_norm_weight", "col", 0);
        std::string ffn_key = format_chunk_key(prefix + ".ffn_norm_weight", "col", 0);
        if (!write_float32_chunk(db, attn_key, ones) || !write_float32_chunk(db, ffn_key, ones)) {
            throw std::runtime_error("Failed to write norm weights");
        }
    }
    std::vector<float> ones(cfg.hidden_dim, 1.0f);
    meta.write(encode_metadata_json("output.output_norm_weight",
                                    1, cfg.hidden_dim,
                                    "float32",
                                    "column",
                                    cfg.chunk_size));
    std::string out_norm_key = format_chunk_key("output.output_norm_weight", "col", 0);
    if (!write_float32_chunk(db, out_norm_key, ones)) {
        throw std::runtime_error("Failed to write output norm");
    }

    // Output projection
    meta.write(encode_metadata_json("output.output_proj",
                                    cfg.hidden_dim, cfg.vocab_size,
                                    dtype_string(cfg.precision),
                                    "column",
                                    cfg.chunk_size));
    int out_chunks = (cfg.vocab_size + cfg.chunk_size - 1) / cfg.chunk_size;
    for (int j = 0; j < out_chunks; ++j) {
        int c0 = j * cfg.chunk_size;
        int c1 = std::min(c0 + cfg.chunk_size, cfg.vocab_size);
        size_t count = static_cast<size_t>(cfg.hidden_dim) * static_cast<size_t>(c1 - c0);
        auto chunk = make_normal_chunk(count, rng);
        std::string key = format_chunk_key("output.output_proj", "col", j);
        if (!write_chunk(db, key, chunk, cfg.precision)) {
            throw std::runtime_error("Failed to write output projection chunk");
        }
    }

    db.flush_index();
    std::cout << "✓ SimpleKVStore weight generation complete." << std::endl;
}

Precision parse_precision(const std::string& value) {
    if (value == "float32") {
        return Precision::Float32;
    }
    if (value == "int8") {
        return Precision::Int8;
    }
    if (value == "bfloat16") {
        return Precision::BFloat16;
    }
    throw std::runtime_error("Unsupported precision: " + value);
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto read_int = [&](int& target) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            target = std::stoi(argv[++i]);
        };
        if (arg == "--db-path") {
            cfg.db_path = argv[++i];
        } else if (arg == "--hidden-dim") {
            read_int(cfg.hidden_dim);
        } else if (arg == "--num-layers") {
            read_int(cfg.num_layers);
        } else if (arg == "--num-heads") {
            read_int(cfg.num_heads);
        } else if (arg == "--num-kv-heads") {
            read_int(cfg.num_kv_heads);
        } else if (arg == "--chunk-size") {
            read_int(cfg.chunk_size);
        } else if (arg == "--vocab-size") {
            read_int(cfg.vocab_size);
        } else if (arg == "--seed") {
            read_int(cfg.seed);
        } else if (arg == "--precision") {
            cfg.precision = parse_precision(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --db-path PATH\n"
                      << "  --hidden-dim N\n"
                      << "  --num-layers N\n"
                      << "  --num-heads N\n"
                      << "  --num-kv-heads N\n"
                      << "  --chunk-size N\n"
                      << "  --vocab-size N\n"
                      << "  --seed N\n"
                      << "  --precision float32|int8|bfloat16\n";
            std::exit(0);
        }
    }

    if (cfg.hidden_dim % cfg.num_heads != 0) {
        throw std::runtime_error("hidden_dim must be divisible by num_heads");
    }
    if (cfg.num_kv_heads == 0) {
        cfg.num_kv_heads = 1;
    }
    if (cfg.num_heads % cfg.num_kv_heads != 0) {
        throw std::runtime_error("num_heads must be divisible by num_kv_heads");
    }
    return cfg;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        generate_weights(cfg);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

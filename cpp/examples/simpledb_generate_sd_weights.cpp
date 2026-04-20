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
    std::string db_path = "./sd_weights";
    std::string graph_path = "";
    int chunk_size = 512;
    int seed = 42;
    Precision precision = Precision::Float32;

    int text_hidden = 768;
    int text_layers = 12;
    int text_vocab_size = 49408;
    int text_ffn_mult = 4;

    int unet_hidden = 320;
    int unet_down_blocks = 4;
    int unet_mid_blocks = 1;
    int unet_up_blocks = 4;
    int unet_ffn_mult = 4;

    int latent_channels = 4;
    int vae_hidden = 128;
    int image_channels = 3;
    int vae_conv_layers = 4;
    int denoise_steps = 20;
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
        // Ensure data file ends on alignment for O_DIRECT reads.
        uint64_t current = static_cast<uint64_t>(data_.tellp());
        uint64_t aligned = align_up(current, kAlignment);
        if (aligned > current) {
            std::vector<char> zeros(static_cast<size_t>(aligned - current), 0);
            data_.write(zeros.data(), zeros.size());
            if (!data_) {
                throw std::runtime_error("Failed to pad data file for alignment");
            }
        }

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

void write_row_matrix(SimpleDBWriter& db,
                      MetadataWriter& meta,
                      const std::string& matrix_id,
                      int64_t rows,
                      int64_t cols,
                      int64_t chunk_size,
                      Precision precision,
                      std::mt19937& rng) {
    meta.write(encode_metadata_json(matrix_id, rows, cols, dtype_string(precision), "row", chunk_size));
    int64_t num_chunks = (rows + chunk_size - 1) / chunk_size;
    for (int64_t i = 0; i < num_chunks; ++i) {
        int64_t r0 = i * chunk_size;
        int64_t r1 = std::min(r0 + chunk_size, rows);
        size_t count = static_cast<size_t>((r1 - r0) * cols);
        auto chunk = make_normal_chunk(count, rng);
        std::string key = format_chunk_key(matrix_id, "row", i);
        if (!write_chunk(db, key, chunk, precision)) {
            throw std::runtime_error("Failed to write row chunk: " + matrix_id);
        }
    }
}

void write_column_matrix(SimpleDBWriter& db,
                         MetadataWriter& meta,
                         const std::string& matrix_id,
                         int64_t rows,
                         int64_t cols,
                         int64_t chunk_size,
                         Precision precision,
                         std::mt19937& rng) {
    meta.write(encode_metadata_json(matrix_id, rows, cols, dtype_string(precision), "column", chunk_size));
    int64_t num_chunks = (cols + chunk_size - 1) / chunk_size;
    for (int64_t j = 0; j < num_chunks; ++j) {
        int64_t c0 = j * chunk_size;
        int64_t c1 = std::min(c0 + chunk_size, cols);
        size_t count = static_cast<size_t>(rows) * static_cast<size_t>(c1 - c0);
        auto chunk = make_normal_chunk(count, rng);
        std::string key = format_chunk_key(matrix_id, "col", j);
        if (!write_chunk(db, key, chunk, precision)) {
            throw std::runtime_error("Failed to write col chunk: " + matrix_id);
        }
    }
}

void write_qkv_fused(SimpleDBWriter& db,
                     MetadataWriter& meta,
                     const std::string& matrix_id,
                     int64_t hidden,
                     int64_t chunk_size,
                     Precision precision,
                     std::mt19937& rng) {
    int64_t qkv_dim = hidden * 3;
    meta.write(encode_metadata_json(matrix_id, hidden, qkv_dim, dtype_string(precision), "column", chunk_size));

    auto Wq = make_normal_chunk(static_cast<size_t>(hidden * hidden), rng);
    auto Wk = make_normal_chunk(static_cast<size_t>(hidden * hidden), rng);
    auto Wv = make_normal_chunk(static_cast<size_t>(hidden * hidden), rng);

    int64_t num_chunks = (qkv_dim + chunk_size - 1) / chunk_size;
    for (int64_t j = 0; j < num_chunks; ++j) {
        int64_t c0 = j * chunk_size;
        int64_t c1 = std::min(c0 + chunk_size, qkv_dim);
        int64_t width = c1 - c0;
        std::vector<float> chunk(static_cast<size_t>(hidden * width));
        for (int64_t r = 0; r < hidden; ++r) {
            for (int64_t c = 0; c < width; ++c) {
                int64_t abs_c = c0 + c;
                float v = 0.0f;
                if (abs_c < hidden) {
                    v = Wq[static_cast<size_t>(r * hidden + abs_c)];
                } else if (abs_c < 2 * hidden) {
                    v = Wk[static_cast<size_t>(r * hidden + (abs_c - hidden))];
                } else {
                    v = Wv[static_cast<size_t>(r * hidden + (abs_c - 2 * hidden))];
                }
                chunk[static_cast<size_t>(r * width + c)] = v;
            }
        }
        std::string key = format_chunk_key(matrix_id, "col", j);
        if (!write_chunk(db, key, chunk, precision)) {
            throw std::runtime_error("Failed to write qkv chunk: " + matrix_id);
        }
    }
}

void write_gate_up_interleaved(SimpleDBWriter& db,
                               MetadataWriter& meta,
                               const std::string& matrix_id,
                               int64_t hidden,
                               int64_t ffn_dim,
                               int64_t chunk_size,
                               Precision precision,
                               std::mt19937& rng) {
    meta.write(encode_metadata_json(matrix_id, hidden, 2 * ffn_dim,
                                    dtype_string(precision), "column", chunk_size));
    int64_t output_chunk_size = chunk_size / 2;
    int64_t gate_up_chunks = (2 * ffn_dim + chunk_size - 1) / chunk_size;
    for (int64_t j = 0; j < gate_up_chunks; ++j) {
        int64_t gate_start = j * output_chunk_size;
        int64_t gate_end = std::min(gate_start + output_chunk_size, ffn_dim);
        int64_t width = gate_end - gate_start;
        size_t count = static_cast<size_t>(hidden) * static_cast<size_t>(width);
        auto gate_chunk = make_normal_chunk(count, rng);
        auto up_chunk = make_normal_chunk(count, rng);

        std::vector<float> combined(static_cast<size_t>(hidden) * static_cast<size_t>(width) * 2);
        for (int r = 0; r < hidden; ++r) {
            size_t row_offset = static_cast<size_t>(r) * static_cast<size_t>(width);
            size_t combined_offset = static_cast<size_t>(r) * static_cast<size_t>(width) * 2;
            std::memcpy(combined.data() + combined_offset,
                        gate_chunk.data() + row_offset,
                        sizeof(float) * static_cast<size_t>(width));
            std::memcpy(combined.data() + combined_offset + static_cast<size_t>(width),
                        up_chunk.data() + row_offset,
                        sizeof(float) * static_cast<size_t>(width));
        }

        std::string key = format_chunk_key(matrix_id, "col", j);
        if (!write_chunk(db, key, combined, precision)) {
            throw std::runtime_error("Failed to write gate_up chunk: " + matrix_id);
        }
    }
}

void write_norm_vector(SimpleDBWriter& db,
                       MetadataWriter& meta,
                       const std::string& matrix_id,
                       int64_t hidden,
                       int64_t chunk_size,
                       Precision precision,
                       float value = 1.0f) {
    meta.write(encode_metadata_json(matrix_id, 1, hidden, dtype_string(precision), "column", chunk_size));
    int64_t num_chunks = (hidden + chunk_size - 1) / chunk_size;
    for (int64_t j = 0; j < num_chunks; ++j) {
        int64_t c0 = j * chunk_size;
        int64_t c1 = std::min(c0 + chunk_size, hidden);
        int64_t width = c1 - c0;
        std::vector<float> vals(static_cast<size_t>(width), value);
        std::string key = format_chunk_key(matrix_id, "col", j);
        if (!write_chunk(db, key, vals, precision)) {
            throw std::runtime_error("Failed to write norm: " + matrix_id);
        }
    }
}

void write_bias_vector(SimpleDBWriter& db,
                       MetadataWriter& meta,
                       const std::string& matrix_id,
                       int64_t hidden,
                       int64_t chunk_size,
                       std::mt19937& rng,
                       Precision precision) {
    meta.write(encode_metadata_json(matrix_id, 1, hidden, dtype_string(precision), "column", chunk_size));
    int64_t num_chunks = (hidden + chunk_size - 1) / chunk_size;
    for (int64_t j = 0; j < num_chunks; ++j) {
        int64_t c0 = j * chunk_size;
        int64_t c1 = std::min(c0 + chunk_size, hidden);
        int64_t width = c1 - c0;
        auto data = make_normal_chunk(static_cast<size_t>(width), rng);
        std::string key = format_chunk_key(matrix_id, "col", j);
        if (!write_chunk(db, key, data, precision)) {
            throw std::runtime_error("Failed to write bias: " + matrix_id);
        }
    }
}

void generate_text_encoder(SimpleDBWriter& db, MetadataWriter& meta, const Config& cfg, std::mt19937& rng) {
    write_row_matrix(db, meta, "text.token_embedding", cfg.text_vocab_size,
                     cfg.text_hidden, cfg.chunk_size, cfg.precision, rng);

    for (int i = 0; i < cfg.text_layers; ++i) {
        std::string prefix = "text.block." + std::to_string(i);
        write_qkv_fused(db, meta, prefix + ".attn_qkv_proj", cfg.text_hidden,
                        cfg.chunk_size, cfg.precision, rng);
        write_column_matrix(db, meta, prefix + ".attn_out_proj", cfg.text_hidden,
                            cfg.text_hidden, cfg.chunk_size, cfg.precision, rng);

        int ffn_dim = cfg.text_hidden * cfg.text_ffn_mult;
        write_gate_up_interleaved(db, meta, prefix + ".ffn_gate_up_proj",
                                  cfg.text_hidden, ffn_dim, cfg.chunk_size,
                                  cfg.precision, rng);

        int down_chunk_size = cfg.chunk_size / 2;
        write_column_matrix(db, meta, prefix + ".ffn_down_proj", ffn_dim,
                            cfg.text_hidden, down_chunk_size, cfg.precision, rng);

        write_norm_vector(db, meta, prefix + ".norm1_weight", cfg.text_hidden, cfg.chunk_size, cfg.precision, 1.0f);
        write_norm_vector(db, meta, prefix + ".norm2_weight", cfg.text_hidden, cfg.chunk_size, cfg.precision, 1.0f);
    }

    write_norm_vector(db, meta, "text.final_norm_weight", cfg.text_hidden, cfg.chunk_size, cfg.precision, 1.0f);
}

void write_conv(SimpleDBWriter& db, MetadataWriter& meta,
                const std::string& prefix,
                int out_c, int in_c, int k,
                const Config& cfg,
                std::mt19937& rng) {
    int64_t cols = static_cast<int64_t>(in_c) * k * k;
    write_column_matrix(db, meta, prefix + ".weight", out_c, cols,
                        cfg.chunk_size, cfg.precision, rng);
    write_bias_vector(db, meta, prefix + ".bias", out_c, cfg.chunk_size, rng, cfg.precision);
}

void write_group_norm(SimpleDBWriter& db, MetadataWriter& meta,
                      const std::string& prefix, int channels, const Config& cfg) {
    write_norm_vector(db, meta, prefix + ".weight", channels, cfg.chunk_size, cfg.precision, 1.0f);
    write_norm_vector(db, meta, prefix + ".bias", channels, cfg.chunk_size, cfg.precision, 0.0f);
}

void write_linear(SimpleDBWriter& db, MetadataWriter& meta,
                  const std::string& prefix,
                  int in_features, int out_features,
                  const Config& cfg,
                  std::mt19937& rng) {
    write_column_matrix(db, meta, prefix + ".weight", in_features, out_features,
                        cfg.chunk_size, cfg.precision, rng);
    write_bias_vector(db, meta, prefix + ".bias", out_features, cfg.chunk_size, rng, cfg.precision);
}

void write_attention(SimpleDBWriter& db, MetadataWriter& meta,
                     const std::string& prefix,
                     int channels,
                     const Config& cfg,
                     std::mt19937& rng) {
    write_group_norm(db, meta, prefix + ".group_norm", channels, cfg);
    write_qkv_fused(db, meta, prefix + ".attn_qkv_proj", channels,
                    cfg.chunk_size, cfg.precision, rng);
    write_column_matrix(db, meta, prefix + ".attn_out_proj", channels,
                        channels, cfg.chunk_size, cfg.precision, rng);
}

void write_layer_norm(SimpleDBWriter& db, MetadataWriter& meta,
                      const std::string& prefix, int channels, const Config& cfg) {
    write_norm_vector(db, meta, prefix + ".weight", channels, cfg.chunk_size, cfg.precision, 1.0f);
    write_norm_vector(db, meta, prefix + ".bias", channels, cfg.chunk_size, cfg.precision, 0.0f);
}

void write_transformer2d(SimpleDBWriter& db, MetadataWriter& meta,
                         const std::string& prefix,
                         int hidden, int cross_dim,
                         const Config& cfg,
                         std::mt19937& rng) {
    write_conv(db, meta, prefix + ".proj_in", hidden, hidden, 1, cfg, rng);
    write_conv(db, meta, prefix + ".proj_out", hidden, hidden, 1, cfg, rng);

    std::string blk = prefix + ".transformer_blocks.0";
    write_layer_norm(db, meta, blk + ".norm1", hidden, cfg);
    write_layer_norm(db, meta, blk + ".norm2", hidden, cfg);
    write_layer_norm(db, meta, blk + ".norm3", hidden, cfg);

    write_column_matrix(db, meta, blk + ".attn1.to_q.weight", hidden, hidden, cfg.chunk_size, cfg.precision, rng);
    write_column_matrix(db, meta, blk + ".attn1.to_k.weight", hidden, hidden, cfg.chunk_size, cfg.precision, rng);
    write_column_matrix(db, meta, blk + ".attn1.to_v.weight", hidden, hidden, cfg.chunk_size, cfg.precision, rng);
    write_column_matrix(db, meta, blk + ".attn1.to_out.weight", hidden, hidden, cfg.chunk_size, cfg.precision, rng);
    write_bias_vector(db, meta, blk + ".attn1.to_out.bias", hidden, cfg.chunk_size, rng, cfg.precision);

    write_column_matrix(db, meta, blk + ".attn2.to_q.weight", hidden, hidden, cfg.chunk_size, cfg.precision, rng);
    write_column_matrix(db, meta, blk + ".attn2.to_k.weight", cross_dim, hidden, cfg.chunk_size, cfg.precision, rng);
    write_column_matrix(db, meta, blk + ".attn2.to_v.weight", cross_dim, hidden, cfg.chunk_size, cfg.precision, rng);
    write_column_matrix(db, meta, blk + ".attn2.to_out.weight", hidden, hidden, cfg.chunk_size, cfg.precision, rng);
    write_bias_vector(db, meta, blk + ".attn2.to_out.bias", hidden, cfg.chunk_size, rng, cfg.precision);

    int inner = hidden * 4;
    int proj_out = hidden * 8;
    write_column_matrix(db, meta, blk + ".ff.proj.weight", hidden, proj_out, cfg.chunk_size, cfg.precision, rng);
    write_bias_vector(db, meta, blk + ".ff.proj.bias", proj_out, cfg.chunk_size, rng, cfg.precision);
    write_column_matrix(db, meta, blk + ".ff.out.weight", inner, hidden, cfg.chunk_size, cfg.precision, rng);
    write_bias_vector(db, meta, blk + ".ff.out.bias", hidden, cfg.chunk_size, rng, cfg.precision);
}

void write_resnet(SimpleDBWriter& db, MetadataWriter& meta,
                  const std::string& prefix,
                  int in_c, int out_c, int time_embed_dim, bool shortcut,
                  const Config& cfg,
                  std::mt19937& rng) {
    write_group_norm(db, meta, prefix + ".norm1", in_c, cfg);
    write_conv(db, meta, prefix + ".conv1", out_c, in_c, 3, cfg, rng);
    write_linear(db, meta, prefix + ".time_emb_proj", time_embed_dim, out_c, cfg, rng);
    write_group_norm(db, meta, prefix + ".norm2", out_c, cfg);
    write_conv(db, meta, prefix + ".conv2", out_c, out_c, 3, cfg, rng);
    if (shortcut) {
        write_conv(db, meta, prefix + ".conv_shortcut", out_c, in_c, 1, cfg, rng);
    }
}

void generate_unet(SimpleDBWriter& db, MetadataWriter& meta,
                   const Config& cfg, std::mt19937& rng) {
    if (cfg.unet_down_blocks != 4 || cfg.unet_mid_blocks != 1 || cfg.unet_up_blocks != 4) {
        throw std::runtime_error("UNet block counts must be down=4, mid=1, up=4 for SD1.5");
    }

    int base = cfg.unet_hidden;
    int time_dim = base;
    int time_embed_dim = base * 4;
    int cross_dim = cfg.text_hidden;

    write_conv(db, meta, "unet.conv_in", base, cfg.latent_channels, 3, cfg, rng);
    write_linear(db, meta, "unet.time_embedding.linear_1", time_dim, time_embed_dim, cfg, rng);
    write_linear(db, meta, "unet.time_embedding.linear_2", time_embed_dim, time_embed_dim, cfg, rng);

    // Down blocks
    write_resnet(db, meta, "unet.down_blocks.0.resnets.0", base, base, time_embed_dim, false, cfg, rng);
    write_transformer2d(db, meta, "unet.down_blocks.0.attentions.0", base, cross_dim, cfg, rng);
    write_resnet(db, meta, "unet.down_blocks.0.resnets.1", base, base, time_embed_dim, false, cfg, rng);
    write_transformer2d(db, meta, "unet.down_blocks.0.attentions.1", base, cross_dim, cfg, rng);
    write_conv(db, meta, "unet.down_blocks.0.downsamplers.0.conv", base, base, 3, cfg, rng);

    write_resnet(db, meta, "unet.down_blocks.1.resnets.0", base, base * 2, time_embed_dim, true, cfg, rng);
    write_transformer2d(db, meta, "unet.down_blocks.1.attentions.0", base * 2, cross_dim, cfg, rng);
    write_resnet(db, meta, "unet.down_blocks.1.resnets.1", base * 2, base * 2, time_embed_dim, false, cfg, rng);
    write_transformer2d(db, meta, "unet.down_blocks.1.attentions.1", base * 2, cross_dim, cfg, rng);
    write_conv(db, meta, "unet.down_blocks.1.downsamplers.0.conv", base * 2, base * 2, 3, cfg, rng);

    write_resnet(db, meta, "unet.down_blocks.2.resnets.0", base * 2, base * 4, time_embed_dim, true, cfg, rng);
    write_transformer2d(db, meta, "unet.down_blocks.2.attentions.0", base * 4, cross_dim, cfg, rng);
    write_resnet(db, meta, "unet.down_blocks.2.resnets.1", base * 4, base * 4, time_embed_dim, false, cfg, rng);
    write_transformer2d(db, meta, "unet.down_blocks.2.attentions.1", base * 4, cross_dim, cfg, rng);
    write_conv(db, meta, "unet.down_blocks.2.downsamplers.0.conv", base * 4, base * 4, 3, cfg, rng);

    write_resnet(db, meta, "unet.down_blocks.3.resnets.0", base * 4, base * 4, time_embed_dim, false, cfg, rng);
    write_resnet(db, meta, "unet.down_blocks.3.resnets.1", base * 4, base * 4, time_embed_dim, false, cfg, rng);

    // Mid block
    write_resnet(db, meta, "unet.mid_block.resnets.0", base * 4, base * 4, time_embed_dim, false, cfg, rng);
    write_transformer2d(db, meta, "unet.mid_block.attentions.0", base * 4, cross_dim, cfg, rng);
    write_resnet(db, meta, "unet.mid_block.resnets.1", base * 4, base * 4, time_embed_dim, false, cfg, rng);

    // Up blocks
    write_resnet(db, meta, "unet.up_blocks.0.resnets.0", base * 8, base * 4, time_embed_dim, true, cfg, rng);
    write_resnet(db, meta, "unet.up_blocks.0.resnets.1", base * 8, base * 4, time_embed_dim, true, cfg, rng);
    write_resnet(db, meta, "unet.up_blocks.0.resnets.2", base * 8, base * 4, time_embed_dim, true, cfg, rng);
    write_conv(db, meta, "unet.up_blocks.0.upsamplers.0.conv", base * 4, base * 4, 3, cfg, rng);

    write_resnet(db, meta, "unet.up_blocks.1.resnets.0", base * 8, base * 4, time_embed_dim, true, cfg, rng);
    write_transformer2d(db, meta, "unet.up_blocks.1.attentions.0", base * 4, cross_dim, cfg, rng);
    write_resnet(db, meta, "unet.up_blocks.1.resnets.1", base * 8, base * 4, time_embed_dim, true, cfg, rng);
    write_transformer2d(db, meta, "unet.up_blocks.1.attentions.1", base * 4, cross_dim, cfg, rng);
    write_resnet(db, meta, "unet.up_blocks.1.resnets.2", base * 6, base * 4, time_embed_dim, true, cfg, rng);
    write_transformer2d(db, meta, "unet.up_blocks.1.attentions.2", base * 4, cross_dim, cfg, rng);
    write_conv(db, meta, "unet.up_blocks.1.upsamplers.0.conv", base * 4, base * 4, 3, cfg, rng);

    write_resnet(db, meta, "unet.up_blocks.2.resnets.0", base * 6, base * 2, time_embed_dim, true, cfg, rng);
    write_transformer2d(db, meta, "unet.up_blocks.2.attentions.0", base * 2, cross_dim, cfg, rng);
    write_resnet(db, meta, "unet.up_blocks.2.resnets.1", base * 4, base * 2, time_embed_dim, true, cfg, rng);
    write_transformer2d(db, meta, "unet.up_blocks.2.attentions.1", base * 2, cross_dim, cfg, rng);
    write_resnet(db, meta, "unet.up_blocks.2.resnets.2", base * 3, base * 2, time_embed_dim, true, cfg, rng);
    write_transformer2d(db, meta, "unet.up_blocks.2.attentions.2", base * 2, cross_dim, cfg, rng);
    write_conv(db, meta, "unet.up_blocks.2.upsamplers.0.conv", base * 2, base * 2, 3, cfg, rng);

    write_resnet(db, meta, "unet.up_blocks.3.resnets.0", base * 3, base, time_embed_dim, true, cfg, rng);
    write_transformer2d(db, meta, "unet.up_blocks.3.attentions.0", base, cross_dim, cfg, rng);
    write_resnet(db, meta, "unet.up_blocks.3.resnets.1", base * 2, base, time_embed_dim, true, cfg, rng);
    write_transformer2d(db, meta, "unet.up_blocks.3.attentions.1", base, cross_dim, cfg, rng);
    write_resnet(db, meta, "unet.up_blocks.3.resnets.2", base * 2, base, time_embed_dim, true, cfg, rng);
    write_transformer2d(db, meta, "unet.up_blocks.3.attentions.2", base, cross_dim, cfg, rng);

    write_group_norm(db, meta, "unet.conv_norm_out", base, cfg);
    write_conv(db, meta, "unet.conv_out", cfg.latent_channels, base, 3, cfg, rng);
}

void append_resnet_ids(std::vector<std::string>& ids, const std::string& prefix, bool shortcut) {
    ids.push_back(prefix + ".norm1.weight");
    ids.push_back(prefix + ".norm1.bias");
    ids.push_back(prefix + ".conv1.weight");
    ids.push_back(prefix + ".conv1.bias");
    ids.push_back(prefix + ".time_emb_proj.weight");
    ids.push_back(prefix + ".time_emb_proj.bias");
    ids.push_back(prefix + ".norm2.weight");
    ids.push_back(prefix + ".norm2.bias");
    ids.push_back(prefix + ".conv2.weight");
    ids.push_back(prefix + ".conv2.bias");
    if (shortcut) {
        ids.push_back(prefix + ".conv_shortcut.weight");
        ids.push_back(prefix + ".conv_shortcut.bias");
    }
}

void append_attention_ids(std::vector<std::string>& ids, const std::string& prefix) {
    ids.push_back(prefix + ".group_norm.weight");
    ids.push_back(prefix + ".group_norm.bias");
    ids.push_back(prefix + ".attn_qkv_proj");
    ids.push_back(prefix + ".attn_out_proj");
}

void generate_vae(SimpleDBWriter& db, MetadataWriter& meta, const Config& cfg, std::mt19937& rng) {
    int in_channels = cfg.latent_channels;
    int out_channels = cfg.vae_hidden;
    for (int i = 0; i < cfg.vae_conv_layers; ++i) {
        if (i == cfg.vae_conv_layers - 1) {
            out_channels = cfg.image_channels;
        }
        std::string prefix = "vae.decoder.conv" + std::to_string(i);
        write_column_matrix(db, meta, prefix + ".conv_weight", out_channels, in_channels,
                            cfg.chunk_size, cfg.precision, rng);
        write_bias_vector(db, meta, prefix + ".conv_bias", out_channels, cfg.chunk_size, rng, cfg.precision);
        in_channels = out_channels;
    }
}

std::string default_graph_path(const Config& cfg) {
    std::filesystem::path base(cfg.db_path);
    return (base / "prefetch_graph.txt").string();
}

void write_prefetch_graph(const Config& cfg) {
    std::string graph_path = cfg.graph_path.empty() ? default_graph_path(cfg) : cfg.graph_path;
    std::filesystem::create_directories(std::filesystem::path(graph_path).parent_path());
    std::ofstream out(graph_path);
    if (!out) {
        throw std::runtime_error("Failed to open prefetch graph file: " + graph_path);
    }
    std::cout << "Writing prefetch graph to " << graph_path << std::endl;

    out << "# Stable Diffusion prefetch graph generated with weights\n";
    out << "# text blocks=" << cfg.text_layers
        << " unet down/mid/up=" << cfg.unet_down_blocks << "/"
        << cfg.unet_mid_blocks << "/" << cfg.unet_up_blocks
        << " vae convs=" << cfg.vae_conv_layers
        << " steps=" << cfg.denoise_steps << "\n";

    if (cfg.text_layers > 0) {
        out << "start text.block.0.norm1_weight\n";
        for (int i = 0; i < cfg.text_layers; ++i) {
            std::string prefix = "text.block." + std::to_string(i);
            out << "link " << prefix << ".norm1_weight " << prefix << ".attn_qkv_proj 0\n";
            out << "link " << prefix << ".attn_qkv_proj " << prefix << ".attn_out_proj 0\n";
            out << "link " << prefix << ".attn_out_proj " << prefix << ".norm2_weight 0\n";
            out << "link " << prefix << ".norm2_weight " << prefix << ".ffn_gate_up_proj 0\n";
            out << "link " << prefix << ".ffn_gate_up_proj " << prefix << ".ffn_down_proj 0\n";
            if (i + 1 < cfg.text_layers) {
                out << "link " << prefix << ".ffn_down_proj text.block." << (i + 1)
                    << ".norm1_weight 0\n";
            } else {
                out << "link " << prefix << ".ffn_down_proj text.final_norm_weight 0\n";
            }
        }
        out << "link text.final_norm_weight vae.decoder.conv0.conv_weight 0\n";
    } else {
        out << "start vae.decoder.conv0.conv_weight\n";
    }

    for (int i = 0; i < cfg.vae_conv_layers; ++i) {
        std::string prefix = "vae.decoder.conv" + std::to_string(i);
        out << "link " << prefix << ".conv_weight " << prefix << ".conv_bias 0\n";
        if (i + 1 < cfg.vae_conv_layers) {
            out << "link " << prefix << ".conv_bias vae.decoder.conv" << (i + 1) << ".conv_weight 0\n";
        }
    }
}

void generate_weights(const Config& cfg) {
    SimpleDBWriter db(cfg.db_path);
    MetadataWriter meta(cfg.db_path);

    std::mt19937 rng(cfg.seed);

    std::cout << "Creating Stable Diffusion weights in SimpleKVStore..." << std::endl;
    std::cout << "  Path: " << cfg.db_path << std::endl;
    std::cout << "  Chunk size: " << cfg.chunk_size << std::endl;

    generate_text_encoder(db, meta, cfg, rng);
    generate_unet(db, meta, cfg, rng);
    generate_vae(db, meta, cfg, rng);
    write_prefetch_graph(cfg);

    db.flush_index();
    std::cout << "Stable Diffusion weight generation complete." << std::endl;
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
        } else if (arg == "--graph-path") {
            cfg.graph_path = argv[++i];
        } else if (arg == "--chunk-size") {
            read_int(cfg.chunk_size);
        } else if (arg == "--seed") {
            read_int(cfg.seed);
        } else if (arg == "--precision") {
            cfg.precision = parse_precision(argv[++i]);
        } else if (arg == "--text-hidden") {
            read_int(cfg.text_hidden);
        } else if (arg == "--text-layers") {
            read_int(cfg.text_layers);
        } else if (arg == "--text-vocab-size") {
            read_int(cfg.text_vocab_size);
        } else if (arg == "--text-ffn-mult") {
            read_int(cfg.text_ffn_mult);
        } else if (arg == "--unet-hidden") {
            read_int(cfg.unet_hidden);
        } else if (arg == "--unet-down-blocks") {
            read_int(cfg.unet_down_blocks);
        } else if (arg == "--unet-mid-blocks") {
            read_int(cfg.unet_mid_blocks);
        } else if (arg == "--unet-up-blocks") {
            read_int(cfg.unet_up_blocks);
        } else if (arg == "--unet-ffn-mult") {
            read_int(cfg.unet_ffn_mult);
        } else if (arg == "--latent-channels") {
            read_int(cfg.latent_channels);
        } else if (arg == "--vae-hidden") {
            read_int(cfg.vae_hidden);
        } else if (arg == "--image-channels") {
            read_int(cfg.image_channels);
        } else if (arg == "--vae-conv-layers") {
            read_int(cfg.vae_conv_layers);
        } else if (arg == "--steps") {
            read_int(cfg.denoise_steps);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --db-path PATH\n"
                      << "  --graph-path PATH\n"
                      << "  --chunk-size N\n"
                      << "  --seed N\n"
                      << "  --precision float32|int8|bfloat16\n"
                      << "  --text-hidden N\n"
                      << "  --text-layers N\n"
                      << "  --text-vocab-size N\n"
                      << "  --text-ffn-mult N\n"
                      << "  --unet-hidden N\n"
                      << "  --unet-down-blocks N\n"
                      << "  --unet-mid-blocks N\n"
                      << "  --unet-up-blocks N\n"
                      << "  --unet-ffn-mult N\n"
                      << "  --latent-channels N\n"
                      << "  --vae-hidden N\n"
                      << "  --image-channels N\n"
                      << "  --vae-conv-layers N\n"
                      << "  --steps N\n";
            std::exit(0);
        }
    }
    if (cfg.chunk_size <= 0) {
        throw std::runtime_error("chunk_size must be > 0");
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

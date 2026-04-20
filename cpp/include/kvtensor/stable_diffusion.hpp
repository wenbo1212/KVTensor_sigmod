#pragma once

#include "kvtensor/context.hpp"
#include "kvtensor/sd_ops.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kvtensor {

// Text encoder (CLIP) config
struct SDTextEncoderConfig {
    std::string token_embedding_id;
    std::vector<std::string> transformer_block_ids; // optional placeholder
    std::string final_layer_norm_id;
    int64_t vocab_size{49408};
    int64_t hidden_size{768};
    int64_t max_length{77};
    DType output_dtype{DType::FLOAT32};
};

// UNet config
struct SDUNetConfig {
    std::vector<std::string> down_block_weight_ids;
    std::vector<std::string> mid_block_weight_ids;
    std::vector<std::string> up_block_weight_ids;
    std::string input_conv_weight_id;
    std::string input_conv_bias_id;
    std::string output_conv_weight_id;
    std::string output_conv_bias_id;
    int64_t latent_channels{4};
    int64_t model_channels{320};
    int64_t sample_height{64}; // latent H (for 512px images)
    int64_t sample_width{64};  // latent W (for 512px images)
    DType output_dtype{DType::FLOAT32};
};

// VAE decoder config
struct SDVAEConfig {
    std::vector<std::string> decoder_weight_ids;
    int64_t latent_channels{4};
    int64_t scale_factor{8}; // latent -> image upsample factor
    int64_t image_channels{3};
    int64_t image_size{512};
    DType output_dtype{DType::FLOAT32};
};

// Scheduler config (DDIM/PLMS placeholder)
struct SDSchedulerConfig {
    int64_t num_inference_steps{20};
    float guidance_scale{7.5f};
    uint64_t seed{1234};
};

// Pipeline config
struct StableDiffusionConfig {
    SDTextEncoderConfig text_encoder;
    SDUNetConfig unet;
    SDVAEConfig vae;
    SDSchedulerConfig scheduler;
    size_t arena_size_mb{512};
    size_t prefetch_window{64};
    std::string preload_file_path;
};

class SDTextEncoder {
public:
    explicit SDTextEncoder(const SDTextEncoderConfig& config) : config_(config) {}
    std::shared_ptr<InMemoryMatrix> forward(
        const std::vector<int32_t>& input_ids,
        const std::string& result_id,
        OperatorContext& ctx,
        bool profile = false,
        const std::string& input_embeddings_id = ""
    );

private:
    SDTextEncoderConfig config_;
};

class SDUNet {
public:
    explicit SDUNet(const SDUNetConfig& config) : config_(config) {}
    std::shared_ptr<InMemoryMatrix> forward(
        const std::string& latents_id,
        const std::string& text_embeddings_id,
        const std::string& result_id,
        OperatorContext& ctx,
        int64_t timestep,
        bool profile = false
    );

private:
    SDUNetConfig config_;
};

class SDVAE {
public:
    explicit SDVAE(const SDVAEConfig& config) : config_(config) {}
    std::shared_ptr<InMemoryMatrix> decode(
        const std::string& latents_id,
        const std::string& result_id,
        OperatorContext& ctx,
        bool profile = false
    );

private:
    SDVAEConfig config_;
};

class StableDiffusionPipeline {
public:
    explicit StableDiffusionPipeline(const StableDiffusionConfig& config);

    std::shared_ptr<InMemoryMatrix> generate(
        const std::string& prompt,
        const std::string& result_id,
        OperatorContext& ctx,
        bool profile = false
    );

    const StableDiffusionConfig& config() const { return config_; }

private:
    std::shared_ptr<InMemoryMatrix> encode_prompt(
        const std::string& prompt,
        const std::string& result_id,
        OperatorContext& ctx,
        bool profile
    );

    std::shared_ptr<InMemoryMatrix> run_denoising_loop(
        const std::string& text_emb_id,
        const std::string& latents_id,
        const std::string& result_id,
        OperatorContext& ctx,
        bool profile
    );

    std::shared_ptr<InMemoryMatrix> decode_latents(
        const std::string& latents_id,
        const std::string& result_id,
        OperatorContext& ctx,
        bool profile
    );

    StableDiffusionConfig config_;
    SDTextEncoder text_encoder_;
    SDUNet unet_;
    SDVAE vae_;
};

} // namespace kvtensor

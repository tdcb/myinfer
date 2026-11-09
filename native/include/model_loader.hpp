#pragma once

#include "model_config.hpp"
#include "memory_arena.hpp"
#include "tokenizer.hpp"
#include <string>

namespace m5engine {

class ModelLoader {
public:
    // Parse HuggingFace config.json and auto-detect Dense vs. MoE architecture
    static ModelConfig parse_config_json(const std::string& json_path);

    // Built-in architectural profiles for Qwen Dense & Qwen MoE on Apple M5 Pro (64GB)
    static ModelConfig make_preset_config(const std::string& preset_name);

    // Load from a HuggingFace / MLX directory (containing config.json + .safetensors)
    // or a zero-copy .m5e model file, or synthesize a calibrated model in Unified Memory
    static ModelWeights load_or_build_model(
        const std::string& path_or_preset,
        UnifiedMemoryArena& weight_arena,
        QwenTokenizer& tokenizer
    );

    // Save quantized weights to a 16KB page-aligned .m5e file for instant zero-copy mmap startup
    static bool save_zero_copy_m5e(const std::string& output_path, const ModelWeights& weights);

    // Load a 16KB page-aligned .m5e file via zero-copy mmap -> MTLBuffer
    static ModelWeights load_zero_copy_m5e(const std::string& m5e_path, UnifiedMemoryArena& weight_arena);
};

} // namespace m5engine

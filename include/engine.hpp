#pragma once

#include "hardware_m5.hpp"
#include "memory_arena.hpp"
#include "metal_backend.hpp"
#include "kv_cache.hpp"
#include "model_config.hpp"
#include "tokenizer.hpp"
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace m5engine {

struct GenerationParams {
    uint32_t max_tokens = 128;
    float temperature = 0.0f;       // 0.0 = greedy / exact MTP verification
    float top_p = 0.9f;
    bool use_prefix_cache = true;
    bool use_speculative_mtp = true; // Used automatically when model is Dense
};

struct GenerationMetrics {
    std::string model_id;
    std::string arch_type;
    uint32_t prompt_tokens = 0;
    uint32_t prefix_cached_tokens = 0;
    uint32_t prefilled_tokens = 0;
    uint32_t generated_tokens = 0;

    double ttft_ms = 0.0;                 // Time To First Token (wall clock ms)
    double decode_ms = 0.0;               // Total decode time (wall clock ms)
    double total_gpu_ms = 0.0;            // Pure hardware GPU execution time (MTLCommandBuffer)
    double prefill_tokens_per_sec = 0.0;
    double decode_tokens_per_sec = 0.0;
    double effective_bw_gbps = 0.0;       // Effective Unified Memory bandwidth achieved (GB/s)

    // Adaptive Dense MTP / Speculative telemetry
    uint32_t forward_passes = 0;
    uint32_t speculative_drafted = 0;
    uint32_t speculative_accepted = 0;
    double mean_tokens_per_pass = 1.0;

    // Adaptive MoE telemetry
    uint32_t active_experts_per_tok = 0;
    uint32_t total_experts = 0;
    uint32_t pinned_hot_experts = 0;
    std::vector<uint32_t> top_routed_experts;

    ThermalLevel thermal_state = ThermalLevel::Nominal;
};

struct GenerationResult {
    std::string text;
    std::vector<int32_t> token_ids;
    GenerationMetrics metrics;
};

using StreamTokenCallback = std::function<bool(const std::string& token_piece, int32_t token_id, bool is_final, const GenerationMetrics& metrics)>;

class M5InferenceEngine {
public:
    explicit M5InferenceEngine(const std::string& model_path_or_preset = "dense");
    ~M5InferenceEngine() = default;

    // Switch or reload model dynamically (e.g., between Dense and MoE presets or paths)
    void load_model(const std::string& model_path_or_preset);

    // Lifecycle control from UI: unload/stop engine (freeing 100% of GPU Unified Memory) or bring it back up
    void unload_model();
    void start_engine(const std::string& model_path_or_preset = "");
    bool is_loaded() const { return engine_loaded_; }
    const std::string& current_model_spec() const { return last_model_spec_; }

    // Run full generation and return result + hardware telemetry
    GenerationResult generate(
        const std::vector<ChatMessage>& messages,
        const GenerationParams& params = GenerationParams{},
        StreamTokenCallback stream_cb = nullptr
    );

    GenerationResult generate_from_prompt(
        const std::string& raw_prompt,
        const GenerationParams& params = GenerationParams{},
        StreamTokenCallback stream_cb = nullptr
    );

    const M5HardwareProfile& hardware_profile() const { return hw_profile_; }
    const ModelConfig& model_config() const { return weights_.config; }
    const QwenTokenizer& tokenizer() const { return tokenizer_; }
    const GenerationMetrics& last_metrics() const { return last_metrics_; }
    PagedRadixKVCache* kv_cache() const { return kv_cache_.get(); }

    size_t weight_memory_bytes() const { return weights_.total_weight_bytes; }
    size_t total_unified_memory_bytes() const;

    void clear_prefix_cache();

private:
    std::mutex engine_mutex_;
    M5HardwareProfile hw_profile_;
    MetalBackend backend_;
    std::unique_ptr<UnifiedMemoryArena> weight_arena_;
    std::unique_ptr<UnifiedMemoryArena> activation_arena_;
    std::unique_ptr<PagedRadixKVCache> kv_cache_;
    QwenTokenizer tokenizer_;
    ModelWeights weights_;
    GenerationMetrics last_metrics_;
    bool engine_loaded_ = false;
    std::string last_model_spec_ = "dense";

    // Pre-allocated zero-copy activation scratch buffers in Unified Memory
    // Sized for max_batch_tokens (e.g. 512 tokens prefill / 8 tokens MTP verify)
    uint32_t max_batch_tokens_ = 512;
    BufferSlice act_token_ids_;      // INT32 [max_batch_tokens]
    BufferSlice act_block_table_;    // INT32 [max_blocks]
    BufferSlice act_x_;              // FP16 [max_batch_tokens, hidden_size]
    BufferSlice act_normed_;         // FP16 [max_batch_tokens, hidden_size]
    BufferSlice act_q_;              // FP16 [max_batch_tokens, q_dim]
    BufferSlice act_k_;              // FP16 [max_batch_tokens, kv_dim]
    BufferSlice act_v_;              // FP16 [max_batch_tokens, kv_dim]
    BufferSlice act_attn_out_;       // FP16 [max_batch_tokens, q_dim]
    BufferSlice act_proj_out_;       // FP16 [max_batch_tokens, hidden_size]
    BufferSlice act_mlp_interm_;     // FP16 [max_batch_tokens, intermediate_size]
    BufferSlice act_logits_;         // FP16 [max_batch_tokens, vocab_size]
    BufferSlice act_argmax_tokens_;  // INT32 [max_batch_tokens]

    // MoE-specific zero-copy device buffers (zero CPU-GPU sync!)
    BufferSlice moe_topk_indices_;   // INT32 [max_batch_tokens, top_k]
    BufferSlice moe_topk_weights_;   // FP16 [max_batch_tokens, top_k]
    BufferSlice moe_interm_;         // FP16 [max_batch_tokens * top_k, moe_intermediate_size]
    BufferSlice moe_expert_counts_;  // UINT32 [num_experts]

    void allocate_activation_buffers();

    // Execute a single batched forward pass across all layers for M tokens starting at start_pos
    // Writes argmax token predictions for all M positions into act_argmax_tokens_
    double forward_pass_batched(
        const int32_t* input_tokens,
        uint32_t M,
        uint32_t start_pos,
        const std::vector<int32_t>& block_table
    );

    // Propose K draft tokens for Dense Speculative / Multi-Token Prediction (MTP)
    std::vector<int32_t> propose_dense_mtp_draft_tokens(
        int32_t last_token,
        const std::vector<int32_t>& context_tokens,
        uint32_t K
    );

    // Update MoE expert Unified Memory pinning based on GPU router telemetry
    void update_moe_expert_residency(GenerationMetrics& metrics);
};

} // namespace m5engine

#pragma once

#include "tensor.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace m5engine {

enum class ModelArchType : uint8_t {
    Dense = 0,       // Standard Dense Transformer (e.g., Qwen 27B Dense, Qwen2.5/3 Dense)
    MoE = 1,         // Sparse Mixture-of-Experts (e.g., Qwen-MoE, Qwen3-30B-A3B, Mixtral)
    HybridMoE = 2    // Interleaved Dense + MoE layers
};

inline const char* arch_type_name(ModelArchType t) {
    switch (t) {
        case ModelArchType::Dense:     return "Dense Transformer";
        case ModelArchType::MoE:       return "Mixture-of-Experts (MoE)";
        case ModelArchType::HybridMoE: return "Hybrid Dense+MoE";
    }
    return "Unknown";
}

struct ModelConfig {
    std::string model_id = "qwen3.8-27b-m5-q4";
    std::string architecture = "QwenForCausalLM";
    ModelArchType arch_type = ModelArchType::Dense;

    uint32_t vocab_size = 32768;
    uint32_t hidden_size = 5120;
    uint32_t intermediate_size = 13824;
    uint32_t num_layers = 48;
    uint32_t num_attention_heads = 40;
    uint32_t num_key_value_heads = 8;
    uint32_t head_dim = 128;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    uint32_t max_seq_len = 8192;

    // MoE parameters (active when arch_type == MoE or HybridMoE)
    uint32_t num_experts = 0;
    uint32_t num_experts_per_tok = 0; // Top-K active experts per token
    uint32_t moe_intermediate_size = 0;
    bool has_shared_expert = false;
    uint32_t shared_expert_intermediate_size = 0;

    // Adaptive Execution Parameters tuned for Apple M5 Pro
    bool enable_speculative_mtp = true; // Multi-token prediction / Prompt-Lookup + Draft verification for Dense
    uint32_t speculative_draft_k = 4;   // Verify 1 + 4 = 5 tokens per single weight-read pass on Dense
    bool pin_hot_moe_experts = true;    // Pin router, shared expert & hot experts in M5 Pro Unified Memory

    uint64_t approximate_total_params() const;
    uint64_t approximate_active_params_per_token() const;
};

struct TransformerLayerWeights {
    bool is_moe_layer = false;

    // Attention Norms & Projections
    BufferSlice input_layernorm;         // FP16 [hidden_size]
    QuantizedWeight q_proj;              // Q4_G64 [num_heads * head_dim, hidden_size]
    QuantizedWeight k_proj;              // Q4_G64 [num_kv_heads * head_dim, hidden_size]
    QuantizedWeight v_proj;              // Q4_G64 [num_kv_heads * head_dim, hidden_size]
    QuantizedWeight o_proj;              // Q4_G64 [hidden_size, num_heads * head_dim]
    BufferSlice post_attention_layernorm;// FP16 [hidden_size]

    // Dense MLP (or Shared Expert in Qwen-MoE)
    bool has_dense_or_shared_mlp = true;
    QuantizedWeight gate_proj;           // Q4_G64 [inter_dim, hidden_size]
    QuantizedWeight up_proj;             // Q4_G64 [inter_dim, hidden_size]
    QuantizedWeight down_proj;           // Q4_G64 [hidden_size, inter_dim]

    // Sparse Routed MoE Experts
    BufferSlice router_gate;             // FP16 [num_experts, hidden_size]
    QuantizedWeight experts_gate_proj;   // Q4_G64 [num_experts * moe_inter_dim, hidden_size]
    QuantizedWeight experts_up_proj;     // Q4_G64 [num_experts * moe_inter_dim, hidden_size]
    QuantizedWeight experts_down_proj;   // Q4_G64 [num_experts * hidden_size, moe_inter_dim]
};

struct ModelWeights {
    ModelConfig config;
    QuantizedWeight embed_tokens;        // Q4_G64 [vocab_size, hidden_size]
    std::vector<TransformerLayerWeights> layers;
    BufferSlice final_norm;              // FP16 [hidden_size]
    QuantizedWeight lm_head;             // Q4_G64 [vocab_size, hidden_size] (can share embed_tokens)

    // Optional lightweight MTP / Draft Head for Dense Speculative Decoding
    bool has_mtp_draft_head = false;
    BufferSlice mtp_norm;                // FP16 [hidden_size]
    QuantizedWeight mtp_proj;            // Q4_G64 [hidden_size, hidden_size]

    size_t total_weight_bytes = 0;
};

} // namespace m5engine

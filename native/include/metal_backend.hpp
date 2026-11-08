#pragma once

#include "tensor.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>

namespace m5engine {

class MetalBackend {
public:
    explicit MetalBackend(const std::string& shader_dir_or_path = "src/kernels.metal");
    ~MetalBackend();

    MetalBackend(const MetalBackend&) = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;

    void* device_handle() const { return device_; }
    std::string device_name() const { return device_name_; }

    // Begin recording a single batched command buffer & compute encoder across all layers
    void begin_batch();

    // Commit the recorded command buffer and block via zero-spin Darwin semaphore
    // Returns GPU hardware execution time in milliseconds (via GPUStartTime / GPUEndTime)
    double commit_and_wait();

    // Kernel dispatch methods (recorded into active single command encoder)
    void encode_embedding_q4_g64(
        const BufferSlice& token_ids,
        const QuantizedWeight& emb_weight,
        const BufferSlice& out_f16,
        uint32_t M,
        uint32_t dim
    );

    void encode_rmsnorm_f16(
        const BufferSlice& x,
        const BufferSlice& weight,
        const BufferSlice& out,
        uint32_t M,
        uint32_t dim,
        float eps
    );

    void encode_fused_add_rmsnorm_f16(
        const BufferSlice& x_inout,
        const BufferSlice& residual,
        const BufferSlice& weight,
        const BufferSlice& normed_out,
        uint32_t M,
        uint32_t dim,
        float eps
    );

    // Automatically chooses between 128B coalesced SIMD-group GEMV/batched-GEMM (M <= 16)
    // and M5 Pro hardware simdgroup_matrix<half, 8, 8> MMA kernel (M > 16)
    void encode_linear_q4_g64(
        const BufferSlice& x,
        const QuantizedWeight& weight,
        const BufferSlice& out,
        uint32_t M
    );

    void encode_linear_q4_g64_residual_add(
        const BufferSlice& x,
        const QuantizedWeight& weight,
        const BufferSlice& residual_inout,
        uint32_t M
    );

    void encode_fused_gate_up_swiglu_q4_g64(
        const BufferSlice& x,
        const QuantizedWeight& gate_weight,
        const QuantizedWeight& up_weight,
        const BufferSlice& out,
        uint32_t M
    );

    void encode_qkv_rope_paged_kv(
        const BufferSlice& q_inout,
        const BufferSlice& k_in,
        const BufferSlice& v_in,
        const BufferSlice& k_cache,
        const BufferSlice& v_cache,
        const BufferSlice& block_table,
        uint32_t M,
        uint32_t num_q_heads,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t block_size,
        uint32_t start_pos,
        float rope_theta
    );

    void encode_flash_attn_paged_gqa(
        const BufferSlice& q_in,
        const BufferSlice& k_cache,
        const BufferSlice& v_cache,
        const BufferSlice& block_table,
        const BufferSlice& out,
        uint32_t M,
        uint32_t num_q_heads,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t block_size,
        uint32_t start_pos
    );

    // Adaptive MoE Kernels (Zero CPU-GPU sync!)
    void encode_moe_router_topk_softmax(
        const BufferSlice& x,
        const BufferSlice& router_weight_f16,
        const BufferSlice& topk_indices_out,
        const BufferSlice& topk_weights_out,
        const BufferSlice& expert_hit_counts,
        uint32_t M,
        uint32_t hidden_dim,
        uint32_t num_experts,
        uint32_t top_k
    );

    void encode_moe_fused_gate_up_swiglu_q4_g64(
        const BufferSlice& x,
        const BufferSlice& topk_indices,
        const QuantizedWeight& experts_gate_weight,
        const QuantizedWeight& experts_up_weight,
        const BufferSlice& moe_interm_out,
        uint32_t M,
        uint32_t top_k,
        uint32_t inter_dim,
        uint32_t hidden_dim
    );

    void encode_moe_down_accumulate_q4_g64(
        const BufferSlice& moe_interm,
        const BufferSlice& topk_indices,
        const BufferSlice& topk_weights,
        const QuantizedWeight& experts_down_weight,
        const BufferSlice& residual_inout,
        uint32_t M,
        uint32_t top_k,
        uint32_t hidden_dim,
        uint32_t inter_dim
    );

    void encode_argmax_f16(
        const BufferSlice& logits,
        const BufferSlice& out_tokens,
        uint32_t M,
        uint32_t vocab_size
    );

private:
    void* device_ = nullptr;        // id<MTLDevice>
    void* command_queue_ = nullptr; // id<MTLCommandQueue>
    void* library_ = nullptr;       // id<MTLLibrary>
    void* active_cmd_buf_ = nullptr; // id<MTLCommandBuffer>
    void* active_encoder_ = nullptr; // id<MTLComputeCommandEncoder>
    std::string device_name_;
    std::unordered_map<std::string, void*> pipelines_; // name -> id<MTLComputePipelineState>

    void* get_pipeline(const char* name);
};

} // namespace m5engine

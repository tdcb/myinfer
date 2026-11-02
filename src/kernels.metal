#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

// ============================================================================
// 1. RMSNorm & Fused Residual + RMSNorm (Batched [M, dim])
// ============================================================================

kernel void rmsnorm_f16(
    device const half* x        [[buffer(0)]],
    device const half* weight   [[buffer(1)]],
    device half* out            [[buffer(2)]],
    constant uint& dim          [[buffer(3)]],
    constant float& eps         [[buffer(4)]],
    uint tgid                   [[threadgroup_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tgs                    [[threads_per_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[32];
    device const half* row_in = x + tgid * dim;
    device half* row_out = out + tgid * dim;

    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tgs) {
        float v = float(row_in[i]);
        sum_sq = fma(v, v, sum_sq);
    }

    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) {
        shared_sum[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd = (tgs + 31) / 32;
    float total = (tid < num_simd) ? shared_sum[tid] : 0.0f;
    if (simd_id == 0) {
        total = simd_sum(total);
        if (simd_lane == 0) {
            shared_sum[0] = rsqrt(total / float(dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_sum[0];
    for (uint i = tid; i < dim; i += tgs) {
        row_out[i] = half(float(row_in[i]) * inv_rms * float(weight[i]));
    }
}

kernel void fused_add_rmsnorm_f16(
    device half* x              [[buffer(0)]], // in-place updated: x += residual
    device const half* residual [[buffer(1)]],
    device const half* weight   [[buffer(2)]],
    device half* normed_out     [[buffer(3)]],
    constant uint& dim          [[buffer(4)]],
    constant float& eps         [[buffer(5)]],
    uint tgid                   [[threadgroup_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tgs                    [[threads_per_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[32];
    device half* row_x = x + tgid * dim;
    device const half* row_res = residual + tgid * dim;
    device half* row_out = normed_out + tgid * dim;

    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tgs) {
        float v = float(row_x[i]) + float(row_res[i]);
        row_x[i] = half(v);
        sum_sq = fma(v, v, sum_sq);
    }

    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) {
        shared_sum[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simd = (tgs + 31) / 32;
    float total = (tid < num_simd) ? shared_sum[tid] : 0.0f;
    if (simd_id == 0) {
        total = simd_sum(total);
        if (simd_lane == 0) {
            shared_sum[0] = rsqrt(total / float(dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_sum[0];
    for (uint i = tid; i < dim; i += tgs) {
        row_out[i] = half(float(row_x[i]) * inv_rms * float(weight[i]));
    }
}

// ============================================================================
// 2. Fused QKV RoPE (NeoX/Qwen split-half) + Paged KV-Cache Scatter Write
// ============================================================================

kernel void qkv_rope_paged_kv_f16(
    device half* q                  [[buffer(0)]], // [M, num_q_heads * head_dim]
    device const half* k            [[buffer(1)]], // [M, num_kv_heads * head_dim]
    device const half* v            [[buffer(2)]], // [M, num_kv_heads * head_dim]
    device half* k_cache            [[buffer(3)]], // [num_blocks, block_size, num_kv_heads, head_dim]
    device half* v_cache            [[buffer(4)]], // [num_blocks, block_size, num_kv_heads, head_dim]
    device const int* block_table   [[buffer(5)]], // [max_blocks_per_seq]
    constant uint& num_q_heads      [[buffer(6)]],
    constant uint& num_kv_heads     [[buffer(7)]],
    constant uint& head_dim         [[buffer(8)]],
    constant uint& block_size       [[buffer(9)]],
    constant uint& start_pos        [[buffer(10)]],
    constant float& rope_theta      [[buffer(11)]],
    uint3 gid                       [[thread_position_in_grid]]
) {
    // gid.x = d in [0, head_dim/2 - 1]
    // gid.y = head_idx in [0, num_q_heads + num_kv_heads - 1]
    // gid.z = token m in [0, M - 1]
    uint half_dim = head_dim / 2;
    if (gid.x >= half_dim) return;

    uint m = gid.z;
    uint pos = start_pos + m;
    uint d = gid.x;

    float freq = 1.0f / pow(rope_theta, float(2 * d) / float(head_dim));
    float angle = float(pos) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    if (gid.y < num_q_heads) {
        uint h = gid.y;
        uint base = (m * num_q_heads + h) * head_dim;
        float q0 = float(q[base + d]);
        float q1 = float(q[base + d + half_dim]);
        q[base + d]            = half(q0 * cos_val - q1 * sin_val);
        q[base + d + half_dim] = half(q1 * cos_val + q0 * sin_val);
    } else if (gid.y < num_q_heads + num_kv_heads) {
        uint kh = gid.y - num_q_heads;
        uint kv_base = (m * num_kv_heads + kh) * head_dim;
        float k0 = float(k[kv_base + d]);
        float k1 = float(k[kv_base + d + half_dim]);

        uint logical_block = pos / block_size;
        uint block_offset  = pos % block_size;
        int phys_block = block_table[logical_block];

        uint cache_base = ((uint(phys_block) * block_size + block_offset) * num_kv_heads + kh) * head_dim;
        k_cache[cache_base + d]            = half(k0 * cos_val - k1 * sin_val);
        k_cache[cache_base + d + half_dim] = half(k1 * cos_val + k0 * sin_val);

        v_cache[cache_base + d]            = v[kv_base + d];
        v_cache[cache_base + d + half_dim] = v[kv_base + d + half_dim];
    }
}

// ============================================================================
// 3. Coalesced 128-Byte Cache-Line Q4_G64 Batched GEMV / Multi-Token GEMM
//    Supports M = 1 (Decode) and M = 2..8 (Speculative MTP Verify & Chunked Prefill)
//    Weights are read ONCE from Unified Memory and multiplied across all M tokens!
// ============================================================================

constant uint kRowsPerTG = 4; // 4 SIMD-groups (128 threads) per threadgroup

kernel void gemm_q4_g64_batched_f16(
    device const half* x            [[buffer(0)]], // [M, K]
    device const uint* qweight      [[buffer(1)]], // [N, K / 8]
    device const half* scales       [[buffer(2)]], // [N, K / 64]
    device const half* biases       [[buffer(3)]], // [N, K / 64]
    device half* out                [[buffer(4)]], // [M, N]
    constant uint& M                [[buffer(5)]],
    constant uint& N                [[buffer(6)]],
    constant uint& K                [[buffer(7)]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]]
) {
    uint row = tgid * kRowsPerTG + simd_id;
    if (row >= N) return;

    // 32 lanes = 4 sub-groups of 8 lanes.
    // Each sub-group of 8 lanes processes 1 group of 64 weights (8 uint32 words = 32 bytes).
    // All 32 lanes together load 128 bytes = 1 full M5 Pro cache line per loop step!
    uint sub_grp  = simd_lane >> 3;   // 0..3
    uint sub_lane = simd_lane & 7u;   // 0..7

    uint num_groups = K >> 6;         // K / 64
    uint u32_per_row = K >> 3;        // K / 8

    device const uint* row_qw = qweight + row * u32_per_row;
    device const half* row_sc = scales  + row * num_groups;
    device const half* row_bi = biases  + row * num_groups;

    // Process up to 8 tokens in registers simultaneously (reads weights only ONCE!)
    for (uint m_base = 0; m_base < M; m_base += 8) {
        uint m_cnt = min(8u, M - m_base);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (uint g = sub_grp; g < num_groups; g += 4) {
            float sc = float(row_sc[g]);
            float bi = float(row_bi[g]);
            uint packed = row_qw[g * 8 + sub_lane];
            uint col = g * 64 + sub_lane * 8;

            // Unpack 8 x 4-bit nibbles into registers once
            float w0 = float((packed >>  0) & 0xFu) * sc + bi;
            float w1 = float((packed >>  4) & 0xFu) * sc + bi;
            float w2 = float((packed >>  8) & 0xFu) * sc + bi;
            float w3 = float((packed >> 12) & 0xFu) * sc + bi;
            float w4 = float((packed >> 16) & 0xFu) * sc + bi;
            float w5 = float((packed >> 20) & 0xFu) * sc + bi;
            float w6 = float((packed >> 24) & 0xFu) * sc + bi;
            float w7 = float((packed >> 28) & 0xFu) * sc + bi;

            for (uint mi = 0; mi < m_cnt; ++mi) {
                device const half* x_ptr = x + (m_base + mi) * K + col;
                float s = w0 * float(x_ptr[0]) + w1 * float(x_ptr[1]) +
                          w2 * float(x_ptr[2]) + w3 * float(x_ptr[3]) +
                          w4 * float(x_ptr[4]) + w5 * float(x_ptr[5]) +
                          w6 * float(x_ptr[6]) + w7 * float(x_ptr[7]);
                acc[mi] += s;
            }
        }

        for (uint mi = 0; mi < m_cnt; ++mi) {
            float sum = simd_sum(acc[mi]);
            if (simd_lane == 0) {
                out[(m_base + mi) * N + row] = half(sum);
            }
        }
    }
}

// ============================================================================
// 4. Fused Dense Gate + Up + SwiGLU Kernel (Q4_G64)
//    Computes out[m, row] = SiLU(x[m] @ W_gate[row]) * (x[m] @ W_up[row])
//    Eliminates 2 intermediate global memory buffers and halves input X reads!
// ============================================================================

kernel void fused_gate_up_swiglu_q4_g64(
    device const half* x            [[buffer(0)]], // [M, K]
    device const uint* gate_qw      [[buffer(1)]], // [N, K / 8]
    device const half* gate_sc      [[buffer(2)]], // [N, K / 64]
    device const half* gate_bi      [[buffer(3)]], // [N, K / 64]
    device const uint* up_qw        [[buffer(4)]], // [N, K / 8]
    device const half* up_sc        [[buffer(5)]], // [N, K / 64]
    device const half* up_bi        [[buffer(6)]], // [N, K / 64]
    device half* out                [[buffer(7)]], // [M, N]
    constant uint& M                [[buffer(8)]],
    constant uint& N                [[buffer(9)]],
    constant uint& K                [[buffer(10)]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]]
) {
    uint row = tgid * kRowsPerTG + simd_id;
    if (row >= N) return;

    uint sub_grp  = simd_lane >> 3;
    uint sub_lane = simd_lane & 7u;
    uint num_groups = K >> 6;
    uint u32_per_row = K >> 3;

    device const uint* g_qw = gate_qw + row * u32_per_row;
    device const half* g_sc = gate_sc + row * num_groups;
    device const half* g_bi = gate_bi + row * num_groups;

    device const uint* u_qw = up_qw   + row * u32_per_row;
    device const half* u_sc = up_sc   + row * num_groups;
    device const half* u_bi = up_bi   + row * num_groups;

    for (uint m_base = 0; m_base < M; m_base += 4) {
        uint m_cnt = min(4u, M - m_base);
        float acc_g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float acc_u[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (uint g = sub_grp; g < num_groups; g += 4) {
            float gsc = float(g_sc[g]);
            float gbi = float(g_bi[g]);
            uint gpack = g_qw[g * 8 + sub_lane];

            float usc = float(u_sc[g]);
            float ubi = float(u_bi[g]);
            uint upack = u_qw[g * 8 + sub_lane];

            uint col = g * 64 + sub_lane * 8;

            for (uint mi = 0; mi < m_cnt; ++mi) {
                device const half* x_ptr = x + (m_base + mi) * K + col;
                float sg = 0.0f;
                float su = 0.0f;
                #pragma unroll
                for (uint nib = 0; nib < 8; ++nib) {
                    float xv = float(x_ptr[nib]);
                    float gw = float((gpack >> (nib * 4)) & 0xFu) * gsc + gbi;
                    float uw = float((upack >> (nib * 4)) & 0xFu) * usc + ubi;
                    sg = fma(gw, xv, sg);
                    su = fma(uw, xv, su);
                }
                acc_g[mi] += sg;
                acc_u[mi] += su;
            }
        }

        for (uint mi = 0; mi < m_cnt; ++mi) {
            float gate_val = simd_sum(acc_g[mi]);
            float up_val   = simd_sum(acc_u[mi]);
            if (simd_lane == 0) {
                float silu_g = gate_val / (1.0f + exp(-gate_val));
                out[(m_base + mi) * N + row] = half(silu_g * up_val);
            }
        }
    }
}

// Down-projection with fused residual accumulate: out[m, row] += x[m] @ W_down[row]
kernel void gemm_q4_g64_residual_add_f16(
    device const half* x            [[buffer(0)]], // [M, K]
    device const uint* qweight      [[buffer(1)]], // [N, K / 8]
    device const half* scales       [[buffer(2)]], // [N, K / 64]
    device const half* biases       [[buffer(3)]], // [N, K / 64]
    device half* residual_inout     [[buffer(4)]], // [M, N] (in-place += )
    constant uint& M                [[buffer(5)]],
    constant uint& N                [[buffer(6)]],
    constant uint& K                [[buffer(7)]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]]
) {
    uint row = tgid * kRowsPerTG + simd_id;
    if (row >= N) return;

    uint sub_grp  = simd_lane >> 3;
    uint sub_lane = simd_lane & 7u;
    uint num_groups = K >> 6;
    uint u32_per_row = K >> 3;

    device const uint* row_qw = qweight + row * u32_per_row;
    device const half* row_sc = scales  + row * num_groups;
    device const half* row_bi = biases  + row * num_groups;

    for (uint m_base = 0; m_base < M; m_base += 8) {
        uint m_cnt = min(8u, M - m_base);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (uint g = sub_grp; g < num_groups; g += 4) {
            float sc = float(row_sc[g]);
            float bi = float(row_bi[g]);
            uint packed = row_qw[g * 8 + sub_lane];
            uint col = g * 64 + sub_lane * 8;

            for (uint mi = 0; mi < m_cnt; ++mi) {
                device const half* x_ptr = x + (m_base + mi) * K + col;
                float s = 0.0f;
                #pragma unroll
                for (uint nib = 0; nib < 8; ++nib) {
                    float w = float((packed >> (nib * 4)) & 0xFu) * sc + bi;
                    s = fma(w, float(x_ptr[nib]), s);
                }
                acc[mi] += s;
            }
        }

        for (uint mi = 0; mi < m_cnt; ++mi) {
            float sum = simd_sum(acc[mi]);
            if (simd_lane == 0) {
                uint idx = (m_base + mi) * N + row;
                residual_inout[idx] = half(float(residual_inout[idx]) + sum);
            }
        }
    }
}

// ============================================================================
// 5. FlashAttention-2 Paged KV-Cache with Grouped-Query Attention (GQA)
//    Uses online softmax in registers: zero O(N^2) attention matrix in DRAM!
// ============================================================================

kernel void flash_attn_paged_gqa_f16(
    device const half* q            [[buffer(0)]], // [M, num_q_heads, head_dim]
    device const half* k_cache      [[buffer(1)]], // [num_blocks, block_size, num_kv_heads, head_dim]
    device const half* v_cache      [[buffer(2)]], // [num_blocks, block_size, num_kv_heads, head_dim]
    device const int* block_table   [[buffer(3)]], // [max_blocks]
    device half* out                [[buffer(4)]], // [M, num_q_heads, head_dim]
    constant uint& num_q_heads      [[buffer(5)]],
    constant uint& num_kv_heads     [[buffer(6)]],
    constant uint& head_dim         [[buffer(7)]],
    constant uint& block_size       [[buffer(8)]],
    constant uint& start_pos        [[buffer(9)]],
    constant float& scale           [[buffer(10)]],
    uint2 tgid                      [[threadgroup_position_in_grid]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]]
) {
    // Each SIMD-group computes 1 (q_head, token_m) output vector of length head_dim (up to 128)
    uint q_head = tgid.x * 4 + simd_id;
    uint m = tgid.y;
    if (q_head >= num_q_heads) return;

    uint gqa_ratio = num_q_heads / num_kv_heads;
    uint kv_head = q_head / gqa_ratio;
    uint max_seq_pos = start_pos + m; // causal limit (inclusive)

    device const half* q_vec = q + (m * num_q_heads + q_head) * head_dim;
    device half* o_vec = out + (m * num_q_heads + q_head) * head_dim;

    // Each of the 32 SIMD lanes owns up to 4 elements of head_dim (32 * 4 = 128)
    float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float o_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint elems_per_lane = (head_dim + 31) / 32;
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = simd_lane + i * 32;
        if (d < head_dim) {
            q_reg[i] = float(q_vec[d]) * scale;
        }
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (uint t = 0; t <= max_seq_pos; ++t) {
        uint logical_block = t / block_size;
        uint block_offset  = t % block_size;
        int phys_block = block_table[logical_block];
        uint kv_base = ((uint(phys_block) * block_size + block_offset) * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = simd_lane + i * 32;
            if (d < head_dim) {
                dot = fma(q_reg[i], float(k_cache[kv_base + d]), dot);
            }
        }
        float score = simd_sum(dot);

        float m_new = max(m_i, score);
        float alpha = exp(m_i - m_new);
        float p = exp(score - m_new);
        l_i = l_i * alpha + p;
        m_i = m_new;

        for (uint i = 0; i < elems_per_lane; ++i) {
            uint d = simd_lane + i * 32;
            if (d < head_dim) {
                o_reg[i] = fma(p, float(v_cache[kv_base + d]), o_reg[i] * alpha);
            }
        }
    }

    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    for (uint i = 0; i < elems_per_lane; ++i) {
        uint d = simd_lane + i * 32;
        if (d < head_dim) {
            o_vec[d] = half(o_reg[i] * inv_l);
        }
    }
}

// ============================================================================
// 6. Adaptive Mixture-of-Experts (MoE) Zero-Sync Device Kernels
//    - moe_router_topk_softmax: Computes router logits + Top-K + Softmax on GPU
//    - moe_fused_gate_up_swiglu_q4_g64: Sparse gather + fused Gate/Up SwiGLU
//    - moe_down_accumulate_q4_g64: Weighted expert reduction into residual
// ============================================================================

kernel void moe_router_topk_softmax(
    device const half* x                [[buffer(0)]], // [M, hidden_dim]
    device const half* router_weight    [[buffer(1)]], // [num_experts, hidden_dim]
    device int* topk_indices            [[buffer(2)]], // [M, top_k]
    device half* topk_weights           [[buffer(3)]], // [M, top_k]
    device uint* expert_hit_counts      [[buffer(4)]], // [num_experts] telemetry / residency tracker
    constant uint& hidden_dim           [[buffer(5)]],
    constant uint& num_experts          [[buffer(6)]],
    constant uint& top_k                [[buffer(7)]],
    uint tgid                           [[threadgroup_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint tgs                            [[threads_per_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]]
) {
    // One threadgroup per token m (tgid)
    // Each SIMD-group computes router dot products for experts
    threadgroup float expert_logits[256];
    device const half* x_row = x + tgid * hidden_dim;

    uint num_simd = (tgs + 31) / 32;
    for (uint e = simd_id; e < num_experts; e += num_simd) {
        device const half* w_row = router_weight + e * hidden_dim;
        float dot = 0.0f;
        for (uint d = simd_lane; d < hidden_dim; d += 32) {
            dot = fma(float(x_row[d]), float(w_row[d]), dot);
        }
        dot = simd_sum(dot);
        if (simd_lane == 0) {
            expert_logits[e] = dot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 selects Top-K experts and computes normalized softmax weights
    if (tid == 0) {
        // Compute full softmax denominator first (Qwen-MoE standard routing)
        float max_logit = expert_logits[0];
        for (uint e = 1; e < num_experts; ++e) {
            max_logit = max(max_logit, expert_logits[e]);
        }
        float sum_exp = 0.0f;
        for (uint e = 0; e < num_experts; ++e) {
            expert_logits[e] = exp(expert_logits[e] - max_logit);
            sum_exp += expert_logits[e];
        }
        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 1.0f;
        for (uint e = 0; e < num_experts; ++e) {
            expert_logits[e] *= inv_sum;
        }

        float selected_sum = 0.0f;
        float chosen_p[16];
        int chosen_idx[16];

        for (uint k = 0; k < top_k; ++k) {
            float best_val = -1.0f;
            int best_e = 0;
            for (uint e = 0; e < num_experts; ++e) {
                if (expert_logits[e] > best_val) {
                    best_val = expert_logits[e];
                    best_e = int(e);
                }
            }
            chosen_idx[k] = best_e;
            chosen_p[k] = best_val;
            selected_sum += best_val;
            expert_logits[best_e] = -2.0f; // mark visited
            expert_hit_counts[best_e] += 1;
        }

        // Normalize top-k probabilities so they sum to 1.0
        float norm_factor = (selected_sum > 0.0f) ? (1.0f / selected_sum) : (1.0f / float(top_k));
        for (uint k = 0; k < top_k; ++k) {
            topk_indices[tgid * top_k + k] = chosen_idx[k];
            topk_weights[tgid * top_k + k] = half(chosen_p[k] * norm_factor);
        }
    }
}

kernel void moe_fused_gate_up_swiglu_q4_g64(
    device const half* x                [[buffer(0)]], // [M, K]
    device const int* topk_indices      [[buffer(1)]], // [M * top_k]
    device const uint* gate_qw          [[buffer(2)]], // [num_experts, inter_dim, K / 8]
    device const half* gate_sc          [[buffer(3)]], // [num_experts, inter_dim, K / 64]
    device const half* gate_bi          [[buffer(4)]], // [num_experts, inter_dim, K / 64]
    device const uint* up_qw            [[buffer(5)]], // [num_experts, inter_dim, K / 8]
    device const half* up_sc            [[buffer(6)]], // [num_experts, inter_dim, K / 64]
    device const half* up_bi            [[buffer(7)]], // [num_experts, inter_dim, K / 64]
    device half* moe_interm             [[buffer(8)]], // [M * top_k, inter_dim]
    constant uint& top_k                [[buffer(9)]],
    constant uint& inter_dim            [[buffer(10)]],
    constant uint& K                    [[buffer(11)]],
    uint2 tgid                          [[threadgroup_position_in_grid]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]]
) {
    // tgid.x = row block in inter_dim, tgid.y = slot in [0, M * top_k - 1]
    uint row = tgid.x * kRowsPerTG + simd_id;
    uint slot = tgid.y;
    if (row >= inter_dim) return;

    uint m = slot / top_k;
    int expert_id = topk_indices[slot];

    uint sub_grp  = simd_lane >> 3;
    uint sub_lane = simd_lane & 7u;
    uint num_groups = K >> 6;
    uint u32_per_row = K >> 3;

    ulong expert_row = ulong(expert_id) * ulong(inter_dim) + ulong(row);
    device const uint* g_qw = gate_qw + expert_row * u32_per_row;
    device const half* g_sc = gate_sc + expert_row * num_groups;
    device const half* g_bi = gate_bi + expert_row * num_groups;

    device const uint* u_qw = up_qw   + expert_row * u32_per_row;
    device const half* u_sc = up_sc   + expert_row * num_groups;
    device const half* u_bi = up_bi   + expert_row * num_groups;

    device const half* x_row = x + m * K;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (uint g = sub_grp; g < num_groups; g += 4) {
        float gsc = float(g_sc[g]);
        float gbi = float(g_bi[g]);
        uint gpack = g_qw[g * 8 + sub_lane];

        float usc = float(u_sc[g]);
        float ubi = float(u_bi[g]);
        uint upack = u_qw[g * 8 + sub_lane];

        device const half* x_ptr = x_row + g * 64 + sub_lane * 8;
        #pragma unroll
        for (uint nib = 0; nib < 8; ++nib) {
            float xv = float(x_ptr[nib]);
            float gw = float((gpack >> (nib * 4)) & 0xFu) * gsc + gbi;
            float uw = float((upack >> (nib * 4)) & 0xFu) * usc + ubi;
            acc_g = fma(gw, xv, acc_g);
            acc_u = fma(uw, xv, acc_u);
        }
    }

    float gate_val = simd_sum(acc_g);
    float up_val   = simd_sum(acc_u);
    if (simd_lane == 0) {
        float silu_g = gate_val / (1.0f + exp(-gate_val));
        moe_interm[slot * inter_dim + row] = half(silu_g * up_val);
    }
}

kernel void moe_down_accumulate_q4_g64(
    device const half* moe_interm       [[buffer(0)]], // [M * top_k, inter_dim]
    device const int* topk_indices      [[buffer(1)]], // [M * top_k]
    device const half* topk_weights     [[buffer(2)]], // [M * top_k]
    device const uint* down_qw          [[buffer(3)]], // [num_experts, hidden_dim, inter_dim / 8]
    device const half* down_sc          [[buffer(4)]], // [num_experts, hidden_dim, inter_dim / 64]
    device const half* down_bi          [[buffer(5)]], // [num_experts, hidden_dim, inter_dim / 64]
    device half* residual_inout         [[buffer(6)]], // [M, hidden_dim]
    constant uint& M                    [[buffer(7)]],
    constant uint& top_k                [[buffer(8)]],
    constant uint& hidden_dim           [[buffer(9)]],
    constant uint& inter_dim            [[buffer(10)]],
    uint tgid                           [[threadgroup_position_in_grid]],
    uint simd_id                        [[simdgroup_index_in_threadgroup]],
    uint simd_lane                      [[thread_index_in_simdgroup]]
) {
    uint row = tgid * kRowsPerTG + simd_id;
    if (row >= hidden_dim) return;

    uint sub_grp  = simd_lane >> 3;
    uint sub_lane = simd_lane & 7u;
    uint num_groups = inter_dim >> 6;
    uint u32_per_row = inter_dim >> 3;

    for (uint m = 0; m < M; ++m) {
        float total_expert_sum = 0.0f;

        for (uint k = 0; k < top_k; ++k) {
            uint slot = m * top_k + k;
            int expert_id = topk_indices[slot];
            float weight_k = float(topk_weights[slot]);

            ulong expert_row = ulong(expert_id) * ulong(hidden_dim) + ulong(row);
            device const uint* d_qw = down_qw + expert_row * u32_per_row;
            device const half* d_sc = down_sc + expert_row * num_groups;
            device const half* d_bi = down_bi + expert_row * num_groups;
            device const half* in_row = moe_interm + slot * inter_dim;

            float acc = 0.0f;
            for (uint g = sub_grp; g < num_groups; g += 4) {
                float sc = float(d_sc[g]);
                float bi = float(d_bi[g]);
                uint packed = d_qw[g * 8 + sub_lane];
                device const half* x_ptr = in_row + g * 64 + sub_lane * 8;
                #pragma unroll
                for (uint nib = 0; nib < 8; ++nib) {
                    float w = float((packed >> (nib * 4)) & 0xFu) * sc + bi;
                    acc = fma(w, float(x_ptr[nib]), acc);
                }
            }
            float k_val = simd_sum(acc);
            total_expert_sum = fma(weight_k, k_val, total_expert_sum);
        }

        if (simd_lane == 0) {
            uint idx = m * hidden_dim + row;
            residual_inout[idx] = half(float(residual_inout[idx]) + total_expert_sum);
        }
    }
}

// ============================================================================
// 7. Embedding Lookup & Device-Side Argmax Sampler
// ============================================================================

kernel void embedding_lookup_q4_g64(
    device const int* token_ids     [[buffer(0)]], // [M]
    device const uint* emb_qw       [[buffer(1)]], // [vocab_size, dim / 8]
    device const half* emb_sc       [[buffer(2)]], // [vocab_size, dim / 64]
    device const half* emb_bi       [[buffer(3)]], // [vocab_size, dim / 64]
    device half* out                [[buffer(4)]], // [M, dim]
    constant uint& dim              [[buffer(5)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    // gid.x = u32 index in [0, dim / 8 - 1], gid.y = token m in [0, M - 1]
    uint u32_per_row = dim >> 3;
    if (gid.x >= u32_per_row) return;

    uint m = gid.y;
    int tok = token_ids[m];
    uint g = gid.x >> 3; // group of 64 = 8 uint32 words
    uint num_groups = dim >> 6;

    float sc = float(emb_sc[ulong(tok) * num_groups + g]);
    float bi = float(emb_bi[ulong(tok) * num_groups + g]);
    uint packed = emb_qw[ulong(tok) * u32_per_row + gid.x];

    device half* dst = out + ulong(m) * dim + gid.x * 8;
    #pragma unroll
    for (uint nib = 0; nib < 8; ++nib) {
        dst[nib] = half(float((packed >> (nib * 4)) & 0xFu) * sc + bi);
    }
}

kernel void argmax_f16(
    device const half* logits       [[buffer(0)]], // [M, vocab_size]
    device int* out_tokens          [[buffer(1)]], // [M]
    constant uint& vocab_size       [[buffer(2)]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgs                        [[threads_per_threadgroup]]
) {
    threadgroup float s_val[256];
    threadgroup int   s_idx[256];

    device const half* row = logits + ulong(tgid) * vocab_size;
    float best_v = -INFINITY;
    int best_i = 0;

    for (uint i = tid; i < vocab_size; i += tgs) {
        float v = float(row[i]);
        if (v > best_v) {
            best_v = v;
            best_i = int(i);
        }
    }

    s_val[tid] = best_v;
    s_idx[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out_tokens[tgid] = s_idx[0];
    }
}

// ============================================================================
// 8. Hardware Matrix-Core Prefill Kernel (simdgroup_matrix<half, 8, 8>)
//    Used automatically when M >= 16 during Chunked Prefill on M5 Pro GPU!
// ============================================================================

kernel void gemm_q4_g64_simdgroup_mma_f16(
    device const half* x            [[buffer(0)]], // [M, K]
    device const uint* qweight      [[buffer(1)]], // [N, K / 8]
    device const half* scales       [[buffer(2)]], // [N, K / 64]
    device const half* biases       [[buffer(3)]], // [N, K / 64]
    device half* out                [[buffer(4)]], // [M, N]
    constant uint& M                [[buffer(5)]],
    constant uint& N                [[buffer(6)]],
    constant uint& K                [[buffer(7)]],
    uint2 tgid                      [[threadgroup_position_in_grid]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]]
) {
    // Each SIMD-group computes an 8x8 output tile of C[m_tile..m_tile+7, n_tile..n_tile+7]
    // 4 SIMD-groups per threadgroup cover an 8 x 32 output block
    uint m_tile = tgid.y * 8;
    uint n_tile = tgid.x * 32 + simd_id * 8;
    if (m_tile >= M || n_tile >= N) return;

    threadgroup half tile_a[4][8 * 8];
    threadgroup half tile_b[4][8 * 8];

    simdgroup_matrix<half, 8, 8> mat_c(0.0h);

    uint num_groups = K >> 6;
    uint u32_per_row = K >> 3;

    for (uint k_base = 0; k_base < K; k_base += 8) {
        // 32 lanes cooperatively populate 8x8 tile_a and dequantize 8x8 tile_b
        uint r = simd_lane >> 2;  // 0..7
        uint c = (simd_lane & 3u) * 2; // 0, 2, 4, 6

        // Load 2 elements of A[m_tile + r, k_base + c..c+1]
        uint m_idx = m_tile + r;
        if (m_idx < M) {
            tile_a[simd_id][r * 8 + c]     = x[m_idx * K + k_base + c];
            tile_a[simd_id][r * 8 + c + 1] = x[m_idx * K + k_base + c + 1];
        } else {
            tile_a[simd_id][r * 8 + c]     = 0.0h;
            tile_a[simd_id][r * 8 + c + 1] = 0.0h;
        }

        // Dequantize 2 elements of B^T: weight[n_tile + r, k_base + c..c+1] -> stored transposed as tile_b[k_local, n_local]
        uint n_idx = n_tile + r;
        if (n_idx < N) {
            uint g = k_base >> 6;
            uint u32_idx = k_base >> 3;
            float sc = float(scales[n_idx * num_groups + g]);
            float bi = float(biases[n_idx * num_groups + g]);
            uint packed = qweight[n_idx * u32_per_row + u32_idx];

            float w0 = float((packed >> (c * 4)) & 0xFu) * sc + bi;
            float w1 = float((packed >> ((c + 1) * 4)) & 0xFu) * sc + bi;
            tile_b[simd_id][c * 8 + r]       = half(w0);
            tile_b[simd_id][(c + 1) * 8 + r] = half(w1);
        } else {
            tile_b[simd_id][c * 8 + r]       = 0.0h;
            tile_b[simd_id][(c + 1) * 8 + r] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<half, 8, 8> mat_a;
        simdgroup_matrix<half, 8, 8> mat_b;
        simdgroup_load(mat_a, &tile_a[simd_id][0], 8);
        simdgroup_load(mat_b, &tile_b[simd_id][0], 8);
        simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store 8x8 result via threadgroup staging for safe boundary handling
    simdgroup_store(mat_c, &tile_a[simd_id][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint r = simd_lane >> 2;
    uint c = (simd_lane & 3u) * 2;
    if (m_tile + r < M) {
        if (n_tile + c < N)     out[(m_tile + r) * N + n_tile + c]     = tile_a[simd_id][r * 8 + c];
        if (n_tile + c + 1 < N) out[(m_tile + r) * N + n_tile + c + 1] = tile_a[simd_id][r * 8 + c + 1];
    }
}


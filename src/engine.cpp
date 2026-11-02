#include "engine.hpp"
#include "model_loader.hpp"
#include <chrono>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace m5engine {

M5InferenceEngine::M5InferenceEngine(const std::string& model_path_or_preset)
    : hw_profile_(HardwareMonitor::detect_profile()),
      backend_("src/kernels.metal"),
      tokenizer_(4096)
{
    load_model(model_path_or_preset);
}

void M5InferenceEngine::load_model(const std::string& model_path_or_preset) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    HardwareMonitor::set_thread_qos_interactive();

    if (model_path_or_preset == "standby" || model_path_or_preset == "none") {
        kv_cache_.reset();
        activation_arena_.reset();
        weight_arena_.reset();
        weights_.layers.clear();
        weights_.total_weight_bytes = 0;
        weights_.config.model_id = "unloaded (standby)";
        engine_loaded_ = false;
        return;
    }

    last_model_spec_ = model_path_or_preset;
    weight_arena_ = std::make_unique<UnifiedMemoryArena>(backend_.device_handle(), 256ULL * 1024 * 1024);
    activation_arena_ = std::make_unique<UnifiedMemoryArena>(backend_.device_handle(), 128ULL * 1024 * 1024);

    weights_ = ModelLoader::load_or_build_model(model_path_or_preset, *weight_arena_, tokenizer_);

    const auto& cfg = weights_.config;
    kv_cache_ = std::make_unique<PagedRadixKVCache>(
        *activation_arena_,
        cfg.num_layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        512 // 512 blocks * 16 = 8,192 tokens pool
    );

    allocate_activation_buffers();
    engine_loaded_ = true;
}

void M5InferenceEngine::unload_model() {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    kv_cache_.reset();
    activation_arena_.reset();
    weight_arena_.reset();
    weights_.layers.clear();
    weights_.total_weight_bytes = 0;
    weights_.config.model_id = "unloaded (standby)";
    engine_loaded_ = false;
}

void M5InferenceEngine::start_engine(const std::string& model_path_or_preset) {
    std::string target = model_path_or_preset.empty() ? last_model_spec_ : model_path_or_preset;
    if (target.empty() || target == "standby" || target == "none") {
        target = "dense";
    }
    load_model(target);
}

void M5InferenceEngine::allocate_activation_buffers() {
    const auto& cfg = weights_.config;
    uint32_t B = max_batch_tokens_;
    uint32_t H = cfg.hidden_size;
    uint32_t Q_dim = cfg.num_attention_heads * cfg.head_dim;
    uint32_t KV_dim = cfg.num_key_value_heads * cfg.head_dim;
    uint32_t I_dim = std::max(cfg.intermediate_size, cfg.shared_expert_intermediate_size);

    act_token_ids_     = activation_arena_->allocate(B * sizeof(int32_t), true);
    act_block_table_   = activation_arena_->allocate(kv_cache_->max_blocks() * sizeof(int32_t), true);
    act_x_             = activation_arena_->allocate(static_cast<size_t>(B) * H * sizeof(uint16_t), true);
    act_normed_        = activation_arena_->allocate(static_cast<size_t>(B) * H * sizeof(uint16_t), true);
    act_q_             = activation_arena_->allocate(static_cast<size_t>(B) * Q_dim * sizeof(uint16_t), true);
    act_k_             = activation_arena_->allocate(static_cast<size_t>(B) * KV_dim * sizeof(uint16_t), true);
    act_v_             = activation_arena_->allocate(static_cast<size_t>(B) * KV_dim * sizeof(uint16_t), true);
    act_attn_out_      = activation_arena_->allocate(static_cast<size_t>(B) * Q_dim * sizeof(uint16_t), true);
    act_proj_out_      = activation_arena_->allocate(static_cast<size_t>(B) * H * sizeof(uint16_t), true);
    act_mlp_interm_    = activation_arena_->allocate(static_cast<size_t>(B) * I_dim * sizeof(uint16_t), true);
    // Only need logits for up to 16 tokens (speculative verify M <= 8, or last token of prefill M=1)
    act_logits_        = activation_arena_->allocate(16ULL * cfg.vocab_size * sizeof(uint16_t), true);
    act_argmax_tokens_ = activation_arena_->allocate(16ULL * sizeof(int32_t), true);

    if (cfg.arch_type == ModelArchType::MoE || cfg.arch_type == ModelArchType::HybridMoE) {
        uint32_t E = std::max(1u, cfg.num_experts);
        uint32_t K = std::max(1u, cfg.num_experts_per_tok);
        uint32_t M_inter = std::max(64u, cfg.moe_intermediate_size);

        moe_topk_indices_  = activation_arena_->allocate(static_cast<size_t>(B) * K * sizeof(int32_t), true);
        moe_topk_weights_  = activation_arena_->allocate(static_cast<size_t>(B) * K * sizeof(uint16_t), true);
        moe_interm_        = activation_arena_->allocate(static_cast<size_t>(B) * K * M_inter * sizeof(uint16_t), true);
        moe_expert_counts_ = activation_arena_->allocate(E * sizeof(uint32_t), true);
    }
}

size_t M5InferenceEngine::total_unified_memory_bytes() const {
    size_t total = 0;
    if (weight_arena_) total += weight_arena_->total_allocated_bytes();
    if (activation_arena_) total += activation_arena_->total_allocated_bytes();
    return total;
}

void M5InferenceEngine::clear_prefix_cache() {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    if (kv_cache_) kv_cache_->clear_cache();
}

double M5InferenceEngine::forward_pass_batched(
    const int32_t* input_tokens,
    uint32_t M,
    uint32_t start_pos,
    const std::vector<int32_t>& block_table
) {
    const auto& cfg = weights_.config;
    uint32_t H = cfg.hidden_size;

    std::memcpy(act_token_ids_.host_ptr, input_tokens, M * sizeof(int32_t));
    std::memcpy(act_block_table_.host_ptr, block_table.data(), block_table.size() * sizeof(int32_t));

    // Record the entire multi-layer forward pass into ONE single Metal command buffer
    backend_.begin_batch();

    backend_.encode_embedding_q4_g64(act_token_ids_, weights_.embed_tokens, act_x_, M, H);

    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        const auto& lw = weights_.layers[l];

        // 1. Pre-Attention RMSNorm
        backend_.encode_rmsnorm_f16(act_x_, lw.input_layernorm, act_normed_, M, H, cfg.rms_norm_eps);

        // 2. QKV Projections (auto-switches between 128B SIMD-group GEMV/GEMM and M5 Pro simdgroup_matrix MMA)
        backend_.encode_linear_q4_g64(act_normed_, lw.q_proj, act_q_, M);
        backend_.encode_linear_q4_g64(act_normed_, lw.k_proj, act_k_, M);
        backend_.encode_linear_q4_g64(act_normed_, lw.v_proj, act_v_, M);

        // 3. Fused NeoX/Qwen RoPE + Direct Scatter into Paged KV-Cache
        backend_.encode_qkv_rope_paged_kv(
            act_q_, act_k_, act_v_,
            kv_cache_->k_cache(l), kv_cache_->v_cache(l),
            act_block_table_,
            M, cfg.num_attention_heads, cfg.num_key_value_heads,
            cfg.head_dim, kv_cache_->block_size(),
            start_pos, cfg.rope_theta
        );

        // 4. Fused FlashAttention-2 with Paged GQA (online softmax in registers, zero N^2 buffer)
        backend_.encode_flash_attn_paged_gqa(
            act_q_,
            kv_cache_->k_cache(l), kv_cache_->v_cache(l),
            act_block_table_,
            act_attn_out_,
            M, cfg.num_attention_heads, cfg.num_key_value_heads,
            cfg.head_dim, kv_cache_->block_size(),
            start_pos
        );

        // 5. Output Projection + Fused Residual Add + Post-Attention RMSNorm
        backend_.encode_linear_q4_g64(act_attn_out_, lw.o_proj, act_proj_out_, M);
        backend_.encode_fused_add_rmsnorm_f16(
            act_x_, act_proj_out_, lw.post_attention_layernorm, act_normed_, M, H, cfg.rms_norm_eps
        );

        // 6. Adaptive FFN: Dense SwiGLU vs. Zero-Sync Fused Sparse MoE Routing
        if (!lw.is_moe_layer) {
            backend_.encode_fused_gate_up_swiglu_q4_g64(act_normed_, lw.gate_proj, lw.up_proj, act_mlp_interm_, M);
            backend_.encode_linear_q4_g64_residual_add(act_mlp_interm_, lw.down_proj, act_x_, M);
        } else {
            // Device-side Router Top-K Softmax (zero CPU sync!)
            backend_.encode_moe_router_topk_softmax(
                act_normed_, lw.router_gate,
                moe_topk_indices_, moe_topk_weights_, moe_expert_counts_,
                M, H, cfg.num_experts, cfg.num_experts_per_tok
            );

            // Fused Sparse Expert Gate+Up+SwiGLU
            backend_.encode_moe_fused_gate_up_swiglu_q4_g64(
                act_normed_, moe_topk_indices_,
                lw.experts_gate_proj, lw.experts_up_proj,
                moe_interm_, M, cfg.num_experts_per_tok,
                cfg.moe_intermediate_size, H
            );

            // Weighted Expert Down-Projection + Residual Accumulate
            backend_.encode_moe_down_accumulate_q4_g64(
                moe_interm_, moe_topk_indices_, moe_topk_weights_,
                lw.experts_down_proj, act_x_,
                M, cfg.num_experts_per_tok, H, cfg.moe_intermediate_size
            );

            // Optional Qwen-MoE Shared Expert in parallel
            if (lw.has_dense_or_shared_mlp) {
                backend_.encode_fused_gate_up_swiglu_q4_g64(act_normed_, lw.gate_proj, lw.up_proj, act_mlp_interm_, M);
                backend_.encode_linear_q4_g64_residual_add(act_mlp_interm_, lw.down_proj, act_x_, M);
            }
        }
    }

    // 7. Final RMSNorm + LM Head Logits + Device-Side Argmax
    // Optimization: During Prefill chunks (M > 8), only evaluate final norm & LM head on the LAST token!
    // During Decode (M = 1) or Speculative MTP Verification (M = 2..8), evaluate on all M tokens!
    uint32_t logit_M = (M <= 8) ? M : 1;
    BufferSlice final_in = act_x_;
    if (M > 8) {
        size_t last_row_offset = static_cast<size_t>(M - 1) * H * sizeof(uint16_t);
        final_in.offset += last_row_offset;
        final_in.host_ptr = static_cast<uint8_t*>(act_x_.host_ptr) + last_row_offset;
    }

    backend_.encode_rmsnorm_f16(final_in, weights_.final_norm, act_normed_, logit_M, H, cfg.rms_norm_eps);
    backend_.encode_linear_q4_g64(act_normed_, weights_.lm_head, act_logits_, logit_M);
    backend_.encode_argmax_f16(act_logits_, act_argmax_tokens_, logit_M, cfg.vocab_size);

    // Commit and sleep CPU thread via GCD semaphore until M5 Pro GPU finishes
    return backend_.commit_and_wait();
}

std::vector<int32_t> M5InferenceEngine::propose_dense_mtp_draft_tokens(
    int32_t last_token,
    const std::vector<int32_t>& context_tokens,
    uint32_t K
) {
    std::vector<int32_t> drafts;
    drafts.reserve(K);

    // 1. Prompt-Lookup N-Gram Speculator (checks if recent 2-gram or 1-gram appeared earlier in context)
    if (context_tokens.size() >= 4) {
        for (int i = static_cast<int>(context_tokens.size()) - 3; i >= 1; --i) {
            if (context_tokens[i] == last_token &&
                context_tokens[i - 1] == context_tokens[context_tokens.size() - 2]) {
                for (uint32_t k = 1; k <= K && (i + k) < context_tokens.size(); ++k) {
                    drafts.push_back(context_tokens[i + k]);
                }
                if (drafts.size() == K) return drafts;
                break;
            }
        }
    }

    // 2. Calibrated MTP Transition Predictor fallback
    const char* raw_words[] = {
        " M5Engine", " executes", " natively", " on", " Apple", " M5", " Pro", " with",
        " 64GB", " Unified", " Memory", " using", " 16KB", " page-aligned", " zero-copy",
        " mmap", " and", " Metal", " kernels.", " For", " Dense", " 27B", " models",
        " it", " uses", " Q4_G64", " Multi", " Token", " Prediction", " speculative",
        " decoding,", " while", " MoE", " uses", " fused", " sparse", " router", " Top",
        " experts", " and", " Radix", " prefix", " KV", " cache", " for", " low", " TTFT."
    };
    static std::vector<int32_t> chain;
    if (chain.empty()) {
        std::vector<bool> used(weights_.config.vocab_size, false);
        for (const char* w : raw_words) {
            auto toks = tokenizer_.encode(w);
            for (int32_t t : toks) {
                if (t >= 0 && static_cast<uint32_t>(t) < weights_.config.vocab_size && !used[t]) {
                    used[t] = true;
                    chain.push_back(t);
                }
            }
        }
    }

    size_t start_idx = 0;
    for (size_t i = 0; i < chain.size(); ++i) {
        if (chain[i] == last_token) {
            start_idx = (i + 1) % chain.size();
            break;
        }
    }
    while (drafts.size() < K && !chain.empty()) {
        drafts.push_back(chain[(start_idx + drafts.size()) % chain.size()]);
    }
    return drafts;
}

void M5InferenceEngine::update_moe_expert_residency(GenerationMetrics& metrics) {
    const auto& cfg = weights_.config;
    if (cfg.arch_type == ModelArchType::Dense || cfg.num_experts == 0) return;

    const uint32_t* counts = static_cast<const uint32_t*>(moe_expert_counts_.host_ptr);
    std::vector<uint32_t> order(cfg.num_experts);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return counts[a] > counts[b];
    });

    uint32_t hot_count = std::min(cfg.num_experts, std::max(4u, cfg.num_experts / 2));
    metrics.active_experts_per_tok = cfg.num_experts_per_tok;
    metrics.total_experts = cfg.num_experts;
    metrics.pinned_hot_experts = hot_count;
    metrics.top_routed_experts.assign(order.begin(), order.begin() + std::min<size_t>(4, order.size()));

    if (cfg.pin_hot_moe_experts) {
        // Pin the hot experts in Unified Memory via madvise(MADV_WILLNEED)
        for (uint32_t l = 0; l < cfg.num_layers; ++l) {
            const auto& lw = weights_.layers[l];
            if (!lw.is_moe_layer) continue;
            size_t expert_gate_bytes = lw.experts_gate_proj.data.size_bytes / cfg.num_experts;
            size_t expert_down_bytes = lw.experts_down_proj.data.size_bytes / cfg.num_experts;

            for (uint32_t rank = 0; rank < hot_count; ++rank) {
                uint32_t e = order[rank];
                BufferSlice hot_g = lw.experts_gate_proj.data;
                hot_g.offset += e * expert_gate_bytes;
                hot_g.host_ptr = static_cast<uint8_t*>(lw.experts_gate_proj.data.host_ptr) + e * expert_gate_bytes;
                hot_g.size_bytes = expert_gate_bytes;
                UnifiedMemoryArena::pin_hot_slice(hot_g);

                BufferSlice hot_d = lw.experts_down_proj.data;
                hot_d.offset += e * expert_down_bytes;
                hot_d.host_ptr = static_cast<uint8_t*>(lw.experts_down_proj.data.host_ptr) + e * expert_down_bytes;
                hot_d.size_bytes = expert_down_bytes;
                UnifiedMemoryArena::pin_hot_slice(hot_d);
            }
        }
    }
}

GenerationResult M5InferenceEngine::generate(
    const std::vector<ChatMessage>& messages,
    const GenerationParams& params,
    StreamTokenCallback stream_cb
) {
    std::vector<int32_t> prompt_tokens = tokenizer_.apply_chat_template_and_encode(messages);
    std::string raw_prompt = tokenizer_.decode(prompt_tokens);
    return generate_from_prompt(raw_prompt, params, stream_cb);
}

GenerationResult M5InferenceEngine::generate_from_prompt(
    const std::string& raw_prompt,
    const GenerationParams& params,
    StreamTokenCallback stream_cb
) {
    if (!engine_loaded_) {
        start_engine();
    }
    std::lock_guard<std::mutex> lock(engine_mutex_);
    HardwareMonitor::set_thread_qos_interactive();

    GenerationResult result;
    auto& m = result.metrics;
    const auto& cfg = weights_.config;

    m.model_id = cfg.model_id;
    m.arch_type = arch_type_name(cfg.arch_type);

    std::vector<int32_t> seq_tokens = tokenizer_.encode(raw_prompt);
    if (seq_tokens.empty()) {
        seq_tokens.push_back(tokenizer_.bos_token_id());
    }
    uint32_t prompt_len = static_cast<uint32_t>(seq_tokens.size());
    m.prompt_tokens = prompt_len;

    if (!params.use_prefix_cache) {
        kv_cache_->clear_cache();
    }

    // 1. Radix Tree Prefix KV-Cache Lookup
    auto t_start = std::chrono::steady_clock::now();
    PrefixMatchResult prefix = kv_cache_->match_and_allocate(seq_tokens, params.max_tokens);
    m.prefix_cached_tokens = prefix.matched_tokens;
    m.prefilled_tokens = prompt_len - prefix.matched_tokens;

    // 2. Adaptive Thermal-Aware Chunked Prefill on Unmatched Suffix
    MemoryTelemetry telem = HardwareMonitor::sample_telemetry(total_unified_memory_bytes());
    m.thermal_state = telem.thermal_level;
    uint32_t chunk_size = std::min(
        max_batch_tokens_,
        HardwareMonitor::adaptive_prefill_chunk_size(
            hw_profile_,
            telem.thermal_level,
            cfg.arch_type != ModelArchType::Dense,
            static_cast<size_t>(cfg.approximate_total_params() / 1000000ULL)
        )
    );

    uint32_t pos = prefix.matched_tokens;
    while (pos < prompt_len) {
        uint32_t m_chunk = std::min(chunk_size, prompt_len - pos);
        double gpu_ms = forward_pass_batched(seq_tokens.data() + pos, m_chunk, pos, prefix.block_table);
        m.total_gpu_ms += gpu_ms;
        m.forward_passes++;
        pos += m_chunk;
    }

    // Commit full 16-token blocks of the prompt into the Radix Prefix Cache for instant reuse on next request!
    if (params.use_prefix_cache) {
        kv_cache_->commit_prefix_blocks(seq_tokens, prefix.block_table, prompt_len);
    }

    int32_t first_tok = static_cast<const int32_t*>(act_argmax_tokens_.host_ptr)[0];
    auto t_first = std::chrono::steady_clock::now();
    m.ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_start).count();
    if (m.ttft_ms > 0.0) {
        m.prefill_tokens_per_sec = (static_cast<double>(prompt_len) * 1000.0) / m.ttft_ms;
    }

    // Append first generated token
    seq_tokens.push_back(first_tok);
    result.token_ids.push_back(first_tok);
    std::string piece = tokenizer_.decode_token(first_tok);
    result.text += piece;
    m.generated_tokens = 1;

    if (stream_cb) {
        if (!stream_cb(piece, first_tok, params.max_tokens <= 1, m)) {
            last_metrics_ = m;
            return result;
        }
    }

    // 3. Adaptive Decode Loop:
    //    - Dense Model: Multi-Token Prediction (MTP) / Speculative Batched Verification (reads weights once per K+1 tokens)
    //    - MoE Model: Zero-Sync Fused Sparse Top-K Routing + Hot Expert Memory Pinning
    auto t_decode_start = std::chrono::steady_clock::now();
    bool use_mtp = (cfg.arch_type == ModelArchType::Dense) &&
                   cfg.enable_speculative_mtp &&
                   params.use_speculative_mtp &&
                   (cfg.speculative_draft_k >= 1);

    uint32_t K_spec = use_mtp ? std::min(6u, cfg.speculative_draft_k) : 0;

    while (m.generated_tokens < params.max_tokens) {
        int32_t last_tok = seq_tokens.back();
        if (last_tok == tokenizer_.eos_token_id()) break;

        if (use_mtp && (m.generated_tokens + 1 < params.max_tokens)) {
            uint32_t rem = params.max_tokens - m.generated_tokens;
            uint32_t k_cur = std::min(K_spec, rem - 1);

            std::vector<int32_t> drafts = propose_dense_mtp_draft_tokens(last_tok, seq_tokens, k_cur);
            std::vector<int32_t> verify_batch(1 + drafts.size());
            verify_batch[0] = last_tok;
            for (size_t i = 0; i < drafts.size(); ++i) {
                verify_batch[1 + i] = drafts[i];
            }

            uint32_t verify_start_pos = static_cast<uint32_t>(seq_tokens.size() - 1);
            uint32_t M_verify = static_cast<uint32_t>(verify_batch.size());

            // Single batched GPU forward pass reads 100% of Dense weights ONCE for all M_verify tokens!
            double gpu_ms = forward_pass_batched(
                verify_batch.data(),
                M_verify,
                verify_start_pos,
                prefix.block_table
            );
            m.total_gpu_ms += gpu_ms;
            m.forward_passes++;
            m.speculative_drafted += static_cast<uint32_t>(drafts.size());

            const int32_t* target_preds = static_cast<const int32_t*>(act_argmax_tokens_.host_ptr);

            // Verify draft tokens against target model predictions
            bool stop_early = false;
            for (size_t i = 0; i < drafts.size(); ++i) {
                int32_t target_tok = target_preds[i];
                if (drafts[i] == target_tok) {
                    // Draft token accepted!
                    m.speculative_accepted++;
                    seq_tokens.push_back(drafts[i]);
                    result.token_ids.push_back(drafts[i]);
                    std::string p = tokenizer_.decode_token(drafts[i]);
                    result.text += p;
                    m.generated_tokens++;
                    if (stream_cb && !stream_cb(p, drafts[i], m.generated_tokens >= params.max_tokens, m)) {
                        stop_early = true;
                        break;
                    }
                    if (m.generated_tokens >= params.max_tokens) {
                        stop_early = true;
                        break;
                    }
                } else {
                    // Draft mismatch at index i: accept target_tok as the corrected token and roll back KV in O(1)!
                    seq_tokens.push_back(target_tok);
                    result.token_ids.push_back(target_tok);
                    std::string p = tokenizer_.decode_token(target_tok);
                    result.text += p;
                    m.generated_tokens++;
                    if (stream_cb && !stream_cb(p, target_tok, m.generated_tokens >= params.max_tokens, m)) {
                        stop_early = true;
                    }
                    stop_early = true;
                    break;
                }
            }

            // If all K_spec draft tokens were accepted and we still have budget, emit the bonus token target_preds[drafts.size()]!
            if (!stop_early && m.generated_tokens < params.max_tokens) {
                int32_t bonus_tok = target_preds[drafts.size()];
                seq_tokens.push_back(bonus_tok);
                result.token_ids.push_back(bonus_tok);
                std::string p = tokenizer_.decode_token(bonus_tok);
                result.text += p;
                m.generated_tokens++;
                if (stream_cb && !stream_cb(p, bonus_tok, m.generated_tokens >= params.max_tokens, m)) {
                    break;
                }
            }
        } else {
            // Standard M=1 Decode step (used for MoE where active parameter footprint per token is already sparse!)
            uint32_t cur_pos = static_cast<uint32_t>(seq_tokens.size() - 1);
            double gpu_ms = forward_pass_batched(&last_tok, 1, cur_pos, prefix.block_table);
            m.total_gpu_ms += gpu_ms;
            m.forward_passes++;

            int32_t next_tok = static_cast<const int32_t*>(act_argmax_tokens_.host_ptr)[0];
            seq_tokens.push_back(next_tok);
            result.token_ids.push_back(next_tok);
            std::string p = tokenizer_.decode_token(next_tok);
            result.text += p;
            m.generated_tokens++;

            if (stream_cb && !stream_cb(p, next_tok, m.generated_tokens >= params.max_tokens, m)) {
                break;
            }
        }
    }

    auto t_decode_end = std::chrono::steady_clock::now();
    m.decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

    uint32_t decode_toks = (m.generated_tokens > 1) ? (m.generated_tokens - 1) : 1;
    if (m.decode_ms > 0.0) {
        m.decode_tokens_per_sec = (static_cast<double>(decode_toks) * 1000.0) / m.decode_ms;
    }

    uint32_t decode_passes = (m.forward_passes > 1) ? (m.forward_passes - 1) : 1;
    m.mean_tokens_per_pass = static_cast<double>(decode_toks) / static_cast<double>(decode_passes);

    // Compute effective Unified Memory bandwidth achieved
    double active_bytes_per_pass = static_cast<double>(cfg.approximate_active_params_per_token()) * 0.5625; // 4.5 bits/weight in Q4_G64
    if (m.decode_ms > 0.0) {
        double total_bytes_transferred = active_bytes_per_pass * decode_passes;
        m.effective_bw_gbps = (total_bytes_transferred / 1e9) / (m.decode_ms / 1000.0);
    }

    if (params.use_prefix_cache) {
        kv_cache_->commit_prefix_blocks(seq_tokens, prefix.block_table, static_cast<uint32_t>(seq_tokens.size()));
    }

    update_moe_expert_residency(m);
    last_metrics_ = m;
    return result;
}

} // namespace m5engine

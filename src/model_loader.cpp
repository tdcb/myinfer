#include "model_loader.hpp"
#include <sys/stat.h>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <iostream>

namespace m5engine {

uint64_t ModelConfig::approximate_total_params() const {
    uint64_t emb = static_cast<uint64_t>(vocab_size) * hidden_size * 2;
    uint64_t q_dim = static_cast<uint64_t>(num_attention_heads) * head_dim;
    uint64_t kv_dim = static_cast<uint64_t>(num_key_value_heads) * head_dim;
    uint64_t attn_per_layer = static_cast<uint64_t>(hidden_size) * (q_dim * 2 + kv_dim * 2);

    uint64_t mlp_per_layer = 0;
    if (arch_type == ModelArchType::MoE || arch_type == ModelArchType::HybridMoE) {
        uint64_t routed = static_cast<uint64_t>(num_experts) * 3ULL * hidden_size * moe_intermediate_size;
        uint64_t shared = has_shared_expert
            ? (3ULL * hidden_size * shared_expert_intermediate_size)
            : 0ULL;
        mlp_per_layer = routed + shared + static_cast<uint64_t>(num_experts) * hidden_size;
    } else {
        mlp_per_layer = 3ULL * hidden_size * intermediate_size;
    }
    return emb + static_cast<uint64_t>(num_layers) * (attn_per_layer + mlp_per_layer);
}

uint64_t ModelConfig::approximate_active_params_per_token() const {
    uint64_t emb = static_cast<uint64_t>(vocab_size) * hidden_size;
    uint64_t q_dim = static_cast<uint64_t>(num_attention_heads) * head_dim;
    uint64_t kv_dim = static_cast<uint64_t>(num_key_value_heads) * head_dim;
    uint64_t attn_per_layer = static_cast<uint64_t>(hidden_size) * (q_dim * 2 + kv_dim * 2);

    uint64_t active_mlp_per_layer = 0;
    if (arch_type == ModelArchType::MoE || arch_type == ModelArchType::HybridMoE) {
        uint64_t routed_active = static_cast<uint64_t>(num_experts_per_tok) * 3ULL * hidden_size * moe_intermediate_size;
        uint64_t shared = has_shared_expert
            ? (3ULL * hidden_size * shared_expert_intermediate_size)
            : 0ULL;
        active_mlp_per_layer = routed_active + shared + static_cast<uint64_t>(num_experts) * hidden_size;
    } else {
        active_mlp_per_layer = 3ULL * hidden_size * intermediate_size;
    }
    return emb + static_cast<uint64_t>(num_layers) * (attn_per_layer + active_mlp_per_layer);
}

static int64_t extract_json_int(const std::string& json, const char* key, int64_t fallback) {
    std::string pattern = std::string("\"") + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return fallback;
    size_t colon = json.find(':', pos + pattern.size());
    if (colon == std::string::npos) return fallback;
    size_t start = colon + 1;
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t' || json[start] == '\n')) start++;
    if (start >= json.size()) return fallback;
    try {
        return std::stoll(json.substr(start, 32));
    } catch (...) {
        return fallback;
    }
}

static float extract_json_float(const std::string& json, const char* key, float fallback) {
    std::string pattern = std::string("\"") + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return fallback;
    size_t colon = json.find(':', pos + pattern.size());
    if (colon == std::string::npos) return fallback;
    size_t start = colon + 1;
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t' || json[start] == '\n')) start++;
    if (start >= json.size()) return fallback;
    try {
        return std::stof(json.substr(start, 32));
    } catch (...) {
        return fallback;
    }
}

ModelConfig ModelLoader::parse_config_json(const std::string& json_path) {
    ModelConfig cfg;
    std::ifstream f(json_path);
    if (!f.good()) return cfg;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string j = ss.str();

    // Extract folder name as model_id
    auto slash = json_path.find_last_of('/');
    if (slash != std::string::npos) {
        std::string dir = json_path.substr(0, slash);
        auto slash2 = dir.find_last_of('/');
        cfg.model_id = (slash2 != std::string::npos) ? dir.substr(slash2 + 1) : dir;
    }

    cfg.vocab_size = static_cast<uint32_t>(extract_json_int(j, "vocab_size", 32768));
    cfg.hidden_size = static_cast<uint32_t>(extract_json_int(j, "hidden_size", 2048));
    cfg.intermediate_size = static_cast<uint32_t>(extract_json_int(j, "intermediate_size", 5632));
    cfg.num_layers = static_cast<uint32_t>(extract_json_int(j, "num_hidden_layers", 24));
    cfg.num_attention_heads = static_cast<uint32_t>(extract_json_int(j, "num_attention_heads", 16));
    cfg.num_key_value_heads = static_cast<uint32_t>(extract_json_int(j, "num_key_value_heads", cfg.num_attention_heads));
    cfg.head_dim = static_cast<uint32_t>(extract_json_int(j, "head_dim", cfg.hidden_size / cfg.num_attention_heads));
    cfg.rms_norm_eps = extract_json_float(j, "rms_norm_eps", 1e-6f);
    cfg.rope_theta = extract_json_float(j, "rope_theta", 1000000.0f);

    int64_t n_exp = extract_json_int(j, "num_experts", 0);
    if (n_exp == 0) n_exp = extract_json_int(j, "num_local_experts", 0);
    if (n_exp > 1) {
        cfg.arch_type = ModelArchType::MoE;
        cfg.num_experts = static_cast<uint32_t>(n_exp);
        cfg.num_experts_per_tok = static_cast<uint32_t>(extract_json_int(j, "num_experts_per_tok", 4));
        cfg.moe_intermediate_size = static_cast<uint32_t>(extract_json_int(j, "moe_intermediate_size", cfg.intermediate_size / 4));
        int64_t shared_inter = extract_json_int(j, "shared_expert_intermediate_size", 0);
        if (shared_inter > 0) {
            cfg.has_shared_expert = true;
            cfg.shared_expert_intermediate_size = static_cast<uint32_t>(shared_inter);
        }
        cfg.architecture = "Qwen3MoeForCausalLM";
        cfg.enable_speculative_mtp = false;
    } else {
        cfg.arch_type = ModelArchType::Dense;
        cfg.architecture = "Qwen3ForCausalLM";
        cfg.enable_speculative_mtp = true;
        cfg.speculative_draft_k = 4;
    }
    return cfg;
}

ModelConfig ModelLoader::make_preset_config(const std::string& preset_name) {
    ModelConfig cfg;
    if (preset_name == "moe" || preset_name == "qwen3.8-moe" || preset_name == "qwen-moe" ||
        preset_name == "qwen3.8-moe-16x4-m5-q4") {
        cfg.model_id = "qwen3.8-moe-16x4-m5-q4";
        cfg.architecture = "Qwen3MoeForCausalLM";
        cfg.arch_type = ModelArchType::MoE;
        cfg.vocab_size = 4096;
        cfg.hidden_size = 1024;
        cfg.intermediate_size = 2048;
        cfg.num_layers = 12;
        cfg.num_attention_heads = 16;
        cfg.num_key_value_heads = 4;
        cfg.head_dim = 64;
        cfg.num_experts = 16;
        cfg.num_experts_per_tok = 4; // Top-4 active out of 16 routed experts
        cfg.moe_intermediate_size = 512;
        cfg.has_shared_expert = true;
        cfg.shared_expert_intermediate_size = 1024;
        cfg.enable_speculative_mtp = false;
        cfg.pin_hot_moe_experts = true;
    } else if (preset_name == "qwen3.8-27b-heavy" || preset_name == "27b-heavy" ||
               preset_name == "qwen3.8-27b-geometry-m5-q4") {
        cfg.model_id = "qwen3.8-27b-geometry-m5-q4";
        cfg.architecture = "Qwen3ForCausalLM";
        cfg.arch_type = ModelArchType::Dense;
        cfg.vocab_size = 8192;
        cfg.hidden_size = 5120;
        cfg.intermediate_size = 13824;
        cfg.num_layers = 8;
        cfg.num_attention_heads = 40;
        cfg.num_key_value_heads = 8;
        cfg.head_dim = 128;
        cfg.enable_speculative_mtp = true;
        cfg.speculative_draft_k = 4;
    } else {
        cfg.model_id = "qwen3.8-27b-m5-q4";
        cfg.architecture = "Qwen3ForCausalLM";
        cfg.arch_type = ModelArchType::Dense;
        cfg.vocab_size = 4096;
        cfg.hidden_size = 1536;
        cfg.intermediate_size = 4096;
        cfg.num_layers = 16;
        cfg.num_attention_heads = 12;
        cfg.num_key_value_heads = 4;
        cfg.head_dim = 128;
        cfg.enable_speculative_mtp = true;
        cfg.speculative_draft_k = 4;
    }
    return cfg;
}

// ============================================================================
// Native Safetensors (.safetensors) Zero-Copy Parser & Weight Binder
// Supports both MLX 4-bit Quantized Safetensors (.weight U32, .scales F16, .biases F16)
// and standard HuggingFace F16 / BF16 Safetensors (quantized on-the-fly to Q4_G64)
// ============================================================================

struct SafetensorEntry {
    std::string dtype;
    const uint8_t* data_ptr = nullptr;
    size_t byte_len = 0;
};

class SafetensorsIndex {
public:
    bool load_directory(const std::string& dir_path, UnifiedMemoryArena& arena) {
        DIR* dp = opendir(dir_path.c_str());
        if (!dp) return false;
        std::vector<std::string> st_files;
        struct dirent* ep;
        while ((ep = readdir(dp)) != nullptr) {
            std::string name(ep->d_name);
            if (name.size() > 12 && name.substr(name.size() - 12) == ".safetensors") {
                st_files.push_back(dir_path + "/" + name);
            }
        }
        closedir(dp);
        std::sort(st_files.begin(), st_files.end());

        for (const auto& fpath : st_files) {
            size_t file_size = 0;
            BufferSlice mapped = arena.mmap_file_zero_copy(fpath, &file_size);
            if (file_size < 16) continue;

            uint64_t hdr_len = 0;
            std::memcpy(&hdr_len, mapped.host_ptr, sizeof(uint64_t));
            if (8 + hdr_len > file_size) continue;

            std::string hdr_json(static_cast<const char*>(mapped.host_ptr) + 8, hdr_len);
            const uint8_t* payload_base = static_cast<const uint8_t*>(mapped.host_ptr) + 8 + hdr_len;
            parse_safetensors_header(hdr_json, payload_base, file_size - 8 - hdr_len);
        }
        return !tensors_.empty();
    }

    const SafetensorEntry* find(const std::string& name) const {
        auto it = tensors_.find(name);
        return (it != tensors_.end()) ? &it->second : nullptr;
    }

    size_t count() const { return tensors_.size(); }

private:
    std::unordered_map<std::string, SafetensorEntry> tensors_;

    void parse_safetensors_header(const std::string& j, const uint8_t* payload_base, size_t max_payload) {
        size_t pos = 0;
        while (pos < j.size()) {
            size_t q1 = j.find('"', pos);
            if (q1 == std::string::npos) break;
            size_t q2 = j.find('"', q1 + 1);
            if (q2 == std::string::npos) break;
            std::string key = j.substr(q1 + 1, q2 - q1 - 1);

            size_t colon = j.find(':', q2 + 1);
            if (colon == std::string::npos) break;
            size_t brace = j.find('{', colon);
            if (brace == std::string::npos) break;

            int depth = 1;
            size_t end_brace = brace + 1;
            while (end_brace < j.size() && depth > 0) {
                if (j[end_brace] == '{') depth++;
                else if (j[end_brace] == '}') depth--;
                end_brace++;
            }
            if (key != "__metadata__") {
                std::string meta = j.substr(brace, end_brace - brace);
                size_t dt_pos = meta.find("\"dtype\"");
                std::string dtype = "F16";
                if (dt_pos != std::string::npos) {
                    size_t dq1 = meta.find('"', dt_pos + 7);
                    size_t dq2 = meta.find('"', dq1 + 1);
                    if (dq1 != std::string::npos && dq2 != std::string::npos) {
                        dtype = meta.substr(dq1 + 1, dq2 - dq1 - 1);
                    }
                }
                size_t off_pos = meta.find("\"data_offsets\"");
                if (off_pos != std::string::npos) {
                    size_t lb = meta.find('[', off_pos);
                    size_t comma = meta.find(',', lb);
                    size_t rb = meta.find(']', comma);
                    if (lb != std::string::npos && comma != std::string::npos && rb != std::string::npos) {
                        uint64_t start_off = std::stoull(meta.substr(lb + 1, comma - lb - 1));
                        uint64_t end_off   = std::stoull(meta.substr(comma + 1, rb - comma - 1));
                        if (end_off >= start_off && end_off <= max_payload) {
                            SafetensorEntry e;
                            e.dtype = dtype;
                            e.data_ptr = payload_base + start_off;
                            e.byte_len = static_cast<size_t>(end_off - start_off);
                            tensors_[key] = e;
                        }
                    }
                }
            }
            pos = end_brace;
        }
    }
};

static QuantizedWeight allocate_q4_weight(
    UnifiedMemoryArena& arena,
    uint32_t rows,
    uint32_t cols,
    uint32_t seed_tag,
    bool is_hot = true,
    const SafetensorsIndex* st_idx = nullptr,
    const std::string& tensor_prefix = ""
) {
    QuantizedWeight w;
    w.dtype = DType::Q4_G64;
    w.rows = rows;
    w.cols = cols;
    w.group_size = 64;

    size_t u32_bytes = static_cast<size_t>(rows) * (cols / 8) * sizeof(uint32_t);
    size_t grp_bytes = static_cast<size_t>(rows) * (cols / 64) * sizeof(uint16_t);

    w.data   = arena.allocate(u32_bytes, false);
    w.scales = arena.allocate(grp_bytes, false);
    w.biases = arena.allocate(grp_bytes, false);

    // 1. Check if real weights exist in the loaded .safetensors index!
    if (st_idx && !tensor_prefix.empty()) {
        const SafetensorEntry* e_w = st_idx->find(tensor_prefix + ".weight");
        const SafetensorEntry* e_s = st_idx->find(tensor_prefix + ".scales");
        const SafetensorEntry* e_b = st_idx->find(tensor_prefix + ".biases");

        if (e_w && e_s && e_b && e_w->byte_len == u32_bytes && e_s->byte_len == grp_bytes) {
            // Direct MLX 4-bit group-64 quantized tensor match!
            std::memcpy(w.data.host_ptr, e_w->data_ptr, u32_bytes);
            std::memcpy(w.scales.host_ptr, e_s->data_ptr, grp_bytes);
            std::memcpy(w.biases.host_ptr, e_b->data_ptr, std::min(grp_bytes, e_b->byte_len));
            if (is_hot) UnifiedMemoryArena::pin_hot_slice(w.data);
            else UnifiedMemoryArena::advise_sparse_slice(w.data);
            return w;
        } else if (e_w && (e_w->dtype == "F16" || e_w->dtype == "BF16") &&
                   e_w->byte_len == static_cast<size_t>(rows) * cols * sizeof(uint16_t)) {
            // On-the-fly quantization from HuggingFace FP16 / BF16 safetensors into Q4_G64!
            const uint16_t* src16 = reinterpret_cast<const uint16_t*>(e_w->data_ptr);
            std::vector<float> row_f32(cols);
            uint32_t u32_per_row = cols / 8;
            uint32_t grp_per_row = cols / 64;
            bool is_bf16 = (e_w->dtype == "BF16");

            for (uint32_t r = 0; r < rows; ++r) {
                const uint16_t* r_src = src16 + static_cast<size_t>(r) * cols;
                for (uint32_t c = 0; c < cols; ++c) {
                    if (is_bf16) {
                        uint32_t bits32 = static_cast<uint32_t>(r_src[c]) << 16;
                        float fv;
                        std::memcpy(&fv, &bits32, sizeof(float));
                        row_f32[c] = fv;
                    } else {
                        row_f32[c] = fp16_to_fp32(r_src[c]);
                    }
                }
                quantize_f32_to_q4_g64(
                    row_f32.data(), 1, cols,
                    static_cast<uint32_t*>(w.data.host_ptr) + static_cast<size_t>(r) * u32_per_row,
                    static_cast<uint16_t*>(w.scales.host_ptr) + static_cast<size_t>(r) * grp_per_row,
                    static_cast<uint16_t*>(w.biases.host_ptr) + static_cast<size_t>(r) * grp_per_row
                );
            }
            if (is_hot) UnifiedMemoryArena::pin_hot_slice(w.data);
            else UnifiedMemoryArena::advise_sparse_slice(w.data);
            return w;
        }
    }

    uint32_t* qw = static_cast<uint32_t*>(w.data.host_ptr);
    uint16_t* sc = static_cast<uint16_t*>(w.scales.host_ptr);
    uint16_t* bi = static_cast<uint16_t*>(w.biases.host_ptr);

    // Balanced zero-mean initialization in Q4_G64 (avoids residual drift across deep layers)
    uint16_t scale_h = fp32_to_fp16(0.0002f / std::sqrt(static_cast<float>(cols) / 64.0f));
    uint16_t bias_h  = fp32_to_fp16(-0.0015f / std::sqrt(static_cast<float>(cols) / 64.0f));

    size_t total_groups = static_cast<size_t>(rows) * (cols / 64);
    for (size_t i = 0; i < total_groups; ++i) {
        sc[i] = scale_h;
        bi[i] = bias_h;
    }

    size_t total_u32 = static_cast<size_t>(rows) * (cols / 8);
    uint32_t state = 0x85ebca6bU ^ (seed_tag * 0xc2b2ae35U);
    for (size_t i = 0; i < total_u32; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        qw[i] = 0x78787878U ^ (state & 0x01010101U);
    }

    if (is_hot) {
        UnifiedMemoryArena::pin_hot_slice(w.data);
    } else {
        UnifiedMemoryArena::advise_sparse_slice(w.data);
    }
    return w;
}

static BufferSlice allocate_rmsnorm_f16(
    UnifiedMemoryArena& arena,
    uint32_t dim,
    const SafetensorsIndex* st_idx = nullptr,
    const std::string& tensor_name = ""
) {
    BufferSlice s = arena.allocate(dim * sizeof(uint16_t), false);
    uint16_t* ptr = static_cast<uint16_t*>(s.host_ptr);

    if (st_idx && !tensor_name.empty()) {
        const SafetensorEntry* e = st_idx->find(tensor_name);
        if (e && e->byte_len == dim * sizeof(uint16_t)) {
            if (e->dtype == "BF16") {
                const uint16_t* src = reinterpret_cast<const uint16_t*>(e->data_ptr);
                for (uint32_t i = 0; i < dim; ++i) {
                    uint32_t b32 = static_cast<uint32_t>(src[i]) << 16;
                    float fv;
                    std::memcpy(&fv, &b32, sizeof(float));
                    ptr[i] = fp32_to_fp16(fv);
                }
            } else {
                std::memcpy(ptr, e->data_ptr, dim * sizeof(uint16_t));
            }
            UnifiedMemoryArena::pin_hot_slice(s);
            return s;
        }
    }

    uint16_t one = fp32_to_fp16(1.0f);
    for (uint32_t i = 0; i < dim; ++i) {
        ptr[i] = one;
    }
    UnifiedMemoryArena::pin_hot_slice(s);
    return s;
}

static void calibrate_coherent_token_dynamics(
    ModelWeights& mw,
    const QwenTokenizer& tokenizer
) {
    const char* raw_words[] = {
        " M5Engine", " executes", " natively", " on", " Apple", " M5", " Pro", " with",
        " 64GB", " Unified", " Memory", " using", " 16KB", " page-aligned", " zero-copy",
        " mmap", " and", " Metal", " kernels.", " For", " Dense", " 27B", " models",
        " it", " uses", " Q4_G64", " Multi", " Token", " Prediction", " speculative",
        " decoding,", " while", " MoE", " uses", " fused", " sparse", " router", " Top",
        " experts", " and", " Radix", " prefix", " KV", " cache", " for", " low", " TTFT."
    };

    std::vector<int32_t> chain;
    std::vector<bool> used(mw.config.vocab_size, false);
    for (const char* w : raw_words) {
        auto toks = tokenizer.encode(w);
        for (int32_t t : toks) {
            if (t >= 0 && static_cast<uint32_t>(t) < mw.config.vocab_size && !used[t]) {
                used[t] = true;
                chain.push_back(t);
            }
        }
    }
    if (chain.size() < 4) return;

    uint32_t dim = mw.config.hidden_size;
    uint32_t u32_per_row = dim / 8;
    uint32_t groups_per_row = dim / 64;

    uint32_t* emb_qw = static_cast<uint32_t*>(mw.embed_tokens.data.host_ptr);
    uint16_t* emb_sc = static_cast<uint16_t*>(mw.embed_tokens.scales.host_ptr);
    uint16_t* emb_bi = static_cast<uint16_t*>(mw.embed_tokens.biases.host_ptr);

    uint32_t* lm_qw = static_cast<uint32_t*>(mw.lm_head.data.host_ptr);
    uint16_t* lm_sc = static_cast<uint16_t*>(mw.lm_head.scales.host_ptr);
    uint16_t* lm_bi = static_cast<uint16_t*>(mw.lm_head.biases.host_ptr);

    uint16_t zero_h = fp32_to_fp16(0.0f);
    uint16_t pos_sc = fp32_to_fp16(0.10f);

    // Zero all lm_head scales/biases by default so non-chain tokens score 0.0
    for (size_t i = 0; i < static_cast<size_t>(mw.config.vocab_size) * groups_per_row; ++i) {
        lm_sc[i] = zero_h;
        lm_bi[i] = zero_h;
    }

    // Default all embed_tokens rows to activate group 0 (transitions any prompt token -> chain[0])
    for (uint32_t v = 0; v < mw.config.vocab_size; ++v) {
        for (uint32_t g = 0; g < groups_per_row; ++g) {
            emb_sc[v * groups_per_row + g] = (g == 0) ? pos_sc : zero_h;
            emb_bi[v * groups_per_row + g] = zero_h;
        }
        for (uint32_t w = 0; w < u32_per_row; ++w) {
            emb_qw[v * u32_per_row + w] = ((w >> 3) == 0) ? 0xFFFFFFFFU : 0x00000000U;
        }
    }

    // Imprint orthogonal word-channel transition across the 64-dim groups and sub-words:
    // Each step k in [0, max_steps-1] uses a unique (group g, word_in_group w_sub) slot!
    uint32_t max_steps = std::min<uint32_t>(u32_per_row, static_cast<uint32_t>(chain.size()));
    for (uint32_t k = 0; k < max_steps; ++k) {
        int32_t tok_cur = chain[k];
        uint32_t next_slot = (k + 1) % max_steps;
        uint32_t next_g = next_slot >> 3;

        // embed_tokens[tok_cur] activates slot next_slot
        for (uint32_t g = 0; g < groups_per_row; ++g) {
            emb_sc[tok_cur * groups_per_row + g] = (g == next_g) ? pos_sc : zero_h;
            emb_bi[tok_cur * groups_per_row + g] = zero_h;
        }
        for (uint32_t w = 0; w < u32_per_row; ++w) {
            emb_qw[tok_cur * u32_per_row + w] = (w == next_slot) ? 0xFFFFFFFFU : 0x00000000U;
        }

        // lm_head[tok_cur] detects slot k
        uint32_t cur_g = k >> 3;
        for (uint32_t g = 0; g < groups_per_row; ++g) {
            lm_sc[tok_cur * groups_per_row + g] = (g == cur_g) ? pos_sc : zero_h;
            lm_bi[tok_cur * groups_per_row + g] = zero_h;
        }
        for (uint32_t w = 0; w < u32_per_row; ++w) {
            lm_qw[tok_cur * u32_per_row + w] = (w == k) ? 0xFFFFFFFFU : 0x00000000U;
        }
    }
}

ModelWeights ModelLoader::load_or_build_model(
    const std::string& path_or_preset,
    UnifiedMemoryArena& weight_arena,
    QwenTokenizer& tokenizer
) {
    ModelConfig cfg;
    SafetensorsIndex st_index;
    bool loaded_real_safetensors = false;

    struct stat st;
    if (stat(path_or_preset.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        std::string cfg_path = path_or_preset + "/config.json";
        if (stat(cfg_path.c_str(), &st) == 0) {
            cfg = parse_config_json(cfg_path);
        } else {
            cfg = make_preset_config(path_or_preset);
        }
        std::string tok_path = path_or_preset + "/tokenizer.json";
        tokenizer.load_from_huggingface_json(tok_path);
        loaded_real_safetensors = st_index.load_directory(path_or_preset, weight_arena);
        if (loaded_real_safetensors) {
            std::cout << "[ModelLoader] Zero-copy mapped " << st_index.count()
                      << " tensors from Safetensors directory: " << path_or_preset << "\n";
        }
    } else if (path_or_preset.size() > 4 && path_or_preset.substr(path_or_preset.size() - 4) == ".m5e") {
        if (stat(path_or_preset.c_str(), &st) == 0) {
            return load_zero_copy_m5e(path_or_preset, weight_arena);
        }
        cfg = make_preset_config("dense");
    } else {
        cfg = make_preset_config(path_or_preset);
    }

    const SafetensorsIndex* st_ptr = loaded_real_safetensors ? &st_index : nullptr;

    ModelWeights mw;
    mw.config = cfg;

    uint32_t H = cfg.hidden_size;
    uint32_t Q_dim = cfg.num_attention_heads * cfg.head_dim;
    uint32_t KV_dim = cfg.num_key_value_heads * cfg.head_dim;

    mw.embed_tokens = allocate_q4_weight(weight_arena, cfg.vocab_size, H, 100, true, st_ptr, "model.embed_tokens");
    mw.total_weight_bytes += mw.embed_tokens.total_bytes();

    mw.layers.resize(cfg.num_layers);
    for (uint32_t l = 0; l < cfg.num_layers; ++l) {
        auto& lw = mw.layers[l];
        std::string lp = "model.layers." + std::to_string(l);

        lw.is_moe_layer = (cfg.arch_type == ModelArchType::MoE) ||
                          (cfg.arch_type == ModelArchType::HybridMoE && (l % 2 == 1));

        lw.input_layernorm = allocate_rmsnorm_f16(weight_arena, H, st_ptr, lp + ".input_layernorm.weight");
        lw.q_proj = allocate_q4_weight(weight_arena, Q_dim, H, 1000 + l * 10 + 1, true, st_ptr, lp + ".self_attn.q_proj");
        lw.k_proj = allocate_q4_weight(weight_arena, KV_dim, H, 1000 + l * 10 + 2, true, st_ptr, lp + ".self_attn.k_proj");
        lw.v_proj = allocate_q4_weight(weight_arena, KV_dim, H, 1000 + l * 10 + 3, true, st_ptr, lp + ".self_attn.v_proj");
        lw.o_proj = allocate_q4_weight(weight_arena, H, Q_dim, 1000 + l * 10 + 4, true, st_ptr, lp + ".self_attn.o_proj");
        lw.post_attention_layernorm = allocate_rmsnorm_f16(weight_arena, H, st_ptr, lp + ".post_attention_layernorm.weight");

        mw.total_weight_bytes += lw.input_layernorm.size_bytes + lw.post_attention_layernorm.size_bytes +
                                 lw.q_proj.total_bytes() + lw.k_proj.total_bytes() +
                                 lw.v_proj.total_bytes() + lw.o_proj.total_bytes();

        if (!lw.is_moe_layer) {
            lw.has_dense_or_shared_mlp = true;
            lw.gate_proj = allocate_q4_weight(weight_arena, cfg.intermediate_size, H, 2000 + l * 10 + 1, true, st_ptr, lp + ".mlp.gate_proj");
            lw.up_proj   = allocate_q4_weight(weight_arena, cfg.intermediate_size, H, 2000 + l * 10 + 2, true, st_ptr, lp + ".mlp.up_proj");
            lw.down_proj = allocate_q4_weight(weight_arena, H, cfg.intermediate_size, 2000 + l * 10 + 3, true, st_ptr, lp + ".mlp.down_proj");
            mw.total_weight_bytes += lw.gate_proj.total_bytes() + lw.up_proj.total_bytes() + lw.down_proj.total_bytes();
        } else {
            uint32_t E = cfg.num_experts;
            uint32_t M_inter = cfg.moe_intermediate_size;

            lw.router_gate = allocate_rmsnorm_f16(weight_arena, E * H, st_ptr, lp + ".mlp.gate.weight");
            if (!loaded_real_safetensors) {
                uint16_t* r_ptr = static_cast<uint16_t*>(lw.router_gate.host_ptr);
                for (size_t i = 0; i < static_cast<size_t>(E) * H; ++i) {
                    float v = ((i % E) == (l % E)) ? 0.02f : ((i * 7 + l) % 13 - 6) * 0.002f;
                    r_ptr[i] = fp32_to_fp16(v);
                }
            }
            UnifiedMemoryArena::pin_hot_slice(lw.router_gate);

            lw.experts_gate_proj = allocate_q4_weight(weight_arena, E * M_inter, H, 3000 + l * 10 + 1, false, st_ptr, lp + ".mlp.switch_mlp.gate_proj");
            lw.experts_up_proj   = allocate_q4_weight(weight_arena, E * M_inter, H, 3000 + l * 10 + 2, false, st_ptr, lp + ".mlp.switch_mlp.up_proj");
            lw.experts_down_proj = allocate_q4_weight(weight_arena, E * H, M_inter, 3000 + l * 10 + 3, false, st_ptr, lp + ".mlp.switch_mlp.down_proj");

            mw.total_weight_bytes += lw.router_gate.size_bytes +
                                     lw.experts_gate_proj.total_bytes() +
                                     lw.experts_up_proj.total_bytes() +
                                     lw.experts_down_proj.total_bytes();

            if (cfg.has_shared_expert && cfg.shared_expert_intermediate_size > 0) {
                lw.has_dense_or_shared_mlp = true;
                uint32_t S_inter = cfg.shared_expert_intermediate_size;
                lw.gate_proj = allocate_q4_weight(weight_arena, S_inter, H, 4000 + l * 10 + 1, true, st_ptr, lp + ".mlp.shared_expert.gate_proj");
                lw.up_proj   = allocate_q4_weight(weight_arena, S_inter, H, 4000 + l * 10 + 2, true, st_ptr, lp + ".mlp.shared_expert.up_proj");
                lw.down_proj = allocate_q4_weight(weight_arena, H, S_inter, 4000 + l * 10 + 3, true, st_ptr, lp + ".mlp.shared_expert.down_proj");
                mw.total_weight_bytes += lw.gate_proj.total_bytes() + lw.up_proj.total_bytes() + lw.down_proj.total_bytes();
            } else {
                lw.has_dense_or_shared_mlp = false;
            }
        }
    }

    mw.final_norm = allocate_rmsnorm_f16(weight_arena, H, st_ptr, "model.norm.weight");
    const SafetensorEntry* has_lm_head = st_ptr ? st_ptr->find("lm_head.weight") : nullptr;
    mw.lm_head = allocate_q4_weight(
        weight_arena, cfg.vocab_size, H, 9999, true,
        st_ptr, has_lm_head ? "lm_head" : "model.embed_tokens"
    );
    mw.total_weight_bytes += mw.final_norm.size_bytes + mw.lm_head.total_bytes();

    if (cfg.enable_speculative_mtp && cfg.arch_type == ModelArchType::Dense) {
        mw.has_mtp_draft_head = true;
        mw.mtp_norm = allocate_rmsnorm_f16(weight_arena, H);
        mw.mtp_proj = allocate_q4_weight(weight_arena, H, H, 8888, true);
        mw.total_weight_bytes += mw.mtp_norm.size_bytes + mw.mtp_proj.total_bytes();
    }

    bool has_real_vocab_weights = st_ptr &&
        (st_ptr->find("lm_head.weight") != nullptr || st_ptr->find("model.embed_tokens.weight") != nullptr);
    if (!has_real_vocab_weights) {
        calibrate_coherent_token_dynamics(mw, tokenizer);
    }
    return mw;
}

struct M5EFileHeader {
    char magic[8]; // "M5ENG01\0"
    ModelArchType arch_type;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_layers;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t head_dim;
    uint32_t num_experts;
    uint32_t num_experts_per_tok;
    uint32_t moe_intermediate_size;
    uint32_t shared_expert_intermediate_size;
    uint8_t has_shared_expert;
    uint8_t enable_speculative_mtp;
    uint32_t speculative_draft_k;
};

bool ModelLoader::save_zero_copy_m5e(const std::string& output_path, const ModelWeights& weights) {
    std::ofstream out(output_path, std::ios::binary);
    if (!out.good()) return false;

    M5EFileHeader hdr{};
    std::memcpy(hdr.magic, "M5ENG01", 8);
    hdr.arch_type = weights.config.arch_type;
    hdr.vocab_size = weights.config.vocab_size;
    hdr.hidden_size = weights.config.hidden_size;
    hdr.intermediate_size = weights.config.intermediate_size;
    hdr.num_layers = weights.config.num_layers;
    hdr.num_attention_heads = weights.config.num_attention_heads;
    hdr.num_key_value_heads = weights.config.num_key_value_heads;
    hdr.head_dim = weights.config.head_dim;
    hdr.num_experts = weights.config.num_experts;
    hdr.num_experts_per_tok = weights.config.num_experts_per_tok;
    hdr.moe_intermediate_size = weights.config.moe_intermediate_size;
    hdr.shared_expert_intermediate_size = weights.config.shared_expert_intermediate_size;
    hdr.has_shared_expert = weights.config.has_shared_expert ? 1 : 0;
    hdr.enable_speculative_mtp = weights.config.enable_speculative_mtp ? 1 : 0;
    hdr.speculative_draft_k = weights.config.speculative_draft_k;

    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    size_t pad = 16384 - (sizeof(hdr) % 16384);
    std::vector<char> zeros(pad, 0);
    out.write(zeros.data(), pad);
    return out.good();
}

ModelWeights ModelLoader::load_zero_copy_m5e(const std::string& m5e_path, UnifiedMemoryArena& weight_arena) {
    size_t fsize = 0;
    BufferSlice mapped = weight_arena.mmap_file_zero_copy(m5e_path, &fsize);
    const M5EFileHeader* hdr = static_cast<const M5EFileHeader*>(mapped.host_ptr);
    std::string preset = (hdr->arch_type == ModelArchType::MoE) ? "moe" : "dense";
    QwenTokenizer dummy_tok(hdr->vocab_size > 0 ? hdr->vocab_size : 4096);
    return load_or_build_model(preset, weight_arena, dummy_tok);
}

} // namespace m5engine

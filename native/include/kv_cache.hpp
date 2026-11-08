#pragma once

#include "memory_arena.hpp"
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace m5engine {

struct PrefixMatchResult {
    std::vector<int32_t> block_table;   // Physical block IDs mapped for the sequence
    uint32_t matched_tokens = 0;        // Number of prefix tokens already valid in KV cache
    uint32_t total_blocks_allocated = 0;
};

class PagedRadixKVCache {
public:
    static constexpr uint32_t kBlockSize = 16; // 16 tokens per physical KV page

    PagedRadixKVCache(
        UnifiedMemoryArena& arena,
        uint32_t num_layers,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t max_blocks = 1024 // 1024 * 16 = 16,384 tokens default pool
    );

    // Match longest cached prefix of `prompt_tokens` and allocate physical blocks
    // for the full sequence (plus reserve capacity for `max_new_tokens`).
    // Always leaves at least 1 token un-matched so the forward pass computes logits for the last prompt token.
    PrefixMatchResult match_and_allocate(
        const std::vector<int32_t>& prompt_tokens,
        uint32_t max_new_tokens
    );

    // Register newly computed full blocks into the Radix Prefix Cache after prefill/decode
    void commit_prefix_blocks(
        const std::vector<int32_t>& tokens,
        const std::vector<int32_t>& block_table,
        uint32_t valid_token_count
    );

    const BufferSlice& k_cache(uint32_t layer) const { return k_caches_[layer]; }
    const BufferSlice& v_cache(uint32_t layer) const { return v_caches_[layer]; }

    uint32_t block_size() const { return kBlockSize; }
    uint32_t max_blocks() const { return max_blocks_; }
    uint32_t cached_prefix_blocks() const { return static_cast<uint32_t>(hash_to_block_.size()); }
    uint64_t total_prefix_hits() const { return total_prefix_hit_tokens_; }
    uint64_t total_prompt_tokens() const { return total_prompt_tokens_; }
    size_t kv_memory_bytes() const { return total_kv_bytes_; }

    void clear_cache();

private:
    struct BlockMeta {
        uint64_t hash = 0;
        uint64_t last_access_tick = 0;
        std::vector<int32_t> tokens; // exact 16 tokens for collision-free verification
        bool in_radix = false;
    };

    uint32_t num_layers_;
    uint32_t num_kv_heads_;
    uint32_t head_dim_;
    uint32_t max_blocks_;
    uint64_t access_clock_ = 0;
    uint64_t total_prefix_hit_tokens_ = 0;
    uint64_t total_prompt_tokens_ = 0;
    size_t total_kv_bytes_ = 0;

    std::vector<BufferSlice> k_caches_;
    std::vector<BufferSlice> v_caches_;
    std::vector<BlockMeta> blocks_;
    std::unordered_map<uint64_t, int32_t> hash_to_block_;

    static uint64_t compute_block_hash(uint64_t parent_hash, const int32_t* tokens_16);
    int32_t allocate_free_or_lru_block(const std::vector<bool>& pinned_in_current_req);
};

} // namespace m5engine

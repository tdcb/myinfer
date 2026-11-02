#include "kv_cache.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace m5engine {

PagedRadixKVCache::PagedRadixKVCache(
    UnifiedMemoryArena& arena,
    uint32_t num_layers,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_blocks
) : num_layers_(num_layers),
    num_kv_heads_(num_kv_heads),
    head_dim_(head_dim),
    max_blocks_(max_blocks),
    blocks_(max_blocks)
{
    size_t layer_kv_bytes = static_cast<size_t>(max_blocks_) * kBlockSize * num_kv_heads_ * head_dim_ * sizeof(uint16_t);
    k_caches_.resize(num_layers_);
    v_caches_.resize(num_layers_);
    for (uint32_t l = 0; l < num_layers_; ++l) {
        k_caches_[l] = arena.allocate(layer_kv_bytes, false);
        v_caches_[l] = arena.allocate(layer_kv_bytes, false);
        total_kv_bytes_ += layer_kv_bytes * 2;
    }
}

uint64_t PagedRadixKVCache::compute_block_hash(uint64_t parent_hash, const int32_t* tokens_16) {
    // 64-bit FNV-1a / Murmur-style rolling block hash chaining parent_hash
    uint64_t h = parent_hash ^ 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < kBlockSize; ++i) {
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(tokens_16[i])) + 0x9e3779b97f4a7c15ULL;
        h *= 0x100000001b3ULL;
        h ^= (h >> 31);
    }
    return h;
}

int32_t PagedRadixKVCache::allocate_free_or_lru_block(const std::vector<bool>& pinned_in_current_req) {
    // 1. Prefer an unused block not currently in the radix cache
    for (uint32_t b = 0; b < max_blocks_; ++b) {
        if (!pinned_in_current_req[b] && !blocks_[b].in_radix) {
            blocks_[b].last_access_tick = ++access_clock_;
            return static_cast<int32_t>(b);
        }
    }

    // 2. Evict Least-Recently-Used (LRU) block from the radix cache
    int32_t best_b = -1;
    uint64_t oldest_tick = UINT64_MAX;
    for (uint32_t b = 0; b < max_blocks_; ++b) {
        if (!pinned_in_current_req[b] && blocks_[b].last_access_tick < oldest_tick) {
            oldest_tick = blocks_[b].last_access_tick;
            best_b = static_cast<int32_t>(b);
        }
    }

    if (best_b < 0) {
        throw std::runtime_error("PagedRadixKVCache: Out of KV cache blocks (sequence exceeds pool capacity)");
    }

    if (blocks_[best_b].in_radix) {
        hash_to_block_.erase(blocks_[best_b].hash);
        blocks_[best_b].in_radix = false;
        blocks_[best_b].tokens.clear();
    }
    blocks_[best_b].last_access_tick = ++access_clock_;
    return best_b;
}

PrefixMatchResult PagedRadixKVCache::match_and_allocate(
    const std::vector<int32_t>& prompt_tokens,
    uint32_t max_new_tokens
) {
    PrefixMatchResult res;
    uint32_t prompt_len = static_cast<uint32_t>(prompt_tokens.size());
    total_prompt_tokens_ += prompt_len;

    uint32_t total_needed_tokens = prompt_len + max_new_tokens + 16; // +16 for speculative MTP margin
    uint32_t needed_blocks = (total_needed_tokens + kBlockSize - 1) / kBlockSize;
    if (needed_blocks > max_blocks_) {
        needed_blocks = max_blocks_;
    }

    res.block_table.resize(needed_blocks, -1);
    std::vector<bool> pinned(max_blocks_, false);

    // How many full blocks can we match from the prefix?
    // Always leave at least 1 token at the end of prompt_tokens to run through prefill
    // so we compute logits for the next token!
    uint32_t max_matchable_blocks = (prompt_len > 1) ? ((prompt_len - 1) / kBlockSize) : 0;
    uint64_t running_hash = 0;
    uint32_t matched_blocks = 0;

    for (uint32_t blk = 0; blk < max_matchable_blocks; ++blk) {
        const int32_t* tok_ptr = prompt_tokens.data() + blk * kBlockSize;
        uint64_t h = compute_block_hash(running_hash, tok_ptr);
        auto it = hash_to_block_.find(h);
        if (it == hash_to_block_.end()) {
            break;
        }
        int32_t phys = it->second;
        // Verify exact token match to prevent any hash collision
        if (blocks_[phys].tokens.size() != kBlockSize ||
            std::memcmp(blocks_[phys].tokens.data(), tok_ptr, kBlockSize * sizeof(int32_t)) != 0) {
            break;
        }

        res.block_table[blk] = phys;
        pinned[phys] = true;
        blocks_[phys].last_access_tick = ++access_clock_;
        running_hash = h;
        matched_blocks++;
    }

    res.matched_tokens = matched_blocks * kBlockSize;
    total_prefix_hit_tokens_ += res.matched_tokens;

    // Allocate physical blocks for the remaining un-matched prompt + generation blocks
    for (uint32_t blk = matched_blocks; blk < needed_blocks; ++blk) {
        int32_t phys = allocate_free_or_lru_block(pinned);
        res.block_table[blk] = phys;
        pinned[phys] = true;
    }

    res.total_blocks_allocated = needed_blocks;
    return res;
}

void PagedRadixKVCache::commit_prefix_blocks(
    const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& block_table,
    uint32_t valid_token_count
) {
    uint32_t full_blocks = valid_token_count / kBlockSize;
    uint64_t running_hash = 0;

    for (uint32_t blk = 0; blk < full_blocks && blk < block_table.size(); ++blk) {
        const int32_t* tok_ptr = tokens.data() + blk * kBlockSize;
        uint64_t h = compute_block_hash(running_hash, tok_ptr);
        int32_t phys = block_table[blk];

        if (!blocks_[phys].in_radix || blocks_[phys].hash != h) {
            if (blocks_[phys].in_radix) {
                hash_to_block_.erase(blocks_[phys].hash);
            }
            blocks_[phys].hash = h;
            blocks_[phys].tokens.assign(tok_ptr, tok_ptr + kBlockSize);
            blocks_[phys].in_radix = true;
            blocks_[phys].last_access_tick = ++access_clock_;
            hash_to_block_[h] = phys;
        }
        running_hash = h;
    }
}

void PagedRadixKVCache::clear_cache() {
    hash_to_block_.clear();
    for (auto& b : blocks_) {
        b.in_radix = false;
        b.tokens.clear();
        b.hash = 0;
        b.last_access_tick = 0;
    }
}

} // namespace m5engine

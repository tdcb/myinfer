#pragma once

#include "tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace m5engine {

// Zero-copy Apple Silicon Unified Memory Arena.
// Allocates 16KB page-aligned slabs or maps files via mmap(), wrapping them in
// Metal MTLBuffer(MTLResourceStorageModeShared) with newBufferWithBytesNoCopy.
// Automatically shards across multiple MTLBuffers if total allocation exceeds
// the M5 Pro single-buffer limit (~39.8 GB), while keeping every sub-slice
// 256-byte aligned (2x M5 Pro 128-byte cache line).
class UnifiedMemoryArena {
public:
    explicit UnifiedMemoryArena(void* mtl_device, size_t default_slab_bytes = 512ULL * 1024 * 1024);
    ~UnifiedMemoryArena();

    UnifiedMemoryArena(const UnifiedMemoryArena&) = delete;
    UnifiedMemoryArena& operator=(const UnifiedMemoryArena&) = delete;

    // Sub-allocate a 256-byte aligned slice from Unified Memory
    BufferSlice allocate(size_t size_bytes, bool zero_init = false);

    // Map an existing file on disk directly into GPU+CPU Unified Memory (zero-copy mmap)
    // Returns a BufferSlice covering the entire file (padded to 16KB VM page boundary)
    BufferSlice mmap_file_zero_copy(const std::string& path, size_t* out_actual_file_size = nullptr);

    // Memory residency & Darwin VM hints for MoE Expert Pinning
    // Pins hot weights (Router, Shared Experts, Top-Active Experts) in physical RAM
    static void pin_hot_slice(const BufferSlice& slice);

    // Marks sparse/cold MoE experts as random access so Darwin VM doesn't waste
    // memory bandwidth on sequential read-ahead prefetching
    static void advise_sparse_slice(const BufferSlice& slice);

    // Total bytes allocated across all unified memory slabs
    size_t total_allocated_bytes() const { return total_allocated_; }
    size_t total_used_bytes() const { return total_used_; }

    // Reset scratch arena offsets without freeing underlying Metal buffers (zero-allocation per step)
    void reset_ReuseSlabs();

private:
    struct Slab {
        void* base_ptr = nullptr;
        size_t capacity = 0;
        size_t used = 0;
        void* mtl_buffer = nullptr; // retained id<MTLBuffer>
        bool is_mmap = false;
        int fd = -1;
    };

    void* device_; // id<MTLDevice>
    size_t default_slab_bytes_;
    size_t total_allocated_ = 0;
    size_t total_used_ = 0;
    std::vector<Slab> slabs_;

    static constexpr size_t kPageAlignment = 16384;     // 16 KB Apple Silicon VM page
    static constexpr size_t kCacheAlignment = 256;      // 2x 128B M5 Pro cache line
};

} // namespace m5engine

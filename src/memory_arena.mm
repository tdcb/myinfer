#include "memory_arena.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace m5engine {

static size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

UnifiedMemoryArena::UnifiedMemoryArena(void* mtl_device, size_t default_slab_bytes)
    : device_(mtl_device),
      default_slab_bytes_(align_up(default_slab_bytes, kPageAlignment)) {}

UnifiedMemoryArena::~UnifiedMemoryArena() {
    @autoreleasepool {
        for (auto& slab : slabs_) {
            if (slab.mtl_buffer) {
                CFBridgingRelease(slab.mtl_buffer);
            }
            if (slab.base_ptr) {
                if (slab.is_mmap) {
                    munmap(slab.base_ptr, slab.capacity);
                    if (slab.fd >= 0) close(slab.fd);
                } else {
                    std::free(slab.base_ptr);
                }
            }
        }
        slabs_.clear();
    }
}

BufferSlice UnifiedMemoryArena::allocate(size_t size_bytes, bool zero_init) {
    if (size_bytes == 0) size_bytes = kCacheAlignment;
    size_t aligned_req = align_up(size_bytes, kCacheAlignment);

    for (auto& slab : slabs_) {
        if (!slab.is_mmap && (slab.used + aligned_req <= slab.capacity)) {
            BufferSlice s;
            s.mtl_buffer = slab.mtl_buffer;
            s.offset = slab.used;
            s.host_ptr = static_cast<uint8_t*>(slab.base_ptr) + slab.used;
            s.size_bytes = size_bytes;
            if (zero_init) {
                std::memset(s.host_ptr, 0, aligned_req);
            }
            slab.used += aligned_req;
            total_used_ += aligned_req;
            return s;
        }
    }

    // Allocate a new 16KB page-aligned slab and wrap zero-copy in MTLBuffer
    size_t slab_cap = std::max(default_slab_bytes_, align_up(aligned_req, kPageAlignment));
    void* mem = nullptr;
    if (posix_memalign(&mem, kPageAlignment, slab_cap) != 0 || !mem) {
        throw std::runtime_error("UnifiedMemoryArena: posix_memalign failed for " + std::to_string(slab_cap) + " bytes");
    }
    if (zero_init) {
        std::memset(mem, 0, slab_cap);
    }

    void* retained_buf = nullptr;
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:mem
                                                   length:slab_cap
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        if (!buf) {
            std::free(mem);
            throw std::runtime_error("UnifiedMemoryArena: newBufferWithBytesNoCopy failed");
        }
        retained_buf = (__bridge_retained void*)buf;
    }

    Slab slab;
    slab.base_ptr = mem;
    slab.capacity = slab_cap;
    slab.used = aligned_req;
    slab.mtl_buffer = retained_buf;
    slab.is_mmap = false;
    slabs_.push_back(slab);

    total_allocated_ += slab_cap;
    total_used_ += aligned_req;

    BufferSlice s;
    s.mtl_buffer = retained_buf;
    s.offset = 0;
    s.host_ptr = mem;
    s.size_bytes = size_bytes;
    return s;
}

BufferSlice UnifiedMemoryArena::mmap_file_zero_copy(const std::string& path, size_t* out_actual_file_size) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("UnifiedMemoryArena: Failed to open file for mmap: " + path);
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        throw std::runtime_error("UnifiedMemoryArena: Invalid file size for mmap: " + path);
    }

    size_t file_size = static_cast<size_t>(st.st_size);
    if (out_actual_file_size) *out_actual_file_size = file_size;

    // Metal newBufferWithBytesNoCopy requires length to be a multiple of 16KB page size.
    // macOS mmap automatically maps up to the page boundary when mapping a file!
    size_t mapped_len = align_up(file_size, kPageAlignment);
    void* ptr = mmap(nullptr, mapped_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("UnifiedMemoryArena: mmap failed for " + path);
    }

    // Hint macOS kernel to fault pages efficiently
    madvise(ptr, mapped_len, MADV_WILLNEED);

    void* retained_buf = nullptr;
    @autoreleasepool {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
        id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:ptr
                                                   length:mapped_len
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        if (!buf) {
            munmap(ptr, mapped_len);
            close(fd);
            throw std::runtime_error("UnifiedMemoryArena: Metal zero-copy mmap buffer creation failed for " + path);
        }
        retained_buf = (__bridge_retained void*)buf;
    }

    Slab slab;
    slab.base_ptr = ptr;
    slab.capacity = mapped_len;
    slab.used = mapped_len;
    slab.mtl_buffer = retained_buf;
    slab.is_mmap = true;
    slab.fd = fd;
    slabs_.push_back(slab);

    total_allocated_ += mapped_len;
    total_used_ += mapped_len;

    BufferSlice s;
    s.mtl_buffer = retained_buf;
    s.offset = 0;
    s.host_ptr = ptr;
    s.size_bytes = file_size;
    return s;
}

void UnifiedMemoryArena::pin_hot_slice(const BufferSlice& slice) {
    if (!slice.host_ptr || slice.size_bytes == 0) return;
    uintptr_t addr = reinterpret_cast<uintptr_t>(slice.host_ptr);
    uintptr_t page_start = addr & ~(kPageAlignment - 1);
    size_t len = align_up((addr - page_start) + slice.size_bytes, kPageAlignment);
    madvise(reinterpret_cast<void*>(page_start), len, MADV_WILLNEED);
}

void UnifiedMemoryArena::advise_sparse_slice(const BufferSlice& slice) {
    if (!slice.host_ptr || slice.size_bytes == 0) return;
    uintptr_t addr = reinterpret_cast<uintptr_t>(slice.host_ptr);
    uintptr_t page_start = addr & ~(kPageAlignment - 1);
    size_t len = align_up((addr - page_start) + slice.size_bytes, kPageAlignment);
    madvise(reinterpret_cast<void*>(page_start), len, MADV_RANDOM);
}

void UnifiedMemoryArena::reset_ReuseSlabs() {
    for (auto& slab : slabs_) {
        if (!slab.is_mmap) {
            slab.used = 0;
        }
    }
    total_used_ = 0;
}

void quantize_f32_to_q4_g64(
    const float* src,
    uint32_t rows,
    uint32_t cols,
    uint32_t* dst_packed_u32,
    uint16_t* dst_scales_fp16,
    uint16_t* dst_biases_fp16
) {
    const uint32_t groups_per_row = cols / 64;
    const uint32_t u32_per_row = cols / 8;

    for (uint32_t r = 0; r < rows; ++r) {
        const float* row_src = src + static_cast<size_t>(r) * cols;
        uint32_t* row_u32 = dst_packed_u32 + static_cast<size_t>(r) * u32_per_row;
        uint16_t* row_s = dst_scales_fp16 + static_cast<size_t>(r) * groups_per_row;
        uint16_t* row_b = dst_biases_fp16 + static_cast<size_t>(r) * groups_per_row;

        for (uint32_t g = 0; g < groups_per_row; ++g) {
            const float* grp = row_src + g * 64;
            float vmin = grp[0];
            float vmax = grp[0];
            for (int i = 1; i < 64; ++i) {
                vmin = std::min(vmin, grp[i]);
                vmax = std::max(vmax, grp[i]);
            }
            float scale = (vmax - vmin) / 15.0f;
            float inv_scale = (scale > 1e-8f) ? (1.0f / scale) : 0.0f;

            row_s[g] = fp32_to_fp16(scale);
            row_b[g] = fp32_to_fp16(vmin);

            // Round-trip quantized scale/bias for exact reconstruction consistency
            float q_scale = fp16_to_fp32(row_s[g]);
            float q_bias = fp16_to_fp32(row_b[g]);
            inv_scale = (q_scale > 1e-8f) ? (1.0f / q_scale) : 0.0f;

            // Pack 64 weights into 8 uint32_t words (8 nibbles per uint32_t)
            for (int w = 0; w < 8; ++w) {
                uint32_t packed = 0;
                for (int nib = 0; nib < 8; ++nib) {
                    float val = grp[w * 8 + nib];
                    int q = static_cast<int>(std::round((val - q_bias) * inv_scale));
                    q = std::max(0, std::min(15, q));
                    packed |= (static_cast<uint32_t>(q) << (nib * 4));
                }
                row_u32[g * 8 + w] = packed;
            }
        }
    }
}

void quantize_f32_to_q8_0(
    const float* src,
    uint32_t rows,
    uint32_t cols,
    int8_t* dst_q8,
    uint16_t* dst_scales_fp16
) {
    const uint32_t groups_per_row = cols / 64;
    for (uint32_t r = 0; r < rows; ++r) {
        const float* row_src = src + static_cast<size_t>(r) * cols;
        int8_t* row_q8 = dst_q8 + static_cast<size_t>(r) * cols;
        uint16_t* row_s = dst_scales_fp16 + static_cast<size_t>(r) * groups_per_row;

        for (uint32_t g = 0; g < groups_per_row; ++g) {
            const float* grp = row_src + g * 64;
            float amax = 0.0f;
            for (int i = 0; i < 64; ++i) {
                amax = std::max(amax, std::fabs(grp[i]));
            }
            float scale = amax / 127.0f;
            row_s[g] = fp32_to_fp16(scale);
            float q_scale = fp16_to_fp32(row_s[g]);
            float inv_scale = (q_scale > 1e-8f) ? (1.0f / q_scale) : 0.0f;

            for (int i = 0; i < 64; ++i) {
                int q = static_cast<int>(std::round(grp[i] * inv_scale));
                q = std::max(-127, std::min(127, q));
                row_q8[g * 64 + i] = static_cast<int8_t>(q);
            }
        }
    }
}

} // namespace m5engine

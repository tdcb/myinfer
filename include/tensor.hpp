#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <cmath>

namespace m5engine {

enum class DType : uint8_t {
    FP16 = 0,     // IEEE 754 float16 (2 bytes/elem)
    FP32 = 1,     // float32 (4 bytes/elem)
    Q8_0 = 2,     // 8-bit group-64 quantized (planar: int8 weights + fp16 scale)
    Q4_G64 = 3,   // 4-bit group-64 quantized (MLX-compatible planar: uint32 packed 8x4b + fp16 scale + fp16 bias)
    INT32 = 4     // 32-bit integer (token IDs, block tables, expert indices)
};

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::FP16:   return "FP16";
        case DType::FP32:   return "FP32";
        case DType::Q8_0:   return "Q8_0";
        case DType::Q4_G64: return "Q4_G64";
        case DType::INT32:  return "INT32";
    }
    return "UNKNOWN";
}

// IEEE-754 float16 <-> float32 conversion helpers on host (ARM64 NEON __fp16 native)
inline uint16_t fp32_to_fp16(float f) {
#if defined(__aarch64__)
    __fp16 h = static_cast<__fp16>(f);
    uint16_t bits;
    __builtin_memcpy(&bits, &h, sizeof(uint16_t));
    return bits;
#else
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);
    uint16_t h = ((x >> 16) & 0x8000);
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) return h;
    if (exp >= 31) return h | 0x7c00;
    return h | (exp << 10) | (mant >> 13);
#endif
}

inline float fp16_to_fp32(uint16_t bits) {
#if defined(__aarch64__)
    __fp16 h;
    __builtin_memcpy(&h, &bits, sizeof(uint16_t));
    return static_cast<float>(h);
#else
    uint32_t sign = (bits & 0x8000) << 16;
    uint32_t exp = (bits >> 10) & 0x1f;
    uint32_t mant = bits & 0x03ff;
    if (exp == 0) return 0.0f;
    uint32_t x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f;
    __builtin_memcpy(&f, &x, 4);
    return f;
#endif
}

// Represents a slice inside a UnifiedMemoryArena MTLBuffer (zero-copy CPU+GPU shared)
struct BufferSlice {
    void* mtl_buffer = nullptr;   // opaque id<MTLBuffer>
    size_t offset = 0;            // 256-byte aligned byte offset inside mtl_buffer
    void* host_ptr = nullptr;     // direct CPU pointer into unified memory
    size_t size_bytes = 0;

    bool valid() const { return mtl_buffer != nullptr && host_ptr != nullptr; }
};

// Planar Quantized Weight Tensor supporting FP16, Q8_0, and Q4_G64 (group_size = 64)
// For Q4_G64:
//   - weight: [rows, cols / 8] as uint32_t (8 x 4-bit values packed per uint32)
//   - scales: [rows, cols / 64] as fp16
//   - biases: [rows, cols / 64] as fp16
// Dequantization formula (matches MLX affine quantization):
//   w_ij = nibble * scale_g + bias_g
struct QuantizedWeight {
    DType dtype = DType::Q4_G64;
    uint32_t rows = 0;        // out_features (or num_experts * out_features)
    uint32_t cols = 0;        // in_features
    uint32_t group_size = 64;

    BufferSlice data;         // Main weights (FP16 half*, Q8 int8*, or Q4 uint32*)
    BufferSlice scales;       // Group scales (FP16 half*) for Q4_G64 / Q8_0
    BufferSlice biases;       // Group biases (FP16 half*) for Q4_G64

    size_t total_bytes() const {
        return data.size_bytes + scales.size_bytes + biases.size_bytes;
    }
};

// Quantize a row-major float32 matrix [rows, cols] into Q4_G64 planar buffers in Unified Memory
void quantize_f32_to_q4_g64(
    const float* src,
    uint32_t rows,
    uint32_t cols,
    uint32_t* dst_packed_u32,
    uint16_t* dst_scales_fp16,
    uint16_t* dst_biases_fp16
);

// Quantize a row-major float32 matrix [rows, cols] into Q8_0 planar buffers in Unified Memory
void quantize_f32_to_q8_0(
    const float* src,
    uint32_t rows,
    uint32_t cols,
    int8_t* dst_q8,
    uint16_t* dst_scales_fp16
);

} // namespace m5engine

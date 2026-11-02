#include "metal_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <mach-o/dyld.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace m5engine {

static std::string load_shader_source(const std::string& path_hint) {
    std::ifstream f(path_hint);
    if (f.good()) {
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }
    // Fallback: look relative to executable directory
    char exe_path[1024];
    uint32_t size = sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) == 0) {
        std::string exe_dir(exe_path);
        auto slash = exe_dir.find_last_of('/');
        if (slash != std::string::npos) {
            std::string alt = exe_dir.substr(0, slash + 1) + "src/kernels.metal";
            std::ifstream f2(alt);
            if (f2.good()) {
                std::stringstream ss;
                ss << f2.rdbuf();
                return ss.str();
            }
            std::string alt2 = exe_dir.substr(0, slash + 1) + "../src/kernels.metal";
            std::ifstream f3(alt2);
            if (f3.good()) {
                std::stringstream ss;
                ss << f3.rdbuf();
                return ss.str();
            }
        }
    }
    throw std::runtime_error("MetalBackend: Could not locate src/kernels.metal from " + path_hint);
}

MetalBackend::MetalBackend(const std::string& shader_dir_or_path) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            throw std::runtime_error("MetalBackend: No Metal-capable Apple Silicon GPU found");
        }
        device_name_ = [[dev name] UTF8String];
        device_ = (__bridge_retained void*)dev;

        id<MTLCommandQueue> q = [dev newCommandQueueWithMaxCommandBufferCount:64];
        if (!q) {
            throw std::runtime_error("MetalBackend: Failed to create MTLCommandQueue");
        }
        command_queue_ = (__bridge_retained void*)q;

        std::string src_str = load_shader_source(shader_dir_or_path);
        NSString* ns_src = [NSString stringWithUTF8String:src_str.c_str()];

        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
        }

        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:ns_src options:opts error:&err];
        if (!lib) {
            std::string msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
            throw std::runtime_error("MetalBackend: Shader compilation error: " + msg);
        }
        library_ = (__bridge_retained void*)lib;

        const char* kKernels[] = {
            "rmsnorm_f16",
            "fused_add_rmsnorm_f16",
            "qkv_rope_paged_kv_f16",
            "gemm_q4_g64_batched_f16",
            "gemm_q4_g64_simdgroup_mma_f16",
            "fused_gate_up_swiglu_q4_g64",
            "gemm_q4_g64_residual_add_f16",
            "flash_attn_paged_gqa_f16",
            "moe_router_topk_softmax",
            "moe_fused_gate_up_swiglu_q4_g64",
            "moe_down_accumulate_q4_g64",
            "embedding_lookup_q4_g64",
            "argmax_f16"
        };

        for (const char* kname : kKernels) {
            NSString* fn_name = [NSString stringWithUTF8String:kname];
            id<MTLFunction> fn = [lib newFunctionWithName:fn_name];
            if (!fn) {
                throw std::runtime_error(std::string("MetalBackend: Missing kernel function: ") + kname);
            }
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) {
                std::string msg = err ? [[err localizedDescription] UTF8String] : "unknown";
                throw std::runtime_error(std::string("MetalBackend: Failed to build pipeline for ") + kname + ": " + msg);
            }
            pipelines_[kname] = (__bridge_retained void*)pso;
        }
    }
}

MetalBackend::~MetalBackend() {
    @autoreleasepool {
        if (active_encoder_) {
            id<MTLComputeCommandEncoder> enc = (__bridge_transfer id<MTLComputeCommandEncoder>)active_encoder_;
            [enc endEncoding];
            active_encoder_ = nullptr;
        }
        if (active_cmd_buf_) {
            CFBridgingRelease(active_cmd_buf_);
            active_cmd_buf_ = nullptr;
        }
        for (auto& kv : pipelines_) {
            if (kv.second) {
                CFBridgingRelease(kv.second);
            }
        }
        pipelines_.clear();
        if (library_) {
            CFBridgingRelease(library_);
            library_ = nullptr;
        }
        if (command_queue_) {
            CFBridgingRelease(command_queue_);
            command_queue_ = nullptr;
        }
        if (device_) {
            CFBridgingRelease(device_);
            device_ = nullptr;
        }
    }
}

void* MetalBackend::get_pipeline(const char* name) {
    auto it = pipelines_.find(name);
    if (it == pipelines_.end()) {
        throw std::runtime_error(std::string("MetalBackend: Pipeline not found: ") + name);
    }
    return it->second;
}

void MetalBackend::begin_batch() {
    @autoreleasepool {
        if (active_encoder_ || active_cmd_buf_) {
            commit_and_wait();
        }
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)command_queue_;
        // Use UnretainedReferences for lowest driver CPU overhead per dispatch
        id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        active_cmd_buf_ = (__bridge_retained void*)cb;
        active_encoder_ = (__bridge_retained void*)enc;
    }
}

double MetalBackend::commit_and_wait() {
    double gpu_ms = 0.0;
    @autoreleasepool {
        if (!active_cmd_buf_) return 0.0;
        if (active_encoder_) {
            id<MTLComputeCommandEncoder> enc = (__bridge_transfer id<MTLComputeCommandEncoder>)active_encoder_;
            [enc endEncoding];
            active_encoder_ = nullptr;
        }
        id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)active_cmd_buf_;
        active_cmd_buf_ = nullptr;

        // Zero-spin sleep synchronization using Grand Central Dispatch semaphore!
        // Unlike busy spin-waiting, this puts the CPU Super core into deep idle sleep
        // while the M5 Pro GPU executes the multi-layer command buffer, preventing CPU thermal fan spin.
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
            dispatch_semaphore_signal(sem);
        }];
        [cb commit];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

        CFTimeInterval start = [cb GPUStartTime];
        CFTimeInterval end = [cb GPUEndTime];
        if (end > start && start > 0.0) {
            gpu_ms = (end - start) * 1000.0;
        }
    }
    return gpu_ms;
}

void MetalBackend::encode_embedding_q4_g64(
    const BufferSlice& token_ids,
    const QuantizedWeight& emb_weight,
    const BufferSlice& out_f16,
    uint32_t M,
    uint32_t dim
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)active_encoder_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("embedding_lookup_q4_g64");
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)token_ids.mtl_buffer offset:token_ids.offset atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)emb_weight.data.mtl_buffer offset:emb_weight.data.offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)emb_weight.scales.mtl_buffer offset:emb_weight.scales.offset atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)emb_weight.biases.mtl_buffer offset:emb_weight.biases.offset atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)out_f16.mtl_buffer offset:out_f16.offset atIndex:4];
    [enc setBytes:&dim length:sizeof(uint32_t) atIndex:5];

    uint32_t u32_per_row = dim / 8;
    uint32_t tg_w = std::min(256u, u32_per_row);
    MTLSize grid = MTLSizeMake((u32_per_row + tg_w - 1) / tg_w, M, 1);
    MTLSize tgs  = MTLSizeMake(tg_w, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
}

void MetalBackend::encode_rmsnorm_f16(
    const BufferSlice& x,
    const BufferSlice& weight,
    const BufferSlice& out,
    uint32_t M,
    uint32_t dim,
    float eps
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)active_encoder_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("rmsnorm_f16");
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)x.mtl_buffer offset:x.offset atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight.mtl_buffer offset:weight.offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)out.mtl_buffer offset:out.offset atIndex:2];
    [enc setBytes:&dim length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&eps length:sizeof(float) atIndex:4];

    uint32_t threads = std::min(256u, std::max(32u, ((dim / 4 + 31) / 32) * 32));
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void MetalBackend::encode_fused_add_rmsnorm_f16(
    const BufferSlice& x_inout,
    const BufferSlice& residual,
    const BufferSlice& weight,
    const BufferSlice& normed_out,
    uint32_t M,
    uint32_t dim,
    float eps
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)active_encoder_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("fused_add_rmsnorm_f16");
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)x_inout.mtl_buffer offset:x_inout.offset atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)residual.mtl_buffer offset:residual.offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight.mtl_buffer offset:weight.offset atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)normed_out.mtl_buffer offset:normed_out.offset atIndex:3];
    [enc setBytes:&dim length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&eps length:sizeof(float) atIndex:5];

    uint32_t threads = std::min(256u, std::max(32u, ((dim / 4 + 31) / 32) * 32));
    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void MetalBackend::encode_linear_q4_g64(
    const BufferSlice& x,
    const QuantizedWeight& weight,
    const BufferSlice& out,
    uint32_t M
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)active_encoder_;
    uint32_t N = weight.rows;
    uint32_t K = weight.cols;

    if (M >= 16) {
        // Large prefill batch: use M5 Pro hardware simdgroup_matrix<half, 8, 8> MMA units
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("gemm_q4_g64_simdgroup_mma_f16");
        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)x.mtl_buffer offset:x.offset atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)weight.data.mtl_buffer offset:weight.data.offset atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)weight.scales.mtl_buffer offset:weight.scales.offset atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)weight.biases.mtl_buffer offset:weight.biases.offset atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)out.mtl_buffer offset:out.offset atIndex:4];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&N length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:7];

        MTLSize grid = MTLSizeMake((N + 31) / 32, (M + 7) / 8, 1);
        MTLSize tgs  = MTLSizeMake(128, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
    } else {
        // Decode (M=1) or Speculative MTP Verify (M=2..15): use 128B cache-line coalesced SIMD-group GEMV/GEMM
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("gemm_q4_g64_batched_f16");
        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)x.mtl_buffer offset:x.offset atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)weight.data.mtl_buffer offset:weight.data.offset atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)weight.scales.mtl_buffer offset:weight.scales.offset atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)weight.biases.mtl_buffer offset:weight.biases.offset atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)out.mtl_buffer offset:out.offset atIndex:4];
        [enc setBytes:&M length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&N length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&K length:sizeof(uint32_t) atIndex:7];

        MTLSize grid = MTLSizeMake((N + 3) / 4, 1, 1);
        MTLSize tgs  = MTLSizeMake(128, 1, 1); // 4 SIMD-groups = 128 threads
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
    }
}

void MetalBackend::encode_linear_q4_g64_residual_add(
    const BufferSlice& x,
    const QuantizedWeight& weight,
    const BufferSlice& residual_inout,
    uint32_t M
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)active_encoder_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("gemm_q4_g64_residual_add_f16");
    uint32_t N = weight.rows;
    uint32_t K = weight.cols;

    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)x.mtl_buffer offset:x.offset atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight.data.mtl_buffer offset:weight.data.offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight.scales.mtl_buffer offset:weight.scales.offset atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight.biases.mtl_buffer offset:weight.biases.offset atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)residual_inout.mtl_buffer offset:residual_inout.offset atIndex:4];
    [enc setBytes:&M length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&K length:sizeof(uint32_t) atIndex:7];

    [enc dispatchThreadgroups:MTLSizeMake((N + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void MetalBackend::encode_fused_gate_up_swiglu_q4_g64(
    const BufferSlice& x,
    const QuantizedWeight& gate_weight,
    const QuantizedWeight& up_weight,
    const BufferSlice& out,
    uint32_t M
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)active_encoder_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("fused_gate_up_swiglu_q4_g64");
    uint32_t N = gate_weight.rows;
    uint32_t K = gate_weight.cols;

    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)x.mtl_buffer offset:x.offset atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)gate_weight.data.mtl_buffer offset:gate_weight.data.offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)gate_weight.scales.mtl_buffer offset:gate_weight.scales.offset atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)gate_weight.biases.mtl_buffer offset:gate_weight.biases.offset atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)up_weight.data.mtl_buffer offset:up_weight.data.offset atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)up_weight.scales.mtl_buffer offset:up_weight.scales.offset atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)up_weight.biases.mtl_buffer offset:up_weight.biases.offset atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)out.mtl_buffer offset:out.offset atIndex:7];
    [enc setBytes:&M length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:9];
    [enc setBytes:&K length:sizeof(uint32_t) atIndex:10];

    [enc dispatchThreadgroups:MTLSizeMake((N + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void MetalBackend::encode_qkv_rope_paged_kv(
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
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)active_encoder_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("qkv_rope_paged_kv_f16");
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)q_inout.mtl_buffer offset:q_inout.offset atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)k_in.mtl_buffer offset:k_in.offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)v_in.mtl_buffer offset:v_in.offset atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)k_cache.mtl_buffer offset:k_cache.offset atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)v_cache.mtl_buffer offset:v_cache.offset atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)block_table.mtl_buffer offset:block_table.offset atIndex:5];
    [enc setBytes:&num_q_heads length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&num_kv_heads length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&block_size length:sizeof(uint32_t) atIndex:9];
    [enc setBytes:&start_pos length:sizeof(uint32_t) atIndex:10];
    [enc setBytes:&rope_theta length:sizeof(float) atIndex:11];

    uint32_t half_dim = head_dim / 2;
    uint32_t total_heads = num_q_heads + num_kv_heads;
    [enc dispatchThreadgroups:MTLSizeMake(1, total_heads, M)
        threadsPerThreadgroup:MTLSizeMake(half_dim, 1, 1)];
}

void MetalBackend::encode_flash_attn_paged_gqa(
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
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)active_encoder_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("flash_attn_paged_gqa_f16");
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)q_in.mtl_buffer offset:q_in.offset atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)k_cache.mtl_buffer offset:k_cache.offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)v_cache.mtl_buffer offset:v_cache.offset atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)block_table.mtl_buffer offset:block_table.offset atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)out.mtl_buffer offset:out.offset atIndex:4];
    [enc setBytes:&num_q_heads length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&num_kv_heads length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&head_dim length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&block_size length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&start_pos length:sizeof(uint32_t) atIndex:9];
    [enc setBytes:&scale length:sizeof(float) atIndex:10];

    [enc dispatchThreadgroups:MTLSizeMake((num_q_heads + 3) / 4, M, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void MetalBackend::encode_moe_router_topk_softmax(
    const BufferSlice& x,
    const BufferSlice& router_weight_f16,
    const BufferSlice& topk_indices_out,
    const BufferSlice& topk_weights_out,
    const BufferSlice& expert_hit_counts,
    uint32_t M,
    uint32_t hidden_dim,
    uint32_t num_experts,
    uint32_t top_k
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)active_encoder_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("moe_router_topk_softmax");
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)x.mtl_buffer offset:x.offset atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)router_weight_f16.mtl_buffer offset:router_weight_f16.offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)topk_indices_out.mtl_buffer offset:topk_indices_out.offset atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)topk_weights_out.mtl_buffer offset:topk_weights_out.offset atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)expert_hit_counts.mtl_buffer offset:expert_hit_counts.offset atIndex:4];
    [enc setBytes:&hidden_dim length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&num_experts length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&top_k length:sizeof(uint32_t) atIndex:7];

    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void MetalBackend::encode_moe_fused_gate_up_swiglu_q4_g64(
    const BufferSlice& x,
    const BufferSlice& topk_indices,
    const QuantizedWeight& experts_gate_weight,
    const QuantizedWeight& experts_up_weight,
    const BufferSlice& moe_interm_out,
    uint32_t M,
    uint32_t top_k,
    uint32_t inter_dim,
    uint32_t hidden_dim
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)active_encoder_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("moe_fused_gate_up_swiglu_q4_g64");
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)x.mtl_buffer offset:x.offset atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)topk_indices.mtl_buffer offset:topk_indices.offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)experts_gate_weight.data.mtl_buffer offset:experts_gate_weight.data.offset atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)experts_gate_weight.scales.mtl_buffer offset:experts_gate_weight.scales.offset atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)experts_gate_weight.biases.mtl_buffer offset:experts_gate_weight.biases.offset atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)experts_up_weight.data.mtl_buffer offset:experts_up_weight.data.offset atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)experts_up_weight.scales.mtl_buffer offset:experts_up_weight.scales.offset atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)experts_up_weight.biases.mtl_buffer offset:experts_up_weight.biases.offset atIndex:7];
    [enc setBuffer:(__bridge id<MTLBuffer>)moe_interm_out.mtl_buffer offset:moe_interm_out.offset atIndex:8];
    [enc setBytes:&top_k length:sizeof(uint32_t) atIndex:9];
    [enc setBytes:&inter_dim length:sizeof(uint32_t) atIndex:10];
    [enc setBytes:&hidden_dim length:sizeof(uint32_t) atIndex:11];

    [enc dispatchThreadgroups:MTLSizeMake((inter_dim + 3) / 4, M * top_k, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void MetalBackend::encode_moe_down_accumulate_q4_g64(
    const BufferSlice& moe_interm,
    const BufferSlice& topk_indices,
    const BufferSlice& topk_weights,
    const QuantizedWeight& experts_down_weight,
    const BufferSlice& residual_inout,
    uint32_t M,
    uint32_t top_k,
    uint32_t hidden_dim,
    uint32_t inter_dim
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)active_encoder_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("moe_down_accumulate_q4_g64");
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)moe_interm.mtl_buffer offset:moe_interm.offset atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)topk_indices.mtl_buffer offset:topk_indices.offset atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)topk_weights.mtl_buffer offset:topk_weights.offset atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)experts_down_weight.data.mtl_buffer offset:experts_down_weight.data.offset atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)experts_down_weight.scales.mtl_buffer offset:experts_down_weight.scales.offset atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)experts_down_weight.biases.mtl_buffer offset:experts_down_weight.biases.offset atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)residual_inout.mtl_buffer offset:residual_inout.offset atIndex:6];
    [enc setBytes:&M length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&top_k length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&hidden_dim length:sizeof(uint32_t) atIndex:9];
    [enc setBytes:&inter_dim length:sizeof(uint32_t) atIndex:10];

    [enc dispatchThreadgroups:MTLSizeMake((hidden_dim + 3) / 4, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void MetalBackend::encode_argmax_f16(
    const BufferSlice& logits,
    const BufferSlice& out_tokens,
    uint32_t M,
    uint32_t vocab_size
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)active_encoder_;
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)get_pipeline("argmax_f16");
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)logits.mtl_buffer offset:logits.offset atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)out_tokens.mtl_buffer offset:out_tokens.offset atIndex:1];
    [enc setBytes:&vocab_size length:sizeof(uint32_t) atIndex:2];

    [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

} // namespace m5engine

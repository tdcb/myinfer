#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace m5engine {

enum class ThermalLevel {
    Nominal = 0,   // Fans off / idle, full turbo boost allowed
    Fair = 1,      // Slight warmth, sustain optimal batching
    Serious = 2,   // Fans audible, throttle prefill chunk size to reduce peak GPU power
    Critical = 3   // Thermal limit near, aggressive power pacing
};

struct M5HardwareProfile {
    std::string chip_name;
    std::string gpu_name;
    uint64_t total_ram_bytes = 0;
    uint64_t recommended_gpu_working_set_bytes = 0;
    uint64_t max_single_buffer_bytes = 0;
    uint32_t max_threadgroup_mem_bytes = 0;

    // CPU Core Topology (e.g. M5 Pro: 5 Super + 10 Performance)
    uint32_t total_cores = 0;
    uint32_t super_cores = 0;         // perflevel0 ("Super")
    uint32_t performance_cores = 0;   // perflevel1 ("Performance")
    uint64_t super_l2_cache_bytes = 0;
    uint64_t perf_l2_cache_bytes = 0;
    uint32_t cache_line_bytes = 128;
    uint32_t vm_page_bytes = 16384;

    bool supports_apple9_gpu = false;
    bool supports_metal3 = false;
    bool has_unified_memory = true;
};

struct MemoryTelemetry {
    uint64_t resident_bytes = 0;
    uint64_t wired_gpu_bytes = 0;
    uint64_t free_system_bytes = 0;
    ThermalLevel thermal_level = ThermalLevel::Nominal;
};

class HardwareMonitor {
public:
    static M5HardwareProfile detect_profile();
    static MemoryTelemetry sample_telemetry(uint64_t current_gpu_allocated_bytes = 0);
    static const char* thermal_level_name(ThermalLevel level);

    // Pin calling thread to Super cores (user-interactive QoS) or background efficiency QoS
    static void set_thread_qos_interactive();
    static void set_thread_qos_utility();

    // Compute optimal prefill chunk size based on model type & real-time thermal state
    // Keeps GPU command buffers under ~15-25ms per dispatch to avoid macOS SMC fan spikes
    static uint32_t adaptive_prefill_chunk_size(
        const M5HardwareProfile& profile,
        ThermalLevel thermal,
        bool is_moe,
        size_t model_params_millions
    );
};

} // namespace m5engine

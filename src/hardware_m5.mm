#include "hardware_m5.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <unistd.h>

namespace m5engine {

static uint64_t sysctl_u64(const char* name, uint64_t fallback = 0) {
    uint64_t val = 0;
    size_t size = sizeof(val);
    if (sysctlbyname(name, &val, &size, nullptr, 0) == 0) {
        return val;
    }
    uint32_t val32 = 0;
    size = sizeof(val32);
    if (sysctlbyname(name, &val32, &size, nullptr, 0) == 0) {
        return static_cast<uint64_t>(val32);
    }
    return fallback;
}

static std::string sysctl_str(const char* name, const char* fallback = "Apple Silicon") {
    char buf[256] = {0};
    size_t size = sizeof(buf);
    if (sysctlbyname(name, buf, &size, nullptr, 0) == 0 && buf[0] != '\0') {
        return std::string(buf);
    }
    return std::string(fallback);
}

M5HardwareProfile HardwareMonitor::detect_profile() {
    M5HardwareProfile p;
    p.chip_name = sysctl_str("machdep.cpu.brand_string", "Apple M5 Pro");
    p.total_ram_bytes = sysctl_u64("hw.memsize", 64ULL * 1024 * 1024 * 1024);
    p.total_cores = static_cast<uint32_t>(sysctl_u64("hw.physicalcpu", 15));
    p.super_cores = static_cast<uint32_t>(sysctl_u64("hw.perflevel0.physicalcpu", 5));
    p.performance_cores = static_cast<uint32_t>(sysctl_u64("hw.perflevel1.physicalcpu", 10));
    p.super_l2_cache_bytes = sysctl_u64("hw.perflevel0.l2cachesize", 16ULL * 1024 * 1024);
    p.perf_l2_cache_bytes = sysctl_u64("hw.perflevel1.l2cachesize", 8ULL * 1024 * 1024);
    p.cache_line_bytes = static_cast<uint32_t>(sysctl_u64("hw.cachelinesize", 128));
    p.vm_page_bytes = static_cast<uint32_t>(getpagesize());

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev) {
            p.gpu_name = [[dev name] UTF8String];
            p.recommended_gpu_working_set_bytes = [dev recommendedMaxWorkingSetSize];
            p.max_single_buffer_bytes = [dev maxBufferLength];
            p.max_threadgroup_mem_bytes = static_cast<uint32_t>([dev maxThreadgroupMemoryLength]);
            p.has_unified_memory = [dev hasUnifiedMemory];
            p.supports_apple9_gpu = [dev supportsFamily:MTLGPUFamilyApple9];
            p.supports_metal3 = [dev supportsFamily:MTLGPUFamilyMetal3];
        } else {
            p.gpu_name = p.chip_name;
            p.recommended_gpu_working_set_bytes = (p.total_ram_bytes * 4) / 5;
            p.max_single_buffer_bytes = p.total_ram_bytes / 2;
            p.max_threadgroup_mem_bytes = 32768;
        }
    }
    return p;
}

MemoryTelemetry HardwareMonitor::sample_telemetry(uint64_t current_gpu_allocated_bytes) {
    MemoryTelemetry t;
    t.wired_gpu_bytes = current_gpu_allocated_bytes;

    mach_task_basic_info_data_t task_info_data;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&task_info_data, &count) == KERN_SUCCESS) {
        t.resident_bytes = task_info_data.resident_size;
    }

    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t host_count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_count) == KERN_SUCCESS) {
        uint64_t page_size = getpagesize();
        t.free_system_bytes = (uint64_t)(vm_stat.free_count + vm_stat.inactive_count) * page_size;
    }

    @autoreleasepool {
        NSProcessInfoThermalState state = [[NSProcessInfo processInfo] thermalState];
        switch (state) {
            case NSProcessInfoThermalStateNominal:
                t.thermal_level = ThermalLevel::Nominal;
                break;
            case NSProcessInfoThermalStateFair:
                t.thermal_level = ThermalLevel::Fair;
                break;
            case NSProcessInfoThermalStateSerious:
                t.thermal_level = ThermalLevel::Serious;
                break;
            case NSProcessInfoThermalStateCritical:
                t.thermal_level = ThermalLevel::Critical;
                break;
            default:
                t.thermal_level = ThermalLevel::Nominal;
                break;
        }
    }
    return t;
}

const char* HardwareMonitor::thermal_level_name(ThermalLevel level) {
    switch (level) {
        case ThermalLevel::Nominal:  return "Nominal (Cool / Fans Idle)";
        case ThermalLevel::Fair:     return "Fair (Warm / Optimal Sustained)";
        case ThermalLevel::Serious:  return "Serious (Active Cooling / Thermal Pacing)";
        case ThermalLevel::Critical: return "Critical (Max Cooling / Throttled)";
    }
    return "Nominal";
}

void HardwareMonitor::set_thread_qos_interactive() {
    // Pins thread to M5 Pro Super cores (perflevel0) when active on CPU
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

void HardwareMonitor::set_thread_qos_utility() {
    // Schedules background work onto lower-power cores
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
}

uint32_t HardwareMonitor::adaptive_prefill_chunk_size(
    const M5HardwareProfile& /*profile*/,
    ThermalLevel thermal,
    bool is_moe,
    size_t model_params_millions
) {
    // On a 27B Dense model, prefilling 2048 tokens in one giant command buffer locks the GPU
    // at 100% power for >1.5s, spiking die temperature and triggering macOS fan ramp.
    // Chunking prefill into 128-512 token slices keeps GPU dispatches short and responsive.
    uint32_t base_chunk = 512;
    if (model_params_millions >= 20000) {
        base_chunk = is_moe ? 512 : 256;
    } else if (model_params_millions >= 7000) {
        base_chunk = is_moe ? 768 : 384;
    }

    switch (thermal) {
        case ThermalLevel::Nominal:
            return base_chunk;
        case ThermalLevel::Fair:
            return (base_chunk * 3) / 4;
        case ThermalLevel::Serious:
            return base_chunk / 2;
        case ThermalLevel::Critical:
            return 64;
    }
    return base_chunk;
}

} // namespace m5engine

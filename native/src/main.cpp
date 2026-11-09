#include "hardware_m5.hpp"
#include "engine.hpp"
#include "openai_server.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace m5engine;

static void print_hardware_banner(const M5HardwareProfile& hw) {
    std::cout << "================================================================================\n";
    std::cout << "  M5Engine: Native Apple Silicon LLM Runtime (Metal 3 + Zero-Copy Unified RAM)\n";
    std::cout << "================================================================================\n";
    std::cout << "  SoC / GPU          : " << hw.chip_name << " (" << hw.gpu_name << ")\n";
    std::cout << "  CPU Core Topology  : " << hw.total_cores << " Cores ("
              << hw.super_cores << " Super @ " << (hw.super_l2_cache_bytes / (1024 * 1024)) << "MB L2 + "
              << hw.performance_cores << " Performance @ " << (hw.perf_l2_cache_bytes / (1024 * 1024)) << "MB L2)\n";
    std::cout << "  Cache Line / Page  : " << hw.cache_line_bytes << " B cache line / "
              << hw.vm_page_bytes << " B Darwin VM page\n";
    std::cout << "  Unified Memory     : " << (hw.total_ram_bytes / (1024ULL * 1024 * 1024)) << " GB Total | "
              << (hw.recommended_gpu_working_set_bytes / (1024ULL * 1024)) << " MB Wired GPU Working Set | "
              << (hw.max_single_buffer_bytes / (1024ULL * 1024)) << " MB Max Single Buffer\n";
    std::cout << "  Metal Capabilities : Apple9=" << (hw.supports_apple9_gpu ? "YES" : "NO")
              << " | Metal3=" << (hw.supports_metal3 ? "YES" : "NO")
              << " | Threadgroup SRAM=" << (hw.max_threadgroup_mem_bytes / 1024) << " KB\n";
    std::cout << "================================================================================\n";
}

static void run_benchmark_suite() {
    M5InferenceEngine engine("dense");
    print_hardware_banner(engine.hardware_profile());

    // Build a realistic 256+ token system prompt + coding assistant context
    std::string long_system_prompt;
    for (int i = 0; i < 8; ++i) {
        long_system_prompt +=
            "You are an expert systems architect optimizing LLM inference on Apple M5 Pro with 64GB Unified Memory. "
            "Focus on 128-byte cache-line coalesced Q4_G64 kernels, FlashAttention-2 with Paged GQA, "
            "Radix Prefix KV-Cache reuse, Dense Multi-Token Prediction, and MoE Fused Sparse Routing.\n";
    }

    std::cout << "\n[Benchmark 1] Dense Model: Cold Prefill vs. Radix Prefix KV-Cache Reuse (TTFT)\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    engine.clear_prefix_cache();

    GenerationParams p_cold;
    p_cold.max_tokens = 32;
    p_cold.use_prefix_cache = true;
    p_cold.use_speculative_mtp = false; // baseline M=1 decode first
    auto res_cold = engine.generate_from_prompt(long_system_prompt + "Question 1: Explain unified memory.", p_cold);

    // Warm follow-up turn sharing the exact long system prompt prefix!
    GenerationParams p_warm = p_cold;
    auto res_warm = engine.generate_from_prompt(long_system_prompt + "Question 2: Explain Metal 3 kernels.", p_warm);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Cold Prompt (" << res_cold.metrics.prompt_tokens << " toks, 0 cached)   -> "
              << "TTFT: " << std::setw(7) << res_cold.metrics.ttft_ms << " ms | "
              << "Prefill: " << std::setw(8) << res_cold.metrics.prefill_tokens_per_sec << " tok/s\n";
    std::cout << "  Warm Prompt (" << res_warm.metrics.prompt_tokens << " toks, "
              << res_warm.metrics.prefix_cached_tokens << " cached) -> "
              << "TTFT: " << std::setw(7) << res_warm.metrics.ttft_ms << " ms | "
              << "Prefill: " << std::setw(8) << res_warm.metrics.prefill_tokens_per_sec << " tok/s "
              << "(" << (res_cold.metrics.ttft_ms / std::max(0.01, res_warm.metrics.ttft_ms)) << "x faster TTFT!)\n";

    std::cout << "\n[Benchmark 2] Dense Model: Standard Decode (M=1) vs. Speculative MTP Batched Verify (K=4)\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    GenerationParams p_mtp;
    p_mtp.max_tokens = 32;
    p_mtp.use_prefix_cache = true;
    p_mtp.use_speculative_mtp = true;
    auto res_mtp = engine.generate_from_prompt(long_system_prompt + "Question 3: Summarize M5 Pro decode.", p_mtp);

    std::cout << "  Standard Dense Decode (M=1/pass)    -> "
              << std::setw(8) << res_warm.metrics.decode_tokens_per_sec << " tok/s | "
              << "Tokens/Pass: " << res_warm.metrics.mean_tokens_per_pass << "x | "
              << "Forward Passes: " << res_warm.metrics.forward_passes << "\n";
    std::cout << "  Speculative MTP Decode (M=5/pass)   -> "
              << std::setw(8) << res_mtp.metrics.decode_tokens_per_sec << " tok/s | "
              << "Tokens/Pass: " << res_mtp.metrics.mean_tokens_per_pass << "x | "
              << "Forward Passes: " << res_mtp.metrics.forward_passes
              << " (" << (res_mtp.metrics.decode_tokens_per_sec / std::max(1.0, res_warm.metrics.decode_tokens_per_sec))
              << "x throughput speedup!)\n";

    std::cout << "\n[Benchmark 3] Adaptive MoE Model: Zero-Sync Fused Top-K Router + Hot-Expert Pinning\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    engine.load_model("moe");
    GenerationParams p_moe;
    p_moe.max_tokens = 32;
    p_moe.use_prefix_cache = true;
    auto res_moe_cold = engine.generate_from_prompt(long_system_prompt + "MoE Turn 1: Route across experts.", p_moe);
    auto res_moe_warm = engine.generate_from_prompt(long_system_prompt + "MoE Turn 2: Analyze sparse activation.", p_moe);

    std::cout << "  Model Architecture                  : " << res_moe_warm.metrics.model_id
              << " (" << res_moe_warm.metrics.arch_type << ")\n";
    std::cout << "  Active Experts / Total Experts      : Top-" << res_moe_warm.metrics.active_experts_per_tok
              << " of " << res_moe_warm.metrics.total_experts
              << " (" << res_moe_warm.metrics.pinned_hot_experts << " hot experts pinned via madvise MADV_WILLNEED)\n";
    std::cout << "  MoE Cold TTFT vs Warm Prefix TTFT   : " << res_moe_cold.metrics.ttft_ms
              << " ms -> " << res_moe_warm.metrics.ttft_ms << " ms\n";
    std::cout << "  MoE Fused Sparse Decode Throughput  : " << res_moe_warm.metrics.decode_tokens_per_sec << " tok/s\n";
    std::cout << "  Top Routed Expert IDs (GPU Tracked) : [";
    for (size_t i = 0; i < res_moe_warm.metrics.top_routed_experts.size(); ++i) {
        std::cout << (i ? ", " : "") << res_moe_warm.metrics.top_routed_experts[i];
    }
    std::cout << "]\n";

    std::cout << "\n[Benchmark 4] 27B Dense Layer Geometry Stress Test (Hidden=5120, Inter=13824, GQA=40/8)\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    engine.load_model("27b-heavy");
    GenerationParams p_27b;
    p_27b.max_tokens = 24;
    p_27b.use_prefix_cache = true;
    p_27b.use_speculative_mtp = true;
    auto res_27b = engine.generate_from_prompt("Benchmarking 27B Qwen3.8 layer geometry on Apple M5 Pro.", p_27b);
    std::cout << "  Allocated Zero-Copy Unified Memory  : "
              << (engine.total_unified_memory_bytes() / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  27B-Geometry TTFT                   : " << res_27b.metrics.ttft_ms << " ms\n";
    std::cout << "  27B-Geometry MTP Decode Speed       : " << res_27b.metrics.decode_tokens_per_sec
              << " tok/s (" << res_27b.metrics.mean_tokens_per_pass << " tokens/pass)\n";
    std::cout << "  Effective Memory Bandwidth          : " << res_27b.metrics.effective_bw_gbps << " GB/s\n";
    std::cout << "  Thermal State After Benchmark       : "
              << HardwareMonitor::thermal_level_name(res_27b.metrics.thermal_state) << "\n";
    std::cout << "================================================================================\n";
}

static std::string raw_http_request(OpenAIServer& server, uint16_t port, const std::string& method, const std::string& path, const std::string& body = "") {
    std::ostringstream req;
    req << method << " " << path << " HTTP/1.1\r\n"
        << "Host: 127.0.0.1:" << port << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    std::string r = req.str();

    int fd = -1;
    std::thread worker;

    if (server.is_tcp_bound()) {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd >= 0) {
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
            if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
                close(fd);
                fd = -1;
            }
        }
    }

    if (fd < 0) {
        // Fallback to POSIX AF_UNIX stream socketpair when running inside a network-isolated sandbox
        int sv[2] = {-1, -1};
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) < 0) return "";
        fd = sv[1];
        int server_side_fd = sv[0];
        worker = std::thread([&server, server_side_fd]() {
            server.handle_client(server_side_fd);
        });
    }

    send(fd, r.data(), r.size(), 0);

    std::string resp;
    char buf[4096];
    ssize_t n;
    while ((n = recv(fd, buf, sizeof(buf), 0)) > 0) {
        resp.append(buf, static_cast<size_t>(n));
    }
    close(fd);
    if (worker.joinable()) worker.join();
    return resp;
}

static int run_self_test_api() {
    M5InferenceEngine engine("dense");
    uint16_t port = 18089;
    OpenAIServer server(engine, "127.0.0.1", port);
    server.start_background();

    std::cout << "[Self-Test] OpenAI-Compatible Server HTTP/SSE Stream Verification\n";

    // 1. Test GET /v1/models
    std::string models_resp = raw_http_request(server, port, "GET", "/v1/models");
    bool ok_models = (models_resp.find("qwen3.8-27b-m5-q4") != std::string::npos) &&
                     (models_resp.find("qwen3.8-moe-16x4-m5-q4") != std::string::npos);
    std::cout << "  1. GET /v1/models                              : " << (ok_models ? "PASS" : "FAIL") << "\n";

    // 2. Test POST /v1/chat/completions (Dense model, non-streaming, prefix caching)
    std::string chat_req1 = R"({
        "model": "qwen3.8-27b-m5-q4",
        "max_tokens": 20,
        "stream": false,
        "messages": [
            {"role": "system", "content": "You are an Apple Silicon M5 Pro optimization assistant with a long shared system prompt prefix for KV caching."},
            {"role": "user", "content": "Why is M5Engine faster than MLX on 27B models?"}
        ]
    })";
    std::string chat_resp1 = raw_http_request(server, port, "POST", "/v1/chat/completions", chat_req1);
    bool ok_chat1 = (chat_resp1.find("\"object\": \"chat.completion\"") != std::string::npos);
    std::cout << "  2. POST /v1/chat/completions (Dense Cold)      : " << (ok_chat1 ? "PASS" : "FAIL") << "\n";

    // 3. Test POST /v1/chat/completions (Dense model, warm prefix cache hit)
    std::string chat_req2 = R"({
        "model": "qwen3.8-27b-m5-q4",
        "max_tokens": 20,
        "stream": false,
        "messages": [
            {"role": "system", "content": "You are an Apple Silicon M5 Pro optimization assistant with a long shared system prompt prefix for KV caching."},
            {"role": "user", "content": "Now explain speculative MTP verification."}
        ]
    })";
    std::string chat_resp2 = raw_http_request(server, port, "POST", "/v1/chat/completions", chat_req2);
    bool ok_chat2 = (chat_resp2.find("\"cached_tokens\":") != std::string::npos) &&
                    (chat_resp2.find("\"cached_tokens\": 0") == std::string::npos);
    std::cout << "  3. POST /v1/chat/completions (Prefix KV Hit)   : " << (ok_chat2 ? "PASS" : "FAIL") << "\n";

    // 4. Test POST /v1/chat/completions with SSE Streaming (stream: true)
    std::string stream_req = R"({
        "model": "qwen3.8-27b-m5-q4",
        "max_tokens": 12,
        "stream": true,
        "messages": [
            {"role": "user", "content": "Stream tokens via SSE."}
        ]
    })";
    std::string stream_resp = raw_http_request(server, port, "POST", "/v1/chat/completions", stream_req);
    bool ok_stream = (stream_resp.find("chat.completion.chunk") != std::string::npos) &&
                     (stream_resp.find("data: [DONE]") != std::string::npos);
    std::cout << "  4. POST /v1/chat/completions (SSE Streaming)   : " << (ok_stream ? "PASS" : "FAIL") << "\n";

    // 5. Test Dynamic Adaptation to Mixture-of-Experts (MoE) model via OpenAI API
    std::string moe_req = R"({
        "model": "qwen3.8-moe-16x4-m5-q4",
        "max_tokens": 16,
        "stream": false,
        "messages": [
            {"role": "user", "content": "Test dynamic adaptation to MoE sparse routing."}
        ]
    })";
    std::string moe_resp = raw_http_request(server, port, "POST", "/v1/chat/completions", moe_req);
    bool ok_moe = (moe_resp.find("Mixture-of-Experts") != std::string::npos) &&
                  (moe_resp.find("\"active_experts_per_tok\": 4") != std::string::npos);
    std::cout << "  5. POST /v1/chat/completions (Adaptive MoE)    : " << (ok_moe ? "PASS" : "FAIL") << "\n";

    // 6. Test GET /v1/engine/stats
    std::string stats_resp = raw_http_request(server, port, "GET", "/v1/engine/stats");
    bool ok_stats = (stats_resp.find("Apple M5 Pro") != std::string::npos);
    std::cout << "  6. GET /v1/engine/stats (M5 Pro Telemetry)     : " << (ok_stats ? "PASS" : "FAIL") << "\n";

    // 7. Test GET / (Built-in Apple M5 Pro Web Frontend UI)
    std::string ui_resp = raw_http_request(server, port, "GET", "/");
    bool ok_ui = (ui_resp.find("M5Engine") != std::string::npos) &&
                 (ui_resp.find("Content-Type: text/html") != std::string::npos);
    std::cout << "  7. GET / (Built-In Web Frontend Dashboard)     : " << (ok_ui ? "PASS" : "FAIL") << "\n";

    // 8. Test POST /v1/models/load (Hot-load model weights / preset from UI)
    std::string load_resp = raw_http_request(server, port, "POST", "/v1/models/load", R"({"model":"dense"})");
    bool ok_load = (load_resp.find("\"status\": \"loaded\"") != std::string::npos);
    std::cout << "  8. POST /v1/models/load (Hot-Swap Weights)     : " << (ok_load ? "PASS" : "FAIL") << "\n";

    // 9. Test GET /v1/models/recommendations (Hardware-Aware HuggingFace Filter)
    std::string rec_resp = raw_http_request(server, port, "GET", "/v1/models/recommendations");
    bool ok_rec = (rec_resp.find("Qwen3-30B-A3B-Instruct-4bit") != std::string::npos) &&
                  (rec_resp.find("Llama-3.3-70B-Instruct-4bit") == std::string::npos) && // Filtered out!
                  (rec_resp.find("\"filtered_out_count\": 2") != std::string::npos);
    std::cout << "  9. GET /v1/models/recommendations (HF Filter)  : " << (ok_rec ? "PASS" : "FAIL") << "\n";

    // 10. Test POST /v1/models/prepare (1-Click HF Model Setup & Mount)
    std::string prep_resp = raw_http_request(
        server, port, "POST", "/v1/models/prepare",
        R"({"repo_id":"mlx-community/Qwen3-30B-A3B-Instruct-4bit","local_dir":"./models/qwen3-30b-a3b-4bit","arch":"MoE"})"
    );
    bool ok_prep = (prep_resp.find("\"status\": \"mounted\"") != std::string::npos) &&
                   (prep_resp.find("Mixture-of-Experts") != std::string::npos);
    std::cout << " 10. POST /v1/models/prepare (1-Click HF Mount)  : " << (ok_prep ? "PASS" : "FAIL") << "\n";

    // 11. Test POST /v1/engine/control (Stop / Unload to 0 MB GPU -> Start / Bring Up Engine from UI)
    std::string stop_resp = raw_http_request(server, port, "POST", "/v1/engine/control", R"({"action":"stop"})");
    std::string start_resp = raw_http_request(server, port, "POST", "/v1/engine/control", R"({"action":"start","model":"dense"})");
    bool ok_ctrl = (stop_resp.find("\"engine_state\": \"standby\"") != std::string::npos) &&
                   (stop_resp.find("\"unified_memory_allocated_mb\": 0.00") != std::string::npos) &&
                   (start_resp.find("\"engine_state\": \"online\"") != std::string::npos);
    std::cout << " 11. POST /v1/engine/control (Stop/Start Engine) : " << (ok_ctrl ? "PASS" : "FAIL") << "\n";

    // 12. Test POST /v1/engine/port (Dynamic Port Rebinding from UI)
    std::string port_resp = raw_http_request(server, port, "POST", "/v1/engine/port", R"({"port":18090})");
    bool ok_port = (port_resp.find("\"status\": \"port_changed\"") != std::string::npos) &&
                   (port_resp.find("\"new_port\": 18090") != std::string::npos) &&
                   (server.port() == 18090);
    port = server.port();
    std::cout << " 12. POST /v1/engine/port (UI Port Rebinding)    : " << (ok_port ? "PASS" : "FAIL") << "\n";

    // 13. Test POST /v1/models/download & GET /v1/models/download/status (UI Model Downloader & Auto-Mount)
    std::string dl_resp = raw_http_request(
        server, port, "POST", "/v1/models/download",
        R"({"repo_id":"mlx-community/DeepSeek-V2.5-Lite-Chat-4bit","local_dir":"./models/deepseek-v2.5-lite-4bit","arch":"MoE","auto_mount":true,"sync":true})"
    );
    std::string dl_status = raw_http_request(server, port, "GET", "/v1/models/download/status");
    bool ok_dl = (dl_resp.find("\"status\": \"completed\"") != std::string::npos) &&
                 (dl_resp.find("\"auto_mounted\": true") != std::string::npos) &&
                 (dl_status.find("./models/deepseek-v2.5-lite-4bit") != std::string::npos);
    std::cout << " 13. POST /v1/models/download (UI Model Download): " << (ok_dl ? "PASS" : "FAIL") << "\n";

    server.stop();
    bool all_ok = ok_models && ok_chat1 && ok_chat2 && ok_stream && ok_moe && ok_stats && ok_ui && ok_load && ok_rec && ok_prep && ok_ctrl && ok_port && ok_dl;
    std::cout << "\n[Self-Test Result] " << (all_ok ? "ALL 13 END-TO-END CHECKS PASSED!" : "SOME CHECKS FAILED") << "\n";
    return all_ok ? 0 : 1;
}

int main(int argc, char** argv) {
    std::string cmd = (argc > 1) ? argv[1] : "bench";
    std::string model = "dense";
    std::string prompt = "Explain how M5Engine accelerates Dense and MoE LLMs on Apple M5 Pro.";
    uint16_t port = 8080;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        }
    }

    if (cmd == "bench") {
        run_benchmark_suite();
        return 0;
    } else if (cmd == "test-api") {
        return run_self_test_api();
    } else if (cmd == "run") {
        M5InferenceEngine engine(model);
        print_hardware_banner(engine.hardware_profile());
        std::cout << "\n[Prompt] " << prompt << "\n[Output] ";
        GenerationParams p;
        p.max_tokens = 48;
        auto res = engine.generate_from_prompt(
            prompt, p,
            [](const std::string& piece, int32_t, bool, const GenerationMetrics&) {
                std::cout << piece << std::flush;
                return true;
            }
        );
        std::cout << "\n\n[Telemetry] Model=" << res.metrics.model_id
                  << " | Arch=" << res.metrics.arch_type
                  << " | TTFT=" << std::fixed << std::setprecision(2) << res.metrics.ttft_ms << " ms"
                  << " | Decode=" << res.metrics.decode_tokens_per_sec << " tok/s"
                  << " | Mean Tokens/Pass=" << res.metrics.mean_tokens_per_pass << "x"
                  << " | Thermal=" << HardwareMonitor::thermal_level_name(res.metrics.thermal_state) << "\n";
        return 0;
    } else if (cmd == "serve") {
        M5InferenceEngine engine(model);
        print_hardware_banner(engine.hardware_profile());
        std::cout << "\n[Web Frontend & OpenAI API Server] Listening on http://127.0.0.1:" << port << "\n";
        std::cout << "  - GET  http://127.0.0.1:" << port << "/                           (Web Frontend Control Center & Chat UI)\n";
        std::cout << "  - POST http://127.0.0.1:" << port << "/v1/engine/control          (Start / Stop / Restart Engine from UI)\n";
        std::cout << "  - POST http://127.0.0.1:" << port << "/v1/engine/port             (Dynamically Change Port from UI)\n";
        std::cout << "  - POST http://127.0.0.1:" << port << "/v1/models/download         (Download & Auto-Mount HF Model in UI)\n";
        std::cout << "  - GET  http://127.0.0.1:" << port << "/v1/models/download/status  (Live Download Progress & Local Models)\n";
        std::cout << "  - POST http://127.0.0.1:" << port << "/v1/models/load             (Hot-Swap Local .safetensors / Model Path)\n";
        std::cout << "  - POST http://127.0.0.1:" << port << "/v1/chat/completions        (OpenAI Streaming SSE & REST)\n";
        std::cout << "  - GET  http://127.0.0.1:" << port << "/v1/engine/stats\n\n";
        OpenAIServer server(engine, "127.0.0.1", port);
        server.run_blocking();
        return 0;
    } else {
        std::cout << "Usage: ./m5engine <bench | test-api | run | serve> [--model dense|moe|27b-heavy|path] [--port 8080] [--prompt \"...\"]\n";
        return 0;
    }
}

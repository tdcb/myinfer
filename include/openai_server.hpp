#pragma once

#include "engine.hpp"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace m5engine {

struct DownloadJobStatus {
    std::string status = "idle"; // "idle", "downloading", "completed", "error"
    std::string repo_id;
    std::string local_dir;
    std::string arch = "Dense";
    std::string current_file;
    std::string message = "Ready to download models";
    double progress_pct = 0.0;
    uint64_t bytes_downloaded = 0;
    uint64_t total_bytes = 0;
    bool auto_mounted = false;
};

class OpenAIServer {
public:
    OpenAIServer(M5InferenceEngine& engine, const std::string& host = "127.0.0.1", uint16_t port = 8080);
    ~OpenAIServer();

    // Start listening and serving requests (blocking)
    void run_blocking();

    // Start in background thread (useful for embedded tests)
    void start_background();
    void stop();

    // Dynamically rebind server to a new TCP port from the Web UI without dropping engine state
    bool change_port(uint16_t new_port, std::string& err_msg);

    uint16_t port() const { return port_.load(); }
    bool is_tcp_bound() const { return listen_fd_.load() >= 0; }

    // Handles a full HTTP/1.1 or SSE streaming request over any connected POSIX stream socket fd
    void handle_client(int client_fd);

private:
    M5InferenceEngine& engine_;
    std::string host_;
    std::atomic<uint16_t> port_;
    std::atomic<int> listen_fd_{-1};
    std::atomic<bool> running_{false};
    std::thread server_thread_;
    std::mutex port_mutex_;

    // Background model downloader state
    std::mutex download_mutex_;
    DownloadJobStatus download_status_;
    std::thread download_thread_;

    void accept_loop(int sock_fd);
    int create_listen_socket(uint16_t target_port, std::string& err_msg);
    void run_model_download_job(std::string repo_id, std::string local_dir, std::string arch, bool auto_mount, bool use_live_hf);
    std::vector<std::string> scan_local_models() const;
};

} // namespace m5engine

#include "h40/service_protocol.hpp"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <netinet/in.h>

namespace {

struct Config {
    std::uint16_t port{8080};
    std::size_t max_body_bytes{65536};
    std::uint64_t rss_budget_kib{512 * 1024};
    std::string runner;
    std::string source;
    std::string catalog;
    std::string experts;
};

struct Metrics {
    std::uint64_t requests{};
    std::uint64_t inference_requests{};
    std::uint64_t completed_inference_requests{};
    std::uint64_t failures{};
    std::uint64_t last_peak_rss_kib{};
};

std::atomic<bool> running{true};
volatile std::sig_atomic_t active_child{-1};

void stop_handler(int) {
    running = false;
    if (active_child > 0) kill(active_child, SIGTERM);
}

std::uint64_t process_status_kib(std::string_view field) {
    std::ifstream status("/proc/self/status");
    std::string line;
    const std::string prefix = std::string(field) + ":";
    while (std::getline(status, line)) {
        if (line.rfind(prefix, 0) != 0) continue;
        std::istringstream value(line.substr(prefix.size()));
        std::uint64_t kib = 0;
        value >> kib;
        return kib;
    }
    return 0;
}

std::string json_field(std::string_view text, std::string_view field) {
    const std::string needle = "\"" + std::string(field) + "\":";
    const auto start = text.find(needle);
    if (start == std::string_view::npos) return {};
    auto value = start + needle.size();
    while (value < text.size() && text[value] == ' ') ++value;
    auto end = value;
    while (end < text.size() && ((text[end] >= '0' && text[end] <= '9') || text[end] == '.' || text[end] == '-')) ++end;
    return std::string(text.substr(value, end - value));
}

std::string response(int status, std::string_view reason, std::string_view body) {
    std::ostringstream out;
    out << "HTTP/1.1 " << status << ' ' << reason << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n" << body;
    return out.str();
}

void send_all(int fd, std::string_view data) {
    while (!data.empty()) {
        const auto sent = send(fd, data.data(), data.size(), 0);
        if (sent <= 0) return;
        data.remove_prefix(static_cast<std::size_t>(sent));
    }
}

std::string receive_request(int fd, std::size_t limit) {
    std::string data;
    char buffer[4096];
    while (data.size() <= limit + 16384) {
        const auto count = recv(fd, buffer, sizeof(buffer), 0);
        if (count <= 0) break;
        data.append(buffer, static_cast<std::size_t>(count));
        const auto split = data.find("\r\n\r\n");
        if (split != std::string::npos) {
            const auto marker = data.find("Content-Length:");
            std::size_t expected = 0;
            if (marker != std::string::npos && marker < split) {
                expected = std::stoull(data.substr(marker + 15));
            }
            if (data.size() >= split + 4 + expected) break;
        }
    }
    return data;
}

Config parse_config(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::invalid_argument("option requires value");
        const std::string_view option = argv[i];
        const std::string value = argv[i + 1];
        if (option == "--port") config.port = static_cast<std::uint16_t>(std::stoul(value));
        else if (option == "--max-body-bytes") config.max_body_bytes = std::stoull(value);
        else if (option == "--rss-budget-kib") config.rss_budget_kib = std::stoull(value);
        else if (option == "--runner") config.runner = value;
        else if (option == "--source") config.source = value;
        else if (option == "--catalog") config.catalog = value;
        else if (option == "--experts") config.experts = value;
        else throw std::invalid_argument("unknown option: " + std::string(option));
    }
    if (config.runner.empty() || config.source.empty() || config.catalog.empty() || config.experts.empty()) {
        throw std::invalid_argument("runner/source/catalog/experts are required");
    }
    return config;
}

std::string run_inference(const Config& config, std::string tokens, Metrics& metrics) {
    while (!tokens.empty() && (tokens.back() == '\r' || tokens.back() == '\n' || tokens.back() == ' ')) tokens.pop_back();
    if (!h40::valid_token_csv(tokens)) throw std::invalid_argument("body must be comma-separated token ids");
    const auto output = std::filesystem::path("/data/local/tmp/h40m") /
        ("service_response_" + std::to_string(getpid()) + ".json");
    const auto pid = fork();
    if (pid < 0) throw std::runtime_error("fork failed");
    if (pid == 0) {
        execl(config.runner.c_str(), config.runner.c_str(), config.source.c_str(), config.catalog.c_str(),
              config.experts.c_str(), tokens.c_str(), output.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    active_child = pid;
    int status = 0;
    pid_t waited = -1;
    do {
        waited = waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);
    active_child = -1;
    if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("decoder runner failed");
    }
    std::ifstream stream(output);
    if (!stream) throw std::runtime_error("decoder runner did not produce output");
    std::ostringstream json;
    json << stream.rdbuf();
    std::filesystem::remove(output);
    const auto payload = json.str();
    const auto rss = json_field(payload, "peak_rss_kib");
    metrics.last_peak_rss_kib = rss.empty() ? 0 : std::stoull(rss);
    if (metrics.last_peak_rss_kib > config.rss_budget_kib) throw std::runtime_error("inference exceeded RSS budget");
    return payload;
}

std::string metrics_json(const Config& config, const Metrics& metrics) {
    std::ostringstream out;
    out << "{\"requests\":" << metrics.requests
        << ",\"inference_requests\":" << metrics.inference_requests
        << ",\"completed_inference_requests\":" << metrics.completed_inference_requests
        << ",\"failures\":" << metrics.failures
        << ",\"last_peak_rss_kib\":" << metrics.last_peak_rss_kib
        << ",\"service_rss_kib\":" << process_status_kib("VmRSS")
        << ",\"service_peak_rss_kib\":" << process_status_kib("VmHWM")
        << ",\"rss_budget_kib\":" << config.rss_budget_kib << "}";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto config = parse_config(argc, argv);
        if (access(config.runner.c_str(), X_OK) != 0 || !std::filesystem::exists(config.source) ||
            !std::filesystem::is_regular_file(config.catalog) ||
            !std::filesystem::is_regular_file(config.experts)) {
            throw std::invalid_argument("runner or model artifact is unavailable");
        }
        struct sigaction action {};
        action.sa_handler = stop_handler;
        sigemptyset(&action.sa_mask);
        sigaction(SIGINT, &action, nullptr);
        sigaction(SIGTERM, &action, nullptr);
        std::signal(SIGPIPE, SIG_IGN);
        const int server = socket(AF_INET, SOCK_STREAM, 0);
        if (server < 0) throw std::runtime_error("socket failed");
        int reuse = 1;
        setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(config.port);
        if (bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 || listen(server, 8) != 0) {
            throw std::runtime_error("bind/listen failed");
        }
        Metrics metrics;
        while (running) {
            const int client = accept(server, nullptr, nullptr);
            if (client < 0) {
                if (errno == EINTR) continue;
                break;
            }
            ++metrics.requests;
            try {
                const auto request = h40::parse_http_request(receive_request(client, config.max_body_bytes), config.max_body_bytes);
                if (request.method == "GET" && request.target == "/health") {
                    send_all(client, response(200, "OK", "{\"status\":\"ok\"}"));
                } else if (request.method == "GET" && request.target == "/metrics") {
                    send_all(client, response(200, "OK", metrics_json(config, metrics)));
                } else if (request.method == "POST" && request.target == "/infer") {
                    ++metrics.inference_requests;
                    const auto payload = run_inference(config, request.body, metrics);
                    ++metrics.completed_inference_requests;
                    send_all(client, response(200, "OK", payload));
                } else {
                    send_all(client, response(404, "Not Found", "{\"error\":\"not_found\"}"));
                }
            } catch (const std::exception& error) {
                ++metrics.failures;
                send_all(client, response(400, "Bad Request", "{\"error\":\"request_failed\"}"));
                std::cerr << error.what() << '\n';
            }
            close(client);
        }
        close(server);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}

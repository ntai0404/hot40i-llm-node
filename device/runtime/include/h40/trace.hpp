#pragma once

#include <chrono>
#include <cstdint>
#include <ostream>
#include <string_view>

namespace h40 {

struct TraceEvent {
    std::string_view event;
    std::uint64_t token{};
    std::uint64_t layer{};
    std::uint64_t expert{};
    std::uint64_t bytes{};
    std::uint64_t duration_ns{};
    std::string_view request_id;
    bool has_token{};
    bool has_layer{};
    bool has_expert{};
    bool has_bytes{};
    bool has_duration{};
    bool cache_hit{};
    bool has_cache_hit{};
};

class JsonlTraceWriter {
public:
    explicit JsonlTraceWriter(std::ostream& out) : out_(out), start_(std::chrono::steady_clock::now()) {}

    void emit(const TraceEvent& event) {
        out_ << "{\"schema_version\":1,\"ts_ns\":" << now_ns() << ",\"event\":\"" << event.event << "\"";
        if (event.has_token) out_ << ",\"token\":" << event.token;
        if (event.has_layer) out_ << ",\"layer\":" << event.layer;
        if (event.has_expert) out_ << ",\"expert\":" << event.expert;
        if (event.has_bytes) out_ << ",\"bytes\":" << event.bytes;
        if (event.has_duration) out_ << ",\"duration_ns\":" << event.duration_ns;
        if (!event.request_id.empty()) out_ << ",\"request_id\":\"" << event.request_id << "\"";
        if (event.has_cache_hit) out_ << ",\"cache_hit\":" << (event.cache_hit ? "true" : "false");
        out_ << "}\n";
    }

private:
    std::uint64_t now_ns() const {
        const auto elapsed = std::chrono::steady_clock::now() - start_;
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    }

    std::ostream& out_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace h40

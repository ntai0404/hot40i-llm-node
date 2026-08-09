#include "h40/trace.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

h40::TraceEvent base(std::string_view event, std::uint64_t token) {
    h40::TraceEvent row;
    row.event = event;
    row.token = token;
    row.has_token = true;
    return row;
}

h40::TraceEvent layer_event(std::string_view event, std::uint64_t token, std::uint64_t layer) {
    auto row = base(event, token);
    row.layer = layer;
    row.has_layer = true;
    return row;
}

h40::TraceEvent expert_event(std::string_view event, std::uint64_t token, std::uint64_t layer, std::uint64_t expert) {
    auto row = layer_event(event, token, layer);
    row.expert = expert;
    row.has_expert = true;
    return row;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("usage: h40_trace_writer_probe OUT.jsonl");
        }
        std::ofstream out(argv[1], std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to open trace output");
        h40::JsonlTraceWriter trace(out);
        auto token_begin = base("token_begin", 0);
        token_begin.request_id = "tok0";
        trace.emit(token_begin);
        trace.emit(layer_event("layer_begin", 0, 0));
        trace.emit(expert_event("route", 0, 0, 2));
        auto cache_miss = expert_event("cache_miss", 0, 0, 2);
        cache_miss.cache_hit = false;
        cache_miss.has_cache_hit = true;
        trace.emit(cache_miss);
        auto read_begin = expert_event("read_begin", 0, 0, 2);
        read_begin.bytes = 13236480;
        read_begin.has_bytes = true;
        trace.emit(read_begin);
        auto read_end = read_begin;
        read_end.event = "read_end";
        read_end.duration_ns = 5000;
        read_end.has_duration = true;
        trace.emit(read_end);
        trace.emit(expert_event("compute_begin", 0, 0, 2));
        auto compute_end = expert_event("compute_end", 0, 0, 2);
        compute_end.duration_ns = 9000;
        compute_end.has_duration = true;
        trace.emit(compute_end);
        trace.emit(expert_event("prefetch_begin", 0, 1, 4));
        auto prefetch_end = expert_event("prefetch_end", 0, 1, 4);
        prefetch_end.duration_ns = 1000;
        prefetch_end.has_duration = true;
        trace.emit(prefetch_end);
        auto evict = expert_event("evict", 0, 0, 1);
        evict.bytes = 13236480;
        evict.has_bytes = true;
        trace.emit(evict);
        auto cache_hit = expert_event("cache_hit", 0, 0, 2);
        cache_hit.cache_hit = true;
        cache_hit.has_cache_hit = true;
        trace.emit(cache_hit);
        trace.emit(layer_event("layer_end", 0, 0));
        auto token_end = base("token_end", 0);
        token_end.request_id = "tok0";
        trace.emit(token_end);
        std::cout << argv[1] << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "h40_trace_writer_probe: " << exc.what() << "\n";
        return 2;
    }
}

#include "h40/kv_cache.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif
#include <vector>

namespace {

struct Result {
    std::size_t context{};
    std::size_t storage_bytes{};
    std::uint64_t append_ns{};
    std::uint64_t attention_ns{};
    long peak_rss_kib{};
    double checksum{};
};

std::uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

long peak_rss_kib() {
#if defined(_WIN32)
    return 0;
#else
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
#endif
}

Result run(std::size_t context) {
    constexpr h40::GptOssKvCacheConfig base{24, 8, 64, 1, 1, true};
    const h40::GptOssKvCacheConfig config{
        base.layers, base.key_value_heads, base.head_dim, context, h40::kGptOssSlidingWindowTokens, true};
    const auto bytes = h40::GptOssKvCache::required_bytes(config);
    std::vector<std::byte> storage(bytes);
    h40::GptOssKvCache cache(config, storage);
    std::vector<float> key(cache.vector_values());
    std::vector<float> value(cache.vector_values());
    float max_roundtrip_error = 0.0F;

    const auto append_start = std::chrono::steady_clock::now();
    for (std::size_t layer = 0; layer < config.layers; ++layer) {
        for (std::size_t position = 0; position < context; ++position) {
            for (std::size_t index = 0; index < key.size(); ++index) {
                const auto raw = static_cast<int>((position * 17 + layer * 13 + index * 3) % 257) - 128;
                key[index] = static_cast<float>(raw) / 257.0F;
                value[index] = static_cast<float>(raw + static_cast<int>(index % 7) - 3) / 263.0F;
            }
            cache.append(layer, position, key, value);
        }
    }
    const auto append_ns = elapsed_ns(append_start);

    std::vector<float> query(64 * 64);
    std::vector<float> sinks(64);
    std::vector<float> output(query.size());
    std::vector<float> scores(context);
    for (std::size_t index = 0; index < query.size(); ++index) {
        query[index] = static_cast<float>(static_cast<int>(index % 31) - 15) / 64.0F;
    }
    for (std::size_t index = 0; index < sinks.size(); ++index) {
        sinks[index] = static_cast<float>(static_cast<int>(index % 9) - 4) / 16.0F;
    }

    const auto attention_start = std::chrono::steady_clock::now();
    double checksum = 0.0;
    for (std::size_t layer = 0; layer < config.layers; ++layer) {
        h40::gptoss_cached_attention(cache, layer, context - 1, query, sinks, output, scores);
        for (const auto item : output) checksum += item;
    }
    const auto attention_ns = elapsed_ns(attention_start);

    const auto retained = cache.key_bf16(0, context - 1);
    for (std::size_t index = 0; index < std::min<std::size_t>(retained.size(), 64); ++index) {
        const auto bits = static_cast<std::uint32_t>(retained[index]) << 16U;
        const auto roundtrip = std::bit_cast<float>(bits);
        const auto raw = static_cast<int>(((context - 1) * 17 + index * 3) % 257) - 128;
        const auto expected = static_cast<float>(raw) / 257.0F;
        max_roundtrip_error = std::max(max_roundtrip_error, std::fabs(roundtrip - expected));
    }
    if (max_roundtrip_error > 0.01F) throw std::runtime_error("BF16 KV roundtrip error exceeded bound");

    return {context, bytes, append_ns, attention_ns, peak_rss_kib(), checksum};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: kv_cache_probe <output.json>\n";
        return 2;
    }
    const std::vector<std::size_t> contexts{512, 1024, 2048, 4096};
    std::vector<Result> results;
    for (const auto context : contexts) results.push_back(run(context));
    std::ofstream out(argv[1]);
    if (!out) throw std::runtime_error("failed to open probe output");
    out << "{\n  \"schema_version\": 1,\n  \"status\": \"pass\",\n  \"device_config\": {\n";
    out << "    \"layers\": 24, \"key_value_heads\": 8, \"head_dim\": 64, \"sliding_window\": 128,\n";
    out << "    \"first_layer_sliding\": true, \"storage_dtype\": \"BF16\"\n  },\n  \"contexts\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const auto& result = results[index];
        out << "    {\"tokens\": " << result.context
            << ", \"storage_bytes\": " << result.storage_bytes
            << ", \"append_ns\": " << result.append_ns
            << ", \"last_token_24_layer_attention_ns\": " << result.attention_ns
            << ", \"peak_rss_kib\": " << result.peak_rss_kib
            << ", \"checksum\": " << result.checksum << "}";
        out << (index + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    std::cout << "kv_context_sweep=512,1024,2048,4096 status=pass\n";
    return 0;
}

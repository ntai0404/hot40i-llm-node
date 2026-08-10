#include "h40/expert_cache.hpp"
#include "h40/expert_loader.hpp"
#include "h40/expert_read_pipeline.hpp"
#include "h40/moe_scheduler.hpp"

#include <array>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <span>
#include <string>
#include <thread>
#include <memory>

namespace {

constexpr std::size_t kExperts = 4;
constexpr std::size_t kExpertBytes = 16;

class SlowProvider final : public h40::TensorProvider {
public:
    SlowProvider() {
        for (std::size_t expert = 0; expert < kExperts; ++expert) {
            for (std::size_t i = 0; i < kExpertBytes; ++i) {
                data_[expert * kExpertBytes + i] = std::byte(expert + 1);
            }
        }
    }

    void read(const h40::TensorSlice& slice, std::span<std::byte> out) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        assert(slice.length == out.size());
        std::copy_n(data_.begin() + static_cast<std::ptrdiff_t>(slice.offset), out.size(), out.begin());
        ++stats_.operations;
        stats_.bytes += out.size();
    }

    h40::ReadStats stats() const noexcept override { return stats_; }
    std::string name() const override { return "slow-fixture"; }

private:
    std::array<std::byte, kExperts * kExpertBytes> data_{};
    h40::ReadStats stats_{};
};

struct RunResult {
    float output{};
    std::uint64_t elapsed_ms{};
    h40::CacheStats cache{};
    h40::ExpertReadPipelineStats pipeline{};
    std::string trace;
};

RunResult run(bool overlap) {
    h40::ModelIndex index;
    for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
        index.put({0, expert}, {expert * kExpertBytes, kExpertBytes});
    }
    SlowProvider provider;
    h40::ExpertLoader loader(index, provider);
    h40::ExpertCache cache(kExpertBytes * 2, kExpertBytes);
    std::array<std::byte, kExpertBytes> prefetch_buffer{};
    std::unique_ptr<h40::ExpertReadPipeline> pipeline;
    if (overlap) pipeline = std::make_unique<h40::ExpertReadPipeline>(loader, prefetch_buffer);

    const std::array<float, kExperts> router_logits{4.0F, 3.0F, 2.0F, 1.0F};
    std::array<float, 1> output{};
    std::array<std::uint32_t, kExperts> ids{};
    std::array<float, kExperts> weights{};
    std::array<float, 1> expert_output{};
    std::ostringstream trace_out;
    h40::JsonlTraceWriter trace(trace_out);
    const auto start = std::chrono::steady_clock::now();
    h40::run_moe_layer_streaming(
        {0, 1, kExperts, kExperts, 1},
        router_logits,
        cache,
        loader,
        output,
        {ids, weights, expert_output},
        [](std::size_t, std::uint32_t, std::span<const std::byte> bytes, std::span<float> out) {
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
            out[0] = static_cast<float>(std::to_integer<unsigned>(bytes[0]));
        },
        &trace,
        false,
        pipeline.get());
    const auto elapsed_ms = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    return {
        output[0],
        elapsed_ms,
        cache.stats(),
        pipeline ? pipeline->stats() : h40::ExpertReadPipelineStats{},
        trace_out.str(),
    };
}

}  // namespace

int main() {
    {
        h40::ModelIndex index;
        SlowProvider provider;
        h40::ExpertLoader loader(index, provider);
        bool rejected_empty_buffer = false;
        try {
            h40::ExpertReadPipeline invalid(loader, {});
        } catch (const std::invalid_argument&) {
            rejected_empty_buffer = true;
        }
        assert(rejected_empty_buffer);
    }
    const auto serial = run(false);
    const auto overlapped = run(true);
    assert(serial.output == overlapped.output);
    assert(serial.cache.misses == kExperts);
    assert(overlapped.cache.misses == kExperts);
    assert(overlapped.pipeline.submitted == kExperts - 1);
    assert(overlapped.pipeline.completed == kExperts - 1);
    assert(overlapped.pipeline.bytes == (kExperts - 1) * kExpertBytes);
    assert(overlapped.elapsed_ms + 60 < serial.elapsed_ms);
    assert(overlapped.trace.find("\"event\":\"prefetch_begin\"") != std::string::npos);
    assert(overlapped.trace.find("\"event\":\"prefetch_wait_end\"") != std::string::npos);
    return 0;
}

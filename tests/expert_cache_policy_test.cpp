#include "h40/expert_cache.hpp"
#include "h40/ram_arena.hpp"
#include "h40/tensor_provider.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace {

class PatternProvider final : public h40::TensorProvider {
public:
    void read(const h40::TensorSlice& slice, std::span<std::byte> out) override {
        assert(slice.length == out.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = std::byte((slice.offset + i) & 0xffU);
        }
        ++stats_.operations;
        stats_.bytes += out.size();
    }

    h40::ReadStats stats() const noexcept override { return stats_; }
    std::string name() const override { return "pattern"; }

private:
    h40::ReadStats stats_{};
};

struct ExpectedStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::uint64_t bytes_loaded{};
};

ExpectedStats simulate_lru(const std::vector<std::uint32_t>& requests, std::size_t slots, std::size_t bytes) {
    ExpectedStats stats;
    std::vector<std::uint32_t> mru;
    for (const auto key : requests) {
        const auto it = std::find(mru.begin(), mru.end(), key);
        if (it != mru.end()) {
            ++stats.hits;
            mru.erase(it);
            mru.insert(mru.begin(), key);
            continue;
        }
        ++stats.misses;
        stats.bytes_loaded += bytes;
        if (mru.size() == slots) {
            ++stats.evictions;
            mru.pop_back();
        }
        mru.insert(mru.begin(), key);
    }
    return stats;
}

bool is_aligned(std::span<const std::byte> bytes, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(bytes.data()) % alignment == 0;
}

}  // namespace

int main(int argc, char** argv) {
    PatternProvider provider;
    h40::ExpertCache cache(48, 16);
    assert(cache.policy() == h40::CachePolicy::lru);
    assert(cache.slot_count() == 3);
    assert(cache.slot_stride_bytes() == 16);

    auto a = cache.get_or_load({0, 0}, {0, 16}, provider);
    auto b = cache.get_or_load({0, 1}, {16, 16}, provider);
    auto c = cache.get_or_load({0, 2}, {32, 16}, provider);
    assert(a.size() == 16 && b.size() == 16 && c.size() == 16);
    assert(cache.used_bytes() == 48);
    assert(cache.stats().peak_used_bytes == 48);
    assert(cache.get_or_load({0, 0}, {0, 16}, provider).data() == a.data());
    (void)cache.get_or_load({0, 3}, {48, 16}, provider);
    const auto before_reload = cache.stats().misses;
    (void)cache.get_or_load({0, 1}, {16, 16}, provider);
    assert(cache.stats().misses == before_reload + 1);
    assert(cache.stats().hits == 1);
    assert(cache.stats().evictions == 2);
    assert(cache.used_bytes() == 48);

    constexpr std::size_t kChurnSlots = 4;
    constexpr std::size_t kChurnSlotBytes = 1024;
    h40::ExpertCache churn(kChurnSlots * kChurnSlotBytes, kChurnSlotBytes);
    std::vector<std::uint32_t> requests;
    requests.reserve(1000);
    std::uint32_t state = 0x12345678U;
    for (std::size_t i = 0; i < 1000; ++i) {
        state = state * 1664525U + 1013904223U;
        requests.push_back((state >> 24U) % 17U);
    }
    const auto expected = simulate_lru(requests, kChurnSlots, kChurnSlotBytes);
    for (std::size_t i = 0; i < requests.size(); ++i) {
        (void)churn.get_or_load({3, requests[i]}, {requests[i] * kChurnSlotBytes, kChurnSlotBytes}, provider);
        assert(churn.used_bytes() <= churn.budget_bytes());
        assert(churn.stats().peak_used_bytes <= churn.budget_bytes());
        assert(churn.stats().hits + churn.stats().misses == i + 1);
    }
    assert(churn.stats().hits == expected.hits);
    assert(churn.stats().misses == expected.misses);
    assert(churn.stats().evictions == expected.evictions);
    assert(churn.stats().bytes_loaded == expected.bytes_loaded);

    constexpr std::size_t kH40MExpertBytes = 13236480;
    constexpr std::size_t kH40MAlignment = 1048576;
    constexpr std::size_t kH40MStride = 13631488;
    h40::FixedMemoryArenas arenas({
        kH40MStride * 3 + kH40MAlignment,
        {{h40::ArenaRegionKind::expert_cache, "expert_cache", kH40MStride * 3, kH40MAlignment}},
    });
    h40::ExpertCache h40m_cache(arenas.region(h40::ArenaRegionKind::expert_cache), kH40MExpertBytes, kH40MAlignment);
    assert(h40m_cache.slot_bytes() == kH40MExpertBytes);
    assert(h40m_cache.slot_stride_bytes() == kH40MStride);
    assert(h40m_cache.slot_count() == 3);
    auto h0 = h40m_cache.get_or_load({0, 0}, {0, kH40MExpertBytes}, provider);
    auto h1 = h40m_cache.get_or_load({0, 1}, {kH40MExpertBytes, kH40MExpertBytes}, provider);
    auto h2 = h40m_cache.get_or_load({0, 2}, {2 * kH40MExpertBytes, kH40MExpertBytes}, provider);
    assert(is_aligned(h0, kH40MAlignment));
    assert(is_aligned(h1, kH40MAlignment));
    assert(is_aligned(h2, kH40MAlignment));
    assert(h40m_cache.used_bytes() == kH40MExpertBytes * 3);
    assert(h40m_cache.used_bytes() <= h40m_cache.budget_bytes());

    const std::vector<std::uint32_t> hot_requests{0, 1, 0, 2, 3, 0};
    h40::ExpertCache hot_lru(32, 16, 1, h40::CachePolicy::lru);
    h40::ExpertCache hot_lfu(32, 16, 1, h40::CachePolicy::lfu_decay);
    h40::ExpertCache hot_layer(32, 16, 1, h40::CachePolicy::per_layer_hotset);
    for (const auto expert : hot_requests) {
        const h40::ExpertKey key{0, expert};
        const h40::TensorSlice slice{expert * 16, 16};
        (void)hot_lru.get_or_load(key, slice, provider);
        (void)hot_lfu.get_or_load(key, slice, provider);
        (void)hot_layer.get_or_load(key, slice, provider);
    }
    assert(hot_lru.stats().hits == 1);
    assert(hot_lfu.stats().hits == 2);
    assert(hot_layer.stats().hits == 2);
    assert(hot_lru.used_bytes() == hot_lfu.used_bytes());
    assert(hot_lru.used_bytes() == hot_layer.used_bytes());

    if (argc > 1) {
        std::filesystem::create_directories(std::filesystem::path(argv[1]).parent_path());
        std::ofstream out(argv[1]);
        out << "{\n";
        out << "  \"schema_version\": 1,\n";
        out << "  \"status\": \"pass\",\n";
        out << "  \"policy\": \"lru\",\n";
        out << "  \"deterministic_hits\": " << cache.stats().hits << ",\n";
        out << "  \"deterministic_misses\": " << cache.stats().misses << ",\n";
        out << "  \"deterministic_evictions\": " << cache.stats().evictions << ",\n";
        out << "  \"churn_requests\": " << requests.size() << ",\n";
        out << "  \"churn_hits\": " << churn.stats().hits << ",\n";
        out << "  \"churn_misses\": " << churn.stats().misses << ",\n";
        out << "  \"churn_evictions\": " << churn.stats().evictions << ",\n";
        out << "  \"churn_bytes_loaded\": " << churn.stats().bytes_loaded << ",\n";
        out << "  \"churn_budget_bytes\": " << churn.budget_bytes() << ",\n";
        out << "  \"churn_peak_used_bytes\": " << churn.stats().peak_used_bytes << ",\n";
        out << "  \"h40m_expert_bytes\": " << kH40MExpertBytes << ",\n";
        out << "  \"h40m_slot_stride_bytes\": " << h40m_cache.slot_stride_bytes() << ",\n";
        out << "  \"h40m_slot_alignment\": " << h40m_cache.slot_alignment() << ",\n";
        out << "  \"h40m_slot_count\": " << h40m_cache.slot_count() << ",\n";
        out << "  \"h40m_peak_used_bytes\": " << h40m_cache.stats().peak_used_bytes << "\n";
        out << "}\n";
    }

    return 0;
}

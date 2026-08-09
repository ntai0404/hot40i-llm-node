#include "h40/expert_cache.hpp"
#include "h40/ram_arena.hpp"
#include "h40/tensor_provider.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <utility>
#include <vector>

class MemoryProvider final : public h40::TensorProvider {
public:
    explicit MemoryProvider(std::size_t bytes) : data_(bytes) {
        for (std::size_t i = 0; i < data_.size(); ++i) data_[i] = std::byte(i & 0xff);
    }

    void read(const h40::TensorSlice& slice, std::span<std::byte> out) override {
        assert(slice.offset + slice.length <= data_.size());
        assert(slice.length == out.size());
        std::copy_n(data_.begin() + slice.offset, slice.length, out.begin());
        ++stats_.operations;
        stats_.bytes += slice.length;
    }

    h40::ReadStats stats() const noexcept override { return stats_; }
    std::string name() const override { return "memory"; }

private:
    std::vector<std::byte> data_;
    h40::ReadStats stats_{};
};

static bool is_aligned(std::span<std::byte> bytes, std::size_t alignment) {
    const auto address = reinterpret_cast<std::uintptr_t>(bytes.data());
    return address % alignment == 0;
}

int main(int argc, char** argv) {
    h40::RamArena arena(256);
    auto first = arena.allocate(13, 64);
    auto second = arena.allocate(7, 128);
    assert(first.size() == 13);
    assert(second.size() == 7);
    assert(is_aligned(first, 64));
    assert(is_aligned(second, 128));

    bool rejected = false;
    try {
        (void)arena.allocate(4096, 64);
    } catch (const std::bad_alloc&) {
        rejected = true;
    }
    assert(rejected);

    h40::FixedMemoryPlan plan;
    plan.budget_bytes = 4096;
    plan.regions = {
        {h40::ArenaRegionKind::resident, "resident", 512, 64},
        {h40::ArenaRegionKind::kv_state, "kv_state", 1024, 64},
        {h40::ArenaRegionKind::expert_cache, "expert_cache", 1024, 64},
        {h40::ArenaRegionKind::scratch, "scratch", 768, 64},
    };
    h40::FixedMemoryArenas arenas(std::move(plan));
    assert(arenas.committed_bytes() <= arenas.budget_bytes());
    assert(arenas.headroom_bytes() == arenas.budget_bytes() - arenas.committed_bytes());
    assert(arenas.region("resident").size() == 512);
    assert(arenas.region(h40::ArenaRegionKind::expert_cache).size() == 1024);

    bool over_budget_rejected = false;
    try {
        h40::FixedMemoryArenas too_small({128, {{h40::ArenaRegionKind::scratch, "too_big", 256, 64}}});
    } catch (const std::bad_alloc&) {
        over_budget_rejected = true;
    }
    assert(over_budget_rejected);

    MemoryProvider provider(4096);
    h40::ExpertCache cache(arenas.region(h40::ArenaRegionKind::expert_cache), 256);
    assert(cache.slot_count() == 4);
    assert(cache.slot_bytes() == 256);

    auto a = cache.get_or_load({0, 0}, {0, 128}, provider);
    auto b = cache.get_or_load({0, 1}, {256, 128}, provider);
    auto c = cache.get_or_load({0, 2}, {512, 128}, provider);
    auto d = cache.get_or_load({0, 3}, {768, 128}, provider);
    assert(a.size() == 128);
    assert(b.size() == 128);
    assert(c.size() == 128);
    assert(d.size() == 128);
    assert(cache.used_bytes() == 512);

    auto hit = cache.get_or_load({0, 0}, {0, 128}, provider);
    assert(hit.data() == a.data());

    auto e = cache.get_or_load({0, 4}, {1024, 128}, provider);
    assert(e.size() == 128);
    assert(cache.stats().evictions == 1);
    assert(cache.used_bytes() == 512);

    bool oversize_rejected = false;
    try {
        (void)cache.get_or_load({0, 5}, {1536, 257}, provider);
    } catch (const std::bad_alloc&) {
        oversize_rejected = true;
    }
    assert(oversize_rejected);

    if (argc > 1) {
        std::filesystem::create_directories(std::filesystem::path(argv[1]).parent_path());
        std::ofstream out(argv[1]);
        out << "{\n";
        out << "  \"schema_version\": 1,\n";
        out << "  \"status\": \"pass\",\n";
        out << "  \"safe_rss_budget_bytes\": 646080512,\n";
        out << "  \"configured_budget_bytes\": " << arenas.budget_bytes() << ",\n";
        out << "  \"committed_bytes\": " << arenas.committed_bytes() << ",\n";
        out << "  \"headroom_bytes\": " << arenas.headroom_bytes() << ",\n";
        out << "  \"expert_cache_slot_bytes\": " << cache.slot_bytes() << ",\n";
        out << "  \"expert_cache_slot_count\": " << cache.slot_count() << ",\n";
        out << "  \"over_budget_rejected\": true,\n";
        out << "  \"oversize_expert_rejected\": true,\n";
        out << "  \"cache_evictions\": " << cache.stats().evictions << "\n";
        out << "}\n";
    }

    return 0;
}

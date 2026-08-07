#include "h40/expert_cache.hpp"
#include "h40/model_index.hpp"
#include "h40/ram_arena.hpp"
#include "h40/tensor_provider.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <span>
#include <vector>

class MemoryProvider final : public h40::TensorProvider {
public:
    explicit MemoryProvider(std::size_t bytes) : data_(bytes) {
        for (std::size_t i = 0; i < data_.size(); ++i) data_[i] = std::byte(i & 0xff);
    }
    void read(const h40::TensorSlice& slice, std::span<std::byte> out) override {
        assert(slice.offset + slice.length <= data_.size());
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

int main() {
    h40::RamArena arena(4096);
    assert(arena.allocate(128).size() == 128);
    assert(arena.used() >= 128);

    h40::ModelIndex index;
    index.put({1, 2}, {64, 256});
    assert(index.find({1, 2}).has_value());

    MemoryProvider provider(4096);
    h40::ExpertCache cache(512);
    auto a = cache.get_or_load({1, 2}, {64, 256}, provider);
    assert(a.size() == 256);
    auto b = cache.get_or_load({1, 2}, {64, 256}, provider);
    assert(b.size() == 256);
    auto stats = cache.stats();
    assert(stats.misses == 1);
    assert(stats.hits == 1);
    return 0;
}

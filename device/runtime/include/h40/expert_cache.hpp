#pragma once

#include "h40/model_index.hpp"
#include "h40/tensor_provider.hpp"

#include <cstddef>
#include <memory>
#include <list>
#include <span>
#include <unordered_map>
#include <vector>

namespace h40 {

struct CacheStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::uint64_t bytes_loaded{};
};

class ExpertCache {
public:
    explicit ExpertCache(std::size_t budget_bytes);
    ExpertCache(std::size_t budget_bytes, std::size_t slot_bytes);
    ExpertCache(std::span<std::byte> storage, std::size_t slot_bytes);

    [[nodiscard]] std::span<const std::byte> get_or_load(
        ExpertKey key,
        const TensorSlice& slice,
        TensorProvider& provider);

    [[nodiscard]] CacheStats stats() const noexcept { return stats_; }
    [[nodiscard]] std::size_t used_bytes() const noexcept { return used_bytes_; }
    [[nodiscard]] std::size_t budget_bytes() const noexcept { return budget_bytes_; }
    [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }
    [[nodiscard]] std::size_t slot_count() const noexcept { return slot_count_; }

private:
    struct Entry {
        std::size_t slot{};
        std::size_t bytes{};
        std::list<ExpertKey>::iterator lru_it;
    };

    void touch(ExpertKey key, Entry& entry);
    void evict_one();
    [[nodiscard]] std::span<std::byte> slot_span(std::size_t slot, std::size_t bytes);
    [[nodiscard]] std::span<const std::byte> slot_span(std::size_t slot, std::size_t bytes) const;

    std::unique_ptr<std::byte[]> owned_storage_;
    std::span<std::byte> storage_;
    std::size_t budget_bytes_{};
    std::size_t slot_bytes_{};
    std::size_t slot_count_{};
    std::size_t used_bytes_{};
    std::vector<std::size_t> free_slots_;
    std::list<ExpertKey> lru_;
    std::unordered_map<ExpertKey, Entry, ExpertKeyHash> entries_;
    CacheStats stats_{};
};

} // namespace h40

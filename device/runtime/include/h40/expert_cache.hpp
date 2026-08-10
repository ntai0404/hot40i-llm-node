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

class ExpertLoader;

struct CacheStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::uint64_t bytes_loaded{};
    std::uint64_t peak_used_bytes{};
};

struct CacheLoadResult {
    std::span<const std::byte> bytes;
    bool hit{};
};

enum class CachePolicy {
    lru,
    lfu_decay,
    per_layer_hotset,
};

[[nodiscard]] const char* cache_policy_name(CachePolicy policy) noexcept;

class ExpertCache {
public:
    explicit ExpertCache(std::size_t budget_bytes);
    ExpertCache(std::size_t budget_bytes, std::size_t slot_bytes);
    ExpertCache(std::size_t budget_bytes, std::size_t slot_bytes, std::size_t slot_alignment);
    ExpertCache(
        std::size_t budget_bytes,
        std::size_t slot_bytes,
        std::size_t slot_alignment,
        CachePolicy policy);
    ExpertCache(std::span<std::byte> storage, std::size_t slot_bytes);
    ExpertCache(std::span<std::byte> storage, std::size_t slot_bytes, std::size_t slot_alignment);
    ExpertCache(
        std::span<std::byte> storage,
        std::size_t slot_bytes,
        std::size_t slot_alignment,
        CachePolicy policy);

    [[nodiscard]] std::span<const std::byte> get_or_load(
        ExpertKey key,
        const TensorSlice& slice,
        TensorProvider& provider);
    [[nodiscard]] CacheLoadResult get_or_load(
        ExpertKey key,
        const ExpertLoader& loader,
        bool verify_checksum);
    [[nodiscard]] CacheLoadResult insert_loaded(
        ExpertKey key,
        std::span<const std::byte> bytes);

    [[nodiscard]] CacheStats stats() const noexcept { return stats_; }
    [[nodiscard]] bool contains(ExpertKey key) const noexcept;
    [[nodiscard]] std::size_t used_bytes() const noexcept { return used_bytes_; }
    [[nodiscard]] std::size_t budget_bytes() const noexcept { return budget_bytes_; }
    [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }
    [[nodiscard]] std::size_t slot_stride_bytes() const noexcept { return slot_stride_bytes_; }
    [[nodiscard]] std::size_t slot_alignment() const noexcept { return slot_alignment_; }
    [[nodiscard]] std::size_t slot_count() const noexcept { return slot_count_; }
    [[nodiscard]] CachePolicy policy() const noexcept { return policy_; }

private:
    static constexpr std::size_t kMaxPolicyHistoryEntries = 4096;
    static constexpr std::size_t kMaxPolicyLayers = 256;

    struct Entry {
        std::size_t slot{};
        std::size_t bytes{};
        std::list<ExpertKey>::iterator lru_it;
    };

    void initialize_metadata();
    void record_access(ExpertKey key);
    void touch(ExpertKey key, Entry& entry);
    void evict_one();
    [[nodiscard]] std::span<std::byte> slot_span(std::size_t slot, std::size_t bytes);
    [[nodiscard]] std::span<const std::byte> slot_span(std::size_t slot, std::size_t bytes) const;

    std::unique_ptr<std::byte[]> owned_storage_;
    std::span<std::byte> storage_;
    std::size_t budget_bytes_{};
    std::size_t slot_bytes_{};
    std::size_t slot_stride_bytes_{};
    std::size_t slot_alignment_{1};
    std::size_t slot_count_{};
    std::size_t used_bytes_{};
    std::vector<std::size_t> free_slots_;
    std::list<ExpertKey> lru_;
    std::unordered_map<ExpertKey, Entry, ExpertKeyHash> entries_;
    std::unordered_map<ExpertKey, std::uint64_t, ExpertKeyHash> frequencies_;
    std::unordered_map<std::uint32_t, std::uint64_t> layer_requests_;
    std::uint64_t requests_{};
    CachePolicy policy_{CachePolicy::lru};
    CacheStats stats_{};
};

} // namespace h40

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace h40 {

class RamArena {
public:
    explicit RamArena(std::size_t capacity_bytes);

    [[nodiscard]] std::span<std::byte> allocate(std::size_t bytes, std::size_t alignment = 64);
    void reset() noexcept;

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t used() const noexcept { return cursor_; }
    [[nodiscard]] std::size_t available() const noexcept { return capacity_ - cursor_; }

private:
    std::unique_ptr<std::byte[]> storage_;
    std::size_t capacity_{};
    std::size_t cursor_{};
};

enum class ArenaRegionKind {
    resident,
    kv_state,
    expert_cache,
    scratch,
    io,
    output_head,
    embedding,
};

struct ArenaRegionPlan {
    ArenaRegionKind kind;
    std::string name;
    std::size_t bytes{};
    std::size_t alignment{64};
};

struct FixedMemoryPlan {
    std::size_t budget_bytes{};
    std::vector<ArenaRegionPlan> regions;
};

class FixedMemoryArenas {
public:
    explicit FixedMemoryArenas(FixedMemoryPlan plan);

    [[nodiscard]] std::span<std::byte> region(ArenaRegionKind kind);
    [[nodiscard]] std::span<const std::byte> region(ArenaRegionKind kind) const;
    [[nodiscard]] std::span<std::byte> region(std::string_view name);
    [[nodiscard]] std::span<const std::byte> region(std::string_view name) const;

    [[nodiscard]] std::size_t budget_bytes() const noexcept { return budget_bytes_; }
    [[nodiscard]] std::size_t committed_bytes() const noexcept { return committed_bytes_; }
    [[nodiscard]] std::size_t headroom_bytes() const noexcept { return budget_bytes_ - committed_bytes_; }
    [[nodiscard]] const std::vector<ArenaRegionPlan>& plan() const noexcept { return plan_; }

private:
    struct Region {
        ArenaRegionKind kind;
        std::string name;
        std::span<std::byte> bytes;
    };

    [[nodiscard]] Region& find_region(ArenaRegionKind kind);
    [[nodiscard]] const Region& find_region(ArenaRegionKind kind) const;
    [[nodiscard]] Region& find_region(std::string_view name);
    [[nodiscard]] const Region& find_region(std::string_view name) const;

    RamArena backing_;
    std::size_t budget_bytes_{};
    std::size_t committed_bytes_{};
    std::vector<ArenaRegionPlan> plan_;
    std::vector<Region> regions_;
};

} // namespace h40

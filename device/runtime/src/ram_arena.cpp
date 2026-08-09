#include "h40/ram_arena.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace h40 {

RamArena::RamArena(std::size_t capacity_bytes)
    : storage_(std::make_unique<std::byte[]>(capacity_bytes)), capacity_(capacity_bytes) {}

std::span<std::byte> RamArena::allocate(std::size_t bytes, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("alignment must be a non-zero power of two");
    }
    const auto base = reinterpret_cast<std::uintptr_t>(storage_.get());
    const auto current = base + cursor_;
    const auto aligned_address = (current + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    const auto aligned = static_cast<std::size_t>(aligned_address - base);
    if (aligned > capacity_ || bytes > capacity_ - aligned) {
        throw std::bad_alloc();
    }
    auto* ptr = storage_.get() + aligned;
    cursor_ = aligned + bytes;
    return {ptr, bytes};
}

void RamArena::reset() noexcept { cursor_ = 0; }

FixedMemoryArenas::FixedMemoryArenas(FixedMemoryPlan plan)
    : backing_(plan.budget_bytes), budget_bytes_(plan.budget_bytes), plan_(std::move(plan.regions)) {
    if (budget_bytes_ == 0) {
        throw std::invalid_argument("memory budget must be > 0");
    }
    regions_.reserve(plan_.size());
    for (const auto& region_plan : plan_) {
        if (region_plan.bytes == 0) {
            continue;
        }
        auto bytes = backing_.allocate(region_plan.bytes, region_plan.alignment);
        committed_bytes_ = backing_.used();
        regions_.push_back({region_plan.kind, region_plan.name, bytes});
    }
}

std::span<std::byte> FixedMemoryArenas::region(ArenaRegionKind kind) {
    return find_region(kind).bytes;
}

std::span<const std::byte> FixedMemoryArenas::region(ArenaRegionKind kind) const {
    return find_region(kind).bytes;
}

std::span<std::byte> FixedMemoryArenas::region(std::string_view name) {
    return find_region(name).bytes;
}

std::span<const std::byte> FixedMemoryArenas::region(std::string_view name) const {
    return find_region(name).bytes;
}

FixedMemoryArenas::Region& FixedMemoryArenas::find_region(ArenaRegionKind kind) {
    auto it = std::find_if(regions_.begin(), regions_.end(), [kind](const Region& region) {
        return region.kind == kind;
    });
    if (it == regions_.end()) {
        throw std::out_of_range("arena region kind is not configured");
    }
    return *it;
}

const FixedMemoryArenas::Region& FixedMemoryArenas::find_region(ArenaRegionKind kind) const {
    auto it = std::find_if(regions_.begin(), regions_.end(), [kind](const Region& region) {
        return region.kind == kind;
    });
    if (it == regions_.end()) {
        throw std::out_of_range("arena region kind is not configured");
    }
    return *it;
}

FixedMemoryArenas::Region& FixedMemoryArenas::find_region(std::string_view name) {
    auto it = std::find_if(regions_.begin(), regions_.end(), [name](const Region& region) {
        return region.name == name;
    });
    if (it == regions_.end()) {
        throw std::out_of_range("arena region name is not configured");
    }
    return *it;
}

const FixedMemoryArenas::Region& FixedMemoryArenas::find_region(std::string_view name) const {
    auto it = std::find_if(regions_.begin(), regions_.end(), [name](const Region& region) {
        return region.name == name;
    });
    if (it == regions_.end()) {
        throw std::out_of_range("arena region name is not configured");
    }
    return *it;
}

} // namespace h40

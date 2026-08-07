#include "h40/ram_arena.hpp"

#include <stdexcept>

namespace h40 {

RamArena::RamArena(std::size_t capacity_bytes)
    : storage_(std::make_unique<std::byte[]>(capacity_bytes)), capacity_(capacity_bytes) {}

std::span<std::byte> RamArena::allocate(std::size_t bytes, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("alignment must be a non-zero power of two");
    }
    const auto aligned = (cursor_ + alignment - 1) & ~(alignment - 1);
    if (aligned > capacity_ || bytes > capacity_ - aligned) {
        throw std::bad_alloc();
    }
    auto* ptr = storage_.get() + aligned;
    cursor_ = aligned + bytes;
    return {ptr, bytes};
}

void RamArena::reset() noexcept { cursor_ = 0; }

} // namespace h40

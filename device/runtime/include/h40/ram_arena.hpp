#pragma once

#include <cstddef>
#include <memory>
#include <span>

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

} // namespace h40

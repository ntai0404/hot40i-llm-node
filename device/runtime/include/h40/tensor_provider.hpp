#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace h40 {

struct TensorSlice {
    std::uint64_t offset{};
    std::uint64_t length{};
};

struct ReadStats {
    std::uint64_t operations{};
    std::uint64_t bytes{};
    std::uint64_t nanoseconds{};
};

class TensorProvider {
public:
    virtual ~TensorProvider() = default;
    virtual void read(const TensorSlice& slice, std::span<std::byte> out) = 0;
    [[nodiscard]] virtual ReadStats stats() const noexcept = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};

} // namespace h40

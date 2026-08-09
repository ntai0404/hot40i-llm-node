#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace h40 {

inline constexpr std::size_t kMxfp4BlockValues = 32;
inline constexpr std::size_t kMxfp4BlockPackedBytes = 1 + kMxfp4BlockValues / 2;

struct Mxfp4MatrixView {
    std::size_t rows{};
    std::size_t cols{};
    std::span<const std::uint8_t> blocks;
};

[[nodiscard]] float mxfp4_scale_to_float(std::uint8_t e8m0) noexcept;
[[nodiscard]] float mxfp4_value_to_float(std::uint8_t fp4, std::uint8_t e8m0) noexcept;

void dequantize_mxfp4_row(
    std::span<const std::uint8_t> row_blocks,
    std::size_t cols,
    std::span<float> out
);

void mxfp4_matvec(
    Mxfp4MatrixView matrix,
    std::span<const float> input,
    std::span<float> output
);

} // namespace h40

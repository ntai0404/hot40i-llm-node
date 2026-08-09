#include "h40/mxfp4.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace h40 {

namespace {

constexpr std::array<int, 16> kE2M1DoubledValues{
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

void validate_mxfp4_shape(std::size_t cols, std::span<const std::uint8_t> blocks) {
    if (cols == 0 || cols % kMxfp4BlockValues != 0) {
        throw std::invalid_argument("MXFP4 column count must be a non-zero multiple of 32");
    }
    const std::size_t expected = (cols / kMxfp4BlockValues) * kMxfp4BlockPackedBytes;
    if (blocks.size() != expected) {
        throw std::invalid_argument("MXFP4 row block byte count does not match columns");
    }
}

} // namespace

float mxfp4_scale_to_float(std::uint8_t e8m0) noexcept {
    return std::ldexp(1.0F, static_cast<int>(e8m0) - 128);
}

float mxfp4_value_to_float(std::uint8_t fp4, std::uint8_t e8m0) noexcept {
    return static_cast<float>(kE2M1DoubledValues[fp4 & 0x0FU]) * mxfp4_scale_to_float(e8m0);
}

void dequantize_mxfp4_row(
    std::span<const std::uint8_t> row_blocks,
    std::size_t cols,
    std::span<float> out
) {
    validate_mxfp4_shape(cols, row_blocks);
    if (out.size() != cols) {
        throw std::invalid_argument("MXFP4 dequant output size must equal columns");
    }

    const std::size_t blocks_per_row = cols / kMxfp4BlockValues;
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
        const auto base = block * kMxfp4BlockPackedBytes;
        const std::uint8_t scale = row_blocks[base];
        const std::uint8_t* packed = row_blocks.data() + base + 1;
        float* dst = out.data() + block * kMxfp4BlockValues;

        for (std::size_t j = 0; j < kMxfp4BlockValues / 2; ++j) {
            dst[j] = mxfp4_value_to_float(packed[j] & 0x0FU, scale);
            dst[j + kMxfp4BlockValues / 2] = mxfp4_value_to_float(packed[j] >> 4U, scale);
        }
    }
}

void mxfp4_matvec(
    Mxfp4MatrixView matrix,
    std::span<const float> input,
    std::span<float> output
) {
    if (matrix.rows == 0) {
        throw std::invalid_argument("MXFP4 matrix must have at least one row");
    }
    if (input.size() != matrix.cols) {
        throw std::invalid_argument("MXFP4 matvec input size must equal columns");
    }
    if (output.size() != matrix.rows) {
        throw std::invalid_argument("MXFP4 matvec output size must equal rows");
    }
    if (matrix.cols == 0 || matrix.cols % kMxfp4BlockValues != 0) {
        throw std::invalid_argument("MXFP4 column count must be a non-zero multiple of 32");
    }

    const std::size_t blocks_per_row = matrix.cols / kMxfp4BlockValues;
    const std::size_t row_bytes = blocks_per_row * kMxfp4BlockPackedBytes;
    if (matrix.blocks.size() != matrix.rows * row_bytes) {
        throw std::invalid_argument("MXFP4 matrix block byte count does not match shape");
    }

    for (std::size_t row = 0; row < matrix.rows; ++row) {
        float acc = 0.0F;
        const auto row_blocks = matrix.blocks.subspan(row * row_bytes, row_bytes);
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const auto base = block * kMxfp4BlockPackedBytes;
            const std::uint8_t scale = row_blocks[base];
            const std::uint8_t* packed = row_blocks.data() + base + 1;
            const float* x = input.data() + block * kMxfp4BlockValues;

            for (std::size_t j = 0; j < kMxfp4BlockValues / 2; ++j) {
                acc += x[j] * mxfp4_value_to_float(packed[j] & 0x0FU, scale);
                acc += x[j + kMxfp4BlockValues / 2] * mxfp4_value_to_float(packed[j] >> 4U, scale);
            }
        }
        output[row] = acc;
    }
}

} // namespace h40

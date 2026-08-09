#include "h40/mxfp4.hpp"

#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::array<int, 16> kReferenceE2M1Doubled{
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

void assert_close(float actual, float expected, float tolerance = 1.0e-6F) {
    assert(std::fabs(actual - expected) <= tolerance);
}

std::array<std::uint8_t, h40::kMxfp4BlockPackedBytes> make_block(std::uint8_t scale) {
    std::array<std::uint8_t, h40::kMxfp4BlockPackedBytes> block{};
    block[0] = scale;
    for (std::size_t i = 0; i < h40::kMxfp4BlockValues / 2; ++i) {
        block[i + 1] = static_cast<std::uint8_t>(i | ((15U - i) << 4U));
    }
    return block;
}

float reference_scale(std::uint8_t e8m0) {
    return std::ldexp(1.0F, static_cast<int>(e8m0) - 128);
}

float reference_value(std::uint8_t fp4, std::uint8_t e8m0) {
    return static_cast<float>(kReferenceE2M1Doubled[fp4 & 0x0FU]) * reference_scale(e8m0);
}

void dequant_matches_reference_layout() {
    const auto block = make_block(128);
    std::array<float, h40::kMxfp4BlockValues> out{};
    h40::dequantize_mxfp4_row(block, h40::kMxfp4BlockValues, out);

    for (std::size_t i = 0; i < h40::kMxfp4BlockValues / 2; ++i) {
        assert_close(out[i], reference_value(static_cast<std::uint8_t>(i), 128));
        assert_close(
            out[i + h40::kMxfp4BlockValues / 2],
            reference_value(static_cast<std::uint8_t>(15U - i), 128)
        );
    }
}

void scale_decodes_like_e8m0_half() {
    assert_close(h40::mxfp4_scale_to_float(128), 1.0F);
    assert_close(h40::mxfp4_scale_to_float(127), 0.5F);
    assert_close(h40::mxfp4_value_to_float(5, 127), 3.0F);
    assert_close(h40::mxfp4_value_to_float(13, 128), -6.0F);
}

void matvec_streams_packed_blocks() {
    std::array<std::uint8_t, h40::kMxfp4BlockPackedBytes * 2> blocks{};
    const auto row0 = make_block(127);
    const auto row1 = make_block(128);
    std::copy(row0.begin(), row0.end(), blocks.begin());
    std::copy(row1.begin(), row1.end(), blocks.begin() + h40::kMxfp4BlockPackedBytes);

    std::array<float, h40::kMxfp4BlockValues> input{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
    }

    std::array<float, 2> output{};
    h40::mxfp4_matvec({2, h40::kMxfp4BlockValues, blocks}, input, output);

    float expected0 = 0.0F;
    float expected1 = 0.0F;
    for (std::size_t packed = 0; packed < h40::kMxfp4BlockValues / 2; ++packed) {
        const std::uint8_t lo = row0[packed + 1] & 0x0FU;
        const std::uint8_t hi = row0[packed + 1] >> 4U;
        expected0 += input[packed] * reference_value(lo, row0[0]);
        expected0 += input[packed + h40::kMxfp4BlockValues / 2] * reference_value(hi, row0[0]);
        expected1 += input[packed] * reference_value(lo, row1[0]);
        expected1 += input[packed + h40::kMxfp4BlockValues / 2] * reference_value(hi, row1[0]);
    }
    assert_close(output[0], expected0);
    assert_close(output[1], expected1);
}

void invalid_shapes_throw() {
    const auto block = make_block(128);
    std::array<float, h40::kMxfp4BlockValues> out{};
    bool bad_cols = false;
    try {
        h40::dequantize_mxfp4_row(block, 16, out);
    } catch (const std::invalid_argument&) {
        bad_cols = true;
    }
    assert(bad_cols);

    bool bad_output = false;
    try {
        h40::mxfp4_matvec({1, h40::kMxfp4BlockValues, block}, std::span<const float>{}, out);
    } catch (const std::invalid_argument&) {
        bad_output = true;
    }
    assert(bad_output);
}

} // namespace

int main() {
    dequant_matches_reference_layout();
    scale_decodes_like_e8m0_half();
    matvec_streams_packed_blocks();
    invalid_shapes_throw();
    return 0;
}

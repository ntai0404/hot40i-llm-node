#include "h40/gptoss_expert.hpp"
#include "h40/mxfp4.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace {

void assert_close(std::span<const float> actual, std::span<const float> expected, float tolerance) {
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        assert(std::fabs(actual[i] - expected[i]) <= tolerance);
    }
}

std::uint8_t pack(std::uint8_t low, std::uint8_t high) {
    return static_cast<std::uint8_t>((high << 4U) | (low & 0x0FU));
}

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

void split_mxfp4_matvec_matches_direct_reference() {
    const std::size_t rows = 2;
    const std::size_t cols = 32;
    std::vector<std::uint8_t> blocks(rows * 16);
    std::vector<std::uint8_t> scales(rows, 128);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t i = 0; i < 16; ++i) {
            blocks[row * 16 + i] = pack(static_cast<std::uint8_t>((i + row) % 8), static_cast<std::uint8_t>(15 - i % 8));
        }
    }
    std::vector<float> input(cols);
    for (std::size_t i = 0; i < cols; ++i) input[i] = static_cast<float>(i % 7) * 0.25F - 0.5F;

    std::vector<float> actual(rows);
    h40::mxfp4_split_scales_matvec(rows, cols, blocks, scales, input, actual);

    std::vector<float> expected(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t i = 0; i < 16; ++i) {
            const auto byte = blocks[row * 16 + i];
            expected[row] += input[i] * h40::mxfp4_value_to_float(byte & 0x0FU, scales[row]);
            expected[row] += input[i + 16] * h40::mxfp4_value_to_float(byte >> 4U, scales[row]);
        }
    }
    assert_close(actual, expected, 1.0e-6F);
}

void gptoss_expert_matches_direct_reference() {
    constexpr std::size_t hidden = 32;
    constexpr std::size_t intermediate = 32;
    std::vector<float> input(hidden);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 5) * 0.1F - 0.2F;

    std::vector<std::uint8_t> gate_blocks(intermediate * 2 * 16, pack(1, 2));
    std::vector<std::uint8_t> gate_scales(intermediate * 2, 126);
    std::vector<std::uint8_t> down_blocks(hidden * 16, pack(3, 4));
    std::vector<std::uint8_t> down_scales(hidden, 125);
    std::vector<std::uint16_t> gate_bias(intermediate * 2);
    std::vector<std::uint16_t> down_bias(hidden);
    for (std::size_t i = 0; i < gate_bias.size(); ++i) gate_bias[i] = h40::float_to_bf16((i % 3) * 0.01F);
    for (std::size_t i = 0; i < down_bias.size(); ++i) down_bias[i] = h40::float_to_bf16(-0.02F + (i % 4) * 0.01F);

    h40::GptOssExpertView view{
        hidden,
        intermediate,
        down_bias,
        down_blocks,
        down_scales,
        gate_bias,
        gate_blocks,
        gate_scales,
    };
    std::vector<float> gate_up(intermediate * 2);
    std::vector<float> hidden_scratch(intermediate);
    std::vector<float> actual(hidden);
    h40::run_gptoss_expert(view, input, actual, {gate_up, hidden_scratch});

    std::vector<float> expected_gate_up(intermediate * 2);
    h40::mxfp4_split_scales_matvec(intermediate * 2, hidden, gate_blocks, gate_scales, input, expected_gate_up);
    for (std::size_t i = 0; i < expected_gate_up.size(); ++i) expected_gate_up[i] += h40::bf16_to_float(gate_bias[i]);
    std::vector<float> expected_hidden(intermediate);
    for (std::size_t i = 0; i < intermediate; ++i) {
        const float gate = std::min(expected_gate_up[i * 2], 7.0F);
        const float up = std::max(-7.0F, std::min(7.0F, expected_gate_up[i * 2 + 1]));
        expected_hidden[i] = (up + 1.0F) * gate * sigmoid(gate * 1.702F);
    }
    std::vector<float> expected(hidden);
    h40::mxfp4_split_scales_matvec(hidden, intermediate, down_blocks, down_scales, expected_hidden, expected);
    for (std::size_t i = 0; i < expected.size(); ++i) expected[i] += h40::bf16_to_float(down_bias[i]);

    assert_close(actual, expected, 1.0e-6F);
}

}  // namespace

int main() {
    split_mxfp4_matvec_matches_direct_reference();
    gptoss_expert_matches_direct_reference();
    return 0;
}

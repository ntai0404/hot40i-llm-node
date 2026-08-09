#include "h40/gptoss_expert.hpp"

#include "h40/mxfp4.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace h40 {

namespace {

constexpr float kGptOssExpertAlpha = 1.702F;
constexpr float kGptOssExpertLimit = 7.0F;
constexpr std::size_t kPackedBytesPerMxfp4Block = 16;

void validate_split_mxfp4_shape(
    std::size_t rows,
    std::size_t cols,
    std::span<const std::uint8_t> blocks,
    std::span<const std::uint8_t> scales) {
    if (rows == 0 || cols == 0 || cols % kMxfp4BlockValues != 0) {
        throw std::invalid_argument("split MXFP4 matrix shape must be non-empty and cols must be multiple of 32");
    }
    const auto blocks_per_row = cols / kMxfp4BlockValues;
    if (blocks.size() != rows * blocks_per_row * kPackedBytesPerMxfp4Block) {
        throw std::invalid_argument("split MXFP4 block byte count does not match shape");
    }
    if (scales.size() != rows * blocks_per_row) {
        throw std::invalid_argument("split MXFP4 scale byte count does not match shape");
    }
}

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

}  // namespace

float bf16_to_float(std::uint16_t value) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
    return std::bit_cast<float>(bits);
}

std::uint16_t float_to_bf16(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb = (bits >> 16U) & 1U;
    const std::uint32_t rounded = bits + 0x7fffU + lsb;
    return static_cast<std::uint16_t>(rounded >> 16U);
}

void mxfp4_split_scales_matvec(
    std::size_t rows,
    std::size_t cols,
    std::span<const std::uint8_t> blocks,
    std::span<const std::uint8_t> scales,
    std::span<const float> input,
    std::span<float> output) {
    validate_split_mxfp4_shape(rows, cols, blocks, scales);
    if (input.size() != cols) throw std::invalid_argument("split MXFP4 input size mismatch");
    if (output.size() != rows) throw std::invalid_argument("split MXFP4 output size mismatch");

    const auto blocks_per_row = cols / kMxfp4BlockValues;
    const auto row_block_bytes = blocks_per_row * kPackedBytesPerMxfp4Block;
    for (std::size_t row = 0; row < rows; ++row) {
        float acc = 0.0F;
        const auto row_blocks = blocks.subspan(row * row_block_bytes, row_block_bytes);
        const auto row_scales = scales.subspan(row * blocks_per_row, blocks_per_row);
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const auto scale = row_scales[block];
            const auto packed = row_blocks.subspan(block * kPackedBytesPerMxfp4Block, kPackedBytesPerMxfp4Block);
            const auto x = input.subspan(block * kMxfp4BlockValues, kMxfp4BlockValues);
            for (std::size_t i = 0; i < kPackedBytesPerMxfp4Block; ++i) {
                acc += x[i] * mxfp4_value_to_float(packed[i] & 0x0FU, scale);
                acc += x[i + kPackedBytesPerMxfp4Block] * mxfp4_value_to_float(packed[i] >> 4U, scale);
            }
        }
        output[row] = acc;
    }
}

void run_gptoss_expert(
    const GptOssExpertView& expert,
    std::span<const float> input,
    std::span<float> output,
    GptOssExpertScratch scratch) {
    if (expert.hidden_size == 0 || expert.intermediate_size == 0) {
        throw std::invalid_argument("gpt-oss expert dimensions must be non-zero");
    }
    if (input.size() != expert.hidden_size || output.size() != expert.hidden_size) {
        throw std::invalid_argument("gpt-oss expert input/output shape mismatch");
    }
    if (scratch.gate_up.size() < expert.intermediate_size * 2 || scratch.hidden.size() < expert.intermediate_size) {
        throw std::invalid_argument("gpt-oss expert scratch is too small");
    }
    if (expert.gate_up_bias_bf16.size() != expert.intermediate_size * 2 ||
        expert.down_bias_bf16.size() != expert.hidden_size) {
        throw std::invalid_argument("gpt-oss expert bias shape mismatch");
    }

    auto gate_up = scratch.gate_up.first(expert.intermediate_size * 2);
    auto hidden = scratch.hidden.first(expert.intermediate_size);
    mxfp4_split_scales_matvec(
        expert.intermediate_size * 2,
        expert.hidden_size,
        expert.gate_up_blocks,
        expert.gate_up_scales,
        input,
        gate_up);
    for (std::size_t i = 0; i < gate_up.size(); ++i) {
        gate_up[i] += bf16_to_float(expert.gate_up_bias_bf16[i]);
    }

    for (std::size_t i = 0; i < expert.intermediate_size; ++i) {
        float gate = std::min(gate_up[i * 2], kGptOssExpertLimit);
        float up = std::clamp(gate_up[i * 2 + 1], -kGptOssExpertLimit, kGptOssExpertLimit);
        const float glu = gate * sigmoid(gate * kGptOssExpertAlpha);
        hidden[i] = (up + 1.0F) * glu;
    }

    mxfp4_split_scales_matvec(
        expert.hidden_size,
        expert.intermediate_size,
        expert.down_blocks,
        expert.down_scales,
        hidden,
        output);
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] += bf16_to_float(expert.down_bias_bf16[i]);
    }
}

}  // namespace h40

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace h40 {

struct GptOssExpertView {
    std::size_t hidden_size{};
    std::size_t intermediate_size{};
    std::span<const std::uint16_t> down_bias_bf16;
    std::span<const std::uint8_t> down_blocks;
    std::span<const std::uint8_t> down_scales;
    std::span<const std::uint16_t> gate_up_bias_bf16;
    std::span<const std::uint8_t> gate_up_blocks;
    std::span<const std::uint8_t> gate_up_scales;
};

struct GptOssExpertScratch {
    std::span<float> gate_up;
    std::span<float> hidden;
};

[[nodiscard]] float bf16_to_float(std::uint16_t value) noexcept;
[[nodiscard]] std::uint16_t float_to_bf16(float value) noexcept;

void mxfp4_split_scales_matvec(
    std::size_t rows,
    std::size_t cols,
    std::span<const std::uint8_t> blocks,
    std::span<const std::uint8_t> scales,
    std::span<const float> input,
    std::span<float> output);

void run_gptoss_expert(
    const GptOssExpertView& expert,
    std::span<const float> input,
    std::span<float> output,
    GptOssExpertScratch scratch);

}  // namespace h40

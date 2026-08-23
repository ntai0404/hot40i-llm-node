#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace h40 {

struct GptOssKvCacheConfig {
    std::size_t layers{};
    std::size_t key_value_heads{};
    std::size_t head_dim{};
    std::size_t max_context_tokens{};
    std::size_t sliding_window{};
    bool first_layer_sliding{true};
};

inline constexpr std::size_t kGptOssDefaultContextTokens = 4096;
inline constexpr std::size_t kGptOssSlidingWindowTokens = 128;

class GptOssKvCache {
public:
    GptOssKvCache(GptOssKvCacheConfig config, std::span<std::byte> storage);

    [[nodiscard]] static std::size_t required_bytes(GptOssKvCacheConfig config);
    [[nodiscard]] static std::size_t layer_capacity(GptOssKvCacheConfig config, std::size_t layer);

    void append(
        std::size_t layer,
        std::size_t position,
        std::span<const float> key,
        std::span<const float> value);
    void clear() noexcept;

    [[nodiscard]] std::span<const std::uint16_t> key_bf16(std::size_t layer, std::size_t position) const;
    [[nodiscard]] std::span<const std::uint16_t> value_bf16(std::size_t layer, std::size_t position) const;
    [[nodiscard]] std::size_t valid_begin(std::size_t layer) const;
    [[nodiscard]] std::size_t next_position(std::size_t layer) const;
    [[nodiscard]] std::size_t capacity(std::size_t layer) const;
    [[nodiscard]] std::size_t storage_bytes() const noexcept { return required_bytes_; }
    [[nodiscard]] std::size_t vector_values() const noexcept { return vector_values_; }
    [[nodiscard]] bool is_sliding_layer(std::size_t layer) const;
    [[nodiscard]] GptOssKvCacheConfig config() const noexcept { return config_; }

private:
    [[nodiscard]] std::span<const std::uint16_t> vector_at(
        std::size_t layer,
        std::size_t position,
        bool value) const;

    GptOssKvCacheConfig config_{};
    std::span<std::byte> storage_;
    std::vector<std::size_t> layer_offsets_;
    std::vector<std::size_t> next_positions_;
    std::size_t vector_values_{};
    std::size_t vector_bytes_{};
    std::size_t required_bytes_{};
};

void gptoss_cached_attention(
    const GptOssKvCache& cache,
    std::size_t layer,
    std::size_t position,
    std::span<const float> query,
    std::span<const float> sinks,
    std::span<float> output,
    std::span<float> score_scratch);

}  // namespace h40

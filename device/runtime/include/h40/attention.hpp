#pragma once

#include <cstddef>
#include <span>

namespace h40 {

struct AttentionConfig {
    std::size_t sequence_length{};
    std::size_t model_dim{};
    std::size_t query_heads{};
    std::size_t key_value_heads{};
    std::size_t head_dim{};
};

void rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    float eps,
    std::span<float> output
);

void apply_rope_to_head(
    std::span<float> values,
    std::span<const float> cos,
    std::span<const float> sin
);

void causal_attention_projection(
    AttentionConfig config,
    std::span<const float> x_norm,
    std::span<const float> wq,
    std::span<const float> wk,
    std::span<const float> wv,
    std::span<const float> wo,
    std::span<const float> sinks,
    std::span<float> output
);

} // namespace h40

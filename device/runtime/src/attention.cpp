#include "h40/attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace h40 {

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void matmul_row(
    std::span<const float> row,
    std::span<const float> weights,
    std::size_t out_dim,
    std::span<float> output
) {
    require(output.size() == out_dim, "projection output size mismatch");
    require(weights.size() == row.size() * out_dim, "projection weight size mismatch");
    for (std::size_t col = 0; col < out_dim; ++col) {
        float acc = 0.0F;
        for (std::size_t i = 0; i < row.size(); ++i) {
            acc += row[i] * weights[i * out_dim + col];
        }
        output[col] = acc;
    }
}

float dot(std::span<const float> lhs, std::span<const float> rhs) {
    require(lhs.size() == rhs.size(), "dot input size mismatch");
    float acc = 0.0F;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        acc += lhs[i] * rhs[i];
    }
    return acc;
}

} // namespace

void rms_norm(
    std::span<const float> input,
    std::span<const float> weight,
    float eps,
    std::span<float> output
) {
    require(!input.empty(), "RMSNorm input must not be empty");
    require(input.size() == weight.size(), "RMSNorm weight size mismatch");
    require(input.size() == output.size(), "RMSNorm output size mismatch");
    require(eps >= 0.0F, "RMSNorm epsilon must be non-negative");

    double sum_sq = 0.0;
    for (const float value : input) {
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    const float inv_rms = 1.0F / std::sqrt(static_cast<float>(sum_sq / input.size()) + eps);
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

void apply_rope_to_head(
    std::span<float> values,
    std::span<const float> cos,
    std::span<const float> sin
) {
    require(values.size() % 2 == 0, "RoPE head dimension must be even");
    const std::size_t half = values.size() / 2;
    require(cos.size() == half && sin.size() == half, "RoPE table size mismatch");

    for (std::size_t i = 0; i < half; ++i) {
        const float x1 = values[i];
        const float x2 = values[i + half];
        values[i] = x1 * cos[i] - x2 * sin[i];
        values[i + half] = x2 * cos[i] + x1 * sin[i];
    }
}

void causal_attention_projection(
    AttentionConfig config,
    std::span<const float> x_norm,
    std::span<const float> wq,
    std::span<const float> wk,
    std::span<const float> wv,
    std::span<const float> wo,
    std::span<const float> sinks,
    std::span<float> output
) {
    require(config.sequence_length > 0, "attention sequence length must be non-zero");
    require(config.model_dim > 0, "attention model dimension must be non-zero");
    require(config.query_heads > 0, "attention query heads must be non-zero");
    require(config.key_value_heads > 0, "attention key/value heads must be non-zero");
    require(config.head_dim > 0, "attention head dimension must be non-zero");
    require(config.query_heads % config.key_value_heads == 0, "query heads must be a multiple of key/value heads");
    require(config.query_heads * config.head_dim == config.model_dim, "query heads/head_dim must equal model_dim");
    require(x_norm.size() == config.sequence_length * config.model_dim, "attention input size mismatch");
    require(wq.size() == config.model_dim * config.model_dim, "wq size mismatch");
    require(wk.size() == config.model_dim * config.key_value_heads * config.head_dim, "wk size mismatch");
    require(wv.size() == config.model_dim * config.key_value_heads * config.head_dim, "wv size mismatch");
    require(wo.size() == config.model_dim * config.model_dim, "wo size mismatch");
    require(sinks.empty() || sinks.size() == config.query_heads, "attention sink size mismatch");
    require(output.size() == config.sequence_length * config.model_dim, "attention output size mismatch");

    const std::size_t kv_dim = config.key_value_heads * config.head_dim;
    std::vector<float> q(config.sequence_length * config.model_dim);
    std::vector<float> k(config.sequence_length * kv_dim);
    std::vector<float> v(config.sequence_length * kv_dim);
    for (std::size_t pos = 0; pos < config.sequence_length; ++pos) {
        const auto row = x_norm.subspan(pos * config.model_dim, config.model_dim);
        matmul_row(row, wq, config.model_dim, std::span<float>(q).subspan(pos * config.model_dim, config.model_dim));
        matmul_row(row, wk, kv_dim, std::span<float>(k).subspan(pos * kv_dim, kv_dim));
        matmul_row(row, wv, kv_dim, std::span<float>(v).subspan(pos * kv_dim, kv_dim));
    }

    const float scale = 1.0F / std::sqrt(static_cast<float>(config.head_dim));
    std::vector<float> merged(config.model_dim);
    std::vector<float> projected(config.model_dim);
    std::vector<float> scores(config.sequence_length);

    for (std::size_t pos = 0; pos < config.sequence_length; ++pos) {
        std::fill(merged.begin(), merged.end(), 0.0F);
        for (std::size_t q_head = 0; q_head < config.query_heads; ++q_head) {
            const std::size_t kv_head = q_head / (config.query_heads / config.key_value_heads);
            const auto qv = std::span<const float>(q).subspan(
                pos * config.model_dim + q_head * config.head_dim,
                config.head_dim
            );

            float max_score = sinks.empty() ? -std::numeric_limits<float>::infinity() : sinks[q_head];
            for (std::size_t src = 0; src <= pos; ++src) {
                const auto kv = std::span<const float>(k).subspan(
                    src * kv_dim + kv_head * config.head_dim,
                    config.head_dim
                );
                scores[src] = dot(qv, kv) * scale;
                max_score = std::max(max_score, scores[src]);
            }

            double normalizer = sinks.empty() ? 0.0 : std::exp(static_cast<double>(sinks[q_head] - max_score));
            for (std::size_t src = 0; src <= pos; ++src) {
                normalizer += std::exp(static_cast<double>(scores[src] - max_score));
            }

            for (std::size_t dim = 0; dim < config.head_dim; ++dim) {
                double acc = 0.0;
                for (std::size_t src = 0; src <= pos; ++src) {
                    const double prob = std::exp(static_cast<double>(scores[src] - max_score)) / normalizer;
                    const auto vv = std::span<const float>(v).subspan(
                        src * kv_dim + kv_head * config.head_dim,
                        config.head_dim
                    );
                    acc += prob * vv[dim];
                }
                merged[q_head * config.head_dim + dim] = static_cast<float>(acc);
            }
        }

        matmul_row(merged, wo, config.model_dim, projected);
        std::copy(projected.begin(), projected.end(), output.begin() + static_cast<std::ptrdiff_t>(pos * config.model_dim));
    }
}

void gptoss_attention_projection(
    GptOssAttentionConfig config,
    std::span<const float> x_norm,
    std::span<const float> wq,
    std::span<const float> q_bias,
    std::span<const float> wk,
    std::span<const float> k_bias,
    std::span<const float> wv,
    std::span<const float> v_bias,
    std::span<const float> wo,
    std::span<const float> o_bias,
    std::span<const float> sinks,
    std::span<const float> rope_cos,
    std::span<const float> rope_sin,
    std::span<float> output
) {
    require(config.sequence_length > 0, "gpt-oss attention sequence length must be non-zero");
    require(config.model_dim > 0, "gpt-oss attention model dimension must be non-zero");
    require(config.query_heads > 0, "gpt-oss attention query heads must be non-zero");
    require(config.key_value_heads > 0, "gpt-oss attention key/value heads must be non-zero");
    require(config.head_dim > 0 && config.head_dim % 2 == 0, "gpt-oss attention head_dim must be even");
    require(config.query_heads % config.key_value_heads == 0, "gpt-oss query heads must be multiple of key/value heads");
    const std::size_t q_dim = config.query_heads * config.head_dim;
    require(x_norm.size() == config.sequence_length * config.model_dim, "gpt-oss attention input size mismatch");
    const std::size_t kv_dim = config.key_value_heads * config.head_dim;
    require(wq.size() == config.model_dim * q_dim, "gpt-oss wq size mismatch");
    require(wk.size() == config.model_dim * kv_dim, "gpt-oss wk size mismatch");
    require(wv.size() == config.model_dim * kv_dim, "gpt-oss wv size mismatch");
    require(wo.size() == q_dim * config.model_dim, "gpt-oss wo size mismatch");
    require(q_bias.size() == q_dim, "gpt-oss q bias size mismatch");
    require(k_bias.size() == kv_dim, "gpt-oss k bias size mismatch");
    require(v_bias.size() == kv_dim, "gpt-oss v bias size mismatch");
    require(o_bias.size() == config.model_dim, "gpt-oss o bias size mismatch");
    require(sinks.size() == config.query_heads, "gpt-oss attention sinks size mismatch");
    require(rope_cos.size() == config.sequence_length * (config.head_dim / 2), "gpt-oss rope cos size mismatch");
    require(rope_sin.size() == config.sequence_length * (config.head_dim / 2), "gpt-oss rope sin size mismatch");
    require(output.size() == config.sequence_length * config.model_dim, "gpt-oss attention output size mismatch");

    std::vector<float> q(config.sequence_length * q_dim);
    std::vector<float> k(config.sequence_length * kv_dim);
    std::vector<float> v(config.sequence_length * kv_dim);
    for (std::size_t pos = 0; pos < config.sequence_length; ++pos) {
        const auto row = x_norm.subspan(pos * config.model_dim, config.model_dim);
        auto q_row = std::span<float>(q).subspan(pos * q_dim, q_dim);
        auto k_row = std::span<float>(k).subspan(pos * kv_dim, kv_dim);
        auto v_row = std::span<float>(v).subspan(pos * kv_dim, kv_dim);
        matmul_row(row, wq, q_dim, q_row);
        matmul_row(row, wk, kv_dim, k_row);
        matmul_row(row, wv, kv_dim, v_row);
        for (std::size_t i = 0; i < q_dim; ++i) q_row[i] += q_bias[i];
        for (std::size_t i = 0; i < kv_dim; ++i) {
            k_row[i] += k_bias[i];
            v_row[i] += v_bias[i];
        }
        const auto cos = rope_cos.subspan(pos * (config.head_dim / 2), config.head_dim / 2);
        const auto sin = rope_sin.subspan(pos * (config.head_dim / 2), config.head_dim / 2);
        for (std::size_t q_head = 0; q_head < config.query_heads; ++q_head) {
            apply_rope_to_head(q_row.subspan(q_head * config.head_dim, config.head_dim), cos, sin);
        }
        for (std::size_t kv_head = 0; kv_head < config.key_value_heads; ++kv_head) {
            apply_rope_to_head(k_row.subspan(kv_head * config.head_dim, config.head_dim), cos, sin);
        }
    }

    const float scale = 1.0F / std::sqrt(static_cast<float>(config.head_dim));
    std::vector<float> merged(q_dim);
    std::vector<float> projected(config.model_dim);
    std::vector<float> scores(config.sequence_length);
    const std::size_t kv_group = config.query_heads / config.key_value_heads;

    for (std::size_t pos = 0; pos < config.sequence_length; ++pos) {
        std::fill(merged.begin(), merged.end(), 0.0F);
        const std::size_t min_src =
            config.sliding_window == 0 || pos + 1 <= config.sliding_window ? 0 : pos + 1 - config.sliding_window;
        for (std::size_t q_head = 0; q_head < config.query_heads; ++q_head) {
            const std::size_t kv_head = q_head / kv_group;
            const auto qv = std::span<const float>(q).subspan(
                pos * q_dim + q_head * config.head_dim,
                config.head_dim);
            float max_score = sinks[q_head];
            for (std::size_t src = min_src; src <= pos; ++src) {
                const auto kv = std::span<const float>(k).subspan(
                    src * kv_dim + kv_head * config.head_dim,
                    config.head_dim);
                scores[src] = dot(qv, kv) * scale;
                max_score = std::max(max_score, scores[src]);
            }
            double normalizer = std::exp(static_cast<double>(sinks[q_head] - max_score));
            for (std::size_t src = min_src; src <= pos; ++src) {
                normalizer += std::exp(static_cast<double>(scores[src] - max_score));
            }
            for (std::size_t dim = 0; dim < config.head_dim; ++dim) {
                double acc = 0.0;
                for (std::size_t src = min_src; src <= pos; ++src) {
                    const double prob = std::exp(static_cast<double>(scores[src] - max_score)) / normalizer;
                    const auto vv = std::span<const float>(v).subspan(
                        src * kv_dim + kv_head * config.head_dim,
                        config.head_dim);
                    acc += prob * vv[dim];
                }
                merged[q_head * config.head_dim + dim] = static_cast<float>(acc);
            }
        }
        matmul_row(merged, wo, config.model_dim, projected);
        for (std::size_t i = 0; i < config.model_dim; ++i) output[pos * config.model_dim + i] = projected[i] + o_bias[i];
    }
}

} // namespace h40

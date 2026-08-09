#include "h40/attention.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

namespace {

float max_abs_diff(std::span<const float> lhs, std::span<const float> rhs) {
    assert(lhs.size() == rhs.size());
    float out = 0.0F;
    for (std::size_t i = 0; i < lhs.size(); ++i) out = std::max(out, std::fabs(lhs[i] - rhs[i]));
    return out;
}

std::vector<float> matrix(std::size_t rows, std::size_t cols, float scale) {
    std::vector<float> values(rows * cols);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = (static_cast<float>(static_cast<int>(i % 17) - 8)) * scale;
    }
    return values;
}

void matmul_row(std::span<const float> row, std::span<const float> weights, std::span<float> out) {
    const std::size_t cols = out.size();
    for (std::size_t col = 0; col < cols; ++col) {
        float acc = 0.0F;
        for (std::size_t i = 0; i < row.size(); ++i) acc += row[i] * weights[i * cols + col];
        out[col] = acc;
    }
}

float dot(std::span<const float> a, std::span<const float> b) {
    float out = 0.0F;
    for (std::size_t i = 0; i < a.size(); ++i) out += a[i] * b[i];
    return out;
}

std::vector<float> reference_attention(
    h40::GptOssAttentionConfig cfg,
    std::span<const float> x,
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
    std::span<const float> rope_sin) {
    const std::size_t kv_dim = cfg.key_value_heads * cfg.head_dim;
    const std::size_t q_dim = cfg.query_heads * cfg.head_dim;
    std::vector<float> q(cfg.sequence_length * q_dim);
    std::vector<float> k(cfg.sequence_length * kv_dim);
    std::vector<float> v(cfg.sequence_length * kv_dim);
    for (std::size_t pos = 0; pos < cfg.sequence_length; ++pos) {
        auto row = x.subspan(pos * cfg.model_dim, cfg.model_dim);
        auto q_row = std::span<float>(q).subspan(pos * q_dim, q_dim);
        auto k_row = std::span<float>(k).subspan(pos * kv_dim, kv_dim);
        auto v_row = std::span<float>(v).subspan(pos * kv_dim, kv_dim);
        matmul_row(row, wq, q_row);
        matmul_row(row, wk, k_row);
        matmul_row(row, wv, v_row);
        for (std::size_t i = 0; i < q_dim; ++i) q_row[i] += q_bias[i];
        for (std::size_t i = 0; i < kv_dim; ++i) {
            k_row[i] += k_bias[i];
            v_row[i] += v_bias[i];
        }
        auto cos = rope_cos.subspan(pos * cfg.head_dim / 2, cfg.head_dim / 2);
        auto sin = rope_sin.subspan(pos * cfg.head_dim / 2, cfg.head_dim / 2);
        for (std::size_t head = 0; head < cfg.query_heads; ++head) {
            h40::apply_rope_to_head(q_row.subspan(head * cfg.head_dim, cfg.head_dim), cos, sin);
        }
        for (std::size_t head = 0; head < cfg.key_value_heads; ++head) {
            h40::apply_rope_to_head(k_row.subspan(head * cfg.head_dim, cfg.head_dim), cos, sin);
        }
    }

    std::vector<float> out(cfg.sequence_length * cfg.model_dim);
    std::vector<float> merged(q_dim);
    std::vector<float> projected(cfg.model_dim);
    std::vector<float> scores(cfg.sequence_length);
    const float scale = 1.0F / std::sqrt(static_cast<float>(cfg.head_dim));
    const std::size_t group = cfg.query_heads / cfg.key_value_heads;
    for (std::size_t pos = 0; pos < cfg.sequence_length; ++pos) {
        std::fill(merged.begin(), merged.end(), 0.0F);
        const std::size_t min_src = cfg.sliding_window == 0 || pos + 1 <= cfg.sliding_window ? 0 : pos + 1 - cfg.sliding_window;
        for (std::size_t qh = 0; qh < cfg.query_heads; ++qh) {
            const std::size_t kvh = qh / group;
            auto qv = std::span<const float>(q).subspan(pos * q_dim + qh * cfg.head_dim, cfg.head_dim);
            float max_score = sinks[qh];
            for (std::size_t src = min_src; src <= pos; ++src) {
                auto kv = std::span<const float>(k).subspan(src * kv_dim + kvh * cfg.head_dim, cfg.head_dim);
                scores[src] = dot(qv, kv) * scale;
                max_score = std::max(max_score, scores[src]);
            }
            double denom = std::exp(static_cast<double>(sinks[qh] - max_score));
            for (std::size_t src = min_src; src <= pos; ++src) denom += std::exp(static_cast<double>(scores[src] - max_score));
            for (std::size_t dim = 0; dim < cfg.head_dim; ++dim) {
                double acc = 0.0;
                for (std::size_t src = min_src; src <= pos; ++src) {
                    const double prob = std::exp(static_cast<double>(scores[src] - max_score)) / denom;
                    auto vv = std::span<const float>(v).subspan(src * kv_dim + kvh * cfg.head_dim, cfg.head_dim);
                    acc += prob * vv[dim];
                }
                merged[qh * cfg.head_dim + dim] = static_cast<float>(acc);
            }
        }
        matmul_row(merged, wo, projected);
        for (std::size_t i = 0; i < cfg.model_dim; ++i) out[pos * cfg.model_dim + i] = projected[i] + o_bias[i];
    }
    return out;
}

void gptoss_attention_matches_reference() {
    h40::GptOssAttentionConfig cfg{5, 6, 2, 1, 4, 3};
    const std::size_t q_dim = cfg.query_heads * cfg.head_dim;
    auto x = matrix(cfg.sequence_length, cfg.model_dim, 0.03F);
    auto wq = matrix(cfg.model_dim, q_dim, 0.02F);
    auto wk = matrix(cfg.model_dim, cfg.key_value_heads * cfg.head_dim, 0.025F);
    auto wv = matrix(cfg.model_dim, cfg.key_value_heads * cfg.head_dim, 0.015F);
    auto wo = matrix(q_dim, cfg.model_dim, 0.018F);
    auto q_bias = matrix(1, q_dim, 0.005F);
    auto k_bias = matrix(1, cfg.key_value_heads * cfg.head_dim, 0.006F);
    auto v_bias = matrix(1, cfg.key_value_heads * cfg.head_dim, 0.007F);
    auto o_bias = matrix(1, cfg.model_dim, 0.004F);
    std::vector<float> sinks{0.1F, -0.2F};
    std::vector<float> cos(cfg.sequence_length * cfg.head_dim / 2);
    std::vector<float> sin(cfg.sequence_length * cfg.head_dim / 2);
    for (std::size_t pos = 0; pos < cfg.sequence_length; ++pos) {
        for (std::size_t i = 0; i < cfg.head_dim / 2; ++i) {
            const float angle = static_cast<float>(pos + 1) * static_cast<float>(i + 1) * 0.1F;
            cos[pos * cfg.head_dim / 2 + i] = std::cos(angle);
            sin[pos * cfg.head_dim / 2 + i] = std::sin(angle);
        }
    }

    const auto expected = reference_attention(cfg, x, wq, q_bias, wk, k_bias, wv, v_bias, wo, o_bias, sinks, cos, sin);
    std::vector<float> actual(expected.size());
    h40::gptoss_attention_projection(cfg, x, wq, q_bias, wk, k_bias, wv, v_bias, wo, o_bias, sinks, cos, sin, actual);
    const float diff = max_abs_diff(actual, expected);
    std::cout << "gptoss_attention_max_abs_diff=" << diff << "\n";
    assert(diff <= 1.0e-6F);
}

void full_attention_differs_from_sliding_attention() {
    h40::GptOssAttentionConfig sliding{5, 6, 2, 1, 4, 2};
    h40::GptOssAttentionConfig full = sliding;
    full.sliding_window = 0;
    const std::size_t q_dim = sliding.query_heads * sliding.head_dim;
    auto x = matrix(sliding.sequence_length, sliding.model_dim, 0.05F);
    auto wq = matrix(sliding.model_dim, q_dim, 0.02F);
    auto wk = matrix(sliding.model_dim, sliding.key_value_heads * sliding.head_dim, 0.03F);
    auto wv = matrix(sliding.model_dim, sliding.key_value_heads * sliding.head_dim, 0.04F);
    auto wo = matrix(q_dim, sliding.model_dim, 0.01F);
    std::vector<float> q_bias(q_dim, 0.0F);
    std::vector<float> k_bias(sliding.key_value_heads * sliding.head_dim, 0.0F);
    std::vector<float> v_bias(sliding.key_value_heads * sliding.head_dim, 0.0F);
    std::vector<float> o_bias(sliding.model_dim, 0.0F);
    std::vector<float> sinks(sliding.query_heads, -3.0F);
    std::vector<float> cos(sliding.sequence_length * sliding.head_dim / 2, 1.0F);
    std::vector<float> sin(sliding.sequence_length * sliding.head_dim / 2, 0.0F);
    std::vector<float> sliding_out(sliding.sequence_length * sliding.model_dim);
    std::vector<float> full_out(sliding.sequence_length * sliding.model_dim);
    h40::gptoss_attention_projection(sliding, x, wq, q_bias, wk, k_bias, wv, v_bias, wo, o_bias, sinks, cos, sin, sliding_out);
    h40::gptoss_attention_projection(full, x, wq, q_bias, wk, k_bias, wv, v_bias, wo, o_bias, sinks, cos, sin, full_out);
    const float diff = max_abs_diff(sliding_out, full_out);
    std::cout << "gptoss_sliding_vs_full_max_abs_diff=" << diff << "\n";
    assert(diff > 1.0e-7F);
}

}  // namespace

int main() {
    gptoss_attention_matches_reference();
    full_attention_differs_from_sliding_attention();
    return 0;
}

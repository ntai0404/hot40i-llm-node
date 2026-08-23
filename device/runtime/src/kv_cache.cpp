#include "h40/kv_cache.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace h40 {

namespace {

std::size_t checked_add(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error("KV cache byte count overflow");
    }
    return left + right;
}

std::size_t checked_mul(std::size_t left, std::size_t right) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error("KV cache byte count overflow");
    }
    return left * right;
}

void validate_config(GptOssKvCacheConfig config) {
    if (config.layers == 0 || config.key_value_heads == 0 || config.head_dim == 0 ||
        config.max_context_tokens == 0 || config.sliding_window == 0) {
        throw std::invalid_argument("KV cache dimensions and sliding window must be non-zero");
    }
    if (config.sliding_window > config.max_context_tokens) {
        throw std::invalid_argument("KV sliding window exceeds maximum context");
    }
}

std::uint16_t float_to_bf16(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb = (bits >> 16U) & 1U;
    return static_cast<std::uint16_t>((bits + 0x7fffU + lsb) >> 16U);
}

float bf16_to_float(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

}  // namespace

std::size_t GptOssKvCache::layer_capacity(GptOssKvCacheConfig config, std::size_t layer) {
    validate_config(config);
    if (layer >= config.layers) throw std::out_of_range("KV cache layer outside model");
    const bool even = layer % 2 == 0;
    const bool sliding = config.first_layer_sliding ? even : !even;
    return sliding ? config.sliding_window : config.max_context_tokens;
}

std::size_t GptOssKvCache::required_bytes(GptOssKvCacheConfig config) {
    validate_config(config);
    const auto values = checked_mul(config.key_value_heads, config.head_dim);
    const auto token_bytes = checked_mul(checked_mul(values, sizeof(std::uint16_t)), 2);
    std::size_t total = 0;
    for (std::size_t layer = 0; layer < config.layers; ++layer) {
        total = checked_add(total, checked_mul(layer_capacity(config, layer), token_bytes));
    }
    return total;
}

GptOssKvCache::GptOssKvCache(GptOssKvCacheConfig config, std::span<std::byte> storage)
    : config_(config),
      storage_(storage),
      layer_offsets_(config.layers),
      next_positions_(config.layers, 0),
      vector_values_(checked_mul(config.key_value_heads, config.head_dim)),
      vector_bytes_(checked_mul(vector_values_, sizeof(std::uint16_t))),
      required_bytes_(required_bytes(config)) {
    if (storage_.size() < required_bytes_) throw std::invalid_argument("KV cache storage is below fixed budget");
    if (reinterpret_cast<std::uintptr_t>(storage_.data()) % alignof(std::uint16_t) != 0) {
        throw std::invalid_argument("KV cache storage is not BF16-aligned");
    }
    std::size_t offset = 0;
    for (std::size_t layer = 0; layer < config_.layers; ++layer) {
        layer_offsets_[layer] = offset;
        offset = checked_add(offset, checked_mul(capacity(layer), checked_mul(vector_bytes_, 2)));
    }
}

bool GptOssKvCache::is_sliding_layer(std::size_t layer) const {
    if (layer >= config_.layers) throw std::out_of_range("KV cache layer outside model");
    const bool even = layer % 2 == 0;
    return config_.first_layer_sliding ? even : !even;
}

std::size_t GptOssKvCache::capacity(std::size_t layer) const {
    return layer_capacity(config_, layer);
}

std::size_t GptOssKvCache::next_position(std::size_t layer) const {
    if (layer >= config_.layers) throw std::out_of_range("KV cache layer outside model");
    return next_positions_[layer];
}

std::size_t GptOssKvCache::valid_begin(std::size_t layer) const {
    const auto next = next_position(layer);
    const auto retained = capacity(layer);
    return next > retained ? next - retained : 0;
}

void GptOssKvCache::append(
    std::size_t layer,
    std::size_t position,
    std::span<const float> key,
    std::span<const float> value) {
    if (layer >= config_.layers) throw std::out_of_range("KV cache layer outside model");
    if (position != next_positions_[layer]) throw std::invalid_argument("KV cache append must be sequential per layer");
    if (position >= config_.max_context_tokens) throw std::length_error("KV cache maximum context exceeded");
    if (key.size() != vector_values_ || value.size() != vector_values_) {
        throw std::invalid_argument("KV cache vector shape mismatch");
    }
    const auto slot = is_sliding_layer(layer) ? position % capacity(layer) : position;
    const auto byte_offset = layer_offsets_[layer] + slot * vector_bytes_ * 2;
    auto* words = reinterpret_cast<std::uint16_t*>(storage_.data() + byte_offset);
    for (std::size_t index = 0; index < vector_values_; ++index) {
        words[index] = float_to_bf16(key[index]);
        words[vector_values_ + index] = float_to_bf16(value[index]);
    }
    ++next_positions_[layer];
}

std::span<const std::uint16_t> GptOssKvCache::vector_at(
    std::size_t layer,
    std::size_t position,
    bool value) const {
    const auto next = next_position(layer);
    if (position >= next || position < valid_begin(layer)) {
        throw std::out_of_range("KV cache position is not retained");
    }
    const auto slot = is_sliding_layer(layer) ? position % capacity(layer) : position;
    const auto byte_offset = layer_offsets_[layer] + slot * vector_bytes_ * 2 + (value ? vector_bytes_ : 0);
    const auto* words = reinterpret_cast<const std::uint16_t*>(storage_.data() + byte_offset);
    return {words, vector_values_};
}

std::span<const std::uint16_t> GptOssKvCache::key_bf16(std::size_t layer, std::size_t position) const {
    return vector_at(layer, position, false);
}

std::span<const std::uint16_t> GptOssKvCache::value_bf16(std::size_t layer, std::size_t position) const {
    return vector_at(layer, position, true);
}

void GptOssKvCache::clear() noexcept {
    auto used = storage_.first(required_bytes_);
    std::fill(used.begin(), used.end(), std::byte{0});
    std::fill(next_positions_.begin(), next_positions_.end(), 0);
}

void gptoss_cached_attention(
    const GptOssKvCache& cache,
    std::size_t layer,
    std::size_t position,
    std::span<const float> query,
    std::span<const float> sinks,
    std::span<float> output,
    std::span<float> score_scratch) {
    const auto config = cache.config();
    if (position >= cache.next_position(layer) || position < cache.valid_begin(layer)) {
        throw std::out_of_range("cached attention position is not retained");
    }
    const auto q_dim = query.size();
    if (q_dim == 0 || q_dim % config.head_dim != 0 || output.size() != q_dim) {
        throw std::invalid_argument("cached attention query/output shape mismatch");
    }
    const auto query_heads = q_dim / config.head_dim;
    if (query_heads % config.key_value_heads != 0 || sinks.size() != query_heads) {
        throw std::invalid_argument("cached attention GQA/sink shape mismatch");
    }
    const auto begin = cache.valid_begin(layer);
    const auto count = position - begin + 1;
    if (score_scratch.size() < count) throw std::invalid_argument("cached attention score scratch is too small");

    const auto group = query_heads / config.key_value_heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(config.head_dim));
    std::fill(output.begin(), output.end(), 0.0F);
    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
        const auto kv_head = query_head / group;
        const auto q = query.subspan(query_head * config.head_dim, config.head_dim);
        float max_score = sinks[query_head];
        for (std::size_t source = begin; source <= position; ++source) {
            const auto key = cache.key_bf16(layer, source).subspan(kv_head * config.head_dim, config.head_dim);
            float score = 0.0F;
            for (std::size_t dim = 0; dim < config.head_dim; ++dim) {
                score += q[dim] * bf16_to_float(key[dim]);
            }
            score *= scale;
            score_scratch[source - begin] = score;
            max_score = std::max(max_score, score);
        }
        double denominator = std::exp(static_cast<double>(sinks[query_head] - max_score));
        for (std::size_t source = begin; source <= position; ++source) {
            denominator += std::exp(static_cast<double>(score_scratch[source - begin] - max_score));
        }
        auto result = output.subspan(query_head * config.head_dim, config.head_dim);
        for (std::size_t source = begin; source <= position; ++source) {
            const auto probability = static_cast<float>(
                std::exp(static_cast<double>(score_scratch[source - begin] - max_score)) / denominator);
            const auto value = cache.value_bf16(layer, source).subspan(kv_head * config.head_dim, config.head_dim);
            for (std::size_t dim = 0; dim < config.head_dim; ++dim) {
                result[dim] += probability * bf16_to_float(value[dim]);
            }
        }
    }
}

}  // namespace h40

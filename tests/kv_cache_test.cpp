#include "h40/kv_cache.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

std::uint16_t float_to_bf16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto lsb = (bits >> 16U) & 1U;
    return static_cast<std::uint16_t>((bits + 0x7fffU + lsb) >> 16U);
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

float max_abs_diff(std::span<const float> left, std::span<const float> right) {
    assert(left.size() == right.size());
    float result = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result = std::max(result, std::fabs(left[index] - right[index]));
    }
    return result;
}

std::vector<float> reference_attention(
    std::span<const std::vector<float>> keys,
    std::span<const std::vector<float>> values,
    std::size_t begin,
    std::span<const float> query,
    std::span<const float> sinks,
    std::size_t kv_heads,
    std::size_t head_dim) {
    const auto query_heads = query.size() / head_dim;
    const auto group = query_heads / kv_heads;
    const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> output(query.size(), 0.0F);
    std::vector<float> scores(keys.size() - begin);
    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
        const auto kv_head = query_head / group;
        float max_score = sinks[query_head];
        for (std::size_t position = begin; position < keys.size(); ++position) {
            float score = 0.0F;
            for (std::size_t dim = 0; dim < head_dim; ++dim) {
                score += query[query_head * head_dim + dim] * keys[position][kv_head * head_dim + dim];
            }
            scores[position - begin] = score * scale;
            max_score = std::max(max_score, scores[position - begin]);
        }
        double denominator = std::exp(static_cast<double>(sinks[query_head] - max_score));
        for (const auto score : scores) denominator += std::exp(static_cast<double>(score - max_score));
        for (std::size_t position = begin; position < keys.size(); ++position) {
            const auto probability = static_cast<float>(
                std::exp(static_cast<double>(scores[position - begin] - max_score)) / denominator);
            for (std::size_t dim = 0; dim < head_dim; ++dim) {
                output[query_head * head_dim + dim] +=
                    probability * values[position][kv_head * head_dim + dim];
            }
        }
    }
    return output;
}

void official_memory_sweep_matches() {
    constexpr std::size_t mib = 1024 * 1024;
    static_assert(h40::kGptOssDefaultContextTokens == 4096);
    static_assert(h40::kGptOssSlidingWindowTokens == 128);
    assert(h40::GptOssKvCache::required_bytes({24, 8, 64, 512, 128, true}) == 15 * mib);
    assert(h40::GptOssKvCache::required_bytes({24, 8, 64, 1024, 128, true}) == 27 * mib);
    assert(h40::GptOssKvCache::required_bytes({24, 8, 64, 2048, 128, true}) == 51 * mib);
    assert(h40::GptOssKvCache::required_bytes({24, 8, 64, 4096, 128, true}) == 99 * mib);
}

void ring_and_attention_match_reference() {
    const h40::GptOssKvCacheConfig config{2, 1, 4, 8, 3, true};
    assert(h40::GptOssKvCache::required_bytes(config) == 176);
    std::vector<std::byte> storage(h40::GptOssKvCache::required_bytes(config));
    h40::GptOssKvCache cache(config, storage);
    std::vector<std::vector<float>> keys(8, std::vector<float>(4));
    std::vector<std::vector<float>> values(8, std::vector<float>(4));
    for (std::size_t position = 0; position < 8; ++position) {
        for (std::size_t dim = 0; dim < 4; ++dim) {
            keys[position][dim] = static_cast<float>(
                static_cast<int>(position) * 7 + static_cast<int>(dim) - 11) * 0.031F;
            values[position][dim] = static_cast<float>(
                static_cast<int>(position) * 5 + static_cast<int>(dim) - 8) * 0.027F;
        }
        cache.append(0, position, keys[position], values[position]);
        cache.append(1, position, keys[position], values[position]);
    }
    assert(cache.valid_begin(0) == 5);
    assert(cache.valid_begin(1) == 0);
    assert(cache.capacity(0) == 3);
    assert(cache.capacity(1) == 8);
    bool old_position_rejected = false;
    try {
        (void)cache.key_bf16(0, 4);
    } catch (const std::out_of_range&) {
        old_position_rejected = true;
    }
    assert(old_position_rejected);

    std::vector<std::vector<float>> quantized_keys = keys;
    std::vector<std::vector<float>> quantized_values = values;
    float roundtrip_error = 0.0F;
    for (std::size_t position = 0; position < 8; ++position) {
        for (std::size_t dim = 0; dim < 4; ++dim) {
            quantized_keys[position][dim] = bf16_to_float(float_to_bf16(keys[position][dim]));
            quantized_values[position][dim] = bf16_to_float(float_to_bf16(values[position][dim]));
            roundtrip_error = std::max(roundtrip_error, std::fabs(quantized_keys[position][dim] - keys[position][dim]));
            roundtrip_error = std::max(roundtrip_error, std::fabs(quantized_values[position][dim] - values[position][dim]));
        }
    }
    const std::vector<float> query{0.2F, -0.1F, 0.4F, 0.3F, -0.2F, 0.5F, 0.1F, -0.4F};
    const std::vector<float> sinks{0.15F, -0.25F};
    std::vector<float> actual(query.size());
    std::vector<float> scratch(8);
    h40::gptoss_cached_attention(cache, 0, 7, query, sinks, actual, scratch);
    const auto quantized_reference = reference_attention(quantized_keys, quantized_values, 5, query, sinks, 1, 4);
    const auto fp32_reference = reference_attention(keys, values, 5, query, sinks, 1, 4);
    const auto cache_diff = max_abs_diff(actual, quantized_reference);
    const auto quantization_diff = max_abs_diff(actual, fp32_reference);
    std::cout << "kv_cache_bf16_reference_max_abs_diff=" << cache_diff << "\n";
    std::cout << "kv_cache_fp32_reference_max_abs_diff=" << quantization_diff << "\n";
    std::cout << "kv_cache_roundtrip_max_abs_diff=" << roundtrip_error << "\n";
    assert(cache_diff <= 1.0e-6F);
    assert(quantization_diff <= 1.0e-3F);
    assert(roundtrip_error <= 4.0e-3F);

    bool nonsequential_rejected = false;
    try {
        cache.append(0, 7, keys[0], values[0]);
    } catch (const std::invalid_argument&) {
        nonsequential_rejected = true;
    }
    assert(nonsequential_rejected);
    cache.clear();
    assert(cache.next_position(0) == 0 && cache.next_position(1) == 0);
}

}  // namespace

int main() {
    official_memory_sweep_matches();
    ring_and_attention_match_reference();
    return 0;
}

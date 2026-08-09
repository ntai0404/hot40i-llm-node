#pragma once

#include "h40/expert_cache.hpp"
#include "h40/expert_loader.hpp"
#include "h40/trace.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace h40 {

struct MoELayerConfig {
    std::uint32_t layer{};
    std::size_t token_count{};
    std::size_t expert_count{};
    std::size_t top_k{};
    std::size_t model_dim{};
};

struct MoESchedulerScratch {
    std::span<std::uint32_t> expert_ids;
    std::span<float> expert_weights;
    std::span<float> expert_output;
};

namespace detail {

inline TraceEvent moe_event(
    std::string_view event,
    std::uint64_t token,
    std::uint64_t layer,
    std::uint64_t expert) {
    TraceEvent row;
    row.event = event;
    row.token = token;
    row.layer = layer;
    row.expert = expert;
    row.has_token = true;
    row.has_layer = true;
    row.has_expert = true;
    return row;
}

inline std::uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

inline bool better_router_logit(std::span<const float> logits, std::uint32_t lhs, std::uint32_t rhs) {
    const float a = logits[lhs];
    const float b = logits[rhs];
    if (a == b) return lhs < rhs;
    return a > b;
}

inline void select_top_k_noalloc(
    std::span<const float> logits,
    std::span<std::uint32_t> expert_ids,
    std::span<float> weights) {
    if (expert_ids.empty()) throw std::invalid_argument("top_k must be greater than zero");
    if (expert_ids.size() != weights.size()) throw std::invalid_argument("top-k scratch size mismatch");
    if (expert_ids.size() > logits.size()) throw std::invalid_argument("top_k cannot exceed expert count");
    for (const float logit : logits) {
        if (!std::isfinite(logit)) throw std::invalid_argument("router logits must be finite");
    }

    for (std::size_t i = 0; i < expert_ids.size(); ++i) expert_ids[i] = static_cast<std::uint32_t>(i);
    for (std::size_t candidate = expert_ids.size(); candidate < logits.size(); ++candidate) {
        std::size_t worst = 0;
        for (std::size_t i = 1; i < expert_ids.size(); ++i) {
            if (better_router_logit(logits, expert_ids[worst], expert_ids[i])) worst = i;
        }
        const auto id = static_cast<std::uint32_t>(candidate);
        if (better_router_logit(logits, id, expert_ids[worst])) expert_ids[worst] = id;
    }
    std::sort(expert_ids.begin(), expert_ids.end(), [&](std::uint32_t lhs, std::uint32_t rhs) {
        return better_router_logit(logits, lhs, rhs);
    });

    float max_selected = logits[expert_ids.front()];
    for (const auto id : expert_ids) max_selected = std::max(max_selected, logits[id]);
    double normalizer = 0.0;
    for (std::size_t i = 0; i < expert_ids.size(); ++i) {
        const double value = std::exp(static_cast<double>(logits[expert_ids[i]] - max_selected));
        weights[i] = static_cast<float>(value);
        normalizer += value;
    }
    for (auto& weight : weights) weight = static_cast<float>(static_cast<double>(weight) / normalizer);
}

}  // namespace detail

template <typename ComputeExpert>
void run_moe_layer_streaming(
    const MoELayerConfig& config,
    std::span<const float> router_logits,
    ExpertCache& cache,
    const ExpertLoader& loader,
    std::span<float> output,
    MoESchedulerScratch scratch,
    ComputeExpert&& compute_expert,
    JsonlTraceWriter* trace = nullptr,
    bool verify_checksum = false) {
    if (router_logits.size() != config.token_count * config.expert_count) {
        throw std::invalid_argument("router logits shape mismatch");
    }
    if (output.size() != config.token_count * config.model_dim) {
        throw std::invalid_argument("MoE output shape mismatch");
    }
    if (scratch.expert_ids.size() < config.top_k || scratch.expert_weights.size() < config.top_k) {
        throw std::invalid_argument("top-k scratch is too small");
    }
    if (scratch.expert_output.size() < config.model_dim) {
        throw std::invalid_argument("expert output scratch is too small");
    }

    std::fill(output.begin(), output.end(), 0.0F);
    if (trace) {
        TraceEvent row;
        row.event = "layer_begin";
        row.layer = config.layer;
        row.has_layer = true;
        trace->emit(row);
    }

    const auto ids = scratch.expert_ids.first(config.top_k);
    const auto weights = scratch.expert_weights.first(config.top_k);
    const auto expert_output = scratch.expert_output.first(config.model_dim);

    for (std::size_t token = 0; token < config.token_count; ++token) {
        detail::select_top_k_noalloc(
            router_logits.subspan(token * config.expert_count, config.expert_count),
            ids,
            weights);
        for (std::size_t choice = 0; choice < config.top_k; ++choice) {
            const auto expert_id = ids[choice];
            if (trace) trace->emit(detail::moe_event("route", token, config.layer, expert_id));

            const ExpertKey key{config.layer, expert_id};
            const bool will_hit = cache.contains(key);
            const auto before_cache = cache.stats();
            auto read_start = std::chrono::steady_clock::now();
            if (trace && !will_hit) {
                auto row = detail::moe_event("read_begin", token, config.layer, expert_id);
                row.bytes = loader.index().find_record(key).value().slice.length;
                row.has_bytes = true;
                trace->emit(row);
            }
            const auto loaded = cache.get_or_load(key, loader, verify_checksum);
            const auto read_ns = detail::elapsed_ns(read_start);
            if (trace) {
                auto cache_row = detail::moe_event(loaded.hit ? "cache_hit" : "cache_miss", token, config.layer, expert_id);
                cache_row.cache_hit = loaded.hit;
                cache_row.has_cache_hit = true;
                trace->emit(cache_row);
                if (!loaded.hit) {
                    auto row = detail::moe_event("read_end", token, config.layer, expert_id);
                    row.bytes = loaded.bytes.size();
                    row.duration_ns = read_ns;
                    row.has_bytes = true;
                    row.has_duration = true;
                    trace->emit(row);
                }
            }
            const auto after_cache = cache.stats();
            if (!loaded.hit && after_cache.bytes_loaded <= before_cache.bytes_loaded) {
                throw std::logic_error("cache miss did not load expert bytes");
            }

            std::fill(expert_output.begin(), expert_output.end(), 0.0F);
            const auto compute_start = std::chrono::steady_clock::now();
            if (trace) trace->emit(detail::moe_event("compute_begin", token, config.layer, expert_id));
            compute_expert(token, expert_id, loaded.bytes, expert_output);
            const auto compute_ns = detail::elapsed_ns(compute_start);
            if (trace) {
                auto row = detail::moe_event("compute_end", token, config.layer, expert_id);
                row.duration_ns = compute_ns;
                row.has_duration = true;
                trace->emit(row);
            }

            auto token_output = output.subspan(token * config.model_dim, config.model_dim);
            for (std::size_t i = 0; i < config.model_dim; ++i) {
                token_output[i] += weights[choice] * expert_output[i];
            }
        }
    }

    if (trace) {
        TraceEvent row;
        row.event = "layer_end";
        row.layer = config.layer;
        row.has_layer = true;
        trace->emit(row);
    }
}

}  // namespace h40

#include "h40/expert_loader.hpp"
#include "h40/flash_tensor_provider.hpp"
#include "h40/moe_scheduler.hpp"
#include "h40/router.hpp"
#include "h40/runtime_graph.hpp"
#include "h40/trace.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kTokens = 6;
constexpr std::size_t kModelDim = 16;
constexpr std::size_t kVocab = 24;
constexpr std::size_t kExperts = 8;
constexpr std::size_t kTopK = 4;
constexpr std::size_t kPackedExpertFloats = kModelDim * kModelDim;
constexpr std::size_t kPackedExpertBytes = kPackedExpertFloats * sizeof(float);
constexpr std::size_t kCacheSlots = 4;
constexpr float kTolerance = 1.0e-5F;

struct GraphRunMetrics {
    h40::BoundedRuntimeGraphMetrics graph{};
    h40::CacheStats cache{};
    h40::ReadStats provider{};
    std::size_t rejected_full_checkpoint_bytes{};
    float max_abs_moe_diff{};
    float max_abs_logit_diff{};
    bool full_checkpoint_residency_rejected{};
};

float deterministic_weight(std::size_t a, std::size_t b, std::size_t c) {
    const auto value = static_cast<int>((a * 17 + b * 31 + c * 13) % 29) - 14;
    return static_cast<float>(value) * 0.0075F;
}

float max_abs_diff(std::span<const float> lhs, std::span<const float> rhs) {
    assert(lhs.size() == rhs.size());
    float out = 0.0F;
    for (std::size_t i = 0; i < lhs.size(); ++i) out = std::max(out, std::fabs(lhs[i] - rhs[i]));
    return out;
}

template <typename T>
void copy_to_region(std::span<std::byte> region, std::span<const T> values) {
    assert(region.size() >= values.size_bytes());
    std::memcpy(region.data(), values.data(), values.size_bytes());
}

std::vector<float> make_embeddings() {
    std::vector<float> values(kTokens * kModelDim);
    for (std::size_t token = 0; token < kTokens; ++token) {
        for (std::size_t dim = 0; dim < kModelDim; ++dim) {
            values[token * kModelDim + dim] = deterministic_weight(token + 1, dim + 3, 0) + 0.01F * static_cast<float>(token);
        }
    }
    return values;
}

std::vector<float> make_output_head() {
    std::vector<float> values(kModelDim * kVocab);
    for (std::size_t row = 0; row < kModelDim; ++row) {
        for (std::size_t col = 0; col < kVocab; ++col) {
            values[row * kVocab + col] = deterministic_weight(row + 7, col + 11, 3);
        }
    }
    return values;
}

std::vector<float> make_expert_matrix(std::size_t expert) {
    std::vector<float> values(kPackedExpertFloats);
    for (std::size_t row = 0; row < kModelDim; ++row) {
        for (std::size_t col = 0; col < kModelDim; ++col) {
            values[row * kModelDim + col] = deterministic_weight(expert + 5, row + 1, col + 9);
        }
    }
    return values;
}

std::vector<float> make_router_logits() {
    const std::uint32_t choices[kTokens][kTopK] = {
        {0, 1, 2, 3},
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {4, 5, 6, 7},
        {0, 2, 4, 6},
        {1, 3, 5, 7},
    };
    std::vector<float> logits(kTokens * kExperts, -8.0F);
    for (std::size_t token = 0; token < kTokens; ++token) {
        for (std::size_t rank = 0; rank < kTopK; ++rank) {
            logits[token * kExperts + choices[token][rank]] = 4.0F - static_cast<float>(rank);
        }
    }
    return logits;
}

void compute_expert(std::span<const float> token_row, std::span<const std::byte> packed, std::span<float> out) {
    assert(packed.size() == kPackedExpertBytes);
    const auto* weights = reinterpret_cast<const float*>(packed.data());
    for (std::size_t col = 0; col < kModelDim; ++col) {
        float acc = 0.0F;
        for (std::size_t row = 0; row < kModelDim; ++row) {
            acc += token_row[row] * weights[row * kModelDim + col];
        }
        out[col] = std::tanh(acc);
    }
}

std::vector<float> matmul_rows(std::span<const float> rows, std::span<const float> weights) {
    std::vector<float> out(kTokens * kVocab);
    for (std::size_t token = 0; token < kTokens; ++token) {
        for (std::size_t col = 0; col < kVocab; ++col) {
            float acc = 0.0F;
            for (std::size_t row = 0; row < kModelDim; ++row) {
                acc += rows[token * kModelDim + row] * weights[row * kVocab + col];
            }
            out[token * kVocab + col] = acc;
        }
    }
    return out;
}

std::filesystem::path write_expert_arena(h40::ModelIndex& index, std::vector<float>& resident_experts) {
    const auto path = std::filesystem::temp_directory_path() / "h40_bounded_graph_experts.bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    assert(out);
    resident_experts.clear();
    resident_experts.reserve(kExperts * kPackedExpertFloats);
    for (std::size_t expert = 0; expert < kExperts; ++expert) {
        const auto matrix = make_expert_matrix(expert);
        const auto offset = expert * kPackedExpertBytes;
        index.put({0, static_cast<std::uint32_t>(expert)}, {offset, kPackedExpertBytes});
        out.write(
            reinterpret_cast<const char*>(matrix.data()),
            static_cast<std::streamsize>(matrix.size() * sizeof(float)));
        resident_experts.insert(resident_experts.end(), matrix.begin(), matrix.end());
    }
    return path;
}

std::vector<float> resident_reference(
    std::span<const float> embeddings,
    std::span<const float> router_logits,
    std::span<const float> resident_experts) {
    std::vector<float> output(kTokens * kModelDim, 0.0F);
    std::vector<float> expert_out(kModelDim);
    for (std::size_t token = 0; token < kTokens; ++token) {
        const auto selected = h40::select_top_k_experts(router_logits.subspan(token * kExperts, kExperts), kTopK);
        for (std::size_t choice = 0; choice < kTopK; ++choice) {
            const auto expert = selected.expert_ids[choice];
            const auto expert_bytes = std::as_bytes(
                resident_experts.subspan(static_cast<std::size_t>(expert) * kPackedExpertFloats, kPackedExpertFloats));
            compute_expert(embeddings.subspan(token * kModelDim, kModelDim), expert_bytes, expert_out);
            for (std::size_t dim = 0; dim < kModelDim; ++dim) {
                output[token * kModelDim + dim] += selected.weights[choice] * expert_out[dim];
            }
        }
    }
    return output;
}

h40::BoundedRuntimeGraphConfig scaled_config() {
    constexpr std::size_t resident = 32 * 1024;
    constexpr std::size_t embedding = kTokens * kModelDim * sizeof(float);
    constexpr std::size_t output_head = kModelDim * kVocab * sizeof(float);
    constexpr std::size_t attention = 24 * 1024;
    constexpr std::size_t expert_cache = kCacheSlots * kPackedExpertBytes;
    constexpr std::size_t kv = kTokens * kModelDim * 2 * sizeof(float);
    constexpr std::size_t io = 8 * 1024;
    constexpr std::size_t scratch = 16 * 1024;
    constexpr std::size_t budget = resident + embedding + output_head + attention + expert_cache + kv + io + scratch + 4096;
    return {
        budget,
        resident,
        embedding,
        output_head,
        attention,
        expert_cache,
        kv,
        io,
        scratch,
        kPackedExpertBytes,
        64,
        0,
        false,
        "cpu_sync_scaled_fixture",
    };
}

void write_report(
    const std::filesystem::path& path,
    const h40::BoundedRuntimeGraphConfig& config,
    const GraphRunMetrics& metrics,
    bool stopped_cleanly) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"status\": \"pass\",\n";
    out << "  \"fixture\": \"scaled_bounded_graph_v1\",\n";
    out << "  \"tokens\": " << kTokens << ",\n";
    out << "  \"model_dim\": " << kModelDim << ",\n";
    out << "  \"experts\": " << kExperts << ",\n";
    out << "  \"top_k\": " << kTopK << ",\n";
    out << "  \"safe_rss_budget_bytes\": " << metrics.graph.safe_rss_budget_bytes << ",\n";
    out << "  \"committed_bytes\": " << metrics.graph.committed_bytes << ",\n";
    out << "  \"headroom_bytes\": " << metrics.graph.headroom_bytes << ",\n";
    out << "  \"peak_allocation_bytes\": " << metrics.graph.peak_allocation_bytes << ",\n";
    out << "  \"resident_bytes\": " << config.resident_bytes << ",\n";
    out << "  \"embedding_cache_bytes\": " << config.embedding_cache_bytes << ",\n";
    out << "  \"output_head_chunk_bytes\": " << config.output_head_chunk_bytes << ",\n";
    out << "  \"attention_stream_bytes\": " << config.attention_stream_bytes << ",\n";
    out << "  \"expert_cache_bytes\": " << config.expert_cache_bytes << ",\n";
    out << "  \"kv_state_bytes\": " << config.kv_state_bytes << ",\n";
    out << "  \"io_bytes\": " << config.io_bytes << ",\n";
    out << "  \"scratch_bytes\": " << config.scratch_bytes << ",\n";
    out << "  \"full_checkpoint_residency_allowed\": false,\n";
    out << "  \"full_checkpoint_residency_rejected\": " << (metrics.full_checkpoint_residency_rejected ? "true" : "false") << ",\n";
    out << "  \"rejected_full_checkpoint_bytes\": " << metrics.rejected_full_checkpoint_bytes << ",\n";
    out << "  \"runtime_started\": true,\n";
    out << "  \"runtime_stopped_cleanly\": " << (stopped_cleanly ? "true" : "false") << ",\n";
    out << "  \"max_abs_moe_diff\": " << metrics.max_abs_moe_diff << ",\n";
    out << "  \"max_abs_logit_diff\": " << metrics.max_abs_logit_diff << ",\n";
    out << "  \"cache_hits\": " << metrics.cache.hits << ",\n";
    out << "  \"cache_misses\": " << metrics.cache.misses << ",\n";
    out << "  \"cache_evictions\": " << metrics.cache.evictions << ",\n";
    out << "  \"cache_bytes_loaded\": " << metrics.cache.bytes_loaded << ",\n";
    out << "  \"peak_cache_used_bytes\": " << metrics.cache.peak_used_bytes << ",\n";
    out << "  \"provider_operations\": " << metrics.provider.operations << ",\n";
    out << "  \"provider_bytes\": " << metrics.provider.bytes << ",\n";
    out << "  \"expert_bytes_per_miss\": " << kPackedExpertBytes << ",\n";
    out << "  \"tolerance\": " << kTolerance << "\n";
    out << "}\n";
}

GraphRunMetrics run_graph(
    const std::filesystem::path& report_path,
    const std::filesystem::path& trace_path) {
    auto config = scaled_config();
    const std::size_t synthetic_full_checkpoint_bytes =
        config.resident_bytes + kExperts * kPackedExpertBytes + config.output_head_chunk_bytes + config.embedding_cache_bytes;
    auto invalid_config = config;
    invalid_config.full_checkpoint_bytes = synthetic_full_checkpoint_bytes;
    bool full_checkpoint_rejected = false;
    try {
        h40::BoundedRuntimeGraph rejected(invalid_config);
        (void)rejected;
    } catch (const std::invalid_argument&) {
        full_checkpoint_rejected = true;
    }
    assert(full_checkpoint_rejected);

    h40::BoundedRuntimeGraph graph(config);
    graph.start();
    assert(graph.started());

    const auto embeddings = make_embeddings();
    const auto output_head = make_output_head();
    const auto router_logits = make_router_logits();
    copy_to_region(graph.embedding_cache(), std::span<const float>(embeddings));
    copy_to_region(graph.output_head_chunk(), std::span<const float>(output_head));
    std::fill(graph.kv_state().begin(), graph.kv_state().end(), std::byte{0});
    std::fill(graph.attention_stream().begin(), graph.attention_stream().end(), std::byte{0});
    std::fill(graph.io().begin(), graph.io().end(), std::byte{0});
    std::fill(graph.scratch().begin(), graph.scratch().end(), std::byte{0});

    h40::ModelIndex index;
    std::vector<float> resident_experts;
    const auto arena_path = write_expert_arena(index, resident_experts);
    const auto reference_moe = resident_reference(embeddings, router_logits, resident_experts);
    const auto reference_logits = matmul_rows(reference_moe, output_head);

    std::vector<float> streaming_moe(kTokens * kModelDim);
    h40::CacheStats cache_stats{};
    h40::ReadStats provider_stats{};
    {
        h40::FlashTensorProvider provider(arena_path);
        h40::ExpertLoader loader(index, provider);
        auto cache = graph.make_expert_cache();
        std::uint32_t selected_ids[kTopK]{};
        float selected_weights[kTopK]{};
        float expert_output[kModelDim]{};
        std::filesystem::create_directories(trace_path.parent_path());
        std::ofstream trace_out(trace_path);
        h40::JsonlTraceWriter trace(trace_out);
        h40::run_moe_layer_streaming(
            {0, kTokens, kExperts, kTopK, kModelDim},
            router_logits,
            cache,
            loader,
            streaming_moe,
            {selected_ids, selected_weights, expert_output},
            [&](std::size_t token, std::uint32_t, std::span<const std::byte> bytes, std::span<float> out) {
                compute_expert(std::span<const float>(embeddings).subspan(token * kModelDim, kModelDim), bytes, out);
            },
            &trace,
            false);
        cache_stats = cache.stats();
        provider_stats = provider.stats();
        assert(cache.used_bytes() <= cache.budget_bytes());
        assert(cache_stats.peak_used_bytes <= cache.budget_bytes());
    }
    std::filesystem::remove(arena_path);

    const auto streaming_logits = matmul_rows(streaming_moe, output_head);
    GraphRunMetrics metrics;
    metrics.graph = graph.metrics();
    metrics.cache = cache_stats;
    metrics.provider = provider_stats;
    metrics.rejected_full_checkpoint_bytes = synthetic_full_checkpoint_bytes;
    metrics.max_abs_moe_diff = max_abs_diff(streaming_moe, reference_moe);
    metrics.max_abs_logit_diff = max_abs_diff(streaming_logits, reference_logits);
    metrics.full_checkpoint_residency_rejected = full_checkpoint_rejected;

    assert(metrics.graph.committed_bytes <= metrics.graph.safe_rss_budget_bytes);
    assert(metrics.graph.peak_allocation_bytes <= metrics.graph.safe_rss_budget_bytes);
    assert(metrics.cache.hits > 0);
    assert(metrics.cache.misses > 0);
    assert(metrics.cache.evictions > 0);
    assert(metrics.cache.bytes_loaded == metrics.cache.misses * kPackedExpertBytes);
    assert(metrics.provider.operations == metrics.cache.misses);
    assert(metrics.provider.bytes == metrics.cache.bytes_loaded);
    assert(metrics.max_abs_moe_diff <= kTolerance);
    assert(metrics.max_abs_logit_diff <= kTolerance);

    graph.stop();
    const bool stopped_cleanly = !graph.started();
    assert(stopped_cleanly);
    write_report(report_path, config, metrics, stopped_cleanly);
    return metrics;
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path report_path = argc > 1 ? argv[1] : "benchmarks/custom/bounded_graph.json";
    const std::filesystem::path trace_path = argc > 2 ? argv[2] : "artifacts/runs/bounded_graph_trace.jsonl";
    (void)run_graph(report_path, trace_path);
    return 0;
}

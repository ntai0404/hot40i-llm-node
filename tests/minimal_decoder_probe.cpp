#include "h40/attention.hpp"
#include "h40/expert_cache.hpp"
#include "h40/expert_loader.hpp"
#include "h40/flash_tensor_provider.hpp"
#include "h40/gptoss_expert.hpp"
#include "h40/h40m_tensor_catalog.hpp"
#include "h40/model_index.hpp"
#include "h40/moe_scheduler.hpp"
#include "h40/parallel_bf16.hpp"
#include "h40/trace.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

constexpr std::size_t kLayers = 24;
constexpr std::size_t kHidden = 2880;
constexpr std::size_t kIntermediate = 2880;
constexpr std::size_t kExperts = 32;
constexpr std::size_t kTopK = 4;
constexpr std::size_t kQHeads = 64;
constexpr std::size_t kKvHeads = 8;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kQDim = kQHeads * kHeadDim;
constexpr std::size_t kKvDim = kKvHeads * kHeadDim;
constexpr std::size_t kVocab = 201088;
constexpr std::size_t kExpertPayloadBytes = 13236480;
constexpr std::size_t kExpertStrideBytes = 13631488;
constexpr std::size_t kLmHeadChunkRows = 8192;
constexpr std::size_t kMaxDenseThreads = 8;
constexpr std::size_t kDefaultDenseThreads = 6;

struct Metrics {
    std::uint64_t dense_bytes{};
    std::uint64_t expert_flash_bytes{};
    std::uint64_t expert_cache_hits{};
    std::uint64_t expert_cache_misses{};
    std::uint64_t layers_run{};
    std::uint32_t token_id{};
    float token_logit{-std::numeric_limits<float>::infinity()};
    std::uint64_t peak_rss_kib{};
    std::uint64_t prefetched_experts{};
    std::uint64_t prefetch_read_ns{};
    std::uint64_t prefetch_wait_ns{};
    std::uint64_t embedding_ns{};
    std::uint64_t dense_matvec_ns{};
    std::uint64_t attention_ns{};
    std::uint64_t moe_ns{};
    std::uint64_t lm_head_ns{};
    std::size_t dense_threads{1};
    std::string cache_policy{"lru"};
    bool io_overlap_enabled{};
};

std::uint64_t elapsed_ms(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
}

std::vector<std::uint32_t> parse_tokens(std::string_view text) {
    std::vector<std::uint32_t> tokens;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto comma = text.find(',', start);
        const auto part = text.substr(start, comma == std::string_view::npos ? text.size() - start : comma - start);
        if (!part.empty()) tokens.push_back(static_cast<std::uint32_t>(std::stoul(std::string(part))));
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    if (tokens.empty()) throw std::invalid_argument("at least one token id is required");
    return tokens;
}

h40::H40mTensorRecord must_find(const h40::H40mTensorCatalog& catalog, const std::string& name) {
    auto record = catalog.find(name);
    if (!record.has_value()) throw std::runtime_error("missing tensor: " + name);
    return *record;
}

void add_bias(std::span<float> values, std::span<const float> bias) {
    if (values.size() != bias.size()) throw std::invalid_argument("bias size mismatch");
    for (std::size_t i = 0; i < values.size(); ++i) values[i] += bias[i];
}

void add_inplace(std::span<float> lhs, std::span<const float> rhs) {
    if (lhs.size() != rhs.size()) throw std::invalid_argument("residual size mismatch");
    for (std::size_t i = 0; i < lhs.size(); ++i) lhs[i] += rhs[i];
}

void bf16_matvec_counted(
    h40::ParallelBf16Matvec& executor,
    const h40::H40mTensorRecord& record,
    std::span<const float> input,
    std::span<float> output,
    std::size_t workers,
    Metrics& metrics) {
    const auto start = std::chrono::steady_clock::now();
    executor.matvec(record, input, output, workers);
    metrics.dense_matvec_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    metrics.dense_bytes += record.length;
}

void bf16_vector_counted(
    const h40::FileTensorReader& reader,
    const h40::H40mTensorRecord& record,
    std::span<float> output,
    Metrics& metrics) {
    reader.read_bf16_vector(record, output);
    metrics.dense_bytes += record.length;
}

void bf16_row_counted(
    const h40::FileTensorReader& reader,
    const h40::H40mTensorRecord& record,
    std::size_t row,
    std::span<float> output,
    Metrics& metrics) {
    reader.read_bf16_row(record, row, output);
    metrics.dense_bytes += record.shape[1] * sizeof(std::uint16_t);
}

h40::ModelIndex build_expert_index() {
    h40::ModelIndex index;
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
            const auto ordinal = static_cast<std::uint64_t>(layer) * kExperts + expert;
            index.put({layer, expert}, {ordinal * kExpertStrideBytes, kExpertPayloadBytes});
        }
    }
    return index;
}

h40::GptOssExpertView expert_view(std::span<const std::byte> bytes) {
    if (bytes.size() != kExpertPayloadBytes) throw std::invalid_argument("unexpected expert payload size");
    const auto* base = bytes.data();
    auto u16 = [](const std::byte* ptr, std::size_t count) {
        return std::span<const std::uint16_t>(reinterpret_cast<const std::uint16_t*>(ptr), count);
    };
    auto u8 = [](const std::byte* ptr, std::size_t count) {
        return std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(ptr), count);
    };
    return {
        kHidden,
        kIntermediate,
        u16(base + 0, kHidden),
        u8(base + 5760, kHidden * 90 * 16),
        u8(base + 4152960, kHidden * 90),
        u16(base + 4412160, kIntermediate * 2),
        u8(base + 4423680, kIntermediate * 2 * 90 * 16),
        u8(base + 12718080, kIntermediate * 2 * 90),
    };
}

void yarn_rope_tables(std::size_t seq_len, std::span<float> cos, std::span<float> sin) {
    if (cos.size() != seq_len * (kHeadDim / 2) || sin.size() != cos.size()) {
        throw std::invalid_argument("rope table shape mismatch");
    }
    constexpr double base = 150000.0;
    constexpr double factor = 32.0;
    constexpr double beta_fast = 32.0;
    constexpr double beta_slow = 1.0;
    constexpr double original_max_position_embeddings = 4096.0;
    constexpr double pi = 3.14159265358979323846264338327950288;
    const double attention_factor = 0.1 * std::log(factor) + 1.0;
    auto correction_dim = [](double rotations) {
        return (static_cast<double>(kHeadDim) * std::log(original_max_position_embeddings / (rotations * 2.0 * pi))) /
               (2.0 * std::log(base));
    };
    const double low = std::max(correction_dim(beta_fast), 0.0);
    const double high = std::min(correction_dim(beta_slow), static_cast<double>(kHeadDim - 1));
    for (std::size_t i = 0; i < kHeadDim / 2; ++i) {
        const double pos_freq = std::pow(base, static_cast<double>(i * 2) / static_cast<double>(kHeadDim));
        const double inv_extrapolate = 1.0 / pos_freq;
        const double inv_interpolate = 1.0 / (factor * pos_freq);
        const double ramp = std::clamp((static_cast<double>(i) - low) / (high - low), 0.0, 1.0);
        const double extrapolate_factor = 1.0 - ramp;
        const double inv_freq = inv_interpolate * (1.0 - extrapolate_factor) + inv_extrapolate * extrapolate_factor;
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            const double angle = static_cast<double>(pos) * inv_freq;
            cos[pos * (kHeadDim / 2) + i] = static_cast<float>(std::cos(angle) * attention_factor);
            sin[pos * (kHeadDim / 2) + i] = static_cast<float>(std::sin(angle) * attention_factor);
        }
    }
}

void apply_rope_all(std::size_t seq_len, std::span<float> q, std::span<float> k, std::span<const float> cos, std::span<const float> sin) {
    for (std::size_t pos = 0; pos < seq_len; ++pos) {
        const auto c = cos.subspan(pos * (kHeadDim / 2), kHeadDim / 2);
        const auto s = sin.subspan(pos * (kHeadDim / 2), kHeadDim / 2);
        auto q_row = q.subspan(pos * kQDim, kQDim);
        auto k_row = k.subspan(pos * kKvDim, kKvDim);
        for (std::size_t head = 0; head < kQHeads; ++head) {
            h40::apply_rope_to_head(q_row.subspan(head * kHeadDim, kHeadDim), c, s);
        }
        for (std::size_t head = 0; head < kKvHeads; ++head) {
            h40::apply_rope_to_head(k_row.subspan(head * kHeadDim, kHeadDim), c, s);
        }
    }
}

void sequence_attention(
    std::size_t seq_len,
    bool sliding,
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    std::span<const float> sinks,
    std::span<float> merged) {
    if (q.size() != seq_len * kQDim || k.size() != seq_len * kKvDim || v.size() != seq_len * kKvDim ||
        merged.size() != seq_len * kQDim) {
        throw std::invalid_argument("sequence attention shape mismatch");
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    const std::size_t group = kQHeads / kKvHeads;
    constexpr std::size_t window = 128;
    for (std::size_t pos = 0; pos < seq_len; ++pos) {
        auto out_row = merged.subspan(pos * kQDim, kQDim);
        const std::size_t min_src = (!sliding || pos + 1 <= window) ? 0 : pos + 1 - window;
        for (std::size_t qh = 0; qh < kQHeads; ++qh) {
            const std::size_t kvh = qh / group;
            const auto qv = q.subspan(pos * kQDim + qh * kHeadDim, kHeadDim);
            std::vector<float> scores(pos - min_src + 1);
            float max_score = sinks[qh];
            for (std::size_t src = min_src; src <= pos; ++src) {
                const auto kv = k.subspan(src * kKvDim + kvh * kHeadDim, kHeadDim);
                float score = 0.0F;
                for (std::size_t i = 0; i < kHeadDim; ++i) score += qv[i] * kv[i];
                score *= scale;
                scores[src - min_src] = score;
                max_score = std::max(max_score, score);
            }
            double denom = std::exp(static_cast<double>(sinks[qh] - max_score));
            for (const float score : scores) denom += std::exp(static_cast<double>(score - max_score));
            auto out = out_row.subspan(qh * kHeadDim, kHeadDim);
            std::fill(out.begin(), out.end(), 0.0F);
            for (std::size_t src = min_src; src <= pos; ++src) {
                const double prob = std::exp(static_cast<double>(scores[src - min_src] - max_score)) / denom;
                const auto vv = v.subspan(src * kKvDim + kvh * kHeadDim, kHeadDim);
                for (std::size_t i = 0; i < kHeadDim; ++i) out[i] += static_cast<float>(prob * vv[i]);
            }
        }
    }
}

void write_json(const std::filesystem::path& path, const Metrics& metrics, std::uint64_t elapsed, std::size_t input_tokens) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"status\": \"pass\",\n";
    out << "  \"mode\": \"" << (input_tokens == 1 ? "single_token_full_24_layer_h40m_decode" : "multi_token_full_24_layer_h40m_prefill_decode") << "\",\n";
    out << "  \"input_tokens\": " << input_tokens << ",\n";
    out << "  \"layers_run\": " << metrics.layers_run << ",\n";
    out << "  \"emitted_token_id\": " << metrics.token_id << ",\n";
    out << "  \"emitted_token_text\": null,\n";
    out << "  \"emitted_token_logit\": " << metrics.token_logit << ",\n";
    out << "  \"dense_flash_bytes\": " << metrics.dense_bytes << ",\n";
    out << "  \"expert_flash_bytes\": " << metrics.expert_flash_bytes << ",\n";
    out << "  \"cache_hits\": " << metrics.expert_cache_hits << ",\n";
    out << "  \"cache_misses\": " << metrics.expert_cache_misses << ",\n";
    out << "  \"cache_policy\": \"" << metrics.cache_policy << "\",\n";
    out << "  \"peak_rss_kib\": " << metrics.peak_rss_kib << ",\n";
    out << "  \"io_overlap_enabled\": " << (metrics.io_overlap_enabled ? "true" : "false") << ",\n";
    out << "  \"prefetched_experts\": " << metrics.prefetched_experts << ",\n";
    out << "  \"prefetch_read_ns\": " << metrics.prefetch_read_ns << ",\n";
    out << "  \"prefetch_wait_ns\": " << metrics.prefetch_wait_ns << ",\n";
    out << "  \"embedding_ns\": " << metrics.embedding_ns << ",\n";
    out << "  \"dense_matvec_ns\": " << metrics.dense_matvec_ns << ",\n";
    out << "  \"attention_ns\": " << metrics.attention_ns << ",\n";
    out << "  \"moe_ns\": " << metrics.moe_ns << ",\n";
    out << "  \"lm_head_ns\": " << metrics.lm_head_ns << ",\n";
    out << "  \"dense_threads\": " << metrics.dense_threads << ",\n";
    out << "  \"lm_head_chunk_rows\": " << kLmHeadChunkRows << ",\n";
    out << "  \"elapsed_ms\": " << elapsed << "\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: minimal_decoder_probe <source_dir> <catalog.tsv> <expert_arena.bin> <token_id> <out.json> [trace.jsonl]\n";
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    const std::filesystem::path source_dir = argv[1];
    const std::filesystem::path catalog_path = argv[2];
    const std::filesystem::path expert_arena = argv[3];
    const auto input_tokens = parse_tokens(argv[4]);
    const std::size_t seq_len = input_tokens.size();
    const std::filesystem::path out_json = argv[5];
    std::ofstream trace_file;
    std::unique_ptr<h40::JsonlTraceWriter> trace_owner;
    h40::JsonlTraceWriter* trace = nullptr;
    if (argc == 7) {
        trace_file.open(argv[6]);
        if (!trace_file) throw std::runtime_error("failed to open trace output");
        trace_owner = std::make_unique<h40::JsonlTraceWriter>(trace_file);
        trace = trace_owner.get();
    }

    Metrics metrics;
    const auto catalog = h40::H40mTensorCatalog::load_tsv(catalog_path);
    h40::FileTensorReader reader(source_dir);
    std::size_t dense_threads = kDefaultDenseThreads;
    if (const char* setting = std::getenv("H40_THREADS")) {
        dense_threads = static_cast<std::size_t>(std::stoul(setting));
    }
    if (dense_threads == 0 || dense_threads > kMaxDenseThreads) {
        throw std::invalid_argument("H40_THREADS must be in [1, 8]");
    }
    h40::ParallelBf16Matvec dense_executor(reader, kMaxDenseThreads, std::max(kQDim, kHidden));
    metrics.dense_threads = dense_threads;
    h40::FlashTensorProvider expert_provider(expert_arena);
    const auto model_index = build_expert_index();
    h40::ExpertLoader loader(model_index, expert_provider);
    h40::CachePolicy cache_policy = h40::CachePolicy::per_layer_hotset;
    if (const char* setting = std::getenv("H40_CACHE_POLICY")) {
        const std::string_view name(setting);
        if (name == "lfu_decay") {
            cache_policy = h40::CachePolicy::lfu_decay;
        } else if (name == "per_layer_hotset") {
            cache_policy = h40::CachePolicy::per_layer_hotset;
        } else if (name != "lru") {
            throw std::invalid_argument("H40_CACHE_POLICY must be lru, lfu_decay, or per_layer_hotset");
        }
    }
    metrics.cache_policy = h40::cache_policy_name(cache_policy);
    h40::ExpertCache cache(kExpertPayloadBytes * kTopK, kExpertPayloadBytes, 1048576, cache_policy);
    const char* overlap_setting = std::getenv("H40_IO_OVERLAP");
    const bool io_overlap_enabled = overlap_setting == nullptr || std::string_view(overlap_setting) != "0";
    std::vector<std::byte> prefetch_storage;
    std::unique_ptr<h40::ExpertReadPipeline> read_pipeline;
    if (io_overlap_enabled) {
        prefetch_storage.resize(kExpertPayloadBytes);
        read_pipeline = std::make_unique<h40::ExpertReadPipeline>(loader, prefetch_storage);
    }
    metrics.io_overlap_enabled = io_overlap_enabled;

    std::vector<float> hidden(seq_len * kHidden);
    std::vector<float> normed(seq_len * kHidden);
    std::vector<float> norm_weight(kHidden);
    std::vector<float> q(seq_len * kQDim);
    std::vector<float> k(seq_len * kKvDim);
    std::vector<float> v(seq_len * kKvDim);
    std::vector<float> q_bias(kQDim);
    std::vector<float> k_bias(kKvDim);
    std::vector<float> v_bias(kKvDim);
    std::vector<float> o_bias(kHidden);
    std::vector<float> sinks(kQHeads);
    std::vector<float> merged(seq_len * kQDim);
    std::vector<float> attn_out(seq_len * kHidden);
    std::vector<float> router_logits(seq_len * kExperts);
    std::vector<float> router_bias(kExperts);
    std::vector<float> moe_out(seq_len * kHidden);
    std::vector<float> expert_out(kHidden);
    std::vector<float> gate_up(kIntermediate * 2);
    std::vector<float> expert_hidden(kIntermediate);
    std::vector<std::uint32_t> expert_ids(kTopK);
    std::vector<float> expert_weights(kTopK);

    std::vector<float> rope_cos(seq_len * (kHeadDim / 2));
    std::vector<float> rope_sin(seq_len * (kHeadDim / 2));
    yarn_rope_tables(seq_len, rope_cos, rope_sin);

    const auto embedding = must_find(catalog, "model.embed_tokens.weight");
    const auto embedding_start = std::chrono::steady_clock::now();
    for (std::size_t pos = 0; pos < seq_len; ++pos) {
        bf16_row_counted(reader, embedding, input_tokens[pos], std::span<float>(hidden).subspan(pos * kHidden, kHidden), metrics);
    }
    metrics.embedding_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - embedding_start)
            .count());

    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        const auto prefix = std::string("model.layers.") + std::to_string(layer);
        if (trace) {
            h40::TraceEvent row;
            row.event = "decoder_layer_begin";
            row.layer = layer;
            row.has_layer = true;
            trace->emit(row);
        }
        bf16_vector_counted(reader, must_find(catalog, prefix + ".input_layernorm.weight"), norm_weight, metrics);
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            h40::rms_norm(
                std::span<const float>(hidden).subspan(pos * kHidden, kHidden),
                norm_weight,
                1.0e-5F,
                std::span<float>(normed).subspan(pos * kHidden, kHidden));
        }

        const auto q_weight = must_find(catalog, prefix + ".self_attn.q_proj.weight");
        const auto k_weight = must_find(catalog, prefix + ".self_attn.k_proj.weight");
        const auto v_weight = must_find(catalog, prefix + ".self_attn.v_proj.weight");
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            const auto row = std::span<const float>(normed).subspan(pos * kHidden, kHidden);
            bf16_matvec_counted(dense_executor, q_weight, row, std::span<float>(q).subspan(pos * kQDim, kQDim), dense_threads, metrics);
            bf16_matvec_counted(dense_executor, k_weight, row, std::span<float>(k).subspan(pos * kKvDim, kKvDim), dense_threads, metrics);
            bf16_matvec_counted(dense_executor, v_weight, row, std::span<float>(v).subspan(pos * kKvDim, kKvDim), dense_threads, metrics);
        }
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.q_proj.bias"), q_bias, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.k_proj.bias"), k_bias, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.v_proj.bias"), v_bias, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.o_proj.bias"), o_bias, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.sinks"), sinks, metrics);
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            add_bias(std::span<float>(q).subspan(pos * kQDim, kQDim), q_bias);
            add_bias(std::span<float>(k).subspan(pos * kKvDim, kKvDim), k_bias);
            add_bias(std::span<float>(v).subspan(pos * kKvDim, kKvDim), v_bias);
        }
        const auto attention_start = std::chrono::steady_clock::now();
        apply_rope_all(seq_len, q, k, rope_cos, rope_sin);
        sequence_attention(seq_len, layer % 2 == 0, q, k, v, sinks, merged);
        metrics.attention_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - attention_start)
                .count());
        const auto o_weight = must_find(catalog, prefix + ".self_attn.o_proj.weight");
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            bf16_matvec_counted(
                dense_executor,
                o_weight,
                std::span<const float>(merged).subspan(pos * kQDim, kQDim),
                std::span<float>(attn_out).subspan(pos * kHidden, kHidden),
                dense_threads,
                metrics);
            add_bias(std::span<float>(attn_out).subspan(pos * kHidden, kHidden), o_bias);
        }
        add_inplace(hidden, attn_out);
        if (trace) {
            h40::TraceEvent row;
            row.event = "attention_end";
            row.layer = layer;
            row.has_layer = true;
            row.bytes = q.size() * sizeof(float) + k.size() * sizeof(float) + v.size() * sizeof(float);
            row.has_bytes = true;
            trace->emit(row);
        }

        bf16_vector_counted(reader, must_find(catalog, prefix + ".post_attention_layernorm.weight"), norm_weight, metrics);
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            h40::rms_norm(
                std::span<const float>(hidden).subspan(pos * kHidden, kHidden),
                norm_weight,
                1.0e-5F,
                std::span<float>(normed).subspan(pos * kHidden, kHidden));
        }
        const auto router_weight = must_find(catalog, prefix + ".mlp.router.weight");
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            bf16_matvec_counted(
                dense_executor,
                router_weight,
                std::span<const float>(normed).subspan(pos * kHidden, kHidden),
                std::span<float>(router_logits).subspan(pos * kExperts, kExperts),
                dense_threads,
                metrics);
        }
        bf16_vector_counted(reader, must_find(catalog, prefix + ".mlp.router.bias"), router_bias, metrics);
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            add_bias(std::span<float>(router_logits).subspan(pos * kExperts, kExperts), router_bias);
        }

        const auto moe_start = std::chrono::steady_clock::now();
        h40::run_moe_layer_streaming(
            {layer, seq_len, kExperts, kTopK, kHidden},
            router_logits,
            cache,
            loader,
            moe_out,
            {expert_ids, expert_weights, expert_out},
            [&](std::size_t token, std::uint32_t, std::span<const std::byte> packed, std::span<float> out) {
                h40::run_gptoss_expert(
                    expert_view(packed),
                    std::span<const float>(normed).subspan(token * kHidden, kHidden),
                    out,
                    {gate_up, expert_hidden});
            },
            trace,
            false,
            read_pipeline.get());
        metrics.moe_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - moe_start)
                .count());
        add_inplace(hidden, moe_out);
        ++metrics.layers_run;
        if (trace) {
            h40::TraceEvent row;
            row.event = "decoder_layer_end";
            row.layer = layer;
            row.has_layer = true;
            trace->emit(row);
        }
    }

    bf16_vector_counted(reader, must_find(catalog, "model.norm.weight"), norm_weight, metrics);
    const auto last_hidden = std::span<const float>(hidden).subspan((seq_len - 1) * kHidden, kHidden);
    auto last_normed = std::span<float>(normed).subspan((seq_len - 1) * kHidden, kHidden);
    h40::rms_norm(last_hidden, norm_weight, 1.0e-5F, last_normed);

    const auto lm_head = must_find(catalog, "lm_head.weight");
    std::vector<float> logits(kLmHeadChunkRows);
    const auto lm_head_start = std::chrono::steady_clock::now();
    for (std::size_t row = 0; row < kVocab; row += kLmHeadChunkRows) {
        const auto rows = std::min(kLmHeadChunkRows, kVocab - row);
        auto chunk = std::span<float>(logits).first(rows);
        dense_executor.matvec_rows(lm_head, row, last_normed, chunk, dense_threads);
        metrics.dense_bytes += rows * kHidden * sizeof(std::uint16_t);
        for (std::size_t i = 0; i < rows; ++i) {
            const float value = chunk[i];
            const auto id = static_cast<std::uint32_t>(row + i);
            if (value > metrics.token_logit || (value == metrics.token_logit && id < metrics.token_id)) {
                metrics.token_logit = value;
                metrics.token_id = id;
            }
        }
    }
    metrics.lm_head_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - lm_head_start)
            .count());

    const auto stats = cache.stats();
    metrics.expert_cache_hits = stats.hits;
    metrics.expert_cache_misses = stats.misses;
    metrics.expert_flash_bytes = stats.bytes_loaded;
    if (read_pipeline) {
        const auto prefetch_stats = read_pipeline->stats();
        metrics.prefetched_experts = prefetch_stats.completed;
        metrics.prefetch_read_ns = prefetch_stats.read_nanoseconds;
        metrics.prefetch_wait_ns = prefetch_stats.wait_nanoseconds;
    }
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        metrics.peak_rss_kib = static_cast<std::uint64_t>(usage.ru_maxrss);
    }
    write_json(out_json, metrics, elapsed_ms(start), seq_len);
    if (trace) {
        h40::TraceEvent row;
        row.event = "streamed_lm_head_argmax";
        row.token = metrics.token_id;
        row.has_token = true;
        row.bytes = kVocab * kHidden * sizeof(std::uint16_t);
        row.has_bytes = true;
        trace->emit(row);
    }
    std::cout << "emitted_token_id=" << metrics.token_id << "\n";
    std::cout << "emitted_token_logit=" << metrics.token_logit << "\n";
    return 0;
}

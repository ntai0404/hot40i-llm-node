#include "h40/attention.hpp"
#include "h40/expert_cache.hpp"
#include "h40/expert_loader.hpp"
#include "h40/flash_tensor_provider.hpp"
#include "h40/gptoss_expert.hpp"
#include "h40/h40m_tensor_catalog.hpp"
#include "h40/model_index.hpp"
#include "h40/moe_scheduler.hpp"
#include "h40/trace.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
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
constexpr std::size_t kLmHeadChunkRows = 256;

struct Metrics {
    std::uint64_t dense_bytes{};
    std::uint64_t expert_flash_bytes{};
    std::uint64_t expert_cache_hits{};
    std::uint64_t expert_cache_misses{};
    std::uint64_t layers_run{};
    std::uint32_t token_id{};
    float token_logit{-std::numeric_limits<float>::infinity()};
};

std::uint64_t elapsed_ms(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
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
    const h40::FileTensorReader& reader,
    const h40::H40mTensorRecord& record,
    std::span<const float> input,
    std::span<float> output,
    std::span<std::uint16_t> row_buffer,
    Metrics& metrics) {
    reader.bf16_matvec(record, input, output, row_buffer);
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

void single_token_attention(
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    std::span<const float> sinks,
    std::span<float> merged) {
    if (q.size() != kQDim || k.size() != kKvDim || v.size() != kKvDim || sinks.size() != kQHeads || merged.size() != kQDim) {
        throw std::invalid_argument("single-token attention shape mismatch");
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    const std::size_t group = kQHeads / kKvHeads;
    for (std::size_t qh = 0; qh < kQHeads; ++qh) {
        const std::size_t kvh = qh / group;
        const auto qv = q.subspan(qh * kHeadDim, kHeadDim);
        const auto kv = k.subspan(kvh * kHeadDim, kHeadDim);
        float score = 0.0F;
        for (std::size_t i = 0; i < kHeadDim; ++i) score += qv[i] * kv[i];
        score *= scale;
        const float max_score = std::max(score, sinks[qh]);
        const double denom =
            std::exp(static_cast<double>(score - max_score)) + std::exp(static_cast<double>(sinks[qh] - max_score));
        const float prob = static_cast<float>(std::exp(static_cast<double>(score - max_score)) / denom);
        const auto vv = v.subspan(kvh * kHeadDim, kHeadDim);
        auto out = merged.subspan(qh * kHeadDim, kHeadDim);
        for (std::size_t i = 0; i < kHeadDim; ++i) out[i] = prob * vv[i];
    }
}

void write_json(const std::filesystem::path& path, const Metrics& metrics, std::uint64_t elapsed) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"status\": \"pass\",\n";
    out << "  \"mode\": \"single_token_full_24_layer_h40m_decode\",\n";
    out << "  \"layers_run\": " << metrics.layers_run << ",\n";
    out << "  \"emitted_token_id\": " << metrics.token_id << ",\n";
    out << "  \"emitted_token_text\": null,\n";
    out << "  \"emitted_token_logit\": " << metrics.token_logit << ",\n";
    out << "  \"dense_flash_bytes\": " << metrics.dense_bytes << ",\n";
    out << "  \"expert_flash_bytes\": " << metrics.expert_flash_bytes << ",\n";
    out << "  \"cache_hits\": " << metrics.expert_cache_hits << ",\n";
    out << "  \"cache_misses\": " << metrics.expert_cache_misses << ",\n";
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
    const auto input_token = static_cast<std::uint32_t>(std::stoul(argv[4]));
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
    h40::FlashTensorProvider expert_provider(expert_arena);
    const auto model_index = build_expert_index();
    h40::ExpertLoader loader(model_index, expert_provider);
    h40::ExpertCache cache(kExpertPayloadBytes * kTopK, kExpertPayloadBytes, 1048576);

    std::vector<float> hidden(kHidden);
    std::vector<float> normed(kHidden);
    std::vector<float> norm_weight(kHidden);
    std::vector<float> q(kQDim);
    std::vector<float> k(kKvDim);
    std::vector<float> v(kKvDim);
    std::vector<float> q_bias(kQDim);
    std::vector<float> k_bias(kKvDim);
    std::vector<float> v_bias(kKvDim);
    std::vector<float> o_bias(kHidden);
    std::vector<float> sinks(kQHeads);
    std::vector<float> merged(kQDim);
    std::vector<float> attn_out(kHidden);
    std::vector<float> router_logits(kExperts);
    std::vector<float> router_bias(kExperts);
    std::vector<float> moe_out(kHidden);
    std::vector<float> expert_out(kHidden);
    std::vector<float> gate_up(kIntermediate * 2);
    std::vector<float> expert_hidden(kIntermediate);
    std::vector<std::uint32_t> expert_ids(kTopK);
    std::vector<float> expert_weights(kTopK);
    std::vector<std::uint16_t> row_buffer(std::max(kQDim, kHidden));

    bf16_row_counted(reader, must_find(catalog, "model.embed_tokens.weight"), input_token, hidden, metrics);

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
        h40::rms_norm(hidden, norm_weight, 1.0e-5F, normed);

        bf16_matvec_counted(reader, must_find(catalog, prefix + ".self_attn.q_proj.weight"), normed, q, row_buffer, metrics);
        bf16_matvec_counted(reader, must_find(catalog, prefix + ".self_attn.k_proj.weight"), normed, k, row_buffer, metrics);
        bf16_matvec_counted(reader, must_find(catalog, prefix + ".self_attn.v_proj.weight"), normed, v, row_buffer, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.q_proj.bias"), q_bias, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.k_proj.bias"), k_bias, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.v_proj.bias"), v_bias, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.o_proj.bias"), o_bias, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.sinks"), sinks, metrics);
        add_bias(q, q_bias);
        add_bias(k, k_bias);
        add_bias(v, v_bias);
        single_token_attention(q, k, v, sinks, merged);
        bf16_matvec_counted(reader, must_find(catalog, prefix + ".self_attn.o_proj.weight"), merged, attn_out, row_buffer, metrics);
        add_bias(attn_out, o_bias);
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
        h40::rms_norm(hidden, norm_weight, 1.0e-5F, normed);
        bf16_matvec_counted(reader, must_find(catalog, prefix + ".mlp.router.weight"), normed, router_logits, row_buffer, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".mlp.router.bias"), router_bias, metrics);
        add_bias(router_logits, router_bias);

        h40::run_moe_layer_streaming(
            {layer, 1, kExperts, kTopK, kHidden},
            router_logits,
            cache,
            loader,
            moe_out,
            {expert_ids, expert_weights, expert_out},
            [&](std::size_t, std::uint32_t, std::span<const std::byte> packed, std::span<float> out) {
                h40::run_gptoss_expert(expert_view(packed), normed, out, {gate_up, expert_hidden});
            },
            trace);
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
    h40::rms_norm(hidden, norm_weight, 1.0e-5F, normed);

    const auto lm_head = must_find(catalog, "lm_head.weight");
    std::vector<float> logits(kLmHeadChunkRows);
    for (std::size_t row = 0; row < kVocab; row += kLmHeadChunkRows) {
        const auto rows = std::min(kLmHeadChunkRows, kVocab - row);
        auto chunk = std::span<float>(logits).first(rows);
        reader.bf16_matvec_rows(lm_head, row, normed, chunk, row_buffer);
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

    const auto stats = cache.stats();
    metrics.expert_cache_hits = stats.hits;
    metrics.expert_cache_misses = stats.misses;
    metrics.expert_flash_bytes = stats.bytes_loaded;
    write_json(out_json, metrics, elapsed_ms(start));
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

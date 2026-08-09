#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def flatten(value):
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten(item))
        return out
    return [value]


def float_array(name: str, values) -> str:
    formatted = ", ".join(f"{float_literal(value)}" for value in values)
    return f"const std::vector<float> {name}{{\n    {formatted}\n}};\n\n"


def float_literal(value) -> str:
    text = f"{float(value):.10g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "F"


def uint_array(name: str, values) -> str:
    formatted = ", ".join(str(int(value)) + "U" for value in values)
    return f"const std::vector<std::uint32_t> {name}{{\n    {formatted}\n}};\n\n"


def main() -> None:
    fixture = json.loads((ROOT / "tests/fixtures/tiny_gpt_oss/fixture.json").read_text(encoding="utf-8"))
    cfg = fixture["config"]
    weights = fixture["weights"]
    golden = fixture["golden"]
    experts = weights["experts"]

    chunks = [
        '#include "h40/attention.hpp"\n'
        '#include "h40/router.hpp"\n\n'
        "#include <algorithm>\n"
        "#include <cassert>\n"
        "#include <cmath>\n"
        "#include <cstdint>\n"
        "#include <filesystem>\n"
        "#include <fstream>\n"
        "#include <span>\n"
        "#include <string>\n"
        "#include <vector>\n\n"
        "namespace {\n\n",
        f"constexpr std::size_t kSeqLen = {cfg['seq_len']};\n",
        f"constexpr std::size_t kModelDim = {cfg['d_model']};\n",
        f"constexpr std::size_t kHeads = {cfg['n_heads']};\n",
        f"constexpr std::size_t kHeadDim = {cfg['head_dim']};\n",
        f"constexpr std::size_t kExperts = {cfg['n_experts']};\n",
        f"constexpr std::size_t kTopK = {cfg['top_k']};\n",
        f"constexpr std::size_t kExpertHidden = {cfg['expert_hidden']};\n",
        f"constexpr std::size_t kVocab = {cfg['vocab_size']};\n",
        f"constexpr float kRmsEps = {float_literal(cfg['rms_eps'])};\n",
        f"constexpr float kSwigluLimit = {float_literal(cfg['swiglu_limit'])};\n\n",
        float_array("kInput", flatten(golden["input_embeddings"])),
        float_array("kAttnNormWeight", flatten(weights["attn_norm_weight"])),
        float_array("kFfnNormWeight", flatten(weights["ffn_norm_weight"])),
        float_array("kFinalNormWeight", flatten(weights["final_norm_weight"])),
        float_array("kWq", flatten(weights["wq"])),
        float_array("kWk", flatten(weights["wk"])),
        float_array("kWv", flatten(weights["wv"])),
        float_array("kWo", flatten(weights["wo"])),
        float_array("kRouterW", flatten(weights["router_w"])),
        float_array("kLmHead", flatten(weights["lm_head"])),
    ]

    for idx, expert in enumerate(experts):
        chunks.append(float_array(f"kExpert{idx}Gate", flatten(expert["w_gate"])))
        chunks.append(float_array(f"kExpert{idx}Up", flatten(expert["w_up"])))
        chunks.append(float_array(f"kExpert{idx}Down", flatten(expert["w_down"])))

    chunks.extend(
        [
            float_array("kExpectedAttention", flatten(golden["attention_output"])),
            float_array("kExpectedFfnNorm", flatten(golden["ffn_norm"])),
            float_array("kExpectedRouterLogits", flatten(golden["router_logits"])),
            uint_array("kExpectedRouterIds", flatten(golden["router_ids"])),
            float_array("kExpectedRouterScores", flatten(golden["router_scores"])),
            float_array("kExpectedMoe", flatten(golden["moe_output"])),
            float_array("kExpectedLayer", flatten(golden["layer_output"])),
            float_array("kExpectedFinalNorm", flatten(golden["final_norm"])),
            float_array("kExpectedLogits", flatten(golden["logits"])),
            """std::vector<float> matmul_rows(std::span<const float> rows, std::size_t row_count, std::size_t in_dim, std::span<const float> weights, std::size_t out_dim) {
    assert(rows.size() == row_count * in_dim);
    assert(weights.size() == in_dim * out_dim);
    std::vector<float> out(row_count * out_dim);
    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t col = 0; col < out_dim; ++col) {
            float acc = 0.0F;
            for (std::size_t i = 0; i < in_dim; ++i) {
                acc += rows[row * in_dim + i] * weights[i * out_dim + col];
            }
            out[row * out_dim + col] = acc;
        }
    }
    return out;
}

float silu(float x) {
    return x / (1.0F + std::exp(-x));
}

float clamp(float value, float limit) {
    return std::max(-limit, std::min(limit, value));
}

float max_abs_diff(std::span<const float> actual, std::span<const float> expected) {
    assert(actual.size() == expected.size());
    float max_diff = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(actual[i] - expected[i]));
    }
    return max_diff;
}

void assert_close(std::span<const float> actual, std::span<const float> expected, float tolerance) {
    assert(max_abs_diff(actual, expected) <= tolerance);
}

std::span<const float> expert_gate(std::size_t id) {
    switch (id) {
        case 0: return kExpert0Gate;
        case 1: return kExpert1Gate;
        case 2: return kExpert2Gate;
        default: return kExpert3Gate;
    }
}

std::span<const float> expert_up(std::size_t id) {
    switch (id) {
        case 0: return kExpert0Up;
        case 1: return kExpert1Up;
        case 2: return kExpert2Up;
        default: return kExpert3Up;
    }
}

std::span<const float> expert_down(std::size_t id) {
    switch (id) {
        case 0: return kExpert0Down;
        case 1: return kExpert1Down;
        case 2: return kExpert2Down;
        default: return kExpert3Down;
    }
}

std::vector<float> run_tiny_network() {
    std::vector<float> attn_norm(kInput.size());
    for (std::size_t pos = 0; pos < kSeqLen; ++pos) {
        h40::rms_norm(
            std::span<const float>(kInput).subspan(pos * kModelDim, kModelDim),
            kAttnNormWeight,
            kRmsEps,
            std::span<float>(attn_norm).subspan(pos * kModelDim, kModelDim)
        );
    }

    std::vector<float> attn(kInput.size());
    h40::causal_attention_projection({kSeqLen, kModelDim, kHeads, kHeads, kHeadDim}, attn_norm, kWq, kWk, kWv, kWo, {}, attn);
    assert_close(attn, kExpectedAttention, 1.0e-5F);

    std::vector<float> post_attention(kInput.size());
    for (std::size_t i = 0; i < post_attention.size(); ++i) {
        post_attention[i] = kInput[i] + attn[i];
    }

    std::vector<float> ffn_norm(kInput.size());
    for (std::size_t pos = 0; pos < kSeqLen; ++pos) {
        h40::rms_norm(
            std::span<const float>(post_attention).subspan(pos * kModelDim, kModelDim),
            kFfnNormWeight,
            kRmsEps,
            std::span<float>(ffn_norm).subspan(pos * kModelDim, kModelDim)
        );
    }
    assert_close(ffn_norm, kExpectedFfnNorm, 1.0e-5F);

    const auto router_logits = matmul_rows(ffn_norm, kSeqLen, kModelDim, kRouterW, kExperts);
    assert_close(router_logits, kExpectedRouterLogits, 1.0e-5F);

    std::vector<float> moe(kInput.size(), 0.0F);
    for (std::size_t pos = 0; pos < kSeqLen; ++pos) {
        const auto selected = h40::select_top_k_experts(
            std::span<const float>(router_logits).subspan(pos * kExperts, kExperts),
            kTopK
        );
        for (std::size_t i = 0; i < kTopK; ++i) {
            assert(selected.expert_ids[i] == kExpectedRouterIds[pos * kTopK + i]);
            assert(std::fabs(selected.weights[i] - kExpectedRouterScores[pos * kTopK + i]) <= 1.0e-6F);
        }

        const auto row = std::span<const float>(ffn_norm).subspan(pos * kModelDim, kModelDim);
        for (std::size_t choice = 0; choice < kTopK; ++choice) {
            const std::size_t expert = selected.expert_ids[choice];
            auto gate = matmul_rows(row, 1, kModelDim, expert_gate(expert), kExpertHidden);
            auto up = matmul_rows(row, 1, kModelDim, expert_up(expert), kExpertHidden);
            std::vector<float> hidden(kExpertHidden);
            for (std::size_t i = 0; i < kExpertHidden; ++i) {
                hidden[i] = silu(clamp(gate[i], kSwigluLimit)) * clamp(up[i], kSwigluLimit);
            }
            auto down = matmul_rows(hidden, 1, kExpertHidden, expert_down(expert), kModelDim);
            for (std::size_t i = 0; i < kModelDim; ++i) {
                moe[pos * kModelDim + i] += selected.weights[choice] * down[i];
            }
        }
    }
    assert_close(moe, kExpectedMoe, 1.0e-5F);

    std::vector<float> layer(kInput.size());
    for (std::size_t i = 0; i < layer.size(); ++i) {
        layer[i] = post_attention[i] + moe[i];
    }
    assert_close(layer, kExpectedLayer, 1.0e-5F);

    std::vector<float> final_norm(kInput.size());
    for (std::size_t pos = 0; pos < kSeqLen; ++pos) {
        h40::rms_norm(
            std::span<const float>(layer).subspan(pos * kModelDim, kModelDim),
            kFinalNormWeight,
            kRmsEps,
            std::span<float>(final_norm).subspan(pos * kModelDim, kModelDim)
        );
    }
    assert_close(final_norm, kExpectedFinalNorm, 1.0e-5F);

    auto logits = matmul_rows(final_norm, kSeqLen, kModelDim, kLmHead, kVocab);
    assert_close(logits, kExpectedLogits, 1.0e-5F);
    return logits;
}

void write_report(const std::string& path, std::span<const float> logits) {
    const std::filesystem::path report_path(path);
    std::filesystem::create_directories(report_path.parent_path());
    std::ofstream out(report_path);
    out << "{\\n";
    out << "  \\"schema_version\\": 1,\\n";
    out << "  \\"status\\": \\"pass\\",\\n";
    out << "  \\"fixture\\": \\"tiny_gpt_oss_shape_v1\\",\\n";
    out << "  \\"seq_len\\": " << kSeqLen << ",\\n";
    out << "  \\"model_dim\\": " << kModelDim << ",\\n";
    out << "  \\"max_abs_logit_diff\\": " << max_abs_diff(logits, kExpectedLogits) << ",\\n";
    out << "  \\"tolerance\\": 1e-5\\n";
    out << "}\\n";
}

} // namespace

int main(int argc, char** argv) {
    const auto logits = run_tiny_network();
    if (argc > 1) {
        write_report(argv[1], logits);
    }
    return 0;
}
""",
        ]
    )
    (ROOT / "tests/tiny_parity_test.cpp").write_text("".join(chunks), encoding="utf-8")


if __name__ == "__main__":
    main()

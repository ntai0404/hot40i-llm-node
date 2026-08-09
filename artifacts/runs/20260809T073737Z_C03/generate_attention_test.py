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


def array(name: str, values) -> str:
    formatted = ", ".join(f"{float(value):.10g}F" for value in values)
    return f"const std::vector<float> {name}{{\n    {formatted}\n}};\n\n"


def main() -> None:
    fixture = json.loads((ROOT / "tests/fixtures/tiny_gpt_oss/fixture.json").read_text(encoding="utf-8"))
    chunks = [
        '#include "h40/attention.hpp"\n\n'
        "#include <array>\n"
        "#include <cassert>\n"
        "#include <cmath>\n"
        "#include <span>\n"
        "#include <stdexcept>\n"
        "#include <vector>\n\n"
        "namespace {\n\n",
        array("kInput", flatten(fixture["golden"]["input_embeddings"])),
        array("kNormWeight", flatten(fixture["weights"]["attn_norm_weight"])),
        array("kExpectedNorm", flatten(fixture["golden"]["attention_norm"])),
        array("kWq", flatten(fixture["weights"]["wq"])),
        array("kWk", flatten(fixture["weights"]["wk"])),
        array("kWv", flatten(fixture["weights"]["wv"])),
        array("kWo", flatten(fixture["weights"]["wo"])),
        array("kExpectedAttention", flatten(fixture["golden"]["attention_output"])),
        """void assert_close(float actual, float expected, float tolerance = 1.0e-5F) {
    assert(std::fabs(actual - expected) <= tolerance);
}

void fixture_rms_norm_matches() {
    constexpr std::size_t seq = 3;
    constexpr std::size_t dim = 8;
    std::vector<float> out(kInput.size());
    for (std::size_t pos = 0; pos < seq; ++pos) {
        h40::rms_norm(
            std::span<const float>(kInput).subspan(pos * dim, dim),
            kNormWeight,
            1.0e-5F,
            std::span<float>(out).subspan(pos * dim, dim)
        );
    }
    for (std::size_t i = 0; i < out.size(); ++i) {
        assert_close(out[i], kExpectedNorm[i]);
    }
}

void fixture_attention_matches() {
    h40::AttentionConfig cfg{3, 8, 2, 2, 4};
    std::vector<float> out(kExpectedAttention.size());
    h40::causal_attention_projection(cfg, kExpectedNorm, kWq, kWk, kWv, kWo, {}, out);
    for (std::size_t i = 0; i < out.size(); ++i) {
        assert_close(out[i], kExpectedAttention[i]);
    }
}

void rope_matches_official_half_split_rotation() {
    std::array<float, 4> head{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 2> cos{0.0F, 1.0F};
    const std::array<float, 2> sin{1.0F, 0.0F};
    h40::apply_rope_to_head(head, cos, sin);
    assert_close(head[0], -3.0F);
    assert_close(head[1], 2.0F);
    assert_close(head[2], 1.0F);
    assert_close(head[3], 4.0F);
}

void sinks_change_softmax_denominator_without_value_contribution() {
    h40::AttentionConfig cfg{1, 2, 1, 1, 2};
    const std::vector<float> x{1.0F, 0.0F};
    const std::vector<float> identity{1.0F, 0.0F, 0.0F, 1.0F};
    const std::array<float, 1> sinks{0.0F};
    std::array<float, 2> no_sink{};
    std::array<float, 2> with_sink{};
    h40::causal_attention_projection(cfg, x, identity, identity, identity, identity, {}, no_sink);
    h40::causal_attention_projection(cfg, x, identity, identity, identity, identity, sinks, with_sink);
    assert_close(no_sink[0], 1.0F);
    assert(with_sink[0] < no_sink[0]);
    assert_close(with_sink[1], 0.0F);
}

void invalid_gqa_shape_throws() {
    h40::AttentionConfig cfg{1, 4, 3, 2, 1};
    const std::vector<float> values(4, 0.0F);
    bool threw = false;
    try {
        h40::causal_attention_projection(cfg, values, values, values, values, values, {}, std::span<float>{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main() {
    fixture_rms_norm_matches();
    fixture_attention_matches();
    rope_matches_official_half_split_rotation();
    sinks_change_softmax_denominator_without_value_contribution();
    invalid_gqa_shape_throws();
    return 0;
}
""",
    ]
    (ROOT / "tests/attention_test.cpp").write_text("".join(chunks), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tools/model/generate_tiny_gpt_oss_fixture.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("tiny_gpt_oss_generator", GENERATOR)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tiny_gpt_oss_fixture_is_deterministic_and_complete():
    generator = load_generator()
    first = generator.build_fixture()
    second = generator.build_fixture()
    assert first == second

    golden = first["golden"]
    for key in (
        "router_ids",
        "router_scores",
        "expert_outputs_selected",
        "attention_output",
        "layer_output",
        "logits",
    ):
        assert key in golden

    cfg = first["config"]
    assert len(golden["router_ids"]) == cfg["seq_len"]
    assert all(len(ids) == cfg["top_k"] for ids in golden["router_ids"])
    assert len(golden["attention_output"]) == cfg["seq_len"]
    assert len(golden["layer_output"]) == cfg["seq_len"]
    assert len(golden["logits"][0]) == cfg["vocab_size"]

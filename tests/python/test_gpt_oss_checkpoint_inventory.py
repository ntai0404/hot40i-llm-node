import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_inventory_has_expected_gpt_oss_shape_when_present():
    inventory_path = ROOT / "artifacts/model/gpt_oss_20b_inventory.json"
    if not inventory_path.exists():
        return
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory["tensor_count"] == 459
    assert inventory["unknown_tensor_count"] == 0
    assert inventory["tensor_bytes_total"] == inventory["index_total_size"]
    assert inventory["config"]["num_hidden_layers"] == 24
    assert inventory["config"]["num_local_experts"] == 32
    assert inventory["config"]["num_experts_per_tok"] == 4
    assert inventory["totals_by_role"]["expert"] > inventory["totals_by_role"]["attention"]


def test_inventory_spot_checks_known_tensors_when_present():
    inventory_path = ROOT / "artifacts/model/gpt_oss_20b_inventory.json"
    if not inventory_path.exists():
        return
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    tensors = {item["name"]: item for item in inventory["tensors"]}
    embed = tensors["model.embed_tokens.weight"]
    assert embed["role"] == "embedding"
    assert embed["dtype"] == "BF16"
    assert embed["shape"] == [201088, 2880]

    router = tensors["model.layers.0.mlp.router.weight"]
    assert router["role"] == "router"
    assert router["layer"] == 0
    assert router["shape"] == [32, 2880]

    expert = tensors["model.layers.0.mlp.experts.gate_up_proj_blocks"]
    assert expert["role"] == "expert"
    assert expert["layer"] == 0
    assert expert["dtype"] == "U8"

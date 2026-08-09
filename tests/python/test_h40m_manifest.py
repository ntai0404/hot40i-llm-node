import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path("tools/model/h40m_manifest.py")
    spec = importlib.util.spec_from_file_location("h40m_manifest", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_validation_matches_schema() -> None:
    module = _module()
    module.validate(
        {
            "format": "H40M/1",
            "source": {
                "model": "openai/gpt-oss-20b",
                "checkpoint_sha256": "0" * 64,
                "upstream_ref": None,
                "converter_commit": None,
            },
            "files": [],
            "tensors": [],
            "extensions": {},
        }
    )


def test_manifest_rejects_old_incompatible_shape() -> None:
    module = _module()
    with pytest.raises(Exception):
        module.validate({"format": "H40M/1", "model": "x", "files": [], "tensors": []})


def test_converter_produces_deterministic_manifest(tmp_path: Path) -> None:
    module = _module()
    inventory = {
        "model": "openai/gpt-oss-20b",
        "repo_commit": "abc123",
        "tensor_bytes_total": 96,
        "inventory_method": "test",
        "tensors": [
            {
                "name": "model.layers.0.mlp.experts.gate_up_proj_blocks",
                "role": "expert",
                "layer": 0,
                "expert_id": None,
                "shape": [32, 2, 16],
                "dtype": "U8",
                "byte_size": 64,
                "source_shard": "model-00000-of-00002.safetensors",
                "source_shard_checksum": "a" * 64,
                "source_shard_size": 128,
            },
            {
                "name": "model.layers.0.self_attn.q_proj.weight",
                "role": "attention",
                "layer": 0,
                "expert_id": None,
                "shape": [8, 8],
                "dtype": "BF16",
                "byte_size": 32,
                "source_shard": "model-00000-of-00002.safetensors",
                "source_shard_checksum": "a" * 64,
                "source_shard_size": 128,
            },
        ],
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    first = module.convert(inventory_path, tmp_path / "manifest1.json", alignment=64)
    second = module.convert(inventory_path, tmp_path / "manifest2.json", alignment=64)
    assert (tmp_path / "manifest1.json").read_bytes() == (tmp_path / "manifest2.json").read_bytes()
    assert first == second
    assert first["files"][0]["size"] == 128
    assert first["tensors"][0]["offset"] == 0
    assert first["tensors"][1]["offset"] == 64
    assert first["tensors"][0]["placement"] == "cache"
    assert first["tensors"][0]["quant_type"] == "MXFP4_E2M1_PACKED"
    assert first["tensors"][1]["placement"] == "resident"
    module.validate(first)

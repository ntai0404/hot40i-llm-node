import importlib.util
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

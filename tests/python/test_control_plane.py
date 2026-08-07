import importlib.util
import json
from pathlib import Path

import pytest


def _taskctl():
    path = Path("scripts/taskctl.py")
    spec = importlib.util.spec_from_file_location("taskctl_for_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evidence(task_id: str, verification_name: str) -> dict:
    return {
        "schema_version": 1,
        "task_id": task_id,
        "started_at": "2026-08-07T00:00:00+00:00",
        "finished_at": "2026-08-07T00:01:00+00:00",
        "status": "pass",
        "git_commit": None,
        "sources": [],
        "commands": [
            {
                "argv": "python -m pytest -q",
                "exit_code": 0,
                "stdout_file": None,
                "stderr_file": None,
            }
        ],
        "files_changed": ["artifacts/runs/test/evidence.json"],
        "verification": [
            {"name": verification_name, "passed": True, "detail": "ok"}
        ],
        "metrics": {},
        "artifacts": [],
        "limitations": [],
        "notes": [],
    }


def test_task_specific_evidence_is_enforced(tmp_path, monkeypatch):
    module = _taskctl()
    root = tmp_path / "repo"
    run_dir = root / "artifacts/runs/test"
    run_dir.mkdir(parents=True)
    schema_dir = root / "schemas"
    schema_dir.mkdir(parents=True)
    schema_dir.joinpath("task_evidence.schema.json").write_text(
        Path("schemas/task_evidence.schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(
        module, "EVIDENCE_SCHEMA_PATH", schema_dir / "task_evidence.schema.json"
    )

    task = {
        "id": "T00",
        "verification": ["python -m pytest -q"],
        "required_artifacts": ["artifacts/runs/<run>/evidence.json"],
        "allowed_paths": ["artifacts/**"],
    }
    evidence_path = run_dir / "evidence.json"
    evidence_path.write_text(
        json.dumps(_evidence("T00", "python -m pytest -q")), encoding="utf-8"
    )
    module.validate_task_evidence(evidence_path, task)

    broken = _evidence("T00", "wrong verification")
    evidence_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing passed task verification"):
        module.validate_task_evidence(evidence_path, task)


def test_task_scope_is_enforced(tmp_path, monkeypatch):
    module = _taskctl()
    root = tmp_path / "repo"
    run_dir = root / "artifacts/runs/test"
    run_dir.mkdir(parents=True)
    schema_dir = root / "schemas"
    schema_dir.mkdir(parents=True)
    schema_dir.joinpath("task_evidence.schema.json").write_text(
        Path("schemas/task_evidence.schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(
        module, "EVIDENCE_SCHEMA_PATH", schema_dir / "task_evidence.schema.json"
    )

    task = {
        "id": "T00",
        "verification": ["verify"],
        "required_artifacts": ["artifacts/runs/<run>/evidence.json"],
        "allowed_paths": ["artifacts/**"],
    }
    document = _evidence("T00", "verify")
    document["files_changed"] = ["device/runtime/unsafe.cpp"]
    evidence_path = run_dir / "evidence.json"
    evidence_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SystemExit, match="outside task allowed_paths"):
        module.validate_task_evidence(evidence_path, task)

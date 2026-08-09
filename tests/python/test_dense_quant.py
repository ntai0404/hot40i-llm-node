import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_dense_quant_report_selects_accepted_candidate(tmp_path: Path) -> None:
    out = tmp_path / "dense_quant.json"
    policy = tmp_path / "dense.policy.json"
    subprocess.run(
        [
            sys.executable,
            "tools/model/evaluate_dense_quant.py",
            "--fixture",
            "tests/fixtures/tiny_gpt_oss/fixture.json",
            "--inventory",
            "artifacts/model/gpt_oss_20b_inventory.json",
            "--out",
            str(out),
            "--policy",
            str(policy),
        ],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["selected"]["accepted"] is True
    assert report["selected"]["routing_match"] is True
    assert report["selected"]["estimated_reduction_bytes"] > 0

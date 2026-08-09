import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_embedding_lookup_report_uses_token_lookup_manifest(tmp_path: Path) -> None:
    out = tmp_path / "embedding_lookup.json"
    subprocess.run(
        [
            sys.executable,
            "tools/model/embedding_lookup.py",
            "--manifest",
            "artifacts/model/h40m/manifest.json",
            "--header-dir",
            "artifacts/runs/20260809T074843Z_M00",
            "--sample-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--tokens",
            "0",
        ],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["embedding"]["placement"] == "token_lookup"
    assert report["embedding"]["row_bytes"] == 5760
    assert report["cache_policy"]["default_capacity_rows"] == 8

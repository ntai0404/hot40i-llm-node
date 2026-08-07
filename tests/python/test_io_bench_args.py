from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _bench_exe() -> Path:
    path = ROOT / "build" / "h40_io_bench.exe"
    if not path.exists():
        raise AssertionError("build/h40_io_bench.exe is required before running this test")
    return path


def test_io_bench_help_exits_zero() -> None:
    result = subprocess.run([str(_bench_exe()), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--file PATH" in result.stdout


def test_io_bench_rejects_missing_file_value() -> None:
    result = subprocess.run([str(_bench_exe()), "--file"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "missing value for --file" in result.stderr

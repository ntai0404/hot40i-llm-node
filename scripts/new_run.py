#!/usr/bin/env python3
"""Compatibility wrapper for creating a task evidence run."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: new_run.py TASK_ID")
    raise SystemExit(
        subprocess.call(
            [sys.executable, "scripts/evidence.py", "init", sys.argv[1]],
            cwd=ROOT,
        )
    )


if __name__ == "__main__":
    main()

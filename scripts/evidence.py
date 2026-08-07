#!/usr/bin/env python3
"""Create auditable per-task evidence bundles."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_run(run: str) -> tuple[Path, Path, dict[str, Any]]:
    run_dir = Path(run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    evidence_path = run_dir / "evidence.json"
    if not evidence_path.exists():
        raise SystemExit(f"missing {evidence_path}")
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    return run_dir, evidence_path, document


def _save(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def cmd_init(args: argparse.Namespace) -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "artifacts/runs" / f"{timestamp}_{args.task}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "stdout").mkdir()
    (run_dir / "stderr").mkdir()

    document: dict[str, Any] = {
        "schema_version": 1,
        "task_id": args.task,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "finished_at": "",
        "status": "fail",
        "git_commit": _git_commit(),
        "sources": [],
        "commands": [],
        "files_changed": [],
        "verification": [],
        "metrics": {},
        "artifacts": [],
        "limitations": [],
        "notes": [],
    }
    _save(run_dir / "evidence.json", document)
    print(run_dir.relative_to(ROOT))


def cmd_run(args: argparse.Namespace) -> None:
    run_dir, evidence_path, document = _load_run(args.run)
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise SystemExit("missing command after --")

    index = len(document["commands"]) + 1
    stdout_path = run_dir / "stdout" / f"{index:03d}.log"
    stderr_path = run_dir / "stderr" / f"{index:03d}.log"

    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
        return_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTIMEOUT after {args.timeout}s\n"

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    document["commands"].append(
        {
            "argv": shlex.join(command),
            "exit_code": return_code,
            "stdout_file": str(stdout_path.relative_to(ROOT)),
            "stderr_file": str(stderr_path.relative_to(ROOT)),
        }
    )
    _save(evidence_path, document)
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    raise SystemExit(return_code)


def cmd_verify(args: argparse.Namespace) -> None:
    _, evidence_path, document = _load_run(args.run)
    document["verification"].append(
        {"name": args.name, "passed": args.passed, "detail": args.detail}
    )
    _save(evidence_path, document)


def cmd_artifact(args: argparse.Namespace) -> None:
    _, evidence_path, document = _load_run(args.run)
    path = Path(args.path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(f"artifact does not exist: {path}")
    try:
        normalized = str(path.relative_to(ROOT))
    except ValueError:
        normalized = str(path)
    if normalized not in document["artifacts"]:
        document["artifacts"].append(normalized)
    _save(evidence_path, document)


def cmd_source(args: argparse.Namespace) -> None:
    _, evidence_path, document = _load_run(args.run)
    document["sources"].append(
        {"url": args.url, "ref": args.ref, "note": args.note}
    )
    _save(evidence_path, document)


def cmd_note(args: argparse.Namespace) -> None:
    _, evidence_path, document = _load_run(args.run)
    document["notes"].append(args.text)
    _save(evidence_path, document)


def cmd_limitation(args: argparse.Namespace) -> None:
    _, evidence_path, document = _load_run(args.run)
    document["limitations"].append(args.text)
    _save(evidence_path, document)


def cmd_metric(args: argparse.Namespace) -> None:
    _, evidence_path, document = _load_run(args.run)
    try:
        value: Any = json.loads(args.value)
    except Exception:
        value = args.value
    document["metrics"][args.key] = value
    _save(evidence_path, document)


def cmd_finish(args: argparse.Namespace) -> None:
    _, evidence_path, document = _load_run(args.run)
    document["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    document["status"] = args.status
    try:
        changed = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        document["files_changed"] = sorted(
            {line[3:] for line in changed if len(line) > 3}
        )
    except Exception:
        pass
    _save(evidence_path, document)
    print(evidence_path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Task evidence bundle helper")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    command = subparsers.add_parser("init")
    command.add_argument("task")
    command.set_defaults(fn=cmd_init)

    command = subparsers.add_parser("run")
    command.add_argument("run")
    command.add_argument("--timeout", type=float, default=None)
    command.add_argument("command", nargs=argparse.REMAINDER)
    command.set_defaults(fn=cmd_run)

    command = subparsers.add_parser("verify")
    command.add_argument("run")
    command.add_argument("--name", required=True)
    command.add_argument(
        "--passed", action=argparse.BooleanOptionalAction, default=True
    )
    command.add_argument("--detail", default="")
    command.set_defaults(fn=cmd_verify)

    command = subparsers.add_parser("artifact")
    command.add_argument("run")
    command.add_argument("path")
    command.set_defaults(fn=cmd_artifact)

    command = subparsers.add_parser("source")
    command.add_argument("run")
    command.add_argument("--url", required=True)
    command.add_argument("--ref", default="")
    command.add_argument("--note", default="")
    command.set_defaults(fn=cmd_source)

    command = subparsers.add_parser("note")
    command.add_argument("run")
    command.add_argument("text")
    command.set_defaults(fn=cmd_note)

    command = subparsers.add_parser("limitation")
    command.add_argument("run")
    command.add_argument("text")
    command.set_defaults(fn=cmd_limitation)

    command = subparsers.add_parser("metric")
    command.add_argument("run")
    command.add_argument("key")
    command.add_argument("value")
    command.set_defaults(fn=cmd_metric)

    command = subparsers.add_parser("finish")
    command.add_argument("run")
    command.add_argument("--status", choices=["pass", "blocked", "fail"], required=True)
    command.set_defaults(fn=cmd_finish)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

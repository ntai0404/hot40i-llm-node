#!/usr/bin/env python3
"""Machine-readable roadmap/task state controller.

This script is deliberately strict: a task can pass only after it was started,
its dependencies passed, its task-specific verification entries are recorded,
and every required artifact exists. The state file therefore remains an audit
index rather than a narrative progress note.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "roadmap/tasks.yaml"
GATES_PATH = ROOT / "roadmap/gates.yaml"
STATE_PATH = ROOT / "PROJECT_STATE.yaml"
EVIDENCE_SCHEMA_PATH = ROOT / "schemas/task_evidence.schema.json"
CONTROL_PLANE_SCOPE_EXEMPTIONS = {"PROJECT_STATE.yaml"}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(
        yaml.safe_dump(state, sort_keys=False, width=110),
        encoding="utf-8",
    )


def task_map() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    data = load_yaml(TASKS_PATH)
    return data, {task["id"]: task for task in data["tasks"]}


def ready_tasks(
    state: dict[str, Any],
    data: dict[str, Any],
    task_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    del task_by_id  # kept in signature for call-site clarity/future checks
    done = set(state.get("completed_tasks", []))
    blocked = set((state.get("blocked_tasks") or {}).keys())
    in_progress = state.get("in_progress")
    output: list[dict[str, Any]] = []

    for task in data["tasks"]:
        task_id = task["id"]
        if task_id in done or task_id in blocked or task_id == in_progress:
            continue
        if task.get("optional"):
            if task_id.startswith("OS") and not state.get("decisions", {}).get(
                "os_optimization_activated", False
            ):
                continue
        if all(dep in done for dep in task.get("depends_on", [])):
            output.append(task)

    return sorted(output, key=lambda item: (item.get("priority", 50), item["id"]))


def _validate_json_schema(document: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError:
        return
    schema = json.loads(EVIDENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(document, schema)


def _normalize_repo_path(path: str) -> str:
    return Path(path).as_posix().lstrip("./")


def _path_allowed(path: str, patterns: list[str]) -> bool:
    normalized = _normalize_repo_path(path)
    if normalized in CONTROL_PLANE_SCOPE_EXEMPTIONS:
        return True
    return any(fnmatch.fnmatchcase(normalized, pattern) for pattern in patterns)


def _resolve_required_artifact(spec: str, evidence_path: Path) -> Path:
    run_name = evidence_path.parent.name
    resolved = spec.replace("<run>", run_name)
    return ROOT / resolved


def validate_task_evidence(
    evidence_path: Path,
    task: dict[str, Any],
) -> dict[str, Any]:
    task_id = task["id"]
    if not evidence_path.exists():
        raise SystemExit(f"evidence file missing: {evidence_path}")

    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    _validate_json_schema(document)

    if document.get("task_id") != task_id:
        raise SystemExit(f"evidence task_id {document.get('task_id')} != {task_id}")
    if document.get("status") != "pass":
        raise SystemExit("evidence status must be pass")
    if not document.get("finished_at"):
        raise SystemExit("evidence finished_at must be populated")
    if not document.get("commands"):
        raise SystemExit("evidence must record at least one executed command")

    verification = document.get("verification", [])
    passed_names = {
        item.get("name")
        for item in verification
        if isinstance(item, dict) and item.get("passed") is True
    }
    missing_verification = [
        item for item in task.get("verification", []) if item not in passed_names
    ]
    if missing_verification:
        raise SystemExit(
            "evidence missing passed task verification entries: "
            + "; ".join(missing_verification)
        )

    evidence_artifacts = {
        _normalize_repo_path(item) for item in document.get("artifacts", [])
    }
    evidence_rel = _normalize_repo_path(str(evidence_path.relative_to(ROOT)))
    evidence_artifacts.add(evidence_rel)

    for spec in task.get("required_artifacts", []):
        required = _resolve_required_artifact(spec, evidence_path)
        if not required.exists():
            raise SystemExit(f"required task artifact does not exist: {required.relative_to(ROOT)}")
        required_rel = _normalize_repo_path(str(required.relative_to(ROOT)))
        if required_rel != evidence_rel and required_rel not in evidence_artifacts:
            raise SystemExit(
                f"required task artifact is not registered in evidence.artifacts: {required_rel}"
            )

    allowed_paths = task.get("allowed_paths", [])
    out_of_scope = [
        path
        for path in document.get("files_changed", [])
        if not _path_allowed(path, allowed_paths)
    ]
    if out_of_scope:
        raise SystemExit(
            "evidence reports files outside task allowed_paths: " + ", ".join(out_of_scope)
        )

    return document


def cmd_status(_: argparse.Namespace) -> None:
    state = load_yaml(STATE_PATH)
    data, task_by_id = task_map()
    ready = ready_tasks(state, data, task_by_id)
    print(f"goal_gate: {state.get('goal_gate')}")
    print(f"completed: {len(state.get('completed_tasks', []))}/{len(data['tasks'])}")
    print(f"in_progress: {state.get('in_progress')}")
    print(f"blocked: {', '.join((state.get('blocked_tasks') or {}).keys()) or '-'}")
    print("ready:", ", ".join(task["id"] for task in ready[:12]) or "-")


def cmd_next(_: argparse.Namespace) -> None:
    state = load_yaml(STATE_PATH)
    data, task_by_id = task_map()
    ready = ready_tasks(state, data, task_by_id)
    in_progress = state.get("in_progress")
    if in_progress:
        print(yaml.safe_dump(task_by_id[in_progress], sort_keys=False, width=110))
        return
    if not ready:
        print("NO_READY_TASK")
        if state.get("blocked_tasks"):
            print(
                "Blocked tasks:",
                json.dumps(state["blocked_tasks"], ensure_ascii=False, indent=2),
            )
        return
    print(yaml.safe_dump(ready[0], sort_keys=False, width=110))


def cmd_start(args: argparse.Namespace) -> None:
    state = load_yaml(STATE_PATH)
    data, task_by_id = task_map()
    task_id = args.task
    if task_id not in task_by_id:
        raise SystemExit(f"unknown task {task_id}")
    if state.get("in_progress") and state["in_progress"] != task_id:
        raise SystemExit(f"task {state['in_progress']} already in progress")

    ready_ids = {task["id"] for task in ready_tasks(state, data, task_by_id)}
    if task_id not in ready_ids and state.get("in_progress") != task_id:
        raise SystemExit(
            f"{task_id} is not ready; dependencies/optional activation/block state prevent start"
        )

    state["in_progress"] = task_id
    state.setdefault("task_started_at", {})[task_id] = dt.datetime.now(
        dt.timezone.utc
    ).isoformat()
    save_state(state)
    print(f"STARTED {task_id}")


def cmd_pass(args: argparse.Namespace) -> None:
    state = load_yaml(STATE_PATH)
    _, task_by_id = task_map()
    task_id = args.task
    if task_id not in task_by_id:
        raise SystemExit(f"unknown task {task_id}")
    if state.get("in_progress") != task_id:
        raise SystemExit(
            f"task {task_id} must be the current in_progress task before it can pass"
        )

    task = task_by_id[task_id]
    done = set(state.get("completed_tasks", []))
    missing_dependencies = [
        dep for dep in task.get("depends_on", []) if dep not in done
    ]
    if missing_dependencies:
        raise SystemExit(
            f"cannot pass {task_id}; missing deps {missing_dependencies}"
        )

    evidence_path = Path(args.evidence)
    if not evidence_path.is_absolute():
        evidence_path = ROOT / evidence_path
    validate_task_evidence(evidence_path, task)

    if task_id not in done:
        state.setdefault("completed_tasks", []).append(task_id)
    state.setdefault("task_evidence", {})[task_id] = _normalize_repo_path(
        str(evidence_path.relative_to(ROOT))
    )
    state["in_progress"] = None
    state.setdefault("blocked_tasks", {}).pop(task_id, None)
    save_state(state)
    print(f"PASSED {task_id}")


def cmd_block(args: argparse.Namespace) -> None:
    state = load_yaml(STATE_PATH)
    _, task_by_id = task_map()
    task_id = args.task
    if task_id not in task_by_id:
        raise SystemExit(f"unknown task {task_id}")
    state.setdefault("blocked_tasks", {})[task_id] = {
        "reason": args.reason,
        "at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if state.get("in_progress") == task_id:
        state["in_progress"] = None
    save_state(state)
    print(f"BLOCKED {task_id}")


def cmd_unblock(args: argparse.Namespace) -> None:
    state = load_yaml(STATE_PATH)
    state.setdefault("blocked_tasks", {}).pop(args.task, None)
    save_state(state)
    print(f"UNBLOCKED {args.task}")


def _gate_status(
    gate_id: str,
    state: dict[str, Any],
    task_by_id: dict[str, dict[str, Any]],
    gates: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    gate = next((item for item in gates["gates"] if item["id"] == gate_id), None)
    if gate is None:
        raise SystemExit(f"unknown gate {gate_id}")

    done = set(state.get("completed_tasks", []))
    missing_tasks = [task for task in gate.get("requires_tasks", []) if task not in done]
    evidence_map = state.get("task_evidence", {}) or {}
    missing_evidence: list[str] = []

    tasks_to_audit = list(gate.get("requires_tasks", []))
    if gate_id == "FINAL_DEPLOYMENT":
        tasks_to_audit = [
            task_id
            for task_id, task in task_by_id.items()
            if not task.get("optional")
        ]
        blocked = state.get("blocked_tasks", {}) or {}
        mandatory_blocked = [task_id for task_id in tasks_to_audit if task_id in blocked]
        missing_tasks.extend(task_id for task_id in mandatory_blocked if task_id not in missing_tasks)

    for task_id in tasks_to_audit:
        if task_id not in done:
            continue
        evidence = evidence_map.get(task_id)
        if not evidence or not (ROOT / evidence).exists():
            missing_evidence.append(task_id)

    return gate, sorted(set(missing_tasks)), sorted(set(missing_evidence))


def cmd_gate(args: argparse.Namespace) -> None:
    state = load_yaml(STATE_PATH)
    _, task_by_id = task_map()
    gates = load_yaml(GATES_PATH)
    gate, missing_tasks, missing_evidence = _gate_status(
        args.gate, state, task_by_id, gates
    )
    if missing_tasks or missing_evidence:
        parts: list[str] = []
        if missing_tasks:
            parts.append("missing tasks " + ", ".join(missing_tasks))
        if missing_evidence:
            parts.append("missing evidence " + ", ".join(missing_evidence))
        print(f"FAIL {args.gate}: " + "; ".join(parts))
        raise SystemExit(1)

    print(f"PASS {args.gate}")
    for criterion in gate.get("criteria", []):
        print(f" - {criterion}")


def cmd_decide(args: argparse.Namespace) -> None:
    state = load_yaml(STATE_PATH)
    value = args.value.lower()
    if value in ("true", "yes", "1"):
        parsed: Any = True
    elif value in ("false", "no", "0"):
        parsed = False
    elif value in ("none", "null"):
        parsed = None
    else:
        parsed = args.value

    state.setdefault("decisions", {})[args.key] = parsed
    state.setdefault("decision_rationale", {})[args.key] = {
        "value": parsed,
        "rationale": args.rationale,
        "at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    save_state(state)
    print(f"DECISION {args.key}={parsed!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hot40i autonomous roadmap controller")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser("status").set_defaults(fn=cmd_status)
    subparsers.add_parser("next").set_defaults(fn=cmd_next)

    command = subparsers.add_parser("start")
    command.add_argument("task")
    command.set_defaults(fn=cmd_start)

    command = subparsers.add_parser("pass")
    command.add_argument("task")
    command.add_argument("--evidence", required=True)
    command.set_defaults(fn=cmd_pass)

    command = subparsers.add_parser("block")
    command.add_argument("task")
    command.add_argument("--reason", required=True)
    command.set_defaults(fn=cmd_block)

    command = subparsers.add_parser("unblock")
    command.add_argument("task")
    command.set_defaults(fn=cmd_unblock)

    command = subparsers.add_parser("gate")
    command.add_argument("gate")
    command.set_defaults(fn=cmd_gate)

    command = subparsers.add_parser("decide")
    command.add_argument("key")
    command.add_argument("value")
    command.add_argument("--rationale", required=True)
    command.set_defaults(fn=cmd_decide)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

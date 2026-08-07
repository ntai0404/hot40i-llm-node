#!/usr/bin/env python3
"""Repository-level contract validator for autonomous handoff.

Quick mode validates static contracts and CLI importability. Full mode also runs
unit tests and the host C++ build. Keep this script dependency-light because it
is the first command a fresh coding agent executes.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "PROJECT_SPEC.md",
    "HANDOFF.md",
    "AUTONOMOUS_AGENT_PROMPT.md",
    "AGENTS.md",
    "PROJECT_STATE.yaml",
    "roadmap/tasks.yaml",
    "roadmap/gates.yaml",
    "roadmap/requirements.yaml",
    "third_party/manifest.yaml",
    "third_party/LOCK.yaml",
    "schemas/task_evidence.schema.json",
    "docs/13_GPT_OSS_20B_MODEL_CONTRACT.md",
    "docs/18_REQUIREMENTS_TRACEABILITY.md",
]
TASK_REQUIRED_FIELDS = {
    "id",
    "title",
    "phase",
    "priority",
    "depends_on",
    "optional",
    "goal",
    "allowed_paths",
    "implementation",
    "verification",
    "required_artifacts",
    "pass_criteria",
    "failure_policy",
}


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    process = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    print(process.stdout, end="")
    if process.returncode:
        print(process.stderr, end="", file=sys.stderr)
        fail(f"command failed {process.returncode}: {' '.join(command)}")


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"invalid YAML {path.relative_to(ROOT)}: {exc}")
    if not isinstance(value, dict):
        fail(f"expected object in {path.relative_to(ROOT)}")
    return value


def validate_structured_files() -> None:
    for path in sorted(ROOT.rglob("*.json")):
        if any(part.startswith("build") for part in path.parts):
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        if path.parent.name == "schemas":
            try:
                import jsonschema
            except ImportError:
                continue
            try:
                jsonschema.Draft202012Validator.check_schema(document)
            except Exception as exc:
                fail(f"invalid JSON Schema {path.relative_to(ROOT)}: {exc}")

    for pattern in ("*.yaml", "*.yml"):
        for path in sorted(ROOT.rglob(pattern)):
            if any(part.startswith("build") for part in path.parts):
                continue
            load_yaml(path)

    print("OK: structured files parse and schemas are valid")


def validate_markdown_links() -> None:
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    failures: list[str] = []
    for path in sorted(ROOT.rglob("*.md")):
        if any(part.startswith("build") for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in link_pattern.finditer(text):
            raw = match.group(1).strip()
            link = raw.split("#", 1)[0]
            if not link or "://" in link or link.startswith("mailto:"):
                continue
            target = (path.parent / link).resolve()
            if not target.exists():
                failures.append(f"{path.relative_to(ROOT)} -> {raw}")
    if failures:
        fail("broken local Markdown links: " + "; ".join(failures))
    print("OK: local Markdown links resolve")


def validate_dag_and_state() -> tuple[dict[str, dict[str, Any]], set[str], set[str]]:
    task_data = load_yaml(ROOT / "roadmap/tasks.yaml")
    tasks = task_data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        fail("roadmap/tasks.yaml must contain a non-empty tasks list")

    ids = [task.get("id") for task in tasks]
    if any(not isinstance(task_id, str) or not task_id for task_id in ids):
        fail("every task needs a non-empty string id")
    if len(ids) != len(set(ids)):
        fail("duplicate task ids")
    task_by_id = {task["id"]: task for task in tasks}

    for task in tasks:
        missing_fields = TASK_REQUIRED_FIELDS - task.keys()
        if missing_fields:
            fail(f"{task['id']} missing fields {sorted(missing_fields)}")
        for field in (
            "allowed_paths",
            "implementation",
            "verification",
            "required_artifacts",
            "pass_criteria",
            "failure_policy",
        ):
            if not task[field]:
                fail(f"{task['id']} has empty {field}")
        if len(task["verification"]) != len(set(task["verification"])):
            fail(f"{task['id']} contains duplicate verification strings")
        if not any(str(item).endswith("/evidence.json") for item in task["required_artifacts"]):
            fail(f"{task['id']} must require its evidence.json artifact")
        for dependency in task.get("depends_on", []):
            if dependency not in task_by_id:
                fail(f"{task['id']} depends on unknown {dependency}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            fail(f"cycle at {task_id}")
        visiting.add(task_id)
        for dependency in task_by_id[task_id].get("depends_on", []):
            dfs(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in ids:
        dfs(task_id)

    gate_data = load_yaml(ROOT / "roadmap/gates.yaml")
    gates = gate_data.get("gates")
    if not isinstance(gates, list) or not gates:
        fail("roadmap/gates.yaml must contain a non-empty gates list")
    gate_ids = [gate.get("id") for gate in gates]
    if len(gate_ids) != len(set(gate_ids)):
        fail("duplicate gate ids")
    for gate in gates:
        for task_id in gate.get("requires_tasks", []):
            if task_id not in task_by_id:
                fail(f"gate {gate['id']} requires unknown {task_id}")

    if "F03" not in task_by_id:
        fail("F03 final audit task is required")
    final_gate = next((gate for gate in gates if gate.get("id") == "FINAL_DEPLOYMENT"), None)
    if final_gate is None or "F03" not in final_gate.get("requires_tasks", []):
        fail("FINAL_DEPLOYMENT must require F03")

    ancestors: set[str] = set()

    def collect_ancestors(task_id: str) -> None:
        for dependency in task_by_id[task_id].get("depends_on", []):
            if dependency not in ancestors:
                ancestors.add(dependency)
                collect_ancestors(dependency)

    collect_ancestors("F03")
    mandatory = {task["id"] for task in tasks if not task.get("optional")}
    missing_from_final_path = mandatory - ancestors - {"F03"}
    if missing_from_final_path:
        fail(
            "mandatory tasks are not on F03 dependency path: "
            + ", ".join(sorted(missing_from_final_path))
        )

    state = load_yaml(ROOT / "PROJECT_STATE.yaml")
    completed = set(state.get("completed_tasks", []))
    blocked = set((state.get("blocked_tasks") or {}).keys())
    in_progress = state.get("in_progress")
    unknown_state_ids = (completed | blocked | ({in_progress} if in_progress else set())) - set(ids)
    if unknown_state_ids:
        fail("PROJECT_STATE references unknown tasks: " + ", ".join(sorted(unknown_state_ids)))
    if completed & blocked:
        fail("tasks cannot be both completed and blocked")
    if in_progress and (in_progress in completed or in_progress in blocked):
        fail("in_progress task cannot also be completed/blocked")
    for task_id in completed:
        missing = [
            dependency
            for dependency in task_by_id[task_id].get("depends_on", [])
            if dependency not in completed
        ]
        if missing:
            fail(f"completed task {task_id} has incomplete dependencies {missing}")

    evidence_map = state.get("task_evidence", {}) or {}
    for task_id in completed:
        evidence = evidence_map.get(task_id)
        if not evidence:
            fail(f"completed task {task_id} has no task_evidence entry")
        if not (ROOT / evidence).exists():
            fail(f"completed task {task_id} evidence does not exist: {evidence}")

    print(f"OK: DAG {len(ids)} tasks, {len(gates)} gates; mandatory final path is closed")
    return task_by_id, set(gate_ids), mandatory



def validate_upstream_contract() -> None:
    manifest = load_yaml(ROOT / "third_party/manifest.yaml")
    lock = load_yaml(ROOT / "third_party/LOCK.yaml")
    manifest_items = {item["name"]: item for item in manifest.get("upstreams", [])}
    lock_items = {item["name"]: item for item in lock.get("upstreams", [])}
    if set(manifest_items) != set(lock_items):
        fail(
            "third_party manifest/lock upstream sets differ: "
            f"manifest_only={sorted(set(manifest_items) - set(lock_items))}, "
            f"lock_only={sorted(set(lock_items) - set(manifest_items))}"
        )
    for name, item in manifest_items.items():
        if item.get("url") != lock_items[name].get("url"):
            fail(f"upstream URL mismatch for {name}")
        commit = lock_items[name].get("resolved_commit")
        if commit is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", str(commit)):
            fail(f"resolved_commit for {name} must be a full 40-hex SHA or null")
        if lock_items[name].get("resolution_required") is False and commit is None:
            fail(f"upstream {name} says resolution_required=false but has no commit")

    research = load_yaml(ROOT / "research/sources.yaml")
    source_ids = [item.get("id") for item in research.get("sources", [])]
    if len(source_ids) != len(set(source_ids)):
        fail("duplicate research source ids")
    print(f"OK: upstream contract {len(manifest_items)} code refs, {len(source_ids)} research sources")

def validate_requirements(
    task_by_id: dict[str, dict[str, Any]],
    gate_ids: set[str],
) -> None:
    data = load_yaml(ROOT / "roadmap/requirements.yaml")
    requirements = data.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        fail("roadmap/requirements.yaml must contain requirements")
    ids = [item.get("id") for item in requirements]
    if len(ids) != len(set(ids)):
        fail("duplicate requirement ids")
    for requirement in requirements:
        if not requirement.get("statement"):
            fail(f"requirement {requirement.get('id')} missing statement")
        tasks = requirement.get("verified_by_tasks", [])
        if not tasks:
            fail(f"requirement {requirement['id']} has no verifying task")
        unknown_tasks = set(tasks) - set(task_by_id)
        if unknown_tasks:
            fail(
                f"requirement {requirement['id']} references unknown tasks {sorted(unknown_tasks)}"
            )
        gate = requirement.get("acceptance_gate")
        if gate and gate not in gate_ids:
            fail(f"requirement {requirement['id']} references unknown gate {gate}")
    print(f"OK: requirements traceability {len(requirements)} requirements")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            fail(f"missing {relative}")

    validate_structured_files()
    validate_markdown_links()
    validate_upstream_contract()
    task_by_id, gate_ids, _ = validate_dag_and_state()
    validate_requirements(task_by_id, gate_ids)
    run([sys.executable, "scripts/render_requirements.py", "--check"])

    run([sys.executable, "-m", "host.device_lab.cli", "--help"])
    run([sys.executable, "scripts/taskctl.py", "status"])

    if not args.quick:
        run([sys.executable, "-m", "compileall", "-q", "host", "scripts", "tools", "tests/python"])
        run([sys.executable, "-m", "pytest", "-q"])
        if platform.system() == "Windows":
            print("SKIP: POSIX host C++ smoke build on native Windows; use Android NDK or WSL. Linux CI remains canonical.")
        else:
            run(
                [
                    "cmake",
                    "-S",
                    ".",
                    "-B",
                    "build-handoff",
                    "-DCMAKE_BUILD_TYPE=Release",
                ]
            )
            run(["cmake", "--build", "build-handoff", "-j2"])
            run(["ctest", "--test-dir", "build-handoff", "--output-on-failure"])

    print("HANDOFF_CHECK_PASS")


if __name__ == "__main__":
    main()

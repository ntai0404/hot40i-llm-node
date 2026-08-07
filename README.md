# Hot40i LLM Node

Autonomous research-and-engineering codebase for turning an Infinix Hot 40i (4 GB RAM / 128 GB flash) into a USB-attached local inference node for a high-capability sparse/MoE model. The primary target is OpenAI `gpt-oss-20b`, with flash-resident expert weights and a strictly bounded RAM working set.

> **Handoff status:** autonomous-agent-ready scaffold. The repository is intentionally not claiming that `gpt-oss-20b` already runs on the phone. It contains the execution contract, task DAG, gates, evidence schemas, safety boundaries and runtime foundation an agent must follow until the final acceptance gate.

## Start here — human or coding agent

Read in this order:

1. [`HANDOFF.md`](HANDOFF.md) — mission, current state, truth/unknowns, final Definition of Done.
2. [`BASELINE_STATUS.md`](BASELINE_STATUS.md) — verified delivery baseline and known-unproven items.
3. [`PROJECT_SPEC.md`](PROJECT_SPEC.md) — complete system/product/research specification.
4. [`AUTONOMOUS_AGENT_PROMPT.md`](AUTONOMOUS_AGENT_PROMPT.md) — the single prompt to give a code agent.
5. [`AGENTS.md`](AGENTS.md) — binding engineering/safety rules (Codex reads this natively).
6. [`roadmap/tasks.yaml`](roadmap/tasks.yaml) — machine-readable executable task DAG.
7. [`roadmap/requirements.yaml`](roadmap/requirements.yaml) — machine-readable requirements traceability.
8. [`roadmap/gates.yaml`](roadmap/gates.yaml) — pass/fail/decision gates.
9. [`docs/13_GPT_OSS_20B_MODEL_CONTRACT.md`](docs/13_GPT_OSS_20B_MODEL_CONTRACT.md) — exact model facts and memory/I/O equations.

## Local preflight

Requirements: Python 3.11+, CMake 3.20+, a C++20 compiler, Git, Android platform-tools (`adb`). Android NDK is required only when the roadmap reaches device-native compilation.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'

hot40 doctor
python scripts/taskctl.py status
python scripts/handoff_check.py --quick
```

Native host smoke test (Linux/macOS/WSL; Windows should use WSL or the Android NDK build path):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Once the phone is connected with USB debugging authorized:

```bash
hot40 devices
hot40 probe --out artifacts/device-manifest.json
```

Do **not** use `python -m hot40_device`; that module does not exist. The installed CLI is `hot40`, backed by `host.device_lab.cli`.

## Autonomous workflow

The agent never waits for a new task prompt. It repeatedly runs:

```bash
python scripts/taskctl.py next
python scripts/taskctl.py start <TASK_ID>
# implement + verify + collect evidence
python scripts/taskctl.py pass <TASK_ID> --evidence artifacts/runs/<run>/evidence.json
```

If a task is genuinely blocked, it records the blocker and continues any independent ready task:

```bash
python scripts/taskctl.py block <TASK_ID> --reason "..."
python scripts/taskctl.py next
```

The agent stops only when the `FINAL_DEPLOYMENT` gate passes or no task can proceed because of an external/physical blocker.

## Definition of Done

Required final result on the primary stock-Android path:

- official OpenAI `gpt-oss-20b` semantics, weights and Harmony formatting are used;
- model generation completes without unbounded swapping/OOM and stays within a measured safe RSS budget;
- the storage engine explicitly measures and controls expert reads rather than declaring generic `mmap` thrashing a solution;
- stable generation and correctness regression are demonstrated;
- the phone exposes the inference service to the laptop over USB (ADB forwarding is acceptable for the primary path);
- 30-minute sustained run is captured with decode rate, RSS, flash bytes/token, cache statistics, CPU frequency and thermal data;
- final report classifies performance (`proof`, P0/P1/P2/stretch) and identifies the measured bottleneck.

Performance classes are objectives, not excuses to falsify completion: proof = correct service; P0 ≥0.25 tok/s; P1 ≥0.5; P2 ≥1; stretch ≥2 sustained.

## Repository map

```text
HANDOFF.md                       Human/agent entry point
AUTONOMOUS_AGENT_PROMPT.md       One-shot agent instruction
PROJECT_SPEC.md                  Complete project specification
PROJECT_STATE.yaml               Machine-readable execution state
roadmap/tasks.yaml               Detailed executable task DAG
roadmap/requirements.yaml        Requirement→task→gate traceability
roadmap/gates.yaml               Gate/decision definitions
schemas/                         Evidence/run/benchmark/model schemas
host/device_lab/                 Conservative ADB laboratory
host/gateway/                    Laptop-side gateway skeleton
device/runtime/                  C++ storage/runtime primitives
device/bench/                    Native microbenchmarks
tools/model/                     H40M/model tools
tools/trace/                     Trace analysis
third_party/manifest.yaml        Upstream intent
third_party/LOCK.yaml            Verified refs/pinning policy
research/sources.yaml            Research source-of-truth matrix
docs/                            Architecture/research/runbooks
scripts/taskctl.py               Task state machine
scripts/handoff_check.py         Handoff/DAG/repo validator
```

## Primary design thesis

Checkpoint capacity is not working-set capacity. For sparse MoE inference, the project attempts to keep unavoidable dense/shared state resident where beneficial and explicitly load/cache/prefetch only the selected experts from flash. The primary optimization metric is therefore not model-file size; it is **flash bytes read per generated token**, alongside latency, memory and thermal behavior.

See `docs/09_SYSTEM_THESIS.md` and `docs/14_RESEARCH_PLAYBOOK.md`.

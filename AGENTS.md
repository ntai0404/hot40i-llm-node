# AGENTS.md — binding rules for autonomous coding agents

## Mission
Build an evidence-driven ARM64 inference/storage runtime for an Infinix Hot 40i and carry the executable roadmap through `FINAL_DEPLOYMENT` without requiring per-task user prompts.

## Autonomy contract
- After completing a task, immediately pick the next ready task with `scripts/taskctl.py`.
- Do not stop at planning, architecture prose, or a partial milestone when ready tasks remain.
- Do not ask the user to choose among technical options that can be resolved by measurement or upstream evidence.
- If an external blocker prevents one task, record it and continue independent ready tasks.
- `roadmap/tasks.yaml` defines task scope/dependencies; `roadmap/gates.yaml` defines milestone truth; `PROJECT_STATE.yaml` records execution state.

## Hard device-safety boundary
- Primary path is stock Android; final success does not require flashing.
- NEVER execute bootloader unlock, `fastboot flash`, `fastboot erase`, partition writes, factory reset, vbmeta/AVB changes, FRP bypasses or writes to modem/NV/persist/calibration areas unless both: (a) explicit authorization is recorded in `PROJECT_STATE.yaml`; and (b) gate `RECOVERY_READY` is PASS.
- Pushing binaries/models into ordinary user/app/temp storage and launching/stopping test processes is allowed.
- Never reinterpret an old discussion as current destructive authorization.

## Engineering rules
- One narrowly scoped task per commit.
- Do not create speculative planning documents outside the roadmap unless a task requires one.
- Every change finishes with actual evidence: file/diff summary, commands, exit codes, tests, measured runtime behavior and limitations.
- Restart any changed runtime/service before testing it.
- Do not claim a speedup without before/after measurements on the same device/configuration.
- Do not claim model correctness from plausible text output; use golden numerical/operator and token-sequence tests.
- Do not claim memory safety from file size; measure RSS/available memory and OOM/swap behavior.

## Architecture constraints
- `device/runtime` must not depend on ADB, FastAPI or laptop tooling.
- Model semantics, compute backend, scheduling, placement/storage and observability are separate concerns.
- All model-storage access in the final engine goes through explicit storage/provider abstractions.
- Working-set RAM is bounded; caches cannot grow without a byte budget.
- `mmap` is allowed only as a baseline/diagnostic path.
- Preserve OpenAI gpt-oss semantics; official gpt-oss + Harmony are correctness authority.
- Treat existing `ExpertCache` vector-backed storage as a disposable prototype; final cache must integrate with fixed/preallocated memory.

## Source discipline
- Before importing/forking upstream code, resolve the ref and record it in `third_party/LOCK.yaml`.
- Prefer official OpenAI/Apple/Anthropic sources for their respective domains.
- Research systems are design references, not proof that their published phone performance transfers to T606.
- Record source URLs and exact refs in task evidence.

## Evidence discipline
Raw task output goes under `artifacts/runs/<timestamp>_<task>/`. Each completed task needs `evidence.json` matching `schemas/task_evidence.schema.json`. Every declared task verification must appear as a passed evidence entry using the exact specification string, and every required artifact must exist and be registered. `taskctl pass` enforces this contract and records the evidence path in `PROJECT_STATE.yaml`.

Never hide failing logs. Negative benchmark results are valid project results.

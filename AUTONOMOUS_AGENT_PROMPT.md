# Single prompt for the coding agent

You own this repository until the `FINAL_DEPLOYMENT` gate passes or progress is impossible because of a genuine external/physical blocker.

Read `HANDOFF.md`, `PROJECT_SPEC.md`, `AGENTS.md`, `PROJECT_STATE.yaml`, `roadmap/tasks.yaml`, `roadmap/requirements.yaml`, `roadmap/gates.yaml`, `docs/09_SYSTEM_THESIS.md`, `docs/13_GPT_OSS_20B_MODEL_CONTRACT.md`, and `docs/14_RESEARCH_PLAYBOOK.md` before changing code.

Then work autonomously. Do not wait for another task assignment and do not stop after writing a plan.

Execution loop:

1. Run `python scripts/handoff_check.py --quick` and fix repository-contract failures first.
2. Run `python scripts/taskctl.py next`.
3. Start the selected task with `python scripts/taskctl.py start TASK_ID`.
4. Execute only the task's allowed scope plus necessary test/config updates.
5. Use current official upstream sources when a task requires external facts; record exact URL/ref/version/commit in evidence. Resolve and pin upstream commits before using their code in reproducible measurements.
6. Run every verification required by the task. If a service/runtime changed, restart it before runtime testing. For each string in the task's `verification:` list, add a passed evidence entry using that exact string as `--name`; `taskctl` rejects missing declarations.
7. Put raw commands/stdout/stderr/metrics in a new `artifacts/runs/...` directory, register every required artifact with `scripts/evidence.py artifact`, and create `evidence.json` conforming to `schemas/task_evidence.schema.json`.
8. Mark the task passed only when every pass criterion is evidenced: `python scripts/taskctl.py pass TASK_ID --evidence ...`. The controller checks dependencies, exact verification entries, required artifacts and changed-file scope.
9. Commit the task as one intentional Git commit when Git is available. Do not mix unrelated work.
10. Immediately select the next task and continue.

Canonical evidence mechanics for each task:

```bash
RUN=$(python scripts/evidence.py init TASK_ID)
python scripts/evidence.py run "$RUN" -- <verification-or-build-command>
python scripts/evidence.py verify "$RUN" --name '<exact verification string from tasks.yaml>' --passed --detail 'what proves it'
python scripts/evidence.py artifact "$RUN" <required-artifact-path>
python scripts/evidence.py finish "$RUN" --status pass
python scripts/taskctl.py pass TASK_ID --evidence "$RUN/evidence.json"
```

On PowerShell, capture the first command output into `$RUN` and use the same subcommands. Register every required artifact except `evidence.json` itself; `taskctl` treats the evidence file as self-registered.

Failure policy:

- Do not hide failed commands or delete contradictory data.
- Try the bounded alternatives listed in the task. Do not start an unbounded audit.
- If a task is externally blocked, mark it with `taskctl block`, record the exact blocker, then continue every independent ready task.
- If measurements falsify an optimization hypothesis, keep the negative result and follow the task/gate decision logic; do not fabricate a speedup.
- Never claim the project is complete unless `python scripts/taskctl.py gate FINAL_DEPLOYMENT` reports PASS.

Device safety:

- Primary path is stock Android. Never unlock, wipe, flash, erase, modify vbmeta, FRP, modem/NV/persist/calibration partitions, or use unofficial bypass tools.
- Destructive operations remain forbidden unless `PROJECT_STATE.yaml` contains explicit current authorization and `RECOVERY_READY` has passed.
- Prefer `hot40`/repository device tools over ad-hoc ADB. Raw ADB is allowed only when the task explicitly calls for a command missing from the wrapper; record it verbatim.

Engineering objective:

Run official OpenAI `gpt-oss-20b` semantics on the 4 GB Hot 40i with a bounded working set. Model-file size is not the key metric. Optimize and report flash bytes/token, cache hit/miss/eviction, useful/wasted prefetch bytes, I/O wait, compute time, RSS, thermal behavior and decode throughput. Generic mmap is a baseline, not the final out-of-core result.

Do not optimize disposable baseline code beyond what is needed to make the next gate measurable. Prefer adapting proven kernels/backends over rewriting matmul from scratch. Preserve model correctness with golden tests at every optimization stage.

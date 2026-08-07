# Autonomous execution protocol

The roadmap is designed for one coding agent session/assignment that may run for many tasks. The repository itself carries state so the agent does not need conversation memory to know what to do next.

## Loop

1. `python scripts/taskctl.py next`
2. `python scripts/taskctl.py start ID`
3. create a run directory and preserve raw output;
4. implement within task scope;
5. run task verification;
6. restart changed runtime/service before device verification;
7. create evidence JSON;
8. `python scripts/taskctl.py pass ID --evidence ...`;
9. commit;
10. repeat.

## When a task fails

A failed experiment is not a failed project. Record raw data. Try only the bounded alternatives in the task. Do not launch broad unrelated audits.

If a physical dependency blocks one task (device disconnected, USB authorization required, external download unavailable), mark it blocked. `taskctl next` should then expose any independent tasks. Only stop when no task can proceed.

## State integrity

`PROJECT_STATE.yaml` is execution metadata, not a place to declare imaginary progress. A task enters `completed_tasks` only through `taskctl pass`, which requires dependency completion and an evidence file.

Do not manually mark future tasks complete to unblock the DAG.

## Git discipline

If the archive has no `.git`, initialize one and commit the handoff baseline. Thereafter one roadmap task = one intentional commit. Generated model weights and large benchmark blobs should remain ignored or external; manifests/checksums/results belong in Git when practical.

## Research refresh

R00/R01 happen first because this is a fast-moving stack. Use exact current official sources and pin immutable refs. If an upstream release changed since this handoff, update the lock + evidence rather than silently using `main`.

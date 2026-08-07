# Failure/recovery policy

## Software task failures
1. Preserve failing output.
2. Identify the smallest hypothesis that explains it.
3. Try only task-scoped alternatives.
4. Re-run verification after runtime restart where relevant.
5. If still blocked by an external prerequisite, mark task blocked and continue independent DAG work.

## Device instability
If the phone overheats, becomes memory-pressured or repeatedly disconnects, stop the workload, capture final state, cool/reboot normally, and lower workload/budget. Never respond to instability by enabling destructive system changes.

## Model correctness failure
Correctness beats performance. If routing/logits diverge, disable recent optimization and bisect with golden fixtures. No performance task may waive numerical gates.

## Destructive recovery
Not part of the primary path. See `docs/03_DEVICE_SAFETY.md`. No destructive operation until authorization + recovery gate.

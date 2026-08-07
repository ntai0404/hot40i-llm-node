# Evidence and artifact contract

Each task creates one immutable-ish run directory:

```text
artifacts/runs/20260807T120000Z_D04/
  evidence.json
  commands.jsonl
  stdout/
  stderr/
  result.json
  thermal.jsonl          # where relevant
  expert_trace.jsonl     # where relevant
```

`evidence.json` must conform to `schemas/task_evidence.schema.json`.

Every benchmark headline in final docs must point back to a machine-readable result/raw run. It is acceptable for huge model files to live outside Git; manifests/checksums and provenance must remain.

Commands that fail are recorded with their nonzero exit code. Do not overwrite a failed run with a successful retry; create/retain evidence so the history is understandable.

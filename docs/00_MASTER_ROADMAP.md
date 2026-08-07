# Master roadmap — autonomous edition

The canonical executable roadmap is **`roadmap/tasks.yaml`** and the canonical gates are **`roadmap/gates.yaml`**. This document explains the phase logic; do not manually infer task completion from it.

## Final research question
Can the 4 GB Hot 40i run official OpenAI `gpt-oss-20b` semantics by treating its 12.8 GiB checkpoint as a flash-resident sparse model and keeping only a bounded dense/state/expert working set in RAM?

## Phase R — freeze truth before code
Pin exact upstreams, refresh official OpenAI/Apple/Anthropic/research assumptions, and harden the safe device interface.

## Phase D — characterize the exact phone
ADB transport, manifest, safe RSS budget, storage identity/access-pattern benchmarks, CPU/DRAM and sustained thermal behavior. Retail spec sheets are not sufficient evidence.

## Phase B — choose compute kernels empirically
Compare llama.cpp/ggml, MNN, ExecuTorch/XNNPACK and MLC where viable. The project owns storage scheduling; the selected backend supplies mature compute primitives.

## Phase C — numerical semantics before scale
Generate a tiny gpt-oss-shaped fixture and prove router, MXFP4 and attention parity against trusted reference outputs.

## Phase M — exact model inventory and H40M
Inspect official `gpt-oss-20b`; build deterministic H40M; place/quantize dense components based on exact bytes and real RAM budget; keep input embeddings row-addressable; treat output projection as a dedicated bottleneck.

## Phase S — bounded storage runtime
Preallocated memory plan, aligned flash reads, arena-backed expert cache, unified traces, then correctness-first MoE schedule without prefetch.

## Phase P — official model proof
First token, stable 128 tokens, then 512/1024 characterization. Only after this point do optimization claims matter.

## Phase O — measured optimization
Double buffering, cache policies, Apple-style reuse where applicable, ActiveFlow/EdgeMoE-style prediction/prefetch, trace-guided repacking, KV/context, affinity/thermal and dense/output tuning.

## Phase A — productize the USB node
Minimal device service, laptop Harmony/Responses gateway, robust ADB forwarding and a real client/Codex demonstration.

## Phase OS — optional late optimization
Non-destructive Android trimming first. Minimal Linux is optional, late, separately authorized and never a prerequisite for the primary final gate.

## Phase F — acceptance
Full correctness regression, >=30 minute sustained run, reproducible final report and `FINAL_DEPLOYMENT` gate audit.

Run `python scripts/taskctl.py next` for the exact next task.

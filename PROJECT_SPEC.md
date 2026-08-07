# Project Specification — Hot40i LLM Node

**Spec version:** 0.2  
**Research snapshot:** 2026-08-07  
**Primary hardware:** Infinix Hot 40i, 4 GB physical RAM / 128 GB flash, exact retail variant to be measured  
**Primary model target:** OpenAI `gpt-oss-20b`  
**Primary deployment path:** stock Android + native ARM64 runtime + USB-C/ADB forwarding  
**Canonical execution plan:** [`roadmap/tasks.yaml`](roadmap/tasks.yaml)  
**Canonical requirements:** [`roadmap/requirements.yaml`](roadmap/requirements.yaml)

This document is the system-level specification for the repository. It defines what the project is trying to prove, what constitutes a correct implementation, what the coding agent is allowed to change, and how final success is accepted. Where this prose conflicts with the machine-readable task DAG or a higher-authority pinned upstream about model semantics, the source-of-truth rules in [`docs/12_SOURCE_OF_TRUTH.md`](docs/12_SOURCE_OF_TRUTH.md) apply.

---

## 1. Purpose

The project investigates whether a low-memory smartphone can host a substantially larger sparse/Mixture-of-Experts language model by separating **checkpoint capacity** from **RAM working-set capacity**.

The concrete target is to run official OpenAI `gpt-oss-20b` semantics on an Infinix Hot 40i with only 4 GB of physical RAM. The full model checkpoint is expected to remain primarily on device flash. The runtime explicitly manages resident dense/shared tensors, state, expert cache, I/O buffers and selected MoE expert reads so that the process stays inside a measured memory budget.

The project is intentionally not defined as “make a large GGUF file mmap successfully.” A final result must make model-storage behavior observable and explainable.

---

## 2. Research question and hypothesis

### 2.1 Research question

Can the target phone provide a **correct, bounded and usable** local `gpt-oss-20b` inference service when most MoE expert capacity resides on flash and only a bounded working set is retained in RAM?

### 2.2 Core hypothesis

For sparse MoE inference:

```text
checkpoint bytes  >>  required resident bytes
```

can be practical only if the runtime also controls:

```text
flash bytes / generated token
cache hit/miss behavior
I/O-compute overlap
storage layout/read granularity
resident dense/shared tensors
KV/state memory
thermal throttling
```

A large checkpoint fitting in 128 GB storage is necessary but not sufficient.

### 2.3 Falsifiable outcome

The project may conclude that the target hardware is too I/O-bound, compute-bound, memory-bound or thermally constrained for a useful decode rate. That is an acceptable research result if the official-model proof, measurements and bottleneck analysis are reproducible. The agent must not hide negative results or manufacture a speed target.

---

## 3. Authoritative technical assumptions

R00/R01 must revalidate and pin these facts before they affect irreversible implementation choices.

### 3.1 OpenAI

OpenAI is authoritative for the final model semantics and Harmony protocol. The research snapshot treats `gpt-oss-20b` as approximately 21B total parameters with approximately 3.6B active parameters/token, using sparse MoE routing and OpenAI's MXFP4 representation for MoE weights. The official model/repository and Harmony implementation are canonical; aggregate parameter counts are not a substitute for inspecting the exact pinned checkpoint in M00.

Primary sources:

- https://github.com/openai/gpt-oss
- https://github.com/openai/harmony
- https://deploymentsafety.openai.com/gpt-oss/paperbench
- https://github.com/openai/codex

### 3.2 Apple

Apple's *LLM in a Flash* is the primary flash-aware systems reference. The transferable principles are:

- reduce bytes transferred from flash;
- favor larger/more contiguous reads;
- exploit reuse where the model activation pattern allows it;
- use a hardware-aware cost model rather than capacity-only reasoning.

Its studied contextual sparsity mechanism is not assumed to be equivalent to `gpt-oss` top-k MoE routing. Apple MLX is a runtime-design reference, not an Android/T606 backend.

Primary sources:

- https://machinelearning.apple.com/research/efficient-large-language
- https://github.com/ml-explore/mlx
- https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025

### 3.3 Anthropic / MCP

Anthropic is used as an agent-workflow/tool-boundary reference, not as a model-runtime authority for this project. Claude Code may independently review implementation/evidence. MCP may be introduced where it improves the safety and structure of device tooling.

Primary sources:

- https://github.com/anthropics/claude-code
- https://github.com/modelcontextprotocol/modelcontextprotocol

### 3.4 Related systems research

PowerInfer-2, ActiveFlow, EdgeMoE, MoE-Infinity and SmallThinker are design references for caching, expert activation tracing, prefetch, flash/DRAM swapping and model/runtime co-design. Their reported performance is **not** an expected Hot 40i result.

See [`research/sources.yaml`](research/sources.yaml) and [`docs/14_RESEARCH_PLAYBOOK.md`](docs/14_RESEARCH_PLAYBOOK.md).

---

## 4. Scope

### 4.1 In scope

- exact characterization of the user's physical Hot 40i;
- stock-Android ADB transport over USB-C;
- ARM64 native C/C++ inference/storage runtime;
- empirical selection/reuse of mature ARM compute kernels;
- exact `gpt-oss` router/attention/MXFP4 semantics;
- deterministic model inspection/conversion/repacking;
- explicit tensor placement and byte-bounded memory planning;
- flash-resident MoE expert loading/caching;
- cache/prefetch/layout optimization backed by traces;
- laptop-side Harmony/Responses gateway;
- USB-attached local service;
- correctness, memory, storage, thermal and sustained performance evidence.

### 4.2 Optional scope

Only after the mandatory stock-Android path is proven:

- non-destructive Android service trimming;
- recovery research;
- minimal Linux evaluation.

### 4.3 Out of scope for the mandatory path

- bootloader unlock or flashing as a prerequisite;
- FRP/security bypasses;
- writes to modem/NV/persist/calibration partitions;
- replacing exact model semantics with a smaller unrelated model and calling the target complete;
- server/cloud offload of the model's main inference compute;
- unbounded Android swap/zram as a substitute for a bounded runtime;
- rewriting generic matrix kernels from scratch before viable mature backends are measured.

---

## 5. System boundary

```text
┌──────────────────────────── LAPTOP ────────────────────────────┐
│                                                                │
│  Coding agent / task control / evidence                        │
│  Model inspection + conversion + trace analysis                │
│  Harmony + Responses-compatible gateway                        │
│                                                                │
└──────────────────────────────┬─────────────────────────────────┘
                               │ USB-C ↔ USB-C
                               │ ADB control + TCP forwarding
                               ▼
┌──────────────────────────── PHONE ─────────────────────────────┐
│ stock Android                                                  │
│                                                                │
│ minimal device inference service                               │
│          │                                                     │
│          ▼                                                     │
│ ModelRuntime / GraphScheduler                                  │
│          │                                                     │
│          ├── ComputeBackend                                    │
│          │                                                     │
│          └── MoEScheduler                                      │
│                 │                                              │
│                 ├── MemoryPlan / PlacementManager              │
│                 │      ├── resident dense/shared tensors       │
│                 │      ├── KV/state                            │
│                 │      └── fixed-budget expert cache           │
│                 │                                              │
│                 └── TensorStore / FlashTensorProvider          │
│                           │                                    │
│                           └── H40M flash arenas                 │
└────────────────────────────────────────────────────────────────┘
```

The detailed contract is [`docs/01_ARCHITECTURE.md`](docs/01_ARCHITECTURE.md).

---

## 6. Functional requirements

Machine-readable requirement IDs and verification mappings live in [`roadmap/requirements.yaml`](roadmap/requirements.yaml). Major requirements are summarized below.

### 6.1 Device and transport

- **FR-001:** authorized USB-C ADB connectivity must be proven on the actual device.
- **FR-002:** exact device/build/hardware evidence must be captured machine-readably.
- **FR-003:** the runtime memory ceiling is derived from measurements, not the nominal 4 GB value.
- **FR-004/005:** storage, CPU/DRAM and thermal limits are measured before backend/runtime decisions.

### 6.2 Model correctness

- **FR-007:** tiny gpt-oss-shaped numerical parity is a hard gate before full-model out-of-core work.
- **FR-008:** final model semantics, checkpoint provenance and Harmony behavior use official OpenAI sources.
- **FR-014:** the official model must progress from first-token proof to stable multi-token generation.

### 6.3 Storage and memory

- **FR-009:** H40M/1 maps logical tensors to deterministic physical ranges with provenance and validation.
- **FR-010:** embedding lookup must not require a fully resident embedding matrix.
- **FR-011:** dense/shared placement must be explicit and measured.
- **FR-012:** expert loading/caching must be explicit and byte-bounded.
- **FR-013:** storage/cache/router/compute events must be observable.

### 6.4 Optimization

- **FR-015:** each optimization is an experiment with a baseline, same-device comparison and correctness regression.
- prediction may schedule reads early but may never change exact router-selected experts.

### 6.5 Service

- **FR-016:** the phone hosts a minimal inference service; Harmony/tool orchestration stays on the laptop initially.
- **FR-017:** final laptop→phone inference is demonstrated over the physical USB-C path.

---

## 7. Non-functional requirements

### 7.1 Safety

The mandatory path is non-destructive stock Android. Destructive operations are prohibited unless both conditions are true:

1. `PROJECT_STATE.yaml` records explicit current authorization; and
2. gate `RECOVERY_READY` has passed.

The repository device wrapper is a defense-in-depth boundary, not a claim that arbitrary shell access can be technically sandboxed from an external coding agent.

### 7.2 Correctness

Readable output is not proof. The project requires operator/golden parity before full-model scaling and a final regression after optimizations.

### 7.3 Memory boundedness

Every cache/arena has a byte budget. The final sustained run must show no unbounded RSS, OOM loop or uncontrolled swap thrashing.

### 7.4 Reproducibility

Every roadmap task records:

- exact commands and exit codes;
- stdout/stderr files;
- files changed;
- source URL/ref/version/commit where relevant;
- task-specific verification results;
- required artifacts and metrics;
- limitations/negative results.

External code used for benchmark/implementation decisions is pinned in `third_party/LOCK.yaml`.

### 7.5 Sustained performance

Final throughput is measured under a >=30-minute workload. Short peak measurements may be reported only as secondary diagnostics.

### 7.6 Performance classification

- **proof:** correct bounded service exists;
- **P0:** >=0.25 decode tok/s sustained;
- **P1:** >=0.5 tok/s;
- **P2:** >=1 tok/s;
- **stretch:** >=2 tok/s.

These labels classify the measured result; they are not permission to alter correctness or fabricate completion.

---

## 8. Memory model

The process working set is modeled as:

```text
RAM_process =
    runtime/code/stacks
  + resident dense/shared tensors
  + model state / KV
  + expert-cache arena
  + scratch/workspace
  + I/O/prefetch buffers
```

Constraint:

```text
peak_RSS <= measured_safe_RSS_budget
```

The safe budget is established in D02 on the exact device and is stored in project state/evidence. No design document may hard-code “4 GB available to the LLM.”

The exact gpt-oss decomposition and early I/O estimates are maintained in [`docs/13_GPT_OSS_20B_MODEL_CONTRACT.md`](docs/13_GPT_OSS_20B_MODEL_CONTRACT.md).

---

## 9. Storage model

### 9.1 Baseline

Generic file `mmap` is a valid baseline for measuring page faults and page-cache behavior. It is not the final storage architecture when model capacity significantly exceeds safe RAM.

### 9.2 Final storage abstraction

All final model reads pass through explicit model-storage/provider interfaces so the runtime can attribute bytes and latency to token/layer/tensor/expert.

### 9.3 H40M

H40M is project-local and deterministic. It is designed for explicit flash placement, not as a general model exchange standard.

Each tensor includes enough metadata to validate:

- semantic role;
- layer/expert identity where applicable;
- shape/dtype/quantization/layout;
- file/range/alignment;
- placement policy;
- checksum and source provenance.

See [`docs/07_H40M_FORMAT.md`](docs/07_H40M_FORMAT.md) and [`schemas/h40m_manifest.schema.json`](schemas/h40m_manifest.schema.json).

### 9.4 Expert cache

The final cache is fixed-budget and arena/slab-backed. The initial vector-backed `ExpertCache` exists only to validate interfaces and basic behavior; it must not be mistaken for the production cache design.

### 9.5 Prefetch

Prefetch is speculative I/O only. Correctness always uses the exact experts selected by the model router. Wasted prefetch bytes are explicitly measured.

---

## 10. Compute backend strategy

The project separates model/storage scheduling from compute kernels. B00–B04 compare viable mobile/ARM backends on the exact device. Candidate families include ggml/llama.cpp, MNN, ExecuTorch/XNNPACK and MLC where technically viable.

Selection criteria include:

- correctness/operation coverage;
- decode and prompt throughput;
- peak memory/workspace requirements;
- Android/ARM64 maintainability;
- integration effort with explicit tensor providers;
- thermal behavior.

No backend is selected solely because it wins on unrelated hardware.

---

## 11. Correctness strategy

Correctness is staged:

1. generate tiny gpt-oss-shaped fixtures on a trusted reference path;
2. verify router/top-k behavior;
3. verify MXFP4 expert math;
4. verify attention math;
5. verify tiny end-to-end layer/logit/token parity;
6. inspect and convert the exact official checkpoint;
7. prove first official token;
8. prove stable 128/512/1024-token behavior;
9. rerun golden and end-to-end regression after all performance optimizations.

Quantization-specific tolerance must be recorded rather than assumed.

---

## 12. Observability and metrics

The runtime/bench harness must make the following attributable where applicable:

- token index and layer;
- router-selected experts;
- cache hit/miss/eviction;
- bytes loaded;
- read operation count;
- read latency;
- useful and wasted prefetch bytes;
- compute duration;
- I/O wait/overlap;
- TTFT;
- prompt and decode tok/s;
- peak/current RSS;
- CPU frequency/core affinity;
- thermal series;
- battery/current if safely exposed.

Final storage optimization is primarily evaluated by **flash bytes per generated token** plus end-to-end decode time, not model-file size.

---

## 13. Service protocol

### 13.1 Device service

The phone service should remain minimal during memory-constrained stages. Its stable internal contract is allowed to be simpler than the public OpenAI-style API.

### 13.2 Laptop gateway

The laptop owns Harmony rendering/parsing and, later, a Responses-compatible boundary. This keeps protocol/tool complexity and memory overhead off the phone while preserving official model formatting.

### 13.3 USB transport

The mandatory stock-Android path uses ADB TCP forwarding. A later minimal-Linux path may replace this with a USB gadget network protocol, but Linux is not required for final acceptance.

---

## 14. Autonomous coding-agent execution contract

The project is designed to be handed to one coding agent once.

The agent reads:

1. [`HANDOFF.md`](HANDOFF.md)
2. this specification
3. [`AUTONOMOUS_AGENT_PROMPT.md`](AUTONOMOUS_AGENT_PROMPT.md)
4. [`AGENTS.md`](AGENTS.md)
5. `PROJECT_STATE.yaml`
6. task/gate/requirements YAML

The canonical loop is:

```text
next task
  -> start
  -> create evidence run
  -> implement only allowed scope
  -> run each declared verification
  -> register required artifacts
  -> finish evidence
  -> taskctl pass
  -> commit
  -> next task
```

A task pass is intentionally stricter than an agent's prose claim: `taskctl.py` verifies dependency closure, evidence schema, exact declared verification entries, required artifact existence/registration and task file scope.

A physical/external blocker may block one task while independent ready tasks continue. The agent stops only when `FINAL_DEPLOYMENT` passes or no mandatory work can proceed because of an unresolved external blocker.

---

## 15. Roadmap and gates

Phases:

- **R:** pin truth/research and device safety boundary;
- **D:** characterize exact device;
- **B:** choose compute backend;
- **C:** prove tiny numerical semantics;
- **M:** inspect/convert/place model;
- **S:** build bounded storage runtime;
- **P:** official-model proof;
- **O:** measured optimizations;
- **A:** USB service/productization;
- **OS:** optional OS work;
- **F:** final correctness, sustained test and report.

Hard gates:

- `HANDOFF_READY`
- `HARDWARE_CHARACTERIZED`
- `BACKEND_SELECTED`
- `NUMERICAL_PARITY`
- `STORAGE_RUNTIME_READY`
- `OFFICIAL_MODEL_PROOF`
- optional `RECOVERY_READY`
- `FINAL_DEPLOYMENT`

See [`docs/00_MASTER_ROADMAP.md`](docs/00_MASTER_ROADMAP.md), [`roadmap/tasks.yaml`](roadmap/tasks.yaml), and [`roadmap/gates.yaml`](roadmap/gates.yaml).

---

## 16. Final acceptance

`FINAL_DEPLOYMENT` may pass only after mandatory task evidence demonstrates all of the following:

1. official `gpt-oss-20b` model semantics/checkpoint provenance;
2. Harmony-compatible prompt/render/parse behavior;
3. bounded RAM under the measured safe budget;
4. explicit expert storage/cache metrics;
5. stable model generation;
6. laptop-to-phone USB service path after clean restart;
7. final correctness regression;
8. >=30-minute sustained deployment test;
9. final report with reproducible config, performance class, bottleneck and negative results;
10. every mandatory roadmap task has a recorded evidence bundle.

A successful final report may still conclude that performance is below P0. It may not conclude success if correctness, memory boundedness, provenance or evidence is missing.

---

## 17. Key risks and mitigation

| Risk | Consequence | Mitigation / decision point |
|---|---|---|
| Flash random-read bandwidth is too low | expert streaming dominates decode | D04 expert-shaped benchmark; contiguous repack/cache/prefetch; report hard lower bound |
| T606 compute is too weak | decode remains compute-bound | B04 empirical backend selection; O06 affinity; preserve negative result |
| Dense/output tensors consume too much RAM/bandwidth | MoE sparsity alone insufficient | M03/M05/M06 explicit quant/placement; O07 dedicated optimization |
| Android memory pressure | OOM/reclaim invalidates results | D02 safe RSS budget; S01 fixed arenas; F01 sustained memory telemetry |
| Thermal throttling | short benchmark misleading | D06 harness; O06; F01 >=30 min |
| Upstream semantics drift | incorrect runtime | R00 immutable pins; R01 contradiction log; tiny golden parity |
| Agent overclaims progress | unsafe/irreproducible handoff | taskctl evidence enforcement; state/evidence audit; final gate |
| Destructive device action | data/device loss | stock path default; wrapper + AGENTS rules; authorization + RECOVERY_READY |
| Published paper result does not transfer to T606 | false expectation | treat papers as mechanisms, not expected throughput; benchmark exact device |

---

## 18. Deliverables produced by the completed project

At final acceptance, the repository is expected to contain:

- pinned source locks and model provenance;
- exact device manifest and hardware benchmarks;
- selected compute backend evidence;
- tiny numerical golden fixtures;
- H40M converter/repacker and manifest/checksums;
- bounded ARM64 device runtime;
- trace/cache/storage instrumentation;
- official-model proof and characterization results;
- laptop gateway + USB service path;
- final correctness and sustained benchmark artifacts;
- `FINAL_REPORT.md` with reproduction instructions and bottleneck analysis.

Large model weights may remain outside Git, but their deterministic manifests/checksums/provenance must be retained.

---

## 19. Change control

Architecture or target changes that materially alter the research question must be recorded as an explicit project decision with evidence/rationale. A coding agent may choose among implementation alternatives when measurements resolve the choice; it may not silently redefine the final model, safety boundary or Definition of Done.

The first autonomous task remains `R00` until it has passed with evidence.

# Requirements traceability

> Generated from `roadmap/requirements.yaml` by `scripts/render_requirements.py`. Edit the YAML, not this table.

| ID | Type | Requirement | Verified by | Acceptance gate |
|---|---|---|---|---|
| `FR-001` | functional | The laptop shall discover and communicate with the target Hot 40i over USB-C using authorized stock-Android ADB before any inference work proceeds. | `D00` | `HARDWARE_CHARACTERIZED` |
| `FR-002` | functional | The project shall capture a normalized, machine-readable manifest of the exact device build, CPU, memory, partitions, mounts, thermal zones and block-device evidence. | `D01`, `D03` | `HARDWARE_CHARACTERIZED` |
| `FR-003` | functional | The runtime shall use a measured safe process-memory budget derived from the physical device rather than assuming all 4 GB of RAM is available. | `D02`, `S01`, `F01` | `FINAL_DEPLOYMENT` |
| `FR-004` | functional | Storage benchmarking shall reproduce expert-shaped access patterns and report throughput and latency distributions needed to estimate flash cost per generated token. | `D04` | `HARDWARE_CHARACTERIZED` |
| `FR-005` | functional | CPU, DRAM and sustained thermal behavior shall be measured on the exact phone before selecting or tuning the compute backend. | `D05`, `D06` | `HARDWARE_CHARACTERIZED` |
| `FR-006` | functional | The compute backend shall be selected by same-device measurements across viable ARM/mobile candidates rather than by paper or desktop benchmark claims. | `B00`, `B01`, `B02`, `B03`, `B04` | `BACKEND_SELECTED` |
| `FR-007` | correctness | Router, MXFP4 expert path and attention semantics shall achieve numerical parity on a tiny gpt-oss-shaped fixture before full-model out-of-core work. | `C00`, `C01`, `C02`, `C03`, `C04` | `NUMERICAL_PARITY` |
| `FR-008` | provenance | The official OpenAI gpt-oss-20b checkpoint, model semantics and Harmony format shall be pinned and provenance/checksums recorded before final conversion or deployment. | `R00`, `R01`, `M00`, `P00`, `A01` | `FINAL_DEPLOYMENT` |
| `FR-009` | functional | H40M/1 shall provide a deterministic mapping from logical model tensors to validated physical flash ranges, including dtype, quantization, shape, alignment, placement and source provenance. | `M01`, `M02` | `STORAGE_RUNTIME_READY` |
| `FR-010` | functional | Input embedding access shall support token-row lookup without requiring the full embedding matrix to be resident in RAM. | `M04`, `S06` | `STORAGE_RUNTIME_READY` |
| `FR-011` | functional | Dense/shared tensors, including attention and output projection, shall have an explicit measured placement/quantization strategy that fits the safe memory plan. | `M03`, `M05`, `M06`, `S01`, `O07` | `FINAL_DEPLOYMENT` |
| `FR-012` | functional | Expert weights shall be loaded through explicit indexed storage providers and a byte-bounded cache rather than relying on uncontrolled whole-model mmap/page-cache thrashing as the final design. | `S00`, `S02`, `S03`, `S05`, `S06` | `STORAGE_RUNTIME_READY` |
| `FR-013` | observability | The runtime shall emit machine-readable router, cache, I/O, compute and prefetch events sufficient to calculate flash bytes per token and explain bottlenecks. | `S04`, `P02`, `O00`, `O01`, `O03` | `FINAL_DEPLOYMENT` |
| `FR-014` | functional | The official model shall generate stable output beyond a one-token proof while respecting the measured memory budget. | `P00`, `P01`, `P02` | `OFFICIAL_MODEL_PROOF` |
| `FR-015` | optimization | Read/compute overlap, cache policy, reuse/windowing, prediction/prefetch, physical repacking, context/state, CPU affinity and dense bottlenecks shall each be evaluated with before/after evidence and correctness preserved. | `O00`, `O01`, `O02`, `O03`, `O04`, `O05`, `O06`, `O07` | `FINAL_DEPLOYMENT` |
| `FR-016` | service | The phone shall expose a minimal inference service and the laptop shall provide Harmony/Responses orchestration without requiring the full protocol stack to reside on the phone. | `A00`, `A01` | `FINAL_DEPLOYMENT` |
| `FR-017` | service | The laptop shall reach the device service over the physical USB-C path, with stock-Android ADB TCP forwarding accepted as the primary deployment transport. | `A02`, `A03`, `F01` | `FINAL_DEPLOYMENT` |
| `NFR-001` | safety | The mandatory path shall remain non-destructive on stock Android; bootloader unlock, partition writes, erase, AVB/vbmeta changes and bypass tooling are forbidden unless explicit authorization and RECOVERY_READY are both present. | `R02`, `D00`, `OS01` | `FINAL_DEPLOYMENT` |
| `NFR-002` | reproducibility | External code used for implementation or benchmark decisions shall be pinned to immutable commits and every task shall preserve commands, outputs, artifacts and source refs. | `R00`, `F02`, `F03` | `FINAL_DEPLOYMENT` |
| `NFR-003` | correctness | Performance optimizations shall never substitute predicted experts for exact router-selected experts and shall pass regression tests before final acceptance. | `O03`, `F00` | `FINAL_DEPLOYMENT` |
| `NFR-004` | memory | Caches and working buffers shall have explicit byte budgets and the final sustained service shall show no unbounded RSS, swap thrashing or OOM behavior. | `S01`, `S03`, `F01` | `FINAL_DEPLOYMENT` |
| `NFR-005` | performance | Final performance shall be reported as measured proof/P0/P1/P2/stretch classes without inventing a pass threshold that is unsupported by the hardware. | `F01`, `F02` | `FINAL_DEPLOYMENT` |
| `NFR-006` | thermal | Final headline throughput shall be based on sustained behavior, with at least a 30-minute test including CPU frequency and thermal telemetry where exposed by the device. | `D06`, `F01` | `FINAL_DEPLOYMENT` |
| `NFR-007` | autonomy | A coding agent shall be able to determine the next valid task, pass/block it with auditable evidence, and progress through the mandatory DAG without per-task human prompts. | `R00`, `F03` | `FINAL_DEPLOYMENT` |

The task DAG is canonical for execution. This table exists so a human reviewer can trace every system requirement to concrete implementation/verification work.

# Hot 40i gpt-oss-20b Final Report

## Verdict

The project has a correct, bounded proof deployment of official OpenAI `gpt-oss-20b` semantics on an Infinix Hot 40i. The measured final sustained service classification is **proof-only**: 0.03130 emitted tokens/second is below the roadmap P0 threshold of 0.25 tokens/second. This is a measured feasibility result, not a claim of practical chat speed.

The final acceptance evidence is [F01 sustained_30m.json](benchmarks/final/sustained_30m.json). It records 1,885 seconds, 59 successful device requests, zero failures, peak decoder RSS of 68,976 KiB against a 630,938 KiB budget, thermal status 0 and vmstat `oom_kill=0`.

## Reproduction

The runtime configuration was verified at commit `17271011` (the F01 commit; this report is the next task commit). Use the canonical branch and the locked refs in [third_party/LOCK.yaml](third_party/LOCK.yaml). The service command used for the final run was:

```text
H40_THREADS=6 H40_IO_OVERLAP=1 /data/local/tmp/h40m/inference_service_a00 \
  --port 8080 --rss-budget-kib 630938 \
  --runner /data/local/tmp/h40m/minimal_decoder_a00 \
  --source /data/local/tmp/h40m/source \
  --catalog /data/local/tmp/h40m/h40m/tensor_catalog.tsv \
  --experts /data/local/tmp/h40m/h40m/expert_arena.bin
```

The final run used Wi-Fi ADB `192.168.100.189:5555`, with a managed `tcp:18080 -> tcp:8080` forward. USB serial `112193741U000563` was also authorized and retained as a fallback. The raw device list and every request/metrics response are in [the F01 run directory](artifacts/runs/20260827T022000Z_F01/).

## Device And Model

- Device: Infinix X6528 / Hot 40i, Unisoc T606, Android 13, 8 online CPUs.
- CPU topology: CPUs 0-5 Cortex-A55 and CPUs 6-7 Cortex-A75; observed policy range 614,400-1,612,000 kHz.
- Governor: `schedutil`; fixed-performance mode was tested and rejected. Final affinity mask was `ff` with six dense workers and scheduler headroom. See [affinity_thermal.json](benchmarks/optimization/affinity_thermal.json).
- Model: official `openai/gpt-oss-20b`, 24 layers, hidden size 2,880, 64 query heads, 8 KV heads, head dimension 64, 32 local experts, top-4 routing, vocabulary 201,088, alternating full/sliding attention with a 128-token sliding window. See [config.json](artifacts/model/source/config.json).
- Official model commit: `6cee5e81ee83917806bbde320786a8fb61efebee`. Full shard sizes and checksums are in [artifacts/model/checksums.txt](artifacts/model/checksums.txt) and [gpt_oss_20b_inventory.json](artifacts/model/gpt_oss_20b_inventory.json).
- H40M manifest identity: checkpoint inventory SHA-256 `53c467455f35dfd00438ae0d68e1899d432d21c71b1ebc058713a450765896fc`; H40M/1 manifest file SHA-256 `b7c74670360e0ed8655d39da60a7464866a607004ec574d604b82b5b867e0391`. See [H40M manifest](artifacts/model/h40m/manifest.json).
- Production expert arena: H40M_EXPERT_ARENA/1, 10,468,587,776 bytes, SHA-256 `2b1607bf3ea4f164c4df2f78f260bb6b9681c31284c103f10e999cf4cb6c3456`. The v2 physical repack was byte-correct but not retained because its same-binary end-to-end speedup was 0.99984x. See [repacking.json](benchmarks/optimization/repacking.json) and [layout_v2.json](artifacts/model/h40m/layout_v2.json).

## Semantics And Memory

- Official Harmony is pinned to `openai-harmony==0.0.8`, commit `abd677f7ac962629c808197caa1feb9e3e95d2b0`. Rendering and tokenization are covered by [harmony_gateway_golden.json](tests/fixtures/harmony_gateway_golden.json) and [F00 correctness.json](benchmarks/final/correctness.json).
- Attention uses the official GPT-OSS layout: YaRN RoPE, GQA, attention sinks, causal masks and alternating 128-token sliding/full layers. Tiny/reference checks and the device path are recorded in [F00 evidence](artifacts/runs/20260827T022000Z_F00/evidence.json).
- MXFP4 experts use the official split-scale E2M1 semantics. Expert loading goes through the indexed provider, bounded cache and H40M arena; full checkpoint residency is never required.
- Safe RSS budget: 646,080,512 bytes / 630,938 KiB. Initial memory plan total: 331,306,112 bytes with 314,774,400 bytes headroom. See [memory_plan.json](artifacts/model/memory_plan.json) and [ADR_003](docs/decisions/ADR_003_MEMORY_PLACEMENT.md).
- Resident state is limited to small shared tensors, a token-row embedding cache, layer-local dense buffers, KV state and bounded expert/I/O scratch. Output projection is streamed in bounded Q8 chunks; no full vocabulary logits matrix is materialized. See [ADR_002](docs/decisions/ADR_002_OUTPUT_HEAD.md).

## Measured Performance

| Stage | Device result | Source |
| --- | ---: | --- |
| P02 512-token full-trace baseline | 0.035929 tok/s, 27.832 s/token | [long_decode.json](benchmarks/proof/long_decode.json) |
| O00 double-buffer screen | 0.0360303 tok/s, 1.016554x control | [O00 evidence](benchmarks/optimization/double_buffer.json) |
| O01 per-layer hotset screen | 0.039630937 tok/s, 1.004892x LRU | [cache_policies.json](benchmarks/optimization/cache_policies.json) |
| O07 selected six-worker screen | 0.039292924 tok/s; 1.121931x same-sweep one-worker control | [dense_output.json](benchmarks/optimization/dense_output.json) |
| F01 sustained production service | 0.031299735 tok/s wall, 31.390 s/request token | [sustained_30m.json](benchmarks/final/sustained_30m.json) |

F01 totals were 2,437,387,008 dense flash bytes and 1,270,702,080 expert flash bytes per one-token request, with 0 cache hits and 96 cache misses per request. The final production service emitted token 366 on all 59 requests and every response completed all 24 layers. The F01 raw responses and metrics are in [artifacts/runs/20260827T022000Z_F01](artifacts/runs/20260827T022000Z_F01/).

## Final Stability

- Observed duration: 1,885 seconds; configured duration: 1,860 seconds.
- Requests: 59/59 successful; service metrics ended at 59 inference, 59 completed, 0 failures.
- Decoder peak RSS: 68,976 KiB; service RSS HWM: 3,044 KiB; budget headroom at peak: 561,962 KiB.
- MemAvailable minimum/final: 1,411,244 / 1,604,884 KiB. SwapFree minimum/final: 659,236 / 861,156 KiB. SwapTotal: 2,114,396 KiB.
- Thermal status maximum: 0. Battery temperature maximum: 43.0 C while USB-powered. CPU policy observations: 614,400-1,612,000 kHz.
- Kernel `oom_kill` counter: 0. Stock Android denied zram `mm_stat`, `bd_stat` and PSI reads; those denials are preserved in the raw artifacts rather than silently represented as zero.

## Optimization Decisions

- Retained: bounded double-buffer I/O overlap (O00), per-layer hotset cache (O01), exact 128-token KV windowing (O05), six-worker dense projections/streamed output head (O06/O07), and the existing production arena/cache configuration.
- Rejected O02 approximate expert reuse: it produced absolute logit errors of 2.922 and 4.5823 in tested windows. Exact reuse had no net benefit and remains off. See [windowing.json](benchmarks/optimization/windowing.json).
- Rejected O03 cross-layer speculative prefetch: P02 first-choice accuracy was 40.38%; later reads were already hidden by O00, so the predictor was not retained. See [prefetch.json](benchmarks/optimization/prefetch.json).
- Rejected O04 expert arena v2: byte-correct, but end-to-end throughput fell to 0.99984x. See [repacking.json](benchmarks/optimization/repacking.json).
- Rejected fixed-performance mode: it lost the short affinity screen and was not retained. Sustained selection remains six workers, all-core affinity mask `ff`, `schedutil`. See [affinity_thermal.json](benchmarks/optimization/affinity_thermal.json).

## Classification And Limits

The roadmap performance thresholds are [proof, P0 >=0.25, P1 >=0.5, P2 >=1, stretch >=2 tok/s](docs/15_DECISION_GATES.md). The final result is proof-only. The service is correct and bounded, but not a practical interactive-speed deployment on this stock T606 configuration. The dominant measured bottleneck is CPU expert compute; the P02 analyzer measured roughly 248 ms per selected expert versus roughly 7 ms per cache-miss read. See [long_decode.json](benchmarks/proof/long_decode.json).

F01 used a fixed one-token request repeated for more than 30 minutes, which proves service stability and repeated full-stack decode but is not a continuous 128-token generation benchmark. F00 separately validates official Harmony, tiny numerical parity, native attention/MXFP4 primitives, protocol behavior and the fixed two-scenario end-to-end set. No performance improvement is claimed without same-device before/after evidence, and no rejected experiment is hidden.

## Raw Evidence Index

- [F00 correctness evidence](artifacts/runs/20260827T022000Z_F00/evidence.json)
- [F01 sustained evidence](artifacts/runs/20260827T022000Z_F01/evidence.json)
- [Final sustained benchmark](benchmarks/final/sustained_30m.json)
- [Optimization artifacts](benchmarks/optimization/)
- [Model and memory artifacts](artifacts/model/)
- [Pinned upstream locks](third_party/LOCK.yaml)

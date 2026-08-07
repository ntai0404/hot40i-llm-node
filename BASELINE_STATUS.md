# Delivery baseline status

**Release:** handoff v0.2.0  
**Prepared:** 2026-08-07  
**Purpose:** clean autonomous-agent handoff baseline; no device progress is claimed by this archive.

## Verified in the packaged source tree

- Structured YAML/JSON and JSON Schemas parse successfully.
- Local Markdown links resolve.
- Roadmap is a closed acyclic DAG with **56 tasks** and **8 gates**; all mandatory tasks lead to `FINAL_DEPLOYMENT`.
- Requirements traceability contains **24 requirements** and is generated from the machine-readable source.
- Upstream/research registry contains **12 code upstreams** and **22 research sources**.
- Python suite: **15 tests passed**.
- Host C++ Release build: passed.
- CTest: **1/1 passed** (`runtime_smoke`).
- `scripts/handoff_check.py` full validation: passed.
- Autonomous state is clean: **0/56 completed**, no task in progress, first ready task is `R00`.

## Intentionally not claimed

The physical Hot 40i has not yet been measured by this archive. Therefore actual RAM budget, storage controller/throughput, thermal behavior, ARM64 runtime behavior, model correctness on-device, gpt-oss first-token generation, expert-streaming benefit, and final decode performance remain unproven until their roadmap tasks produce evidence.

## Upstream lock state

`third_party/LOCK.yaml` intentionally contains unresolved immutable commits at handoff. Seed refs are hints only. `R00` must resolve and record full immutable commit SHAs before any upstream is treated as reproducible input. This avoids shipping unverified or fabricated commit pins.

## Primary execution path

The mandatory path is **stock Android + native ARM64 runtime + USB ADB forwarding**. Bootloader unlock, destructive flashing, or minimal-Linux work is optional and cannot be activated without explicit authorization and the recovery gate.

## Integrity

`BASELINE_SHA256SUMS.txt` contains SHA-256 hashes for delivery files, excluding the checksum file itself and generated/runtime directories. Build outputs, Python caches, downloaded upstream source, model weights, and device artifacts are intentionally excluded from the release archive.

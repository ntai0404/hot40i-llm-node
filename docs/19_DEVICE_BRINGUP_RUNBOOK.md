# Device bring-up runbook — D00 to D06

> `roadmap/tasks.yaml` is canonical. This runbook explains the intended implementation/measurement sequence and expected artifacts without pre-marking any task complete.

## D00 — establish USB-C ADB transport

Preconditions:

- stock Android is running;
- USB debugging is enabled by the user;
- the host has current Android platform-tools;
- the user has accepted the device-side RSA authorization dialog.

Start with:

```bash
hot40 doctor
hot40 devices
adb devices -l
```

Capture host OS, ADB version, device serial/state and USB connection evidence. Do not infer that a listed `unauthorized` device is ready; that is a physical/user authorization blocker.

Preferred development transport remains ADB over the C-to-C cable. Later service tasks use `adb forward` rather than Wi-Fi so network variability does not contaminate inference latency.

## D01 — normalized manifest

Run the repository read-only probes and persist both raw and normalized data. Important evidence includes:

- complete `getprop` output/build fingerprint;
- `/proc/cpuinfo`;
- `/proc/meminfo`;
- partitions and mounts;
- `/dev/block/by-name` listing where readable;
- block-device sysfs metadata;
- thermal zones;
- CPU online/frequency sysfs.

Do not convert missing permission into a guessed value. Record `UNKNOWN`/probe stderr.

Validate the normalized artifact with `schemas/device_manifest.schema.json`.

## D02 — safe RSS budget

The budget is not `MemTotal`. Measure at minimum:

1. immediately after a clean reboot/settle period;
2. idle with screen on;
3. idle with screen off;
4. representative native process allocations in increasing steps;
5. Android memory pressure/reclaim/zram behavior.

Track:

- `MemTotal`, `MemAvailable`, `Cached`, swap/zram;
- process RSS/PSS where obtainable;
- LMKD/OOM events if exposed;
- whether allocations trigger major reclaim or app/process death.

Choose a conservative `safe_rss_budget_bytes` with safety headroom and record rationale in project state via `taskctl decide`. The runtime later treats this as a hard ceiling.

## D03 — storage identity

Do not rely on a product webpage for the exact retail variant. Gather evidence from Android/sysfs/udev-visible data. Classify UFS/eMMC only when the device exposes convincing evidence; otherwise keep the type unknown and rely on measured access behavior.

Record filesystem/mount used for model storage because filesystem and free-space conditions can affect results.

## D04 — expert-shaped storage benchmark

The benchmark should include:

- 4 KiB, 64 KiB, 256 KiB, 1 MiB random blocks;
- 8 MiB, 16 MiB, 32 MiB expert-like regions;
- sequential controls;
- deterministic random seed;
- at least several repeated samples;
- p50/p95/p99 latency plus throughput;
- warm cache and cold-ish/file-cache-controlled runs when safely possible.

Do not use destructive block-device raw I/O. Benchmark a sufficiently large ordinary file on the same filesystem intended for model storage.

Estimate a first-order lower bound:

```text
I/O_seconds_per_token ≈ uncached_expert_bytes_per_token / measured_effective_read_Bps
```

and explicitly state why random/expert-shaped effective bandwidth, not marketing sequential bandwidth, is relevant.

## D05 — CPU/DRAM

Measure independent kernels before full inference:

- memcpy/read stream bandwidth;
- representative FP16/BF16/INT8/INT4/MXFP4-like matvec paths where available;
- big-core only vs mixed cores vs all cores;
- 1/2/4/8 thread scaling.

The T606's heterogeneous cores and memory bandwidth may make “all cores” slower after contention/thermal effects. Treat thread count as an experiment.

## D06 — thermal harness

Create a reusable sampler that records timestamps, thermal-zone values, CPU frequencies and battery/current data where available without elevated-risk modifications.

Run a sustained CPU/memory workload long enough to expose frequency decay. This establishes the baseline needed to interpret later tok/s over time.

## Hardware characterization gate

`HARDWARE_CHARACTERIZED` is not a spec-sheet gate. It passes only after D02/D04/D05/D06 produce evidence from the actual connected device.

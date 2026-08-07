# Benchmark specification

All performance comparisons must use the same physical phone, cooling/charging condition where practical, model/quant/context/prompt, thread/core configuration and runtime restart policy. If a field differs, report it.

## Required run metadata
- timestamp/timezone and Git commit;
- device serial/build fingerprint/thermal state;
- upstream/backend refs;
- model checksum/H40M manifest checksum;
- memory budget/context/threads/affinity;
- command argv and exit code.

## Storage
Measure sequential and deterministic random access across 4 KiB, 64 KiB, 256 KiB, 1 MiB plus expert-shaped 8/16/32 MiB regions. Report throughput and p50/p95/p99 latency. Separate warm page-cache from cold-ish/file-cache-controlled runs when the platform allows it safely.

## Official-model decode
Report TTFT, prompt tok/s, decode tok/s, peak RSS, configured safe budget, flash bytes/token, read ops/token, cache hit/miss/evictions, useful/wasted prefetch bytes when present, I/O wait, compute time, CPU frequency and thermal series.

## Sustained results
A peak 10-second number cannot be the final headline. Final acceptance uses >=30 minutes sustained service/inference.

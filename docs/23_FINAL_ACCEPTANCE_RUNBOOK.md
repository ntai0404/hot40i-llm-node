# Final acceptance runbook — F00 to F03

## F00 — correctness regression

Run all tiny/golden/operator and official prompt/tokenization fixtures against the optimized runtime. No unexplained divergence may be waived because performance improved.

Capture a machine-readable final correctness result linked to raw artifacts.

## F01 — >=30 minute sustained deployment

Start from a clean device/runtime restart and establish the final USB forwarding/service path. Use a fixed workload/request set for at least 30 minutes.

Collect time series sufficient to show:

- decode tok/s and TTFT;
- peak/current RSS;
- flash bytes/token and read ops/token;
- cache hits/misses/evictions;
- useful/wasted prefetch bytes when enabled;
- I/O wait vs compute time;
- CPU frequency;
- thermal values;
- battery/current where safely available;
- request failures/restarts.

The run fails if the service crashes, OOMs, grows memory without bound or silently drops correctness.

## F02 — final report

`FINAL_REPORT.md` must allow another engineer to reconstruct:

- exact phone/build and USB path;
- Git commit;
- locked upstream refs;
- official model/checkpoint checksum;
- H40M checksum/config;
- compute backend;
- dense/expert quantization;
- memory budget/arena split;
- context/thread/affinity settings;
- cache/prefetch/layout policy;
- final metrics and performance class;
- measured bottleneck;
- negative/rejected experiments.

Every headline number links to a raw machine-readable artifact.

## F03 — gate audit

Run:

```bash
python scripts/handoff_check.py
python scripts/taskctl.py status
```

Then verify the clean-restart USB health and one inference request and confirm every mandatory predecessor has a recorded evidence bundle.

After F03 itself is passed with evidence:

```bash
python scripts/taskctl.py gate FINAL_DEPLOYMENT
```

Only a PASS from that command plus the required service evidence permits the project to be declared complete.

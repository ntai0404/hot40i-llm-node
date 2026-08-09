# ADR 002: Output Head Strategy

## Status

Accepted for initial runtime implementation.

## Decision

Use a chunked streamed Q8 output head as the default exact vocabulary projection path.

## Evidence

- BF16 `lm_head.weight` is 1,158,266,880 bytes; Q8 estimate is 579,133,440 bytes.
- Safe RSS budget is 646,080,512 bytes, leaving only 66,947,072 bytes if Q8 is fully resident.
- Tiny fixture Q8 output-head-only max absolute logit drift is 0.004736612.
- D04 random 8 MiB read median is 1151.15 MiB/s.
- D05 best measured INT8 matvec throughput is 38.16 GOPS.
- Chunked Q8 uses 4,096-vocab chunks (11,796,480 bytes resident) and scans 579,133,440 bytes/token.

## Consequences

Resident Q8 remains a future option only if later memory planning proves enough headroom. The initial path favors correctness and bounded RSS over output-head latency.

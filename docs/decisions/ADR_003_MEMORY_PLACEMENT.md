# ADR 003: Memory Placement

## Status

Accepted for initial runtime implementation.

## Decision

Use layer-streamed Q8 attention matrices with resident small shared tensors. Keep router, normalization, attention
bias and attention sink tensors resident; keep embedding and output projection bounded by their prior row/chunk
policies; budget only a minimum top-4 expert cache until trace-driven cache work proves a larger allocation.

## Evidence

- D02 safe RSS budget is 646,080,512 bytes.
- Full BF16 attention matrices are 1,274,019,840 bytes.
- Q8 attention matrix storage is 637,009,920 bytes; keeping it all resident is rejected at 637,396,992 bytes before KV, experts, output, runtime or I/O buffers.
- Largest layer-local Q8 attention bundle is 26,542,080 bytes.
- M05 output head chunk is 11,796,480 bytes.
- M04 embedding cache is 46,080 bytes.
- Minimum top-4 expert cache is 52,945,920 bytes.
- The complete initial plan totals 331,306,112 bytes with 314,774,400 bytes of headroom.

## Consequences

The initial runtime must not allocate the whole attention stack, resident Q8 output head, or all experts at once.
P-stage implementation should stream layer attention bundles and expert slices through explicit buffers. O05 may
increase context only by spending this headroom with measured RSS evidence.

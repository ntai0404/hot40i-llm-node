# Model inspection and H40M conversion guide — M00 to M06

## M00 — inspect, never infer

The official pinned `gpt-oss-20b` checkpoint is the source for exact tensor names, shapes, dtypes, MXFP4 blocks/scales and byte counts. Produce a machine-readable inventory before designing final placement.

Inventory should include:

```text
logical tensor name
semantic role
layer
expert id (if any)
shape
dtype / quant representation
stored bytes
source shard/file
source checksum/range
```

Aggregate facts in the model card are a planning guide only.

## M01 — deterministic converter

A converter must be deterministic for:

```text
(source checkpoint checksum, converter commit, conversion config)
```

The H40M manifest records provenance. Re-running with identical inputs should produce identical tensor payload/checksums unless a documented nondeterministic step is unavoidable.

The converter must validate source shapes/types rather than silently accepting unexpected tensors from an upstream release.

## M02 — expert layout

Initial layout should prioritize simple contiguous per-layer expert arenas. Do not overfit co-access packing before real P/O traces exist.

Alignment must come from D04 measurements/platform requirements. Preserve MXFP4 block/scales correctly; repacking is not permission to alter numerical values.

## M03 — dense/shared quantization

OpenAI's released non-MoE tensors are BF16 in the official format. On 4 GB hardware, investigate lower-bit representations for dense/shared components one group at a time with golden regression.

Always report:

- byte saving;
- compute/kernel support;
- numerical/quality effect;
- runtime workspace change.

Do not choose a low-bit format that saves storage but requires dequantization workspace that breaks the memory budget.

## M04 — input embedding

Treat embeddings as row-addressable storage when feasible:

```text
token_id -> tensor row offset -> read/cache row
```

Cache common/recent rows only after measuring whether lookup I/O is material.

## M05 — output projection

Output/unembedding is structurally different from sparse experts because logits require broad vocabulary projection each decode step. Evaluate in order:

1. quantized resident head if it fits;
2. optimized backend representation;
3. chunked projection with bounded workspace;
4. hybrid placement only if measurement justifies it.

Streaming a huge output matrix from flash each token can erase all gains from MoE expert sparsity, so report its time/bytes separately.

## M06 — attention placement

Inventory exact attention weights and state cost. Evaluate keeping quantized attention resident because it is reused every token; compare byte cost with the safe budget before deciding.

Context configuration matters independently of weight placement. P/O phases start with small controlled context and scale after the official model is stable.

## Converter acceptance

Before S00 consumes H40M:

- schema validation passes;
- every file/range is inside bounds;
- checksums match;
- source provenance is complete;
- a random sample or full validation reconstructs reference tensors correctly;
- tiny/reference inference remains numerically consistent where conversion changed representation.

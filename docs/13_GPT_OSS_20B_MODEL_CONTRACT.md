# gpt-oss-20b model/memory contract

## Official aggregate facts

From OpenAI's published model card:

| Item | gpt-oss-20b |
|---|---:|
| Layers | 24 |
| Total parameters | 20.91B |
| Active parameters/token | 3.61B |
| MoE/MLP parameters | 19.12B |
| Attention parameters | 0.64B |
| Embed + Unembed | 1.16B |
| Experts/layer | 32 |
| Selected experts/token/layer | 4 |
| Checkpoint size | 12.8 GiB |
| Residual dimension | 2880 |
| Query heads | 64 × 64-dim |
| KV heads | 8 |
| Context support | up to 131,072 on dense layers in the published architecture |

MoE weights are post-trained in MXFP4 at 4.25 bits/parameter; the official repo describes packed fp4 blocks plus block scales. Other released tensors are BF16 unless a downstream conversion changes them.

Source: https://deploymentsafety.openai.com/gpt-oss/paperbench and https://github.com/openai/gpt-oss

## Important non-equivalence

`3.61B active parameters` does **not** imply a ~2 GB total RAM requirement. Active count includes work across layers and does not remove dense attention, unembedding/output projection, state/KV, runtime scratch or I/O buffers. OpenAI's release targets systems with substantially more memory than this phone; this project intentionally researches a different storage hierarchy.

## First-order expert traffic estimate

A rough uniform division of the 19.12B MoE parameters gives:

```text
19.12B / (24 layers * 32 experts) ~= 24.9M parameters per expert per layer
```

At 4.25 bits/parameter, one expert is roughly 13.2 MB decimal before implementation/layout overhead. Four selected experts are about 52.9 MB/layer. Across 24 layers, a zero-cache decode could therefore require on the order of 1.27 GB of expert payload per token.

This is a **planning lower-bound estimate**, not a checkpoint inventory. M00 must replace it with actual tensor bytes/scales/layout. The number explains why merely having 128 GB storage is irrelevant without cache/reuse/prefetch and sufficient flash bandwidth.

## Dense/shared pressure

Published aggregate counts imply 0.64B attention parameters and 1.16B combined embed+unembed parameters. At BF16 those aggregates are too large to casually keep all on a 4 GB Android device alongside model state/cache. Therefore M03–M06 must inspect exact tensors and evaluate selective quantization/placement.

Input embeddings have a structural advantage: inference needs the row(s) for the current token IDs, so token-row lookup can avoid full input-embedding residency. The output/unembedding projection generally needs the full vocabulary projection each decode step and must be benchmarked as its own bottleneck.

Do not assume input and output embeddings are tied or equally sized; M00 must inspect the actual checkpoint.

## Memory plan invariant

Final runtime startup must establish:

```text
safe_RSS_budget
  >= runtime + stacks
   + resident dense/shared tensors
   + KV/state
   + output-head allocation/working chunks
   + expert cache
   + I/O buffers
   + scratch
   + safety headroom
```

If this cannot be satisfied, runtime startup must fail cleanly. It may not fall back to unbounded swap.

## Correctness invariants

- router uses official top-4 selection and selected-expert weighting;
- official SwiGLU/clamping/residual behavior must match the reference implementation;
- attention/RoPE/GQA/sink behavior must match reference for tested context;
- Harmony formatting/tokenization must come from official OpenAI implementation/reference;
- prediction/prefetch may change **when** bytes arrive, never which router-selected expert result contributes to exact inference.

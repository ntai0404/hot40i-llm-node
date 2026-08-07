# System thesis and success metrics

## Research thesis

The project tests a narrow claim: **checkpoint capacity and inference working-set capacity are different for sparse/MoE models**. A 12.8 GiB checkpoint does not necessarily require 12.8 GiB resident DRAM if the runtime can keep the unavoidable dense/shared state resident and explicitly load/cache only selected experts.

The project is not trying to prove that flash is “as good as RAM”. Flash bandwidth/latency is the central adversary.

## Decode pipeline target

```text
full checkpoint / H40M on flash
          |
          +--> token lookup (input embedding row)
          +--> resident/budgeted dense attention + output path
          +--> MoE router
                    |
                    v
               top-4 experts
                    |
             bounded cache lookup
                 /       \
              hit         miss -> aligned read -> cache slot
                 \       /
                  compute
                    |
              next layer/token
```

Later optimizations may overlap current compute with known/predicted next reads, but prediction must never change the actual router-selected experts used for exact inference.

## Primary equations

Let `Bsafe` be the measured safe process budget on this phone.

```text
Bsafe >= Bruntime + Bdense_resident + Bstate/KV + Bexpert_cache + BIO_buffers + Bscratch + Bheadroom
```

No component may assume the leftover is “free RAM”. Each region is budgeted and observable.

For token `t`:

```text
FlashBytes(t) = expert_miss_bytes(t)
              + streamed_dense_bytes(t)
              + token_lookup_bytes(t)
              + wasted_prefetch_bytes(t)
```

The primary storage objective is to minimize `FlashBytes/token` and the unhidden portion of its latency while respecting correctness and RAM limits.

## Required metrics

At minimum every official-model run records:
- TTFT and prompt/decode tok/s;
- peak/current RSS and configured safe budget;
- flash bytes/token, read operations and p50/p95/p99 latency where available;
- expert cache hit/miss/eviction and bytes loaded;
- useful and wasted prefetch bytes after prefetch exists;
- I/O wait versus compute time from traces;
- CPU frequency/core affinity;
- thermal zones and battery current/state when readable.

## Anti-goals

- A model file merely mmap'ing successfully is not success.
- Swap/zram thrashing is not a substitute for bounded working-set design.
- A one-token demo is an intermediate milestone, not final deployment.
- Aggressive 1–2 bit quantization that destroys the “xịn” model objective is not an automatic win.
- Porting Linux before proving the algorithm on stock Android is not the primary plan.

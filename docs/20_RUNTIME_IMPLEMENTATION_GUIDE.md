# Runtime implementation guide — C/S/O phases

> This guide defines subsystem boundaries and invariants. Exact implementation tasks remain canonical in `roadmap/tasks.yaml`.

## 1. Layering

```text
Model semantics
      |
Graph / ModelRuntime
      |
MoE scheduler ----------------------- TraceSink
      |                                   ^
      +---- ComputeBackend                |
      |
      +---- MemoryPlan / PlacementManager |
      |          |                        |
      |          +-- resident arena       |
      |          +-- KV/state             |
      |          +-- expert cache --------+
      |
      +---- TensorStore / ModelIndex
                 |
                 +-- FlashTensorProvider
                 +-- embedding row lookup
                 +-- prefetch queue
```

Do not let backend kernel code decide storage placement, and do not let storage code implement gpt-oss math.

## 2. Proposed interface contracts

The existing scaffold exposes only a subset. Later tasks should evolve toward contracts equivalent to:

```cpp
struct MemoryBudget {
    size_t process_limit;
    size_t resident_limit;
    size_t cache_limit;
    size_t io_limit;
    size_t scratch_limit;
};

class ComputeBackend {
public:
    virtual RouterResult router(...) = 0;
    virtual void moe_expert(...) = 0;
    virtual void attention(...) = 0;
    virtual void output_projection(...) = 0;
};

class TensorStore {
public:
    virtual TensorView resident(TensorId) = 0;
    virtual void read(TensorId, MutableBytes) = 0;
    virtual PrefetchTicket prefetch(TensorId) = 0;
};

class TraceSink {
public:
    virtual void emit(const TraceEvent&) = 0;
};
```

Names may change; separation/invariants should not.

## 3. Memory invariants

- allocate large arenas once where practical;
- no expert-cache path may grow beyond its configured byte budget;
- model startup computes the full planned budget and fails early if it cannot fit;
- avoid per-token heap churn in final hot paths;
- keep safety headroom outside the runtime's nominal allocations;
- page cache/swap is not counted as controlled runtime memory.

The scaffold `ExpertCache` owns `std::vector` entries only as a disposable correctness prototype. S01/S03 replace this with fixed slots/arena ownership.

## 4. Storage invariants

- every final expert read maps to a known H40M tensor/range;
- validate file ID, offset, length and bounds before reading;
- trace bytes and latency by token/layer/expert;
- distinguish demand read from prefetch;
- do not hide read cost inside opaque model mmap;
- keep a generic mmap path only as a diagnostic baseline.

## 5. Correctness-first MoE schedule

Initial S05 path:

```text
router exact top-k
    -> lookup each selected expert
    -> cache hit or blocking demand read
    -> exact expert compute
    -> weighted combine
```

No prediction is needed to prove correctness. O00+ may overlap/prefetch but never change selected experts.

## 6. Prefetch state machine

A later prefetch entry should distinguish:

```text
ABSENT -> QUEUED -> READING -> READY -> CONSUMED
                         \-> CANCELLED/WASTED
```

Metrics must separate:

- demand bytes;
- useful prefetched bytes consumed by exact routing;
- prefetched bytes evicted/unused;
- demand wait hidden by overlap.

A predictor with a high hit rate can still be harmful if it wastes enough flash bandwidth to delay demand reads.

## 7. Concurrency

Start simple:

- one compute thread group;
- one bounded I/O worker/queue;
- double buffering.

Only increase concurrency after traces show idle I/O or compute. Mobile storage and memory bandwidth can degrade with excessive parallel reads/threads.

## 8. Backend integration

The selected B04 backend should receive already-resolved tensor views/bytes. Prefer adapting its proven kernels/operator code while keeping the project's storage scheduler above it.

Do not fork huge runtime subsystems unless required. Record every adapted upstream file/license/ref.

## 9. Observability

Trace timestamps should be monotonic (`steady_clock`/platform equivalent). Event volume must be configurable so profiling itself does not dominate decode. S04 defines canonical JSONL; later a compact binary trace may be added if profiling overhead is measured.

## 10. Optimization order

1. correctness/blocking demand-load baseline;
2. fixed cache;
3. trace real expert reuse;
4. double buffering / I/O-compute overlap;
5. cache policy experiments;
6. predictor/prefetch;
7. trace-guided physical repack;
8. dense/output and context/state tuning;
9. affinity/thermal tuning.

This order prevents optimizing a storage hypothesis before real access patterns exist.

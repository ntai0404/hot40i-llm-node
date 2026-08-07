# Architecture contract

```text
Laptop control plane
  Codex / optional Claude reviewer
  taskctl + evidence state
  device_lab + benchmarks + converters
  Harmony/Responses gateway
             |
             | USB-C / ADB forward (primary path)
             v
Phone process
  inference service
       |
       v
  ModelRuntime / GraphScheduler
        |              \
        |               \-> ComputeBackend (selected by B04)
        v
  MoEScheduler
        |
        +--> PlacementManager / MemoryPlan
        |       +--> Resident tensors
        |       +--> KV/state
        |       +--> ExpertCache (fixed arena slots)
        |
        +--> TensorStore / ModelIndex
                +--> token row lookup
                +--> FlashTensorProvider / prefetcher
                              |
                              v
                           H40M arena(s)
```

## Required separation

### Model semantics
Defines gpt-oss normalization, attention, router/top-k, experts, residuals and output behavior. It must not know whether an expert came from RAM or flash.

### ComputeBackend
Executes kernels on spans/tensors provided by the runtime. Backend choice is empirical. Storage policy must remain replaceable without rewriting model math.

### TensorStore / TensorProvider
Maps logical H40M tensors to physical ranges and performs observable reads. No component may silently mmap the entire model and call that the final storage design.

### PlacementManager / MemoryPlan
Owns the byte budget. It decides resident/cache/stream/token_lookup placement and must be able to prove the sum fits the measured safe budget.

### ExpertCache
Final implementation is arena/slab-backed, byte-bounded and observable. The original vector-backed cache is a prototype only.

### Prefetcher
Optimization layer. It may load bytes early, but never substitutes a predicted expert for the exact router-selected expert.

### TraceSink
Emits token/layer/router/cache/I/O/compute events. Optimization claims without traces/metrics are invalid.

## Host/device split
Harmony/Responses/tool orchestration remains on laptop initially to reduce phone memory/complexity. The phone's service is intentionally small. ADB TCP forwarding is the primary transport until/unless a later Linux path is justified.

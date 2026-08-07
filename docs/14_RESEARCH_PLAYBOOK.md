# Research playbook — what to borrow and what not to copy

## OpenAI gpt-oss

**Borrow directly:** architecture semantics, MXFP4 representation/reference, tokenizer/Harmony behavior, fixed model facts. The official implementation is the numerical reference.

**Do not assume:** the official memory requirement/runtime is optimized for a 4 GB ARM phone. This project adds a new storage/placement engine.

## Apple LLM in a Flash

**Core lesson:** design around the flash cost model. Reduce transferred bytes; turn scattered tiny reads into larger contiguous reads. Apple demonstrates reuse/windowing and row-column bundling for the sparse activation patterns studied in that work.

**Apply here as:** explicit H40M layout, aligned expert arenas, trace-driven reuse, larger coalesced reads where co-access statistics justify them.

**Do not copy blindly:** Apple's original contextual neuron sparsity assumptions are not the same as gpt-oss top-k MoE routing. The gpt-oss router already provides exact expert selection; the main opportunity is cache/prefetch/layout rather than approximating arbitrary dense neurons.

Source: https://machinelearning.apple.com/research/efficient-large-language

## PowerInfer-2

**Core lesson:** smartphone inference beyond DRAM can be structured around small schedulable units, an explicit storage engine, I/O/compute overlap and a segmented cache.

**Apply here as:** separate storage scheduler from compute backend; trace/cache by layer/expert; overlap aligned reads with current work.

**Do not assume:** its published 47B/11.68 tok/s smartphone result transfers to T606 or gpt-oss. Hardware/model differ dramatically.

Source: https://arxiv.org/abs/2406.06282

## ActiveFlow

**Core lesson:** for modern non-ReLU models, use cross-layer active-weight prediction/preload plus a DRAM/flash pipeline that explicitly allocates memory between hot cache, preloaded weights and current compute.

**Apply here as:** O03 predictor/prefetch and explicit cache/I/O buffer budget.

Source: https://arxiv.org/abs/2504.08378

## EdgeMoE

**Core lesson:** non-expert weights resident, expert weights external; predict/preload experts and consider expert-specific bit-width only with quality evidence.

**Apply here as:** initial placement hypothesis and prefetch structure. Do not introduce expert-wise re-quantization before exact baseline works.

Source: https://arxiv.org/abs/2308.14352

## MoE-Infinity

**Core lesson:** activation tracing, activation-aware caching and activation-aware prefetch are measurable, separable mechanisms.

**Apply here as:** S04 trace schema and O01/O03 experiments.

Source: https://github.com/EfficientMoE/MoE-Infinity

## SmallThinker

**Core lesson:** model/runtime co-design can move routing early (pre-attention router) to hide storage latency and reduce KV needs. This is a strong indication that storage latency is not always solvable by runtime tricks alone.

**Apply here as:** comparator/future-model option if gpt-oss routing timing fundamentally prevents sufficient prefetch. Do not change the primary target before gpt-oss feasibility is measured.

Source: https://arxiv.org/abs/2507.20984

## llama.cpp / MNN / ExecuTorch / MLC

These are kernel/backend candidates. Select using B00–B04 on the actual phone. The custom project owns storage placement/scheduling; it should avoid rewriting mature ARM kernels unless profiling proves a missing kernel is the bottleneck.

## PowerServe — mobile deployment reference
Use the open-source PowerServe project as a practical Android/HarmonyOS build/serving reference. Inspect its device deployment, threading, mobile backend integration and service mechanics where useful. Do not treat its supported-model list or flagship-device speed as evidence for gpt-oss on T606.

# Research and open-source references

Checked/selected August 2026.

## OpenAI — priority source
- `https://github.com/openai/gpt-oss` — official open-weight gpt-oss model/reference code. Use as model-semantics and MXFP4 correctness source.
- `https://github.com/openai/harmony` — official response/prompt format for gpt-oss. A custom inference stack must preserve Harmony behavior.
- `https://github.com/openai/codex` — primary laptop coding-agent/client reference.

## Apple — memory/runtime research priority
- Apple paper: *LLM in a Flash: Efficient Large Language Model Inference with Limited Memory* — selective flash loading, reuse/windowing and flash-aware access layout concepts.
- `https://github.com/ml-explore/mlx` — Apple ML research runtime reference. Do not attempt to port MLX wholesale to the T606; study its runtime/memory/quantization design selectively.
- `https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025` — additional Apple on-device quantization/KV-memory design context; Apple-silicon performance does not transfer directly to T606.

## Anthropic — agent workflow priority
- `https://github.com/anthropics/claude-code` — independent code review/terminal workflow reference.
- MCP is useful as a safe tool boundary for device experiments; destructive operations should not be exposed during non-destructive phases.

## Systems/inference references
- `https://github.com/SJTU-IPADS/PowerInfer` — activation locality; includes the project line that led to PowerInfer-2 smartphone inference work.
- `https://github.com/powerserve-project/PowerServe` — Android/HarmonyOS mobile serving/build reference; useful for deployment mechanics, not a gpt-oss correctness source.
- `https://github.com/ggml-org/llama.cpp` — ARM kernels/GGUF/gpt-oss baseline.
- `https://github.com/alibaba/MNN` — Android/mobile inference backend candidate.
- `https://github.com/pytorch/executorch` — XNNPACK/ARM backend candidate.
- `https://github.com/mlc-ai/mlc-llm` — Android/Mali backend candidate.

## How references are used
Do not copy architecture merely because a paper reports a high tokens/s number. Extract the mechanism, reproduce a local baseline, and validate it against the Hot 40i's actual RAM, flash access pattern, T606 CPU and thermal behavior.

## Background systems used to bound the design space
- FlexGen (ICML 2023) — dense CPU/GPU/disk offload background; useful to show why capacity can be solved while single-request latency remains bandwidth-bound.
- DeepSpeed ZeRO-Inference — layer/NVMe streaming background for dense models.
- DejaVu (ICML 2023) — contextual sparsity/predictor background that helps explain the lineage of selective activation approaches.

## Additional mobile/backend code references
- `https://github.com/alibaba/MNN` — Android CPU/Vulkan candidate.
- `https://github.com/pytorch/executorch` — XNNPACK/mobile operator candidate.
- `https://github.com/mlc-ai/mlc-llm` — Android/Mali build/runtime candidate.

# ADR 001: Compute Backend Selection

## Status

Accepted, generated from benchmark artifacts on 2026-08-09T07:17:23.683976+00:00.

## Decision

Use `llama.cpp` as the primary reusable compute-kernel base for the custom storage runtime.

Fallback order:

1. `ExecuTorch/XNNPACK` for isolated operator experiments where PTE export is acceptable.
2. `MNN` only if a complete MNN LLM template/model conversion path is provided.
3. `MLC/Mali` only after a complete pinned TVM/Relax Android toolchain is available.

## Evidence

| Backend | Runnable | Evidence fixture | Key result | Decision impact |
| --- | --- | --- | --- | --- |
| llama.cpp | yes | `stories260K.gguf` | median prompt 8609.739317 tok/s, median decode 2344.292633 tok/s, max RSS 10360 KiB | Selected primary backend |
| ExecuTorch/XNNPACK | yes | `linear_xnnpack_fp32.pte` | median 2.186502 ms per 1000 executions, max RSS 4196 KiB | Useful fallback for operator kernels, not the main LLM path |
| MNN | no | attempted tiny GGUF conversion | blocked: mnn_gguf_conversion_requires_existing_template_model | Not selected |
| MLC/Mali | no | Vulkan/OpenCL support probe plus MLC prepare attempt | Mali-G57 Vulkan/OpenCL present, MLC build rejected by missing TVM/toolchain package | Not selected |

## Rationale

The final runtime needs gpt-oss-compatible semantics, bounded RAM behavior, and a realistic Android path. `llama.cpp` is the only candidate that already executed an LLM-style GGUF fixture on the Hot 40i and exposes the relevant CPU kernel/runtime surface. Its Android build complexity is lower than building a complete TVM/MLC stack, and its model format path is closer to the project's current tiny GGUF evidence than ExecuTorch PTE or MNN template conversion.

`ExecuTorch/XNNPACK` remains valuable as a measured operator fallback. It demonstrated a working Android ARM64 build and successful XNNPACK-delegated execution, but the evidence is not an LLM benchmark and does not prove gpt-oss token/model semantics.

`MNN` and `MLC/Mali` are rejected for the current backend selection because their equivalent LLM execution paths are blocked by required model/template or TVM/toolchain inputs. The MLC GPU probe is still useful: it proves the Hot 40i exposes callable Mali-G57 Vulkan and OpenCL devices, so a future GPU revisit is possible.

## Consequences

Custom storage-runtime work should wrap or reuse llama.cpp/GGML-style CPU kernels first. GPU and alternate mobile runtimes should not block storage placement, paging, cache budgeting, or correctness work. Any later backend change must beat the llama.cpp baseline with same-device evidence and must include gpt-oss correctness coverage.

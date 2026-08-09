#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


llama = load("benchmarks/runtimes/llama_cpp.json")
mnn = load("benchmarks/runtimes/mnn.json")
executorch = load("benchmarks/runtimes/executorch.json")
mlc = load("benchmarks/runtimes/mlc.json")

rows = [
    {
        "backend": "llama.cpp",
        "status": "pass",
        "runnable": True,
        "fixture": "tinyllamas/stories260K.gguf",
        "llm_or_operator": "LLM/GGUF",
        "prompt_tokens_per_second_median": llama["metrics"].get("prompt_tokens_per_second_median"),
        "decode_tokens_per_second_median": llama["metrics"].get("decode_tokens_per_second_median"),
        "rss_peak_kib_max": llama["metrics"].get("rss_peak_kib_max"),
        "android_build_complexity": "moderate",
        "gpt_oss_mxfp4_adaptation_cost": "lowest",
        "kernel_coverage": "broad GGML/GGUF CPU kernels plus gpt-oss-oriented upstream support",
        "evidence": "benchmarks/runtimes/llama_cpp.json",
    },
    {
        "backend": "ExecuTorch/XNNPACK",
        "status": executorch.get("status"),
        "runnable": executorch.get("runnable"),
        "fixture": executorch["config"].get("model_description"),
        "llm_or_operator": "operator fixture",
        "duration_ms_median_per_1000": executorch["metrics"].get("duration_ms_median"),
        "single_execution_ms_median": executorch["metrics"].get("single_execution_ms_median"),
        "rss_peak_kib_max": executorch["metrics"].get("rss_peak_kib_max"),
        "android_build_complexity": "high",
        "gpt_oss_mxfp4_adaptation_cost": "high",
        "kernel_coverage": "strong FP32 mobile operator kernels; no direct GGUF/gpt-oss runtime path measured",
        "evidence": "benchmarks/runtimes/executorch.json",
    },
    {
        "backend": "MNN",
        "status": mnn.get("status"),
        "runnable": mnn.get("runnable"),
        "fixture": mnn["config"]["requested_equivalence"].get("fixture"),
        "llm_or_operator": "blocked LLM conversion",
        "rss_peak_kib_max": None,
        "android_build_complexity": "moderate",
        "gpt_oss_mxfp4_adaptation_cost": "high",
        "kernel_coverage": "mobile inference kernels available, but GGUF conversion blocked by required template model files",
        "evidence": "benchmarks/runtimes/mnn.json",
        "blocker": mnn.get("reason"),
    },
    {
        "backend": "MLC/Mali",
        "status": mlc.get("status"),
        "runnable": mlc.get("runnable"),
        "fixture": "native Vulkan/OpenCL support probe",
        "llm_or_operator": "support probe only",
        "rss_peak_kib_max": None,
        "android_build_complexity": "very high",
        "gpt_oss_mxfp4_adaptation_cost": "high",
        "kernel_coverage": "Mali Vulkan/OpenCL hardware present; MLC Android build blocked by missing TVM/toolchain package",
        "evidence": "benchmarks/runtimes/mlc.json",
        "vulkan_device_count": mlc["metrics"].get("vulkan_device_count"),
        "opencl_device_count": mlc["metrics"].get("opencl_device_count"),
    },
]

scores = {
    "llama.cpp": {
        "kernel_coverage": 5,
        "speed": 5,
        "rss": 4,
        "android_build_complexity": 4,
        "gpt_oss_mxfp4_adaptation_cost": 5,
        "total": 23,
    },
    "ExecuTorch/XNNPACK": {
        "kernel_coverage": 3,
        "speed": 3,
        "rss": 4,
        "android_build_complexity": 2,
        "gpt_oss_mxfp4_adaptation_cost": 2,
        "total": 14,
    },
    "MNN": {
        "kernel_coverage": 2,
        "speed": 0,
        "rss": 0,
        "android_build_complexity": 2,
        "gpt_oss_mxfp4_adaptation_cost": 1,
        "total": 5,
    },
    "MLC/Mali": {
        "kernel_coverage": 2,
        "speed": 0,
        "rss": 0,
        "android_build_complexity": 1,
        "gpt_oss_mxfp4_adaptation_cost": 1,
        "total": 4,
    },
}

comparison = {
    "schema_version": 1,
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "inputs": [
        "benchmarks/runtimes/llama_cpp.json",
        "benchmarks/runtimes/mnn.json",
        "benchmarks/runtimes/executorch.json",
        "benchmarks/runtimes/mlc.json",
    ],
    "selected_compute_backend": "llama.cpp",
    "fallback_order": ["ExecuTorch/XNNPACK", "MNN", "MLC/Mali"],
    "selection_rationale": [
        "llama.cpp is the only measured backend with a runnable LLM/GGUF fixture on the Hot 40i.",
        "llama.cpp produced three successful device runs with median prompt 8609.739317 tok/s, median decode 2344.292633 tok/s, and max RSS 10360 KiB on the tiny GGUF fixture.",
        "ExecuTorch/XNNPACK is runnable but only on a tiny linear operator fixture; using it as the primary backend would require a new model export/runtime integration path.",
        "MNN and MLC/Mali were not runnable for equivalent LLM inference in this workspace because their model/toolchain paths are blocked.",
    ],
    "rows": rows,
    "scores": scores,
}

(ROOT / "benchmarks/runtimes/comparison.json").write_text(
    json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
)

adr = f"""# ADR 001: Compute Backend Selection

## Status

Accepted, generated from benchmark artifacts on {comparison["generated_at"]}.

## Decision

Use `llama.cpp` as the primary reusable compute-kernel base for the custom storage runtime.

Fallback order:

1. `ExecuTorch/XNNPACK` for isolated operator experiments where PTE export is acceptable.
2. `MNN` only if a complete MNN LLM template/model conversion path is provided.
3. `MLC/Mali` only after a complete pinned TVM/Relax Android toolchain is available.

## Evidence

| Backend | Runnable | Evidence fixture | Key result | Decision impact |
| --- | --- | --- | --- | --- |
| llama.cpp | yes | `stories260K.gguf` | median prompt {llama["metrics"].get("prompt_tokens_per_second_median")} tok/s, median decode {llama["metrics"].get("decode_tokens_per_second_median")} tok/s, max RSS {llama["metrics"].get("rss_peak_kib_max")} KiB | Selected primary backend |
| ExecuTorch/XNNPACK | yes | `linear_xnnpack_fp32.pte` | median {executorch["metrics"].get("duration_ms_median")} ms per 1000 executions, max RSS {executorch["metrics"].get("rss_peak_kib_max")} KiB | Useful fallback for operator kernels, not the main LLM path |
| MNN | no | attempted tiny GGUF conversion | blocked: {mnn.get("reason")} | Not selected |
| MLC/Mali | no | Vulkan/OpenCL support probe plus MLC prepare attempt | Mali-G57 Vulkan/OpenCL present, MLC build rejected by missing TVM/toolchain package | Not selected |

## Rationale

The final runtime needs gpt-oss-compatible semantics, bounded RAM behavior, and a realistic Android path. `llama.cpp` is the only candidate that already executed an LLM-style GGUF fixture on the Hot 40i and exposes the relevant CPU kernel/runtime surface. Its Android build complexity is lower than building a complete TVM/MLC stack, and its model format path is closer to the project's current tiny GGUF evidence than ExecuTorch PTE or MNN template conversion.

`ExecuTorch/XNNPACK` remains valuable as a measured operator fallback. It demonstrated a working Android ARM64 build and successful XNNPACK-delegated execution, but the evidence is not an LLM benchmark and does not prove gpt-oss token/model semantics.

`MNN` and `MLC/Mali` are rejected for the current backend selection because their equivalent LLM execution paths are blocked by required model/template or TVM/toolchain inputs. The MLC GPU probe is still useful: it proves the Hot 40i exposes callable Mali-G57 Vulkan and OpenCL devices, so a future GPU revisit is possible.

## Consequences

Custom storage-runtime work should wrap or reuse llama.cpp/GGML-style CPU kernels first. GPU and alternate mobile runtimes should not block storage placement, paging, cache budgeting, or correctness work. Any later backend change must beat the llama.cpp baseline with same-device evidence and must include gpt-oss correctness coverage.
"""

adr_path = ROOT / "docs/decisions/ADR_001_COMPUTE_BACKEND.md"
adr_path.parent.mkdir(parents=True, exist_ok=True)
adr_path.write_text(adr, encoding="utf-8")
print("wrote benchmarks/runtimes/comparison.json")
print("wrote docs/decisions/ADR_001_COMPUTE_BACKEND.md")

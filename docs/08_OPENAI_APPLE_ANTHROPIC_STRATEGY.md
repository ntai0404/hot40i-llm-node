# OpenAI / Apple / Anthropic strategy

## OpenAI — primary model and protocol source
`gpt-oss-20b` is the primary end model because its sparse MoE structure (21B/3.6B active class) creates a meaningful checkpoint-vs-working-set research target. Official gpt-oss code/model card define numerical semantics/MXFP4. Official Harmony defines the conversation/tool format. Codex is the preferred autonomous coding agent because this repo carries `AGENTS.md` + machine-readable task state.

## Apple — primary flash-memory research source
LLM-in-a-Flash supplies the key hardware-aware principles: minimize transferred bytes and improve flash access granularity/contiguity; its windowing/reuse and row-column bundling motivate H40M/cache/layout experiments. MLX is a runtime design reference, not an Android backend.

## Anthropic — independent agent/tooling source
Claude Code can serve as an independent reviewer when desired; root `CLAUDE.md` points it to the same binding contract. MCP is considered for a structured device-lab server, but the primary execution loop does not depend on MCP availability. Anthropic is not used as an inference-semantics source because Claude weights/runtime are not the target open-weight stack.

# Source-of-truth policy

## OpenAI — model/protocol authority

OpenAI's model card is authoritative for gpt-oss architecture facts. The official repository is authoritative for reference tensor/MXFP4 implementation details. OpenAI Harmony is authoritative for rendering/parsing the format the models were trained on.

As of the handoff research snapshot, OpenAI documents `gpt-oss-20b` as 24 layers, 20.91B total parameters, 3.61B active parameters, 32 experts with top-4 selected per token, a 12.8 GiB checkpoint, and MXFP4 quantization for MoE weights at 4.25 bits/parameter. Do not infer future checkpoint tensor details from these aggregate numbers; M00 inspects the actual pinned checkpoint.

Official sources:
- https://deploymentsafety.openai.com/gpt-oss/paperbench
- https://github.com/openai/gpt-oss
- https://github.com/openai/harmony
- https://github.com/openai/codex

## Apple — flash-aware systems authority/reference

Apple's LLM-in-a-Flash paper demonstrates a hardware-informed approach: reduce transferred bytes and prefer larger contiguous reads, using windowing/reuse and row-column bundling in its studied sparse models. It is a design reference, not a drop-in implementation for gpt-oss MoE.

- https://machinelearning.apple.com/research/efficient-large-language
- https://github.com/ml-explore/mlx

## Anthropic/MCP — agent tooling reference

Anthropic Claude Code is useful as an independent reviewer/agent workflow reference. MCP is useful if a device-tool server is added. Neither defines gpt-oss inference semantics.

- https://github.com/anthropics/claude-code
- https://github.com/modelcontextprotocol/modelcontextprotocol

## Systems research

See `research/sources.yaml`. PowerInfer-2, ActiveFlow, EdgeMoE, MoE-Infinity and SmallThinker each solve related but non-identical hardware/model problems. Treat reported speedups as inspiration, not expected Hot 40i performance.

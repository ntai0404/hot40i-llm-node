# Handoff — autonomous implementation contract

## Mission

Take this repository from its current research scaffold to an end-to-end USB-attached Hot 40i inference node running official OpenAI `gpt-oss-20b` semantics with a bounded RAM working set and explicitly managed flash-resident MoE experts. `PROJECT_SPEC.md` is the complete system specification; `roadmap/requirements.yaml` provides requirement-to-task traceability.

The coding agent is expected to execute the complete roadmap without being assigned each task manually.

## Current state (facts, not aspirations)

### Proven in this archive
- Host Python package/CLI exists.
- Conservative ADB inventory helpers exist.
- C++ `TensorProvider`, file-backed provider, basic RAM arena, model index and prototype LRU expert cache exist.
- Host C++ smoke test and Python tests can run.
- Roadmap/task/gate/evidence control plane is present.

### Not yet proven
- The user's physical Hot 40i has not been characterized by this repository.
- Actual usable RAM, flash controller and expert-shaped storage throughput are unknown until D00–D06 run.
- Android ARM64 build/runtime has not been proven on the actual device.
- No backend has been selected from measurements yet.
- Tiny gpt-oss-shaped numerical parity is not yet proven.
- Official `gpt-oss-20b` checkpoint has not yet produced a token on the phone.
- Flash streaming, expert prefetch and practical decode performance remain hypotheses until measured.

Never convert an item from “not proven” to “proven” without evidence artifacts.

## Final Definition of Done

`FINAL_DEPLOYMENT` passes only when all mandatory criteria in `roadmap/gates.yaml` pass. In short:

1. official gpt-oss model semantics and Harmony-compatible prompt/render/parse behavior;
2. official model checkpoint or deterministically derived H40M artifact with provenance/checksums;
3. bounded process memory using the measured device budget;
4. explicit expert storage/cache metrics and no reliance on uncontrolled swap thrashing;
5. stable end-to-end generation;
6. USB laptop→phone service path;
7. 30-minute sustained test and final evidence report.

## Autonomous execution rules

Run `python scripts/taskctl.py next`; take the highest-priority ready task; finish it; verify it; create evidence; mark it passed; continue immediately. A task is not passable unless its exact declared verification entries and required artifacts are present; `taskctl` enforces this.

Do not ask the user to choose implementation details already resolvable by benchmark/research. Use evidence and make the decision. If one approach fails, record it and try the next bounded alternative defined by the task.

If the phone is temporarily absent or a physical authorization dialog blocks a device task, mark that task blocked and continue independent host/research/runtime tasks. Stop only if there are no ready tasks remaining.

## Primary path vs destructive OS path

The **primary target is stock Android + native ARM64 process + ADB USB forwarding**. This path is deliberately capable of reaching the final gate without flashing the phone.

Minimal Linux/bootloader work is optional and late. It is not permitted unless `PROJECT_STATE.yaml` explicitly records destructive-device authorization and the recovery gate has passed. The agent must never infer authorization from enthusiasm or earlier discussion.

## Source-of-truth hierarchy

1. OpenAI official gpt-oss/model card/source for model semantics and MXFP4.
2. OpenAI Harmony for conversation/render/parse behavior.
3. Actual Hot 40i measurements for hardware decisions.
4. Apple LLM-in-a-Flash for flash-aware transfer/layout ideas.
5. PowerInfer-2, ActiveFlow, EdgeMoE, MoE-Infinity, SmallThinker for out-of-core/MoE scheduling ideas.
6. llama.cpp/ggml, MNN, ExecuTorch/XNNPACK, MLC for reusable compute/backend primitives.
7. Anthropic Claude Code/MCP for agent/tool-control patterns, not for model inference semantics.

If two sources disagree, write the discrepancy into the task evidence and prefer the higher authority for the relevant domain.

## First commands

```bash
python scripts/handoff_check.py --quick
python scripts/taskctl.py status
python scripts/taskctl.py next
```

If there is no `.git` directory, initialize Git and commit the untouched handoff baseline before R00 implementation changes.

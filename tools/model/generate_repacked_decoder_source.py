#!/usr/bin/env python3
"""Generate a benchmark-only decoder source that can select a planned arena index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate(args: argparse.Namespace) -> int:
    layout = load_json(args.layout)
    records = {
        (int(record["layer"]), int(record["expert_id"])): record
        for record in layout["records"]
    }
    layer_count = int(layout["trace_provenance"]["layer_count"])
    expert_count = int(layout["trace_provenance"]["expert_count"])
    offsets = [
        int(records[(layer, expert)]["offset"])
        for layer in range(layer_count)
        for expert in range(expert_count)
    ]
    if len(offsets) != layer_count * expert_count:
        raise ValueError("layout mapping is incomplete")
    values = ",\n        ".join(str(offset) + "ULL" for offset in offsets)
    replacement = f"""h40::ModelIndex build_expert_index() {{
    static constexpr std::uint64_t kRepackedOffsets[kLayers * kExperts] = {{
        {values}
    }};
    const char* layout = std::getenv("H40_EXPERT_LAYOUT");
    const bool use_repacked = layout != nullptr && std::string_view(layout) == "v2";
    h40::ModelIndex index;
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {{
        for (std::uint32_t expert = 0; expert < kExperts; ++expert) {{
            const auto ordinal = static_cast<std::uint64_t>(layer) * kExperts + expert;
            const auto offset = use_repacked ? kRepackedOffsets[ordinal] : ordinal * kExpertStrideBytes;
            index.put({{layer, expert}}, {{offset, kExpertPayloadBytes}});
        }}
    }}
    return index;
}}"""

    source = args.source.read_text(encoding="utf-8")
    start_marker = "h40::ModelIndex build_expert_index() {"
    end_marker = "\n\nh40::GptOssExpertView expert_view"
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0:
        raise ValueError("canonical build_expert_index function was not found")
    generated = source[:start] + replacement + source[end:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    return len(offsets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = generate(args)
    print(json.dumps({"status": "pass", "mapping_entries": entries}, sort_keys=True))


if __name__ == "__main__":
    main()

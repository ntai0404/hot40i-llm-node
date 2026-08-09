from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAYOUT = ROOT / "artifacts/model/h40m/expert_arena.layout.json"
ARENA = ROOT / "artifacts/model/h40m/expert_arena.bin"
OUT_BIN = ROOT / "artifacts/runs/20260809T093047Z_S02/h40m_sampled_experts.bin"
OUT_JSON = ROOT / "artifacts/runs/20260809T093047Z_S02/h40m_sampled_experts.json"
SAMPLE_RECORDS = [0, 257, 767]


def main() -> None:
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    records = layout["records"]
    selected = [records[index] for index in SAMPLE_RECORDS]
    copied = []
    offset = 0
    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    with ARENA.open("rb") as source, OUT_BIN.open("wb") as output:
        for record in selected:
            source.seek(record["offset"])
            remaining = record["length"]
            while remaining:
                chunk = source.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError("unexpected EOF while copying sampled expert")
                output.write(chunk)
                remaining -= len(chunk)
            copied.append(
                {
                    "layer": record["layer"],
                    "expert_id": record["expert_id"],
                    "original_offset": record["offset"],
                    "sample_offset": offset,
                    "length": record["length"],
                    "sha256": record["sha256"],
                }
            )
            offset += record["length"]
    OUT_JSON.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_layout": str(LAYOUT.relative_to(ROOT)),
                "source_arena": str(ARENA.relative_to(ROOT)),
                "sample_file": str(OUT_BIN.relative_to(ROOT)),
                "record_count": len(copied),
                "bytes": offset,
                "records": copied,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"record_count": len(copied), "bytes": offset}, separators=(",", ":")))


if __name__ == "__main__":
    main()

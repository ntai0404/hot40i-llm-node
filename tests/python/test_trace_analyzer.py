import json
import subprocess
import sys


def test_trace_analyzer_uses_canonical_route_event(tmp_path):
    trace = tmp_path / "trace.jsonl"
    rows = [
        {"schema_version": 1, "ts_ns": 0, "event": "token_begin", "token": 0},
        {"schema_version": 1, "ts_ns": 1, "event": "route", "token": 0, "layer": 0, "expert": 2},
        {"schema_version": 1, "ts_ns": 2, "event": "route", "token": 0, "layer": 0, "expert": 4},
        {"schema_version": 1, "ts_ns": 3, "event": "cache_miss", "token": 0, "layer": 0, "expert": 2},
        {"schema_version": 1, "ts_ns": 4, "event": "read_end", "token": 0, "layer": 0, "expert": 2, "bytes": 128},
        {"schema_version": 1, "ts_ns": 5, "event": "cache_hit", "token": 0, "layer": 0, "expert": 4, "cache_hit": True},
        {"schema_version": 1, "ts_ns": 6, "event": "token_end", "token": 0},
        {"schema_version": 1, "ts_ns": 7, "event": "route", "token": 1, "layer": 0, "expert": 2},
        {"schema_version": 1, "ts_ns": 8, "event": "cache_miss", "token": 1, "layer": 0, "expert": 2},
        {"schema_version": 1, "ts_ns": 9, "event": "read_end", "token": 1, "layer": 0, "expert": 2, "bytes": 256},
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = subprocess.run(
        [sys.executable, "tools/trace/analyze_experts.py", str(trace)],
        text=True,
        capture_output=True,
        check=True,
    )
    doc = json.loads(result.stdout)
    assert doc["tokens_seen"] == 2
    assert doc["route_events"] == 3
    assert doc["cache_lookups"] == 3
    assert doc["cache_hits"] == 1
    assert doc["cache_misses"] == 2
    assert doc["cache_hit_rate"] == 1 / 3
    assert doc["flash_bytes_total"] == 384
    assert doc["flash_bytes_per_token"] == {"0": 128, "1": 256}
    assert doc["top_experts"][0] == {"layer": 0, "expert": 2, "count": 2}

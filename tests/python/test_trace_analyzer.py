import json
import subprocess
import sys


def test_trace_analyzer_uses_canonical_route_event(tmp_path):
    trace = tmp_path / "trace.jsonl"
    rows = [
        {"schema_version": 1, "ts_ns": 1, "event": "route", "token": 0, "layer": 0, "expert": 2},
        {"schema_version": 1, "ts_ns": 2, "event": "route", "token": 0, "layer": 0, "expert": 4},
        {"schema_version": 1, "ts_ns": 3, "event": "route", "token": 1, "layer": 0, "expert": 2},
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
    assert doc["top_experts"][0] == {"layer": 0, "expert": 2, "count": 2}

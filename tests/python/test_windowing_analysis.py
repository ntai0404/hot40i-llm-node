from research.analyze_windowing import analyze


def test_window_reuse_upper_bound(tmp_path) -> None:
    trace = tmp_path / "routes.jsonl"
    rows = [
        {"event": "route", "token": 0, "layer": 0, "expert": 1},
        {"event": "compute_end", "token": 0, "layer": 0, "expert": 1},
        {"event": "route", "token": 1, "layer": 0, "expert": 1},
        {"event": "route", "token": 3, "layer": 0, "expert": 1},
    ]
    trace.write_text("".join(f"{__import__('json').dumps(row)}\n" for row in rows), encoding="utf-8")
    result = analyze(trace, [1, 2], expert_bytes=16)
    assert result["requests"] == 3
    assert result["windows"][0]["potential_reuses"] == 1
    assert result["windows"][1]["potential_reuses"] == 2
    assert result["windows"][1]["upper_bound_flash_bytes_saved"] == 32

from tools.trace.compare_cache_policies import PolicySimulation, compare


def test_policy_simulations_remain_bounded() -> None:
    requests = [(0, expert) for expert in (0, 1, 2, 0, 3, 0, 1, 0)]
    for name in ("lru", "lfu_decay", "per_layer_hotset"):
        policy = PolicySimulation(name, slots=2, decay_interval=4)
        for key in requests:
            policy.access(key)
            assert len(policy.resident) <= 2
        assert policy.result.hits + policy.result.misses == len(requests)
        assert policy.result.evictions == policy.result.misses - 2


def test_compare_reads_only_route_events(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"event":"route","layer":0,"expert":1}\n'
        '{"event":"compute_end","layer":0,"expert":1}\n'
        '{"event":"route","layer":0,"expert":1}\n',
        encoding="utf-8",
    )
    result = compare(trace, slots=1, expert_bytes=16, decay_interval=8)
    assert result["budget_bytes"] == 16
    assert all(row["requests"] == 2 for row in result["policies"])
    assert all(row["hits"] == 1 for row in result["policies"])

import json

from tools.trace.evaluate_prefetch_predictor import evaluate


def test_transition_predictor_uses_disjoint_training_split(tmp_path) -> None:
    trace = tmp_path / "routes.jsonl"
    rows = []
    for token in range(4):
        route = token % 2
        for layer, experts in ((0, (route, 3)), (1, (route, route + 4))):
            for expert in experts:
                rows.append({"event": "route", "token": token, "layer": layer, "expert": expert})
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = evaluate(trace, train_tokens=2, width=2, expert_bytes=16)
    assert result["evaluation_tokens"] == 2
    assert result["transition_counter_entries_max"] == 1024
    transition = next(row for row in result["predictors"] if row["predictor"] == "cross_layer_transition")
    assert transition["precision"] == 1.0
    assert transition["first_choice_accuracy"] == 1.0
    assert transition["wasted_prefetch_bytes"] == 0

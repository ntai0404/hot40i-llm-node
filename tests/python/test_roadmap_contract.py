from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]

def test_task_ids_unique_and_dependencies_exist():
    data=yaml.safe_load((ROOT/'roadmap/tasks.yaml').read_text())
    tasks=data['tasks']; ids=[t['id'] for t in tasks]; assert len(ids)==len(set(ids)); s=set(ids)
    for t in tasks:
        assert set(t.get('depends_on',[])) <= s
        for field in ['goal','implementation','verification','required_artifacts','pass_criteria','failure_policy']:
            assert t.get(field), (t['id'],field)

def test_final_gate_exists():
    gates=yaml.safe_load((ROOT/'roadmap/gates.yaml').read_text())['gates']
    assert any(g['id']=='FINAL_DEPLOYMENT' for g in gates)

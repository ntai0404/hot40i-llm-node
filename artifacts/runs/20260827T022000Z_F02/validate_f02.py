import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
report_path = ROOT / "FINAL_REPORT.md"
summary_path = ROOT / "benchmarks" / "final" / "summary.json"
sustained_path = ROOT / "benchmarks" / "final" / "sustained_30m.json"

report = report_path.read_text(encoding="utf-8")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
sustained = json.loads(sustained_path.read_text(encoding="utf-8"))

links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", report)
local_links = [link for link in links if not link.startswith(("http://", "https://", "#"))]
missing_links = [link for link in local_links if not (ROOT / link).exists()]
assert not missing_links, f"missing report links: {missing_links}"

for source in summary["sources"]:
    assert (ROOT / source).exists(), f"missing summary source: {source}"

assert summary["status"] == "pass"
assert sustained["status"] == "pass"
assert summary["performance"]["f01_wall_tok_s"] == sustained["throughput"]["tokens_per_second_wall"]
assert "0.03130 emitted tokens/second" in report
assert "1,885 seconds" in report
assert "proof-only" in report

print(f"report_local_links={len(local_links)}")
print(f"summary_sources={len(summary['sources'])}")
print("headline_metric_crosscheck=pass")
print("report_source_validation=pass")

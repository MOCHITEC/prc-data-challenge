"""Append completed measurements without regenerating the previous investigation report."""

import json
from pathlib import Path

root = Path("docs/codex_fe_results")
report = Path("docs/codex_feature_report.md")
marker = "\n### NEW — actual feature-refit measurements\n"
text = report.read_text().split(marker)[0]
lines = [
    marker,
    "Results below use actual feature additions and fresh fits, without residual corrections.",
    "",
    "| Comparison | ΔRMSE [day CI] | 7-day CI | Nov Δ | Dec Δ | Matched Δ | Unmatched Δ |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for filename, section in [("refit_evaluation.json", "confirmation"), ("continuation_evaluation.json", None)]:
    path = root / filename
    if not path.exists():
        continue
    results = json.loads(path.read_text())
    if section:
        results = results[section]
    for name, r in results.items():
        ci = ", ".join(f"{v:+.3f}" for v in r["ci"])
        week = ", ".join(f"{v:+.3f}" for v in r["week_ci"])
        lines.append(
            f"| {name} | {r['delta']:+.3f} [{ci}] | [{week}] | {r['by_month']['11']['delta']:+.3f} | {r['by_month']['12']['delta']:+.3f} | {r['matched']:+.3f} | {r['unmatched']:+.3f} |"
        )
lines += [
    "",
    "Selected/all bundle comparisons retain the original 98.75% intervals. Gap and training-size",
    "comparisons use descriptive 95% intervals: their design follows the already-seen confirmation",
    "outcomes. Seven-day intervals are a dependence sensitivity. These intervals omit training uncertainty.",
    "",
    "Completed fit metadata:",
    "",
]
for path in sorted(root.glob("refit_*.json")):
    d = json.loads(path.read_text())
    if "train_rows" in d:
        lines.append(
            f"- `{path.name}`: {d['train_rows']:,} training rows, {d['rounds_matched']}/{d['rounds_unmatched']} rounds, {d['seconds']:.1f} s."
        )
lines += [
    "",
    "Stage commit attempts remain blocked by the read-only git worktree index lock. Files are saved locally.",
    "",
]
report.write_text(text + "\n".join(lines))

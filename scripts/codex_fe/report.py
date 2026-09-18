"""Render numerical report directly from saved measurements."""

import json
from pathlib import Path

R = Path("docs/codex_fe_results")


def read(name):
    return json.loads((R / name).read_text())


def fmt(x):
    return f"{x:+.3f}"


def ci(r):
    return f"{fmt(r['delta'])} [{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]"


def main():
    screen = read("screen.json")
    confirm = read("confirmation.json")
    base = read("baseline.json")
    sel = read("selection.json")
    desc = read("candidates.json")
    out = [
        "# Independent feature investigation — 2026-09-18",
        "",
        (
            "All differences below are **candidate minus comparator RMSE, in seconds; negative is better**. "
            "Candidate screens measure residual corrections; feature refits are reported separately. No submission was created or uploaded."
        ),
        "",
        "## Scope and provenance",
        "",
        (
            f"Generated **{sel['novel_count']} candidates**, four hash controls, an airport-only control and three existing-feature controls. "
            "The 63 comprise 62 nonconstant training columns plus the constant schedule-second diagnostic. "
            "None of the ten rejected features in brief §7 was rerun. Arrival continuity overlaps the prior audit, "
            "but is retested because that audit did not fit these features in a fresh baseline or validate their corrections within one model. "
            "No operator target encoding, planned duration, distance, queue-ahead or bank-position candidate was included."
        ),
        "",
        (
            "Read the [brief](CODEX_BRIEF.md), [instrument audit](investigation_2026-09-16/results/decision-rule-blind-to-tail.md), "
            "[same-model refutation](data_audit_2026-09-18/results/review-refutations.md), "
            "[arrival audit](data_audit_2026-09-18/results/arrival-rows.md), "
            "[timestamp audit](data_audit_2026-09-18/results/timestamps.md), "
            "[precision investigation](investigation_2026-09-16/results/timestamp-precision-clusters.md), and production code/configuration."
        ),
        "",
        (
            "The existing read-only feature cache has 2,085,047 training and 344,841 ranking departures. "
            "Its recorded feature-source fingerprint `590ceb8f8170ff6b` equals the current feature-code hash. "
            "New features are separate keyed parquet files in `outputs/codex_fe/`; production source and `configs/main.yaml` are unchanged. "
            "All experiment writes stay in this worktree, all LightGBM jobs are serial and use two threads; Polars uses two threads."
        ),
        "",
        "## Protocol and its limits",
        "",
        (
            "The [pre-outcome protocol](codex_fe_results/protocol.md) and [configuration](../configs/codex_fe_screen.yaml) "
            "specify a fresh January–August model predicting September–December. August chooses stopping rounds; "
            "the model is refitted on January–August. **Every calibration and evaluation residual belongs to this same fit.** "
            "The mixed historical OOF file is never loaded."
        ),
        "",
        (
            f"Baseline training: {base['train_rows']:,} departures; {base['rounds_matched']} matched rounds, "
            f"{base['rounds_unmatched']} unmatched rounds; {base['seconds']:.1f} seconds. One seed, full column sampling, "
            "two model regimes, LIRF unmatched schedule reparameterisation. This is a deliberately controlled experiment, "
            "not an exact reproduction of the three-seed production model."
        ),
        "",
        (
            "1. Fit airport × regime × candidate-cell residual means on September. Numeric cells use September-only deciles "
            "(exact levels for ≤20 values); categorical cells use values. Add 100 zero-residual pseudo-observations per cell; "
            "unseen cells get zero. Evaluate every candidate on all 185,674 October departures."
        ),
        (
            "2. Keep at most five novel candidates with October 95% upper CI < 0, a matched-row improvement, and a point "
            "improvement greater than every hash control and airport-only correction. Save selection before evaluating November–December."
        ),
        (
            "3. Refit selected corrections on September+October and confirm on all 328,009 November–December departures. "
            "A separately preregistered shallow joint learner tests interactions across all candidates against existing context "
            "and context plus four hashes. It uses 15 leaves, 120 rounds, learning rate .03 and no evaluation early stopping."
        ),
        (
            "4. Refit the actual January–August baseline with the selected bundle, all candidates, and a hash control. "
            "Keep baseline round counts and seed fixed. No correction is transferred across these refits."
        ),
        "",
        (
            "Intervals resample **whole calendar days**, paired between predictions, preserving within-day airport/traffic dependence. "
            "October intervals use 2,000 draws and are exploratory, with no familywise claim. Confirmation uses 10,000 draws "
            f"and {100 * (1 - confirm['alpha']):.3f}% intervals (Bonferroni for five selected scalars plus the preregistered joint test). "
            "Feature-refit intervals use 98.75% coverage for two bundles × two comparators. Seven-day circular moving-block "
            "intervals are an additional sensitivity analysis. These are conditional test-sample intervals, not model-training "
            "uncertainty or a guarantee about another season/year."
        ),
        "",
        (
            "Full RMSE is primary. Matched/unmatched results, month and airport breakdowns, SSE changes and tail counts "
            "are reported without deleting extremes. The “drop200” diagnostic excludes a fixed set of the baseline’s worst "
            "200 rows, rather than choosing a different set for each candidate; it never determines acceptance. "
            "There are only two confirmation months, so a month-bootstrap is not presented as reliable uncertainty."
        ),
        "",
        (
            "Limits: no local fold reproduces 12 training months → January/July 2026. The existing trainer fits encoding "
            "tables before its inner stopping split, so inner-holdout encodings include that holdout; outer evaluation labels "
            "never enter fitting. Fixed round counts in verification prevent candidate-specific stopping from confounding "
            "feature additions. A univariate binned correction can miss conditional signal; the joint screen and full "
            "all-feature refit address that limitation but do not exhaust all possible feature representations."
        ),
        "",
        "## Candidate rationale and complete October screen",
        "",
        (
            "Hypotheses, not measured facts: clock revisions may distinguish recording regimes; precision and identity "
            "fragments may expose reporting conventions; stand continuity may constrain available turnaround time; "
            "runway/operator histories may describe local proxy-clock behaviour; traffic and weather derivatives may "
            "capture changing operating conditions. The table measures their incremental residual-correction value."
        ),
        "",
        "| Candidate | Full ΔRMSE [95% day CI] | Matched Δ | Unmatched Δ | Drop200 Δ | Selected |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for r in screen:
        out.append(
            f"| `{r['candidate']}` | {ci(r)} | {fmt(r['matched'])} | {fmt(r['unmatched'])} | {fmt(r['baseline_worst200_removed'])} | {'yes' if r['candidate'] in sel['selected'] else ''} |"
        )
    out += [
        "",
        (
            "One hash control has a nominally significant October improvement. This is measured evidence that "
            "a single screen CI is insufficient. The screen is not a declaration that all nonselected features are useless."
        ),
        "",
        "### Joint October screen",
        "",
        "| Arm | Full ΔRMSE [95% day CI] | Matched Δ | Unmatched Δ |",
        "|---|---:|---:|---:|",
    ]
    for r in read("joint_screen.json"):
        out.append(f"| {r['candidate']} | {ci(r)} | {fmt(r['matched'])} | {fmt(r['unmatched'])} |")
    out += [
        "",
        "## Locked November–December correction confirmation",
        "",
        "| Correction | Full ΔRMSE [adjusted day CI] | November Δ | December Δ | Matched Δ | Unmatched Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in confirm["results"]:
        out.append(
            f"| `{r['candidate']}` | {ci(r)} | {fmt(r['by_month']['11']['delta'])} | {fmt(r['by_month']['12']['delta'])} | {fmt(r['matched'])} | {fmt(r['unmatched'])} |"
        )
    joint = next(x for x in confirm["results"] if x["candidate"] == "joint_novel")
    out += [
        "",
        (
            f"Baseline RMSE is **{joint['baseline']:.3f}**; joint-correction RMSE is **{joint['rmse']:.3f}**. "
            f"Joint novel versus context-only: {ci(joint['vs_joint_context'])}; versus context plus hashes: {ci(joint['vs_joint_noise'])}. "
            "These comparator intervals use the same adjusted confidence level and are supporting comparisons, not "
            "additional independently controlled discoveries. Airport-only and noise rows are controls, not selected hypotheses."
        ),
        "",
        (
            "Only `stand_arrival_gap` among the five selected scalar corrections retains an adjusted CI below zero. "
            "The joint correction improves in both months and all ten airports. This supports recoverable signal "
            "within the fixed baseline; it does **not** authorize attaching this correction to a model trained on all 12 months."
        ),
        "",
    ]
    ref = R / "refit_evaluation.json"
    if ref.exists():
        measurements = read("refit_evaluation.json")["confirmation"]
        out += [
            "## Actual feature-addition refits",
            "",
            "| Comparison | Full ΔRMSE [98.75% day CI] | 98.75% 7-day CI | Nov Δ | Dec Δ | Matched Δ | Unmatched Δ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, r in measurements.items():
            out.append(
                f"| {name} | {ci(r)} | [{fmt(r['week_ci'][0])}, {fmt(r['week_ci'][1])}] | {fmt(r['by_month']['11']['delta'])} | {fmt(r['by_month']['12']['delta'])} | {fmt(r['matched'])} | {fmt(r['unmatched'])} |"
            )
        out += [
            "",
            (
                "Tail and airport detail, row-level largest error changes, and October refit scores are in "
                "[refit_evaluation.json](codex_fe_results/refit_evaluation.json). These are the comparisons that measure "
                "adding columns to the original training procedure."
            ),
            "",
        ]
    out += [
        "## Availability and validation",
        "",
        (
            "Candidate generation passes a departure-label-masking test: replacing all departure labels and hidden "
            "block timestamps with nulls leaves candidates identical. Arrival joins use only arrivals already on-block "
            "before the known departure take-off, capped at six hours; they are candidate links, not guaranteed aircraft "
            "rotations. Neighbour histories exclude the current departure with a 60-second lag. Train/rank schemas, "
            "ID uniqueness, unseen-cell behaviour, train-only bin boundaries and exact paired-bootstrap arithmetic pass "
            "[checks.py](../scripts/codex_fe/checks.py). Lint and formatting checks pass on the experiment scripts."
        ),
        "",
        (
            "`schedule_second` is constant in both sets; `precip_mm_trend` is constant in ranking only. "
            "[Availability rates](codex_fe_results/parity.json) and [cardinality audit](codex_fe_results/candidate_audit.json) "
            "are saved for every column. No ranking labels were used."
        ),
        "",
        "## Definitions",
        "",
        "| Candidate | Definition / motivation |",
        "|---|---|",
    ]
    out += [f"| `{name}` | {definition} |" for name, definition in desc.items()]
    out += [
        "",
        "## Reproduction and artifacts",
        "",
        (
            "Run from this worktree with `POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src`. "
            "Use `.venv/bin/python scripts/codex_fe/candidates.py`, then `baseline.py`, "
            "`screen.py screen`, `screen.py confirm`, `refit.py noise`, `refit.py selected`, "
            "`refit.py all`, `evaluate_refits.py`, `checks.py`, and `report.py`. Run training commands serially. "
            "`run_stages.py` coordinates the middle stages after baseline completion; clear stale output metadata "
            "when intentionally starting a new experiment. Predictions are keyed parquet files under ignored "
            "`outputs/codex_fe/`; all aggregate evidence is under `docs/codex_fe_results/`."
        ),
        "",
        (
            "**Commit limitation:** `git add` failed because `/home/user/prc-data-challenge/.git/worktrees/codex-fe/index.lock` "
            "is outside the writable workspace and the filesystem is read-only. No commit could be created on the "
            "requested branch. Files and results are preserved in this worktree; no workaround modified another checkout."
        ),
        "",
    ]
    Path("docs/codex_feature_report.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()

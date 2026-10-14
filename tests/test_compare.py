"""The decision rule has to reject an improvement that lives in a handful of rows (docs/07, docs/08)."""

import numpy as np

from prc.compare import (
    dropped_rows,
    format_tail_report,
    group_diff,
    month_bootstrap_rmse_diff,
    per_group_diff,
    robustness_curve,
    tail_report,
    verdict,
)
from prc.cv import rmse

rng = np.random.default_rng(0)


def _broad_improvement(n=20_000):
    y = rng.normal(900, 200, n)
    p_a = y + rng.normal(0, 220, n)
    p_b = y + rng.normal(0, 200, n)  # better everywhere, by a little
    return y, p_a, p_b


def _tail_only_improvement(n=20_000, k=30):
    y = rng.normal(900, 200, n)
    noise = rng.normal(0, 200, n)
    p_a, p_b = y + noise, y + noise  # identical apart from k rows
    y[:k] = 86_400.0  # the day-rollover rows that dominate the metric
    p_a[:k], p_b[:k] = 1_000.0, 60_000.0  # B happens to catch them; A does not
    return y, p_a, p_b


def test_broad_improvement_is_adopted():
    ok, why = verdict(robustness_curve(*_broad_improvement(), n_boot=200))
    assert ok, why


def test_tail_only_improvement_is_rejected():
    curve = robustness_curve(*_tail_only_improvement(), n_boot=200)
    full = next(r for r in curve if r["dropped"] == 0)
    assert full["diff"] < -100, "the setup should look like a huge win on all rows"
    ok, why = verdict(curve)
    assert not ok and "tail-only" in why, why


def test_month_bootstrap_is_wider_than_the_row_bootstrap():
    y, p_a, p_b = _broad_improvement()
    month = np.repeat(np.arange(4), len(y) // 4)
    mb = month_bootstrap_rmse_diff(y, p_a, p_b, month, n_boot=200)
    row = next(r for r in robustness_curve(y, p_a, p_b, drop_levels=(0,), n_boot=200) if r["dropped"] == 0)
    assert mb["ci95"][1] - mb["ci95"][0] > row["ci95"][1] - row["ci95"][0]


def test_no_difference_is_rejected():
    y = rng.normal(900, 200, 5_000)
    p = y + rng.normal(0, 200, 5_000)
    ok, why = verdict(robustness_curve(y, p, p.copy(), n_boot=200))
    assert not ok and "no difference" in why, why


def test_a_worse_candidate_is_named_as_worse_not_as_a_null():
    """[+18, +53] is not 'includes zero'. The wording decides whether anyone re-runs the experiment."""
    y = rng.normal(900, 200, 5_000)
    p_a = y + rng.normal(0, 200, 5_000)
    p_b = y + rng.normal(0, 400, 5_000)
    ok, why = verdict(robustness_curve(y, p_a, p_b, n_boot=200))
    assert not ok and "worse" in why, why


# --- the second instrument: what the verdict does not look at (docs/investigation_2026-09-16 §1.3) -----------


def _tail_only_with_airports(n=20_000, k=30, n_unmatched=300):
    """The same tail-only setup, with the routing group and the airport the tail sits at.

    The K rows are unmatched rows of one airport, which is the shape the investigation measured: at K=200 the
    drop set held 9 of the 10 rollover rows and 129 of the 537 LIRF unmatched rows."""
    y, p_a, p_b = _tail_only_improvement(n, k)
    airport = np.full(n, "LFPG", dtype=object)
    airport[:k] = "LIRF"
    matched = np.ones(n, bool)
    matched[:n_unmatched] = False  # the k tail rows are inside the unmatched block
    return y, p_a, p_b, matched, airport


def test_dropped_rows_are_exactly_the_rows_the_curve_removes():
    y, p_a, p_b = _tail_only_improvement()
    idx = dropped_rows(y, p_a, p_b, 200)
    assert len(idx) == 200
    assert set(range(30)) <= set(idx.tolist()), "the rows that make the effect must be in the drop set"
    keep = np.ones(len(y), bool)
    keep[idx] = False
    curve = robustness_curve(y, p_a, p_b, drop_levels=(200,), n_boot=200)[0]
    assert curve["rmse_b"] == rmse(y[keep], p_b[keep])  # same rows, not merely the same count


def test_tail_report_names_the_rows_the_verdict_stops_looking_at():
    y, p_a, p_b, matched, airport = _tail_only_with_airports()
    ok, why = verdict(robustness_curve(y, p_a, p_b, n_boot=200))
    assert not ok and "tail-only" in why, why
    rep = tail_report(y, p_a, p_b, matched, airport, ids=np.arange(len(y)), n_boot=200)
    assert rep["n_dropped"] == 200 and rep["n_dropped_unmatched"] >= 30  # the k tail rows are all unmatched
    listed = {r["id"] for r in rep["dropped"]}
    assert set(range(30)) <= listed
    row = next(r for r in rep["dropped"] if r["id"] == 0)
    assert row["y"] == 86_400.0 and row["p_a"] == 1_000.0 and row["p_b"] == 60_000.0  # label and both predictions
    assert rep["dominant_group"] == "LIRF" and rep["dominant_gap_share"] > 0.9
    assert rep["unmatched"]["ci95"][1] < 0, "B is better on the unmatched rows, which the verdict never reports"


def test_per_airport_breakdown_separates_structure_from_a_one_airport_tail():
    y, p_a, p_b, matched, airport = _tail_only_with_airports()
    by = per_group_diff(y, p_a, p_b, airport, ~matched, n_boot=200)
    assert [r["group"] for r in by] == ["LIRF", "LFPG"], "ordered by share of the squared disagreement"
    assert abs(sum(r["gap_share"] for r in by) - 1.0) < 1e-9
    assert by[0]["n"] == 30 and by[0]["diff"] < -1_000  # the whole difference is 30 rows at one airport
    assert abs(by[1]["diff"]) < 1e-9  # every other airport: the two builds are identical


def test_group_diff_on_every_row_is_the_full_data_difference():
    y, p_a, p_b = _broad_improvement()
    full = next(r for r in robustness_curve(y, p_a, p_b, drop_levels=(0,), n_boot=200) if r["dropped"] == 0)
    g = group_diff(y, p_a, p_b, np.ones(len(y), bool), n_boot=200)
    assert g["n"] == len(y) and g["diff"] == full["diff"] and g["ci95"] == full["ci95"]


def test_the_report_is_labelled_as_reporting_only_and_leaves_the_verdict_alone():
    y, p_a, p_b, matched, airport = _tail_only_with_airports()
    curve = robustness_curve(y, p_a, p_b, n_boot=200)
    before = verdict(curve)
    text = "\n".join(format_tail_report(tail_report(y, p_a, p_b, matched, airport, n_boot=200), max_listed=5))
    assert "reporting only" in text and "none of these lines change adopt/reject" in text
    assert "195 more" in text  # the list is truncated only when asked
    assert verdict(curve) == before


def test_holdout_mask_is_stable_by_airport_day_and_near_its_share():
    from prc.compare import holdout_mask

    rng = np.random.default_rng(0)
    ap = rng.choice(["EDDF", "EGLL", "LIRF"], 20000)
    day = np.array([f"2025-01-{d:02d}" for d in rng.integers(1, 29, 20000)])
    m = holdout_mask(ap, day)
    assert np.array_equal(m, holdout_mask(ap, day))
    keys = np.char.add(ap, day)
    for k in np.unique(keys):  # whole airport-days, never split
        assert len(np.unique(m[keys == k])) == 1
    assert 0.05 < m.mean() < 0.4


def test_block_bootstrap_sign_and_ci():
    from prc.compare import block_bootstrap_rmse_diff

    rng = np.random.default_rng(1)
    y = rng.normal(0, 100, 5000)
    p_a = y + rng.normal(0, 50, 5000)
    p_b = y + rng.normal(0, 40, 5000)
    r = block_bootstrap_rmse_diff(y, p_a, p_b, rng.integers(0, 200, 5000), n_boot=300)
    assert r["diff_rmse"] < 0 and r["ci95"][1] < 0 and r["n_blocks"] == 200


def test_strict_verdict_rejects_when_one_fold_does_not_improve():
    from prc.compare import strict_report, strict_verdict

    rng = np.random.default_rng(2)
    n = 6000
    y = rng.normal(0, 100, n)
    fold = np.repeat(["jan", "jul", "forward"], n // 3)
    ap = np.tile(np.array([f"A{i}" for i in range(10)]), n // 10)
    block = np.char.add(ap, rng.integers(0, 30, n).astype(str))
    p_a = y + rng.normal(0, 50, n)
    p_b = y + rng.normal(0, 40, n)
    good = strict_report(y, p_a, p_b, fold, ap, block, k=20, n_boot=200)
    assert strict_verdict(True, good)[0]
    p_c = np.where(fold == "jan", y + rng.normal(0, 60, n), p_b)  # better in jul/forward, worse in jan
    bad = strict_report(y, p_a, p_c, fold, ap, block, k=20, n_boot=200)
    ok, why = strict_verdict(True, bad)
    assert not ok and "jan" in why
    assert not strict_verdict(False, good)[0]

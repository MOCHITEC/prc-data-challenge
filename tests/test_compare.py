"""The decision rule has to reject an improvement that lives in a handful of rows (docs/07, docs/08)."""

import numpy as np

from prc.compare import month_bootstrap_rmse_diff, robustness_curve, verdict

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

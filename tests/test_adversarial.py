"""The adversarial classifier has to answer 0.5 when the two sides are the same data."""

from prc.adversarial import CALENDAR_COLS, build_design, run
from prc.features import CATEGORICALS, build_departure_features
from prc.synth import generate_month


def _feats(month, seed):
    return build_departure_features(generate_month(2025, month, seed=seed, deps_per_airport_day=8))


def test_two_samples_of_the_same_process_are_not_separable():
    a, b = _feats(7, 1), _feats(7, 2)
    n_cat = sum(1 for c in CATEGORICALS if c in a.columns and c in b.columns)
    X, y, cols = build_design(a, b)
    auc, _ = run(X, y, cols, n_cat)
    assert 0.35 < auc < 0.65, f"same generator should be indistinguishable, got AUC {auc:.3f}"


def test_the_calendar_columns_are_excluded_by_default():
    a, b = _feats(1, 3), _feats(7, 4)
    _, _, cols = build_design(a, b)
    assert not any(c in cols for c in CALENDAR_COLS)
    _, _, kept = build_design(a, b, drop_calendar=False)
    assert any(c in kept for c in CALENDAR_COLS)

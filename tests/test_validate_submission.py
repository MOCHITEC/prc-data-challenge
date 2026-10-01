import numpy as np
import pyarrow as pa

from prc.config import SUBMISSION_ID_COL, SUBMISSION_TARGET_COL
from prc.validate_submission import validate


def _template(n=1000):
    ids = np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)
    return pa.table({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": pa.array([None] * n, pa.float64())})


def _fill(template, values):
    return template.set_column(1, "TAXITIME_SEC_mvt", pa.array(values, pa.float64()))


def _preds(n, centre=900.0):
    """Plausible predictions: spread around a centre, as a real model produces. A constant vector is not a
    valid submission - the gate rejects it - so fixtures must not use one."""
    return centre + np.arange(n, dtype=float) % 200 - 100


def test_valid_submission_passes():
    t = _template()
    s = _fill(t, _preds(t.num_rows))
    r = validate(s, t, expected_rows=t.num_rows)
    assert r.ok, r.errors


def test_nan_is_rejected():
    t = _template()
    vals = _preds(t.num_rows)
    vals[5] = np.nan
    r = validate(_fill(t, vals), t, expected_rows=t.num_rows)
    assert not r.ok and any("NaN" in e for e in r.errors)


def test_missing_row_is_rejected():
    t = _template()
    s = _fill(t, _preds(t.num_rows)).slice(0, t.num_rows - 1)
    r = validate(s, t, expected_rows=t.num_rows)
    assert not r.ok and any("row count" in e for e in r.errors)


def test_wrong_ids_are_rejected():
    t = _template()
    s = _fill(t, _preds(t.num_rows))
    s = s.set_column(0, "MVT_ID_mvt", pa.array(np.arange(t.num_rows), pa.int64()))
    r = validate(s, t, expected_rows=t.num_rows)
    assert not r.ok and any("set differs" in e for e in r.errors)


def test_dtype_mismatch_is_rejected():
    t = _template()
    s = t.set_column(1, "TAXITIME_SEC_mvt", pa.array(_preds(t.num_rows), pa.int64()))
    r = validate(s, t, expected_rows=t.num_rows)
    assert not r.ok and any("dtype" in e for e in r.errors)


def test_negative_is_rejected_and_reordered_ids_warn():
    t = _template()
    vals = _preds(t.num_rows)
    vals[0] = -1.0
    r = validate(_fill(t, vals), t, expected_rows=t.num_rows)
    assert not r.ok and any("negative" in e for e in r.errors)
    s = _fill(t, _preds(t.num_rows)).take(pa.array(np.arange(t.num_rows)[::-1]))
    r2 = validate(s, t, expected_rows=t.num_rows)
    assert r2.ok and any("order differs" in w for w in r2.warnings)


def _sub(values, ids=None):
    """A structurally perfect submission carrying the given predictions."""
    n = len(values)
    ids = ids if ids is not None else [float(i) for i in range(n)]
    return pa.table(
        {SUBMISSION_ID_COL: pa.array(ids, type=pa.float64()), SUBMISSION_TARGET_COL: pa.array(values, pa.float64())}
    )


def test_gate_rejects_a_model_trained_on_the_wrong_data():
    """Ids, dtypes and nulls are all fine - only the numbers are wrong. This is the near-miss of 2026-09-10."""
    good = _sub([900.0 + i for i in range(100)])
    res = validate(good, good, expected_rows=None, reference_median=950.0)
    assert res.ok, res.errors

    wrong = _sub([300.0 + i for i in range(100)])  # trained on something else entirely
    res = validate(wrong, good, expected_rows=None, reference_median=950.0)
    assert not res.ok
    assert any("median" in e for e in res.errors)


def test_gate_rejects_a_constant_prediction():
    flat = _sub([912.0] * 100)
    res = validate(flat, flat, expected_rows=None)
    assert not res.ok
    assert any("same value" in e for e in res.errors)


def test_median_check_allows_normal_drift():
    t = _sub([900.0 + i for i in range(100)])
    for med in (760.0, 1180.0):  # within +-30% of the submission's own median (949.5)
        assert validate(t, t, expected_rows=None, reference_median=med).ok


def test_median_check_is_skipped_when_no_reference_is_given():
    t = _sub([10.0 + i for i in range(100)])
    assert validate(t, t, expected_rows=None).ok

import numpy as np
import pyarrow as pa

from prc.validate_submission import validate


def _template(n=1000):
    ids = np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)
    return pa.table({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": pa.array([None] * n, pa.float64())})


def _fill(template, values):
    return template.set_column(1, "TAXITIME_SEC_mvt", pa.array(values, pa.float64()))


def test_valid_submission_passes():
    t = _template()
    s = _fill(t, np.full(t.num_rows, 900.0))
    r = validate(s, t, expected_rows=t.num_rows)
    assert r.ok, r.errors


def test_nan_is_rejected():
    t = _template()
    vals = np.full(t.num_rows, 900.0)
    vals[5] = np.nan
    r = validate(_fill(t, vals), t, expected_rows=t.num_rows)
    assert not r.ok and any("NaN" in e for e in r.errors)


def test_missing_row_is_rejected():
    t = _template()
    s = _fill(t, np.full(t.num_rows, 900.0)).slice(0, t.num_rows - 1)
    r = validate(s, t, expected_rows=t.num_rows)
    assert not r.ok and any("row count" in e for e in r.errors)


def test_wrong_ids_are_rejected():
    t = _template()
    s = _fill(t, np.full(t.num_rows, 900.0))
    s = s.set_column(0, "MVT_ID_mvt", pa.array(np.arange(t.num_rows), pa.int64()))
    r = validate(s, t, expected_rows=t.num_rows)
    assert not r.ok and any("set differs" in e for e in r.errors)


def test_dtype_mismatch_is_rejected():
    t = _template()
    s = t.set_column(1, "TAXITIME_SEC_mvt", pa.array(np.full(t.num_rows, 900), pa.int64()))
    r = validate(s, t, expected_rows=t.num_rows)
    assert not r.ok and any("dtype" in e for e in r.errors)


def test_negative_is_rejected_and_reordered_ids_warn():
    t = _template()
    vals = np.full(t.num_rows, 900.0)
    vals[0] = -1.0
    r = validate(_fill(t, vals), t, expected_rows=t.num_rows)
    assert not r.ok and any("negative" in e for e in r.errors)
    s = _fill(t, np.full(t.num_rows, 900.0)).take(pa.array(np.arange(t.num_rows)[::-1]))
    r2 = validate(s, t, expected_rows=t.num_rows)
    assert r2.ok and any("order differs" in w for w in r2.warnings)

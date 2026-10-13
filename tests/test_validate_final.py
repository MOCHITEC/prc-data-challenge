"""Final gate (prc.validate_final) on a small synthetic final set: 10 airports x Jan/Feb/Jun/Jul 2026."""

import datetime as dt

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prc.config import SUBMISSION_ID_COL, SUBMISSION_TARGET_COL, TARGET_AIRPORTS
from prc.validate_final import file_md5, main, validate_final, validate_final_files

MONTHS = (1, 2, 6, 7)


def make_final_set(tmp_path, per_cell=3, extra_month=None):
    """final_ranking-like (DEP + ARR rows), final_submitting-like template, and a valid submission."""
    ids, times, adep = [], [], []
    k = 0
    for m in MONTHS + ((extra_month,) if extra_month else ()):
        for ap in TARGET_AIRPORTS:
            for i in range(per_cell):
                ids.append(float(202_600_000 + k))
                times.append(dt.datetime(2026, m, 1 + i, 8, 0, tzinfo=dt.UTC))
                adep.append(ap)
                k += 1
    n = len(ids)
    order = np.random.default_rng(0).permutation(n)  # template order is not ranking order
    ids_t = [ids[i] for i in order]
    ranking = pa.table(
        {
            SUBMISSION_ID_COL: ids + [9e8 + i for i in range(5)],
            "PHASE_mvt": ["DEP"] * n + ["ARR"] * 5,
            "ADEP_mvt": adep + ["EGLL"] * 5,
            "MVT_TIME_UTC_mvt": pa.array(
                times + [dt.datetime(2026, 3, 1, tzinfo=dt.UTC)] * 5, pa.timestamp("us", "UTC")
            ),
        }
    )
    template = pa.table(
        {SUBMISSION_ID_COL: pa.array(ids_t, pa.float64()), SUBMISSION_TARGET_COL: pa.nulls(n, pa.int32())}
    )
    values = (600 + np.arange(n) % 500).astype(np.int32)
    sub = template.set_column(1, SUBMISSION_TARGET_COL, pa.array(values, pa.int32()))
    paths = {name: tmp_path / f"{name}.parquet" for name in ("ranking", "template", "sub")}
    pq.write_table(ranking, paths["ranking"])
    pq.write_table(template, paths["template"])
    pq.write_table(sub, paths["sub"])
    return paths, sub, template, ranking


def _with_values(sub, values, type_=None):
    return sub.set_column(1, SUBMISSION_TARGET_COL, pa.array(values, type_ or pa.int32()))


def test_valid_final_passes_and_reports_months_and_airports(tmp_path):
    paths, *_ = make_final_set(tmp_path)
    rep = validate_final_files(paths["sub"], paths["template"], paths["ranking"], expected_rows=None)
    assert rep.ok, rep.errors
    assert {m: v["rows"] for m, v in rep.months.items()} == {"Jan": 30, "Feb": 30, "Jun": 30, "Jul": 30}
    assert set(rep.months["Jan"]["quantiles"]) >= {"p0", "p50", "p100"}
    assert rep.airport_month["EGLL"] == {"Jan": 3, "Feb": 3, "Jun": 3, "Jul": 3}
    assert rep.md5 == file_md5(paths["sub"]) and rep.n_above_long == 0 and rep.warnings == []


def test_order_difference_is_a_hard_failure(tmp_path):
    _, sub, template, ranking = make_final_set(tmp_path)
    rep = validate_final(sub.take(pa.array(np.arange(sub.num_rows)[::-1])), template, ranking, expected_rows=None)
    assert not rep.ok and any("order differs" in e for e in rep.errors)


def test_id_set_difference_is_a_hard_failure(tmp_path):
    _, sub, template, ranking = make_final_set(tmp_path)
    ids = sub[SUBMISSION_ID_COL].to_pylist()
    ids[0] = 1.0
    bad = sub.set_column(0, SUBMISSION_ID_COL, pa.array(ids, pa.float64()))
    rep = validate_final(bad, template, ranking, expected_rows=None)
    assert not rep.ok and any("set differs" in e for e in rep.errors)
    assert any("not departures in final_ranking" in e for e in rep.errors)


@pytest.mark.parametrize(("patch", "message"), [(-5, "negative"), (None, "null")])
def test_negatives_and_nulls_fail(tmp_path, patch, message):
    _, sub, template, ranking = make_final_set(tmp_path)
    vals = sub[SUBMISSION_TARGET_COL].to_pylist()
    vals[7] = patch
    rep = validate_final(_with_values(sub, vals), template, ranking, expected_rows=None)
    assert not rep.ok and any(message in e for e in rep.errors)


def test_long_rows_are_counted_but_do_not_fail(tmp_path):
    _, sub, template, ranking = make_final_set(tmp_path)
    vals = sub[SUBMISSION_TARGET_COL].to_pylist()
    vals[3], vals[4] = 10_801, 50_000
    rep = validate_final(_with_values(sub, vals), template, ranking, expected_rows=None)
    assert rep.ok and rep.n_above_long == 2 and rep.max_value == 50_000
    assert sum(m.get("n_above_10800", 0) for m in rep.months.values()) == 2


def test_dtype_must_equal_the_template(tmp_path):
    _, sub, template, ranking = make_final_set(tmp_path)
    as_float = _with_values(sub, np.asarray(sub[SUBMISSION_TARGET_COL]).astype(float), pa.float64())
    rep = validate_final(as_float, template, ranking, expected_rows=None)
    assert not rep.ok and any("dtype of TAXITIME_SEC_mvt" in e for e in rep.errors)


def test_a_month_outside_the_final_four_fails(tmp_path):
    _, sub, template, ranking = make_final_set(tmp_path, extra_month=3)
    rep = validate_final(sub, template, ranking, expected_rows=None)
    assert not rep.ok and any("outside Jan/Feb/Jun/Jul" in e and "2026-03" in e for e in rep.errors)


def test_md5_mismatch_and_missing_ranking_fail(tmp_path):
    paths, *_ = make_final_set(tmp_path)
    rep = validate_final_files(
        paths["sub"], paths["template"], paths["ranking"], expect_md5="0" * 32, expected_rows=None
    )
    assert not rep.ok and any("md5" in e for e in rep.errors)
    rep = validate_final_files(paths["sub"], paths["template"], None, expected_rows=None)
    assert not rep.ok and any("no --ranking" in e for e in rep.errors)


def test_cli_exit_codes(tmp_path, capsys):
    paths, *_ = make_final_set(tmp_path)
    args = [str(paths["sub"]), "--template", str(paths["template"]), "--ranking", str(paths["ranking"])]
    assert main(args + ["--json", str(tmp_path / "r.json")]) == 0  # the 670,790-row expectation is only a warning
    out = capsys.readouterr().out
    assert "PASS" in out and "WARNING: template has 120 rows" in out and (tmp_path / "r.json").exists()
    assert main(args + ["--expect-md5", "f" * 32]) == 1
    assert "FAIL" in capsys.readouterr().out

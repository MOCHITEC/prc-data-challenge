import polars as pl

from prc.profile import baselines, build_report, clock_profile, coverage_profile, regime_profile
from prc.synth import write_synthetic_dataset


def test_report_on_synthetic(tmp_path):
    write_synthetic_dataset(tmp_path / "raw", months=[(2025, 1), (2025, 7)], deps_per_airport_day=4)
    md = build_report(tmp_path / "raw")
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8."):
        assert heading in md
    assert "定義の整合" in md and "1.000000" in md
    assert "域内便" in md


def test_component_shapes(tmp_path):
    paths = write_synthetic_dataset(tmp_path / "raw", months=[(2025, 1)], deps_per_airport_day=6)
    train = pl.read_parquet(next(p for n, p in paths.items() if n.startswith("training_")))
    dep = train.filter(pl.col("PHASE_mvt") == "DEP")
    rank_dep = pl.read_parquet(paths["ranking.parquet"]).filter(pl.col("PHASE_mvt") == "DEP")
    assert {r["ADEP_mvt"] for r in clock_profile(dep)}
    reg = regime_profile(dep)
    assert all(0.0 <= r["block_eq_sched_rate"] <= 1.0 for r in reg)
    # The day-offset rate reads the label, not BLOCK - SCHED: that was the bug it replaced.
    assert all(0.0 <= r["day_offset_rate"] <= 1.0 and 0.0 <= r["sched_day_offset_rate"] <= 1.0 for r in reg)
    cov = coverage_profile(dep, rank_dep)
    assert any(r["column"] == "STAND_mvt" and 0.0 <= r["rank_rows_covered"] <= 1.0 for r in cov)
    names = {b["predictor"] for b in baselines(dep)}
    assert "global mean" in names and any("MVT - AOBT_3" in n for n in names)

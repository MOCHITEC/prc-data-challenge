"""Item 18 `sched-anchored-delay-shift` (docs/investigation_2026-09-16/plan.md).

No model training. Lazy scans, only the columns needed. Run with the project venv:

    POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/sched-anchored-delay-shift.py

Checks
  A. matched rows (routing definition: mvt_minus_aobt3 not null), per airport x month, Jan/Jul 2025 (feat_train)
     vs Jan/Jul 2026 (feat_rank): n, median of mvt_minus_sched / aobt3_minus_sched / aobt3_minus_eobt1 /
     mvt_minus_aobt3 (proxy), proxy sd; the same pooled over airports.
  B. congestion counts / headways / taxi-in means on the same rows (medians and means) -> is there a shift there?
  C. OOF residuals (experiments/oof_main-48c19bc830.parquet joined to feat_train on MVT_ID_mvt, non-null oof only):
     n, RMSE, mean residual (y - oof) by mvt_minus_sched bin and by aobt3_minus_sched bin
     (bins in minutes: <-15, -15..0, 0..15, 15..30, 30..60, 60..120, >120), all rows and matched rows,
     and by OOF month.
  D. bin shares: ranking matched rows per bin (all / Jan 2026 / Jul 2026) vs training matched rows
     (all 2025 / Jan 2025 / Jul 2025 / OOF-scored rows).
  E. data/work/adv_score.parquet (schema printed): mean `adv` by the same bins.
  F. arithmetic only: OOF MSE per bin re-weighted by the ranking bin shares (no refit).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")
import polars as pl  # noqa: E402

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(260)
pl.Config.set_float_precision(3)

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "data/work/feat_train.parquet"
RANK = ROOT / "data/work/feat_rank.parquet"
OOF = ROOT / "experiments/oof_main-48c19bc830.parquet"
ADV = ROOT / "data/work/adv_score.parquet"

LABEL, PROXY, TIME = "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "MVT_TIME_UTC_mvt"
ANCH = ["mvt_minus_sched", "aobt3_minus_sched", "aobt3_minus_eobt1", PROXY]
CONG = [
    "dep_tko_prev_30m", "arr_land_prev_30m", "arr_inblock_prev_30m", "dep_tko_prev_60m",
    "rwy_headway_prev_s", "rwy_headway_next_s", "arr_taxi_in_mean_30m", "nbr_dep_n_30m",
    "nbr_mvt_minus_aobt_mean_30m", "queue_ahead",
]
BIN_EDGES_MIN = [-15, 0, 15, 30, 60, 120]
BIN_LABELS = ["<-15", "-15..0", "0..15", "15..30", "30..60", "60..120", ">120"]


def ym() -> pl.Expr:
    return (pl.col(TIME).dt.year() * 100 + pl.col(TIME).dt.month()).alias("ym")


def bin_expr(col: str) -> pl.Expr:
    """Bin a seconds column into the plan's minute bins; null stays 'null'. Left edge exclusive, right inclusive,
    except the first bin: x < -15 min; last bin: x > 120 min."""
    e = pl.when(pl.col(col).is_null()).then(pl.lit("null"))
    edges_s = [m * 60 for m in BIN_EDGES_MIN]
    e = e.when(pl.col(col) < edges_s[0]).then(pl.lit(BIN_LABELS[0]))
    for lo, hi, lab in zip(edges_s[:-1], edges_s[1:], BIN_LABELS[1:-1]):
        e = e.when((pl.col(col) >= lo) & (pl.col(col) < hi)).then(pl.lit(lab))
    return e.otherwise(pl.lit(BIN_LABELS[-1])).alias(f"bin_{col}")


BIN_ORDER = {lab: i for i, lab in enumerate(BIN_LABELS + ["null"])}


def order_bins(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return df.with_columns(pl.col(col).replace_strict(BIN_ORDER, return_dtype=pl.Int32).alias("_o")).sort("_o").drop("_o")


def matched_janjul(path: Path, cols: list[str]) -> pl.LazyFrame:
    lf = pl.scan_parquet(path).select([TIME, "ADEP_mvt", PROXY] + [c for c in cols if c != PROXY]).with_columns(ym())
    return lf.filter(pl.col(PROXY).is_not_null() & pl.col(TIME).dt.month().is_in([1, 7]))


def agg_medians(lf: pl.LazyFrame, cols: list[str], by: list[str], sd_cols: tuple[str, ...] = (PROXY,)) -> pl.DataFrame:
    aggs = [pl.len().alias("n")] + [pl.col(c).cast(pl.Float64).median().alias(f"{c}_med") for c in cols]
    aggs += [pl.col(c).cast(pl.Float64).std().alias(f"{c}_sd") for c in sd_cols if c in cols]
    return lf.group_by(by).agg(aggs).sort(by).collect()


def side_by_side(tr: pl.DataFrame, rk: pl.DataFrame, metric_cols: list[str], key: list[str]) -> pl.DataFrame:
    """Long format: one row per (metric, key, calendar month): n_25, n_26, value 2025, value 2026, delta."""
    tr = tr.with_columns((pl.col("ym") % 100).alias("mo")).drop("ym")
    rk = rk.with_columns((pl.col("ym") % 100).alias("mo")).drop("ym")
    j = tr.join(rk, on=key + ["mo"], how="full", coalesce=True, suffix="_26")
    parts = []
    for c in metric_cols:
        parts.append(j.select(key + ["mo", "n", "n_26"]).with_columns(
            pl.lit(c).alias("metric"), j[c].alias("v25"), j[f"{c}_26"].alias("v26"), (j[f"{c}_26"] - j[c]).alias("delta")))
    return pl.concat(parts).select(["metric"] + key + ["mo", "n", "n_26", "v25", "v26", "delta"]).sort(["metric"] + key + ["mo"], maintain_order=True)


def main() -> None:
    # ---------------------------------------------------------------- A/B: matched rows Jan/Jul, 2025 vs 2026
    print("=== A. matched rows (proxy not null), per airport x month: medians of SCHED/EOBT-anchored features ===")
    tr = matched_janjul(TRAIN, ANCH)
    rk = matched_janjul(RANK, ANCH)
    a_tr = agg_medians(tr, ANCH, ["ADEP_mvt", "ym"])
    a_rk = agg_medians(rk, ANCH, ["ADEP_mvt", "ym"])
    mcols = [f"{c}_med" for c in ANCH] + [f"{PROXY}_sd"]
    print(side_by_side(a_tr, a_rk, mcols, ["ADEP_mvt"]))

    print("\n=== A2. same, pooled over airports (per month) ===")
    p_tr = agg_medians(tr, ANCH, ["ym"]).with_columns(pl.lit("all").alias("ADEP_mvt"))
    p_rk = agg_medians(rk, ANCH, ["ym"]).with_columns(pl.lit("all").alias("ADEP_mvt"))
    print(side_by_side(p_tr, p_rk, mcols, ["ADEP_mvt"]))

    print("\n=== A3. matched rows: quantiles of mvt_minus_sched pooled per month (p10/p25/p50/p75/p90) ===")
    def q(lf: pl.LazyFrame) -> pl.DataFrame:
        return lf.group_by("ym").agg(
            pl.len().alias("n"),
            *[pl.col("mvt_minus_sched").quantile(p).alias(f"p{int(p*100)}") for p in (0.1, 0.25, 0.5, 0.75, 0.9)],
            (pl.col("mvt_minus_sched") > 3600).mean().round(4).alias("share_gt60min"),
            (pl.col("mvt_minus_sched") > 7200).mean().round(4).alias("share_gt120min"),
        ).sort("ym")
    print(pl.concat([q(tr), q(rk)]).collect())

    print("\n=== B. congestion counts / headways / taxi-in on the same matched rows: medians (and means) ===")
    trc = matched_janjul(TRAIN, CONG)
    rkc = matched_janjul(RANK, CONG)

    def agg_cong(lf: pl.LazyFrame, by: list[str]) -> pl.DataFrame:
        aggs = [pl.len().alias("n")]
        for c in CONG:
            v = pl.col(c).cast(pl.Float64).fill_nan(None)  # Float32 features carry NaN, which would poison mean()
            aggs += [v.median().alias(f"{c}_med"), v.mean().alias(f"{c}_mean")]
            if c.startswith("rwy_headway"):
                aggs += [v.quantile(0.95).alias(f"{c}_p95"), v.is_null().mean().alias(f"{c}_nullrate")]
        return lf.group_by(by).agg(aggs).sort(by).collect()

    b_tr, b_rk = agg_cong(trc, ["ADEP_mvt", "ym"]), agg_cong(rkc, ["ADEP_mvt", "ym"])
    print("-- medians --")
    print(side_by_side(b_tr, b_rk, [f"{c}_med" for c in CONG], ["ADEP_mvt"]))
    print("-- means --")
    print(side_by_side(b_tr, b_rk, [f"{c}_mean" for c in CONG], ["ADEP_mvt"]))
    hw = [f"{c}_{s}" for c in CONG if c.startswith("rwy_headway") for s in ("p95", "nullrate")]
    print("-- headway p95 / null rate --")
    print(side_by_side(b_tr, b_rk, hw, ["ADEP_mvt"]))
    print("-- pooled over airports --")
    pb_tr = agg_cong(trc, ["ym"]).with_columns(pl.lit("all").alias("ADEP_mvt"))
    pb_rk = agg_cong(rkc, ["ym"]).with_columns(pl.lit("all").alias("ADEP_mvt"))
    print(side_by_side(pb_tr, pb_rk, [f"{c}_med" for c in CONG] + [f"{c}_mean" for c in CONG] + hw, ["ADEP_mvt"]))

    # ---------------------------------------------------------------- C: OOF residuals by bin
    print("\n=== C. OOF residuals by bin (oof_main-48c19bc830 joined to feat_train on MVT_ID_mvt, non-null oof) ===")
    base = pl.scan_parquet(TRAIN).select(
        ["MVT_ID_mvt", TIME, "ADEP_mvt", LABEL, PROXY, "mvt_minus_sched", "aobt3_minus_sched"]
    ).with_columns(ym(), bin_expr("mvt_minus_sched"), bin_expr("aobt3_minus_sched"), pl.col(PROXY).is_not_null().alias("matched"))
    oof = pl.scan_parquet(OOF).filter(pl.col("oof").is_not_null())
    j = base.join(oof, on="MVT_ID_mvt", how="inner").with_columns((pl.col(LABEL) - pl.col("oof")).alias("res"))
    tot = j.select(pl.len().alias("n"), (pl.col("res") ** 2).mean().sqrt().alias("rmse"), pl.col("res").mean().alias("mean_res"),
                   pl.col("matched").sum().alias("n_matched")).collect()
    print("OOF scored rows:", tot.to_dicts()[0])
    sse_tot = j.select((pl.col("res") ** 2).sum()).collect().item()

    def by_bin(lf: pl.LazyFrame, bcol: str, extra_by: list[str] | None = None) -> pl.DataFrame:
        by = [bcol] + (extra_by or [])
        d = lf.group_by(by).agg(
            pl.len().alias("n"),
            (pl.col("res") ** 2).mean().sqrt().alias("rmse"),
            pl.col("res").mean().alias("mean_res"),
            pl.col("res").median().alias("med_res"),
            pl.col(LABEL).median().alias("y_med"),
            pl.col("oof").median().alias("pred_med"),
            ((pl.col("res") ** 2).sum() / sse_tot).alias("sse_share"),
        ).collect()
        d = d.with_columns((pl.col("n") / pl.col("n").sum().over(extra_by) if extra_by else pl.col("n") / pl.col("n").sum()).alias("share"))
        return order_bins(d, bcol).sort(extra_by or [], maintain_order=True) if extra_by else order_bins(d, bcol)

    for bcol in ["bin_mvt_minus_sched", "bin_aobt3_minus_sched"]:
        print(f"\n-- C1. all OOF rows by {bcol} --")
        print(by_bin(j, bcol))
        print(f"-- C2. matched OOF rows only by {bcol} --")
        print(by_bin(j.filter(pl.col("matched")), bcol))
    print("\n-- C3. matched OOF rows by bin_mvt_minus_sched x OOF month (ym) --")
    print(by_bin(j.filter(pl.col("matched")), "bin_mvt_minus_sched", ["ym"]).sort(["ym"], maintain_order=True))
    print("\n-- C4. unmatched OOF rows only by bin_mvt_minus_sched --")
    print(by_bin(j.filter(~pl.col("matched")), "bin_mvt_minus_sched"))

    # ---------------------------------------------------------------- D: bin shares ranking vs training
    print("\n=== D. bin shares: matched rows, ranking (2026) vs training (2025) ===")

    def shares(path: Path, col: str, matched_only: bool = True) -> pl.DataFrame:
        lf = pl.scan_parquet(path).select([TIME, PROXY, col]).with_columns(ym(), bin_expr(col))
        if matched_only:
            lf = lf.filter(pl.col(PROXY).is_not_null())
        lf = lf.with_columns(pl.col("ym").cast(pl.Utf8).alias("grp"))
        both = pl.concat([lf.select(f"bin_{col}", "grp"), lf.select(f"bin_{col}", pl.lit("all").alias("grp"))])
        d = both.group_by(["grp", f"bin_{col}"]).agg(pl.len().alias("n")).with_columns(
            (pl.col("n") / pl.col("n").sum().over("grp") * 100).alias("pct")
        ).collect()
        return d

    for col in ["mvt_minus_sched", "aobt3_minus_sched"]:
        s_tr = shares(TRAIN, col).filter(pl.col("grp").is_in(["all", "202501", "202507"]))
        s_rk = shares(RANK, col)
        # OOF-scored rows (matched) share
        s_oof = (j.filter(pl.col("matched")).group_by(f"bin_{col}").agg(pl.len().alias("n"))
                 .with_columns((pl.col("n") / pl.col("n").sum() * 100).alias("pct"), pl.lit("oof").alias("grp")).collect())
        wide_tr = s_tr.pivot(on="grp", index=f"bin_{col}", values="pct").rename({"all": "train_all"})
        wide_rk = s_rk.pivot(on="grp", index=f"bin_{col}", values="pct").rename({"all": "rank_all"})
        wide_o = s_oof.pivot(on="grp", index=f"bin_{col}", values="pct")
        w = wide_tr.join(wide_o, on=f"bin_{col}", how="full", coalesce=True).join(wide_rk, on=f"bin_{col}", how="full", coalesce=True).fill_null(0.0)
        w = w.with_columns(
            (pl.col("202601") - pl.col("202501")).alias("d_jan"),
            (pl.col("202607") - pl.col("202507")).alias("d_jul"),
            (pl.col("rank_all") - pl.col("train_all")).alias("d_all"),
        )
        print(f"\n-- D1. matched-row share (%) per bin of {col} --")
        print(order_bins(w, f"bin_{col}").select([f"bin_{col}", "train_all", "oof", "rank_all", "d_all", "202501", "202601", "d_jan", "202507", "202607", "d_jul"]))

    print("\n-- D2. all rows (matched + unmatched) share (%) per bin of mvt_minus_sched --")
    s_tr = shares(TRAIN, "mvt_minus_sched", matched_only=False).pivot(on="grp", index="bin_mvt_minus_sched", values="pct")
    s_rk = shares(RANK, "mvt_minus_sched", matched_only=False).pivot(on="grp", index="bin_mvt_minus_sched", values="pct")
    w = s_tr.select("bin_mvt_minus_sched", pl.col("all").alias("train_all"), "202501", "202507").join(
        s_rk.select("bin_mvt_minus_sched", pl.col("all").alias("rank_all"), "202601", "202607"), on="bin_mvt_minus_sched", how="full", coalesce=True
    ).fill_null(0.0)
    print(order_bins(w, "bin_mvt_minus_sched"))

    # ---------------------------------------------------------------- E: adversarial score by bin
    print("\n=== E. data/work/adv_score.parquet ===")
    adv = pl.scan_parquet(ADV)
    print("schema:", {k: str(v) for k, v in adv.collect_schema().items()})
    adv_stats = adv.select(pl.len().alias("n"), pl.col("MVT_ID_mvt").n_unique().alias("uniq"), pl.col("adv").min().alias("min"),
                           pl.col("adv").max().alias("max"), pl.col("adv").mean().alias("mean"), pl.col("adv").median().alias("median"),
                           pl.col("adv").is_null().sum().alias("nulls")).collect()
    print(adv_stats)
    ja = base.join(adv, on="MVT_ID_mvt", how="inner")
    print("adv rows joined to feat_train:", ja.select(pl.len()).collect().item(),
          "| joined to feat_rank:", pl.scan_parquet(RANK).select("MVT_ID_mvt").join(adv, on="MVT_ID_mvt", how="inner").select(pl.len()).collect().item())
    for bcol in ["bin_mvt_minus_sched", "bin_aobt3_minus_sched"]:
        d = ja.group_by([bcol, "matched"]).agg(pl.len().alias("n"), pl.col("adv").mean().alias("adv_mean"), pl.col("adv").median().alias("adv_med"),
                                              (pl.col("adv") > 0.5).mean().alias("adv_gt_0.5")).collect()
        print(f"\n-- E1. mean adv by {bcol} x matched --")
        print(order_bins(d, bcol).sort("matched", maintain_order=True))
    print("\n-- E2. mean adv by month (all training rows) --")
    print(ja.group_by("ym").agg(pl.len().alias("n"), pl.col("adv").mean().alias("adv_mean")).sort("ym").collect())
    print("\n-- E3. mean adv by airport x matched --")
    print(ja.group_by(["ADEP_mvt", "matched"]).agg(pl.len().alias("n"), pl.col("adv").mean().alias("adv_mean")).sort(["ADEP_mvt", "matched"]).collect())

    # ---------------------------------------------------------------- F: reweighting arithmetic (no refit)
    print("\n=== F. arithmetic: matched-row OOF MSE per bin re-weighted by ranking bin shares (no refit) ===")
    for col in ["mvt_minus_sched", "aobt3_minus_sched"]:
        bcol = f"bin_{col}"
        mse = j.filter(pl.col("matched")).group_by(bcol).agg((pl.col("res") ** 2).mean().alias("mse"), pl.col("res").mean().alias("mean_res"), pl.len().alias("n_oof")).collect()
        w_rk = (pl.scan_parquet(RANK).select([PROXY, col]).filter(pl.col(PROXY).is_not_null()).with_columns(bin_expr(col))
                .group_by(bcol).agg(pl.len().alias("n_rk")).collect())
        m = mse.join(w_rk, on=bcol, how="full", coalesce=True).fill_null(0)
        m = m.with_columns((pl.col("n_oof") / pl.col("n_oof").sum()).alias("w_oof"), (pl.col("n_rk") / pl.col("n_rk").sum()).alias("w_rk"))
        rmse_oof = (m["mse"] * m["w_oof"]).sum() ** 0.5
        rmse_rk = (m["mse"] * m["w_rk"]).sum() ** 0.5
        mres_oof = (m["mean_res"] * m["w_oof"]).sum()
        mres_rk = (m["mean_res"] * m["w_rk"]).sum()
        print(f"{col}: matched OOF RMSE with OOF weights {rmse_oof:.2f} -> with ranking bin weights {rmse_rk:.2f}; "
              f"mean residual {mres_oof:+.2f} -> {mres_rk:+.2f}")
        print(order_bins(m, bcol))


if __name__ == "__main__":
    main()

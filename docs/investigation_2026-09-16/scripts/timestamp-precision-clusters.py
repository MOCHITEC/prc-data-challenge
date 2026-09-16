"""Verification of investigation item 19 `timestamp-precision-clusters` (plan.md §1; source E H10).

Independent re-derivation of:
  * per airport (and per airport x month) the share of rows whose MVT_TIME_UTC_mvt, BLOCK_TIME_UTC_mvt,
    AOBT_3_flt, SCHED_TIME_UTC_mvt fall on an exact minute (second == 0), in training and in ranking
    (BLOCK is null on every ranking row - reported as such);
  * the full second-of-minute histogram of MVT_TIME (mass at :00 vs the uniform 1/60 baseline);
  * OOF residual SD / RMSE (experiments/oof_main-48c19bc830.parquet joined to feat_train on MVT_ID_mvt,
    non-null oof only) by (ADEP, MVT second == 0) and by (ADEP, AOBT_3 second == 0), plus label SD by the
    same groups, with and without a matched/unmatched split;
  * whether any second-of-minute indicator is a model feature (code grep is reported in the .md; here the
    algebraic link  mvt_minus_sched mod 60 == 0  <=>  MVT second == 0  is checked on the data);
  * the quantisation noise floor as arithmetic.

No training. POLARS_MAX_THREADS=1. Lazy scans, only the needed columns.
"""
import math
import os

os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl

pl.Config.set_tbl_rows(400)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(260)
pl.Config.set_fmt_str_lengths(60)

ROOT = "/home/user/prc-data-challenge"
FT = f"{ROOT}/data/work/feat_train.parquet"
FR = f"{ROOT}/data/work/feat_rank.parquet"
OOF = f"{ROOT}/experiments/oof_main-48c19bc830.parquet"

CLUSTER = ["EDDM", "EGLL", "LFPG", "LIRF", "LTFM"]  # E H10's "~8.3 %" group

TS = {
    "mvt": "MVT_TIME_UTC_mvt",
    "blk": "BLOCK_TIME_UTC_mvt",
    "aobt": "AOBT_3_flt",
    "sched": "SCHED_TIME_UTC_mvt",
}


def load(path: str) -> pl.DataFrame:
    lf = pl.scan_parquet(path).select(
        "MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "mvt_minus_sched", *TS.values()
    )
    exprs = []
    for k, c in TS.items():
        exprs += [
            pl.col(c).dt.second().alias(f"{k}_sec"),
            (pl.col(c).dt.second() == 0).alias(f"{k}_s0"),  # null when the timestamp is null
            pl.col(c).is_null().alias(f"{k}_null"),
        ]
    return (
        lf.with_columns(
            pl.col("TAXITIME_SEC_mvt").cast(pl.Int64).alias("y"),
            pl.col("mvt_minus_aobt3").is_null().alias("unm"),
            (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym"),
            pl.col("ADEP_mvt").is_in(CLUSTER).alias("cluster"),
        )
        .with_columns(exprs)
        .drop(*TS.values())
        .collect()
    )


def s0_table(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Share of second == 0 per timestamp, over the non-null rows of that timestamp; n_null reported."""
    return (
        df.group_by(by)
        .agg(
            pl.len().alias("n"),
            pl.col("unm").sum().alias("n_unm"),
            *[pl.col(f"{k}_s0").mean().round(4).alias(f"{k}_s0") for k in TS],
            *[pl.col(f"{k}_null").sum().alias(f"{k}_null") for k in TS],
        )
        .sort(by)
    )


# ------------------------------------------------------------------ A. training shares
tr = load(FT)
print(f"### training departures: {tr.height}")
print("\n=== A1. training: share second == 0 per ADEP (denominator: non-null rows of that timestamp) ===")
print(s0_table(tr, ["ADEP_mvt"]))
print("\n=== A1b. pooled: cluster (EDDM/EGLL/LFPG/LIRF/LTFM) vs rest ===")
print(s0_table(tr, ["cluster"]))
print("\n=== A1c. per ADEP x matched/unmatched (routing definition) ===")
print(s0_table(tr, ["ADEP_mvt", "unm"]))

print("\n=== A2. training: MVT second-of-minute histogram, cluster vs rest (share per second value; uniform = 1/60 = 0.01667) ===")
hist = (
    tr.group_by("cluster", "mvt_sec")
    .agg(pl.len().alias("n"))
    .with_columns((pl.col("n") / pl.col("n").sum().over("cluster")).round(5).alias("share"))
    .sort("cluster", "mvt_sec")
)
for cl in (True, False):
    h = hist.filter(pl.col("cluster") == cl)
    p0 = h.filter(pl.col("mvt_sec") == 0)["share"][0]
    others = h.filter(pl.col("mvt_sec") != 0)["share"]
    print(f"-- cluster={cl}: share at :00 = {p0:.5f}; other seconds min/median/max = {others.min():.5f}/{others.median():.5f}/{others.max():.5f}; "
          f"excess mass at :00 over the mean of the other 59 seconds = {p0 - others.mean():.5f} "
          f"(== implied fraction of rows minute-rounded, if the rounded rows sit exactly at :00)")
    print("   full histogram (second: share), only seconds with n > 0:",
          {int(r["mvt_sec"]): r["share"] for r in h.select("mvt_sec", "share").to_dicts()})
    print("   seconds with zero rows:", sorted(set(range(60)) - set(h["mvt_sec"].to_list())))
print("-- per ADEP: MVT second-of-minute values present, count of distinct values, mass on {54..59, 0..6}")
print(
    tr.group_by("ADEP_mvt")
    .agg(
        pl.col("mvt_sec").n_unique().alias("n_distinct_sec"),
        pl.col("mvt_sec").is_in([54, 55, 56, 57, 58, 59, 0, 1, 2, 3, 4, 5, 6]).mean().round(5).alias("mvt_in_54_6"),
        pl.col("blk_sec").n_unique().alias("blk_n_distinct_sec"),
        pl.col("blk_sec").is_in([54, 55, 56, 57, 58, 59, 0, 1, 2, 3, 4, 5, 6]).mean().round(5).alias("blk_in_54_6"),
        pl.col("aobt_sec").drop_nulls().n_unique().alias("aobt_n_distinct_sec"),
    )
    .sort("ADEP_mvt")
)
print("-- MVT seconds 54 and 6 (partial edge cells) and 55/5 per cluster airport")
print(
    tr.filter(pl.col("cluster"))
    .group_by("ADEP_mvt")
    .agg(*[(pl.col("mvt_sec") == s).mean().round(5).alias(f"s{s}") for s in (53, 54, 55, 0, 5, 6, 7)])
    .sort("ADEP_mvt")
)
print("-- per ADEP: MVT share at :00, mean share of the other 59 seconds, excess, and share within 6 s of a minute boundary")
print(
    tr.group_by("ADEP_mvt")
    .agg(
        pl.len().alias("n"),
        (pl.col("mvt_sec") == 0).mean().round(5).alias("mvt_s0"),
        ((pl.col("mvt_sec") != 0).mean() / 59).round(5).alias("mean_other_sec"),
        (pl.min_horizontal(pl.col("mvt_sec"), 60 - pl.col("mvt_sec")) <= 6).mean().round(4).alias("mvt_within6"),
        (pl.min_horizontal(pl.col("blk_sec"), 60 - pl.col("blk_sec")) <= 6).mean().round(4).alias("blk_within6"),
        (pl.min_horizontal(pl.col("aobt_sec"), 60 - pl.col("aobt_sec")) <= 6).mean().round(4).alias("aobt_within6"),
        (pl.col("aobt_sec") == 30).mean().round(4).alias("aobt_s30"),
    )
    .with_columns((pl.col("mvt_s0") - pl.col("mean_other_sec")).round(5).alias("mvt_excess_at_0"))
    .sort("ADEP_mvt")
)
print("-- AOBT_3 second-of-minute histogram, all matched rows (top 6 values)")
print(
    tr.filter(~pl.col("aobt_null"))
    .group_by("aobt_sec").agg(pl.len().alias("n"))
    .with_columns((pl.col("n") / pl.col("n").sum()).round(5).alias("share"))
    .sort("n", descending=True).head(6)
)
print("-- BLOCK second-of-minute histogram, all rows (top 6 values)")
print(
    tr.group_by("blk_sec").agg(pl.len().alias("n"))
    .with_columns((pl.col("n") / pl.col("n").sum()).round(5).alias("share"))
    .sort("n", descending=True).head(6)
)

print("\n=== A3. training: share second == 0 per ADEP x month (ym = year*100+month) ===")
print(s0_table(tr, ["ADEP_mvt", "ym"]))
print("-- ranges over the 12 months, per ADEP")
print(
    s0_table(tr, ["ADEP_mvt", "ym"])
    .group_by("ADEP_mvt")
    .agg(
        *[pl.col(f"{k}_s0").min().alias(f"{k}_s0_min") for k in TS],
        *[pl.col(f"{k}_s0").max().alias(f"{k}_s0_max") for k in TS],
    )
    .sort("ADEP_mvt")
)

# ------------------------------------------------------------------ B. ranking shares
rk = load(FR)
print(f"\n### ranking departures: {rk.height}")
print("\n=== B1. ranking: share second == 0 per ADEP (BLOCK null count reported; y null count) ===")
print(s0_table(rk, ["ADEP_mvt"]))
print("BLOCK null on ranking rows:", rk["blk_null"].sum(), "of", rk.height, "; TAXITIME null:", rk["y"].is_null().sum())
print("\n=== B2. ranking: per ADEP x month ===")
print(s0_table(rk, ["ADEP_mvt", "ym"]))
print("\n=== B3. ranking vs training same calendar month (Jan, Jul): MVT and AOBT_3 second == 0 ===")
cmp = (
    pl.concat([
        s0_table(tr.filter(pl.col("ym").is_in([202501, 202507])), ["ADEP_mvt", "ym"]).with_columns(pl.lit("train").alias("set")),
        s0_table(rk, ["ADEP_mvt", "ym"]).with_columns(pl.lit("rank").alias("set")),
    ])
    .with_columns((pl.col("ym") % 100).alias("mo"))
    .select("ADEP_mvt", "mo", "set", "n", "mvt_s0", "aobt_s0", "sched_s0")
    .sort("ADEP_mvt", "mo", "set")
)
print(cmp)
print("-- EHAM AOBT_3 second == 0: Jan 2025 vs Jan 2026, Jul 2025 vs Jul 2026")
print(cmp.filter(pl.col("ADEP_mvt") == "EHAM"))
print("-- EHAM AOBT_3 second-of-minute histogram (top 6), training vs ranking")
for name, d in (("train", tr), ("rank", rk)):
    print(f"   {name}:",
          d.filter((pl.col("ADEP_mvt") == "EHAM") & ~pl.col("aobt_null")).group_by("aobt_sec").agg(pl.len().alias("n"))
          .with_columns((pl.col("n") / pl.col("n").sum()).round(4).alias("share")).sort("n", descending=True).head(6).to_dicts())

# ------------------------------------------------------------------ C. OOF residuals
print("\n=== C. OOF main-48c19bc830 joined to feat_train on MVT_ID_mvt (non-null, non-NaN oof only) ===")
oof_all = pl.read_parquet(OOF)
oof = oof_all.filter(pl.col("oof").is_not_null() & pl.col("oof").is_not_nan())
j = oof.join(tr, on="MVT_ID_mvt", how="left").with_columns((pl.col("oof") - pl.col("y")).alias("resid"))
print(f"oof rows {oof_all.height}, scored {oof.height}, unjoined {j['y'].is_null().sum()}, "
      f"months scored {sorted(j['ym'].unique().to_list())}, RMSE all {math.sqrt((j['resid'] ** 2).mean()):.2f}")
print("mvt_s0 share among scored rows:", round(j["mvt_s0"].mean(), 4), "; aobt_s0 share among scored matched rows:", round(j.filter(~pl.col("unm"))["aobt_s0"].mean(), 4))


def resid_table(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (
        df.group_by(by)
        .agg(
            pl.len().alias("n"),
            (pl.col("resid") ** 2).mean().sqrt().round(1).alias("rmse"),
            pl.col("resid").std().round(1).alias("resid_sd"),
            pl.col("resid").mean().round(1).alias("resid_mean"),
            pl.col("resid").median().round(1).alias("resid_med"),
            pl.col("y").std().round(1).alias("y_sd"),
            pl.col("y").median().alias("y_med"),
            pl.col("oof").std().round(1).alias("oof_sd"),
            pl.col("unm").mean().round(4).alias("unm_share"),
        )
        .sort(by)
    )


def sd_ratio(t: pl.DataFrame, flag: str, key: str = "ADEP_mvt") -> pl.DataFrame:
    a = t.filter(pl.col(flag)).select(key, pl.col("n").alias("n_s0"), pl.col("rmse").alias("rmse_s0"), pl.col("resid_sd").alias("sd_s0"), pl.col("y_sd").alias("ysd_s0"))
    b = t.filter(~pl.col(flag)).select(key, pl.col("n").alias("n_other"), pl.col("rmse").alias("rmse_other"), pl.col("resid_sd").alias("sd_other"), pl.col("y_sd").alias("ysd_other"))
    return (
        a.join(b, on=key, how="full", coalesce=True)
        .with_columns(
            (pl.col("sd_s0") / pl.col("sd_other")).round(3).alias("sd_ratio"),
            (pl.col("rmse_s0") / pl.col("rmse_other")).round(3).alias("rmse_ratio"),
            (pl.col("ysd_s0") / pl.col("ysd_other")).round(3).alias("ysd_ratio"),
        )
        .sort(key)
    )


print("\n--- C1. by (ADEP, MVT second == 0), all scored rows ---")
t1 = resid_table(j, ["ADEP_mvt", "mvt_s0"])
print(t1)
print("   side-by-side ratios (s0 / other):")
print(sd_ratio(t1, "mvt_s0"))
print("\n--- C1b. by (ADEP, matched, MVT second == 0) ---")
t1m = resid_table(j, ["ADEP_mvt", "unm", "mvt_s0"])
print(t1m)
print("   matched-only ratios:")
print(sd_ratio(t1m.filter(~pl.col("unm")), "mvt_s0"))
print("   unmatched-only ratios:")
print(sd_ratio(t1m.filter(pl.col("unm")), "mvt_s0"))
print("\n--- C1c. pooled cluster vs rest x MVT second == 0 (all rows / matched only) ---")
print(resid_table(j, ["cluster", "mvt_s0"]))
print(resid_table(j.filter(~pl.col("unm")), ["cluster", "mvt_s0"]))

print("\n--- C2. by (ADEP, AOBT_3 second == 0), matched rows only (AOBT_3 non-null) ---")
jm = j.filter(~pl.col("unm"))
t2 = resid_table(jm, ["ADEP_mvt", "aobt_s0"])
print(t2)
print("   ratios (s0 / other):")
print(sd_ratio(t2, "aobt_s0"))
print("   pooled cluster vs rest:")
print(resid_table(jm, ["cluster", "aobt_s0"]))

print("\n--- C3. cross: matched rows by (MVT s0, AOBT_3 s0), pooled ---")
print(resid_table(jm, ["mvt_s0", "aobt_s0"]))
print("   proxy (mvt_minus_aobt3) mod 60 == 0 share by cell:")
print(jm.group_by("mvt_s0", "aobt_s0").agg(pl.len().alias("n"), ((pl.col("mvt_minus_aobt3") % 60) == 0).mean().round(4).alias("proxy_mod60_0")).sort("mvt_s0", "aobt_s0"))

print("\n--- C4. by (ADEP, BLOCK second == 0) — label-side only (BLOCK is null on ranking rows) ---")
print(sd_ratio(resid_table(j, ["ADEP_mvt", "blk_s0"]), "blk_s0"))

print("\n--- C5. per-month OOF residual SD by (ADEP, MVT s0), matched only, months scored ---")
print(resid_table(jm, ["ADEP_mvt", "ym", "mvt_s0"]).select("ADEP_mvt", "ym", "mvt_s0", "n", "rmse", "resid_sd", "y_sd"))

# ------------------------------------------------------------------ C6. are the extreme unmatched-cell ratios single rows?
print("\n--- C6. OOF rows with y >= 72,000 (day-rollover band): seconds-of-minute and cell membership ---")
big = j.filter(pl.col("y") >= 72000).select("ADEP_mvt", "unm", "ym", "mvt_sec", "blk_sec", "aobt_sec", "y", "oof", "resid").sort("ADEP_mvt", "y")
print(big)
print("   per (ADEP, unm, mvt_s0): n rows y >= 72,000 and RMSE with / without them")
print(
    j.filter(pl.col("ADEP_mvt").is_in(["LFPG", "LIRF", "EGLL"]))
    .group_by("ADEP_mvt", "unm", "mvt_s0")
    .agg(
        pl.len().alias("n"),
        (pl.col("y") >= 72000).sum().alias("n_y_ge_72000"),
        (pl.col("resid") ** 2).mean().sqrt().round(1).alias("rmse"),
        (pl.col("resid").filter(pl.col("y") < 72000) ** 2).mean().sqrt().round(1).alias("rmse_excl_ge_72000"),
        pl.col("resid").filter(pl.col("y") < 72000).std().round(1).alias("sd_excl_ge_72000"),
    )
    .sort("ADEP_mvt", "unm", "mvt_s0")
)
print("   same for (ADEP, blk_s0) at LFPG / LIRF / LEBL")
print(
    j.filter(pl.col("ADEP_mvt").is_in(["LFPG", "LIRF", "LEBL"]))
    .group_by("ADEP_mvt", "blk_s0")
    .agg(
        pl.len().alias("n"),
        (pl.col("y") >= 72000).sum().alias("n_y_ge_72000"),
        (pl.col("resid") ** 2).mean().sqrt().round(1).alias("rmse"),
        (pl.col("resid").filter(pl.col("y") < 72000) ** 2).mean().sqrt().round(1).alias("rmse_excl_ge_72000"),
    )
    .sort("ADEP_mvt", "blk_s0")
)

# ------------------------------------------------------------------ F. is the label itself minute-quantised?
print("\n=== F. label y = MVT - BLOCK: y mod 60 per ADEP (share == 0, distinct values, share within 6 of a multiple of 60) ===")
ymod = (pl.col("y") % 60).alias("ymod")
print(
    tr.with_columns(ymod)
    .group_by("ADEP_mvt")
    .agg(
        pl.len().alias("n"),
        (pl.col("ymod") == 0).mean().round(4).alias("y_mod60_eq0"),
        pl.col("ymod").n_unique().alias("y_mod60_distinct"),
        (pl.min_horizontal(pl.col("ymod"), 60 - pl.col("ymod")) <= 6).mean().round(4).alias("y_within6_of_60k"),
        (pl.min_horizontal(pl.col("ymod"), 60 - pl.col("ymod")) <= 12).mean().round(4).alias("y_within12_of_60k"),
        ((pl.col("mvt_sec") - pl.col("blk_sec")) % 60).n_unique().alias("mvt_minus_blk_sec_distinct"),
    )
    .sort("ADEP_mvt")
)
print("-- cluster: histogram of y mod 60 (share per value, values with share > 0.001)")
h = (
    tr.filter(pl.col("cluster")).with_columns(ymod).group_by("ymod").agg(pl.len().alias("n"))
    .with_columns((pl.col("n") / pl.col("n").sum()).round(4).alias("share")).filter(pl.col("share") > 0.001).sort("ymod")
)
print({int(r["ymod"]): r["share"] for r in h.to_dicts()})
print("-- proxy = MVT - AOBT_3 mod 60 per ADEP (matched rows): share == 0, distinct values")
print(
    tr.filter(~pl.col("unm"))
    .with_columns((pl.col("mvt_minus_aobt3") % 60).alias("pmod"))
    .group_by("ADEP_mvt")
    .agg(pl.len().alias("n"), (pl.col("pmod") == 0).mean().round(4).alias("proxy_mod60_eq0"), pl.col("pmod").n_unique().alias("proxy_mod60_distinct"),
         (pl.min_horizontal(pl.col("pmod"), 60 - pl.col("pmod")) <= 6).mean().round(4).alias("proxy_within6_of_60k"))
    .sort("ADEP_mvt")
)

# ------------------------------------------------------------------ D. is the indicator a feature?
print("\n=== D. indicator vs existing features (data-side check; code grep reported in the .md) ===")
print(
    tr.select(
        pl.len().alias("n"),
        (pl.col("sched_sec") == 0).sum().alias("sched_sec0_rows"),
        (((pl.col("mvt_minus_sched") % 60) == 0) == pl.col("mvt_s0")).sum().alias("ms_mod60_iff_mvt_s0"),
        (((pl.col("mvt_minus_aobt3") % 60) == 0) == (pl.col("mvt_sec") == pl.col("aobt_sec"))).sum().alias("proxy_mod60_iff_same_sec"),
        pl.col("aobt_null").sum().alias("aobt_null"),
    )
)

# ------------------------------------------------------------------ E. quantisation arithmetic
print("\n=== E. quantisation noise floor (arithmetic, no data) ===")
sd_u = 60 / math.sqrt(12)
sd_tri = 60 / math.sqrt(6)
print(f"uniform rounding error on [-30, 30) s: sd = 60/sqrt(12) = {sd_u:.2f} s, variance = {sd_u**2:.0f} s^2")
print(f"difference of two independently minute-rounded stamps (triangular on [-60, 60]): sd = 60/sqrt(6) = {sd_tri:.2f} s, variance = {sd_tri**2:.0f} s^2")
for base in (217.7, 334.58, 162.1, 347.1):
    print(f"  added in quadrature to RMSE {base}: sqrt({base}^2 + {sd_u**2:.0f}) = {math.sqrt(base**2 + sd_u**2):.2f} (+{math.sqrt(base**2 + sd_u**2) - base:.2f} s); "
          f"variance share {sd_u**2 / (base**2 + sd_u**2):.4f}; with triangular: +{math.sqrt(base**2 + sd_tri**2) - base:.2f} s")
# measured: the gap between resid variance for s0 vs other rows, pooled matched, expressed in the same units
a = jm.filter(pl.col("mvt_s0"))["resid"]
b = jm.filter(~pl.col("mvt_s0"))["resid"]
print(f"measured pooled matched: sd(resid | MVT s0) = {a.std():.1f} (n={a.len()}), sd(resid | other) = {b.std():.1f} (n={b.len()}); "
      f"variance difference = {a.var() - b.var():.0f} s^2 vs 300 s^2 floor")

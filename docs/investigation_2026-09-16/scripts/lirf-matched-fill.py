"""Item 4 `lirf-matched-fill` — independent re-derivation (no training).

Inputs : data/work/feat_train.parquet, data/work/feat_rank.parquet,
         experiments/oof_main-48c19bc830.parquet
Run    : POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/lirf-matched-fill.py

Definitions (plan §0):
  y       = TAXITIME_SEC_mvt                       (= MVT - BLOCK)
  proxy   = MVT - AOBT_3       (= feature mvt_minus_aobt3)
  resid   = y - proxy          (= AOBT_3 - BLOCK algebraically)
  as_     = AOBT_3 - SCHED     (= feature aobt3_minus_sched)
  bs      = BLOCK - SCHED
  fill    = |bs| <= 10 s  (D);  fill6 = |bs| <= 6 s (M);  fill120 = |bs| <= 120 s (M)
  matched = AOBT_3 non-null (routing definition, model A). FLIGHT_ID-null count is also printed.
"""
import os

os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"
import polars as pl  # noqa: E402

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(260)
pl.Config.set_float_precision(3)

ROOT = "/home/user/prc-data-challenge/"
TR = ROOT + "data/work/feat_train.parquet"
RK = ROOT + "data/work/feat_rank.parquet"
OOF = ROOT + "experiments/oof_main-48c19bc830.parquet"

BASE_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "AIRCRAFT_OPERATOR_flt",
    "MARKET_SEGMENT_flt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AOBT_3_flt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "aobt3_minus_sched", "hour_utc",
    "month", "AIRCRAFT_TYPE_mvt",
]
sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()  # noqa: E731


def derive(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.col("AOBT_3_flt").is_not_null().alias("matched"),
            pl.col("FLIGHT_ID_mvt").is_not_null().alias("matched_D"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
            sec("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("proxy"),
            sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("ms"),
            sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("bs"),
            sec("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("as_"),
            sec("BLOCK_TIME_UTC_mvt", "AOBT_3_flt").alias("blk_minus_aobt"),
        ]
    ).with_columns(
        [
            (pl.col("y") - pl.col("proxy")).alias("resid"),
            (pl.col("bs").abs() <= 10).alias("fill"),
            (pl.col("bs").abs() <= 6).alias("fill6"),
            (pl.col("bs").abs() <= 120).alias("fill120"),
            (pl.col("as_").abs() > 600).alias("far"),
        ]
    )


def sd(c):  # population-agnostic: polars std is sample sd (ddof=1)
    return pl.col(c).std().round(1)


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


tr = derive(pl.scan_parquet(TR).select(BASE_COLS))

# ---------------------------------------------------------------- A. sanity / definitions
hdr("A. frame sanity: rows, PHASE, matched counts under both definitions, identities")
print(
    tr.select(
        [
            pl.len().alias("rows"),
            (pl.col("PHASE_mvt") == "DEP").sum().alias("n_DEP"),
            pl.col("matched").sum().alias("matched_routing(AOBT_3 nn)"),
            (~pl.col("matched")).sum().alias("unmatched_routing"),
            pl.col("matched_D").sum().alias("matched_D(FLIGHT_ID nn)"),
            (pl.col("matched_D") & ~pl.col("matched")).sum().alias("FLIGHT_ID nn & AOBT_3 null"),
            (pl.col("proxy") - pl.col("mvt_minus_aobt3")).abs().max().alias("max|proxy-mvt_minus_aobt3|"),
            (pl.col("as_") - pl.col("aobt3_minus_sched")).abs().max().alias("max|as_-aobt3_minus_sched|"),
            (pl.col("y") - sec("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt")).abs().max().alias("max|y-(MVT-BLOCK)|"),
            (pl.col("hour_utc") != pl.col("MVT_TIME_UTC_mvt").dt.hour()).sum().alias("hour_utc!=hour(MVT)"),
        ]
    ).collect()
)

lm = tr.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("matched")).collect()  # LIRF matched, in memory
print(f"LIRF matched (routing) rows: {lm.height}")
print(
    "LIRF matched_D rows:",
    tr.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("matched_D")).select(pl.len()).collect().item(),
    "| LIRF rows total:",
    tr.filter(pl.col("ADEP_mvt") == "LIRF").select(pl.len()).collect().item(),
)
print("AIRCRAFT_TYPE_mvt null among LIRF matched:", lm["AIRCRAFT_TYPE_mvt"].is_null().sum())
print("Operator null among LIRF matched:", lm["AIRCRAFT_OPERATOR_flt"].is_null().sum(),
      "| segment null:", lm["MARKET_SEGMENT_flt"].is_null().sum())

# ---------------------------------------------------------------- B. fill rates by airport (matched vs unmatched)
hdr("B. fill rate (|BLOCK-SCHED|<=10 s) by ADEP x matched; also +-6 s and <=120 s variants")
print(
    tr.group_by(["ADEP_mvt", "matched"])
    .agg([pl.len().alias("n"), pl.col("fill").mean().round(4).alias("fill10"),
          pl.col("fill6").mean().round(4).alias("fill6"), pl.col("fill120").mean().round(4).alias("fill120")])
    .sort(["ADEP_mvt", "matched"])
    .collect()
)

hdr("B2. matched rows, all airports: fill rate by |AOBT_3-SCHED| band (pivot ADEP x band)")
bands_all = (
    pl.when(pl.col("as_").abs() <= 60).then(pl.lit("0: <=60"))
    .when(pl.col("as_").abs() <= 300).then(pl.lit("1: 60-300"))
    .when(pl.col("as_").abs() <= 600).then(pl.lit("2: 300-600"))
    .when(pl.col("as_").abs() <= 1800).then(pl.lit("3: 600-1800"))
    .when(pl.col("as_").abs() <= 3600).then(pl.lit("4: 1800-3600"))
    .when(pl.col("as_").abs() <= 7200).then(pl.lit("5: 3600-7200"))
    .otherwise(pl.lit("6: >7200"))
    .alias("as_band")
)
b2 = (
    tr.filter(pl.col("matched")).with_columns(bands_all)
    .group_by(["ADEP_mvt", "as_band"]).agg([pl.len().alias("n"), pl.col("fill").mean().round(3).alias("fill")])
    .collect()
)
print(b2.pivot(on="as_band", index="ADEP_mvt", values="fill").sort("ADEP_mvt"))
print(b2.pivot(on="as_band", index="ADEP_mvt", values="n").sort("ADEP_mvt"))

# ---------------------------------------------------------------- C. LIRF matched: fill by month
hdr("C. LIRF matched: fill rate by month")
print(lm.group_by("month").agg([pl.len().alias("n"), pl.col("fill").mean().round(4).alias("fill")]).sort("month"))
print("overall fill:", round(lm["fill"].mean(), 4), "| fill6:", round(lm["fill6"].mean(), 4),
      "| fill120:", round(lm["fill120"].mean(), 4))

# ---------------------------------------------------------------- D. |AOBT_3 - SCHED| bands vs fill / residual sd
hdr("D. LIRF matched: |AOBT_3-SCHED| <=600 vs >600 : fill rate, sd(resid), sd(y), sd by fill state")
agg = [
    pl.len().alias("n"),
    pl.col("fill").mean().round(4).alias("fill"),
    pl.col("fill").sum().alias("n_fill"),
    sd("resid").alias("sd_resid_all"),
    pl.col("resid").filter(pl.col("fill")).std().round(1).alias("sd_resid_fill"),
    pl.col("resid").filter(~pl.col("fill")).std().round(1).alias("sd_resid_nofill"),
    pl.col("as_").filter(pl.col("fill")).std().round(1).alias("sd_as_fill"),
    sd("y").alias("sd_y_all"),
    pl.col("y").filter(pl.col("fill")).std().round(1).alias("sd_y_fill"),
    pl.col("y").filter(~pl.col("fill")).std().round(1).alias("sd_y_nofill"),
    pl.col("y").median().alias("y_med"),
    pl.col("proxy").median().alias("proxy_med"),
    pl.col("resid").median().alias("resid_med"),
]
print(lm.group_by("far").agg(agg).sort("far"))

hdr("D2. finer |AOBT_3-SCHED| bands (seconds)")
bands = (
    pl.when(pl.col("as_").abs() <= 60).then(pl.lit("0: <=60"))
    .when(pl.col("as_").abs() <= 300).then(pl.lit("1: 60-300"))
    .when(pl.col("as_").abs() <= 600).then(pl.lit("2: 300-600"))
    .when(pl.col("as_").abs() <= 1800).then(pl.lit("3: 600-1800"))
    .when(pl.col("as_").abs() <= 3600).then(pl.lit("4: 1800-3600"))
    .when(pl.col("as_").abs() <= 7200).then(pl.lit("5: 3600-7200"))
    .otherwise(pl.lit("6: >7200"))
    .alias("as_band")
)
print(lm.with_columns(bands).group_by("as_band").agg(agg).sort("as_band"))

hdr("D3. identity check: on filled rows resid == AOBT_3 - SCHED (within 10 s)?")
f = lm.filter(pl.col("fill"))
print(
    f.select(
        [
            pl.len().alias("n_fill"),
            ((pl.col("resid") - pl.col("as_")).abs() <= 10).mean().alias("frac |resid-as_|<=10"),
            (pl.col("resid") - pl.col("as_")).abs().max().alias("max|resid-as_|"),
            (pl.col("resid") - pl.col("as_")).abs().median().alias("med|resid-as_|"),
        ]
    )
)

# ---------------------------------------------------------------- E. residual SSE decomposition: how much of LIRF's resid variance is the filled rows
hdr("E. LIRF matched: share of sum(resid^2) and sum((resid-mean)^2) from filled / far cells")
tot_ss = (lm["resid"] ** 2).sum()
print(
    lm.group_by(["fill", "far"])
    .agg([pl.len().alias("n"), (pl.col("resid") ** 2).sum().alias("ss_resid"),
          sd("resid").alias("sd_resid"), pl.col("resid").mean().round(1).alias("mean_resid"),
          pl.col("resid").abs().median().alias("med|resid|")])
    .with_columns((pl.col("ss_resid") / tot_ss).round(4).alias("share_ss"))
    .sort(["fill", "far"])
)
print("LIRF matched sd(resid) all rows:", round(lm["resid"].std(), 1),
      "| excluding filled rows:", round(lm.filter(~pl.col("fill"))["resid"].std(), 1),
      "| excluding filled&far rows:", round(lm.filter(~(pl.col("fill") & pl.col("far")))["resid"].std(), 1))

# ---------------------------------------------------------------- F. residual sd by airport (matched), with/without filled rows
hdr("F. all airports, matched rows: sd(resid), sd(resid | not filled), RMS(resid), fill rate")
print(
    tr.filter(pl.col("matched"))
    .group_by("ADEP_mvt")
    .agg(
        [
            pl.len().alias("n"),
            pl.col("fill").mean().round(4).alias("fill"),
            sd("resid").alias("sd_resid"),
            pl.col("resid").filter(~pl.col("fill")).std().round(1).alias("sd_resid_nofill"),
            pl.col("resid").filter(~(pl.col("fill") & pl.col("far"))).std().round(1).alias("sd_resid_no_fillfar"),
            (pl.col("resid") ** 2).mean().sqrt().round(1).alias("rms_resid"),
            pl.col("resid").abs().quantile(0.99).alias("p99|resid|"),
            (pl.col("blk_minus_aobt").abs() > 3600).sum().alias("n_|BLK-AOBT|>1h"),
        ]
    )
    .sort("sd_resid", descending=True)
    .collect()
)

# ---------------------------------------------------------------- G. |BLOCK - AOBT_3| > 1 h at LIRF matched
hdr("G. LIRF matched rows with |BLOCK-AOBT_3| > 1 h")
g = lm.filter(pl.col("blk_minus_aobt").abs() > 3600)
print(
    g.select(
        [pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("proxy").median().alias("proxy_med"),
         pl.col("fill").mean().round(3).alias("fill"), pl.col("far").mean().round(3).alias("far"),
         (pl.col("blk_minus_aobt") < 0).mean().round(3).alias("frac BLOCK<AOBT_3"),
         pl.col("as_").median().alias("as_med")]
    )
)
print("  by (fill): ", g.group_by("fill").agg([pl.len(), pl.col("y").median().alias("y_med"),
                                                pl.col("proxy").median().alias("proxy_med"),
                                                pl.col("as_").median().alias("as_med")]).sort("fill"))

# ---------------------------------------------------------------- H. fill by operator / segment / hour (LIRF matched)
hdr("H1. LIRF matched: fill by operator (n>2000), sorted by fill")
op = (
    lm.group_by("AIRCRAFT_OPERATOR_flt")
    .agg([pl.len().alias("n"), pl.col("fill").mean().round(4).alias("fill"),
          pl.col("MARKET_SEGMENT_flt").mode().first().alias("seg_mode"),
          pl.col("far").mean().round(3).alias("far"),
          sd("resid").alias("sd_resid")])
    .sort("fill", descending=True)
)
print(op.filter(pl.col("n") > 2000))
print("distinct operators among LIRF matched:", op.height, "| with n>2000:", op.filter(pl.col("n") > 2000).height)
print("operators with n>=200 and fill>=0.30:")
print(op.filter((pl.col("n") >= 200) & (pl.col("fill") >= 0.30)))

hdr("H2. LIRF matched: fill by market segment")
print(lm.group_by("MARKET_SEGMENT_flt").agg([pl.len().alias("n"), pl.col("fill").mean().round(4).alias("fill"),
                                             pl.col("far").mean().round(3).alias("far"),
                                             sd("resid").alias("sd_resid")]).sort("n", descending=True))

hdr("H3. LIRF matched: fill by hour_utc")
print(lm.group_by("hour_utc").agg([pl.len().alias("n"), pl.col("fill").mean().round(3).alias("fill")])
      .sort("hour_utc").transpose(include_header=True, header_name="hour", column_names=None))

hdr("H4. EXTENSION: fill by operator x hour_utc x segment (cells with n>=100), top 40 by n, and top 40 by fill")
cell = (
    lm.group_by(["AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "hour_utc"])
    .agg([pl.len().alias("n"), pl.col("fill").mean().round(3).alias("fill"), pl.col("fill").sum().alias("n_fill")])
)
big = cell.filter(pl.col("n") >= 100)
print("cells total:", cell.height, "| cells n>=100:", big.height, "| rows in n>=100 cells:", big["n"].sum(),
      "| filled rows in n>=100 cells:", big["n_fill"].sum(), "of", lm["fill"].sum())
print("--- top 40 cells by n")
print(big.sort("n", descending=True).head(40))
print("--- top 40 cells (n>=100) by fill")
print(big.sort("fill", descending=True).head(40))
print("--- cells n>=100 with fill >= 0.8:", big.filter(pl.col("fill") >= 0.8).height,
      "rows:", big.filter(pl.col("fill") >= 0.8)["n"].sum(),
      "| fill <= 0.05:", big.filter(pl.col("fill") <= 0.05).height, "rows:", big.filter(pl.col("fill") <= 0.05)["n"].sum())

hdr("H5. Between-group variance share of the fill flag (R^2 of fill on group means; no model fit)")
p = lm["fill"].mean()
tot_var = p * (1 - p) * lm.height


def r2(keys):
    gm = lm.group_by(keys).agg([pl.len().alias("n"), pl.col("fill").mean().alias("m")])
    between = ((gm["m"] - p) ** 2 * gm["n"]).sum()
    return round(between / tot_var, 4), gm.height


for keys in (["AIRCRAFT_OPERATOR_flt"], ["MARKET_SEGMENT_flt"], ["hour_utc"], ["far"],
             ["AIRCRAFT_OPERATOR_flt", "hour_utc"], ["AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt"],
             ["AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "hour_utc"],
             ["AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "hour_utc", "far"],
             ["AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "hour_utc", "month"]):
    v, k = r2(keys)
    print(f"  R2(fill | {' x '.join(keys)}) = {v}   (groups={k})")

# ---------------------------------------------------------------- I. same operators in ranking LIRF matched rows?
hdr("I. EXTENSION: ranking LIRF matched rows — operator overlap with training LIRF matched operators")
rk = derive(pl.scan_parquet(RK).select(BASE_COLS)).filter(pl.col("ADEP_mvt") == "LIRF").collect()
rkm = rk.filter(pl.col("matched"))
print("ranking LIRF rows:", rk.height, "| matched (AOBT_3 nn):", rkm.height, "| unmatched:", rk.height - rkm.height,
      "| matched_D:", rk["matched_D"].sum())
print("ranking LIRF matched by month:", rkm.group_by("month").agg(pl.len().alias("n")).sort("month"))
train_ops = op.select(["AIRCRAFT_OPERATOR_flt", pl.col("n").alias("n_train"), pl.col("fill").alias("fill_train")])
rop = (
    rkm.group_by("AIRCRAFT_OPERATOR_flt").agg([pl.len().alias("n_rank"), pl.col("far").mean().round(3).alias("far_rank")])
    .join(train_ops, on="AIRCRAFT_OPERATOR_flt", how="left")
    .with_columns(pl.col("n_train").is_not_null().alias("seen_in_train_LIRF_matched"))
    .sort("n_rank", descending=True)
)
print("distinct operators ranking LIRF matched:", rop.height,
      "| of which seen among training LIRF matched:", rop["seen_in_train_LIRF_matched"].sum())
print("ranking rows whose operator is seen in training LIRF matched:",
      rop.filter(pl.col("seen_in_train_LIRF_matched"))["n_rank"].sum(), "of", rkm.height,
      "| unseen-operator rows:", rop.filter(~pl.col("seen_in_train_LIRF_matched"))["n_rank"].sum(),
      "| null-operator rows:", rkm["AIRCRAFT_OPERATOR_flt"].is_null().sum())
print("--- ranking LIRF matched operators (n_rank>=100) with training fill rate")
print(rop.filter(pl.col("n_rank") >= 100))
print("--- ranking rows by training-fill bucket of their operator (operators with n_train>2000 only)")
hi = train_ops.filter(pl.col("n_train") > 2000)
print(
    rkm.join(hi, on="AIRCRAFT_OPERATOR_flt", how="left")
    .with_columns(
        pl.when(pl.col("fill_train").is_null()).then(pl.lit("z: op n_train<=2000 or unseen"))
        .when(pl.col("fill_train") >= 0.5).then(pl.lit("a: fill>=0.5"))
        .when(pl.col("fill_train") >= 0.2).then(pl.lit("b: 0.2-0.5"))
        .when(pl.col("fill_train") >= 0.1).then(pl.lit("c: 0.1-0.2"))
        .otherwise(pl.lit("d: <0.1")).alias("bucket")
    )
    .group_by("bucket").agg([pl.len().alias("n_rank"), pl.col("far").mean().round(3).alias("far_rank")]).sort("bucket")
)
print("--- expected filled count in ranking LIRF matched if training operator x hour x segment fill rates apply")
exp = (
    rkm.join(cell.select(["AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "hour_utc", pl.col("fill").alias("cell_fill")]),
             on=["AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "hour_utc"], how="left")
    .join(train_ops.select(["AIRCRAFT_OPERATOR_flt", pl.col("fill_train")]), on="AIRCRAFT_OPERATOR_flt", how="left")
)
print("  rows with a matching train cell:", exp["cell_fill"].is_not_null().sum(), "of", rkm.height,
      "| sum(cell_fill) over those:", round(exp["cell_fill"].sum(), 1),
      "| sum(operator fill) over rows with seen operator:", round(exp["fill_train"].sum(), 1),
      "| flat 17.8 %-type rate applied to all rows:", round(rkm.height * p, 1))
print("  ranking LIRF matched: share far (|AOBT_3-SCHED|>600):", round(rkm["far"].mean(), 4),
      "| training LIRF matched:", round(lm["far"].mean(), 4))
print("  ranking LIRF matched by segment:", rkm.group_by("MARKET_SEGMENT_flt").agg(pl.len().alias("n")).sort("n", descending=True))

# ---------------------------------------------------------------- J. OOF by airport (matched rows), LIRF subsets
hdr("J. OOF main-48c19bc830 joined on MVT_ID_mvt: RMSE and SSE share by ADEP x matched; LIRF matched subsets")
oof = pl.scan_parquet(OOF)
print("oof rows:", oof.select(pl.len()).collect().item(),
      "| distinct MVT_ID:", oof.select(pl.col("MVT_ID_mvt").n_unique()).collect().item())
j = (
    tr.select(["MVT_ID_mvt", "ADEP_mvt", "matched", "fill", "far", "y", "proxy", "resid", "as_", "blk_minus_aobt",
               "AIRCRAFT_OPERATOR_flt", "hour_utc"])
    .join(oof, on="MVT_ID_mvt", how="inner")
    .filter(pl.col("oof").is_not_null())
    .with_columns((pl.col("oof") - pl.col("y")).alias("err"))
    .collect()
)
print("joined rows with non-null oof:", j.height)
tot_sse = (j["err"] ** 2).sum()
print("OOF RMSE all:", round(((j["err"] ** 2).mean()) ** 0.5, 2))
by_ap = (
    j.group_by(["ADEP_mvt", "matched"])
    .agg([pl.len().alias("n"), (pl.col("err") ** 2).mean().sqrt().round(1).alias("rmse"),
          (pl.col("err") ** 2).sum().alias("sse")])
    .with_columns((pl.col("sse") / tot_sse * 100).round(2).alias("sse_pct"))
    .sort(["matched", "rmse"], descending=[False, True])
)
print(by_ap)
lj = j.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("matched"))
print("--- LIRF matched OOF by fill x far")
print(
    lj.group_by(["fill", "far"])
    .agg([pl.len().alias("n"), (pl.col("err") ** 2).mean().sqrt().round(1).alias("rmse"),
          (pl.col("err") ** 2).sum().alias("sse"), pl.col("err").median().round(1).alias("err_med"),
          pl.col("y").median().alias("y_med"), pl.col("oof").median().round(0).alias("oof_med")])
    .with_columns([(pl.col("sse") / tot_sse * 100).round(2).alias("sse_pct_total"),
                   (pl.col("sse") / (lj["err"] ** 2).sum() * 100).round(1).alias("sse_pct_LIRFmatched")])
    .sort(["fill", "far"])
)
print("LIRF matched OOF RMSE excluding filled rows:", round(((lj.filter(~pl.col("fill"))["err"] ** 2).mean()) ** 0.5, 1),
      "| excluding filled&far:", round(((lj.filter(~(pl.col("fill") & pl.col("far")))["err"] ** 2).mean()) ** 0.5, 1))
print("--- matched OOF RMSE by airport excluding filled rows (for ranking comparison)")
print(
    j.filter(pl.col("matched"))
    .group_by("ADEP_mvt")
    .agg([pl.len().alias("n"), (pl.col("err") ** 2).mean().sqrt().round(1).alias("rmse_all"),
          (pl.col("err").filter(~pl.col("fill")) ** 2).mean().sqrt().round(1).alias("rmse_nofill"),
          (pl.col("err").filter(~(pl.col("fill") & pl.col("far"))) ** 2).mean().sqrt().round(1).alias("rmse_no_fillfar")])
    .sort("rmse_all", descending=True)
)
print("--- LIRF matched OOF: |BLOCK-AOBT_3|>1h rows")
gg = lj.filter(pl.col("blk_minus_aobt").abs() > 3600)
print(gg.select([pl.len().alias("n"), (pl.col("err") ** 2).mean().sqrt().round(1).alias("rmse"),
                 ((pl.col("err") ** 2).sum() / tot_sse * 100).round(2).alias("sse_pct_total"),
                 pl.col("fill").mean().round(3).alias("fill")]))
print("--- LIRF matched OOF: does |oof - y| track |as_| on filled&far rows? (correlation of err with as_, and with resid)")
ff = lj.filter(pl.col("fill") & pl.col("far")).with_columns((pl.col("oof") - pl.col("proxy")).alias("oof_minus_proxy"))
print(
    ff.select(
        [pl.len().alias("n"), pl.corr("err", "as_").round(3).alias("corr(err,as_)"),
         pl.corr("oof_minus_proxy", "as_").round(3).alias("corr(oof-proxy,as_)"),
         pl.col("oof_minus_proxy").median().alias("med(oof-proxy)"), pl.col("as_").median().alias("med as_"),
         pl.col("resid").median().alias("med resid"), pl.col("err").median().alias("med err")]
    )
)
print("--- same for LIRF matched NOT filled & far (control)")
nf = lj.filter(~pl.col("fill") & pl.col("far")).with_columns((pl.col("oof") - pl.col("proxy")).alias("oof_minus_proxy"))
print(nf.select([pl.len().alias("n"), pl.corr("err", "as_").round(3).alias("corr(err,as_)"),
                 pl.corr("oof_minus_proxy", "as_").round(3).alias("corr(oof-proxy,as_)"),
                 pl.col("oof_minus_proxy").median().alias("med(oof-proxy)"), pl.col("resid").median().alias("med resid")]))

hdr("K. OOF file coverage check (plan §0 says 672,428 rows)")
print(oof.select([pl.len().alias("rows"), pl.col("oof").is_null().sum().alias("oof_null"),
                  pl.col("oof").is_not_null().sum().alias("oof_notnull")]).collect())
print(
    tr.select(["MVT_ID_mvt", "month"]).join(oof, on="MVT_ID_mvt", how="inner")
    .group_by("month").agg([pl.len().alias("n"), pl.col("oof").is_not_null().sum().alias("oof_notnull")]).sort("month").collect()
)
print("done.")

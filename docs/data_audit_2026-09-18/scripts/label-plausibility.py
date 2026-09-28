#!/usr/bin/env python
"""Label plausibility / row-identity audit (lens 6 of the 2026-09-18 data census).

No training. POLARS_MAX_THREADS=1. Lazy scans, minimal columns.

Question: is each label physically possible, and is each row the flight it claims to be?

Inputs (read-only):
  data/raw/training_*.parquet   (12 files, DEP + ARR rows, 2025)
  data/raw/ranking.parquet      (DEP rows = ranking targets, label null; ARR rows LABELLED, 2026)
  data/external/airports.parquet

Run:  POLARS_MAX_THREADS=1 .venv/bin/python docs/data_audit_2026-09-18/scripts/label-plausibility.py
"""

import glob
import math
import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(250)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN = sorted(glob.glob(os.path.join(ROOT, "data/raw/training_*.parquet")))
RANK = os.path.join(ROOT, "data/raw/ranking.parquet")
APT = os.path.join(ROOT, "data/external/airports.parquet")

COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt",
    "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt",
    "CALLSIGN_flt", "MARKET_SEGMENT_flt", "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt", "ARVT_3_flt",
]


def hdr(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def sub(s):
    print("\n--- " + s)


# ---------------------------------------------------------------- load
tr = pl.scan_parquet(TRAIN).select(COLS)
rk = pl.scan_parquet(RANK).select(COLS)

FLAGS = [
    # |BLOCK - SCHED| <= 10 s  (plan.md §0 f10 == "sched-filled", LIRF pathology)
    (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs().alias("bs_abs"),
    # date(MVT) - date(BLOCK) in days
    (pl.col("MVT_TIME_UTC_mvt").dt.date() - pl.col("BLOCK_TIME_UTC_mvt").dt.date()).dt.total_days().alias("daydiff"),
]

dep = (
    tr.filter(pl.col("PHASE_mvt") == "DEP")
    .with_columns(FLAGS)
    .with_columns(
        y=pl.col("TAXITIME_SEC_mvt"),
        f10=pl.col("bs_abs") <= 10,
        # the established rollover set: y > 20 h AND date(MVT)-date(BLOCK) == 1
        rollover=(pl.col("TAXITIME_SEC_mvt") > 72000) & (pl.col("daydiff") >= 1),
        unm=pl.col("FLIGHT_ID_mvt").is_null(),
        month=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
    )
    .collect()
)
arr = (
    tr.filter(pl.col("PHASE_mvt") == "ARR")
    .select("ADES_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt",
            "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "FLIGHT_ID_mvt", "AIRCRAFT_TYPE_mvt",
            "AIRCRAFT_OPERATOR_flt", "MVT_ID_mvt")
    .collect()
)
rdep = rk.filter(pl.col("PHASE_mvt") == "DEP").collect()
rarr = (
    rk.filter(pl.col("PHASE_mvt") == "ARR")
    .select("ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt")
    .collect()
)

hdr("0. SHAPES AND SCOPE")
print("train DEP", dep.height, " train ARR", arr.height,
      " rank DEP", rdep.height, " rank ARR", rarr.height)
print("rank DEP label nulls", rdep["TAXITIME_SEC_mvt"].null_count(),
      " rank DEP BLOCK nulls", rdep["BLOCK_TIME_UTC_mvt"].null_count())
print("rank ARR label nulls", rarr["TAXITIME_SEC_mvt"].null_count())
print("train DEP months", dep["month"].min(), "->", dep["month"].max())
print("rank months", rdep.select(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m")).to_series().unique().sort().to_list())
print("label identity y == MVT-BLOCK:",
      dep.select(((pl.col("MVT_TIME_UTC_mvt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds()
                  == pl.col("y")).sum()).item(), "/", dep.height)
print("ARR identity taxiin == BLOCK-MVT:",
      arr.select(((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds()
                  == pl.col("TAXITIME_SEC_mvt")).sum()).item(), "/", arr.height)

hdr("A. LABEL DISTRIBUTION, SHORT TAIL THRESHOLDS")
print(dep.select(
    [pl.col("y").quantile(q).alias(f"q{q}") for q in (0, 1e-4, 1e-3, 5e-3, .01, .05, .25, .5, .75, .95, .99, 1)]
))
for t in (0, 1, 30, 60, 90, 120, 150, 180, 240, 300):
    print(f"  y < {t:5d}: {dep.filter(pl.col('y') < t).height:7d}")
print("negative label value_counts (all):")
print(dep.filter(pl.col("y") < 0)["y"].value_counts().sort("y"))

# ---------------------------------------------------------------- B. per stand x runway
hdr("B. PHYSICAL PLAUSIBILITY PER STAND x RUNWAY")
print("NOTE airports.parquet has airport-centroid lat/lon only (columns icao,lat,lon);")
print("     it contains no stand or runway geometry, so no stand->runway distance can be")
print("     computed from it. The physical floor below is derived empirically instead:")
print("     (a) the DEP label distribution of the same (ADEP,STAND,RUNWAY) pair, and")
print("     (b) the ARR taxi-IN distribution at the same (airport,STAND) -- an independent")
print("         traversal of the same apron-to-runway-system path.")

pair = (
    dep.filter(pl.col("STAND_mvt").is_not_null())
    .group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
    .agg(n=pl.len(), med=pl.col("y").median(), p05=pl.col("y").quantile(.05),
         p01=pl.col("y").quantile(.01), mn=pl.col("y").min())
)
print("\n(ADEP,STAND,RUNWAY) pairs:", pair.height,
      " with n>=20:", pair.filter(pl.col("n") >= 20).height,
      " rows covered by n>=20 pairs:", pair.filter(pl.col("n") >= 20)["n"].sum())
big = pair.filter(pl.col("n") >= 20)
print("distribution of pair medians:")
print(big.select([pl.col("med").quantile(q).alias(f"q{q}") for q in (0, .01, .05, .25, .5, .75, .95, 1)]))
print("pairs (n>=20) with median < 120 s:")
print(big.filter(pl.col("med") < 120).sort("n", descending=True))

# how much of the label does the pair explain (variance decomposition, n>=20 pairs)
dj = dep.join(big.select("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "med", "n"),
              on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="inner")
dj = dj.filter(pl.col("y").is_between(-100, 7200))  # exclude the known pathological tail
print("\nvariance decomposition on n>=20 pairs, y in [-100,7200] (", dj.height, "rows ):")
gm = dj["y"].mean()
tot = ((dj["y"] - gm) ** 2).sum()
wit = dj.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]).agg(
    ss=((pl.col("y") - pl.col("y").mean()) ** 2).sum())["ss"].sum()
print(f"  total SS {tot:.4g}  within-pair SS {wit:.4g}  between-pair share {1 - wit / tot:.4f}")

sub("B1. rows too short to be a stand->runway traverse: y < 60 s")
short = dep.filter(pl.col("y") < 60)
print("n =", short.height)
print(short.group_by(["ADEP_mvt", "RUNWAY_mvt"]).agg(
    n=pl.len(), neg=(pl.col("y") < 0).sum()).sort("n", descending=True).head(20))
print("\nby FLIGHT_RULE_mvt:")
print(short.group_by("FLIGHT_RULE_mvt").agg(n=pl.len(), neg=(pl.col("y") < 0).sum()))
print("\nby MARKET_SEGMENT_flt:")
print(short.group_by("MARKET_SEGMENT_flt").agg(n=pl.len()).sort("n", descending=True))

sub("B2. RUNWAY_mvt == 'H' (helipad)")
h_tr = dep.filter(pl.col("RUNWAY_mvt") == "H")
print("training DEP rows with RUNWAY_mvt=='H':", h_tr.height,
      " airports:", h_tr["ADEP_mvt"].unique().to_list())
print(h_tr.select([pl.col("y").quantile(q).alias(f"q{q}") for q in (0, .05, .25, .5, .75, .95, 1)]))
print(" y<60:", h_tr.filter(pl.col("y") < 60).height, " y<=0:", h_tr.filter(pl.col("y") <= 0).height)
print(" top aircraft types:", h_tr["AIRCRAFT_TYPE_mvt"].value_counts().sort("count", descending=True).head(8).to_dicts())
print(" stands:", h_tr["STAND_mvt"].value_counts().sort("count", descending=True).head(8).to_dicts())
print(" LSZH all runways, DEP label median / share y<60:")
print(dep.filter(pl.col("ADEP_mvt") == "LSZH").group_by("RUNWAY_mvt").agg(
    n=pl.len(), med=pl.col("y").median(), lt60=(pl.col("y") < 60).sum()).sort("n", descending=True))
print(" ARR rows landing on 'H':", arr.filter(pl.col("RUNWAY_mvt") == "H").height,
      " median taxi-in:", arr.filter(pl.col("RUNWAY_mvt") == "H")["TAXITIME_SEC_mvt"].median())
print(" RANKING DEP rows with RUNWAY_mvt=='H':", rdep.filter(pl.col("RUNWAY_mvt") == "H").height,
      rdep.filter(pl.col("RUNWAY_mvt") == "H")["ADEP_mvt"].unique().to_list())
print(" RANKING ARR rows on 'H':", rarr.filter(pl.col("RUNWAY_mvt") == "H").height,
      " median taxi-in:", rarr.filter(pl.col("RUNWAY_mvt") == "H")["TAXITIME_SEC_mvt"].median())

sub("B3. short rows after removing runway H and VFR")
s2 = short.filter((pl.col("RUNWAY_mvt") != "H") & (pl.col("FLIGHT_RULE_mvt") != "V"))
print("n =", s2.height, " negative:", s2.filter(pl.col("y") < 0).height)
print(s2.group_by(["ADEP_mvt"]).agg(n=pl.len(), neg=(pl.col("y") < 0).sum(),
                                    unm=pl.col("unm").sum()).sort("n", descending=True))
print(s2.group_by(["ADEP_mvt", "STAND_mvt"]).agg(n=pl.len()).sort("n", descending=True).head(15))
print("aircraft types:", s2["AIRCRAFT_TYPE_mvt"].value_counts().sort("count", descending=True).head(12).to_dicts())

sub("B4. short rows against their own pair median and against ARR taxi-in at the same stand")
stand_arr = (
    arr.filter(pl.col("STAND_mvt").is_not_null())
    .group_by(["ADES_mvt", "STAND_mvt"])
    .agg(n_arr=pl.len(), arr_med=pl.col("TAXITIME_SEC_mvt").median(),
         arr_p05=pl.col("TAXITIME_SEC_mvt").quantile(.05))
    .rename({"ADES_mvt": "ADEP_mvt"})
)
sj = (short.join(big.select("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "med", "n"),
                 on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
           .join(stand_arr, on=["ADEP_mvt", "STAND_mvt"], how="left"))
print("short rows whose own (stand,runway) pair has n>=20:", sj.filter(pl.col("n").is_not_null()).height)
print("  of those, pair median >= 300 s (i.e. the pair normally takes >=5 min):",
      sj.filter(pl.col("med") >= 300).height)
print("  pair median >= 600 s:", sj.filter(pl.col("med") >= 600).height)
print("short rows with ARR taxi-in stats at the same stand:", sj.filter(pl.col("n_arr").is_not_null()).height)
print("  of those, ARR p05 (taxi-IN 5th pct) >= 120 s:", sj.filter(pl.col("arr_p05") >= 120).height)
print("  ARR p05 >= 240 s:", sj.filter(pl.col("arr_p05") >= 240).height)
print("\n the 'hard-impossible' subset: y < 60 AND pair median >= 300 AND ARR p05 at stand >= 120")
hard = sj.filter((pl.col("med") >= 300) & (pl.col("arr_p05") >= 120))
print(" n =", hard.height)
print(hard.group_by(["ADEP_mvt", "RUNWAY_mvt"]).agg(n=pl.len(), ymin=pl.col("y").min(),
                                                    pairmed=pl.col("med").median()).sort("n", descending=True).head(15))
print(hard.select("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "y", "med", "arr_med",
                  "MVT_TIME_UTC_mvt", "unm").sort("y").head(20))

sub("B5. global 'y < 60 while the pair normally needs >= 5 min' rate per airport")
allj = dep.join(big.select("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "med"),
                on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="inner")
print(allj.group_by("ADEP_mvt").agg(
    n=pl.len(),
    lt60_pm300=((pl.col("y") < 60) & (pl.col("med") >= 300)).sum(),
    lt120_pm300=((pl.col("y") < 120) & (pl.col("med") >= 300)).sum(),
    lt_20pct=(pl.col("y") < 0.2 * pl.col("med")).sum(),
).sort("n", descending=True))

# ---------------------------------------------------------------- C. negative / near-zero
hdr("C. NEGATIVE AND NEAR-ZERO LABELS")
neg = dep.filter(pl.col("y") < 0)
nz = dep.filter(pl.col("y").is_between(0, 30, closed="left"))
print("negative:", neg.height, " min:", neg["y"].min(), " max:", neg["y"].max())
print("0 <= y < 30:", nz.height)
for name, df in (("NEG", neg), ("0-30", nz)):
    sub(f"{name}  n={df.height}")
    print(df.group_by(["ADEP_mvt", "RUNWAY_mvt"]).agg(n=pl.len()).sort("n", descending=True).head(10))
    print(" flight rule:", df["FLIGHT_RULE_mvt"].value_counts().sort("count", descending=True).to_dicts())
    print(" heli(H):", df.filter(pl.col("RUNWAY_mvt") == "H").height,
          " unmatched:", df["unm"].sum(),
          " operator null:", df["AIRCRAFT_OPERATOR_flt"].null_count(),
          " actype null:", df["AIRCRAFT_TYPE_mvt"].null_count())
    print(" top stands:", df["STAND_mvt"].value_counts().sort("count", descending=True).head(8).to_dicts())
    print(" top operators (hashed):", [(k[:10] if k else None, v) for k, v in
                                       df["AIRCRAFT_OPERATOR_flt"].value_counts()
                                       .sort("count", descending=True).head(6).rows()])
    print(" per month:", df["month"].value_counts().sort("month").to_dicts())
    print(" per hour-of-day:", df.select(pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("h"))["h"]
          .value_counts().sort("h").to_dicts())
    print(" distinct dates:", df.select(pl.col("MVT_TIME_UTC_mvt").dt.date()).to_series().n_unique())
    print(" max rows on a single date:", df.select(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("d"))["d"]
          .value_counts().sort("count", descending=True).head(3).to_dicts())
    print(" f10 (|BLOCK-SCHED|<=10):", df["f10"].sum(),
          " BLOCK sec-of-min==0:", df.select((pl.col("BLOCK_TIME_UTC_mvt").dt.second() == 0).sum()).item(),
          " MVT sec-of-min==0:", df.select((pl.col("MVT_TIME_UTC_mvt").dt.second() == 0).sum()).item())

sub("C1. is the negative range bounded?")
print("negative labels are confined to [", neg["y"].min(), ",", neg["y"].max(), "]")
print("counts of y in [-20,20]:")
print(dep.filter(pl.col("y").is_between(-20, 20))["y"].value_counts().sort("y"))
print("\nARR side for comparison (taxi-in, training): negatives:",
      arr.filter(pl.col("TAXITIME_SEC_mvt") < 0).height, " min:", arr["TAXITIME_SEC_mvt"].min())
print("ARR negatives with runway H:", arr.filter((pl.col("TAXITIME_SEC_mvt") < 0) & (pl.col("RUNWAY_mvt") == "H")).height)
print("ARR taxi-in value counts in [-20,0]:")
print(arr.filter(pl.col("TAXITIME_SEC_mvt").is_between(-20, 0))["TAXITIME_SEC_mvt"].value_counts().sort("TAXITIME_SEC_mvt"))

sub("C2. does the short-label signature survive into the ranking set?")
sig = {
    "ADEP=LSZH & RUNWAY=H": (pl.col("ADEP_mvt") == "LSZH") & (pl.col("RUNWAY_mvt") == "H"),
    "FLIGHT_RULE_mvt == V": pl.col("FLIGHT_RULE_mvt") == "V",
    "ADEP==ADES (local)": pl.col("ADEP_mvt") == pl.col("ADES_mvt"),
    "LSZH stand MFG": (pl.col("ADEP_mvt") == "LSZH") & (pl.col("STAND_mvt") == "MFG"),
    "LSZH stands HPW/REG/HAN": (pl.col("ADEP_mvt") == "LSZH") & (pl.col("STAND_mvt").is_in(["HPW", "REG", "HAN"])),
    "LFPG stand P73/P82": (pl.col("ADEP_mvt") == "LFPG") & (pl.col("STAND_mvt").is_in(["P73", "P82"])),
    "EHAM stand HG01": (pl.col("ADEP_mvt") == "EHAM") & (pl.col("STAND_mvt") == "HG01"),
    "STAND_mvt null": pl.col("STAND_mvt").is_null(),
    "STAND_mvt == 'UNKNOWN'": pl.col("STAND_mvt") == "UNKNOWN",
}
rows = []
for k, e in sig.items():
    t = dep.filter(e)
    r = rdep.filter(e)
    rows.append({
        "signature": k, "train_n": t.height,
        "train_med_y": None if t.height == 0 else float(t["y"].median()),
        "train_lt60": t.filter(pl.col("y") < 60).height,
        "train_lt60_rate": None if t.height == 0 else round(t.filter(pl.col("y") < 60).height / t.height, 4),
        "rank_n": r.height,
    })
print(pl.DataFrame(rows))

# ---------------------------------------------------------------- D. long tail
hdr("D. LONG TAIL NOT EXPLAINED BY THE TWO KNOWN PATHOLOGIES")
print("rollover set as established (y > 72000 and date(MVT)-date(BLOCK) >= 1):",
      dep.filter(pl.col("rollover")).height)
print("rows with date(MVT)-date(BLOCK) >= 1 at any y:", dep.filter(pl.col("daydiff") >= 1).height,
      "(most are ordinary midnight crossings)")
print("daydiff value counts:", dep["daydiff"].value_counts().sort("daydiff").to_dicts())
for lo in (3600, 7200, 14400, 43200, 72000):
    t = dep.filter(pl.col("y") >= lo)
    print(f"y >= {lo:6d}: n={t.height:5d}  f10={t['f10'].sum():5d}  rollover={t['rollover'].sum():3d} "
          f" neither={t.filter(~pl.col('f10') & ~pl.col('rollover')).height:5d} "
          f" LIRF={t.filter(pl.col('ADEP_mvt') == 'LIRF').height:5d}")

unexp = dep.filter((pl.col("y") >= 7200) & ~pl.col("f10") & ~pl.col("rollover"))
sub(f"D1. unexplained long rows, y >= 7200 s: n={unexp.height}")
print(unexp.group_by("ADEP_mvt").agg(n=pl.len(), med=pl.col("y").median(), mx=pl.col("y").max(),
                                     unm=pl.col("unm").sum()).sort("n", descending=True))
print(" market segment:", unexp["MARKET_SEGMENT_flt"].value_counts().sort("count", descending=True).to_dicts())
print(" per month:", unexp["month"].value_counts().sort("month").to_dicts())
print(" max on any single date:", unexp.select(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("d"))["d"]
      .value_counts().sort("count", descending=True).head(5).to_dicts())
print(" BLOCK-SCHED seconds quantiles (signed):")
print(unexp.select((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("bs"))
      .select([pl.col("bs").quantile(q).alias(f"q{q}") for q in (0, .05, .25, .5, .75, .95, 1)]))
print(" of these, how many have AOBT_3 present and |BLOCK-AOBT_3| > 3600:",
      unexp.filter((pl.col("AOBT_3_flt").is_not_null()) &
                   ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().abs() > 3600)).height,
      " (AOBT_3 present:", unexp.filter(pl.col("AOBT_3_flt").is_not_null()).height, ")")

sub("D2. unexplained long rows y >= 14400 (listed)")
print(dep.filter((pl.col("y") >= 14400) & ~pl.col("f10") & ~pl.col("rollover"))
      .select("ADEP_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "y",
              "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "unm").sort("y", descending=True))

sub("D3. same-day repeated BLOCK: is a long label a stale BLOCK from an earlier attempt?")
# a DEP row whose BLOCK precedes a *different* DEP row of the same (ADEP, FLIGHT) on that day
tw = (dep.filter(pl.col("FLIGHT_mvt").is_not_null())
      .group_by(["ADEP_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"])
      .agg(k=pl.len()))
dtw = dep.join(tw, on=["ADEP_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"], how="left")
print("DEP rows in an (ADEP,FLIGHT,SCHED) group of size >= 2:", dtw.filter(pl.col("k") >= 2).height,
      " groups:", tw.filter(pl.col("k") >= 2).height,
      " group sizes:", tw.filter(pl.col("k") >= 2)["k"].value_counts().sort("k").to_dicts())
print(" of the unexplained y>=7200 rows, how many are in such a group:",
      dtw.filter((pl.col("y") >= 7200) & ~pl.col("f10") & ~pl.col("rollover") & (pl.col("k") >= 2)).height)

# ---------------------------------------------------------------- E. identity
hdr("E. IDENTITY")

sub("E1. key uniqueness")
allt = pl.concat([dep.select("MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt"),
                  arr.select("MVT_ID_mvt", "FLIGHT_ID_mvt").with_columns(PHASE_mvt=pl.lit("ARR"))])
print("training rows", allt.height, " MVT_ID unique", allt["MVT_ID_mvt"].n_unique(),
      " MVT_ID null", allt["MVT_ID_mvt"].null_count())
print("ranking rows", rdep.height + rarr.height, " rank DEP MVT_ID unique", rdep["MVT_ID_mvt"].n_unique())
print("MVT_ID overlap train/rankDEP:",
      len(set(rdep["MVT_ID_mvt"].to_list()) & set(allt["MVT_ID_mvt"].to_list())))
print("FLIGHT_ID: nunique", allt["FLIGHT_ID_mvt"].n_unique(), " null", allt["FLIGHT_ID_mvt"].null_count())
g = (allt.filter(pl.col("FLIGHT_ID_mvt").is_not_null())
     .group_by("FLIGHT_ID_mvt")
     .agg(nd=(pl.col("PHASE_mvt") == "DEP").sum(), na=(pl.col("PHASE_mvt") == "ARR").sum()))
print("FLIGHT_ID multiplicity (nd, na):")
print(g.group_by(["nd", "na"]).agg(n=pl.len()).sort("n", descending=True))
dupfid = g.filter(pl.col("nd") >= 2)
print("\nFLIGHT_IDs carrying 2 DEP rows:", dupfid.height, " -- listed:")
dl = dep.join(dupfid.select("FLIGHT_ID_mvt"), on="FLIGHT_ID_mvt").sort("FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt")
print(dl.select("FLIGHT_ID_mvt", "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "STAND_mvt",
                "RUNWAY_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "y"))
print(" are these also (ADEP,FLIGHT,SCHED) twin-group members?",
      dl.join(tw.filter(pl.col("k") >= 2), on=["ADEP_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"], how="semi").height,
      "/", dl.height)

sub("E2. CALLSIGN_flt collisions -- same callsign, two DEP rows with overlapping [BLOCK, MVT]")
cs = (dep.filter(pl.col("CALLSIGN_flt").is_not_null())
      .select("CALLSIGN_flt", "ADEP_mvt", "MVT_ID_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "y", "unm")
      .sort("CALLSIGN_flt", "BLOCK_TIME_UTC_mvt"))
cs = cs.with_columns(
    prev_mvt=pl.col("MVT_TIME_UTC_mvt").shift(1).over("CALLSIGN_flt"),
    prev_blk=pl.col("BLOCK_TIME_UTC_mvt").shift(1).over("CALLSIGN_flt"),
    prev_adep=pl.col("ADEP_mvt").shift(1).over("CALLSIGN_flt"),
    prev_id=pl.col("MVT_ID_mvt").shift(1).over("CALLSIGN_flt"),
)
ov = cs.filter(pl.col("prev_mvt").is_not_null() & (pl.col("BLOCK_TIME_UTC_mvt") < pl.col("prev_mvt")))
print("overlapping DEP ground windows under one callsign:", ov.height)
print("  same airport:", ov.filter(pl.col("ADEP_mvt") == pl.col("prev_adep")).height,
      " different airport (aircraft in two places):",
      ov.filter(pl.col("ADEP_mvt") != pl.col("prev_adep")).height)
print(ov.filter(pl.col("ADEP_mvt") != pl.col("prev_adep"))
      .select("CALLSIGN_flt", "prev_adep", "ADEP_mvt", "prev_blk", "prev_mvt",
              "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "y").head(20))
print("  of the same-airport overlaps, how many are twin-group rows:",
      ov.filter(pl.col("ADEP_mvt") == pl.col("prev_adep"))
        .join(dtw.filter(pl.col("k") >= 2).select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi").height)

sub("E3. same stand, overlapping pushback windows (two aircraft on one stand)")
st = (dep.filter(pl.col("STAND_mvt").is_not_null())
      .select("ADEP_mvt", "STAND_mvt", "MVT_ID_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt",
              "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "y", "AIRCRAFT_TYPE_mvt")
      .sort("ADEP_mvt", "STAND_mvt", "BLOCK_TIME_UTC_mvt"))
st = st.with_columns(
    prev_mvt=pl.col("MVT_TIME_UTC_mvt").shift(1).over(["ADEP_mvt", "STAND_mvt"]),
    prev_blk=pl.col("BLOCK_TIME_UTC_mvt").shift(1).over(["ADEP_mvt", "STAND_mvt"]),
    prev_fl=pl.col("FLIGHT_mvt").shift(1).over(["ADEP_mvt", "STAND_mvt"]),
    prev_sched=pl.col("SCHED_TIME_UTC_mvt").shift(1).over(["ADEP_mvt", "STAND_mvt"]),
    prev_type=pl.col("AIRCRAFT_TYPE_mvt").shift(1).over(["ADEP_mvt", "STAND_mvt"]),
)
so = st.filter(pl.col("prev_mvt").is_not_null() & (pl.col("BLOCK_TIME_UTC_mvt") < pl.col("prev_mvt")))
print("DEP rows whose off-block precedes the take-off of the previous DEP from the same stand:", so.height)
print(" same FLIGHT_mvt as previous (i.e. the known return-to-stand twin shape):",
      so.filter(pl.col("FLIGHT_mvt") == pl.col("prev_fl")).height)
print(" different FLIGHT_mvt (two different flights overlapping on one stand):",
      so.filter((pl.col("FLIGHT_mvt") != pl.col("prev_fl"))).height)
print(so.group_by("ADEP_mvt").agg(n=pl.len(),
                                  same_flight=(pl.col("FLIGHT_mvt") == pl.col("prev_fl")).sum())
      .sort("n", descending=True))
diff = so.filter(pl.col("FLIGHT_mvt") != pl.col("prev_fl"))
print(" overlap seconds quantiles for the different-flight overlaps:")
print(diff.select((pl.col("prev_mvt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("ov"))
      .select([pl.col("ov").quantile(q).alias(f"q{q}") for q in (0, .25, .5, .75, .95, 1)]))
print(diff.select("ADEP_mvt", "STAND_mvt", "prev_fl", "FLIGHT_mvt", "prev_blk", "prev_mvt",
                  "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "y").head(15))

sub("E4. DEP row vs ARR row of the SAME FLIGHT_ID (the flight's own arrival)")
apair = arr.select("FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
                   "BLOCK_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "TAXITIME_SEC_mvt").rename(
    {c: c + "_a" for c in ("ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                           "AIRCRAFT_TYPE_mvt", "TAXITIME_SEC_mvt")})
leg = dep.join(apair.filter(pl.col("FLIGHT_ID_mvt").is_not_null()), on="FLIGHT_ID_mvt", how="inner")
leg = leg.with_columns(
    air=(pl.col("MVT_TIME_UTC_mvt_a") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds(),
    b2b=(pl.col("BLOCK_TIME_UTC_mvt_a") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds(),
)
print("DEP rows joined to an ARR row on FLIGHT_ID:", leg.height, "of", dep.height)
print(" ADEP disagrees (DEP.ADEP != ARR.ADEP):", (leg["ADEP_mvt"] != leg["ADEP_mvt_a"]).sum())
print(" ADES disagrees (DEP.ADES != ARR.ADES):", (leg["ADES_mvt"] != leg["ADES_mvt_a"]).sum())
print(" AIRCRAFT_TYPE disagrees:", (leg["AIRCRAFT_TYPE_mvt"] != leg["AIRCRAFT_TYPE_mvt_a"]).sum())
print(" of the ADEP-disagreeing rows, ARR.ADES == DEP.ADEP (i.e. the ARR row is the RETURN leg):",
      leg.filter((pl.col("ADEP_mvt") != pl.col("ADEP_mvt_a")) &
                 (pl.col("ADES_mvt_a") == pl.col("ADEP_mvt"))).height)
print(" airborne time (ARR take-off-to-landing) quantiles:")
print(leg.select([pl.col("air").quantile(q).alias(f"q{q}") for q in (0, .001, .01, .5, .99, 1)]))
print(" airborne <= 0 (lands before it takes off):", leg.filter(pl.col("air") <= 0).height,
      " < 300 s:", leg.filter(pl.col("air") < 300).height,
      " block-to-block <= 0:", leg.filter(pl.col("b2b") <= 0).height)
imp = leg.filter(pl.col("air") < 300)
print(imp.select("FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "ADEP_mvt_a", "ADES_mvt_a",
                 "MVT_TIME_UTC_mvt", "MVT_TIME_UTC_mvt_a", "air", "y", "unm").sort("air"))

sub("E5. implied ground speed from airport coordinates (only where ADEP/ADES both known)")
ap = pl.read_parquet(APT).unique(subset="icao")
legc = (leg.filter((pl.col("ADEP_mvt") == pl.col("ADEP_mvt_a")) & (pl.col("ADES_mvt") == pl.col("ADES_mvt_a"))
                   & (pl.col("ADEP_mvt") != pl.col("ADES_mvt")))
        .join(ap.rename({"icao": "ADEP_mvt", "lat": "lat1", "lon": "lon1"}), on="ADEP_mvt", how="inner")
        .join(ap.rename({"icao": "ADES_mvt", "lat": "lat2", "lon": "lon2"}), on="ADES_mvt", how="inner"))
R = 6371.0088
legc = legc.with_columns(
    gc=(2 * R * (
        ((pl.col("lat2") - pl.col("lat1")) * math.pi / 180 / 2).sin() ** 2
        + (pl.col("lat1") * math.pi / 180).cos() * (pl.col("lat2") * math.pi / 180).cos()
        * (((pl.col("lon2") - pl.col("lon1")) * math.pi / 180 / 2).sin() ** 2)
    ).sqrt().arcsin())
).with_columns(kt=pl.col("gc") / 1.852 / (pl.col("air") / 3600.0))
print("legs with both endpoints geocoded:", legc.height)
print(legc.select([pl.col("kt").quantile(q).alias(f"q{q}") for q in (0, .001, .01, .5, .99, .999, 1)]))
print(" implied ground speed > 700 kt:", legc.filter(pl.col("kt") > 700).height,
      "  > 1000 kt:", legc.filter(pl.col("kt") > 1000).height,
      "  <= 0 or non-finite:", legc.filter(~pl.col("kt").is_finite() | (pl.col("kt") <= 0)).height)
print(legc.filter((pl.col("kt") > 700) | (pl.col("kt") <= 0))
      .select("FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "gc", "air", "kt", "y").sort("kt", descending=True).head(15))

sub("E6. previous leg of the same aircraft: nearest preceding ARR at the same (airport, stand)")
ae = (arr.filter(pl.col("STAND_mvt").is_not_null())
      .select(apt="ADES_mvt", STAND_mvt="STAND_mvt", t="BLOCK_TIME_UTC_mvt",
              atype="AIRCRAFT_TYPE_mvt", aop="AIRCRAFT_OPERATOR_flt", aid="MVT_ID_mvt",
              a_taxiin="TAXITIME_SEC_mvt", a_rwy="RUNWAY_mvt")
      .drop_nulls("t").with_columns(a_t=pl.col("t")).sort("t"))
de = (dep.filter(pl.col("STAND_mvt").is_not_null())
      .select(apt="ADEP_mvt", STAND_mvt="STAND_mvt", t="BLOCK_TIME_UTC_mvt",
              dtype="AIRCRAFT_TYPE_mvt", dop="AIRCRAFT_OPERATOR_flt", did="MVT_ID_mvt",
              y="y", d_rwy="RUNWAY_mvt", unm="unm")
      .sort("t"))
chain = de.join_asof(ae, on="t", by=["apt", "STAND_mvt"], strategy="backward")
ok = chain.filter(pl.col("aid").is_not_null())
ok = ok.with_columns(turn=(pl.col("t") - pl.col("a_t")).dt.total_seconds())
print("DEP rows with a preceding same-stand ARR:", ok.height, "of", de.height)
print(" aircraft type matches the arriving aircraft:", (ok["dtype"] == ok["atype"]).sum(),
      f"({(ok['dtype'] == ok['atype']).sum() / ok.height:.3f})")
print(" operator matches:", (ok["dop"] == ok["aop"]).sum())
print(" turnaround (off-block - on-block) quantiles:")
print(ok.select([pl.col("turn").quantile(q).alias(f"q{q}") for q in (0, .001, .01, .05, .5, .95, .99, 1)]))
same = ok.filter((pl.col("dtype") == pl.col("atype")) & (pl.col("dop") == pl.col("aop")))
print(" same type AND operator (best same-aircraft proxy):", same.height)
print("   turnaround < 0:", same.filter(pl.col("turn") < 0).height,
      "  < 300 s:", same.filter(pl.col("turn") < 300).height,
      "  < 600 s:", same.filter(pl.col("turn") < 600).height)
print("   turnaround quantiles:")
print(same.select([pl.col("turn").quantile(q).alias(f"q{q}") for q in (0, .001, .01, .05, .5, .95, 1)]))
print(" label of DEP rows whose same-aircraft turnaround < 300 s:")
print(same.filter(pl.col("turn") < 300).select(
    n=pl.len(), ymed=pl.col("y").median(), ylt60=(pl.col("y") < 60).sum()))
print(" for reference all same-type/op rows: y median", same["y"].median())
print(" correlation between the DEP label and the preceding ARR taxi-in at the same stand:")
cc = same.filter(pl.col("y").is_between(0, 3600) & pl.col("a_taxiin").is_between(0, 3600))
print("   n =", cc.height, " pearson r =", round(float(pl.DataFrame(cc).select(
    pl.corr("y", "a_taxiin")).item()), 4))

# ---------------------------------------------------------------- F. 2025 vs 2026
hdr("F. 2025 vs 2026")
print("NOTE the ranking set's DEP rows carry NULL BLOCK_TIME and NULL TAXITIME_SEC,")
print("     so every label-based check above is training-only (2025). What can be compared")
print("     across the two years is (i) the composition of the DEP rows and (ii) the ARR")
print("     rows, which ARE labelled in the ranking file.")

sub("F1. DEP-row composition, 2025 (train) vs 2026 (rank), restricted to Jan+Jul")
dj25 = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7]))
print("train Jan+Jul DEP rows:", dj25.height, " rank DEP rows:", rdep.height)
for name, e in sig.items():
    a1 = dj25.filter(e).height / dj25.height
    a2 = rdep.filter(e).height / rdep.height
    print(f"  {name:28s}  2025 Jan+Jul {a1 * 100:7.4f}%   2026 {a2 * 100:7.4f}%   "
          f"(n {dj25.filter(e).height} / {rdep.filter(e).height})")

sub("F2. ARR (taxi-in) labels, 2025 vs 2026 -- the only labelled 2026 quantity")
a25 = arr.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7]))
for nm, df, apt_col in (("2025 Jan+Jul", a25, "ADES_mvt"), ("2026", rarr, "ADES_mvt")):
    print(f" {nm}: n={df.height}  median={df['TAXITIME_SEC_mvt'].median()} "
          f" <60s={df.filter(pl.col('TAXITIME_SEC_mvt') < 60).height} "
          f" <0={df.filter(pl.col('TAXITIME_SEC_mvt') < 0).height} "
          f" min={df['TAXITIME_SEC_mvt'].min()} max={df['TAXITIME_SEC_mvt'].max()}")
print(" per airport median taxi-in:")
m25 = a25.group_by("ADES_mvt").agg(n25=pl.len(), med25=pl.col("TAXITIME_SEC_mvt").median(),
                                   lt60_25=(pl.col("TAXITIME_SEC_mvt") < 60).mean())
m26 = rarr.group_by("ADES_mvt").agg(n26=pl.len(), med26=pl.col("TAXITIME_SEC_mvt").median(),
                                    lt60_26=(pl.col("TAXITIME_SEC_mvt") < 60).mean())
print(m25.join(m26, on="ADES_mvt", how="full", coalesce=True).sort("n25", descending=True))

sub("F3. negative ARR taxi-in, 2025 vs 2026 (bound of the negative range)")
print(" 2025 Jan+Jul ARR min:", a25["TAXITIME_SEC_mvt"].min(),
      " 2026 ARR min:", rarr["TAXITIME_SEC_mvt"].min())
print(" 2026 ARR negatives by airport:",
      rarr.filter(pl.col("TAXITIME_SEC_mvt") < 0).group_by("ADES_mvt").agg(pl.len()).sort("len", descending=True).to_dicts())
print(" 2026 ARR negative value counts:",
      rarr.filter(pl.col("TAXITIME_SEC_mvt").is_between(-30, 0))["TAXITIME_SEC_mvt"]
      .value_counts().sort("TAXITIME_SEC_mvt").to_dicts())

sub("F4. stand x runway pairs present in 2026 but never seen in 2025")
p25 = dep.select("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt").unique()
p26 = rdep.select("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt").unique()
newp = p26.join(p25, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="anti")
print(" 2025 pairs:", p25.height, " 2026 pairs:", p26.height, " 2026-only pairs:", newp.height)
cov = rdep.join(p25, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="anti")
print(" ranking DEP rows on a pair never seen in 2025:", cov.height,
      f"({cov.height / rdep.height * 100:.3f}%)")
print(cov.group_by("ADEP_mvt").agg(pl.len()).sort("len", descending=True))
newst = rdep.join(dep.select("ADEP_mvt", "STAND_mvt").unique(), on=["ADEP_mvt", "STAND_mvt"], how="anti")
print(" ranking DEP rows on a STAND never seen in 2025:", newst.height)
print(newst.group_by("ADEP_mvt").agg(pl.len()).sort("len", descending=True))

sub("F5. identity checks repeated on the 2026 ranking file")
print(" rank DEP MVT_ID duplicated:", rdep.height - rdep["MVT_ID_mvt"].n_unique())
rg = (pl.concat([rdep.select("FLIGHT_ID_mvt", "PHASE_mvt"),
                 rk.filter(pl.col("PHASE_mvt") == "ARR").select("FLIGHT_ID_mvt", "PHASE_mvt").collect()])
      .filter(pl.col("FLIGHT_ID_mvt").is_not_null())
      .group_by("FLIGHT_ID_mvt")
      .agg(nd=(pl.col("PHASE_mvt") == "DEP").sum(), na=(pl.col("PHASE_mvt") == "ARR").sum()))
print(rg.group_by(["nd", "na"]).agg(n=pl.len()).sort("n", descending=True))
rtw = (rdep.filter(pl.col("FLIGHT_mvt").is_not_null())
       .group_by(["ADEP_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"]).agg(k=pl.len()))
print(" rank DEP rows in an (ADEP,FLIGHT,SCHED) group of size>=2:",
      rdep.join(rtw.filter(pl.col("k") >= 2), on=["ADEP_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"], how="semi").height)
rcs = (rdep.filter(pl.col("CALLSIGN_flt").is_not_null())
       .select("CALLSIGN_flt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
       .sort("CALLSIGN_flt", "MVT_TIME_UTC_mvt"))
rcs = rcs.with_columns(pm=pl.col("MVT_TIME_UTC_mvt").shift(1).over("CALLSIGN_flt"),
                       pa=pl.col("ADEP_mvt").shift(1).over("CALLSIGN_flt"))
print(" rank DEP: same callsign, two take-offs < 600 s apart at different airports:",
      rcs.filter(pl.col("pm").is_not_null()
                 & ((pl.col("MVT_TIME_UTC_mvt") - pl.col("pm")).dt.total_seconds() < 600)
                 & (pl.col("ADEP_mvt") != pl.col("pa"))).height)

hdr("DONE")

# ---------------------------------------------------------------- G. follow-ups
hdr("G. FOLLOW-UPS")

sub("G1. the +/-12 s band")
print("rows with |y| <= 12:", dep.filter(pl.col("y").abs() <= 12).height,
      "  y in [-12,-1]:", dep.filter(pl.col("y").is_between(-12, -1)).height,
      "  y in [0,12]:", dep.filter(pl.col("y").is_between(0, 12)).height,
      "  y in [13,24]:", dep.filter(pl.col("y").is_between(13, 24)).height)
print("ARR (training) taxi-in < -12:", arr.filter(pl.col("TAXITIME_SEC_mvt") < -12).height)
print(arr.filter(pl.col("TAXITIME_SEC_mvt") < -12).group_by(["ADES_mvt", "RUNWAY_mvt"]).agg(pl.len())
      .sort("len", descending=True))
print("ARR (training) taxi-in in [-12,-1]:", arr.filter(pl.col("TAXITIME_SEC_mvt").is_between(-12, -1)).height)

sub("G2. STAND_mvt is not a unique physical stand at EDDM")
print("stand-overlap rows (E3, different flights) per (ADEP, STAND), top 20:")
print(diff.group_by(["ADEP_mvt", "STAND_mvt"]).agg(n=pl.len()).sort("n", descending=True).head(20))
tot_by_stand = dep.group_by(["ADEP_mvt", "STAND_mvt"]).agg(tot=pl.len())
print("with the stand's total DEP volume:")
print(diff.group_by(["ADEP_mvt", "STAND_mvt"]).agg(n=pl.len())
      .join(tot_by_stand, on=["ADEP_mvt", "STAND_mvt"])
      .with_columns(rate=pl.col("n") / pl.col("tot")).sort("n", descending=True).head(15))

sub("G3. turnaround check E6 is one-sided by construction")
print("the asof join is 'backward', so turn >= 0 always; 'turnaround < 0' cannot be found this way.")
print("what IS informative: DEP type != preceding ARR type rate =",
      round(1 - (ok['dtype'] == ok['atype']).sum() / ok.height, 4))
print("per airport:")
print(ok.group_by("apt").agg(n=pl.len(), type_mismatch=(pl.col("dtype") != pl.col("atype")).mean(),
                             turn_med=pl.col("turn").median()).sort("n", descending=True))

sub("G4. 2026 stands with no 2025 history (cold start), per airport")
new_stand = (rdep.select("ADEP_mvt", "STAND_mvt").unique()
             .join(dep.select("ADEP_mvt", "STAND_mvt").unique(), on=["ADEP_mvt", "STAND_mvt"], how="anti"))
print("distinct (ADEP,STAND) new in 2026:", new_stand.height)
print(rdep.join(new_stand, on=["ADEP_mvt", "STAND_mvt"], how="semi")
      .group_by(["ADEP_mvt", "STAND_mvt"]).agg(n=pl.len()).sort("n", descending=True).head(20))

sub("G5. the 4 same-callsign overlapping DEP windows")
print(ov.select("CALLSIGN_flt", "ADEP_mvt", "prev_adep", "prev_blk", "prev_mvt",
                "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "y", "unm"))

sub("G6. ADEP==ADES local flights")
loc = dep.filter(pl.col("ADEP_mvt") == pl.col("ADES_mvt"))
print(loc.group_by("ADEP_mvt").agg(n=pl.len(), med=pl.col("y").median(), lt60=(pl.col("y") < 60).sum(),
                                   unm=pl.col("unm").sum()).sort("n", descending=True))
print("all-DEP median for comparison:", dep["y"].median())
print("ranking rows with ADEP==ADES:", rdep.filter(pl.col("ADEP_mvt") == pl.col("ADES_mvt")).height)

hdr("DONE G")

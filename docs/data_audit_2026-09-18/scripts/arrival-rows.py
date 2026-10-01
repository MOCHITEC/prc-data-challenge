"""Census of the ARRIVAL rows (PHASE_mvt == 'ARR') and of what they join to.

Lens: data_audit_2026-09-18, agent "arrival-rows".
No training. POLARS_MAX_THREADS=1. Lazy scans, minimal columns.
Reads only data/raw/*.parquet. Writes nothing.

Sections
  A  census: ARR vs DEP row counts per airport x month, training and ranking
  B  column population on ARR vs DEP (null rates), both sets
  C  taxi-in distribution per airport / stand / runway / hour
  D  the stand join: DEP -> most recent ARR at same airport+stand, coverage + gap
  E  inbound-leg identity: aircraft type / operator / callsign linkage
  F  signal check (correlations only, no training)
  G  pathologies on ARR rows: sched-fill, day rollover, negative durations
"""

from __future__ import annotations

import glob
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

RAW = "/home/user/prc-data-challenge/data/raw"
TRAIN_FILES = sorted(glob.glob(f"{RAW}/training_*.parquet"))
RANK_FILE = f"{RAW}/ranking.parquet"

MVT_COLS = [
    "MVT_ID_mvt",
    "FLIGHT_ID_mvt",
    "FLIGHT_mvt",
    "FLIGHT_RULE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "TAXITIME_SEC_mvt",
]
FLT_COLS = [
    "CALLSIGN_flt",
    "MARKET_SEGMENT_flt",
    "AIRCRAFT_TYPE_flt",
    "AIRCRAFT_OPERATOR_flt",
    "AOBT_3_flt",
    "ARVT_3_flt",
    "EOBT_1_flt",
    "LOBT_flt",
]
COLS = MVT_COLS + FLT_COLS


def hdr(s: str) -> None:
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def load(files: list[str]) -> pl.DataFrame:
    return pl.concat([pl.scan_parquet(f).select(COLS) for f in files]).collect()


def split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (dep, arr) with a uniform `apt` column (DEP -> ADEP, ARR -> ADES)."""
    dep = df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        pl.col("ADEP_mvt").alias("apt")
    )
    arr = df.filter(pl.col("PHASE_mvt") == "ARR").with_columns(
        pl.col("ADES_mvt").alias("apt")
    )
    return dep, arr


def q(s: pl.Series, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> list:
    s = s.drop_nulls()
    if s.len() == 0:
        return [None] * len(qs)
    return [round(float(s.quantile(x)), 1) for x in qs]


def corr(a: pl.Series, b: pl.Series) -> tuple[int, float | None]:
    m = a.is_not_null() & b.is_not_null() & a.is_finite() & b.is_finite()
    a2, b2 = a.filter(m), b.filter(m)
    n = a2.len()
    if n < 30 or a2.std() in (None, 0.0) or b2.std() in (None, 0.0):
        return n, None
    return n, round(float(np.corrcoef(a2.to_numpy(), b2.to_numpy())[0, 1]), 4)


print("loading training ...")
TR = load(TRAIN_FILES)
print("loading ranking ...")
RK = pl.scan_parquet(RANK_FILE).select(COLS).collect()

TR_DEP, TR_ARR = split(TR)
RK_DEP, RK_ARR = split(RK)
del TR, RK

# ----------------------------------------------------------------------------- A
hdr("A. ROW COUNTS")
for name, dep, arr in (("TRAIN", TR_DEP, TR_ARR), ("RANK", RK_DEP, RK_ARR)):
    print(f"{name}: DEP {dep.height:,}  ARR {arr.height:,}  ratio {arr.height/dep.height:.4f}")

print("\nA1. ARR rows per airport (train / rank) and ARR/DEP ratio")
a = (
    TR_ARR.group_by("apt").len().rename({"len": "tr_arr"})
    .join(TR_DEP.group_by("apt").len().rename({"len": "tr_dep"}), on="apt")
    .join(RK_ARR.group_by("apt").len().rename({"len": "rk_arr"}), on="apt")
    .join(RK_DEP.group_by("apt").len().rename({"len": "rk_dep"}), on="apt")
    .with_columns(
        (pl.col("tr_arr") / pl.col("tr_dep")).round(4).alias("tr_ratio"),
        (pl.col("rk_arr") / pl.col("rk_dep")).round(4).alias("rk_ratio"),
    )
    .sort("tr_arr", descending=True)
)
print(a)

print("\nA2. ARR rows per month (anchor = MVT_TIME = landing)")
for name, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    m = (
        arr.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"))
        .group_by("ym").len().sort("ym")
    )
    print(f"-- {name}"); print(m)

print("\nA3. ARR per airport x month, TRAIN (wide)")
print(
    TR_ARR.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%m").alias("mo"))
    .pivot(values="MVT_ID_mvt", index="apt", on="mo", aggregate_function="len")
    .sort("apt")
)
print("\nA3b. ARR per airport x month, RANK")
print(
    RK_ARR.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("mo"))
    .pivot(values="MVT_ID_mvt", index="apt", on="mo", aggregate_function="len")
    .sort("apt")
)

# ----------------------------------------------------------------------------- B
hdr("B. COLUMN POPULATION: null rate on ARR vs DEP, train vs rank")
rows = []
for c in COLS:
    if c == "PHASE_mvt":
        continue
    r = {"col": c}
    for tag, df in (("tr_dep", TR_DEP), ("tr_arr", TR_ARR), ("rk_dep", RK_DEP), ("rk_arr", RK_ARR)):
        r[tag] = round(df[c].null_count() / df.height, 5)
    rows.append(r)
print(pl.DataFrame(rows))

print("\nB2. distinct-value counts of the ARR-side join keys")
for tag, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    print(
        tag,
        "n_stand", arr["STAND_mvt"].n_unique(),
        "n_runway", arr["RUNWAY_mvt"].n_unique(),
        "n_actype", arr["AIRCRAFT_TYPE_mvt"].n_unique(),
        "n_operator", arr["AIRCRAFT_OPERATOR_flt"].n_unique(),
    )

print("\nB3. do DEP and ARR stand vocabularies agree? (per airport: |ARR stands|, |DEP stands|, |intersection|)")
for tag, dep, arr in (("TRAIN", TR_DEP, TR_ARR), ("RANK", RK_DEP, RK_ARR)):
    out = []
    for apt in sorted(arr["apt"].unique().to_list()):
        sa = set(arr.filter(pl.col("apt") == apt)["STAND_mvt"].drop_nulls().unique().to_list())
        sd = set(dep.filter(pl.col("apt") == apt)["STAND_mvt"].drop_nulls().unique().to_list())
        out.append({"apt": apt, "arr_stands": len(sa), "dep_stands": len(sd),
                    "both": len(sa & sd), "dep_only": len(sd - sa)})
    print(f"-- {tag}"); print(pl.DataFrame(out))

# ----------------------------------------------------------------------------- C
hdr("C. TAXI-IN DISTRIBUTION")
print("C0. identity check: TAXITIME_SEC_mvt == BLOCK_TIME - MVT_TIME on ARR rows?")
for tag, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    d = arr.select(
        (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().alias("calc"),
        pl.col("TAXITIME_SEC_mvt").alias("t"),
    )
    eq = (d["calc"] == d["t"]).sum()
    print(f"{tag}: exact {eq:,}/{arr.height:,}  null_taxi {arr['TAXITIME_SEC_mvt'].null_count()}")

for tag, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    print(f"\nC1. taxi-in per airport ({tag}): n, mean, p01..p99, share<0, share>3600")
    g = (
        arr.group_by("apt").agg(
            pl.len().alias("n"),
            pl.col("TAXITIME_SEC_mvt").mean().round(1).alias("mean"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.01).alias("p01"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.25).alias("p25"),
            pl.col("TAXITIME_SEC_mvt").median().alias("p50"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.75).alias("p75"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.95).alias("p95"),
            pl.col("TAXITIME_SEC_mvt").quantile(0.99).alias("p99"),
            pl.col("TAXITIME_SEC_mvt").max().alias("max"),
            (pl.col("TAXITIME_SEC_mvt") < 0).mean().round(5).alias("neg"),
            (pl.col("TAXITIME_SEC_mvt") > 3600).mean().round(5).alias("gt1h"),
        ).sort("apt")
    )
    print(g)

print("\nC2. taxi-in by hour of landing (TRAIN, median per airport x hour bucket)")
print(
    TR_ARR.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("h"))
    .group_by("h").agg(pl.len().alias("n"), pl.col("TAXITIME_SEC_mvt").median().alias("p50"),
                       pl.col("TAXITIME_SEC_mvt").mean().round(1).alias("mean")).sort("h")
)

print("\nC3. taxi-in by arrival runway (TRAIN, top 25 by n)")
print(
    TR_ARR.group_by(["apt", "RUNWAY_mvt"]).agg(
        pl.len().alias("n"), pl.col("TAXITIME_SEC_mvt").median().alias("p50"),
        pl.col("TAXITIME_SEC_mvt").quantile(0.9).alias("p90"),
    ).sort("n", descending=True).head(25)
)

print("\nC4. taxi-in by stand: dispersion of per-stand medians, per airport (stands with n>=30, TRAIN)")
st = (
    TR_ARR.group_by(["apt", "STAND_mvt"]).agg(
        pl.len().alias("n"), pl.col("TAXITIME_SEC_mvt").median().alias("p50")
    ).filter(pl.col("n") >= 30)
)
print(
    st.group_by("apt").agg(
        pl.len().alias("n_stands"),
        pl.col("p50").min().alias("min_stand_med"),
        pl.col("p50").quantile(0.25).alias("q25"),
        pl.col("p50").median().alias("med"),
        pl.col("p50").quantile(0.75).alias("q75"),
        pl.col("p50").max().alias("max_stand_med"),
        pl.col("p50").std().round(1).alias("sd_of_medians"),
    ).sort("apt")
)

print("\nC5. train vs rank taxi-in shift, per airport x month-of-year (Jan / Jul only)")
tr_j = TR_ARR.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7])).with_columns(
    pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
rk_j = RK_ARR.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
cmp = (
    tr_j.group_by(["apt", "mo"]).agg(pl.len().alias("n25"),
                                     pl.col("TAXITIME_SEC_mvt").median().alias("p50_25"),
                                     pl.col("TAXITIME_SEC_mvt").mean().round(1).alias("mean_25"))
    .join(rk_j.group_by(["apt", "mo"]).agg(pl.len().alias("n26"),
                                           pl.col("TAXITIME_SEC_mvt").median().alias("p50_26"),
                                           pl.col("TAXITIME_SEC_mvt").mean().round(1).alias("mean_26")),
          on=["apt", "mo"])
    .with_columns((pl.col("mean_26") - pl.col("mean_25")).round(1).alias("d_mean"),
                  (pl.col("p50_26") - pl.col("p50_25")).alias("d_p50"))
    .sort(["apt", "mo"])
)
print(cmp)

# ----------------------------------------------------------------------------- D
hdr("D. THE STAND JOIN  (DEP -> most recent ARR, same airport + same STAND, before DEP)")

# Departure anchor time: SCHED_TIME_UTC_mvt (scheduled off-block).  It is the only
# departure clock that is non-null in BOTH sets (BLOCK_TIME is null on all ranking DEP).
# Arrival anchor time: BLOCK_TIME_UTC_mvt (in-block), non-null in both sets.


def stand_join(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    d = (
        dep.select(
            pl.col("MVT_ID_mvt").alias("dep_id"),
            pl.col("apt"),
            pl.col("STAND_mvt").alias("stand"),
            pl.col("SCHED_TIME_UTC_mvt").alias("dep_sched"),
            pl.col("MVT_TIME_UTC_mvt").alias("dep_mvt"),
            pl.col("BLOCK_TIME_UTC_mvt").alias("dep_block"),
            pl.col("TAXITIME_SEC_mvt").alias("y"),
            pl.col("AIRCRAFT_TYPE_mvt").alias("d_actype"),
            pl.col("AIRCRAFT_OPERATOR_flt").alias("d_op"),
            pl.col("CALLSIGN_flt").alias("d_cs"),
            pl.col("FLIGHT_mvt").alias("d_flt"),
            pl.col("FLIGHT_ID_mvt").alias("d_fid"),
            pl.col("AOBT_3_flt").alias("d_aobt3"),
            pl.col("RUNWAY_mvt").alias("d_rwy"),
        )
        .with_columns(pl.col("d_aobt3").is_null().alias("unmatched_nm"))
        .drop_nulls(["dep_sched"])
        .sort("dep_sched")
    )
    a = (
        arr.select(
            pl.col("MVT_ID_mvt").alias("arr_id"),
            pl.col("apt"),
            pl.col("STAND_mvt").alias("stand"),
            pl.col("BLOCK_TIME_UTC_mvt").alias("arr_block"),
            pl.col("MVT_TIME_UTC_mvt").alias("arr_land"),
            pl.col("SCHED_TIME_UTC_mvt").alias("arr_sched"),
            pl.col("TAXITIME_SEC_mvt").alias("taxi_in"),
            pl.col("AIRCRAFT_TYPE_mvt").alias("a_actype"),
            pl.col("AIRCRAFT_OPERATOR_flt").alias("a_op"),
            pl.col("CALLSIGN_flt").alias("a_cs"),
            pl.col("FLIGHT_mvt").alias("a_flt"),
            pl.col("FLIGHT_ID_mvt").alias("a_fid"),
            pl.col("ADEP_mvt").alias("a_from"),
            pl.col("RUNWAY_mvt").alias("a_rwy"),
        )
        .drop_nulls(["arr_block"])
        .sort("arr_block")
    )
    j = d.join_asof(
        a, left_on="dep_sched", right_on="arr_block", by=["apt", "stand"], strategy="backward"
    )
    return j.with_columns(
        (pl.col("dep_sched") - pl.col("arr_block")).dt.total_seconds().alias("gap_s")
    )


print("D0. departure anchor availability")
for tag, dep in (("TRAIN", TR_DEP), ("RANK", RK_DEP)):
    print(tag, "SCHED null", dep["SCHED_TIME_UTC_mvt"].null_count(),
          "STAND null", dep["STAND_mvt"].null_count(),
          "AOBT_3 null (unmatched)", dep["AOBT_3_flt"].null_count())

TJ = stand_join(TR_DEP, TR_ARR)
RJ = stand_join(RK_DEP, RK_ARR)

print("\nD1. coverage: fraction of departures with ANY previous same-stand arrival, and within cut-offs")
CUTS = [3600, 2 * 3600, 4 * 3600, 6 * 3600, 12 * 3600, 24 * 3600, 72 * 3600]
rows = []
for tag, j in (("TRAIN", TJ), ("RANK", RJ)):
    for grp, sub in (("all", j), ("matched", j.filter(~pl.col("unmatched_nm"))),
                     ("unmatched", j.filter(pl.col("unmatched_nm")))):
        r = {"set": tag, "group": grp, "n_dep": sub.height,
             "any_match": round(sub["arr_id"].is_not_null().mean(), 4)}
        for c in CUTS:
            r[f"<={c//3600}h"] = round(float((sub["gap_s"].fill_null(10**9) <= c).mean()), 4)
        rows.append(r)
print(pl.DataFrame(rows))

print("\nD2. gap distribution (seconds) for matched-to-a-stand departures")
rows = []
for tag, j in (("TRAIN", TJ), ("RANK", RJ)):
    for grp, sub in (("all", j), ("matched", j.filter(~pl.col("unmatched_nm"))),
                     ("unmatched", j.filter(pl.col("unmatched_nm")))):
        s = sub["gap_s"].drop_nulls()
        qq = q(s, (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
        rows.append({"set": tag, "group": grp, "n": s.len(), "min": int(s.min()) if s.len() else None,
                     "p01": qq[0], "p05": qq[1], "p10": qq[2], "p25": qq[3], "p50": qq[4],
                     "p75": qq[5], "p90": qq[6], "p95": qq[7], "p99": qq[8],
                     "max": int(s.max()) if s.len() else None})
print(pl.DataFrame(rows))

print("\nD2b. how many gaps are NEGATIVE or zero (arrival in-block after/at dep sched off-block)?")
for tag, j in (("TRAIN", TJ), ("RANK", RJ)):
    s = j["gap_s"].drop_nulls()
    print(tag, "n", s.len(), "gap<0", int((s < 0).sum()), "gap==0", int((s == 0).sum()),
          "gap<300", int((s < 300).sum()))

print("\nD3. coverage per airport (any match, <=6h, <=12h), TRAIN and RANK")
rows = []
for tag, j in (("TRAIN", TJ), ("RANK", RJ)):
    g = j.group_by("apt").agg(
        pl.len().alias("n"),
        pl.col("arr_id").is_not_null().mean().round(4).alias("any"),
        (pl.col("gap_s").fill_null(10**9) <= 6 * 3600).mean().round(4).alias("le6h"),
        (pl.col("gap_s").fill_null(10**9) <= 12 * 3600).mean().round(4).alias("le12h"),
        pl.col("gap_s").median().alias("gap_p50"),
    ).with_columns(pl.lit(tag).alias("set")).sort("apt")
    rows.append(g)
print(pl.concat(rows).sort(["apt", "set"]))

print("\nD4. coverage on the rows that carry our error: NM-unmatched, per airport (TRAIN)")
print(
    TJ.filter(pl.col("unmatched_nm")).group_by("apt").agg(
        pl.len().alias("n"),
        pl.col("arr_id").is_not_null().mean().round(4).alias("any"),
        (pl.col("gap_s").fill_null(10**9) <= 6 * 3600).mean().round(4).alias("le6h"),
        pl.col("gap_s").median().alias("gap_p50"),
    ).sort("n", descending=True)
)
print("\nD4b. same, RANK")
print(
    RJ.filter(pl.col("unmatched_nm")).group_by("apt").agg(
        pl.len().alias("n"),
        pl.col("arr_id").is_not_null().mean().round(4).alias("any"),
        (pl.col("gap_s").fill_null(10**9) <= 6 * 3600).mean().round(4).alias("le6h"),
        pl.col("gap_s").median().alias("gap_p50"),
    ).sort("n", descending=True)
)

print("\nD5. why do departures miss? (TRAIN) stand null / stand unseen among arrivals / no earlier arrival")
for tag, j, arr in (("TRAIN", TJ, TR_ARR), ("RANK", RJ, RK_ARR)):
    seen = arr.select(pl.col("apt"), pl.col("STAND_mvt").alias("stand")).unique().with_columns(
        pl.lit(True).alias("stand_seen"))
    m = j.join(seen, on=["apt", "stand"], how="left")
    miss = m.filter(pl.col("arr_id").is_null())
    print(tag, "misses", miss.height,
          "| stand null", int(miss["stand"].is_null().sum()),
          "| stand never used by an arrival", int((miss["stand_seen"].is_null() & miss["stand"].is_not_null()).sum()),
          "| stand seen but no earlier arrival", int((miss["stand_seen"] == True).sum()))

# ----------------------------------------------------------------------------- E
hdr("E. DOES THE INBOUND LEG IDENTIFY THE OUTBOUND ONE?")
print("E0. is FLIGHT_ID_mvt ever shared between an ARR and a DEP row? (turnaround as one flight id)")
for tag, dep, arr in (("TRAIN", TR_DEP, TR_ARR), ("RANK", RK_DEP, RK_ARR)):
    fd = set(dep["FLIGHT_ID_mvt"].drop_nulls().unique().to_list())
    fa = set(arr["FLIGHT_ID_mvt"].drop_nulls().unique().to_list())
    print(tag, "dep fids", len(fd), "arr fids", len(fa), "shared", len(fd & fa))

print("\nE0b. does an ARR row's ADES_flt / a DEP row's ADEP_flt let the two legs be tied? "
      "(count of ARR rows whose ADES == airport)")
for tag, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    print(tag, "ARR rows with ADES_mvt == apt:", int((arr["ADES_mvt"] == arr["apt"]).sum()), "/", arr.height)

print("\nE1. among stand-matched departures: agreement of aircraft type / operator / callsign with the inbound leg")
rows = []
for tag, j in (("TRAIN", TJ), ("RANK", RJ)):
    for lo, hi, lbl in ((0, 3600, "0-1h"), (3600, 4 * 3600, "1-4h"), (4 * 3600, 12 * 3600, "4-12h"),
                        (12 * 3600, 10**12, ">12h")):
        sub = j.filter(pl.col("arr_id").is_not_null() & (pl.col("gap_s") >= lo) & (pl.col("gap_s") < hi))
        if sub.height == 0:
            continue
        both_t = sub.filter(pl.col("d_actype").is_not_null() & pl.col("a_actype").is_not_null())
        both_o = sub.filter(pl.col("d_op").is_not_null() & pl.col("a_op").is_not_null())
        rows.append({
            "set": tag, "gap": lbl, "n": sub.height,
            "n_actype_both": both_t.height,
            "actype_eq": round(float((both_t["d_actype"] == both_t["a_actype"]).mean()), 4) if both_t.height else None,
            "n_op_both": both_o.height,
            "op_eq": round(float((both_o["d_op"] == both_o["a_op"]).mean()), 4) if both_o.height else None,
            "callsign_eq": round(float((sub["d_cs"] == sub["a_cs"]).mean()), 4),
            "flightno_eq": round(float((sub["d_flt"] == sub["a_flt"]).mean()), 4),
        })
print(pl.DataFrame(rows))

print("\nE2. 'tied to a specific inbound flight' = stand match AND same aircraft type AND same operator "
      "AND gap in (0, 12h].  Share of ALL departures.")
rows = []
for tag, j in (("TRAIN", TJ), ("RANK", RJ)):
    for grp, sub in (("all", j), ("matched", j.filter(~pl.col("unmatched_nm"))),
                     ("unmatched", j.filter(pl.col("unmatched_nm")))):
        tie = (
            sub["arr_id"].is_not_null()
            & (sub["gap_s"] > 0) & (sub["gap_s"] <= 12 * 3600)
            & (sub["d_actype"] == sub["a_actype"])
            & (sub["d_op"] == sub["a_op"])
        )
        tie_t = (
            sub["arr_id"].is_not_null() & (sub["gap_s"] > 0) & (sub["gap_s"] <= 12 * 3600)
            & (sub["d_actype"] == sub["a_actype"])
        )
        rows.append({"set": tag, "group": grp, "n": sub.height,
                     "stand+type+op<=12h": round(float(tie.fill_null(False).mean()), 4),
                     "stand+type<=12h": round(float(tie_t.fill_null(False).mean()), 4)})
print(pl.DataFrame(rows))

print("\nE3. is the tie unique? for stand-matched deps with gap<=12h, how many OTHER arrivals at the "
      "same airport+stand fall in the same window? (TRAIN sample: count via second-most-recent)")
# second-most-recent same-stand arrival: shift within (apt, stand) groups of the arrival table
a2 = (
    TR_ARR.select(pl.col("apt"), pl.col("STAND_mvt").alias("stand"),
                  pl.col("BLOCK_TIME_UTC_mvt").alias("arr_block"))
    .drop_nulls(["arr_block"]).sort("arr_block")
    .with_columns(pl.col("arr_block").shift(1).over(["apt", "stand"]).alias("prev_arr_block"))
)
tj2 = TJ.join(a2.unique(subset=["apt", "stand", "arr_block"]), left_on=["apt", "stand", "arr_block"],
              right_on=["apt", "stand", "arr_block"], how="left")
sub = tj2.filter(pl.col("arr_id").is_not_null() & (pl.col("gap_s") <= 12 * 3600))
gap2 = (sub["dep_sched"] - sub["prev_arr_block"]).dt.total_seconds()
print("stand-matched gap<=12h:", sub.height,
      "| a SECOND arrival on the same stand also within 12h:", int((gap2 <= 12 * 3600).sum()),
      f"({(gap2 <= 12*3600).mean():.4f})",
      "| within 4h:", int((gap2 <= 4 * 3600).sum()))

# ----------------------------------------------------------------------------- F
hdr("F. SIGNAL CHECK (correlations only, no training)")

# arrivals in the preceding 30 / 60 minutes at the airport, relative to dep_sched
def add_arr_counts(j: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    out = []
    for apt in sorted(j["apt"].unique().to_list()):
        ja = j.filter(pl.col("apt") == apt)
        aa = arr.filter(pl.col("apt") == apt)
        land = np.sort(aa["MVT_TIME_UTC_mvt"].drop_nulls().to_numpy().astype("datetime64[s]").astype("int64"))
        blk = np.sort(aa["BLOCK_TIME_UTC_mvt"].drop_nulls().to_numpy().astype("datetime64[s]").astype("int64"))
        t = ja["dep_sched"].to_numpy().astype("datetime64[s]").astype("int64")
        n30 = np.searchsorted(land, t, "left") - np.searchsorted(land, t - 1800, "left")
        n60 = np.searchsorted(land, t, "left") - np.searchsorted(land, t - 3600, "left")
        b30 = np.searchsorted(blk, t, "left") - np.searchsorted(blk, t - 1800, "left")
        b60 = np.searchsorted(blk, t, "left") - np.searchsorted(blk, t - 3600, "left")
        out.append(ja.with_columns(
            pl.Series("arr_land_30", n30), pl.Series("arr_land_60", n60),
            pl.Series("arr_blk_30", b30), pl.Series("arr_blk_60", b60)))
    return pl.concat(out)


TJ = add_arr_counts(TJ, TR_ARR)
RJ = add_arr_counts(RJ, RK_ARR)

TJ = TJ.with_columns(
    (pl.col("dep_sched") - pl.col("arr_block")).dt.total_seconds().alias("turn_gap"),
)
RJ = RJ.with_columns(
    (pl.col("dep_sched") - pl.col("arr_block")).dt.total_seconds().alias("turn_gap"),
)

FEATS = ["taxi_in", "turn_gap", "arr_land_30", "arr_land_60", "arr_blk_30", "arr_blk_60"]

print("F1. corr(y, feature) on TRAIN, whole set and by NM-match status. "
      "'stand-matched' restricts to rows with a previous same-stand arrival within 12h.")
rows = []
for grp, sub in (("all", TJ), ("matched", TJ.filter(~pl.col("unmatched_nm"))),
                 ("unmatched", TJ.filter(pl.col("unmatched_nm")))):
    for scope, s2 in (("stand-matched<=12h", sub.filter(pl.col("arr_id").is_not_null() & (pl.col("gap_s") <= 12 * 3600))),
                      ("no-stand-match", sub.filter(pl.col("arr_id").is_null() | (pl.col("gap_s") > 12 * 3600)))):
        r = {"group": grp, "scope": scope, "n": s2.height,
             "y_p50": float(s2["y"].median()) if s2.height else None,
             "y_sd": round(float(s2["y"].std()), 1) if s2.height > 2 else None}
        for f in FEATS:
            n, c = corr(s2["y"].cast(pl.Float64), s2[f].cast(pl.Float64))
            r[f] = c
        rows.append(r)
print(pl.DataFrame(rows))

print("\nF2. same, per airport, NM-matched rows with a same-stand arrival <=12h")
rows = []
base = TJ.filter(~pl.col("unmatched_nm") & pl.col("arr_id").is_not_null() & (pl.col("gap_s") <= 12 * 3600))
for apt in sorted(base["apt"].unique().to_list()):
    s2 = base.filter(pl.col("apt") == apt)
    r = {"apt": apt, "n": s2.height, "y_p50": float(s2["y"].median()), "y_sd": round(float(s2["y"].std()), 1)}
    for f in FEATS:
        r[f] = corr(s2["y"].cast(pl.Float64), s2[f].cast(pl.Float64))[1]
    rows.append(r)
print(pl.DataFrame(rows))

print("\nF3. same, per airport, NM-UNMATCHED rows (all of them; scope column says how many have a stand match)")
rows = []
base = TJ.filter(pl.col("unmatched_nm"))
for apt in sorted(base["apt"].unique().to_list()):
    s2 = base.filter(pl.col("apt") == apt)
    r = {"apt": apt, "n": s2.height,
         "n_stand<=12h": int((s2["gap_s"].fill_null(10**9) <= 12 * 3600).sum()),
         "y_p50": float(s2["y"].median()), "y_sd": round(float(s2["y"].std()), 1)}
    for f in FEATS:
        r[f] = corr(s2["y"].cast(pl.Float64), s2[f].cast(pl.Float64))[1]
    rows.append(r)
print(pl.DataFrame(rows))

print("\nF4. LIRF unmatched only (the tail): corr with y, split by sched-fill state")
li = TJ.filter((pl.col("apt") == "LIRF") & pl.col("unmatched_nm")).with_columns(
    ((pl.col("dep_block") - pl.col("dep_sched")).dt.total_seconds().abs() <= 10).alias("fill"))
for fl in (True, False):
    s2 = li.filter(pl.col("fill") == fl)
    r = {"fill": fl, "n": s2.height, "n_stand<=12h": int((s2["gap_s"].fill_null(10**9) <= 12 * 3600).sum()),
         "y_p50": float(s2["y"].median())}
    for f in FEATS:
        r[f] = corr(s2["y"].cast(pl.Float64), s2[f].cast(pl.Float64))[1]
    print(r)

print("\nF5. spearman-style robustness: corr on log1p(y) clipped to [0, 7200] (matched, stand-matched<=12h)")
s2 = TJ.filter(~pl.col("unmatched_nm") & pl.col("arr_id").is_not_null() & (pl.col("gap_s") <= 12 * 3600)
               & (pl.col("y") >= 0) & (pl.col("y") <= 7200))
for f in FEATS:
    print(" ", f, corr(s2["y"].cast(pl.Float64).log1p(), s2[f].cast(pl.Float64)))

print("\nF6. binned means: y by taxi_in decile and by arr_land_60 decile (matched, stand-matched<=12h)")
s2 = TJ.filter(~pl.col("unmatched_nm") & pl.col("arr_id").is_not_null() & (pl.col("gap_s") <= 12 * 3600))
for f in ("taxi_in", "arr_land_60", "turn_gap"):
    b = s2.with_columns((pl.col(f).rank("ordinal") * 10 // (s2.height + 1)).alias("dec"))
    print(f"-- {f}")
    print(b.group_by("dec").agg(pl.len().alias("n"), pl.col(f).median().alias(f"{f}_p50"),
                                pl.col("y").mean().round(1).alias("y_mean"),
                                pl.col("y").median().alias("y_p50")).sort("dec"))

print("\nF7. distribution of arr_land_30 / arr_land_60 in TRAIN vs RANK (2026 window onto congestion)")
for tag, j in (("TRAIN", TJ), ("RANK", RJ)):
    print(tag, {f: q(j[f].cast(pl.Float64), (0.1, 0.5, 0.9)) for f in
                ("arr_land_30", "arr_land_60", "arr_blk_30", "arr_blk_60")})

# ----------------------------------------------------------------------------- G
hdr("G. PATHOLOGIES ON ARRIVAL ROWS")
print("G1. sched-fill analogue: on ARR rows the label-like quantity is taxi_in = BLOCK - MVT; "
      "the fill pathology on DEP is BLOCK == SCHED.  On ARR, SCHED is the scheduled IN-BLOCK time, "
      "so the analogue is BLOCK == SCHED (in-block filled with STA).")
for tag, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    g = arr.with_columns(
        (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("b_minus_s"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("m_minus_s"),
    ).group_by("apt").agg(
        pl.len().alias("n"),
        (pl.col("b_minus_s").abs() <= 10).mean().round(5).alias("blk_eq_sched_10s"),
        (pl.col("b_minus_s").abs() <= 120).mean().round(5).alias("blk_eq_sched_120s"),
        (pl.col("m_minus_s").abs() <= 10).mean().round(5).alias("land_eq_sched_10s"),
        pl.col("b_minus_s").median().alias("b_minus_s_p50"),
    ).sort("apt")
    print(f"-- {tag}"); print(g)

print("\nG1b. LIRF arrivals specifically, by month (TRAIN): fill rate and taxi-in")
print(
    TR_ARR.filter(pl.col("apt") == "LIRF").with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
        ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10).alias("f10"),
    ).group_by("ym").agg(pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10"),
                         pl.col("TAXITIME_SEC_mvt").median().alias("taxi_p50"),
                         (pl.col("TAXITIME_SEC_mvt") > 7200).sum().alias("gt2h")).sort("ym")
)
print("\nG1c. LIRF arrivals, RANK (2026)")
print(
    RK_ARR.filter(pl.col("apt") == "LIRF").with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
        ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10).alias("f10"),
    ).group_by("ym").agg(pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10"),
                         pl.col("TAXITIME_SEC_mvt").median().alias("taxi_p50"),
                         (pl.col("TAXITIME_SEC_mvt") > 7200).sum().alias("gt2h")).sort("ym")
)

print("\nG2. negative and extreme taxi-in")
for tag, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    t = arr["TAXITIME_SEC_mvt"]
    print(tag, "n", arr.height, "| <0", int((t < 0).sum()), "| ==0", int((t == 0).sum()),
          "| >7200", int((t > 7200).sum()), "| >43200", int((t > 43200).sum()),
          "| >=86400-ish (80000..90000)", int(((t >= 80000) & (t <= 90000)).sum()),
          "| min", int(t.min()), "| max", int(t.max()))
    bad = arr.filter(pl.col("TAXITIME_SEC_mvt") < 0)
    if bad.height:
        print("   negatives per airport:", bad.group_by("apt").len().sort("len", descending=True).to_dicts())

print("\nG3. day-rollover analogue on ARR: taxi_in in [80000, 90000] rows")
for tag, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    ro = arr.filter((pl.col("TAXITIME_SEC_mvt") >= 80000) & (pl.col("TAXITIME_SEC_mvt") <= 90000))
    print(tag, "n", ro.height)
    if ro.height:
        print(ro.group_by("apt").agg(pl.len().alias("n"),
                                     (pl.col("TAXITIME_SEC_mvt") - 86400).median().alias("resid_p50")).sort("n", descending=True))

print("\nG4. taxi-in > 2h rows per airport (the arrival analogue of the departure tail)")
for tag, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    print(f"-- {tag}")
    print(arr.filter(pl.col("TAXITIME_SEC_mvt") > 7200).group_by("apt").agg(
        pl.len().alias("n"), pl.col("TAXITIME_SEC_mvt").median().alias("p50"),
        pl.col("TAXITIME_SEC_mvt").max().alias("max")).sort("n", descending=True))

print("\nG5. seconds-of-minute alignment on ARR (recording granularity), per airport TRAIN")
print(
    TR_ARR.with_columns(
        pl.min_horizontal(pl.col("BLOCK_TIME_UTC_mvt").dt.second(),
                          60 - pl.col("BLOCK_TIME_UTC_mvt").dt.second()).alias("bsec"),
        pl.min_horizontal(pl.col("MVT_TIME_UTC_mvt").dt.second(),
                          60 - pl.col("MVT_TIME_UTC_mvt").dt.second()).alias("msec"),
    ).group_by("apt").agg(pl.len().alias("n"),
                          (pl.col("bsec") <= 6).mean().round(4).alias("block_within6s"),
                          (pl.col("msec") <= 6).mean().round(4).alias("land_within6s")).sort("apt")
)

print("\nG6. ARR rows with null AIRCRAFT_TYPE_mvt / null FLIGHT_ID_mvt per airport (the LIRF 'second source')")
for tag, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    print(f"-- {tag}")
    print(arr.group_by("apt").agg(
        pl.len().alias("n"),
        pl.col("AIRCRAFT_TYPE_mvt").is_null().mean().round(5).alias("actype_null"),
        pl.col("FLIGHT_ID_mvt").is_null().mean().round(5).alias("fid_null"),
        pl.col("ARVT_3_flt").is_null().mean().round(5).alias("arvt3_null"),
    ).sort("apt"))

print("\nG7. among ARR rows with null FLIGHT_ID (the unmatched arrivals): taxi-in profile")
for tag, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    u = arr.filter(pl.col("FLIGHT_ID_mvt").is_null())
    print(tag, "n", u.height,
          "taxi p50", float(u["TAXITIME_SEC_mvt"].median()) if u.height else None,
          "| >2h", int((u["TAXITIME_SEC_mvt"] > 7200).sum()) if u.height else 0,
          "| fill10", round(float(((u["BLOCK_TIME_UTC_mvt"] - u["SCHED_TIME_UTC_mvt"]).dt.total_seconds().abs() <= 10).mean()), 4) if u.height else None)
    if u.height:
        print("   per airport:", u.group_by("apt").len().sort("len", descending=True).to_dicts())

print("\nDONE")

# ----------------------------------------------------------------------------- H
hdr("H. FOLLOW-UPS")

print("H1. what IS the shared FLIGHT_ID between an ARR row and a DEP row?")
for tag, dep, arr in (("TRAIN", TR_DEP, TR_ARR), ("RANK", RK_DEP, RK_ARR)):
    d = dep.select(pl.col("FLIGHT_ID_mvt").alias("fid"), pl.col("apt").alias("d_apt"),
                   pl.col("ADES_mvt").alias("d_des"),
                   pl.col("MVT_TIME_UTC_mvt").alias("d_off")).drop_nulls(["fid"])
    a = arr.select(pl.col("FLIGHT_ID_mvt").alias("fid"), pl.col("apt").alias("a_apt"),
                   pl.col("ADEP_mvt").alias("a_org"),
                   pl.col("MVT_TIME_UTC_mvt").alias("a_land")).drop_nulls(["fid"])
    j = d.join(a, on="fid", how="inner")
    same_leg = (j["a_apt"] == j["d_des"]).fill_null(False)
    turn = (j["a_apt"] == j["d_apt"]).fill_null(False)
    print(tag, "dep-fid x arr-fid pairs", j.height,
          "| arrival airport == departure's ADES (SAME LEG)", int(same_leg.sum()),
          "| arrival airport == departure airport (TURNAROUND)", int(turn.sum()),
          "| arrival lands after departure off-block", int(((j["a_land"] > j["d_off"]).fill_null(False)).sum()))

print("\nH2. LIRF arrivals: does the DEP sched-fill pathology appear on ARR rows?")
for tag, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    li = arr.filter(pl.col("apt") == "LIRF").with_columns(
        ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10).alias("f10"))
    print(f"-- {tag}")
    print(li.group_by("f10").agg(
        pl.len().alias("n"),
        pl.col("TAXITIME_SEC_mvt").quantile(0.05).alias("p05"),
        pl.col("TAXITIME_SEC_mvt").median().alias("p50"),
        pl.col("TAXITIME_SEC_mvt").quantile(0.95).alias("p95"),
        pl.col("TAXITIME_SEC_mvt").max().alias("max"),
        (pl.col("TAXITIME_SEC_mvt") > 7200).sum().alias("gt2h"),
        (pl.col("TAXITIME_SEC_mvt") < 0).sum().alias("neg"),
    ).sort("f10"))
print("   (comparison at a non-LIRF airport, EGLL TRAIN:)")
eg = TR_ARR.filter(pl.col("apt") == "EGLL").with_columns(
    ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10).alias("f10"))
print(eg.group_by("f10").agg(pl.len().alias("n"), pl.col("TAXITIME_SEC_mvt").median().alias("p50"),
                             (pl.col("TAXITIME_SEC_mvt") > 7200).sum().alias("gt2h")).sort("f10"))

print("\nH2b. the LIRF 'second source': actype-null counts, DEP vs ARR, per airport")
for tag, dep, arr in (("TRAIN", TR_DEP, TR_ARR), ("RANK", RK_DEP, RK_ARR)):
    dn = dep.filter(pl.col("AIRCRAFT_TYPE_mvt").is_null()).group_by("apt").len().rename({"len": "dep_actype_null"})
    an = arr.filter(pl.col("AIRCRAFT_TYPE_mvt").is_null()).group_by("apt").len().rename({"len": "arr_actype_null"})
    print(f"-- {tag}"); print(dn.join(an, on="apt", how="full", coalesce=True).sort("dep_actype_null", descending=True))

print("\nH3. taxi-in signal at tighter turnaround gaps (TRAIN, NM-matched)")
rows = []
for lo, hi, lbl in ((0, 1800, "0-30m"), (1800, 3600, "30-60m"), (3600, 7200, "1-2h"),
                    (7200, 14400, "2-4h"), (14400, 43200, "4-12h"), (43200, 10**12, ">12h")):
    s2 = TJ.filter(~pl.col("unmatched_nm") & (pl.col("gap_s") >= lo) & (pl.col("gap_s") < hi))
    rows.append({"gap": lbl, "n": s2.height,
                 "corr_taxi_in": corr(s2["y"].cast(pl.Float64), s2["taxi_in"].cast(pl.Float64))[1],
                 "y_p50": float(s2["y"].median()) if s2.height else None})
print(pl.DataFrame(rows))

print("\nH4. residualised signal: y and taxi_in each centred on their (airport, month, hour-of-day) median, "
      "TRAIN NM-matched with a same-stand arrival <=12h")
base = TJ.filter(~pl.col("unmatched_nm") & pl.col("arr_id").is_not_null() & (pl.col("gap_s") <= 12 * 3600)).with_columns(
    pl.col("dep_sched").dt.month().alias("mo"), pl.col("dep_sched").dt.hour().alias("hr"))
base = base.with_columns(
    (pl.col("y") - pl.col("y").median().over(["apt", "mo", "hr"])).alias("y_res"),
    (pl.col("taxi_in") - pl.col("taxi_in").median().over(["apt", "mo", "hr"])).alias("ti_res"),
    (pl.col("arr_land_60") - pl.col("arr_land_60").median().over(["apt", "mo", "hr"])).alias("a60_res"),
    (pl.col("turn_gap") - pl.col("turn_gap").median().over(["apt", "mo", "hr"])).alias("tg_res"),
)
print("  overall:", "taxi_in", corr(base["y_res"].cast(pl.Float64), base["ti_res"].cast(pl.Float64)),
      "| arr_land_60", corr(base["y_res"].cast(pl.Float64), base["a60_res"].cast(pl.Float64)),
      "| turn_gap", corr(base["y_res"].cast(pl.Float64), base["tg_res"].cast(pl.Float64)))
rows = []
for apt in sorted(base["apt"].unique().to_list()):
    s2 = base.filter(pl.col("apt") == apt)
    rows.append({"apt": apt, "n": s2.height,
                 "r_taxi_in": corr(s2["y_res"].cast(pl.Float64), s2["ti_res"].cast(pl.Float64))[1],
                 "r_arr60": corr(s2["y_res"].cast(pl.Float64), s2["a60_res"].cast(pl.Float64))[1],
                 "r_turngap": corr(s2["y_res"].cast(pl.Float64), s2["tg_res"].cast(pl.Float64))[1]})
print(pl.DataFrame(rows))

print("\nH5. anchor sensitivity: coverage with the departure's ACTUAL off-block (TRAIN only; "
      "BLOCK_TIME is null on every ranking departure, so this anchor does not exist in the ranking set)")
d2 = TR_DEP.select(pl.col("apt"), pl.col("STAND_mvt").alias("stand"),
                   pl.col("BLOCK_TIME_UTC_mvt").alias("dep_block"),
                   pl.col("AOBT_3_flt")).drop_nulls(["dep_block"]).sort("dep_block")
a2b = TR_ARR.select(pl.col("apt"), pl.col("STAND_mvt").alias("stand"),
                    pl.col("BLOCK_TIME_UTC_mvt").alias("arr_block")).drop_nulls(["arr_block"]).sort("arr_block")
j2 = d2.join_asof(a2b, left_on="dep_block", right_on="arr_block", by=["apt", "stand"], strategy="backward")
g2 = (j2["dep_block"] - j2["arr_block"]).dt.total_seconds()
print("  BLOCK anchor: any", round(float(j2["arr_block"].is_not_null().mean()), 4),
      "| <=6h", round(float((g2.fill_null(10**9) <= 6 * 3600).mean()), 4),
      "| <=12h", round(float((g2.fill_null(10**9) <= 12 * 3600).mean()), 4),
      "| gap p50", float(g2.median()))
print("  (SCHED anchor, same set, from D1: any 0.9991, <=6h 0.8170, <=12h 0.9398, gap p50 5215)")

print("\nH6. ARR rows whose airport is NOT one of the 10 (sanity) and DEP/ARR airport coverage")
for tag, arr in (("TRAIN", TR_ARR), ("RANK", RK_ARR)):
    print(tag, "distinct arrival airports:", sorted(arr["apt"].unique().to_list()))

print("\nH7. turn_gap vs the LIRF unmatched tail: is the previous arrival informative about y>=2h there?")
li = TJ.filter((pl.col("apt") == "LIRF") & pl.col("unmatched_nm"))
print(li.with_columns((pl.col("y") >= 7200).alias("tail")).group_by("tail").agg(
    pl.len().alias("n"),
    (pl.col("gap_s").fill_null(10**9) <= 12 * 3600).mean().round(4).alias("stand_le12h"),
    pl.col("gap_s").median().alias("gap_p50"),
    pl.col("taxi_in").median().alias("taxi_in_p50"),
    pl.col("arr_land_60").median().alias("arr60_p50"),
).sort("tail"))

print("\nH8. DONE-2")

# ----------------------------------------------------------------------------- I
hdr("I. TWO LAST CHECKS")
print("I1. is the *_flt block all-or-nothing? (AOBT_3 null <=> CALLSIGN null <=> OPERATOR null)")
for tag, df in (("tr_dep", TR_DEP), ("tr_arr", TR_ARR), ("rk_dep", RK_DEP), ("rk_arr", RK_ARR)):
    n1 = df["AOBT_3_flt"].is_null()
    print(tag, "AOBT_3 null", int(n1.sum()),
          "| disagreements with CALLSIGN null", int((n1 != df["CALLSIGN_flt"].is_null()).sum()),
          "| with OPERATOR null", int((n1 != df["AIRCRAFT_OPERATOR_flt"].is_null()).sum()),
          "| with MARKET_SEGMENT null", int((n1 != df["MARKET_SEGMENT_flt"].is_null()).sum()))

print("\nI2. BLOCK - SCHED on ARR rows: mass at exactly 0 vs the neighbouring minutes, per airport (TRAIN)")
d = TR_ARR.select(
    pl.col("apt"),
    ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds() / 60)
    .round(0).cast(pl.Int64).alias("dmin"),
)
print(
    d.group_by("apt").agg(
        pl.len().alias("n"),
        (pl.col("dmin") == 0).mean().round(5).alias("at_0"),
        (pl.col("dmin").is_in([-2, -1])).mean().round(5).alias("at_m1m2"),
        (pl.col("dmin").is_in([1, 2])).mean().round(5).alias("at_p1p2"),
    ).with_columns(
        (pl.col("at_0") / ((pl.col("at_m1m2") + pl.col("at_p1p2")) / 4)).round(2).alias("spike_ratio")
    ).sort("spike_ratio", descending=True)
)

print("\nI3. implausibly short turnarounds: stand-join gap < 300 s, per airport (TRAIN / RANK)")
for tag, j in (("TRAIN", TJ), ("RANK", RJ)):
    s = j.filter(pl.col("gap_s") < 300)
    print(tag, "n", s.height, s.group_by("apt").len().sort("len", descending=True).to_dicts())

print("\nI4. DONE-3")

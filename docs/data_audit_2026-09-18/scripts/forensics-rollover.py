#!/usr/bin/env python
"""Row-level forensics on the 16 training rows with TAXITIME_SEC_mvt > 72,000 s
("day-rollover" rows), 2026-09-18.

No training. POLARS_MAX_THREADS=1. Lazy scans where possible.

Questions answered:
  S1  what is in every one of the 16 rows (feat_train non-null + raw + OOF prediction)
  S2  what actually happened to each flight (arrival leg, stand timeline, neighbouring traffic)
  S3  which timestamp carries the wrong date (single-shift tests + stand-occupancy test)
  S4  identifiability at prediction time (rules using only ranking-populated columns)
  S5  how many candidates in the 2026 ranking set; training base rate per month
  S6  leaderboard arithmetic (honest ceiling)
  S7  false-positive cost

Inputs (read-only):
  data/work/feat_train.parquet, data/work/feat_rank.parquet
  data/raw/training_*.parquet, data/raw/ranking.parquet
  experiments/oof_main-48c19bc830.parquet, outputs/submit_v10_reparam.parquet

Run: POLARS_MAX_THREADS=1 .venv/bin/python docs/data_audit_2026-09-18/scripts/forensics-rollover.py
"""

import datetime as dt
import glob
import json
import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import polars as pl

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(300)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN_FILES = sorted(glob.glob(os.path.join(ROOT, "data/raw/training_*.parquet")))
RAW_COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
    "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "CALLSIGN_flt", "AIRCRAFT_OPERATOR_flt",
    "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
]
SHOW = ["FLIGHT_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "STAND_mvt",
        "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "CALLSIGN_flt", "AOBT_3_flt", "ARVT_3_flt"]
DAY = dt.timedelta(days=1)
H = dt.timedelta(hours=1)
DATASET_APTS = {"EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"}


def head(t):
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


# ---------------------------------------------------------------- S1
head("S1  The 16 training rows with y > 72,000 s")

feat = pl.scan_parquet(os.path.join(ROOT, "data/work/feat_train.parquet"))
r16 = feat.filter(pl.col("TAXITIME_SEC_mvt") > 72_000).collect()
oof = pl.read_parquet(os.path.join(ROOT, "experiments/oof_main-48c19bc830.parquet"))
r16 = r16.join(oof, on="MVT_ID_mvt", how="left").sort("TAXITIME_SEC_mvt", descending=True)
print(f"rows: {r16.height}")

raw_all = pl.concat([pl.scan_parquet(f).select(RAW_COLS) for f in TRAIN_FILES]).collect()
print(f"raw training rows loaded: {raw_all.height}")
raw_dep = raw_all.filter(pl.col("PHASE_mvt") == "DEP")
raw_arr = raw_all.filter(pl.col("PHASE_mvt") == "ARR")

ids = r16["MVT_ID_mvt"].to_list()
raw16 = raw_all.filter(pl.col("MVT_ID_mvt").is_in(ids))

for i, r in enumerate(r16.iter_rows(named=True), 1):
    print("\n" + "-" * 110)
    print(f"ROW #{i}  {r['ADEP_mvt']}->{r['ADES_mvt']}  {r['FLIGHT_mvt']}  y={r['TAXITIME_SEC_mvt']}  "
          f"MVT_ID={r['MVT_ID_mvt']:.0f}  OOF={r['oof']}")
    nn = {k: str(v) for k, v in r.items() if v is not None}
    print("feat_train non-null columns (%d of 132):" % len(nn))
    print(json.dumps(nn, indent=0))
    print("raw row:")
    print(raw16.filter(pl.col("MVT_ID_mvt") == r["MVT_ID_mvt"]))

# ---------------------------------------------------------------- S2
head("S2  Per-row forensics: same flight, arrival leg, stand timeline, neighbours +-3 h")

for i, r in enumerate(r16.iter_rows(named=True), 1):
    f, apt, des, st = r["FLIGHT_mvt"], r["ADEP_mvt"], r["ADES_mvt"], r["STAND_mvt"]
    B, M, S = r["BLOCK_TIME_UTC_mvt"], r["MVT_TIME_UTC_mvt"], r["SCHED_TIME_UTC_mvt"]
    print("\n" + "#" * 110)
    print(f"#{i} {apt}->{des} {f}  y={r['TAXITIME_SEC_mvt']}  SCHED={S}  BLOCK={B}  MVT={M}  stand={st} rwy={r['RUNWAY_mvt']}")

    if r["FLIGHT_ID_mvt"] is not None:
        fid = raw_all.filter(pl.col("FLIGHT_ID_mvt") == r["FLIGHT_ID_mvt"])
        print(f"-- A0) rows sharing FLIGHT_ID_mvt={r['FLIGHT_ID_mvt']:.0f}: {fid.height}")
        if fid.height:
            print(fid.select(SHOW))
    else:
        print("-- A0) FLIGHT_ID_mvt is null: no FLIGHT_ID join possible")

    a = raw_all.filter((pl.col("FLIGHT_mvt") == f)
                       & (pl.col("MVT_TIME_UTC_mvt") >= B - 4 * DAY)
                       & (pl.col("MVT_TIME_UTC_mvt") <= M + 4 * DAY)).sort("MVT_TIME_UTC_mvt") if f else raw_all.head(0)
    print(f"-- A) rows with FLIGHT_mvt=={f} within +-4 d: {a.height}")
    if a.height:
        print(a.select(SHOW))

    if des in DATASET_APTS:
        b = raw_arr.filter((pl.col("ADES_mvt") == des) & (pl.col("ADEP_mvt") == apt)
                           & (pl.col("MVT_TIME_UTC_mvt") >= B - 6 * H)
                           & (pl.col("MVT_TIME_UTC_mvt") <= M + 12 * H)).sort("MVT_TIME_UTC_mvt")
        print(f"-- B) ARR rows at {des} from {apt} in [BLOCK-6h, MVT+12h]: {b.height}")
        if b.height:
            print(b.select(SHOW))
    else:
        print(f"-- B) destination {des} is not one of the 10 observed airports: no arrival row exists")

    c = raw_all.filter((pl.col("STAND_mvt") == st)
                       & (((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == apt))
                          | ((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == apt)))
                       & (pl.col("MVT_TIME_UTC_mvt") >= B - 12 * H)
                       & (pl.col("MVT_TIME_UTC_mvt") <= M + 12 * H)).sort("MVT_TIME_UTC_mvt")
    print(f"-- C) ALL movements at {apt} stand {st} in [BLOCK-12h, MVT+12h]: {c.height}")
    if c.height:
        print(c.select(SHOW))

    for label, anchor in (("BLOCK", B), ("MVT", M)):
        d = raw_dep.filter((pl.col("ADEP_mvt") == apt) & (pl.col("STAND_mvt") == st)
                           & ((pl.col("MVT_TIME_UTC_mvt") - anchor).abs() <= 3 * H))
        e = raw_dep.filter((pl.col("ADEP_mvt") == apt)
                           & ((pl.col("MVT_TIME_UTC_mvt") - anchor).abs() <= 3 * H))
        print(f"-- D) departures within +-3 h of {label}: same stand {d.height}, same airport {e.height}")

# ---------------------------------------------------------------- S3
head("S3  Which timestamp carries the wrong date")

# LIRF/LSZH/LFPG sched-fill prevalence (context for BLOCK==SCHED rows)
fill = (raw_dep.filter(pl.col("ADEP_mvt").is_in(list(DATASET_APTS)))
        .with_columns(((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
                       .dt.total_seconds().abs() <= 10).alias("f10"))
        .group_by("ADEP_mvt").agg([pl.len().alias("n"), pl.col("f10").mean().alias("fill_rate")])
        .sort("fill_rate", descending=True))
print("share of DEP rows with |BLOCK-SCHED| <= 10 s, per airport:")
print(fill)

verdicts = []
for i, r in enumerate(r16.iter_rows(named=True), 1):
    apt, st = r["ADEP_mvt"], r["STAND_mvt"]
    B, M, S = r["BLOCK_TIME_UTC_mvt"], r["MVT_TIME_UTC_mvt"], r["SCHED_TIME_UTC_mvt"]
    stand_rows = raw_all.filter((pl.col("STAND_mvt") == st)
                                & (((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == apt))
                                   | ((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == apt)))
                                & (pl.col("MVT_ID_mvt") != r["MVT_ID_mvt"]))
    # last ARR in-block at the stand strictly before MVT, and next movement after MVT
    pa = stand_rows.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("BLOCK_TIME_UTC_mvt") < M)) \
                   .sort("BLOCK_TIME_UTC_mvt").tail(1)
    prev_arr = pa["BLOCK_TIME_UTC_mvt"][0] if pa.height else None
    prev_arr_flight = pa["FLIGHT_mvt"][0] if pa.height else None
    prev_arr_sched = pa["SCHED_TIME_UTC_mvt"][0] if pa.height else None
    nx = stand_rows.filter(pl.col("MVT_TIME_UTC_mvt") > M).sort("MVT_TIME_UTC_mvt").head(1)
    nxt = nx["MVT_TIME_UTC_mvt"][0] if nx.height else None
    # is there any *other* departure from this stand between prev_arr and M?  (an unexplained gap)
    between = stand_rows.filter((pl.col("PHASE_mvt") == "DEP")
                                & (pl.col("MVT_TIME_UTC_mvt") > (prev_arr if prev_arr else M))
                                & (pl.col("MVT_TIME_UTC_mvt") < M)).height if prev_arr else None
    d = dict(
        row=i, apt=apt, flight=r["FLIGHT_mvt"], stand=st,
        y=r["TAXITIME_SEC_mvt"], ms=int(r["mvt_minus_sched"]),
        y_minus_1d=r["TAXITIME_SEC_mvt"] - 86_400,
        block_minus_sched=int((B - S).total_seconds()),
        # hypothesis H_B: BLOCK date is one day early -> true off-block = BLOCK + 1 d
        HB_taxi=int((M - (B + DAY)).total_seconds()),
        HB_offblock_minus_sched=int(((B + DAY) - S).total_seconds()),
        HB_after_prev_arr=None if prev_arr is None else bool((B + DAY) > prev_arr),
        # hypothesis H_M: MVT date is one day late -> true take-off = MVT - 1 d
        HM_taxi=int(((M - DAY) - B).total_seconds()),
        HM_takeoff_minus_sched=int(((M - DAY) - S).total_seconds()),
        prev_same_stand_arr=str(prev_arr), prev_arr_flight=prev_arr_flight,
        prev_arr_sched=str(prev_arr_sched),
        deps_between_prev_arr_and_MVT=between,
        next_stand_movement=str(nxt),
    )
    verdicts.append(d)
vt = pl.DataFrame(verdicts)
print("\nsingle-shift arithmetic and stand-occupancy test (one row per rollover row):")
for c in [["row", "apt", "flight", "stand", "y", "ms", "y_minus_1d", "block_minus_sched"],
          ["row", "flight", "HB_taxi", "HB_offblock_minus_sched", "HB_after_prev_arr",
           "HM_taxi", "HM_takeoff_minus_sched"],
          ["row", "flight", "prev_same_stand_arr", "prev_arr_flight", "prev_arr_sched",
           "deps_between_prev_arr_and_MVT", "next_stand_movement"]]:
    print(vt.select(c))

# independent flight-plan clocks
print("\nindependent clocks on the 16 rows (null = absent):")
print(r16.select(["ADEP_mvt", "FLIGHT_mvt", "nm_matched", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt",
                  "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt",
                  "AOBT_3_flt", "ARVT_3_flt"]))

# ---------------------------------------------------------------- S4
head("S4  Identifiability at prediction time")

TR_COLS = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
           "FLIGHT_mvt", "FLIGHT_RULE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
           "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "mvt_minus_sched", "nm_matched",
           "month", "hour_utc", "MARKET_SEGMENT_flt", "AIRCRAFT_OPERATOR_flt"]
tr = feat.select(TR_COLS).collect()
rk = pl.scan_parquet(os.path.join(ROOT, "data/work/feat_rank.parquet")).select(TR_COLS).collect()

y = pl.col("TAXITIME_SEC_mvt")
ms = pl.col("mvt_minus_sched")
tr = tr.with_columns([
    (y > 72_000).alias("target"),
    ((y - ms).abs() <= 60).alias("filled"),
    pl.col("AIRCRAFT_TYPE_mvt").is_null().alias("actype_null"),
])
print("training target rows:", tr["target"].sum())
print("\nranking: BLOCK/label nullness ->",
      rk.select([pl.col("TAXITIME_SEC_mvt").is_null().sum()]).item(), "of", rk.height)


def rule(name, tr_mask, rk_mask):
    t = tr.filter(tr_mask)
    tp = int(t["target"].sum())
    n = t.height
    prec = tp / n if n else float("nan")
    rec = tp / 16
    nrk = rk.filter(rk_mask).height
    print(f"{name:<62s} n_train={n:<8d} tp={tp:<3d} prec={prec:.4f} rec={rec:.4f} n_rank={nrk}")
    return dict(rule=name, n_train=n, tp=tp, precision=prec, recall=rec, n_rank=nrk)


unm_t = pl.col("nm_matched") == 0
lirf = pl.col("ADEP_mvt") == "LIRF"
res = []
print("\nrules built ONLY from columns populated on ranking rows:")
for thr in (7_200, 14_400, 28_800, 43_200, 57_600, 72_000, 80_000, 86_400):
    res.append(rule(f"ms >= {thr}", ms >= thr, ms >= thr))
for thr in (43_200, 57_600, 72_000, 86_400):
    res.append(rule(f"unmatched & ms >= {thr}", unm_t & (ms >= thr), unm_t & (ms >= thr)))
    res.append(rule(f"LIRF & unmatched & ms >= {thr}", lirf & unm_t & (ms >= thr),
                    lirf & unm_t & (ms >= thr)))
res.append(rule("LIRF & unmatched & ms >= 43200 & RUNWAY != 16R",
                lirf & unm_t & (ms >= 43_200) & (pl.col("RUNWAY_mvt") != "16R"),
                lirf & unm_t & (ms >= 43_200) & (pl.col("RUNWAY_mvt") != "16R")))
res.append(rule("LIRF & unmatched & actype null & ms >= 43200",
                lirf & unm_t & pl.col("AIRCRAFT_TYPE_mvt").is_null() & (ms >= 43_200),
                lirf & unm_t & pl.col("AIRCRAFT_TYPE_mvt").is_null() & (ms >= 43_200)))
res.append(rule("LIRF & unmatched & ms >= 43200 & stand starts 6/7/8",
                lirf & unm_t & (ms >= 43_200) & pl.col("STAND_mvt").str.slice(0, 1).is_in(["6", "7", "8"]),
                lirf & unm_t & (ms >= 43_200) & pl.col("STAND_mvt").str.slice(0, 1).is_in(["6", "7", "8"])))
res.append(rule("unmatched (any airport)", unm_t, unm_t))
res.append(rule("LIRF & unmatched", lirf & unm_t, lirf & unm_t))
print("\nbest F1 over the sweep:")
best = sorted(res, key=lambda d: 0 if (d["precision"] + d["recall"]) == 0
              else 2 * d["precision"] * d["recall"] / (d["precision"] + d["recall"]), reverse=True)[:5]
for d in best:
    f1 = 0 if (d["precision"] + d["recall"]) == 0 else 2 * d["precision"] * d["recall"] / (d["precision"] + d["recall"])
    print(f"  F1={f1:.4f}  {d['rule']}  (prec {d['precision']:.3f}, rec {d['recall']:.3f}, n_rank {d['n_rank']})")

# which of the 16 are invisible to the best structural rule
vis = lirf & unm_t & (ms >= 43_200)
print("\nthe 16 rows against 'LIRF & unmatched & ms >= 43200':")
print(tr.filter(pl.col("target")).with_columns(vis.alias("in_rule"))
      .select(["ADEP_mvt", "FLIGHT_mvt", "nm_matched", "mvt_minus_sched", "TAXITIME_SEC_mvt", "in_rule"])
      .sort("mvt_minus_sched"))

# the discriminating population: LIRF unmatched ms >= 12 h
pop = tr.filter(lirf & unm_t & (ms >= 43_200)).sort("mvt_minus_sched")
print("\nLIRF unmatched ms >= 12 h population (n=%d): rollover vs sched-filled" % pop.height)
print(pop.select(["FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt", "month", "hour_utc",
                  "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
                  "mvt_minus_sched", "filled", "target"]))

# extra ranking-visible signals on that population
lirf_mv = raw_all.filter(((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF"))
                         | ((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF")))
lirf_dep = raw_dep.filter(pl.col("ADEP_mvt") == "LIRF")
extra = []
for r in pop.iter_rows(named=True):
    M, st, f = r["MVT_TIME_UTC_mvt"], r["STAND_mvt"], r["FLIGHT_mvt"]
    twin = lirf_dep.filter(((pl.col("MVT_TIME_UTC_mvt") - M).abs() <= dt.timedelta(seconds=5))
                           & (pl.col("MVT_ID_mvt") != r["MVT_ID_mvt"])).height
    pa = lirf_mv.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("STAND_mvt") == st)
                        & (pl.col("BLOCK_TIME_UTC_mvt") < M)).sort("BLOCK_TIME_UTC_mvt").tail(1)
    gap = int((M - pa["BLOCK_TIME_UTC_mvt"][0]).total_seconds()) if pa.height else None
    same = lirf_dep.filter((pl.col("FLIGHT_mvt") == f)
                           & ((pl.col("MVT_TIME_UTC_mvt") - M).abs() <= 2 * DAY)
                           & (pl.col("MVT_ID_mvt") != r["MVT_ID_mvt"])).height if f else 0
    extra.append(dict(flight=f, stand=st, ms=int(r["mvt_minus_sched"]), y=r["TAXITIME_SEC_mvt"],
                      target=bool(r["target"]), filled=bool(r["filled"]),
                      twin_takeoff_5s=twin, prev_stand_arr_gap_s=gap, same_flight_pm2d=same))
ex = pl.DataFrame(extra)
print("\nadditional ranking-visible signals on the same population:")
print(ex.sort("ms"))
print("\nby class (target = y > 72,000):")
print(ex.group_by("target").agg([pl.len().alias("n"),
                                 pl.col("twin_takeoff_5s").mean().alias("twin_mean"),
                                 pl.col("prev_stand_arr_gap_s").median().alias("gap_med"),
                                 pl.col("prev_stand_arr_gap_s").min().alias("gap_min"),
                                 pl.col("prev_stand_arr_gap_s").max().alias("gap_max"),
                                 pl.col("same_flight_pm2d").mean().alias("samefl_mean")]))

# ---------------------------------------------------------------- S5
head("S5  How many in the 2026 ranking set")

print("ranking rows per month:")
print(rk.group_by("month").len().sort("month"))
print("\ntraining rows and rollover rows per month:")
print(tr.group_by("month").agg([pl.len().alias("n"), pl.col("target").sum().alias("rollover")]).sort("month"))
jj = tr.filter(pl.col("month").is_in([1, 7]))
print(f"\ntraining Jan+Jul 2025: n={jj.height}, rollover={int(jj['target'].sum())}, "
      f"rate={jj['target'].sum() / jj.height:.3e}")
print(f"training all 2025:     n={tr.height}, rollover={int(tr['target'].sum())}, "
      f"rate={tr['target'].sum() / tr.height:.3e}")
print(f"ranking (Jan+Jul 2026): n={rk.height}")
print(f"expected rollovers from Jan+Jul-2025 rate: {jj['target'].sum() / jj.height * rk.height:.2f}")
print(f"expected rollovers from all-2025 rate:     {tr['target'].sum() / tr.height * rk.height:.2f}")

cand_rk = rk.filter(lirf & unm_t & (ms >= 43_200))
sub = pl.read_parquet(os.path.join(ROOT, "outputs/submit_v10_reparam.parquet"))
cand_rk = cand_rk.join(sub.rename({"TAXITIME_SEC_mvt": "pred_v10"}), on="MVT_ID_mvt", how="left")
print("\nthe %d ranking rows matching 'LIRF & unmatched & ms >= 43200', with the v10 prediction:"
      % cand_rk.height)
print(cand_rk.select(["FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt", "ADES_mvt", "month",
                      "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_sched", "pred_v10"])
      .sort("mvt_minus_sched"))

# ---------------------------------------------------------------- S6 / S7
head("S6/S7  Leaderboard arithmetic: ceiling and false-positive cost")

N, NM, NU = 344_841, 339_551, 5_290
LB = 284.3584
SSE_TOT = LB ** 2 * N
SSE_M = 218.0 ** 2 * NM
SSE_U = SSE_TOT - SSE_M
print(f"N={N} (matched {NM} + unmatched {NU});  LB {LB} -> total SSE {SSE_TOT:.4e}")
print(f"matched at 218.0 -> SSE {SSE_M:.4e};  implied unmatched SSE {SSE_U:.4e} -> RMSE {(SSE_U / NU) ** .5:.1f}")


def lb(sse):
    return (sse / N) ** 0.5


# rollover label model taken from training: y in [86,700, 88,800], median 87,300
Y_ROLL = 87_300.0
cr = cand_rk.sort("mvt_minus_sched")
mss = cr["mvt_minus_sched"].to_list()
pv = [float(x) for x in cr["pred_v10"].to_list()]
print("\nper candidate row: SSE now if it IS a rollover, and SSE if we fire and it is NOT")
tot_if_roll = 0.0
rows_arith = []
for f_, m_, p_ in zip(cr["FLIGHT_mvt"].to_list(), mss, pv):
    sse_missed = (Y_ROLL - p_) ** 2          # we keep pred = v10, row is a rollover
    sse_fp = (m_ - Y_ROLL) ** 2              # we predict 87,300, row is sched-filled (y = ms)
    sse_now_fill = (m_ - p_) ** 2            # v10 today on a sched-filled row
    tot_if_roll += sse_missed
    rows_arith.append(dict(flight=f_, ms=int(m_), pred_v10=int(p_),
                           sse_if_rollover_missed=sse_missed, sse_if_fired_and_filled=sse_fp,
                           sse_v10_if_filled=sse_now_fill,
                           gain_s=LB - lb(SSE_TOT - sse_missed),
                           loss_s=lb(SSE_TOT + sse_fp - sse_now_fill) - LB))
ra = pl.DataFrame(rows_arith)
print(ra)
print(f"\nsum over the 11 candidates of (87,300 - v10 pred)^2 = {tot_if_roll:.4e} "
      f"({tot_if_roll / SSE_U:.3f} of the whole unmatched SSE budget)")

print("\nLB implied bound: if j of the 11 were rollovers, their SSE alone would be at least")
srt = sorted(r["sse_if_rollover_missed"] for r in rows_arith)
cum = 0.0
for j, s in enumerate(srt, 1):
    cum += s
    print(f"  j={j:<3d} min SSE {cum:.4e}  = {cum / SSE_U:.3f} of unmatched SSE  "
          f"(residual RMSE for the other {NU - j} unmatched rows: "
          f"{((SSE_U - cum) / (NU - j)) ** .5 if cum < SSE_U else float('nan'):.0f})")

print("\nceiling A: perfect oracle on k rollover rows currently predicted by v10 at ~ms")
for k in (1, 2, 3, 5, 7):
    s = sum(srt[-k:])
    print(f"  k={k} (the k costliest): LB {LB} -> {lb(SSE_TOT - s):.4f}  gain {LB - lb(SSE_TOT - s):+.4f} s")
    s2 = sum(srt[:k])
    print(f"  k={k} (the k cheapest ): LB {LB} -> {lb(SSE_TOT - s2):.4f}  gain {LB - lb(SSE_TOT - s2):+.4f} s")

print("\nceiling B: predict 86,400 + LIRF median taxi on every one of the 11, as a function of the")
print("           number j that are really rollovers (rest are sched-filled, y = ms)")
lirf_med = tr.filter(lirf & (y > 0) & (y < 7_200))["TAXITIME_SEC_mvt"].median()
print(f"  LIRF median taxi (0 < y < 2 h) = {lirf_med:.0f} s -> prediction {86_400 + lirf_med:.0f} s")
PRED = 86_400 + float(lirf_med)
for j in range(0, 12):
    # assume the j rollovers are the ones with the largest |ms - 87,300| (most favourable) and
    # the least favourable ordering, to bracket
    order = sorted(range(11), key=lambda t: -rows_arith[t]["sse_if_rollover_missed"])
    best_set, worst_set = order[:j], order[::-1][:j]
    for tag, sel in (("best case", best_set), ("worst case", worst_set)):
        d = 0.0
        for t in range(11):
            m_, p_ = mss[t], pv[t]
            if t in sel:       # really a rollover
                d += (PRED - Y_ROLL) ** 2 - (p_ - Y_ROLL) ** 2
            else:              # really sched-filled
                d += (PRED - m_) ** 2 - (p_ - m_) ** 2
        if tag == "best case":
            b = LB - lb(SSE_TOT + d)
        else:
            w = LB - lb(SSE_TOT + d)
    print(f"  j={j:<3d} gain best case {b:+.4f} s   worst case {w:+.4f} s")

print("\nexpected value at the training mixing rate")
for pi in (11 / 29, 12 / 29, 0.5, 0.62):
    d = 0.0
    for t in range(11):
        m_, p_ = mss[t], pv[t]
        cur = pi * (p_ - Y_ROLL) ** 2 + (1 - pi) * (p_ - m_) ** 2
        new = pi * (PRED - Y_ROLL) ** 2 + (1 - pi) * (PRED - m_) ** 2
        d += new - cur
    mix = 0.0
    for t in range(11):
        m_, p_ = mss[t], pv[t]
        q = pi * Y_ROLL + (1 - pi) * m_
        cur = pi * (p_ - Y_ROLL) ** 2 + (1 - pi) * (p_ - m_) ** 2
        new = pi * (q - Y_ROLL) ** 2 + (1 - pi) * (q - m_) ** 2
        mix += new - cur
    print(f"  pi={pi:.3f}: hard rule (predict {PRED:.0f}) {LB - lb(SSE_TOT + d):+.4f} s; "
          f"mixture mean pi*87,300+(1-pi)*ms {LB - lb(SSE_TOT + mix):+.4f} s")

print("\nfalse-positive cost of firing on a single non-rollover candidate:")
print(ra.select(["flight", "ms", "pred_v10", "loss_s", "gain_s"]))

print("\ndone.")

# ---------------------------------------------------------------- S4b
head("S4b  ms threshold sweep inside the LIRF-unmatched population, strict rollover target")

# strict target: y > 72,000 AND not sched-filled (|y - ms| > 60).  NOSOS739 (y = ms = 131,163)
# is a sched-filled row that happens to exceed 20 h and is NOT a day-rollover.
tr = tr.with_columns(((y > 72_000) & ((y - ms).abs() > 60)).alias("strict"))
print("strict rollover rows in training:", int(tr["strict"].sum()))
print(tr.filter(pl.col("strict")).select(["ADEP_mvt", "FLIGHT_mvt", "nm_matched",
      "mvt_minus_sched", "TAXITIME_SEC_mvt"]).sort("mvt_minus_sched"))

sweep = []
for thr in (43_200, 46_800, 50_400, 54_000, 57_600, 61_200, 64_800, 72_000, 86_400):
    m_tr = lirf & unm_t & (ms >= thr)
    t = tr.filter(m_tr)
    tp = int(t["strict"].sum())
    nrk = rk.filter(lirf & unm_t & (ms >= thr)).height
    sweep.append(dict(thr=thr, n_train=t.height, rollover=tp, filled=t.height - tp,
                      precision=tp / t.height if t.height else float("nan"),
                      recall=tp / 15, n_rank=nrk))
sw = pl.DataFrame(sweep)
print("\nLIRF & unmatched & ms >= thr, strict target (15 rollover rows in training):")
print(sw)

# 2x2 contingency at the best threshold
lo = tr.filter(lirf & unm_t & (ms >= 43_200) & (ms < 57_600))
hi = tr.filter(lirf & unm_t & (ms >= 57_600))
a, b_, c_, d_ = int(hi["strict"].sum()), hi.height - int(hi["strict"].sum()), \
                int(lo["strict"].sum()), lo.height - int(lo["strict"].sum())
print(f"\n2x2 inside the 29-row population, split at ms = 57,600:")
print(f"   ms >= 57,600 : rollover {a}, sched-filled {b_}")
print(f"   ms <  57,600 : rollover {c_}, sched-filled {d_}")
try:
    from math import comb
    n = a + b_ + c_ + d_
    r1, r2, cc1 = a + b_, c_ + d_, a + c_
    obs = comb(r1, a) * comb(r2, c_) / comb(n, cc1)
    p = sum(comb(r1, k) * comb(r2, cc1 - k) / comb(n, cc1)
            for k in range(max(0, cc1 - r2), min(r1, cc1) + 1)
            if comb(r1, k) * comb(r2, cc1 - k) / comb(n, cc1) <= obs + 1e-12)
    print(f"   Fisher exact two-sided p = {p:.5f}")
except Exception as e:
    print("   Fisher failed:", e)

# Wilson interval for the high-ms precision
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (c - h, c + h)
print(f"   Wilson 95% CI for {a}/{a + b_}: {wilson(a, a + b_)}")
print(f"   Wilson 95% CI for 11/29 (whole bin): {wilson(11, 29)}")

# per-flight precedent for the 4 high-ms ranking rows
print("\nper-flight precedent in training for the ranking candidates (same FLIGHT_mvt at LIRF):")
for f_ in rk.filter(lirf & unm_t & (ms >= 43_200))["FLIGHT_mvt"].to_list():
    t = tr.filter(lirf & (pl.col("FLIGHT_mvt") == f_))
    print(f"  {str(f_):<12s} training rows {t.height:<4d} strict-rollover {int(t['strict'].sum()) if t.height else 0:<3d} "
          f"sched-filled(|y-ms|<=60 & ms>=43200) "
          f"{int(t.filter((ms >= 43_200) & pl.col('filled')).height) if t.height else 0}")

# ---------------------------------------------------------------- S6b
head("S6b  Expected leaderboard effect of the two candidate rules, as a function of the true rate")

def dlb(delta_sse):
    return LB - lb(SSE_TOT + delta_sse)

for thr, pi_hat in ((43_200, 11 / 29), (57_600, a / (a + b_))):
    cr2 = cand_rk.filter(pl.col("mvt_minus_sched") >= thr).sort("mvt_minus_sched")
    ms2 = [float(x) for x in cr2["mvt_minus_sched"].to_list()]
    pv2 = [float(x) for x in cr2["pred_v10"].to_list()]
    sd2 = sum((Y_ROLL - m_) ** 2 for m_ in ms2)
    print(f"\nrule: LIRF & unmatched & ms >= {thr}   n_rank={len(ms2)}   "
          f"training rollover rate pi_hat={pi_hat:.4f}   sum (87,300-ms)^2 = {sd2:.4e}")
    print("  flights:", cr2["FLIGHT_mvt"].to_list())
    print("   pi_true |  hard rule (predict 87,300)  |  mixture mean pi_hat*87,300+(1-pi_hat)*ms")
    for pi in (0.0, 0.1, 0.2, 0.3, pi_hat / 2, 0.4, 0.5, pi_hat, 0.8, 1.0):
        dh = dm = 0.0
        for m_, p_ in zip(ms2, pv2):
            q = pi_hat * Y_ROLL + (1 - pi_hat) * m_
            cur = pi * (p_ - Y_ROLL) ** 2 + (1 - pi) * (p_ - m_) ** 2
            dh += pi * 0.0 + (1 - pi) * (Y_ROLL - m_) ** 2 - cur
            dm += pi * (q - Y_ROLL) ** 2 + (1 - pi) * (q - m_) ** 2 - cur
        print(f"   {pi:6.3f}  |  {dlb(dh):+8.3f} s                  |  {dlb(dm):+8.3f} s")
    print(f"  break-even pi: hard rule 0.500 ; mixture rule {pi_hat / 2:.3f}")

print("\ndone (S4b/S6b).")

# ---------------------------------------------------------------- S8
head("S8  Is the mechanism visible anywhere in 2026?  The labelled ranking ARR rows")

RCOLS = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt",
         "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "STAND_mvt",
         "AIRCRAFT_TYPE_mvt", "AOBT_3_flt"]
rank_raw = pl.scan_parquet(os.path.join(ROOT, "data/raw/ranking.parquet")).select(RCOLS).collect()
for tag, df in (("TRAIN 2025", raw_all.select([c for c in RCOLS])), ("RANK 2026", rank_raw)):
    for ph in ("ARR", "DEP"):
        s = df.filter(pl.col("PHASE_mvt") == ph)
        lab = s.filter(pl.col("TAXITIME_SEC_mvt").is_not_null())
        big = lab.filter(pl.col("TAXITIME_SEC_mvt") > 72_000)
        print(f"{tag} {ph}: rows={s.height} labelled={lab.height} y>72,000={big.height}")
        if 0 < big.height < 40:
            print(big.with_columns((pl.col("TAXITIME_SEC_mvt") - 86_400).alias("y_minus_1d")))
arr25 = raw_all.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("TAXITIME_SEC_mvt").is_not_null())
n25, k25 = arr25.height, arr25.filter(pl.col("TAXITIME_SEC_mvt") > 72_000).height
narr26 = rank_raw.filter(pl.col("PHASE_mvt") == "ARR").height
print(f"\n2025 ARR rollover-band rate {k25}/{n25} = {k25 / n25:.3e}; "
      f"expected count in the {narr26} labelled 2026 ARR rows = {k25 / n25 * narr26:.2f}")
print("observed in 2026 ARR:",
      rank_raw.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("TAXITIME_SEC_mvt") > 72_000)).height)
print("\ndone (S8).")

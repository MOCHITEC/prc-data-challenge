"""Row-level forensics on the top-200 OOF squared-error rows of oof_main-48c19bc830.

No model training. POLARS_MAX_THREADS=1. Read-only on data/ and experiments/.
Writes nothing but stdout (redirect to capture).

Definitions (plan.md 2026-09-16 §0):
  y     = TAXITIME_SEC_mvt = MVT_TIME - BLOCK_TIME
  proxy = mvt_minus_aobt3  = MVT_TIME - AOBT_3
  d     = y - proxy        = AOBT_3 - BLOCK_TIME        (matched rows only)
  ms    = mvt_minus_sched  = MVT_TIME - SCHED_TIME      (ranking null rate 0)
  err   = y - oof ;  unmatched = proxy is null (model-B routing rule)
"""
import glob
import polars as pl

RAW = sorted(glob.glob("data/raw/training_*.parquet"))
OOF = "experiments/oof_main-48c19bc830.parquet"
FT = "data/work/feat_train.parquet"
FR = "data/work/feat_rank.parquet"

# leaderboard arithmetic constants given by the task
N_RANK, N_MATCHED, N_UNMATCHED = 344841, 339551, 5290
RMSE_MATCHED, RMSE_UNMATCHED, LB_BEST = 218.0, 1490.2, 284.3584

def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)

# ---------------------------------------------------------------- 0. the 141
hdr("0. THE RANKING SET AND THE 141 ROWS")
oof = pl.read_parquet(OOF).filter(pl.col("oof").is_not_null())
ft = pl.read_parquet(FT)
d = ft.join(oof, on="MVT_ID_mvt", how="inner").with_columns(
    (pl.col("TAXITIME_SEC_mvt") - pl.col("oof")).alias("err")
).with_columns(
    (pl.col("err") ** 2).alias("se"),
    pl.col("mvt_minus_aobt3").is_null().alias("unm"),
    (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("blk_sched"),
    (pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_aobt3")).alias("d"),
)
TOT = d["se"].sum()
print(f"scored rows {d.height}  total SSE {TOT:.5e}  RMSE {(TOT/d.height)**0.5:.3f}")
top = d.sort("se", descending=True).head(200)
roll = top.filter(pl.col("TAXITIME_SEC_mvt") > 72000)
lirf_fill = top.filter(
    (pl.col("TAXITIME_SEC_mvt") <= 72000) & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("blk_sched").abs() <= 10))
mine = top.filter(~(
    (pl.col("TAXITIME_SEC_mvt") > 72000)
    | ((pl.col("ADEP_mvt") == "LIRF") & (pl.col("blk_sched").abs() <= 10))))
print(f"top200 SSE share {top['se'].sum()/TOT:.4f}; rollover {roll.height}; LIRF fill {lirf_fill.height}; "
      f"mine {mine.height} carrying {mine['se'].sum()/TOT:.4f}; unmatched {mine['unm'].sum()}")
print(mine.group_by("ADEP_mvt").agg(pl.len(), (pl.col("se").sum()/TOT).alias("sse_share")).sort("sse_share", descending=True))

# ------------------------------------------------- 1. per-row raw assembly
hdr("1. PER-ROW CONTEXT FROM data/raw (all movements, DEP and ARR, at the 10 airports)")
cols = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
        "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "AOBT_3_flt", "ARVT_3_flt"]
raw = pl.scan_parquet(RAW).select(cols).with_columns(
    pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("apt")
).collect()
dep_raw, arr_raw = raw.filter(pl.col("PHASE_mvt") == "DEP"), raw.filter(pl.col("PHASE_mvt") == "ARR")
print(f"raw movements {raw.height} (DEP {dep_raw.height} / ARR {arr_raw.height})")

# same flight-number history at the same airport (whole year, the row itself included)
fnh = dep_raw.group_by(["apt", "FLIGHT_mvt"]).agg(
    pl.len().alias("fn_n"), pl.col("TAXITIME_SEC_mvt").median().alias("fn_med"),
    pl.col("TAXITIME_SEC_mvt").quantile(0.9).alias("fn_p90"))

ctx = []
for row in mine.sort("se", descending=True).iter_rows(named=True):
    a, mvt, blk, st = row["ADEP_mvt"], row["MVT_TIME_UTC_mvt"], row["BLOCK_TIME_UTC_mvt"], row["STAND_mvt"]
    nb = dep_raw.filter(
        (pl.col("apt") == a) & (pl.col("MVT_ID_mvt") != row["MVT_ID_mvt"])
        & (pl.col("MVT_TIME_UTC_mvt") >= mvt - pl.duration(hours=1))
        & (pl.col("MVT_TIME_UTC_mvt") <= mvt + pl.duration(hours=1)))["TAXITIME_SEC_mvt"].drop_nulls()
    pa = None
    if st is not None:
        pas = arr_raw.filter((pl.col("apt") == a) & (pl.col("STAND_mvt") == st)
                             & (pl.col("BLOCK_TIME_UTC_mvt") <= blk)
                             & (pl.col("BLOCK_TIME_UTC_mvt") >= blk - pl.duration(hours=24))
                             ).sort("BLOCK_TIME_UTC_mvt")
        if pas.height:
            pa = pas.tail(1).to_dicts()[0]
    sf = None
    if row["FLIGHT_ID_mvt"] is not None:
        s = raw.filter((pl.col("FLIGHT_ID_mvt") == row["FLIGHT_ID_mvt"]) & (pl.col("PHASE_mvt") == "ARR"))
        if s.height:
            sf = s.head(1).to_dicts()[0]
    ctx.append(dict(
        MVT_ID_mvt=row["MVT_ID_mvt"], nb_n=nb.len(),
        nb_med=float(nb.median()) if nb.len() else None,
        nb_p90=float(nb.quantile(0.9)) if nb.len() else None,
        nb_frac_gt3600=float((nb > 3600).mean()) if nb.len() else None,
        pa_gap_s=(blk - pa["BLOCK_TIME_UTC_mvt"]).total_seconds() if pa else None,
        pa_flight=pa["FLIGHT_mvt"] if pa else None,
        sf_arr_apt=sf["apt"] if sf else None))
ctx = pl.DataFrame(ctx, infer_schema_length=None)
m = (mine.sort("se", descending=True)
     .join(ctx, on="MVT_ID_mvt")
     .join(fnh, left_on=["ADEP_mvt", "FLIGHT_mvt"], right_on=["apt", "FLIGHT_mvt"], how="left")
     .with_columns((pl.col("TAXITIME_SEC_mvt") / pl.col("nb_med")).alias("y_over_nb"),
                   (pl.col("TAXITIME_SEC_mvt") / pl.col("fn_med")).alias("y_over_fn")))
print("ARRIVAL row of the same FLIGHT_ID present for", m["sf_arr_apt"].is_not_null().sum(), "of 141 rows")
print("previous ARR at the same stand within 24 h found for", m["pa_gap_s"].is_not_null().sum(), "of 141 rows")

hdr("1b. THE FULL 141, ONE LINE EACH (sorted by squared error)")
for x in m.iter_rows(named=True):
    print(f"{x['ADEP_mvt']} {str(x['FLIGHT_mvt']):>9} id={x['MVT_ID_mvt']:.0f} y={x['TAXITIME_SEC_mvt']:6d} "
          f"pred={x['oof']:8.0f} err={x['err']:8.0f} sse%={100*x['se']/TOT:5.3f} "
          f"{'U' if x['unm'] else 'M'} ms={x['mvt_minus_sched']:9.0f} blk-sched={x['blk_sched']:9.0f} "
          f"d={x['d']} proxy={x['mvt_minus_aobt3']} | nbr n={x['nb_n']:3d} med={x['nb_med']} p90={x['nb_p90']} "
          f"f>1h={x['nb_frac_gt3600']:.2f} y/nb={x['y_over_nb']:6.2f} | fltnum n={x['fn_n']} med={x['fn_med']} "
          f"y/fn={x['y_over_fn'] if x['y_over_fn'] is None else round(x['y_over_fn'],1)} | "
          f"BLK={x['BLOCK_TIME_UTC_mvt']} MVT={x['MVT_TIME_UTC_mvt']} SCHED={x['SCHED_TIME_UTC_mvt']} "
          f"stand={x['STAND_mvt']} rwy={x['RUNWAY_mvt']} act={x['AIRCRAFT_TYPE_mvt']} prevARRgap={x['pa_gap_s']}")

# ------------------------------------------------------------ 2. the kinds
hdr("2. THE SPLIT: y RELATIVE TO THE CONCURRENT AIRPORT MEDIAN vs THE SIGN OF err")
print(m.with_columns(pl.when(pl.col("err") < 0).then(pl.lit("over-predicted")).otherwise(pl.lit("under-predicted")).alias("g"))
      .group_by("g").agg(pl.len(), pl.col("y_over_nb").min().alias("min_y/nb"),
                         pl.col("y_over_nb").max().alias("max_y/nb"),
                         pl.col("y_over_nb").median().alias("med_y/nb"),
                         (pl.col("se").sum()/TOT).alias("sse_share")))

# K1 BLOCK from another day; K2 stale SCHED non-LIRF; K3 LIRF non-filled branch;
# K4/K5 label inflated (unmatched / matched); K6 model-B output with no visible driver
m = m.with_columns(
    pl.when(pl.col("err") < 0).then(
        pl.when(pl.col("mvt_minus_sched") < 3600).then(pl.lit("K6-no-visible-driver"))
        .when(pl.col("ADEP_mvt") == "LIRF").then(pl.lit("K3-LIRF-nonfilled-branch"))
        .otherwise(pl.lit("K2-stale-sched-nonLIRF")))
    .otherwise(
        pl.when(pl.col("TAXITIME_SEC_mvt") > 21600).then(pl.lit("K1-block-from-another-day"))
        .when(pl.col("unm")).then(pl.lit("K4-inflated-label-unmatched"))
        .otherwise(pl.lit("K5-inflated-label-matched"))).alias("kind"))
print(m.group_by("kind").agg(
    pl.len(), (pl.col("se").sum()/TOT).alias("sse_share"),
    pl.col("err").mean().alias("bias"), (pl.col("se").mean().sqrt()).alias("rmse"),
    pl.col("TAXITIME_SEC_mvt").median().alias("med_y"), pl.col("oof").median().alias("med_pred"),
    pl.col("mvt_minus_sched").median().alias("med_ms")).sort("sse_share", descending=True))
print("\nper airport x kind:")
print(m.group_by(["kind", "ADEP_mvt"]).agg(pl.len(), (pl.col("se").sum()/TOT).alias("sse_share")).sort(["kind", "sse_share"], descending=[False, True]))

hdr("2b. LFPG, ALL 18 ROWS")
for x in m.filter(pl.col("ADEP_mvt") == "LFPG").iter_rows(named=True):
    print(f"  {x['kind']:26s} {str(x['FLIGHT_mvt']):>9} y={x['TAXITIME_SEC_mvt']:6d} pred={x['oof']:8.0f} "
          f"sse%={100*x['se']/TOT:5.3f} ms={x['mvt_minus_sched']:9.0f} d={x['d']} y/fn={None if x['y_over_fn'] is None else round(x['y_over_fn'],1)} "
          f"nb_med={x['nb_med']} BLK={x['BLOCK_TIME_UTC_mvt']} MVT={x['MVT_TIME_UTC_mvt']}")

hdr("2c. WORKED EXAMPLE FOR KIND B: the whole flight-number history of LFPG EJU42AY")
print(dep_raw.filter((pl.col("apt") == "LFPG") & (pl.col("FLIGHT_mvt") == "EJU42AY")).sort("MVT_TIME_UTC_mvt")
      .select(["MVT_ID_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
               "TAXITIME_SEC_mvt", "STAND_mvt"]).head(60))

hdr("2d. THE THREE K6 ROWS (over-predicted with ms < 1 h)")
print(m.filter(pl.col("kind") == "K6-no-visible-driver").select(
    ["ADEP_mvt", "FLIGHT_mvt", "MVT_ID_mvt", "TAXITIME_SEC_mvt", "oof", "mvt_minus_sched", "blk_sched",
     "nb_med", "fn_n", "fn_med", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"]))

hdr("2e. K5: DIRECT EVIDENCE ON THE MATCHED ROWS (d = AOBT_3 - BLOCK)")
print(m.filter(pl.col("kind") == "K5-inflated-label-matched").select(
    ["ADEP_mvt", "FLIGHT_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "d", "fn_med", "nb_med", "oof", "err"]))

hdr("2f. K4/K5: IS THE AIRPORT-HOUR ITSELF DISRUPTED? episode windows vs the previous day")
ep = d.select(["ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "d"])
import datetime as dt
T = lambda s: dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)
for apt, t0, t1, tag in [("LTFM", "2025-07-26 12:00", "2025-07-26 16:30", "episode"),
                         ("LTFM", "2025-07-25 12:00", "2025-07-25 16:30", "control d-1"),
                         ("EGLL", "2025-07-30 14:00", "2025-07-30 18:00", "episode"),
                         ("EGLL", "2025-07-29 14:00", "2025-07-29 18:00", "control d-1"),
                         ("LFPG", "2025-11-22 09:00", "2025-11-22 14:00", "episode"),
                         ("LFPG", "2025-11-21 09:00", "2025-11-21 14:00", "control d-1"),
                         ("LFPG", "2025-12-05 09:00", "2025-12-05 14:00", "episode"),
                         ("LFPG", "2025-12-04 09:00", "2025-12-04 14:00", "control d-1")]:
    s = ep.filter((pl.col("ADEP_mvt") == apt) & (pl.col("MVT_TIME_UTC_mvt") >= T(t0)) & (pl.col("MVT_TIME_UTC_mvt") <= T(t1)))
    mm = s.filter(pl.col("mvt_minus_aobt3").is_not_null())
    print(f"{apt} {t0[:16]} {tag:11s} n={s.height:4d} y med={s['TAXITIME_SEC_mvt'].median():7.0f} p90={s['TAXITIME_SEC_mvt'].quantile(.9):7.0f} "
          f"f(y>1h)={float((s['TAXITIME_SEC_mvt']>3600).mean()):.3f} || matched n={mm.height:4d} "
          f"proxy med={mm['mvt_minus_aobt3'].median():7.0f} p90={mm['mvt_minus_aobt3'].quantile(.9):7.0f} "
          f"f(proxy>1h)={float((mm['mvt_minus_aobt3']>3600).mean()):.3f} d med={mm['d'].median():6.0f} p90={mm['d'].quantile(.9):7.0f} "
          f"f(d>1800)={float((mm['d']>1800).mean()):.3f}")

# ------------------------------------- 3. (a)/(b)/(c): the kind-A candidate rule
hdr("3. K2 / K3 -- (a) IDENTIFIABILITY, (b) BIAS vs SPREAD, (c) CONTROL")
print("ranking null rates, columns the rule would use:")
fr = pl.read_parquet(FR, columns=["MVT_ID_mvt", "ADEP_mvt", "mvt_minus_sched", "mvt_minus_aobt3", "aobt3_null"])
for c in ["ADEP_mvt", "mvt_minus_sched", "mvt_minus_aobt3", "aobt3_null"]:
    print(f"  {c:20s} null {fr[c].null_count():7d} / {fr.height}")

EDGES = [-1e18, 0, 1800, 3600, 5400, 7200, 10800, 14400, 21600, 43200, 86400, 1e18]
def band(col):
    e = pl.when(pl.col(col) < EDGES[1]).then(pl.lit("<0"))
    for i in range(1, len(EDGES) - 1):
        e = e.when((pl.col(col) >= EDGES[i]) & (pl.col(col) < EDGES[i + 1])).then(pl.lit(f"{EDGES[i]:.0f}..{EDGES[i+1]:.0f}"))
    return e.otherwise(pl.lit("?")).alias("ms_band")

un = d.filter(pl.col("unm"))
print(f"\nscored unmatched {un.height} rows, {un['se'].sum()/TOT:.4f} of total SSE")
for nm, sub in [("LIRF unmatched", un.filter(pl.col("ADEP_mvt") == "LIRF")),
                ("non-LIRF unmatched", un.filter(pl.col("ADEP_mvt") != "LIRF"))]:
    g = sub.with_columns(band("mvt_minus_sched")).group_by("ms_band").agg(
        pl.len().alias("n"), pl.col("TAXITIME_SEC_mvt").mean().alias("mean_y"),
        pl.col("TAXITIME_SEC_mvt").median().alias("med_y"),
        (pl.col("TAXITIME_SEC_mvt") <= 3600).mean().alias("frac_y_le_1h"),
        (pl.col("blk_sched").abs() <= 10).mean().alias("fill_rate"),
        pl.col("oof").mean().alias("mean_pred"), pl.col("err").mean().alias("bias"),
        (pl.col("se").mean().sqrt()).alias("rmse"),
        (pl.col("err").mean() ** 2 / pl.col("se").mean()).alias("bias_share_of_mse"),
        (pl.col("se").sum() / TOT).alias("sse_share")).sort("ms_band")
    print(f"\n-- {nm}"); print(g)

print("\n(a) precision / recall of `unmatched & ADEP!=LIRF & ms>=T` for the kind "
      "(kind = the model over-predicts by >3000 s, i.e. oof - y > 3000):")
tgt = (pl.col("oof") - pl.col("TAXITIME_SEC_mvt") > 3000)
base = d.select(tgt.mean()).item()
print(f"  base rate of the kind over all {d.height} scored rows: {base:.6f} ({d.select(tgt.sum()).item()} rows)")
for T_ in [7200, 14400, 21600, 28800, 43200, 86400]:
    sel = d.filter(pl.col("unm") & (pl.col("ADEP_mvt") != "LIRF") & (pl.col("mvt_minus_sched") >= T_))
    n_sel, n_hit = sel.height, sel.select(tgt.sum()).item()
    n_tot = d.select(tgt.sum()).item()
    print(f"  T={T_:6d}s  flagged {n_sel:5d}  precision {n_hit/max(n_sel,1):.4f}  recall {n_hit/n_tot:.4f}  "
          f"bias {sel['err'].mean():9.1f}  rmse {(sel['se'].mean())**0.5:9.1f}  "
          f"bias share of MSE {sel['err'].mean()**2/sel['se'].mean():.4f}  SSE share {sel['se'].sum()/TOT:.4f}")
print("  same rule at LIRF (contrast):")
for T_ in [3600, 7200, 14400, 21600, 43200]:
    sel = d.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("mvt_minus_sched") >= T_))
    n_hit = sel.select(tgt.sum()).item()
    print(f"  T={T_:6d}s  flagged {sel.height:5d}  precision {n_hit/max(sel.height,1):.4f}  "
          f"bias {sel['err'].mean():9.1f}  rmse {(sel['se'].mean())**0.5:9.1f}  "
          f"bias share of MSE {sel['err'].mean()**2/sel['se'].mean():.4f}  "
          f"fill rate {float((sel['blk_sched'].abs()<=10).mean()):.3f}  frac y<=1h {float((sel['TAXITIME_SEC_mvt']<=3600).mean()):.3f}")

print("\n(c) control: the visible cell vs the label-selected subset inside it "
      "(non-LIRF unmatched, ms>=21600)")
cell = d.filter(pl.col("unm") & (pl.col("ADEP_mvt") != "LIRF") & (pl.col("mvt_minus_sched") >= 21600))
inset = cell.join(m.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi")
outset = cell.join(m.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti")
for nm, s in [("whole visible cell", cell), ("the part inside the 141", inset), ("the rest of the cell", outset)]:
    if s.height:
        print(f"  {nm:24s} n={s.height:4d} bias={s['err'].mean():10.1f} rmse={(s['se'].mean())**0.5:10.1f} "
              f"bias share of MSE={s['err'].mean()**2/s['se'].mean():.4f} med_y={s['TAXITIME_SEC_mvt'].median():.0f} "
              f"max_y={s['TAXITIME_SEC_mvt'].max():.0f} SSE share={s['se'].sum()/TOT:.4f}")
print("  -- the same question for the 39,224-row precedent: is the harm a property of the "
      "visible cell or only of its label-selected part?")

# ---------------------------------------------- 4. kind B identifiability
hdr("4. K4 / K5 -- (a) IS IT IDENTIFIABLE? (c) IS IT JUST A LARGE RESIDUAL?")
# define the kind over all scored rows: y >= 3 x the median y of departures at the same
# airport with MVT within +-1 h (computable in training only; the label is null in ranking)
loc = (d.select(["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
       .join(ft.select(["MVT_ID_mvt"]), on="MVT_ID_mvt"))
allday = (pl.scan_parquet(RAW).filter(pl.col("PHASE_mvt") == "DEP")
          .select(["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]).collect()
          .with_columns((pl.col("MVT_TIME_UTC_mvt") + pl.duration(hours=1)).alias("t_end"))
          .sort(["ADEP_mvt", "t_end"]))
allday = allday.with_columns(
    pl.col("TAXITIME_SEC_mvt").rolling_median_by("t_end", window_size="2h", closed="both")
    .over("ADEP_mvt").alias("local_med"))
dd = d.join(allday.select(["MVT_ID_mvt", "local_med"]), on="MVT_ID_mvt", how="left")
dd = dd.with_columns((pl.col("TAXITIME_SEC_mvt") / pl.col("local_med")).alias("y_over_local"))
kb = (pl.col("y_over_local") >= 3.0)
print(f"kind-B target (y >= 3 x local 2 h median): {dd.select(kb.sum()).item()} of {dd.height} scored rows, "
      f"base rate {dd.select(kb.mean()).item():.5f}, carrying "
      f"{dd.filter(kb)['se'].sum()/TOT:.4f} of total SSE")
print(f"  of the 141: {m.height} rows, of which in kind B by the sign of err: {m.filter(pl.col('err')>0).height}")
print("\nAUC of every prediction-time-visible column for the kind-B indicator "
      "(scored rows with a defined local median):")
vis = [c for c in ["mvt_minus_sched", "mvt_minus_eobt1", "mvt_minus_lobt", "mvt_minus_iobt",
                   "aobt3_minus_eobt1", "aobt3_minus_sched", "lobt_minus_eobt1", "queue_ahead",
                   "rwy_headway_prev_s", "rwy_headway_next_s", "dep_tko_prev_60m", "dep_tko_next_60m",
                   "arr_land_prev_60m", "nbr_mvt_minus_aobt_mean_30m", "nbr_aobt_delay_mean_30m",
                   "arr_taxi_in_mean_30m", "congestion_vs_usual_30m", "active_runways_60m",
                   "dep_arr_ratio_30m", "stand_rwy_share", "bank_position", "planned_flight_s",
                   "vis_km", "low_vis", "precip_mm", "frozen_wx", "deicing_gate", "hour_utc"] if c in dd.columns]
sub = dd.filter(pl.col("local_med").is_not_null())
lab = sub.select(kb.cast(pl.Int8).alias("t"))["t"].to_numpy()
import numpy as np
npos, nneg = int(lab.sum()), int((1 - lab).sum())
rows = []
for c in vis:
    v = sub[c].to_numpy().astype("float64")
    ok = ~np.isnan(v)
    if ok.sum() < 1000:
        continue
    vv, ll = np.abs(v[ok]), lab[ok]
    order = np.argsort(vv, kind="stable")
    ranks = np.empty(len(vv), dtype="float64"); ranks[order] = np.arange(1, len(vv) + 1)
    p, n = ll.sum(), (1 - ll).sum()
    auc = (ranks[ll == 1].sum() - p * (p + 1) / 2) / (p * n) if p and n else float("nan")
    rows.append((c, max(auc, 1 - auc)))
for c, a in sorted(rows, key=lambda t: -t[1])[:12]:
    print(f"  |{c:32s}| AUC {a:.4f}")
print(f"  (positives {npos}, negatives {nneg})")

print("\nbest visible flag: precision / recall sweep on the best column")
bestc = sorted(rows, key=lambda t: -t[1])[0][0]
v = np.abs(sub[bestc].to_numpy().astype("float64"))
for q in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]:
    thr = np.nanquantile(v, q)
    f = (v >= thr) & ~np.isnan(v)
    tp = int((f & (lab == 1)).sum())
    print(f"  q={q:<6} thr={thr:12.1f} flagged {int(f.sum()):7d} precision {tp/max(f.sum(),1):.4f} recall {tp/max(npos,1):.4f}")

print("\n(b)/(c) bias vs spread on the kind and on the best visible flag:")
kbs = dd.filter(kb)
print(f"  kind B (label-selected)   n={kbs.height:6d} bias={kbs['err'].mean():9.1f} rmse={(kbs['se'].mean())**0.5:9.1f} "
      f"bias share of MSE={kbs['err'].mean()**2/kbs['se'].mean():.4f}")
thr90 = np.nanquantile(v, 0.9)
flagids = sub.filter(pl.col(bestc).abs() >= thr90)["MVT_ID_mvt"]
fs = dd.filter(pl.col("MVT_ID_mvt").is_in(flagids))
print(f"  best visible flag (q90)   n={fs.height:6d} bias={fs['err'].mean():9.1f} rmse={(fs['se'].mean())**0.5:9.1f} "
      f"bias share of MSE={fs['err'].mean()**2/fs['se'].mean():.4f}")

# ------------------------------------------- 5. (d)/(e) ranking set and ceiling
hdr("5. (d) THE RANKING SET UNDER THE K2 RULE, (e) HONEST CEILING")
frf = pl.read_parquet(FR, columns=["MVT_ID_mvt", "ADEP_mvt", "mvt_minus_sched", "mvt_minus_aobt3"])
frf = frf.with_columns(pl.col("mvt_minus_aobt3").is_null().alias("unm"))
print(f"ranking rows {frf.height}, unmatched {frf['unm'].sum()}")
for T_ in [7200, 14400, 21600, 43200, 86400]:
    s = frf.filter(pl.col("unm") & (pl.col("ADEP_mvt") != "LIRF") & (pl.col("mvt_minus_sched") >= T_))
    print(f"  non-LIRF unmatched ms>={T_:6d}: {s.height:5d} ranking rows")
print(frf.filter(pl.col("unm") & (pl.col("ADEP_mvt") != "LIRF") & (pl.col("mvt_minus_sched") >= 21600))
      .group_by("ADEP_mvt").agg(pl.len()).sort("len", descending=True))

subs = {"v7 outputs/main.parquet": "outputs/main.parquet",
        "v10 outputs/submit_v10_reparam.parquet": "outputs/submit_v10_reparam.parquet"}
sel_ids = frf.filter(pl.col("unm") & (pl.col("ADEP_mvt") != "LIRF") & (pl.col("mvt_minus_sched") >= 21600))["MVT_ID_mvt"]
for nm, p in subs.items():
    try:
        sp = pl.read_parquet(p)
    except Exception as e:
        print("  (missing)", p, e); continue
    pc = [c for c in sp.columns if c != "MVT_ID_mvt"][0]
    q = sp.filter(pl.col("MVT_ID_mvt").is_in(sel_ids))[pc]
    print(f"  {nm:42s} on those rows: n={q.len()} median={q.median():.0f} p90={q.quantile(.9):.0f} "
          f"max={q.max():.0f} n>=7200={int((q>=7200).sum())} n>=14400={int((q>=14400).sum())}")

print("\nleaderboard arithmetic (task's constants):")
SSE_TOT = LB_BEST ** 2 * N_RANK
print(f"  LB {LB_BEST} -> total SSE {SSE_TOT:.5e}; matched {N_MATCHED} at {RMSE_MATCHED} -> "
      f"implied unmatched RMSE {(((SSE_TOT - N_MATCHED*RMSE_MATCHED**2)/N_UNMATCHED)**0.5):.1f}")
for nm, p in subs.items():
    try:
        sp = pl.read_parquet(p)
    except Exception:
        continue
    pc = [c for c in sp.columns if c != "MVT_ID_mvt"][0]
    q = sp.filter(pl.col("MVT_ID_mvt").is_in(sel_ids))[pc].to_numpy()
    for repl, tag in [(1200.0, "replace by 1200 s (the cell's training mean y)")]:
        # worst case: current prediction is right and we break it; best case: label ~1200
        dsse_best = float(((q - 1200.0) ** 2).sum())   # SSE removed if every label were 1200
        print(f"  {nm}: {tag}; if every label in the cell were 1,200 s, SSE removed = {dsse_best:.4e} "
              f"-> LB {((SSE_TOT - dsse_best)/N_RANK)**0.5:.4f} (gain {LB_BEST - ((SSE_TOT - dsse_best)/N_RANK)**0.5:+.4f} s)")
print("\nKind B ceiling: the 38 rows are selected by the label; no visible rule exists (section 4). "
      "Oracle arithmetic for the prevalence of the kind-B target in the ranking set:")
prev = dd.filter(pl.col("local_med").is_not_null()).select(kb.mean()).item()
n_rank_kb = prev * N_RANK
mse_kb = dd.filter(kb)["se"].mean()
print(f"  prevalence {prev:.5f} -> {n_rank_kb:.0f} ranking rows, own MSE {mse_kb:.0f}; "
      f"perfect oracle on them -> LB {((SSE_TOT - n_rank_kb*mse_kb)/N_RANK)**0.5:.4f} "
      f"(gain {LB_BEST - ((SSE_TOT - n_rank_kb*mse_kb)/N_RANK)**0.5:+.4f} s) -- label-selected, not achievable")

# ------------------------------------------------ 6. supplementary sections
hdr("6. HOW RARE IS THE BLOCK-FROM-ANOTHER-DAY KIND? (whole training year)")
ftx = ft.with_columns((pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_aobt3")).alias("d"))
for ylim, mslim in [(10800, 3600), (21600, 7200), (43200, 7200)]:
    s = ftx.filter((pl.col("TAXITIME_SEC_mvt") > ylim) & (pl.col("mvt_minus_sched").abs() < mslim))
    ss = d.filter((pl.col("TAXITIME_SEC_mvt") > ylim) & (pl.col("mvt_minus_sched").abs() < mslim))
    print(f"  y>{ylim} and |ms|<{mslim}: {s.height} of 2,085,047 training rows; {ss.height} scored, "
          f"SSE share {ss['se'].sum()/TOT:.4f}")
print(ftx.filter((pl.col("TAXITIME_SEC_mvt") > 21600) & (pl.col("mvt_minus_sched").abs() < 7200))
      .select(["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "TAXITIME_SEC_mvt", "mvt_minus_sched",
               "mvt_minus_aobt3", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]))

hdr("7. THE THREE K6 ROWS IN CONTEXT: model-B output >= 4000 s where ms < 3600 s")
q = d.filter(pl.col("unm") & (pl.col("mvt_minus_sched") < 3600))
qq = q.filter(pl.col("oof") >= 4000)
print(f"  unmatched scored rows with ms < 3600: {q.height}; of those the model predicts >= 4000 s on "
      f"{qq.height}; their labels {sorted(qq['TAXITIME_SEC_mvt'].to_list())}; SSE share {qq['se'].sum()/TOT:.4f}")

hdr("8. BLOCK-SIDE EPISODES: airport x 3 h blocks where the label inflates but the real taxi does not")
mm = d.filter(~pl.col("unm")).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("3h").alias("blk3"))
g3 = (mm.group_by(["ADEP_mvt", "blk3"]).agg(
    pl.len().alias("n"), (pl.col("d") > 1800).mean().alias("f_d_gt1800"),
    (pl.col("mvt_minus_aobt3") > 3600).mean().alias("f_proxy_gt1h"),
    (pl.col("TAXITIME_SEC_mvt") > 3600).mean().alias("f_y_gt1h"),
    (pl.col("se").sum() / TOT).alias("sse_share")).filter(pl.col("n") >= 20))
print(f"  airport x 3 h blocks with >= 20 matched scored departures: {g3.height}")
for thr in [0.10, 0.15, 0.20, 0.30]:
    e = g3.filter(pl.col("f_d_gt1800") >= thr)
    print(f"    f(d>1800 s) >= {thr}: {e.height} blocks, {e['n'].sum()} rows, SSE share {e['sse_share'].sum():.4f}, "
          f"median f(proxy>1 h) in them {e['f_proxy_gt1h'].median():.3f} (over all blocks {g3['f_proxy_gt1h'].median():.3f})")
print(g3.filter(pl.col("f_d_gt1800") >= 0.15).sort("sse_share", descending=True))

hdr("9. THE LIRF CELL IN THE RANKING SET (K3) UNDER v10, WHICH ALREADY REPARAMETRISES IT")
sel_lirf = frf.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("mvt_minus_sched") >= 3600))["MVT_ID_mvt"]
print(f"  LIRF unmatched ranking rows with ms >= 3600: {sel_lirf.len()}")
for nm, p in subs.items():
    try:
        sp = pl.read_parquet(p)
    except Exception:
        continue
    pc = [c for c in sp.columns if c != "MVT_ID_mvt"][0]
    q2 = sp.filter(pl.col("MVT_ID_mvt").is_in(sel_lirf))[pc]
    print(f"  {nm:42s} median={q2.median():.0f} p90={q2.quantile(.9):.0f} max={q2.max():.0f} "
          f"n>=7200={int((q2>=7200).sum())}")

"""Item 6 `return-to-stand-twins` -- independent re-derivation (no training).

Hypothesis (plan.md §1 item 6): departures recorded twice with the same (ADEP, FLIGHT, SCHED)
and take-off within 2 min; the matched twin carries the earlier pushback's BLOCK, so its label
is inflated.

Inputs : data/raw/training_*.parquet (DEP rows), data/raw/ranking.parquet (DEP rows),
         experiments/oof_main-48c19bc830.parquet, outputs/*.parquet (main.parquet primarily).
Run    : POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/return-to-stand-twins.py
"""
import os
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"
import glob, math
import polars as pl

pl.Config.set_tbl_rows(120); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260)
ROOT = "/home/user/prc-data-challenge/"
R = ROOT + "data/raw/"
KEY = ["ADEP_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"]
COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
        "CALLSIGN_flt", "AIRCRAFT_OPERATOR_flt"]
sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()

def prep(lf):
    return (lf.filter(pl.col("PHASE_mvt") == "DEP").select(COLS)
            .with_columns([pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),     # D's definition
                           pl.col("AOBT_3_flt").is_null().alias("unm_route"),        # routing definition (model B)
                           pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
                           sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("ms"),
                           sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("bs")]))

tr = prep(pl.scan_parquet(sorted(glob.glob(R + "training_*.parquet"))))
rk = prep(pl.scan_parquet(R + "ranking.parquet"))

# ---------------------------------------------------------------- A. group sizes (training)
print("=== A. training DEP rows and (ADEP, FLIGHT, SCHED) group sizes ===")
n_tr = tr.select(pl.len()).collect().item()
n_tr_fl = tr.filter(pl.col("FLIGHT_mvt").is_not_null()).select(pl.len()).collect().item()
print("training DEP rows:", n_tr, "| with FLIGHT_mvt non-null:", n_tr_fl)
gsz = (tr.filter(pl.col("FLIGHT_mvt").is_not_null()).group_by(KEY).len()
       .group_by("len").agg(pl.len().alias("n_groups")).sort("len").collect())
print(gsz)
n_multi_groups = gsz.filter(pl.col("len") > 1)["n_groups"].sum()
n_multi_rows = (gsz.filter(pl.col("len") > 1)["len"] * gsz.filter(pl.col("len") > 1)["n_groups"]).sum()
print(f"groups with len>1: {n_multi_groups}  rows in them: {n_multi_rows}  size-2 groups (pairs): {gsz.filter(pl.col('len')==2)['n_groups'].sum()}")

# all rows in multi-row groups, sorted by MVT within group
tw = (tr.filter(pl.col("FLIGHT_mvt").is_not_null())
      .with_columns(pl.len().over(KEY).alias("k"))
      .filter(pl.col("k") >= 2)
      .sort(KEY + ["MVT_TIME_UTC_mvt"])
      .with_columns(pl.int_range(pl.len()).over(KEY).alias("pos"))
      .collect())
print("twin rows collected:", tw.height, "| by ADEP:")
print(tw.group_by("ADEP_mvt").agg([pl.len(), pl.col("unm_fid").mean().round(3).alias("unm_fid_rate"),
                                   pl.col("unm_route").mean().round(3).alias("unm_route_rate"),
                                   pl.col("y").median().alias("y_med")]).sort("len", descending=True))
print("unm_fid vs unm_route disagreement on twin rows:", (tw["unm_fid"] != tw["unm_route"]).sum())

# ---------------------------------------------------------------- B. size-2 pairs
print("\n=== B. size-2 pairs (first = earlier MVT) ===")
p2 = tw.filter(pl.col("k") == 2)
def side(col, i, name):
    return pl.col(col).filter(pl.col("pos") == i).first().alias(name)
pairs = p2.group_by(KEY).agg([
    side("unm_fid", 0, "unm1"), side("unm_fid", 1, "unm2"),
    side("y", 0, "y1"), side("y", 1, "y2"),
    side("MVT_TIME_UTC_mvt", 0, "mvt1"), side("MVT_TIME_UTC_mvt", 1, "mvt2"),
    side("BLOCK_TIME_UTC_mvt", 0, "blk1"), side("BLOCK_TIME_UTC_mvt", 1, "blk2"),
    side("AOBT_3_flt", 0, "aobt1"), side("AOBT_3_flt", 1, "aobt2"),
    side("STAND_mvt", 0, "st1"), side("STAND_mvt", 1, "st2"),
    side("RUNWAY_mvt", 0, "rw1"), side("RUNWAY_mvt", 1, "rw2"),
    side("MVT_ID_mvt", 0, "id1"), side("MVT_ID_mvt", 1, "id2"),
]).with_columns([
    (pl.col("mvt2") - pl.col("mvt1")).dt.total_seconds().alias("dt_mvt"),
    (pl.col("blk2") - pl.col("blk1")).dt.total_seconds().alias("dt_blk"),
    pl.when(pl.col("unm1") & ~pl.col("unm2")).then(pl.lit("unm,mat"))
      .when(~pl.col("unm1") & pl.col("unm2")).then(pl.lit("mat,unm"))
      .when(pl.col("unm1") & pl.col("unm2")).then(pl.lit("unm,unm")).otherwise(pl.lit("mat,mat")).alias("kind"),
])
print("pairs:", pairs.height, "| MVT_IDs distinct within pair:", (pairs["id1"] != pairs["id2"]).all())
print(pairs.group_by("kind").agg([
    pl.len(),
    (pl.col("dt_mvt") <= 120).mean().round(3).alias("dt_mvt<=120"),
    (pl.col("dt_mvt") == 0).mean().round(3).alias("dt_mvt==0"),
    pl.col("dt_mvt").median().alias("dt_mvt_med"),
    pl.col("y1").median().alias("y1_med"), pl.col("y2").median().alias("y2_med"),
    (pl.col("y1") > pl.col("y2")).mean().round(3).alias("y1>y2"),
    pl.col("dt_blk").median().alias("dt_blk_med"),
    (pl.col("st1") == pl.col("st2")).mean().round(3).alias("same_stand"),
    (pl.col("rw1") == pl.col("rw2")).mean().round(3).alias("same_rwy"),
]).sort("kind"))
print("dt_mvt distribution (all pairs) quantiles 0/.1/.25/.5/.75/.9/1:",
      [pairs["dt_mvt"].quantile(q) for q in (0, .1, .25, .5, .75, .9, 1)])

# mixed pairs: orient as (matched twin, unmatched twin)
mixed = pairs.filter(pl.col("unm1") != pl.col("unm2")).with_columns([
    pl.when(pl.col("unm1")).then(pl.col("y2")).otherwise(pl.col("y1")).alias("y_m"),
    pl.when(pl.col("unm1")).then(pl.col("y1")).otherwise(pl.col("y2")).alias("y_u"),
    pl.when(pl.col("unm1")).then(pl.col("blk2")).otherwise(pl.col("blk1")).alias("blk_m"),
    pl.when(pl.col("unm1")).then(pl.col("blk1")).otherwise(pl.col("blk2")).alias("blk_u"),
    pl.when(pl.col("unm1")).then(pl.col("mvt2")).otherwise(pl.col("mvt1")).alias("mvt_m"),
    pl.when(pl.col("unm1")).then(pl.col("mvt1")).otherwise(pl.col("mvt2")).alias("mvt_u"),
    pl.when(pl.col("unm1")).then(pl.col("aobt2")).otherwise(pl.col("aobt1")).alias("aobt_m"),
]).with_columns([
    (pl.col("blk_u") - pl.col("blk_m")).dt.total_seconds().alias("blk_u_minus_blk_m"),
    (pl.col("mvt_u") - pl.col("mvt_m")).dt.total_seconds().alias("mvt_u_minus_mvt_m"),
    (pl.col("blk_m") - pl.col("aobt_m")).dt.total_seconds().alias("blk_m_minus_aobt_m"),
    (pl.col("blk_u") - pl.col("aobt_m")).dt.total_seconds().alias("blk_u_minus_aobt_m"),
    (pl.col("y_m") - pl.col("y_u")).alias("y_m_minus_y_u"),
])
print("\n--- mixed pairs (one matched, one unmatched) ---")
print("n mixed:", mixed.height,
      "| same take-off (|dt_mvt|<=120):", (mixed["dt_mvt"].abs() <= 120).sum(),
      "| matched twin has EARLIER BLOCK (blk_m < blk_u):", (mixed["blk_u_minus_blk_m"] > 0).sum(),
      "| blk_m == blk_u:", (mixed["blk_u_minus_blk_m"] == 0).sum(),
      "| y_m > y_u:", (mixed["y_m"] > mixed["y_u"]).sum())
print("median y_m:", mixed["y_m"].median(), "median y_u:", mixed["y_u"].median(),
      "| y_m quantiles .1/.5/.9:", [mixed["y_m"].quantile(q) for q in (.1, .5, .9)],
      "| y_u quantiles .1/.5/.9:", [mixed["y_u"].quantile(q) for q in (.1, .5, .9)])
print("y_m >= 7200:", (mixed["y_m"] >= 7200).sum(), "| y_u >= 7200:", (mixed["y_u"] >= 7200).sum())
print("blk_u - blk_m quantiles .1/.5/.9:", [mixed["blk_u_minus_blk_m"].quantile(q) for q in (.1, .5, .9)])
print("blk_m - aobt_m (matched twin BLOCK vs its own AOBT_3) quantiles .1/.5/.9:",
      [mixed["blk_m_minus_aobt_m"].quantile(q) for q in (.1, .5, .9)],
      "| |blk_m - aobt_m| <= 300:", (mixed["blk_m_minus_aobt_m"].abs() <= 300).sum())
print("blk_u - aobt_m quantiles .1/.5/.9:", [mixed["blk_u_minus_aobt_m"].quantile(q) for q in (.1, .5, .9)],
      "| |blk_u - aobt_m| <= 300:", (mixed["blk_u_minus_aobt_m"].abs() <= 300).sum())
print("matched twin is the EARLIER MVT row (kind mat,unm):", (mixed["kind"] == "mat,unm").sum(),
      "| LATER (unm,mat):", (mixed["kind"] == "unm,mat").sum())
same_to = mixed.filter(pl.col("dt_mvt").abs() <= 120)
print("same-take-off mixed pairs:", same_to.height,
      "| y_m median", same_to["y_m"].median(), "| y_u median", same_to["y_u"].median(),
      "| y_m > y_u:", (same_to["y_m"] > same_to["y_u"]).sum(),
      "| blk_m < blk_u:", (same_to["blk_u_minus_blk_m"] > 0).sum())
print("same-take-off mixed by ADEP:")
print(same_to.group_by("ADEP_mvt").agg([pl.len(), pl.col("y_m").median().alias("y_m_med"),
                                        pl.col("y_u").median().alias("y_u_med")]).sort("len", descending=True))
print("examples (same-take-off mixed, largest y_m):")
print(same_to.sort("y_m", descending=True).select(KEY + ["kind", "dt_mvt", "y_m", "y_u", "blk_m", "blk_u", "aobt_m", "mvt_m", "st1", "st2"]).head(8))

# ---------------------------------------------------------------- C. twins vs extreme labels
print("\n=== C. twin rows vs the training label tail ===")
tail = tr.select([(pl.col("y") >= 7200).sum().alias("n_y>=2h"), (pl.col("y") >= 14400).sum().alias("n_y>=4h"),
                  pl.col("unm_fid").sum().alias("n_unm_fid"), pl.col("unm_route").sum().alias("n_unm_route")]).collect()
print(tail)
print("twin rows y>=2h:", (tw["y"] >= 7200).sum(), "| y>=4h:", (tw["y"] >= 14400).sum(),
      "| twin rows y quantiles .5/.9/.99/max:", [tw["y"].quantile(q) for q in (.5, .9, .99)], tw["y"].max())
print("twin rows by (unm_fid): ")
print(tw.group_by("unm_fid").agg([pl.len(), pl.col("y").median().alias("y_med"), (pl.col("y") >= 7200).sum().alias("n_y>=2h")]))

# ---------------------------------------------------------------- D. OOF error carried by twin rows
print("\n=== D. OOF (oof_main-48c19bc830) error on twin rows ===")
oof = pl.scan_parquet(ROOT + "experiments/oof_main-48c19bc830.parquet").filter(pl.col("oof").is_not_null())
oof_all = (tr.select(["MVT_ID_mvt", "y", "unm_route"]).join(oof, on="MVT_ID_mvt", how="inner")
           .with_columns(((pl.col("oof") - pl.col("y")) ** 2).alias("se"))
           .select([pl.len().alias("n"), pl.col("se").sum().alias("sse"), (pl.col("se").mean() ** 0.5).alias("rmse")]).collect())
print("OOF rows:", oof_all)
n_oof, sse_oof = oof_all["n"][0], oof_all["sse"][0]
tw_oof = tw.join(oof.collect(), on="MVT_ID_mvt", how="inner").with_columns(((pl.col("oof") - pl.col("y")) ** 2).alias("se"))
print("twin rows with OOF pred:", tw_oof.height, "| SSE:", round(tw_oof["se"].sum()), "| share of OOF SSE:",
      round(tw_oof["se"].sum() / sse_oof * 100, 3), "% | RMSE on twin rows:", round(tw_oof["se"].mean() ** 0.5, 1))
print(tw_oof.group_by(["k", "unm_fid"]).agg([pl.len(), (pl.col("se").mean() ** 0.5).round(1).alias("rmse"),
                                             pl.col("se").sum().round(0).alias("sse"),
                                             pl.col("y").median().alias("y_med"), pl.col("oof").median().alias("oof_med"),
                                             (pl.col("oof") - pl.col("y")).median().alias("resid_med")]).sort(["k", "unm_fid"]))
# mixed same-take-off pairs' matched twin specifically
mm_ids = pl.concat([same_to.select(pl.when(pl.col("unm1")).then(pl.col("id2")).otherwise(pl.col("id1")).alias("MVT_ID_mvt"))])
mt = tw_oof.filter(pl.col("MVT_ID_mvt").is_in(mm_ids["MVT_ID_mvt"].implode()))
print("matched twin of same-take-off mixed pairs with OOF:", mt.height, "| RMSE:", round(mt["se"].mean() ** 0.5, 1) if mt.height else None,
      "| SSE share:", round(mt["se"].sum() / sse_oof * 100, 3) if mt.height else None, "% | resid median:", mt.select((pl.col("oof") - pl.col("y")).median()).item() if mt.height else None)
# for reference: what share do the top-N rows take (context for 'share' numbers)
print("(context) OOF rows y>=2h share of SSE:",
      round(tr.select(["MVT_ID_mvt", "y"]).join(oof, on="MVT_ID_mvt").filter(pl.col("y") >= 7200)
            .select(((pl.col("oof") - pl.col("y")) ** 2).sum()).collect().item() / sse_oof * 100, 2), "%")

# ---------------------------------------------------------------- E. ranking twins and predictions
print("\n=== E. ranking (ADEP, FLIGHT, SCHED) groups with k>=2 and outputs/main.parquet predictions ===")
n_rk = rk.select(pl.len()).collect().item()
rtw = (rk.filter(pl.col("FLIGHT_mvt").is_not_null()).with_columns(pl.len().over(KEY).alias("k")).filter(pl.col("k") >= 2)
       .sort(KEY + ["MVT_TIME_UTC_mvt"]).with_columns(pl.int_range(pl.len()).over(KEY).alias("pos")).collect())
print("ranking DEP rows:", n_rk, "| twin rows:", rtw.height, "| groups:", rtw.select(pl.struct(KEY).n_unique()).item(),
      "| k distribution:", rtw.group_by("k").len().sort("k").to_dicts())
print(rtw.group_by("ADEP_mvt").agg([pl.len(), pl.col("unm_fid").mean().round(2).alias("unm_fid_rate"),
                                    pl.col("unm_route").mean().round(2).alias("unm_route_rate")]).sort("len", descending=True))
print("by month:", rtw.group_by(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo")).len().sort("mo").to_dicts())
# pair structure in ranking
rp = rtw.filter(pl.col("k") == 2).group_by(KEY).agg([
    side("unm_fid", 0, "unm1"), side("unm_fid", 1, "unm2"),
    side("MVT_TIME_UTC_mvt", 0, "mvt1"), side("MVT_TIME_UTC_mvt", 1, "mvt2"),
    side("BLOCK_TIME_UTC_mvt", 0, "blk1"), side("BLOCK_TIME_UTC_mvt", 1, "blk2"),
    side("MVT_ID_mvt", 0, "id1"), side("MVT_ID_mvt", 1, "id2"),
]).with_columns([(pl.col("mvt2") - pl.col("mvt1")).dt.total_seconds().alias("dt_mvt"),
                 (pl.col("blk2") - pl.col("blk1")).dt.total_seconds().alias("dt_blk"),
                 pl.when(pl.col("unm1") != pl.col("unm2")).then(pl.lit("mixed"))
                   .when(pl.col("unm1")).then(pl.lit("unm,unm")).otherwise(pl.lit("mat,mat")).alias("kind")])
print("ranking pairs:", rp.height)
print(rp.group_by("kind").agg([pl.len(), (pl.col("dt_mvt") <= 120).sum().alias("n_dt<=120"),
                               pl.col("dt_mvt").median().alias("dt_mvt_med"), pl.col("dt_blk").median().alias("dt_blk_med")]).sort("kind"))
n_mixed_same = rp.filter((pl.col("kind") == "mixed") & (pl.col("dt_mvt") <= 120)).height
print("ranking mixed pairs with same take-off (dt<=120):", n_mixed_same, "-> matched-twin rows at risk:", n_mixed_same)
# BLOCK ordering in ranking mixed pairs (BLOCK is present in ranking rows; only TAXITIME is hidden?)
print("ranking twin rows: BLOCK_TIME null:", rtw["BLOCK_TIME_UTC_mvt"].is_null().sum(), "| TAXITIME null:", rtw["TAXITIME_SEC_mvt"].is_null().sum())
rmix = rp.filter(pl.col("kind") == "mixed").with_columns([
    pl.when(pl.col("unm1")).then(pl.col("blk2")).otherwise(pl.col("blk1")).alias("blk_m"),
    pl.when(pl.col("unm1")).then(pl.col("blk1")).otherwise(pl.col("blk2")).alias("blk_u")])
if rmix.height:
    d = (rmix["blk_u"] - rmix["blk_m"]).dt.total_seconds()
    print("ranking mixed pairs: blk_m < blk_u:", (d > 0).sum(), "| ==:", (d == 0).sum(), "| >:", (d < 0).sum(), "| nulls:", d.is_null().sum(),
          "| (blk_u - blk_m) quantiles .1/.5/.9:", [d.quantile(q) for q in (.1, .5, .9)])

# predictions
preds = {}
for f in sorted(glob.glob(ROOT + "outputs/*.parquet")):
    name = os.path.basename(f).replace(".parquet", "")
    s = pl.read_parquet(f)
    if s.height != 344841:  # probe file covers only unmatched rows; keep but note
        pass
    preds[name] = s.rename({"TAXITIME_SEC_mvt": "pred"}).with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
main = preds["main"]
rtw = rtw.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(main, on="MVT_ID_mvt", how="left")
print("main.parquet rows:", main.height, "| twin rows with a main prediction:", rtw["pred"].is_not_null().sum())
print("main predictions on twin rows: by unm_fid")
print(rtw.group_by("unm_fid").agg([pl.len(), pl.col("pred").median().alias("pred_med"), pl.col("pred").mean().round(0).alias("pred_mean"),
                                   pl.col("pred").min().alias("pred_min"), pl.col("pred").max().alias("pred_max"),
                                   (pl.col("pred") >= 3000).sum().alias("n_pred>=3000"), (pl.col("pred") >= 7200).sum().alias("n_pred>=2h")]))
print("main predictions on twin rows: by ADEP x unm_fid")
print(rtw.group_by(["ADEP_mvt", "unm_fid"]).agg([pl.len(), pl.col("pred").median().alias("pred_med"), pl.col("pred").max().alias("pred_max")]).sort(["ADEP_mvt", "unm_fid"]))
# per-pair prediction difference in mixed pairs
rp_pred = rp.join(main.rename({"MVT_ID_mvt": "id1", "pred": "pred1"}).with_columns(pl.col("id1")), left_on=pl.col("id1").cast(pl.Float64), right_on="id1", how="left") \
            .join(main.rename({"MVT_ID_mvt": "id2", "pred": "pred2"}), left_on=pl.col("id2").cast(pl.Float64), right_on="id2", how="left")
print("pairs with predictions: pred difference (later - earlier MVT) by kind")
print(rp_pred.group_by("kind").agg([pl.len(), (pl.col("pred2") - pl.col("pred1")).median().alias("d_pred_med"),
                                    (pl.col("pred2") - pl.col("pred1")).abs().median().alias("|d_pred|_med"),
                                    (pl.col("pred2") - pl.col("pred1")).abs().max().alias("|d_pred|_max")]).sort("kind"))
rmp = rp_pred.filter(pl.col("kind") == "mixed").with_columns([
    pl.when(pl.col("unm1")).then(pl.col("pred2")).otherwise(pl.col("pred1")).alias("pred_m"),
    pl.when(pl.col("unm1")).then(pl.col("pred1")).otherwise(pl.col("pred2")).alias("pred_u")])
print("mixed ranking pairs: pred_m median", rmp["pred_m"].median(), "| pred_u median", rmp["pred_u"].median(),
      "| pred_m > pred_u:", (rmp["pred_m"] > rmp["pred_u"]).sum(), "of", rmp.height)
print("full list of the ranking twin rows (main prediction):")
print(rtw.select(["ADEP_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt", "pos", "k", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "bs", "ms", "unm_fid", "unm_route", "STAND_mvt", "RUNWAY_mvt", "pred"]))
# other submissions on the same rows (median by unm_fid)
print("other submissions on twin rows (median pred, by unm_fid; probe file covers unmatched rows only):")
rows = []
for name, s in preds.items():
    j = rtw.select(["MVT_ID_mvt", "unm_fid"]).join(s, on="MVT_ID_mvt", how="left")
    rows.append({"file": name, "n_pred": j["pred"].is_not_null().sum(),
                 "med_matched": j.filter(~pl.col("unm_fid"))["pred"].median(), "med_unmatched": j.filter(pl.col("unm_fid"))["pred"].median(),
                 "max": j["pred"].max()})
print(pl.DataFrame(rows))

# ---------------------------------------------------------------- F. ID overlap ranking vs training
print("\n=== F. ID overlap between ranking DEP and training DEP ===")
tid = tr.select("MVT_ID_mvt").collect()["MVT_ID_mvt"]
rid = rk.select("MVT_ID_mvt").collect()["MVT_ID_mvt"]
print("MVT_ID overlap:", rid.is_in(tid).sum())
tf = tr.select("FLIGHT_ID_mvt").drop_nulls().collect()["FLIGHT_ID_mvt"]
rf = rk.select("FLIGHT_ID_mvt").drop_nulls().collect()["FLIGHT_ID_mvt"]
print("FLIGHT_ID overlap (ranking rows whose FLIGHT_ID appears in training):", rf.is_in(tf).sum())
print("twin ranking rows whose FLIGHT_ID appears in training:", rtw["FLIGHT_ID_mvt"].drop_nulls().is_in(tf).sum())
# (ADEP, FLIGHT, SCHED) key overlap between ranking twins and training twins
print("ranking twin keys also twin keys in training:",
      rtw.select(KEY).unique().join(tw.select(KEY).unique(), on=KEY, how="inner").height)

# ---------------------------------------------------------------- G. upper-bound arithmetic
print("\n=== G. LB upper-bound arithmetic ===")
N = 344841; LB = 305.03; SSE_LB = LB ** 2 * N
n_twin_rank = rtw.height
r_twin_oof = tw_oof["se"].mean() ** 0.5
r_mt = mt["se"].mean() ** 0.5 if mt.height else float("nan")
print(f"N_sub={N}, LB(v4)={LB} -> SSE_LB = {SSE_LB:,.0f}")
def bound(label, n, e):
    sse = n * e ** 2
    new = math.sqrt(max(LB ** 2 - sse / N, 0))
    print(f"{label}: n={n}, per-row error {e:,.1f} s -> SSE {sse:,.0f} = {sse / SSE_LB * 100:.3f} % of SSE_LB; RMSE {LB} -> {new:.2f} (delta {LB - new:.2f})")
bound("(a) all twin rows, error = OOF RMSE on training twin rows", n_twin_rank, r_twin_oof)
bound("(b) matched twins of same-take-off mixed pairs, error = OOF RMSE on that class", n_mixed_same, r_mt)
bound("(c) same as (b), error = median(y_m) - median(y_u) in training", n_mixed_same, float(same_to["y_m"].median() - same_to["y_u"].median()))
bound("(d) pessimistic: all twin rows, error = training matched-twin median label", n_twin_rank, float(mixed["y_m"].median()))
print("done")

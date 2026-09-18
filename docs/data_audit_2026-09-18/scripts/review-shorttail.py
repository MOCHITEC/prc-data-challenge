"""Adversarial review of label-plausibility.md items 1-5 (short tail of the label).

Question it must answer: does the trained model ALREADY handle these rows?
No training. POLARS_MAX_THREADS=1. Lazy scans, minimal columns.

Run: POLARS_MAX_THREADS=1 .venv/bin/python docs/data_audit_2026-09-18/scripts/review-shorttail.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

ROOT = "/home/user/prc-data-challenge"
LABEL = "TAXITIME_SEC_mvt"
OOF = f"{ROOT}/experiments/oof_main-48c19bc830.parquet"
FT = f"{ROOT}/data/work/feat_train.parquet"
FR = f"{ROOT}/data/work/feat_rank.parquet"

COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
    "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", LABEL, "nm_matched", "mvt_minus_aobt3",
]


def hr(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def q(s: pl.Series, ps=(0.0, 0.25, 0.5, 0.75, 0.95)) -> list:
    return [None if s.is_empty() else round(float(s.quantile(p, "nearest") or 0.0), 1) for p in ps]


# ---------------------------------------------------------------- 0. load
hr("0. inputs")
oof = pl.scan_parquet(OOF).collect()
print("oof rows", oof.height, "unique MVT_ID", oof["MVT_ID_mvt"].n_unique(), "null oof", oof["oof"].null_count())

tr = pl.scan_parquet(FT).select(COLS).collect()
print("feat_train DEP rows", tr.height, "labelled", tr[LABEL].is_not_null().sum())

rk = pl.scan_parquet(FR).select(COLS).collect()
print("feat_rank DEP rows", rk.height)

d = tr.join(oof, on="MVT_ID_mvt", how="inner")
print("joined rows (train x oof)", d.height)
d = d.filter(pl.col(LABEL).is_not_null() & pl.col("oof").is_not_null())
print("with label+pred", d.height)
d = d.with_columns((pl.col("oof") - pl.col(LABEL)).alias("res"))
TOT_SSE = float((d["res"] ** 2).sum())
N = d.height
print("global: n", N, "RMSE", round((TOT_SSE / N) ** 0.5, 2), "bias", round(float(d["res"].mean()), 2))
print("total SSE", f"{TOT_SSE:.6g}")

# which folds cover which rows: the oof file is one prediction per row
print("oof rows not in feat_train:", oof.height - tr.join(oof, on="MVT_ID_mvt", how="semi").height)

# ---------------------------------------------------------------- 1. populations
hr("1. the five populations, re-derived independently")

POPS = {
    "P1_helipad_H": (pl.col("RUNWAY_mvt") == "H"),
    "P1b_helipad_H_LSZH": (pl.col("RUNWAY_mvt") == "H") & (pl.col("ADEP_mvt") == "LSZH"),
    "P2_EHAM_HG01": (pl.col("ADEP_mvt") == "EHAM") & (pl.col("STAND_mvt") == "HG01"),
    "P2b_A139_any": (pl.col("AIRCRAFT_TYPE_mvt") == "A139"),
    "P3_local": (pl.col("ADEP_mvt") == pl.col("ADES_mvt")),
    "P4_VFR": (pl.col("FLIGHT_RULE_mvt") == "V"),
}

rows = []
for name, pred in POPS.items():
    ntr = int(tr.select(pred.fill_null(False).sum()).item())
    ntr_lab = int(tr.filter(pl.col(LABEL).is_not_null()).select(pred.fill_null(False).sum()).item())
    nrk = int(rk.select(pred.fill_null(False).sum()).item())
    sub = tr.filter(pred.fill_null(False) & pl.col(LABEL).is_not_null())
    rows.append(dict(
        pop=name, n_train=ntr, n_train_lab=ntr_lab, n_rank=nrk,
        med_y=None if sub.is_empty() else float(sub[LABEL].median()),
        lt60=0 if sub.is_empty() else int((sub[LABEL] < 60).sum()),
        neg=0 if sub.is_empty() else int((sub[LABEL] < 0).sum()),
        unmatched=0 if sub.is_empty() else int(sub["FLIGHT_ID_mvt"].is_null().sum()),
    ))
print(pl.DataFrame(rows))

print("\nglobal training median y:", float(tr.filter(pl.col(LABEL).is_not_null())[LABEL].median()))
print("training negatives total:", int((tr[LABEL] < 0).sum()), " y<60 total:", int((tr[LABEL] < 60).sum()))
print("negatives that are RUNWAY=='H':",
      int(tr.filter((pl.col(LABEL) < 0) & (pl.col("RUNWAY_mvt") == "H")).height))

hr("1b. per-airport / per-value detail")
print("RUNWAY=='H' by ADEP (train):")
print(tr.filter(pl.col("RUNWAY_mvt") == "H").group_by("ADEP_mvt").len())
print("RUNWAY=='H' by ADEP (rank):")
print(rk.filter(pl.col("RUNWAY_mvt") == "H").group_by("ADEP_mvt").len())
print("\nlocal ADEP==ADES by ADEP (train, labelled):")
print(tr.filter((pl.col("ADEP_mvt") == pl.col("ADES_mvt")) & pl.col(LABEL).is_not_null())
      .group_by("ADEP_mvt").agg(pl.len(), pl.col(LABEL).median().alias("med"),
                                (pl.col(LABEL) < 60).sum().alias("lt60")).sort("len", descending=True))
print("local by ADEP (rank):")
print(rk.filter(pl.col("ADEP_mvt") == pl.col("ADES_mvt")).group_by("ADEP_mvt").len().sort("len", descending=True))
print("\nFLIGHT_RULE_mvt value counts train:")
print(tr.group_by("FLIGHT_RULE_mvt").agg(pl.len(), pl.col(LABEL).median().alias("med"),
                                         (pl.col(LABEL) < 60).sum().alias("lt60"),
                                         (pl.col(LABEL) < 0).sum().alias("neg")))
print("FLIGHT_RULE_mvt value counts rank:")
print(rk.group_by("FLIGHT_RULE_mvt").len())
print("\nEHAM/HG01 aircraft types (train):")
print(tr.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("STAND_mvt") == "HG01"))
      .group_by("AIRCRAFT_TYPE_mvt").len().sort("len", descending=True).head(6))
print("EHAM airport median y:", float(tr.filter((pl.col("ADEP_mvt") == "EHAM") & pl.col(LABEL).is_not_null())[LABEL].median()))
print("A139 by ADEP (train):")
print(tr.filter(pl.col("AIRCRAFT_TYPE_mvt") == "A139").group_by("ADEP_mvt").agg(
    pl.len(), pl.col(LABEL).median().alias("med")).sort("len", descending=True))
print("A139 by ADEP (rank):")
print(rk.filter(pl.col("AIRCRAFT_TYPE_mvt") == "A139").group_by("ADEP_mvt").len().sort("len", descending=True))

# ---------------------------------------------------------------- 2. THE DECISIVE TEST
hr("2. DECISIVE: the model's actual residual on each population")

def stats(sub: pl.DataFrame, label: str) -> dict:
    if sub.is_empty():
        return dict(pop=label, n=0)
    r = sub["res"].to_numpy()
    y = sub[LABEL].to_numpy().astype(float)
    p = sub["oof"].to_numpy()
    sse = float((r ** 2).sum())
    return dict(
        pop=label, n=len(r),
        med_y=round(float(np.median(y)), 1), med_pred=round(float(np.median(p)), 1),
        mean_pred=round(float(p.mean()), 1),
        rmse=round(float(np.sqrt(sse / len(r))), 1),
        bias=round(float(r.mean()), 1),
        mae=round(float(np.abs(r).mean()), 1),
        sse_share_pct=round(100.0 * sse / TOT_SSE, 4),
        rmse_if_perfect_global=round(float(np.sqrt((TOT_SSE - sse) / N)), 3),
    )

res_rows = []
for name, pred in POPS.items():
    res_rows.append(stats(d.filter(pred.fill_null(False)), name))
res_rows.append(stats(d, "ALL"))
tab = pl.DataFrame(res_rows)
with pl.Config(tbl_cols=20, tbl_width_chars=200, fmt_str_lengths=40):
    print(tab)

hr("2b. helipad rows: does the model already predict near zero?")
h = d.filter(pl.col("RUNWAY_mvt") == "H")
print("n", h.height)
print("y quantiles      ", q(h[LABEL]))
print("pred quantiles   ", q(h["oof"]))
print("residual quantiles", q(h["res"]))
print("pred < 60 s:", int((h["oof"] < 60).sum()), "/", h.height,
      " pred < 120:", int((h["oof"] < 120).sum()),
      " pred < 300:", int((h["oof"] < 300).sum()))
print("what a non-helipad-aware model would do (LSZH non-H median pred):",
      float(d.filter((pl.col("ADEP_mvt") == "LSZH") & (pl.col("RUNWAY_mvt") != "H"))["oof"].median()))
print("matched/unmatched split of helipad rows:", h.group_by("nm_matched").agg(
    pl.len(), pl.col("res").pow(2).mean().sqrt().round(1).alias("rmse"),
    pl.col("oof").median().alias("med_pred")).to_dicts())

hr("2c. residual RMSE by population, matched vs unmatched (the score decomposition)")
for name, pred in POPS.items():
    sub = d.filter(pred.fill_null(False))
    if sub.is_empty():
        continue
    for m in (1, 0):
        s = sub.filter(pl.col("nm_matched") == m)
        if s.height:
            print(f"  {name:20s} {'matched' if m else 'unmatched':9s} n={s.height:6d} "
                  f"rmse={float(np.sqrt((s['res'].to_numpy()**2).mean())):9.1f} "
                  f"bias={float(s['res'].mean()):8.1f} sse_share={100*float((s['res'].to_numpy()**2).sum())/TOT_SSE:.4f}%")

# ---------------------------------------------------------------- 3. overlap / dedup
hr("3. overlap and deduplicated totals")
FOUR = ["P1_helipad_H", "P2_EHAM_HG01", "P3_local", "P4_VFR"]
def flags(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([POPS[k].fill_null(False).alias(k) for k in FOUR])

trf, rkf = flags(tr), flags(rk)
print("pairwise overlap counts (training, all rows):")
for i, a in enumerate(FOUR):
    for b in FOUR[i + 1:]:
        print(f"  {a} & {b}: train {int(trf.filter(pl.col(a) & pl.col(b)).height)}, "
              f"rank {int(rkf.filter(pl.col(a) & pl.col(b)).height)}")
for nm, df in (("train", trf), ("rank", rkf)):
    n_any = int(df.filter(pl.any_horizontal([pl.col(c) for c in FOUR])).height)
    naive = sum(int(df.select(pl.col(c).sum()).item()) for c in FOUR)
    print(f"{nm}: naive sum of the four = {naive}, deduplicated union = {n_any}")
    print("   membership-count histogram:",
          df.with_columns(pl.sum_horizontal([pl.col(c).cast(pl.Int8) for c in FOUR]).alias("k"))
            .group_by("k").len().sort("k").to_dicts())

hr("3b. union residual (the honest joint effect)")
u = flags(d).filter(pl.any_horizontal([pl.col(c) for c in FOUR]))
print(stats(u, "UNION_of_four"))
u2 = flags(d).filter(pl.any_horizontal([pl.col(c) for c in FOUR]) | (pl.col("AIRCRAFT_TYPE_mvt") == "A139"))
print(stats(u2, "UNION_plus_A139"))

# ---------------------------------------------------------------- 4. identifiability in 2026
hr("4. prediction-time identifiability on the ranking set")
for c in ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "FLIGHT_RULE_mvt", "AIRCRAFT_TYPE_mvt"]:
    nn = int(rk[c].is_not_null().sum())
    print(f"  {c:20s} non-null {nn}/{rk.height} = {nn/rk.height:.6f}   distinct {rk[c].n_unique()}")
print("\n'H' present in ranking RUNWAY_mvt:", "H" in rk["RUNWAY_mvt"].unique().to_list())
print("'HG01' present in ranking STAND_mvt:", "HG01" in rk["STAND_mvt"].unique().to_list())
print("'V' present in ranking FLIGHT_RULE_mvt:", "V" in rk["FLIGHT_RULE_mvt"].unique().to_list())
print("'A139' present in ranking AIRCRAFT_TYPE_mvt:", "A139" in rk["AIRCRAFT_TYPE_mvt"].unique().to_list())
print("\nIS FLIGHT_RULE_mvt A MODEL FEATURE? -> see features.CATEGORICALS / numeric_feature_columns")
from prc.features import CATEGORICALS, numeric_feature_columns  # noqa: E402
schema_frame = pl.scan_parquet(FT).head(1).collect()
num = numeric_feature_columns(schema_frame)
print("  CATEGORICALS:", CATEGORICALS)
print("  FLIGHT_RULE_mvt in CATEGORICALS:", "FLIGHT_RULE_mvt" in CATEGORICALS)
print("  FLIGHT_RULE_mvt in numeric feature columns:", "FLIGHT_RULE_mvt" in num)
print("  FLIGHT_RULE_flt in either:", "FLIGHT_RULE_flt" in CATEGORICALS or "FLIGHT_RULE_flt" in num)
print("  ADEP_mvt / ADES_mvt / RUNWAY_mvt / STAND_mvt / AIRCRAFT_TYPE_mvt all categorical:",
      all(c in CATEGORICALS for c in ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]))

hr("4b. cold start: do the ranking rows' identifying groups exist in training?")
for name, keys in (("(ADEP,RUNWAY)", ["ADEP_mvt", "RUNWAY_mvt"]),
                   ("(ADEP,STAND,RUNWAY)", ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])):
    seen = tr.select(keys).unique()
    for pname, pred in (("helipad", POPS["P1_helipad_H"]), ("EHAM_HG01", POPS["P2_EHAM_HG01"])):
        sub = rk.filter(pred.fill_null(False)).select(keys).unique()
        got = sub.join(seen, on=keys, how="semi").height
        print(f"  {name} {pname}: {got}/{sub.height} ranking groups present in training")

# ---------------------------------------------------------------- 5. effect-size ceiling
hr("5. effect-size ceiling, in leaderboard arithmetic")
# team decomposition: 344,841 ranking rows, 339,551 matched at ~218 RMSE, 5,290 unmatched
N_ALL, N_M, RMSE_M, N_U = 344841, 339551, 218.0, 5290
# infer the unmatched RMSE that reproduces the last leaderboard number, and also use the OOF one
m_oof = d.filter(pl.col("nm_matched") == 1)
u_oof = d.filter(pl.col("nm_matched") == 0)
rm = float(np.sqrt((m_oof["res"].to_numpy() ** 2).mean()))
ru = float(np.sqrt((u_oof["res"].to_numpy() ** 2).mean()))
print(f"OOF matched   n={m_oof.height} rmse={rm:.1f}")
print(f"OOF unmatched n={u_oof.height} rmse={ru:.1f}")
tot = (N_M * RMSE_M ** 2 + N_U * ru ** 2) / N_ALL
print(f"implied leaderboard RMSE with matched=218, unmatched={ru:.1f}: {tot**0.5:.2f}")

# rank-side population sizes, split matched/unmatched, and the OOF per-row SSE they carry
print("\nceiling per population: remove its OOF sse contribution, scaled to ranking prevalence")
for name, pred in POPS.items():
    sub = d.filter(pred.fill_null(False))
    if sub.is_empty():
        continue
    rk_sub = rk.filter(pred.fill_null(False))
    out = {}
    for m, n_lb in ((1, N_M), (0, N_U)):
        s = sub.filter(pl.col("nm_matched") == m)
        r = rk_sub.filter(pl.col("nm_matched") == m)
        if not s.height or not r.height:
            out[m] = (r.height, 0.0)
            continue
        mse_pop = float((s["res"].to_numpy() ** 2).mean())
        out[m] = (r.height, mse_pop)
    # leaderboard SSE with and without perfect prediction of these rows
    sse_base = N_M * RMSE_M ** 2 + N_U * ru ** 2
    # replace the population's own rows: matched rows carry 218^2 on average is wrong; use their own mse
    n_m_pop, mse_m_pop = out.get(1, (0, 0.0))
    n_u_pop, mse_u_pop = out.get(0, (0, 0.0))
    sse_removed = n_m_pop * mse_m_pop + n_u_pop * mse_u_pop
    new = (sse_base - sse_removed) / N_ALL
    print(f"  {name:20s} rank n={n_m_pop + n_u_pop:5d} (m {n_m_pop}, u {n_u_pop})  "
          f"own MSE m={mse_m_pop:12.1f} u={mse_u_pop:12.1f}  "
          f"LB {tot**0.5:.3f} -> {new**0.5:.3f}  gain {tot**0.5 - new**0.5:+.4f} s")

# union
uf = flags(rk).filter(pl.any_horizontal([pl.col(c) for c in FOUR]))
ud = flags(d).filter(pl.any_horizontal([pl.col(c) for c in FOUR]))
sse_base = N_M * RMSE_M ** 2 + N_U * ru ** 2
srem = 0.0
det = []
for m in (1, 0):
    s = ud.filter(pl.col("nm_matched") == m)
    r = uf.filter(pl.col("nm_matched") == m)
    if s.height and r.height:
        mse = float((s["res"].to_numpy() ** 2).mean())
        srem += r.height * mse
        det.append((m, r.height, round(mse, 1)))
new = (sse_base - srem) / N_ALL
print(f"\n  UNION of four: rank n={uf.height} detail(matched_flag,n,own_mse)={det}")
print(f"  LB {tot**0.5:.3f} -> {new**0.5:.3f}   ceiling gain {tot**0.5 - new**0.5:+.4f} s")

# ---------------------------------------------------------------- 6. the 527 rows
hr("6. the 527 'too short for their stand x runway pair' rows")
lab = tr.filter(pl.col(LABEL).is_not_null())
trip = (lab.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
        .agg(pl.len().alias("trip_n"), pl.col(LABEL).median().alias("trip_med")))
big = trip.filter((pl.col("trip_n") >= 20) & pl.col("STAND_mvt").is_not_null())
print("triples n>=20 with non-null stand:", big.height)
short = lab.filter(pl.col(LABEL) < 60)
print("y<60:", short.height)
sj = short.join(big, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="inner")
print("  on a triple with n>=20:", sj.height)
print("  ... triple median >= 300:", int((sj["trip_med"] >= 300).sum()))
print("  ... triple median >= 600:", int((sj["trip_med"] >= 600).sum()))

# ARR taxi-in p05 per (airport, stand) from the raw training files (ARR rows are labelled)
arr = (pl.scan_parquet(f"{ROOT}/data/raw/training_*.parquet")
       .filter(pl.col("PHASE_mvt") == "ARR")
       .select(["ADES_mvt", "STAND_mvt", LABEL])
       .filter(pl.col(LABEL).is_not_null())
       .group_by(["ADES_mvt", "STAND_mvt"])
       .agg(pl.len().alias("arr_n"), pl.col(LABEL).quantile(0.05, "nearest").alias("arr_p05"),
            pl.col(LABEL).median().alias("arr_med"))
       .collect())
print("(airport,stand) ARR groups:", arr.height)
sj2 = sj.join(arr.rename({"ADES_mvt": "ADEP_mvt"}), on=["ADEP_mvt", "STAND_mvt"], how="left")
hard = sj2.filter((pl.col("trip_med") >= 300) & (pl.col("arr_p05") >= 120))
print("HARD set (y<60 & trip_med>=300 & arr_p05>=120):", hard.height, "   audit said 527")
print("  also with arr_p05>=240:", int(sj2.filter((pl.col("trip_med") >= 300) & (pl.col("arr_p05") >= 240)).height))
print("  short rows with ARR stats at all:", int(sj2["arr_p05"].is_not_null().sum()))

hd = (hard.join(oof, on="MVT_ID_mvt", how="inner").filter(pl.col("oof").is_not_null())
      .with_columns((pl.col("oof") - pl.col(LABEL)).alias("res")))
print("HARD rows with an OOF prediction:", hd.height, "of", hard.height)
print("\nmodel behaviour on the HARD set:", stats(hd, "HARD_527"))
print("  y quantiles   ", q(hd[LABEL]))
print("  pred quantiles", q(hd["oof"]))
print("  triple median quantiles", q(hd["trip_med"]))

hr("6b. is the test circular? compare it to the plain residual ranking")
# the HARD test selects rows by |y - group median| being large and negative. A residual does the same thing
# with a better conditional. Rank all labelled+predicted rows by residual and see how many HARD rows land in
# the worst tail.
dd = d.with_columns(pl.col("res").abs().alias("ares"))
k = hd.height
top = dd.sort("ares", descending=True).head(50000).select("MVT_ID_mvt")
hard_ids = set(hd["MVT_ID_mvt"].to_list())
k = hd.height
hard_ids = set(hd["MVT_ID_mvt"].to_list())
for kk in (k, 1000, 5000, 20000, 50000):
    t = set(dd.sort("ares", descending=True).head(kk)["MVT_ID_mvt"].to_list())
    print(f"  HARD rows inside the {kk} largest |residual| rows: {len(hard_ids & t)}/{k}")
# the converse: how well does the group-median rule separate, i.e. what does the model's own residual
# distribution look like on HARD vs on all rows on the same triples
print("\n  |residual| quantiles, HARD set:", q(hd["res"].abs(), (0.25, 0.5, 0.75, 0.95, 1.0)))
print("  |residual| quantiles, all rows :", q(dd["ares"], (0.25, 0.5, 0.75, 0.95, 1.0)))
# does a model that already knows the triple (via ref_med encoding) predict the triple median on these rows?
print("\n  mean(pred) on HARD:", round(float(hd['oof'].mean()), 1),
      " mean(trip_med):", round(float(hd['trip_med'].mean()), 1),
      " mean(y):", round(float(hd[LABEL].mean()), 1))
print("  correlation pred vs trip_med on HARD:",
      round(float(np.corrcoef(hd['oof'].to_numpy().astype(float), hd['trip_med'].to_numpy().astype(float))[0, 1]), 3))

hr("6c. can the HARD population be identified at prediction time at all?")
# rate at which a (ADEP,STAND,RUNWAY) triple produces a HARD row
rate = (hard.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]).len().rename({"len": "hard_n"})
        .join(big, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
        .with_columns((pl.col("hard_n") / pl.col("trip_n")).alias("rate")).sort("rate", descending=True))
print("triples producing >=1 HARD row:", rate.height)
print("their HARD rate quantiles:", q(rate["rate"], (0.5, 0.9, 0.99, 1.0)))
with pl.Config(tbl_rows=12):
    print(rate.head(10))
print("\nthe best-case precision of any triple-level rule: max rate =",
      round(float(rate['rate'].max()), 4))

# ------------------------------------------------------- 7. robustness of the residuals
hr("7. are the population RMSEs outlier-driven?")
for name, pred in POPS.items():
    sub = d.filter(pred.fill_null(False))
    if sub.is_empty():
        continue
    r = np.sort(np.abs(sub["res"].to_numpy()))[::-1]
    n = len(r)
    full = float(np.sqrt((r ** 2).mean()))
    d1 = float(np.sqrt((r[1:] ** 2).mean())) if n > 1 else 0.0
    d5 = float(np.sqrt((r[5:] ** 2).mean())) if n > 5 else 0.0
    print(f"  {name:20s} n={n:5d} rmse={full:8.1f}  drop-1={d1:8.1f}  drop-5={d5:8.1f}  "
          f"median|res|={float(np.median(r)):7.1f}  max|res|={r[0]:9.1f}")

hr("7b. what the model is HANDED: the (ADEP,STAND,RUNWAY) label encoding on these rows")
ref = (lab.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
       .agg(pl.len().alias("ref_n"), pl.col(LABEL).median().alias("ref_med"),
            pl.col(LABEL).quantile(0.10, "nearest").alias("ref_p10"))
       .filter(pl.col("ref_n") >= 30))
print("triples with n>=30 (the reference_min_count the trainer uses):", ref.height)
print("helipad triples the model is handed:")
print(ref.filter(pl.col("RUNWAY_mvt") == "H"))
print("EHAM HG01 triples the model is handed:")
print(ref.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("STAND_mvt") == "HG01")).head(8))
rk_ref = rk.join(ref, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
for name, pred in (("helipad", POPS["P1_helipad_H"]), ("EHAM_HG01", POPS["P2_EHAM_HG01"]),
                   ("local", POPS["P3_local"]), ("VFR", POPS["P4_VFR"])):
    s_ = rk_ref.filter(pred.fill_null(False))
    hit = int(s_["ref_med"].is_not_null().sum())
    print(f"  ranking {name:10s}: {hit}/{s_.height} rows get a ref_med; "
          f"median ref_med where present = {None if not hit else float(s_['ref_med'].median())}")

hr("7c. ranking matched/unmatched split (the team decomposition, verified)")
print(rk.group_by("nm_matched").len().sort("nm_matched"))
print("rank rows with PROXY (mvt_minus_aobt3) null:", int(rk["mvt_minus_aobt3"].is_null().sum()))
for name, pred in POPS.items():
    s_ = rk.filter(pred.fill_null(False))
    print(f"  {name:20s} rank n={s_.height:5d} proxy-null (model B) = {int(s_['mvt_minus_aobt3'].is_null().sum())}")

hr("7d. realistic (not perfect) ceiling: replace the prediction by the population training median")
for name, pred in POPS.items():
    sub = d.filter(pred.fill_null(False))
    if sub.is_empty():
        continue
    med = float(tr.filter(pred.fill_null(False))[LABEL].median())
    y = sub[LABEL].to_numpy().astype(float)
    cur = float(np.sqrt(((sub["oof"].to_numpy() - y) ** 2).mean()))
    alt = float(np.sqrt(((med - y) ** 2).mean()))
    print(f"  {name:20s} train median={med:7.1f}  model rmse={cur:8.1f}  median-rule rmse={alt:8.1f}  "
          f"{'BETTER' if alt < cur else 'WORSE'}")

hr("DONE")

"""Final fit of the learned ADS-B residual model on all 62 labelled days, applied to the 2026 scored rows on top of
`outputs/hybrid_H2.parquet`. Writes `outputs/hybrid_H2_adsb_learned.parquet`. Nothing is submitted.

Pre-registration: `docs/competitor_research_2026-09-21/results/adsb-learned-residual.md` section 0 (and 0a:
c = 0 at LTFM). Run `adsb_learn_cv.py` first: it writes `$SCRATCH/learn/cv.json`, and this script reads the
number of rounds from there.

    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_apply.py
"""

from __future__ import annotations

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_learn_cv as CV
import adsb_learn_features as L

ROOT = L.s1.REPO_DIR
# `--base plainy` (see adsb_learn_cv.py): base outputs/hybrid_H2_plainy.parquet, CV json cv_plainy.json
BASE_TAG = {"h2": "H2", "plainy": "H2_plainy", "cb": "plainy_cb", "l2": "blend_l2",
            **{v: f"blend_{v}" for v in CV.L2_VARIANTS}}[CV.BASE_NAME]
PREV_OUT = {"plainy": "outputs/hybrid_H2_adsb_learned.parquet",
            "cb": "outputs/hybrid_H2_plainy_adsb_learned.parquet",
            "l2": "outputs/hybrid_plainy_cb_adsb_learned.parquet",
            **{v: "outputs/hybrid_blend_l2_adsb_learned.parquet" for v in CV.L2_VARIANTS}}.get(CV.BASE_NAME)
OUT_FILE = ROOT + f"outputs/hybrid_{BASE_TAG}_adsb_learned.parquet"
BASE_FILE = ROOT + f"outputs/hybrid_{BASE_TAG}.parquet"
V20_FILE = ROOT + "outputs/v20_hybridH2_adsb.parquet"
NO_CORR = ["LTFM"]  # section 0a


def q(x, qs=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    return {p: round(float(np.quantile(x, p)), 1) for p in qs} if len(x) else {}


def main() -> None:
    L.mem_gate()
    cvj = json.load(open(os.path.join(L.OUT, f"cv{CV.SFX}.json")))
    rounds = cvj["final_rounds"]
    print(f"base = {BASE_FILE}; output = {OUT_FILE}")
    CV.head(f"S0  final fit: all 62 days, {rounds} rounds (median fold best iteration {cvj['best_iters']})")
    D = CV.load_labelled()
    F = pl.read_parquet(os.path.join(L.OUT, "feat_2025.parquet")).drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="inner")
    X, feats, cats = L.design(D, "b")
    assert feats == cvj["features"]
    r = np.clip(D["y"].to_numpy() - D["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    dtr = lgb.Dataset(X.select(feats).to_numpy().astype(np.float64), r, feature_name=feats, categorical_feature=cats)
    model = lgb.train(CV.PARAMS, dtr, num_boost_round=rounds)
    model.save_model(os.path.join(L.OUT, f"final_model{CV.SFX}.txt"))
    print(f"trained on {D.height} ADS-B rows; model saved to {L.OUT}/final_model{CV.SFX}.txt")
    c_in = np.clip(model.predict(X.select(feats).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)
    print("in-sample correction quantiles (reference only):", q(c_in))

    CV.head("S1  apply to 2026 scored rows")
    tpl = pl.read_parquet(ROOT + "data/raw/submitting.parquet")
    base = pl.read_parquet(BASE_FILE)
    assert base.schema == tpl.schema and base.height == tpl.height and (base["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
    rk = pl.scan_parquet(ROOT + "data/raw/ranking.parquet").select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt").collect()
    F26 = pl.read_parquet(os.path.join(L.OUT, "feat_2026.parquet"))
    assert not (set(F26.columns) & L.LABEL_COLS)
    S = (base.rename({"TAXITIME_SEC_mvt": "base"}).join(rk, on="MVT_ID_mvt", how="left", maintain_order="left"))
    assert S["ADEP_mvt"].null_count() == 0
    G = F26.drop("ADEP_mvt").join(S.select("MVT_ID_mvt", "ADEP_mvt", "base", "MVT_TIME_UTC_mvt"), on="MVT_ID_mvt",
                                  how="inner")
    print(f"2026 ADS-B feature rows {F26.height}; matched to template {G.height}; "
          f"max |t_mvt - MVT| {(G['t_mvt'] - G['MVT_TIME_UTC_mvt'].dt.epoch('ms') / 1000).abs().max()}")
    G = G.with_columns(pl.col("base").cast(pl.Float64).alias("b"))
    X26, feats26, _ = L.design(G, "b")
    assert feats26 == feats
    c26 = np.clip(model.predict(X26.select(feats).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)
    G = G.with_columns(pl.Series("c", c26)).with_columns(
        pl.when(pl.col("ADEP_mvt").is_in(NO_CORR)).then(0.0).otherwise(pl.col("c")).alias("c"))
    G = G.with_columns((pl.col("b") + pl.col("c")).round(0).alias("new_f"))
    neg = G.filter(pl.col("new_f") < 0)
    print(f"negative predictions reverted to base: {neg.height}")
    G = G.with_columns(pl.when(pl.col("new_f") < 0).then(pl.col("base")).otherwise(pl.col("new_f").cast(pl.Int32))
                       .alias("new"))
    cand = (base.join(G.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left", maintain_order="left")
            .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt"))
            .select(tpl.columns))

    CV.head("S2  validation")
    checks = {
        "rows == 344,841": cand.height == tpl.height == 344841,
        "schema == template": cand.schema == tpl.schema,
        "ids and order == template": bool((cand["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()),
        "no nulls": cand.null_count().sum_horizontal().item() == 0,
        "no negatives": int((cand["TAXITIME_SEC_mvt"] < 0).sum()) == 0,
        "rows without ADS-B == base": bool(
            (cand.filter(~pl.col("MVT_ID_mvt").is_in(G["MVT_ID_mvt"].implode()))["TAXITIME_SEC_mvt"]
             == base.filter(~pl.col("MVT_ID_mvt").is_in(G["MVT_ID_mvt"].implode()))["TAXITIME_SEC_mvt"]).all()),
        "LTFM unchanged": bool((cand.join(S.select("MVT_ID_mvt", "ADEP_mvt", "base"), on="MVT_ID_mvt")
                                .filter(pl.col("ADEP_mvt") == "LTFM")
                                .select((pl.col("TAXITIME_SEC_mvt") == pl.col("base")).all()).item())),
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    assert all(checks.values())
    cand.write_parquet(OUT_FILE)
    back = pl.read_parquet(OUT_FILE)
    assert back.equals(cand)
    print(f"wrote {OUT_FILE}: {back.shape}, schema {back.schema}")

    CV.head("S3  what changed")
    M = cand.rename({"TAXITIME_SEC_mvt": "new"}).join(S, on="MVT_ID_mvt").with_columns(
        (pl.col("new") - pl.col("base")).alias("delta"))
    ch = M.filter(pl.col("delta") != 0)
    print(f"rows with an ADS-B feature row {G.height}; changed vs base ({os.path.basename(BASE_FILE)}) {ch.height} "
          f"({100 * ch.height / M.height:.2f} % of scored rows); max |delta| {ch['delta'].abs().max()}; "
          f"RMS delta over changed {float(np.sqrt((ch['delta'].cast(pl.Float64) ** 2).mean())):.1f} s")
    print("signed delta quantiles (changed rows):", q(ch["delta"].to_numpy()))
    print("|delta| quantiles (changed rows):", q(np.abs(ch["delta"].to_numpy()), (0.5, 0.9, 0.99)))
    print(f"mean prediction: base {M['base'].mean():.2f}, new {M['new'].mean():.2f}")
    print("per airport:")
    print(M.group_by("ADEP_mvt").agg(
        pl.len().alias("n"), (pl.col("delta") != 0).sum().alias("changed"),
        pl.col("delta").mean().alias("mean_delta"), pl.col("delta").median().alias("median_delta"),
        pl.col("delta").abs().filter(pl.col("delta") != 0).median().alias("med_abs_changed"),
        pl.col("delta").abs().quantile(0.99).alias("p99_abs"), pl.col("delta").abs().max().alias("max_abs"))
        .sort("ADEP_mvt"))
    cvr = pl.read_parquet(os.path.join(L.OUT, f"cv_rows{CV.SFX}.parquet")).filter(pl.col("has"))
    print("per airport correction quantiles, 2025 CV vs 2026 production (p10 / p50 / p90):")
    for a in sorted(set(G["ADEP_mvt"].to_list())):
        x25 = cvr.filter(pl.col("ADEP_mvt") == a)["c_learned"].to_numpy()
        x26 = G.filter(pl.col("ADEP_mvt") == a)["c"].to_numpy()
        f = lambda x: "/".join(f"{np.quantile(x, p):.0f}" for p in (0.1, 0.5, 0.9)) if len(x) else "-"  # noqa: E731
        print(f"  {a}: 2025 n {len(x25):6d} {f(x25):>14s} | 2026 n {len(x26):6d} {f(x26):>14s}")
    v21 = ROOT + PREV_OUT if PREV_OUT else ""
    if PREV_OUT and os.path.exists(v21):
        W = M.join(pl.read_parquet(v21).rename({"TAXITIME_SEC_mvt": "v21"}), on="MVT_ID_mvt")
        dv = (W["new"] - W["v21"]).cast(pl.Float64)
        print(f"vs {v21} (previous base + learned): rows differing {int((dv != 0).sum())}; RMS diff {float(np.sqrt((dv ** 2).mean())):.1f} s; "
              f"max |diff| {int(dv.abs().max())}")
    if CV.BASE_NAME == "h2" and os.path.exists(V20_FILE):
        v20 = pl.read_parquet(V20_FILE).rename({"TAXITIME_SEC_mvt": "v20"})
        W = M.join(v20, on="MVT_ID_mvt").with_columns((pl.col("v20") - pl.col("base")).alias("d_rule"))
        R = W.filter(pl.col("d_rule") != 0)
        print(f"vs v20 (H2 + rule): rows the rule changed {R.height}; on them corr(delta_learned, delta_rule) "
              f"{np.corrcoef(R['delta'].to_numpy(), R['d_rule'].to_numpy())[0, 1]:.3f}; median delta_learned "
              f"{R['delta'].median():.0f} vs median delta_rule {R['d_rule'].median():.0f}; "
              f"RMS(new - v20) over all rows {float(np.sqrt(((W['new'] - W['v20']).cast(pl.Float64) ** 2).mean())):.1f} s")
    json.dump({"rounds": rounds, "n_train": D.height, "n_2026_adsb": G.height, "changed": ch.height,
               "negatives_reverted": neg.height, "checks": checks},
              open(os.path.join(L.OUT, f"apply{CV.SFX}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

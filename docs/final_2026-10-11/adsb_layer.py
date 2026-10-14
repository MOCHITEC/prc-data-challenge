"""Learned ADS-B residual layer of submission v47, rebuilt as one clean step for the FINAL ranking set (Jan, Feb, Jun,
Jul 2026). See adsb_out/README.md for the provenance of every rule.

Subcommands (run from anywhere; env SCRATCH is forced to the shared ADS-B scratch dir):

  fit                      refit the residual models that were never saved (12:v2+apron, 7:v2+apron, 4:v2) on the
                           2025 training rows exactly as production did; writes adsb_out/models/*.txt
                           (--check-7v2 also refits 7:v2 and compares it with the saved production model)
  feat --days D,D,...      per-day learn features for new days (callsign join, positional join, positional learn
                           features, long-stop dwell) -> adsb_out/feat/, adsb_out/pos/   (read-only on SCRATCH)
  apply --base B --out O   O = B + c on every row the layer covers, B elsewhere; also writes O's sidecar
                           <O stem>_detail.parquet (MVT_ID_mvt, ADEP_mvt, month, comp, b, c, new)

Components of c (b = the pre-residual base value of the row):
  G12A  callsign ADS-B row, ground fix, airport not LTFM/LIRF/LFPG : 12-month V2 residual with the apron category
  G7A   same at LFPG                                                : 7-month V2 residual with the apron category
  GL4   same at LIRF                                                : 4-month V2 residual (LIRF excluded from later swaps)
  AIR   callsign ADS-B row without a ground fix, not LTFM           : Jan+Jul residual (original params), the v35 layer
  POS   positional-join row, not LIRF/LTFM, not a RULE row          : 7-month V2 residual
  RULE  unmatched positional-join row with a push signature          : b + w (a - b), a = ADS-B taxi, w = 1 or 0.75
  LTFM: c = 0 for every learned component.
c is clipped to +-1800 s; a negative new value reverts to b; new is rounded to an integer.

Data: adsb.lol, ODbL 1.0 ("Contains data from adsb.lol, made available under the ODbL 1.0").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

SCR = "/tmp/claude-0/-home-user-prc-data-challenge/7ae154e2-3baf-5cb4-beae-354c18126ab1/scratchpad/adsb"
os.environ["SCRATCH"] = SCR
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
FINAL = "/home/user/prc-final"
WORK = "/home/user/prc-work"          # older repo copy that still holds the 2025 OOF files and feat_train.parquet
HERE = os.path.join(FINAL, "adsb_out")
MODELS = os.environ.get("ADSB_MODELS") or os.path.join(HERE, "models")  # ADSB_MODELS: alternative model dir (seed runs)
FEAT = os.path.join(HERE, "feat")
POS = os.path.join(HERE, "pos")
LEARN = os.path.join(SCR, "learn")
RES = os.path.join(SCR, "agents", "rescat")
RANK = os.path.join(FINAL, "data/raw/ranking.parquet")

ARGV = list(sys.argv)
sys.path[:0] = [os.path.join(FINAL, "scripts"), os.path.join(FINAL, "analysis")]

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import lightgbm as lgb  # noqa: E402

import adsb_stage1_feasibility as s1  # noqa: E402
import adsb_learn_features as L  # noqa: E402

assert L.SCR == SCR and L.OUT == LEARN
sys.argv = [ARGV[0], "--base", "cb"]
import adsb_learn_cv as CV  # noqa: E402  (production residual: base cb, PARAMS, clips)
sys.argv = ARGV

CLIP_C = CV.CLIP_C
V2 = dict(num_leaves=255, min_data_in_leaf=300, feature_fraction=0.7)
M4 = ["2025-01", "2025-07", "2025-11", "2025-12"]
M7 = M4 + ["2025-08", "2025-09", "2025-10"]
# rounds exactly as production: 12:v2 cv_months12_v2.json C, 7:v2+apron rescat eval A3, 4:v2 cv_months_v2.json V2,
# 7:v2 cv_months7_v2.json C
SPEC = {"m12a": (None, True, ("cv_months12_v2.json", "C")),
        "m7a": (M7, True, (os.path.join(RES, "eval.json"), "A3")),
        "m4v2": (M4, False, ("cv_months_v2.json", "V2")),
        "m7v2_check": (M7, False, ("cv_months7_v2.json", "C"))}
SAVED = {"mA": os.path.join(LEARN, "final_model_months_A.txt"),     # Jan+Jul, PARAMS, 386 rounds (cb base)
         "m7v2": os.path.join(LEARN, "final_model_7v2.txt")}       # 7 months, V2, 505 rounds (posjoin rows)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


# ------------------------------------------------------------------------------------------------ apron codes
def apron_vocab() -> list[str]:
    return json.load(open(os.path.join(RES, "vocab.json")))["apron"]


def apron_expr():
    """Same code as local_adsb_rescat.raw_codes: ADEP|<leading letters (<= 2)> or ADEP|<first digit>d<n digits>."""
    s = pl.col("STAND_mvt").str.to_uppercase().str.strip_chars()
    lead_alpha = s.str.extract(r"^([A-Z]+)", 1).str.slice(0, 2)
    lead_dig = s.str.extract(r"^([0-9]+)", 1)
    code = (pl.when(s.is_null() | (s == "")).then(pl.lit("null"))
            .when(lead_alpha.is_not_null()).then(lead_alpha)
            .when(lead_dig.is_not_null()).then(lead_dig.str.slice(0, 1) + pl.lit("d") + lead_dig.str.len_chars().cast(pl.String))
            .otherwise(pl.lit("misc")))
    return (pl.col("ADEP_mvt") + pl.lit("|") + code).alias("apron")


def add_apron(D: pl.DataFrame, codes: pl.DataFrame) -> pl.DataFrame:
    kept = apron_vocab()
    D = D.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64).alias("_id")).join(
        codes.select(pl.col("MVT_ID_mvt").cast(pl.Float64).alias("_id"), "apron"), on="_id", how="left").drop("_id")
    return D.with_columns(pl.col("apron").replace_strict(kept, list(range(len(kept))), default=len(kept))
                          .cast(pl.Int32).alias("cat_apron"))


# ------------------------------------------------------------------------------------------------ fit
def cmd_fit(check7: bool) -> None:
    os.makedirs(MODELS, exist_ok=True)
    import local_adsb_learn_months as M  # noqa: F401  (relative paths: needs cwd = WORK during loading)
    import local_adsb_apply_swap as W
    CV.ROOT = WORK + "/"
    cwd = os.getcwd()
    os.chdir(WORK)
    try:
        D = W.data()
    finally:
        os.chdir(cwd)
    log(f"training rows {D.height}, rss {rss_mb():.0f} MB")
    codes = pl.read_parquet(os.path.join(RES, "codes.parquet"), columns=["MVT_ID_mvt", "apron"])
    D = add_apron(D, codes)
    del codes
    ym = D["day"].str.slice(0, 7).to_numpy()
    print("rows per month:", dict(zip(*np.unique(ym, return_counts=True))))
    r = np.clip(D["y"].to_numpy() - D["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    X, feats, cats = L.design(D, "b")
    allf = feats + ["cat_apron"]
    del D
    Xn = X.select([pl.col(c).cast(pl.Float32) for c in allf]).to_numpy()
    del X
    log(f"matrix {Xn.shape} float32, rss {rss_mb():.0f} MB")
    P = {**CV.PARAMS, **V2}
    if os.environ.get("ADSB_SEED"):
        _s = int(os.environ["ADSB_SEED"])
        P = {**P, "seed": _s, "bagging_seed": _s, "feature_fraction_seed": _s}
        log(f"seed override {_s}")
    meta = {"features": feats, "features_apron": allf, "cats": cats, "params": P, "models": {}}
    todo = ["m12a", "m7a", "m4v2"] + (["m7v2_check"] if check7 else [])
    for tag in todo:
        months, apron, (jf, key) = SPEC[tag]
        jp = jf if os.path.isabs(jf) else os.path.join(LEARN, jf)
        n = int(json.load(open(jp))["final_rounds"][key])
        fn = allf if apron else feats
        idx = [allf.index(f) for f in fn]
        sel = np.ones(len(r), bool) if months is None else np.isin(ym, months)
        dst = os.path.join(MODELS, f"{tag}.txt")
        if os.path.exists(dst):
            log(f"{tag}: exists, skipped")
            continue
        t0 = time.time()
        ds = lgb.Dataset(Xn[np.ix_(sel, idx)], r[sel], feature_name=fn,
                         categorical_feature=[c for c in cats + ["cat_apron"] if c in fn], free_raw_data=True)
        m = lgb.train(P, ds, num_boost_round=n)
        m.save_model(dst)
        del ds
        meta["models"][tag] = {"rows": int(sel.sum()), "rounds": n, "apron": apron, "months": months or "all 2025"}
        log(f"{tag}: {int(sel.sum())} rows, {n} rounds, {time.time() - t0:.0f} s, rss {rss_mb():.0f} MB")
    old = json.load(open(os.path.join(MODELS, "meta.json"))) if os.path.exists(os.path.join(MODELS, "meta.json")) else {}
    old_models = old.get("models", {})
    old_models.update(meta["models"])
    meta["models"] = old_models
    json.dump(meta, open(os.path.join(MODELS, "meta.json"), "w"), indent=1)
    if check7:
        a = lgb.Booster(model_file=SAVED["m7v2"]).predict(Xn[:200000, :len(feats)].astype(np.float64))
        b = lgb.Booster(model_file=os.path.join(MODELS, "m7v2_check.txt")).predict(Xn[:200000, :len(feats)].astype(np.float64))
        print(f"7:v2 refit vs saved production model on 200k training rows: corr {np.corrcoef(a, b)[0, 1]:.5f}, "
              f"RMS diff {np.sqrt(np.mean((a - b) ** 2)):.2f} s, RMS c {np.sqrt(np.mean(a ** 2)):.1f} s, max |diff| {np.abs(a - b).max():.1f}")


# ------------------------------------------------------------------------------------------------ feat
def posjoin_learnfeat(day: str, P: pl.DataFrame) -> pl.DataFrame | None:
    """local_posjoin_learnfeat.py for one day, temp files under adsb_out/pos/tmp_scr instead of $SCRATCH/pos."""
    FILL = ["ground_fix", "t_first", "t_dep1", "t_move", "t_move_lo", "t_to", "n_ground", "stand_dur", "dep1_gs",
            "gap_before_dep1", "opens_with_landing", "stand_rule", "stand_nfix", "push", "post_dwell"]
    tag = L.tag_of(day)
    P = P.filter(pl.col("icao_pos").is_not_null())
    if P.height == 0:
        return None
    tmp = os.path.join(POS, "tmp_scr")
    os.makedirs(tmp, exist_ok=True)
    J = pl.read_parquet(os.path.join(SCR, f"{tag}_joined.parquet"))
    sub = P.select(["MVT_ID_mvt", pl.col("icao_pos").alias("icao")] + [c for c in FILL if c in P.columns])
    keep_cols = [c for c in sub.columns if c != "MVT_ID_mvt"]
    J2 = (J.join(sub.rename({c: c + "__p" for c in keep_cols}), on="MVT_ID_mvt", how="left")
          .with_columns([pl.when(pl.col("icao__p").is_not_null()).then(pl.col(c + "__p")).otherwise(pl.col(c)).alias(c)
                         for c in keep_cols if c in J.columns])
          .with_columns(pl.when(pl.col("icao__p").is_not_null()).then(pl.lit("cs")).otherwise(pl.col("join")).alias("join"),
                        pl.when(pl.col("icao__p").is_not_null()).then(True).otherwise(pl.col("seen")).alias("seen"))
          .drop([c + "__p" for c in keep_cols]))
    J2.write_parquet(os.path.join(tmp, f"{tag}_joined.parquet"))
    lk = os.path.join(tmp, f"{tag}.parquet")
    if not os.path.exists(lk):
        os.symlink(os.path.join(SCR, f"{tag}.parquet"), lk)
    L.SCR = tmp
    try:
        f = L.day_features(day)
    finally:
        L.SCR = SCR
    os.remove(os.path.join(tmp, f"{tag}_joined.parquet"))
    os.remove(lk)
    return f.filter(pl.col("MVT_ID_mvt").is_in(P["MVT_ID_mvt"].implode()))


def dwell_day(day: str, P: pl.DataFrame | None) -> pl.DataFrame:
    """local_dwell_end.py for one day (callsign + positional rows with a push and a later stop >= 900 s)."""
    tag = L.tag_of(day)
    cols = ["MVT_ID_mvt", "ADEP_mvt", "t_mvt", "t_aobt3", "t_block", "t_sched", "icao", "t_move", "t_to", "push", "post_dwell"]
    J = pl.read_parquet(os.path.join(SCR, f"{tag}_joined.parquet"), columns=cols).with_columns(pl.lit("cs").alias("src"))
    if P is not None and P.height and "icao_pos" in P.columns:
        Pp = P.filter(pl.col("icao_pos").is_not_null()).select(
            [c for c in cols if c != "icao"] + [pl.col("icao_pos").alias("icao")]).with_columns(pl.lit("pos").alias("src"))
        J = pl.concat([J.filter(pl.col("icao").is_not_null()), Pp.select(J.columns)], how="vertical_relaxed")
    J = J.filter(pl.col("t_move").is_not_null() & (pl.col("push").fill_null(0) >= s1.PUSH_MIN)
                 & (pl.col("post_dwell").fill_null(0) >= s1.POST_DWELL_MAX) & (pl.col("ADEP_mvt") != "LIRF"))
    out = []
    if J.height:
        fx = pl.read_parquet(os.path.join(SCR, f"{tag}.parquet")).filter(pl.col("icao").is_in(J["icao"].unique().to_list()))
        g = {k: v.sort("t") for k, v in fx.partition_by(["icao", "apt"], as_dict=True).items()}
        for r in J.iter_rows(named=True):
            f = g.get((r["icao"], r["ADEP_mvt"]))
            if f is None:
                continue
            end = r["t_to"] if r["t_to"] is not None else r["t_mvt"] + 120
            f = f.filter(pl.col("ground") & (pl.col("t") > r["t_move"]) & (pl.col("t") < end))
            t, la, lo = f["t"].to_numpy(), f["lat"].to_numpy(), f["lon"].to_numpy()
            best, s0 = None, 0
            while s0 < len(t):
                e = s0
                while e + 1 < len(t) and s1.dist_m(la[s0], lo[s0], la[e + 1], lo[e + 1]) <= s1.STAT_R:
                    e += 1
                if best is None or t[e] - t[s0] > best[1] - best[0]:
                    best = (t[s0], t[e], float(s1.dist_m(la[0], lo[0], la[s0], lo[s0])) if len(t) else None)
                s0 = e + 1
            if best:
                out.append({**r, "dw_start": float(best[0]), "dw_end": float(best[1]), "dw_dist_from_push_m": best[2]})
    return pl.DataFrame(out, infer_schema_length=None)


def cmd_feat(days: list[str], force: bool) -> None:
    os.makedirs(FEAT, exist_ok=True)
    os.makedirs(POS, exist_ok=True)
    import local_adsb_posjoin as PJ
    PJ.OUT = POS  # write positional joins under adsb_out/pos, never $SCRATCH/pos
    for day in days:
        tag = L.tag_of(day)
        if not os.path.exists(os.path.join(SCR, f"{tag}_joined.parquet")):
            log(f"{day}: no joined file yet, skipped")
            continue
        done = os.path.join(FEAT, f"done_{day}")
        if os.path.exists(done) and not force:
            continue
        t0 = time.time()
        f = L.day_features(day)
        f.write_parquet(os.path.join(FEAT, f"feat_{day}.parquet"))
        PJ.run(tag)
        P = pl.read_parquet(os.path.join(POS, f"{tag}_pos.parquet"))
        fp = posjoin_learnfeat(day, P) if "icao_pos" in P.columns else None
        if fp is not None and fp.height:
            fp.write_parquet(os.path.join(FEAT, f"featpos_{day}.parquet"))
        dw = dwell_day(day, P)
        if dw.height:
            dw.write_parquet(os.path.join(POS, f"dwell_{day}.parquet"))
        open(done, "w").write("ok\n")
        log(f"{day}: callsign rows {f.height} (ground_fix {int(f['ground_fix'].fill_null(False).sum())}), positional "
            f"{0 if fp is None else fp.height}, dwell {dw.height}; {time.time() - t0:.0f} s, rss {rss_mb():.0f} MB")


# ------------------------------------------------------------------------------------------------ apply
def load_features():
    """Callsign features, positional learn features, positional joins and dwell rows for all 2026 days available:
    Jan+Jul from the production files in $SCRATCH (learn/feat_2026, learn/feat_pos_2026, pos/), the other days from
    adsb_out (cmd_feat)."""
    cs = [pl.read_parquet(os.path.join(LEARN, "feat_2026.parquet"))]
    cs += [pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(FEAT, "feat_2026-*.parquet")))]
    ps = [pl.read_parquet(os.path.join(LEARN, "feat_pos_2026.parquet"))]
    ps += [pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(FEAT, "featpos_2026-*.parquet")))]
    pj_files = sorted(glob.glob(os.path.join(SCR, "pos", "v2026.*_pos.parquet"))) + sorted(glob.glob(os.path.join(POS, "v2026.*_pos.parquet")))
    pj = pl.concat([pl.read_parquet(f) for f in pj_files], how="diagonal_relaxed")
    dw = [pl.read_parquet(os.path.join(SCR, "pos", "dwell_2026.parquet"))]
    dw += [pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(POS, "dwell_2026-*.parquet")))]
    out = []
    for parts in (cs, ps, None, dw):
        if parts is None:
            out.append(pj)
            continue
        X = pl.concat(parts, how="diagonal_relaxed")
        out.append(X)
    for i, X in enumerate(out):
        key = ["MVT_ID_mvt"] + (["src"] if "src" in X.columns else [])
        n = X.select(key).n_unique()
        assert n == X.height, f"duplicate rows in feature set {i}: {X.height} vs {n}"
    return out


def cmd_apply(base: str, dst: str, months: list[int] | None, tmvt: str = "final") -> None:
    meta = json.load(open(os.path.join(MODELS, "meta.json")))
    feats, allf = meta["features"], meta["features_apron"]
    models = {k: lgb.Booster(model_file=os.path.join(MODELS, f"{k}.txt")) for k in ("m12a", "m7a", "m4v2")}
    models.update({k: lgb.Booster(model_file=p) for k, p in SAVED.items()})
    for k, m in models.items():
        want = allf if k in ("m12a", "m7a") else feats
        assert m.feature_name() == want, k
    B = pl.read_parquet(base).select("MVT_ID_mvt", "TAXITIME_SEC_mvt")
    rk = pl.read_parquet(RANK, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "STAND_mvt"])
    S = B.join(rk, on="MVT_ID_mvt", how="left", maintain_order="left").with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("b"))
    assert S["ADEP_mvt"].null_count() == 0, "base has ids that are not in the final ranking set"
    codes = rk.select("MVT_ID_mvt", apron_expr())
    F, FP, PJ, DW = load_features()
    # The FINAL ranking set re-releases Jan/Jul with MVT_TIME (and BLOCK) jittered by up to +-12 s against the old
    # ranking set that the Jan/Jul ADS-B files were joined with. --tmvt final (default) re-anchors every ADS-B row
    # on the final MVT_TIME, so features, base and label share one clock; --tmvt stored keeps the stored t_mvt
    # (only for comparisons with old-ranking submissions such as v47).
    tm = rk.select("MVT_ID_mvt", (pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000).alias("_tm"))
    shifts = {}
    if tmvt == "final":
        def reanchor(X, name):
            X = X.join(tm, on="MVT_ID_mvt", how="left")
            dlt = (X["_tm"] - X["t_mvt"]).abs()
            shifts[name] = (int((dlt > 0.5).sum()), float(dlt.max() or 0.0))
            return X.with_columns(pl.coalesce("_tm", "t_mvt").alias("t_mvt")).drop("_tm")
        F, FP, PJ, DW = reanchor(F, "callsign"), reanchor(FP, "posfeat"), reanchor(PJ, "posjoin"), reanchor(DW, "dwell")
        print("t_mvt re-anchored to the final MVT_TIME (rows moved, max |shift| s):", shifts)
    sel = pl.col("month").is_in(months) if months else pl.lit(True)
    Sb = S.filter(sel).select("MVT_ID_mvt", "ADEP_mvt", "month", "b", "MVT_TIME_UTC_mvt")

    def predict(G: pl.DataFrame, tag: str) -> np.ndarray:
        if G.height == 0:
            return np.zeros(0)
        Xg, f2, _ = L.design(G, "b")
        fn = allf if tag in ("m12a", "m7a") else feats
        assert f2 == feats
        return np.clip(models[tag].predict(Xg.select(fn).to_numpy().astype(np.float64)), -CLIP_C, CLIP_C)

    parts = []
    # callsign rows
    G = F.drop("ADEP_mvt").join(Sb, on="MVT_ID_mvt", how="inner")
    tchk = (G["t_mvt"] - G["MVT_TIME_UTC_mvt"].dt.epoch("ms") / 1000).abs().max()
    assert tchk is None or tchk <= 12.5, f"t_mvt does not match MVT_TIME ({tchk})"
    G = add_apron(G, codes)
    gf = pl.col("ground_fix").fill_null(False)
    rules = [("G12A", "m12a", gf & ~pl.col("ADEP_mvt").is_in(["LTFM", "LIRF", "LFPG"])),
             ("G7A", "m7a", gf & (pl.col("ADEP_mvt") == "LFPG")),
             ("GL4", "m4v2", gf & (pl.col("ADEP_mvt") == "LIRF")),
             ("AIR", "mA", ~gf & (pl.col("ADEP_mvt") != "LTFM"))]
    for comp, tag, cond in rules:
        g = G.filter(cond)
        parts.append(g.select("MVT_ID_mvt").with_columns(pl.lit(comp).alias("comp"), pl.Series("c", predict(g, tag)),
                                                         pl.lit(None, dtype=pl.Float64).alias("a"), pl.lit(None, dtype=pl.Float64).alias("w")))
    n_cs_ltfm = G.filter(pl.col("ADEP_mvt") == "LTFM").height
    # positional-join rule rows (local_posjoin_overlay.py), unmatched only, LIRF excluded
    T = (PJ.filter(pl.col("t_aobt3").is_null() & pl.col("t_move").is_not_null() & (pl.col("ADEP_mvt") != "LIRF")
                   & (pl.col("push").fill_null(0) >= s1.PUSH_MIN) & (pl.col("post_dwell").fill_null(0) < s1.POST_DWELL_MAX))
         .select("MVT_ID_mvt", (pl.col("t_mvt") - pl.col("t_move")).alias("a")).with_columns(pl.lit(1.0).alias("w")))
    Dw = (DW.filter((pl.col("src") == "pos") & pl.col("t_aobt3").is_null())
          .select("MVT_ID_mvt", (pl.col("t_mvt") - pl.when(pl.col("dw_dist_from_push_m") < 100).then(pl.col("dw_end"))
                                 .otherwise(pl.col("t_move"))).alias("a")))
    Lg = Dw.join(T.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti").with_columns(pl.lit(0.75).alias("w"))
    T = pl.concat([T, Lg]).join(Sb.select("MVT_ID_mvt", "b"), on="MVT_ID_mvt", how="inner")
    T = T.join(G.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti")  # never on a callsign row (cannot happen by construction)
    parts.append(T.select("MVT_ID_mvt", pl.lit("RULE").alias("comp"), (pl.col("w") * (pl.col("a") - pl.col("b"))).alias("c"), "a", "w"))
    # positional-join learned rows
    GP = (FP.drop("ADEP_mvt").join(Sb, on="MVT_ID_mvt", how="inner")
          .filter(~pl.col("ADEP_mvt").is_in(["LIRF", "LTFM"]))
          .join(T.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti").join(G.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti"))
    parts.append(GP.select("MVT_ID_mvt").with_columns(pl.lit("POS").alias("comp"), pl.Series("c", predict(GP, "m7v2")),
                                                      pl.lit(None, dtype=pl.Float64).alias("a"), pl.lit(None, dtype=pl.Float64).alias("w")))
    C = pl.concat(parts, how="vertical_relaxed")
    assert C["MVT_ID_mvt"].n_unique() == C.height
    Dt = (S.select("MVT_ID_mvt", "ADEP_mvt", "month", "b").join(C, on="MVT_ID_mvt", how="left", maintain_order="left")
          .with_columns(pl.col("comp").fill_null("none"), pl.col("c").fill_null(0.0)))
    newf = (pl.col("b") + pl.col("c")).round(0)
    Dt = Dt.with_columns(pl.when(newf < 0).then(pl.col("b")).otherwise(newf).alias("new"))
    out = B.with_columns(Dt["new"].cast(pl.Int32).alias("TAXITIME_SEC_mvt"))
    assert out.height == B.height and out["MVT_ID_mvt"].equals(B["MVT_ID_mvt"]) and out.null_count().sum_horizontal().item() == 0
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    out.write_parquet(dst)
    det = os.path.splitext(dst)[0] + "_detail.parquet"
    Dt.with_columns((pl.col("new") - pl.col("b")).alias("d")).write_parquet(det)
    d = (Dt["new"] - Dt["b"]).to_numpy()
    print(f"wrote {dst} ({out.height} rows) and {det}")
    print(f"rows changed {int((d != 0).sum())}; sum d^2 / N {float((d ** 2).sum()) / len(d):.1f}; callsign LTFM rows left at 0: {n_cs_ltfm}")
    summary(Dt)


def summary(Dt: pl.DataFrame) -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    Dt = Dt.with_columns((pl.col("new") - pl.col("b")).alias("d"))
    print("by component:")
    print(Dt.group_by("comp").agg(pl.len().alias("n"), (pl.col("d") != 0).sum().alias("changed"),
                                  pl.col("d").mean().round(1).alias("mean"), (pl.col("d") ** 2).mean().sqrt().round(1).alias("rms"),
                                  *[pl.col("d").quantile(q).alias(f"q{int(q * 100):02d}") for q in (0.01, 0.1, 0.5, 0.9, 0.99)])
          .sort("comp"))
    print("by airport x month (cov = share of departures with any component; g = ground-fix learned share):")
    print(Dt.group_by("ADEP_mvt", "month").agg(
        pl.len().alias("n"), (pl.col("comp") != "none").mean().round(3).alias("cov"),
        pl.col("comp").is_in(["G12A", "G7A", "GL4"]).mean().round(3).alias("g"),
        (pl.col("comp") == "AIR").mean().round(3).alias("air"),
        pl.col("comp").is_in(["POS", "RULE"]).sum().alias("pos+rule"),
        pl.col("d").filter(pl.col("comp") != "none").mean().round(1).alias("mean_d"),
        (pl.col("d") ** 2).mean().sqrt().round(1).alias("rms_d_all"),
        pl.col("d").filter(pl.col("comp") != "none").quantile(0.1).alias("q10"),
        pl.col("d").filter(pl.col("comp") != "none").quantile(0.5).alias("q50"),
        pl.col("d").filter(pl.col("comp") != "none").quantile(0.9).alias("q90")).sort("ADEP_mvt", "month"))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--check-7v2", action="store_true")
    g = sp.add_parser("feat")
    g.add_argument("--days", required=True)
    g.add_argument("--force", action="store_true")
    a = sp.add_parser("apply")
    a.add_argument("--base", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--months", default="", help="comma list of months to correct (default: all)")
    a.add_argument("--tmvt", default="final", choices=["final", "stored"])
    ns = ap.parse_args(ARGV[1:])
    if ns.cmd == "fit":
        cmd_fit(ns.check_7v2)
    elif ns.cmd == "feat":
        cmd_feat([d for d in ns.days.split(",") if d], ns.force)
    else:
        cmd_apply(ns.base, ns.out, [int(m) for m in ns.months.split(",") if m] or None, ns.tmvt)


if __name__ == "__main__":
    main()

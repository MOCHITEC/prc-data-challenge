"""plan_rebase_c12.md section 3: the 12-month appear as a per-row delta on C12 (v42), evaluated at C12's base.

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/apply_rebase_c12.py <dir with v38 v39 v42 v43>

b38 is recovered from v38 exactly as v39 was built (fixed point with the plain 4-month P4 only); with it, (c_A4 - c_P4)
reproduces v39 - v38 on every row (plan_rebase_c12.md section 4). Recovering b from v39 instead (as apply_delta12.py did)
gives a different b and a 4-month term RMS 24.5 s off the one in the file: the cause of v46's miss. The 4-month term C12
carries is v39 - v38. C12's base on the appear rows is b_C12 = b38 + (v42 - v43) (the requester's estimate).
Deltas relative to v42, on the rows with appear columns (LTFM 0):
  swap = -(v39 - v38) + (c_A12 - c_P12)(b_C12)
  old  = -(v39 - v38) + (c_A12 - c_P12)(b38)               (the 12-month term at the same b as C12's other ADS-B layers)
  req  = (c_A12 - c_P12 - c_A4 + c_P4)(b_C12)            (the request's formula; re-computes the 4-month term, for the record)
  full = swap + P(b_C12) - P(b38)                        (also refresh C12's plain residual at C12's base, as a change;
                                                          P = c_P4 at LIRF, c_P12 elsewhere; apron stays as it is.
                                                          Post hoc: cv_stale's C0full -0.20 / -0.29 s)
  clean = b_C12 + c_A12(b_C12) - v42                     (rebuild every ADS-B layer of C12 in this model frame: drops the
                                                          other session's plain swaps and apron, which local_adsb_apply_swap.py
                                                          evaluated at b = file - c_OLD(b), i.e. with v39's 4-month appear
                                                          term left inside b; cv_stale's arm "full" -1.15 / -1.25 s)
Writes appear12_c12_{swap,old,req,full,clean}_delta.parquet (MVT_ID_mvt, delta in whole seconds) next to the inputs; the overlay
exclusions and the LOBT clip are then applied by c12_apply.py. Nothing is uploaded."""

import os
import sys

SRC = sys.argv[1]
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_months12 as M  # noqa: E402  (sets --base a2)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

CV, L, AP = M.CV, M.L, M.AP
CV.PARAMS = {**CV.PARAMS, **M.V2}


def main() -> None:
    models = {k: lgb.Booster(model_file=os.path.join(L.OUT, f"final_model_a2_delta_months_fill_v2_{k[0]}.txt")) for k in ("P4", "A4")}
    models.update({k: lgb.Booster(model_file=os.path.join(L.OUT, f"final_model_a2_delta_m12_v2_{k[0]}.txt")) for k in ("P12", "A12")})
    tpl = pl.read_parquet(os.path.join(CV.ROOT, "data/raw/submitting.parquet"))
    files = {}
    for v in (38, 39, 42, 43):
        s = pl.read_parquet(os.path.join(SRC, f"v{v}.parquet"))
        assert s.schema == tpl.schema and (s["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
        files[v] = s.select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(f"v{v}"))
    rk = pl.scan_parquet(os.path.join(CV.ROOT, "data/raw/ranking.parquet")).select("MVT_ID_mvt", "ADEP_mvt").collect()
    G = pl.read_parquet(os.path.join(L.OUT, "feat_2026.parquet")).drop("ADEP_mvt").join(rk, on="MVT_ID_mvt")
    for v in files:
        G = G.join(files[v], on="MVT_ID_mvt")
    G = AP.add(G.filter(pl.col("ADEP_mvt") != "LTFM"), "fill_2026")
    has = G["own_ok"].is_not_null().to_numpy()
    print(f"2026 ADS-B feature rows (not LTFM) {G.height:,}; with appear columns {int(has.sum()):,}", flush=True)

    def corr(k, b):
        Xg, f2, _ = L.design(G.with_columns(pl.Series("b", b)), "b")
        if k[0] == "A":
            Xg, f2 = AP.design(Xg, f2, "b")
        assert f2 == models[k].feature_name()
        return np.clip(models[k].predict(Xg.select(f2).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)

    f38 = G["v38"].to_numpy()
    b = f38.copy()
    for it in range(3):
        b_new = f38 - corr("P4", b)
        print(f"b38 fixed point step {it}: max |db| {np.abs(b_new - b).max():.1f}", flush=True)
        b = b_new
    app4_file = (G["v39"] - G["v38"]).to_numpy()
    app4_model = np.where(has, corr("A4", b) - corr("P4", b), 0.0)
    rep = np.where(has, np.round(f38 + app4_model), f38)
    rep = np.where(rep < 0, f38, rep)
    print(f"4-month term in the file (v39 - v38) vs the model at b38: RMS diff {np.sqrt(np.mean((app4_file - app4_model)[has] ** 2)):.2f} s "
          f"(term RMS {np.sqrt(np.mean(app4_file[has] ** 2)):.2f} s); v39 reproduced on {int((rep == G['v39'].to_numpy()).sum()):,} "
          f"of {G.height:,} rows", flush=True)
    assert (rep == G["v39"].to_numpy()).mean() > 0.999, "b38 does not reproduce v39"
    shift = (G["v42"] - G["v43"]).to_numpy()
    b_c12 = b + shift
    print(f"b shift v39 -> C12 on appear rows: RMS {np.sqrt(np.mean(shift[has] ** 2)):.1f} s, median |.| {np.median(np.abs(shift[has])):.1f} s", flush=True)
    a12 = corr("A12", b_c12) - corr("P12", b_c12)
    a4n = corr("A4", b_c12) - corr("P4", b_c12)
    a12_old = corr("A12", b) - corr("P12", b)
    lirf = (G["ADEP_mvt"] == "LIRF").to_numpy()
    plain_ref = np.where(lirf, corr("P4", b_c12) - corr("P4", b), corr("P12", b_c12) - corr("P12", b))
    deltas = {"swap": np.where(has, a12 - app4_file, 0.0), "old": np.where(has, a12_old - app4_file, 0.0),
              "req": np.where(has, a12 - a4n, 0.0), "full": np.where(has, a12 - app4_file + plain_ref, 0.0),
              "clean": np.where(has, b_c12 + corr("A12", b_c12) - G["v42"].to_numpy(), 0.0)}
    old = pl.read_parquet(os.path.join(CV.ROOT, "handoff/2026-10-06/appear12_delta.parquet"))
    oldv = G.select("MVT_ID_mvt").join(old, on="MVT_ID_mvt", how="left")["delta"].fill_null(0).to_numpy().astype(float)
    for k, d in deltas.items():
        print(f"{k}: rows {int((np.round(d) != 0).sum()):,}, RMS {np.sqrt(np.mean(d[has] ** 2)):.2f} s, sum d^2/N(344,841) "
              f"{float((d ** 2).sum()) / 344841:.0f}, mean {d[has].mean():+.2f}; vs the v39-based table: RMS diff "
              f"{np.sqrt(np.mean((d - oldv)[has] ** 2)):.2f} s, corr {np.corrcoef(d[has], oldv[has])[0, 1]:.3f}", flush=True)
        out = G.select("MVT_ID_mvt").with_columns(pl.Series("delta", np.round(d)).cast(pl.Int32)).filter(pl.col("delta") != 0)
        out.write_parquet(os.path.join(SRC, f"appear12_c12_{k}_delta.parquet"))


if __name__ == "__main__":
    main()

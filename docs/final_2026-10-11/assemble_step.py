"""Assemble a v48-based 4-month final file from a (possibly new) pre-ADS-B blend.

    python assemble_step.py --a2o <a2o member parquet (matched rows: pred_fix[,pred_orig])> --tag <tag> [--c13]

Chain for the given blend: adsb_layer.py apply -> apply_overlays.py (laneB + LIRF overlays + LOBT clip) and the
appear increment (other thread's final_adsb.py --increment) on the SAME base.
Rows:
  Jan/Jul : v48 + (chain_new - chain_orig) [+ (1/8)(delta_a2o_exact - delta_pao_applied) on matched rows, --c13]
  Feb/Jun : chain_new + appear increment (2026-06-11 rows: A12 chain + (b_new - b0))
Then LOBT clip on every matched row, clip at 0, round, int32. Writes handoff/<tag>.parquet + janjul_only_<tag>.parquet.
"""
import argparse
import os
import subprocess
import sys

import polars as pl

A = "/home/user/prc-final/assemble/"
H = "/home/user/prc-data-challenge/handoff/2026-10-13/"
F = "/home/user/pb-work/outputs/final/"
PY = "/home/user/prc-work/.venv/bin/python"
ENV = {**os.environ, "POLARS_MAX_THREADS": "2", "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage"}

ap = argparse.ArgumentParser()
ap.add_argument("--a2o", default=None, help="parquet with MVT_ID_mvt, pred_fix (matched rows); default: production a2o")
ap.add_argument("--pao", default=F + "main_flseq_osm_nolirf_l2_pa_opt_l6.parquet")
ap.add_argument("--p2o", default=F + "main_flseq_osm_nolirf_plainy_l2_opt_l6.parquet")
ap.add_argument("--m4", default=F + "main_flseq_osm_nolirf_l2_mlp_l6.parquet")
ap.add_argument("--cb", nargs="+", default=[F + "main_flseq_osm_nolirf_cb_l2.parquet", F + "main_flseq_osm_nolirf_cb_l2_s43.parquet",
                                             F + "main_flseq_osm_nolirf_cb_l2_s44.parquet"])
ap.add_argument("--tag", required=True)
ap.add_argument("--c13", action="store_true", help="add the exact a2o C13 correction on Jan/Jul matched rows")
ap.add_argument("--skip-chain", action="store_true", help="reuse chain files already built for this tag")
a = ap.parse_args()


def member(path, name, col=None):
    d = pl.read_parquet(path)
    col = col or next(c for c in d.columns if c in ("TAXITIME_SEC_mvt", "pred_fix", "pred"))
    return d.select("MVT_ID_mvt", pl.col(col).cast(pl.Float64).alias(name))


tpl = pl.read_parquet("/home/user/prc-work/data/final/final_submitting.parquet").select("MVT_ID_mvt")
fr = pl.read_parquet("/home/user/pb-work/data/work_final/feat_rank_unfixed.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "mvt_minus_aobt3", "mvt_minus_lobt"])
mon = (pl.read_parquet("/home/user/prc-work/data/final/final_ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
       .filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"), pl.col("MVT_TIME_UTC_mvt").dt.day().alias("dd")))
d = tpl.join(fr, on="MVT_ID_mvt", how="left", maintain_order="left").join(mon, on="MVT_ID_mvt", how="left", maintain_order="left")
mat = pl.col("mvt_minus_aobt3").is_not_null()

# 1. blend
a2o_prod = member(F + "main_flseq_osm_nolirf_l2_opt_l6_cmix.parquet", "a2o_prod")
d = d.join(a2o_prod, on="MVT_ID_mvt", how="left", maintain_order="left")
if a.a2o:
    d = d.join(member(a.a2o, "a2o", "pred_fix"), on="MVT_ID_mvt", how="left", maintain_order="left")
    d = d.with_columns(pl.coalesce("a2o", "a2o_prod").alias("a2o"))
else:
    d = d.with_columns(pl.col("a2o_prod").alias("a2o"))
d = d.join(member(a.pao, "pao"), on="MVT_ID_mvt", how="left", maintain_order="left") \
     .join(member(a.p2o, "p2o"), on="MVT_ID_mvt", how="left", maintain_order="left") \
     .join(member(a.m4, "m4"), on="MVT_ID_mvt", how="left", maintain_order="left")
for i, p in enumerate(a.cb):
    d = d.join(member(p, f"cb{i}"), on="MVT_ID_mvt", how="left", maintain_order="left")
c2 = sum(pl.col(f"cb{i}") for i in range(len(a.cb))) / len(a.cb)
blend = ((0.5 * pl.col("a2o") + 0.5 * pl.col("pao")) + pl.col("p2o") + c2 + pl.col("m4")) / 4
d = d.with_columns(pl.when(mat).then(blend).otherwise(pl.col("a2o_prod")).alias("b_new"))
assert d["b_new"].null_count() == 0
base = A + f"base_blend_{a.tag}.parquet"
d.select("MVT_ID_mvt", pl.col("b_new").alias("TAXITIME_SEC_mvt")).write_parquet(base)
b0 = member(A + "base_blend.parquet", "b0")
d = d.join(b0, on="MVT_ID_mvt", how="left", maintain_order="left")
print(f"blend written; rms(b_new - b0) = {float(((d['b_new'] - d['b0']) ** 2).mean()) ** .5:.2f}", flush=True)

# 2. chain + increment on this base
adsb = A + f"blend_{a.tag}_adsb.parquet"
ovl = A + f"blend_{a.tag}_adsb_ovl.parquet"
inc = A + f"incr_{a.tag}_febjun.parquet"
if not a.skip_chain:
    subprocess.run([PY, "adsb_layer.py", "apply", "--base", base, "--out", adsb], cwd="/home/user/prc-final/adsb_out", env=ENV, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([PY, "overlays/apply_overlays.py", "--base", adsb, "--out", ovl, "--laneB", "--lobt-clip", "--copy-feat", "unfixed"],
                   cwd="/home/user/prc-final", env=ENV, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([PY, "docs/appear_tier_2026-10-05/final_adsb.py", base, inc, "--months", "02,06", "--increment"],
                   cwd="/home/user/bd-work", env={**ENV, "SCRATCH": "/home/user/bd-work/handoff/2026-10-11/adsb"}, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("chain + increment built", flush=True)

d = d.join(member(ovl, "new"), on="MVT_ID_mvt", how="left", maintain_order="left") \
     .join(member(A + "blend_adsb2_ovl.parquet", "orig"), on="MVT_ID_mvt", how="left", maintain_order="left") \
     .join(member(A + "blend_A12_ovl.parquet", "a12"), on="MVT_ID_mvt", how="left", maintain_order="left") \
     .join(member(inc, "inc_out"), on="MVT_ID_mvt", how="left", maintain_order="left") \
     .join(member("/home/user/bd-work/handoff/2026-10-08/C15_appear12_cov26eddf.parquet", "v48"), on="MVT_ID_mvt", how="left", maintain_order="left")
d = d.with_columns((pl.col("inc_out") - pl.col("b_new").round(0)).alias("inc"))

jj = pl.col("m").is_in([1, 7])
jun11 = (pl.col("m") == 6) & (pl.col("dd") == 11)
# 2026-06-11: the A12 store has no appear rows for that day (audit 2026-10-13), so it is treated like every other
# Feb/Jun day: own ADS-B layer (adsb_out/feat+pos for 06-11 copied from beautiful-davinci handoff) + increment.
p = (pl.when(jj).then(pl.col("v48") + (pl.col("new") - pl.col("orig")))
       .otherwise(pl.col("new") + pl.col("inc")))
d = d.with_columns(p.alias("p"))

# 3. exact C13 on Jan/Jul matched rows
if a.c13:
    assert a.a2o, "--c13 needs --a2o with pred_fix and pred_orig"
    # v47's a2o slot carried s * d_pao / 8 with s = 1 - 0.4 q (LIRF copy overlay scaling), d_pao from the old-frame pao
    # flfix run. Exact: s * d_a2o / 8. Correction e = s * (d_a2o_new - d_pao_old) / 8 on the old Jan/Jul matched ids.
    x = pl.read_parquet(a.a2o).select("MVT_ID_mvt", (pl.col("pred_fix") - pl.col("pred_orig")).alias("d_a2o"))
    pao_old = pl.read_parquet("/home/user/pb-work/outputs/flfix/main_flseq_osm_nolirf_l2_pa_opt_l6.parquet").select(
        "MVT_ID_mvt", (pl.col("pred_fix") - pl.col("pred_orig")).alias("d_pao"))
    q = pl.read_parquet("/tmp/claude-0/-home-user-prc-data-challenge/7ae154e2-3baf-5cb4-beae-354c18126ab1/scratchpad/adsb/flfix/copyq_LIRF.parquet")
    q = q.select("MVT_ID_mvt", pl.col("q_orig").cast(pl.Float64))
    d = d.join(x, on="MVT_ID_mvt", how="left", maintain_order="left").join(pao_old, on="MVT_ID_mvt", how="inner" if False else "left", maintain_order="left") \
         .join(q, on="MVT_ID_mvt", how="left", maintain_order="left")
    s = pl.when(pl.col("ADEP_mvt") == "LIRF").then(1.0 - 0.4 * pl.col("q_orig").fill_null(0.0)).otherwise(1.0)
    ok13 = jj & mat & pl.col("d_a2o").is_not_null() & pl.col("d_pao").is_not_null()
    d = d.with_columns(pl.when(ok13).then(s * (pl.col("d_a2o") - pl.col("d_pao")) / 8).otherwise(0.0).alias("c13"))
    n13 = int(d.select(ok13.sum()).item())
    e2 = float((d["c13"] ** 2).sum()) / 344841
    emax = float(d["c13"].abs().max())
    da2 = d.filter(ok13)["d_a2o"]
    print(f"C13: rows {n13} (expect 339551), sum e^2/344841 {e2:.2f} (gate < 5), max|e| {emax:.1f} (gate < 150), "
          f"sum d_a2o^2/344841 {float((da2 ** 2).sum()) / 344841:.1f} (expect 40-300), mean d_a2o {float(da2.mean()):+.2f} (gate |.| < 3)", flush=True)
    if not (n13 == 339551 and e2 < 5 and emax < 150 and abs(float(da2.mean())) < 3):
        print("C13 gates FAILED -> correction dropped", flush=True)
        d = d.with_columns(pl.lit(0.0).alias("c13"))
    d = d.with_columns((pl.col("p") + pl.col("c13")).alias("p"))

# 4. LOBT clip on matched rows, floor, round
lo, hi = pl.col("mvt_minus_lobt") - 3606, pl.col("mvt_minus_lobt") + 3606
p2 = pl.when(pl.col("mvt_minus_lobt").is_not_null()).then(pl.max_horizontal(pl.min_horizontal(pl.col("p"), hi.ceil()), lo.ceil())).otherwise(pl.col("p"))
out = d.select("MVT_ID_mvt", p2.clip(0).round(0).cast(pl.Int32).alias("TAXITIME_SEC_mvt"))
assert out.null_count().sum_horizontal().item() == 0 and out.height == 670790
cur = pl.read_parquet(H + "final_recommended.parquet") if os.path.exists(H + "final_recommended.parquet") else None
if cur is not None:
    dd = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - cur["TAXITIME_SEC_mvt"].cast(pl.Float64)
    rep = d.select("m").with_columns(dd.alias("chg")).group_by("m").agg((pl.col("chg") != 0).sum().alias("changed"), (pl.col("chg") ** 2).mean().sqrt().alias("rms")).sort("m")
    print("vs current recommended:\n", rep, flush=True)
out.write_parquet(H + f"{a.tag}.parquet")
sub = pl.read_parquet("/home/user/prc-work/data/raw/submitting.parquet").select("MVT_ID_mvt")
sub.join(out, on="MVT_ID_mvt", how="left", maintain_order="left").write_parquet(H + f"janjul_only_{a.tag}.parquet")
print("written", H + f"{a.tag}.parquet")

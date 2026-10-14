"""Lane-B (unmatched, AOBT_3 null) post-processing deltas for the FINAL evaluation set: NM-blind blend (w 0.25,
unmatched rows outside LIRF) and ucmix (gated LIRF copy mixture, 0 < MVT-SCHED < 8 h).

Same method as prc-data-challenge analysis/local_nmb_laneB.py and analysis/local_ucmix_laneB.py (which produced
/home/user/pb-work/outputs/{nmb,ucmix}_laneB_delta.parquet for the old Jan+Jul ranking set):
only lane B is fitted (the target frame holds the unmatched ranking rows only), and the blend functions are wrapped
to record their input (plain model-B prediction) and output, so delta = blended - plain.

One fit gives both deltas: the config is main_flseq_osm_nolirf_l2_opt_nmb.yaml + ucmix_airports [LIRF]
(= main_flseq_osm_nolirf_l2_opt_ucmix.yaml + nmblind_weight 0.25). The two blends touch disjoint rows (NM-blind
skips the sched-reparam airport LIRF; ucmix is LIRF-only), model B and its seeds do not depend on either flag, so
each delta is identical to what the corresponding single-flag run gives. --separate runs the two configs one after
the other instead (two fits).

    cd /home/user/pb-work && PYTHONPATH=src POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \\
      /home/user/prc-work/.venv/bin/python /home/user/prc-final/overlays/laneB_deltas.py \\
      --raw-dir data/raw_final --work-dir data/work_final --out-dir /home/user/prc-final/overlays/out [--tag final]

Outputs: <out-dir>/nmb_laneB_delta_<tag>.parquet, <out-dir>/ucmix_laneB_delta_<tag>.parquet
(MVT_ID_mvt, plain, blend, delta, ADEP_mvt; one row per unmatched ranking departure, delta = 0 where not touched).
"""
import argparse
import time
from pathlib import Path

import polars as pl
import yaml

import prc.train as T
from prc.levers import attach_levers
from prc.pipeline import drop_hidden_candidates, load_raw, train_config_from_spec

ap = argparse.ArgumentParser()
ap.add_argument("--raw-dir", default="data/raw_final")
ap.add_argument("--work-dir", default="data/work_final")
ap.add_argument("--out-dir", default="/home/user/prc-final/overlays/out")
ap.add_argument("--tag", default="final")
ap.add_argument("--separate", action="store_true", help="two fits (nmb config, ucmix config) instead of one")
a = ap.parse_args()
out_dir = Path(a.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

rec = {}


def wrap(name, orig):
    def _w(tr, u_tg, g, cfg_, cols_b, cat_cols, info):
        rec[name] = {"ids": u_tg["MVT_ID_mvt"], "plain": g.copy()}
        out = orig(tr, u_tg, g, cfg_, cols_b, cat_cols, info)
        rec[name]["blend"] = out.copy()
        return out
    return _w


T._nmblind_blend = wrap("nmb", T._nmblind_blend)
T._ucmix_blend = wrap("ucmix", T._ucmix_blend)

t0 = time.time()
spec_nmb = yaml.safe_load(open("configs/main_flseq_osm_nolirf_l2_opt_nmb.yaml"))
spec_uc = yaml.safe_load(open("configs/main_flseq_osm_nolirf_l2_opt_ucmix.yaml"))
assert {k: v for k, v in spec_nmb.items() if k not in ("name", "nmblind_weight")} == \
       {k: v for k, v in spec_uc.items() if k not in ("name", "ucmix_airports")}, "configs differ beyond the two flags"
if a.separate:
    specs = [spec_nmb, spec_uc]
else:
    both = dict(spec_nmb, name="main_flseq_osm_nolirf_l2_opt_nmb_ucmix", ucmix_airports=spec_uc["ucmix_airports"])
    specs = [both]
cfg0 = train_config_from_spec(specs[0])

work = Path(a.work_dir)
f_tr, f_rk = pl.read_parquet(work / "feat_train.parquet"), pl.read_parquet(work / "feat_rank.parquet")
raw_tr, raw_rk, _ = load_raw(Path(a.raw_dir))
# same lever step as prc.pipeline.main (fe_fwd_nm on, fe_retention off in these configs)
f_tr, f_rk = attach_levers(f_tr, f_rk, raw_tr, raw_rk, l1=cfg0.fe_plan_nm_taxi, l2=cfg0.fe_fwd_nm or cfg0.fe_fwd_nm_unmatched,
                           l3=cfg0.fe_push_queue, l5=cfg0.fe_tx_nm, l6=cfg0.fe_retention)
del raw_tr, raw_rk
f_tr = drop_hidden_candidates(f_tr, cfg0)
f_rk = drop_hidden_candidates(f_rk, cfg0).filter(pl.col("mvt_minus_aobt3").is_null())
print(f"frames: train {f_tr.height:,}, unmatched target {f_rk.height:,} ({time.time() - t0:.0f}s)", flush=True)

infos = []
for spec in specs:
    cfg = train_config_from_spec(spec)
    _, info = T.predict_fold(f_tr, f_rk, cfg)
    infos.append({k: v for k, v in info.items() if "nmblind" in k or "ucmix" in k or "round" in k})
    print(spec["name"], infos[-1], f"({time.time() - t0:.0f}s)", flush=True)

apt = f_rk.select("MVT_ID_mvt", "ADEP_mvt")
if not a.separate:
    # model B is shared by construction in a single fit. ucmix ran after the NM-blind blend, so its recorded input
    # is the NM-blended vector; NM-blind never touches LIRF and ucmix only touches LIRF, so re-base ucmix on the plain
    # model-B prediction to keep the original file semantics (plain = model B, blend = plain + delta).
    assert (rec["nmb"]["ids"] == rec["ucmix"]["ids"]).all()
    du = rec["ucmix"]["blend"] - rec["ucmix"]["plain"]
    dn = rec["nmb"]["blend"] - rec["nmb"]["plain"]
    assert not ((abs(du) > 1e-9) & (abs(dn) > 1e-9)).any(), "nmb and ucmix touched the same rows"
    rec["ucmix"] = {"ids": rec["nmb"]["ids"], "plain": rec["nmb"]["plain"], "blend": rec["nmb"]["plain"] + du}
for name in ("nmb", "ucmix"):
    r = rec[name]
    d = (pl.DataFrame({"MVT_ID_mvt": r["ids"], "plain": r["plain"], "blend": r["blend"]})
         .with_columns((pl.col("blend") - pl.col("plain")).alias("delta")).join(apt, on="MVT_ID_mvt"))
    p = out_dir / f"{name}_laneB_delta_{a.tag}.parquet"
    d.write_parquet(p)
    print(f"{p}: rows {d.height} changed {int((d['delta'].abs() > 0.5).sum())} "
          f"sum d^2 {float((d['delta'] ** 2).sum()):.4g}")
    print(d.group_by("ADEP_mvt").agg(pl.len(), pl.col("delta").mean().round(1),
                                     pl.col("delta").abs().mean().round(1).alias("abs")).sort("ADEP_mvt"))
print(f"done {time.time() - t0:.0f}s")

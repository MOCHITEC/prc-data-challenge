"""Final phase: re-join the Jan/Jul 2026 ADS-B fixes to the final-phase departures, whose MVT times are dithered by up to
+-12 s (handoff/2026-10-11/FINAL_PHASE.md), and rebuild the residual features and appear (fill) columns on them.

    SRC=<adsb dir with the extracted v2026.01/07 fixes> SCRATCH=<new final dir> PRC_FINAL=1 POLARS_MAX_THREADS=2 \
        .venv/bin/python docs/appear_tier_2026-10-05/final_janjul_adsb.py

No download: the per-day filtered fixes of SRC are symlinked into SCRATCH, then stage-1 `analyse` (departures from
final_ranking.parquet, PRC_FINAL=1), `adsb_learn_features.day_features` and `adsb_appear.day_appear` (stand table
"fill", the 2026 stand map of SRC) run per day. Writes SCRATCH/learn/feat_2026.parquet and appear_fill_2026.parquet,
and links the models and the Feb/Jun files of SRC/learn next to them, so that final_adsb.py with this SCRATCH covers
all four months. SRC is never written."""

import os
import shutil
import sys
from contextlib import redirect_stdout

import polars as pl

SRC = os.environ["SRC"]
assert os.environ.get("PRC_FINAL") == "1", "set PRC_FINAL=1 (Jan/Jul departures from final_ranking)"
assert os.path.abspath(SRC) != os.path.abspath(os.environ["SCRATCH"]), "SCRATCH must be the new final dir"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_appear as A  # noqa: E402
import adsb_learn_features as L  # noqa: E402
import adsb_stage1_feasibility as s1  # noqa: E402

LINKS = ("final_model_a2_delta_m12_v2_A.txt", "final_model_a2_delta_m12_v2_A_thinEDDF.txt",
         "feat_2026fj.parquet", "appear_fill_2026fj.parquet")


def main() -> None:
    os.makedirs(L.OUT, exist_ok=True)
    m = os.path.join(L.OUT, "stand_map_2026.parquet")
    if not os.path.exists(m):
        shutil.copy(os.path.join(SRC, "learn", "stand_map_2026.parquet"), m)
    import stand_map as SM
    ST = SM.stand_table("fill")
    have = []
    for d in L.DAYS["2026"]:
        tag = L.tag_of(d)
        src_fx = os.path.join(SRC, f"{tag}.parquet")
        if not os.path.exists(src_fx):
            continue
        dst_fx = os.path.join(s1.SCRATCH, f"{tag}.parquet")
        if not os.path.exists(dst_fx):
            os.symlink(src_fx, dst_fx)
        fp, ap = os.path.join(L.OUT, f"feat_{d}.parquet"), os.path.join(L.OUT, f"appear_fill_{d}.parquet")
        if not (os.path.exists(fp) and os.path.exists(ap)):
            with open(os.devnull, "w") as dn, redirect_stdout(dn):
                s1.analyse(tag, d)
            L.day_features(d).write_parquet(fp)
            A.day_appear(d, ST).write_parquet(ap)
        have.append(d)
    print(f"Jan/Jul days re-joined: {len(have)} of {len(L.DAYS['2026'])}", flush=True)
    for kind, name in (("feat", "feat_2026"), ("appear_fill", "appear_fill_2026")):
        D = pl.concat([pl.read_parquet(os.path.join(L.OUT, f"{kind}_{d}.parquet")) for d in have], how="diagonal_relaxed")
        assert D["MVT_ID_mvt"].n_unique() == D.height
        D.write_parquet(os.path.join(L.OUT, f"{name}.parquet"))
        print(f"{name}: {D.shape}", flush=True)
    for f in LINKS:
        src, dst = os.path.join(SRC, "learn", f), os.path.join(L.OUT, f)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
        print(f"link {f}: {'ok' if os.path.exists(dst) else 'source not there yet'}", flush=True)


if __name__ == "__main__":
    main()

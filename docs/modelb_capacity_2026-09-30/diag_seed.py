"""Post-hoc diagnostic (results.md §3), not part of the verdict: is mix255's gain capacity, or only averaging?

    .venv/bin/python docs/modelb_capacity_2026-09-30/diag_seed.py BASE_ID SEED_ID

mixseed averages the production model B with itself under three other seeds (six seeds in all). If mixseed matches
mix255, the gain is more averaging and a seed count would buy it; if mix255 beats mixseed, the 255-leaf model adds
something the production model does not have under any seed. Needs splice_eval.py's outputs/mb_splice/ first."""

import subprocess
import sys

import numpy as np
import polars as pl
from splice_eval import LABEL, OUT, PROXY, WORK, boot_diff, oof, rmse


def main(base: str, seed_id: str) -> None:
    feats = pl.read_parquet(WORK / "feat_train.parquet", columns=["MVT_ID_mvt", LABEL, PROXY, "ADEP_mvt"])
    d = feats.join(oof(base, "p_base"), on="MVT_ID_mvt", how="inner").filter(pl.col("p_base").is_not_null())
    d = d.join(oof(seed_id, "p_seed"), on="MVT_ID_mvt", how="left")
    d = d.join(pl.read_parquet(OUT / "oof_mbs_mix255.parquet").rename({"oof": "p_mix255"}), on="MVT_ID_mvt")
    unm = d[PROXY].is_null().to_numpy()
    pb, ps = d["p_base"].to_numpy(), d["p_seed"].to_numpy()
    if np.isnan(ps[unm]).any():
        raise AssertionError("seed arm is missing unmatched rows the base scored")
    mixseed = np.where(unm, 0.5 * (pb + ps), pb)
    pl.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "oof": mixseed}).write_parquet(OUT / "oof_mbs_mixseed.parquet")

    y = d[LABEL].to_numpy().astype(float)
    clean = unm & np.isfinite(y) & ~(y > 43_200)
    pm = d["p_mix255"].to_numpy()
    print(f"unmatched seed-to-seed spread: mean |seed - base| = {np.abs(ps[unm] - pb[unm]).mean():.1f} s")
    for label, sel in (
        ("all unm", unm),
        ("clean", clean),
        ("clean LIRF", clean & (d["ADEP_mvt"] == "LIRF").to_numpy()),
    ):
        for name, (pa, pz) in (("base->mixseed", (pb, mixseed)), ("mixseed->mix255", (mixseed, pm))):
            diff, lo, hi = boot_diff(y[sel], pa[sel], pz[sel])
            print(
                f"{label:>10} {name:>16} n={int(sel.sum()):>6,} {rmse(y[sel], pa[sel]):8.1f} -> "
                f"{rmse(y[sel], pz[sel]):8.1f} {diff:+8.2f} [{lo:+8.2f},{hi:+8.2f}]"
            )
    for a, b in (("mbs_base", "mbs_mixseed"), ("mbs_mixseed", "mbs_mix255")):
        print(f"\n######## prc.compare {a} {b}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "prc.compare",
                a,
                b,
                "--work-dir",
                str(WORK),
                "--exp-dir",
                str(OUT),
                "--max-listed",
                "10",
            ],
            check=True,
        )


if __name__ == "__main__":
    main(*sys.argv[1:3])

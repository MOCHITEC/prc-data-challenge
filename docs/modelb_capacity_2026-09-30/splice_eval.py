"""Model B capacity re-test: splice each arm's unmatched predictions into the base OOF and apply plan.md §2.

    .venv/bin/python docs/modelb_capacity_2026-09-30/splice_eval.py BASE_ID CTL_ID B255_ID MD5_ID

Every arm runs model A for 10 rounds only, so its matched rows are thrown away: the matched rows of every spliced
file are the base's, and only the unmatched rows differ. The control arm (production model B) must reproduce the
base's unmatched predictions, or nothing below is measured. Writes the spliced files to outputs/mb_splice/ and
runs prc.compare on them (criterion 1), then the clean-unmatched bootstrap (criterion 2)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

LABEL, PROXY = "TAXITIME_SEC_mvt", "mvt_minus_aobt3"
WORK, EXP, OUT = Path("data/work_flseq"), Path("experiments"), Path("outputs/mb_splice")
N_BOOT = 2000


def oof(exp_id: str, col: str) -> pl.DataFrame:
    return pl.read_parquet(EXP / f"oof_{exp_id}.parquet").rename({"oof": col})


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def boot_diff(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """RMSE(B) - RMSE(A) with a paired row bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    ea, eb = (y - pa) ** 2, (y - pb) ** 2
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    d = np.sqrt(eb[idx].mean(1)) - np.sqrt(ea[idx].mean(1))
    return float(np.sqrt(eb.mean()) - np.sqrt(ea.mean())), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main(base: str, ctl: str, b255: str, md5: str) -> None:
    feats = pl.read_parquet(WORK / "feat_train.parquet", columns=["MVT_ID_mvt", LABEL, PROXY, "ADEP_mvt"])
    d = feats.join(oof(base, "p_base"), on="MVT_ID_mvt", how="inner")
    for name, eid in (("ctl", ctl), ("b255", b255), ("md5", md5)):
        d = d.join(oof(eid, f"p_{name}"), on="MVT_ID_mvt", how="left")
    unm = d[PROXY].is_null().to_numpy()
    for name in ("ctl", "b255", "md5"):
        if d.filter(pl.Series(unm))[f"p_{name}"].null_count():
            raise AssertionError(f"arm {name} is missing unmatched rows the base scored")

    # Validation (plan §1): the control arm is the production model B, so it must give the base's numbers.
    gap = np.abs(d["p_ctl"].to_numpy()[unm] - d["p_base"].to_numpy()[unm])
    print(f"control vs base on {unm.sum():,} unmatched rows: max |diff| = {gap.max():.6g} s")
    if gap.max() > 1e-6:
        sys.exit("VALIDATION FAILED: the control arm does not reproduce the base's model B; nothing is measured")

    OUT.mkdir(parents=True, exist_ok=True)
    pb = d["p_base"].to_numpy()
    arms = {
        "base": pb,
        "b255": np.where(unm, d["p_b255"].to_numpy(), pb),
        "md5": np.where(unm, d["p_md5"].to_numpy(), pb),
    }
    arms["mix255"] = np.where(unm, 0.5 * (pb + arms["b255"]), pb)
    arms["mix5"] = np.where(unm, 0.5 * (pb + arms["md5"]), pb)
    for name, p in arms.items():
        pl.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"], "oof": p}).write_parquet(OUT / f"oof_mbs_{name}.parquet")

    # Criterion 2: unmatched rows without the two corrupted-label populations (evaluation only).
    y = d[LABEL].to_numpy().astype(float)
    labelled = np.isfinite(y)
    clean = unm & labelled & ~(np.abs(y - 86_400) < 5_400) & ~(y > 43_200)
    adep = d["ADEP_mvt"].to_numpy()
    print(
        f"\nunmatched labelled {int((unm & labelled).sum()):,}; clean {int(clean.sum()):,} "
        f"(removed: rollover band {int((unm & (np.abs(y - 86_400) < 5_400)).sum())}, "
        f"> 12 h {int((unm & (y > 43_200)).sum())})"
    )
    print(f"\n{'arm':>7} {'rows':>9} {'n':>7} {'base':>8} {'arm':>8} {'diff':>8} {'CI95':>20}")
    for name in ("b255", "md5", "mix255", "mix5"):
        for label, sel in (
            ("all unm", unm & labelled),
            ("clean", clean),
            ("clean LIRF", clean & (adep == "LIRF")),
            ("clean LFPG", clean & (adep == "LFPG")),
            ("clean oth", clean & (adep != "LIRF") & (adep != "LFPG")),
        ):
            diff, lo, hi = boot_diff(y[sel], pb[sel], arms[name][sel])
            print(
                f"{name:>7} {label:>9} {int(sel.sum()):>7,} {rmse(y[sel], pb[sel]):8.1f} "
                f"{rmse(y[sel], arms[name][sel]):8.1f} {diff:+8.2f} [{lo:+8.2f},{hi:+8.2f}]"
            )

    # Criterion 1: the repo's gate on the spliced files.
    for name in ("b255", "md5", "mix255", "mix5"):
        print(f"\n######## prc.compare mbs_base mbs_{name}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "prc.compare",
                "mbs_base",
                f"mbs_{name}",
                "--work-dir",
                str(WORK),
                "--exp-dir",
                str(OUT),
                "--max-listed",
                "25",
            ],
            check=True,
        )


if __name__ == "__main__":
    main(*sys.argv[1:5])

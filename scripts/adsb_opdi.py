"""OPDI stand-exit columns for the learned ADS-B residual (handoff/2026-10-05/README.md, docs/opdi_events_2026-10-05/).

    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_cv.py --base opt                  # control (v33's run)
    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_cv.py --base opt --opdi-shuffle   # noise control
    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_cv.py --base opt --opdi           # the arm
    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_apply.py --base opt --opdi        # only if adopted

`--opdi` adds six columns to the residual's design: the OPDI `exit-parking_position` event linked to the departure
(MVT - exit, exit - AOBT_3, stand match, number of candidate exits, OPDI's time-in-use) and MVT - exit - base. They
are read from handoff/2026-10-05/opdi_exitpark.parquet (built by docs/opdi_events_2026-10-05/link_events.py; no label
in it) and kept only at APTS: the residual is fitted on Jan + Jul 2025, where OPDI covers LSZH (58 % / 79 % of
departures) and almost nothing else, so elsewhere (EDDF 15-25 % in 2026) the trees would extrapolate.

`--opdi-shuffle` adds the same six columns with the OPDI block permuted across the rows that have one, within each
airport: same column count, same marginals and the same rows filled, no per-flight signal. It is the control for
the column-sampling lottery (feature_fraction 0.8). Both arms' S4 compare with the same base WITHOUT the columns
(cv_rows_<base>.parquet), so the plain `--base <b>` run must exist; `report()` also compares --opdi with the shuffle.

LICENCE FLAG: OPDI is published by EUROCONTROL under its general disclaimer (attribution, non-commercial, no
modification without permission). The prize rule asks for additional datasets "openly available under an open source
license"; whether OPDI qualifies is the user's call. The leaderboard is not affected."""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

MODE = "opdi" if "--opdi" in sys.argv else "opdishuf" if "--opdi-shuffle" in sys.argv else ""
SFX = f"_{MODE}" if MODE else ""
APTS = ("LSZH",)
COLS = ("opdi_exit_est_s", "opdi_exit_minus_aobt3_s", "opdi_exit_stand_match", "opdi_exit_n", "opdi_exit_in_use_s")
FEATS = [*COLS, "res_opdi_exit"]
SEED = 20261005
N_BOOT = 2000


def table(root: str) -> pl.DataFrame:
    path = os.environ.get("PRC_OPDI_EXIT", root + "handoff/2026-10-05/opdi_exitpark.parquet")
    t = pl.read_parquet(path, columns=["MVT_ID_mvt", *COLS])
    assert t["MVT_ID_mvt"].n_unique() == t.height, "one OPDI row per departure"
    return t


def add(D: pl.DataFrame, root: str) -> pl.DataFrame:
    """Left-join the OPDI columns by MVT_ID_mvt, null outside APTS; under --opdi-shuffle permute the block."""
    D = D.drop([c for c in COLS if c in D.columns]).join(table(root), on="MVT_ID_mvt", how="left")
    keep = pl.col("ADEP_mvt").is_in(list(APTS))
    D = D.with_columns([pl.when(keep).then(pl.col(c)).otherwise(None).cast(pl.Float64).alias(c) for c in COLS])
    if MODE != "opdishuf":
        return D
    rng = np.random.default_rng(SEED)
    vals = {c: D[c].to_numpy().astype(np.float64) for c in COLS}
    filled = D["opdi_exit_est_s"].is_not_null().to_numpy()
    apt = D["ADEP_mvt"].to_numpy()
    for a in APTS:
        idx = np.flatnonzero(filled & (apt == a))
        src = idx[rng.permutation(len(idx))]
        for c in COLS:
            vals[c][idx] = vals[c][src]
    return D.with_columns([pl.Series(c, vals[c]).fill_nan(None) for c in COLS])


def design(X: pl.DataFrame, feats: list[str], base_col: str = "b") -> tuple[pl.DataFrame, list[str]]:
    X = X.with_columns((pl.col("opdi_exit_est_s") - pl.col(base_col).cast(pl.Float64)).alias("res_opdi_exit"))
    return X, feats + FEATS


def _paired(y, pa, pb, day_idx, n_days):
    """RMSE(b) - RMSE(a) and its day-block bootstrap 95 % CI (negative = b better)."""
    ea, eb = (y - pa) ** 2, (y - pb) ** 2
    sa = np.bincount(day_idx, weights=ea, minlength=n_days)
    sb = np.bincount(day_idx, weights=eb, minlength=n_days)
    cnt = np.bincount(day_idx, minlength=n_days).astype(float)
    pick = np.random.default_rng(SEED).integers(0, n_days, size=(N_BOOT, n_days))
    c = cnt[pick].sum(1)
    d = np.sqrt(sb[pick].sum(1) / c) - np.sqrt(sa[pick].sum(1) / c)
    return float(np.sqrt(eb.mean()) - np.sqrt(ea.mean())), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def report(D: pl.DataFrame, p_new: np.ndarray, out_dir: str, base_sfx: str) -> dict:
    """Paired comparisons of this run (b + c) with the same base's run without the columns and, for --opdi, with
    the --opdi-shuffle run: all rows, APTS rows, APTS rows with an OPDI exit. Pre-registered decision (README):
    ADOPT when the all-row CI excludes zero on the improving side against BOTH references."""
    refs = {"plain": os.path.join(out_dir, f"cv_rows{base_sfx}.parquet")}
    if MODE == "opdi":
        refs["shuffle"] = os.path.join(out_dir, f"cv_rows{base_sfx}_opdishuf.parquet")
    y = D["y"].to_numpy()
    day_idx = D["day_idx"].to_numpy()
    n_days = int(day_idx.max()) + 1
    at = D["ADEP_mvt"].is_in(list(APTS)).to_numpy()
    filled = D["opdi_exit_est_s"].is_not_null().to_numpy()
    print(f"\nOPDI ({MODE}): rows with the columns {int(filled.sum()):,} of {D.height:,}; "
          f"at {'/'.join(APTS)} {int((filled & at).sum()):,} of {int(at.sum()):,}")
    out, ok = {}, {}
    for name, path in refs.items():
        if not os.path.exists(path):
            print(f"  vs {name}: {path} missing - run it first")
            ok[name] = None
            continue
        R = pl.read_parquet(path).select("MVT_ID_mvt", (pl.col("b") + pl.col("c_learned")).alias("p_ref"))
        p_ref = D.select("MVT_ID_mvt").join(R, on="MVT_ID_mvt", how="left", maintain_order="left")["p_ref"].to_numpy()
        assert np.isfinite(p_ref).all(), f"{path} does not cover this run's rows"
        out[name] = {}
        for label, m in (("all rows", np.ones(len(y), bool)), ("/".join(APTS), at), ("with an OPDI exit", filled)):
            dd, lo, hi = _paired(y[m], p_ref[m], p_new[m], day_idx[m], n_days)
            out[name][label] = {"n": int(m.sum()), "d_rmse": dd, "lo": lo, "hi": hi}
            print(f"  vs {name:7s} {label:>18s} n={int(m.sum()):>7,}  RMSE diff {dd:+.3f} s  day-CI95 [{lo:+.3f}, {hi:+.3f}]")
        ok[name] = out[name]["all rows"]["hi"] < 0
    if MODE == "opdi":
        verdict = "ADOPT" if ok.get("plain") and ok.get("shuffle") else "INCOMPLETE" if None in ok.values() else "REJECT"
        print(f"DECISION (handoff/2026-10-05/README.md): vs plain {ok.get('plain')}, vs shuffle {ok.get('shuffle')} "
              f"-> {verdict}")
        out["verdict"] = verdict
    return out


def parity(D: pl.DataFrame, G: pl.DataFrame) -> None:
    """Share of ADS-B rows at APTS with an OPDI exit: fitting rows (2025) vs the 2026 rows it is applied to."""
    for a in APTS:
        d = D.filter(pl.col("ADEP_mvt") == a)
        g = G.filter(pl.col("ADEP_mvt") == a).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
        s = {mo: round(float(x["opdi_exit_est_s"].is_not_null().mean()), 3) for (mo,), x in g.group_by("mo")}
        print(f"OPDI parity {a}: 2025 fit rows {d['opdi_exit_est_s'].is_not_null().mean():.3f} (n {d.height:,}); "
              f"2026 by month {dict(sorted(s.items()))} (n {g.height:,})")

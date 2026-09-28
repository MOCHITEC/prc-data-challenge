"""Diagnosis for the airport-qualified categorical levels candidate.

`STAND_mvt` and `RUNWAY_mvt` are encoded as global categoricals in `prc.train._matrix`: the Enum is built
over `sorted(s.unique())` of the *name*, so "A12" at EDDF and "A12" at LFPG share one integer code. This
script measures (a) how much pooling there is, (b) how much of it is genuinely new information the model
does not already have, and (c) what qualifying the level would cost in level count / rows-per-level /
unseen levels on the scored frame.

Read-only. Touches nothing under outputs/. Sections match the results document.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

TRAIN = ROOT / "data" / "work" / "feat_train.parquet"
RANK = ROOT / "data" / "work" / "feat_rank.parquet"
LABEL = "TAXITIME_SEC_mvt"
QUAL = ("STAND_mvt", "RUNWAY_mvt")
SEP = "‖"  # the qualifier separator, ADEP{SEP}NAME


def _say(x) -> None:
    print(x, flush=True)


# ---------------------------------------------------------------- §1


def s1_collisions(tr: pl.DataFrame) -> dict:
    """The four motivating numbers, recomputed from the cached feature frame."""
    out = {"n_rows": tr.height}
    for col in QUAL:
        pairs = tr.select(["ADEP_mvt", col]).unique()
        per_name = pairs.group_by(col).agg(pl.col("ADEP_mvt").n_unique().alias("n_apt"))
        names = per_name.filter(pl.col(col).is_not_null())
        multi = names.filter(pl.col("n_apt") > 1)[col].to_list()
        rows_multi = tr.filter(pl.col(col).is_in(multi)).height if multi else 0
        out[col] = {
            "n_names": names.height,
            "n_names_multi_airport": len(multi),
            "rows_multi_airport": rows_multi,
            "frac_rows_multi_airport": rows_multi / tr.height,
            "max_airports_per_name": int(names["n_apt"].max()) if names.height else 0,
            "worst_names": names.sort("n_apt", descending=True).head(5).to_dicts(),
            "n_qualified_levels": pairs.filter(pl.col(col).is_not_null()).height,
        }
    return out


# ---------------------------------------------------------------- §2


def s2_label_distributions(tr: pl.DataFrame, min_n: int = 30) -> dict:
    """How much do the colliding (airport, name) groups actually disagree about the label?

    A pooled code is only harmful to the extent the pooled groups have different label distributions. The
    measure is a one-way decomposition WITHIN each colliding name: between-airport sum of squares over the
    total sum of squares of that name's rows, and the spread of group means in seconds.
    """
    lab = tr.filter(pl.col(LABEL).is_not_null() & (pl.col(LABEL) > 0) & (pl.col(LABEL) < 7200))
    res = {}
    for col in QUAL:
        g = (
            lab.group_by([col, "ADEP_mvt"])
            .agg([pl.col(LABEL).mean().alias("m"), pl.col(LABEL).var().alias("v"), pl.len().alias("n")])
            .filter(pl.col(col).is_not_null() & (pl.col("n") >= min_n))
        )
        multi = g.group_by(col).agg(pl.col("ADEP_mvt").n_unique().alias("k")).filter(pl.col("k") > 1)
        g = g.join(multi.select(col), on=col, how="inner")
        if g.height == 0:
            res[col] = {"n_names": 0}
            continue
        rows = []
        for name, sub in g.group_by(col):
            n = sub["n"].to_numpy().astype(float)
            m = sub["m"].to_numpy().astype(float)
            v = np.nan_to_num(sub["v"].to_numpy().astype(float))
            gm = float((n * m).sum() / n.sum())
            ssb = float((n * (m - gm) ** 2).sum())
            ssw = float(((n - 1) * v).sum())
            rows.append(
                {
                    "name": name[0] if isinstance(name, tuple) else name,
                    "k_airports": len(n),
                    "n": int(n.sum()),
                    "mean_spread_s": float(m.max() - m.min()),
                    "sd_between_s": float(np.sqrt(ssb / n.sum())),
                    "eta2": ssb / (ssb + ssw) if (ssb + ssw) > 0 else 0.0,
                    "ssb": ssb,
                    "sst": ssb + ssw,
                }
            )
        d = pl.DataFrame(rows)
        w = d["n"].to_numpy().astype(float)
        res[col] = {
            "n_names": d.height,
            "n_rows": int(w.sum()),
            "row_weighted_eta2": float(d["ssb"].sum() / d["sst"].sum()),
            "row_weighted_mean_spread_s": float((d["mean_spread_s"].to_numpy() * w).sum() / w.sum()),
            "median_mean_spread_s": float(d["mean_spread_s"].median()),
            "p90_mean_spread_s": float(d["mean_spread_s"].quantile(0.90)),
            "frac_rows_spread_gt_60s": float(w[(d["mean_spread_s"].to_numpy() > 60)].sum() / w.sum()),
            "frac_rows_spread_gt_120s": float(w[(d["mean_spread_s"].to_numpy() > 120)].sum() / w.sum()),
            "worst": d.sort("mean_spread_s", descending=True).head(8).drop(["ssb", "sst"]).to_dicts(),
        }
    return res


# ---------------------------------------------------------------- §3


def s3_already_encoded(tr: pl.DataFrame, min_count: int = 30) -> dict:
    """How much of the collision is already resolved by the label-derived tables?

    `reference_table` is keyed on (ADEP, STAND, RUNWAY) and `airport_runway_hour_table` on
    (ADEP, RUNWAY, hour) - both already airport-qualified. A row whose (ADEP, STAND, RUNWAY) cell clears
    `reference_min_count` is handed `ref_p10/p20/med` that ALREADY separate the two airports' A12. The
    residual is the rows where that table is missing and the tree has only the pooled code.
    """
    from prc.train import reference_table

    lab = tr.filter(pl.col(LABEL).is_not_null())
    ref = reference_table(lab, min_count)
    j = tr.join(ref, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left", maintain_order="left")
    has_ref = j["ref_n"].is_not_null()
    out = {"ref_rows": ref.height, "frac_rows_with_ref": float(has_ref.mean())}
    for col in QUAL:
        pairs = tr.select(["ADEP_mvt", col]).unique()
        per_name = pairs.group_by(col).agg(pl.col("ADEP_mvt").n_unique().alias("n_apt"))
        multi = per_name.filter(pl.col(col).is_not_null() & (pl.col("n_apt") > 1))[col].to_list()
        m = j[col].is_in(multi).fill_null(False)
        n_multi = int(m.sum())
        out[col] = {
            "rows_colliding": n_multi,
            "frac_colliding_with_ref": float((m & has_ref).sum() / n_multi) if n_multi else 0.0,
            "frac_all_rows_colliding_and_no_ref": float((m & ~has_ref).sum() / tr.height),
        }
    return out


# ---------------------------------------------------------------- §4


def s4_level_budget(tr: pl.DataFrame, rk: pl.DataFrame) -> dict:
    """Cost side: level counts, rows-per-level, and unseen levels on the scored frame.

    LightGBM's categorical split is over code values and is bounded by max_cat_to_onehot / max_cat_threshold;
    more levels means a larger split search and fewer rows behind each level. Unseen levels matter because
    `_matrix` builds the Enum over the STACKED train+target frame, so a 2026 level absent from 2025 gets a
    code of its own that no training row carries - it lands wherever the learned split's default sends it.
    """
    out = {}
    for col in QUAL:
        cur_tr = tr[col].cast(pl.Utf8).fill_null("__NA__").n_unique()
        q_tr = (
            tr.select(pl.col("ADEP_mvt").cast(pl.Utf8) + SEP + pl.col(col).cast(pl.Utf8).fill_null("__NA__"))
            .to_series()
            .n_unique()
        )
        tr_names = set(tr[col].cast(pl.Utf8).fill_null("__NA__").unique().to_list())
        rk_names = set(rk[col].cast(pl.Utf8).fill_null("__NA__").unique().to_list())
        tr_q = set(
            tr.select(pl.col("ADEP_mvt").cast(pl.Utf8) + SEP + pl.col(col).cast(pl.Utf8).fill_null("__NA__"))
            .to_series()
            .unique()
            .to_list()
        )
        rk_qs = rk.select(
            (pl.col("ADEP_mvt").cast(pl.Utf8) + SEP + pl.col(col).cast(pl.Utf8).fill_null("__NA__")).alias("q")
        )["q"]
        unseen_plain = rk[col].cast(pl.Utf8).fill_null("__NA__").is_in(list(tr_names)).not_()
        unseen_qual = rk_qs.is_in(list(tr_q)).not_()
        counts = tr.group_by(["ADEP_mvt", col]).len()["len"].to_numpy()
        plain_counts = tr.group_by(col).len()["len"].to_numpy()
        out[col] = {
            "levels_plain": int(cur_tr),
            "levels_qualified": int(q_tr),
            "level_multiplier": q_tr / cur_tr,
            "median_rows_per_level_plain": float(np.median(plain_counts)),
            "median_rows_per_level_qualified": float(np.median(counts)),
            "qual_levels_under_30_rows": int((counts < 30).sum()),
            "frac_train_rows_in_qual_levels_under_30": float(counts[counts < 30].sum() / tr.height),
            "rank_rows": rk.height,
            "rank_unseen_plain": int(unseen_plain.sum()),
            "rank_frac_unseen_plain": float(unseen_plain.mean()),
            "rank_unseen_qualified": int(unseen_qual.sum()),
            "rank_frac_unseen_qualified": float(unseen_qual.mean()),
            "rank_names_absent_from_train": len(rk_names - tr_names),
        }
    return out


# ---------------------------------------------------------------- §5


def s5_determinism(tr: pl.DataFrame, n: int = 200_000) -> dict:
    """The G11 guard, applied to the qualified encoding: the code must be a function of the data alone.

    Prints a hash of the Enum level list and of the resulting physical codes on a sample, so two processes
    can be compared. `_matrix` already rebuilds the Enum over sorted(unique) - the qualified variant must do
    the same over the qualified strings, NOT over a per-frame categorical cast.
    """
    import hashlib

    sub = tr.head(n)
    res = {}
    for col in QUAL:
        q = pl.col("ADEP_mvt").cast(pl.Utf8).fill_null("__NA__") + SEP + pl.col(col).cast(pl.Utf8).fill_null("__NA__")
        s = sub.select(q.alias("q"))["q"]
        levels = sorted(s.unique().to_list())
        codes = s.cast(pl.Enum(levels)).to_physical().to_numpy()
        res[col] = {
            "n_levels": len(levels),
            "levels_sha1": hashlib.sha1("\n".join(levels).encode()).hexdigest()[:12],
            "codes_sha1": hashlib.sha1(codes.astype("int64").tobytes()).hexdigest()[:12],
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="1,2,3,4,5")
    ap.add_argument("--oof", default=str(ROOT / "experiments" / "oof_main_gap-62dd7c6e88.parquet"))
    ap.add_argument("--seed", type=int, default=0, help="noise-arm membership seed (section 6)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    want = {s.strip() for s in a.sections.split(",")}

    need_rank = "4" in want
    cols = sorted({"ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", LABEL, "hour_local", "MVT_TIME_UTC_mvt"})
    sc = pl.scan_parquet(TRAIN)
    have = set(sc.collect_schema().names())
    tr = sc.select([c for c in cols if c in have]).collect()
    rk = (
        pl.scan_parquet(RANK).select([c for c in ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"] if c in have]).collect()
        if need_rank
        else None
    )
    _say(f"train rows {tr.height:,}   rank rows {rk.height if rk is not None else '-'}")

    res = {}
    if "1" in want:
        res["s1_collisions"] = s1_collisions(tr)
    if "2" in want:
        res["s2_label_distributions"] = s2_label_distributions(tr)
    if "3" in want:
        res["s3_already_encoded"] = s3_already_encoded(tr)
    if "4" in want:
        res["s4_level_budget"] = s4_level_budget(tr, rk)
    if "5" in want:
        res["s5_determinism"] = s5_determinism(tr)
    if "6" in want:
        res["s6_residual_recovery"] = s6_residual_recovery(Path(a.oof), a.seed)

    txt = json.dumps(res, indent=2, default=str)
    _say(txt)
    if a.out:
        Path(a.out).write_text(txt)


# ---------------------------------------------------------------- §6


def _cross_fitted_sse_gain(resid: np.ndarray, key: np.ndarray, mvt: np.ndarray, n_fold: int = 2) -> dict:
    """SSE removed by giving each level of `key` its own intercept, estimated honestly.

    The in-sample version (subtract each group's own mean residual) is what a level "could" buy if the group
    mean were known exactly; with 2,722 groups over 664k rows it is optimistic by roughly n_groups * sigma^2.
    The cross-fitted version estimates each group's mean on the other half and applies it here, so a level
    that carries nothing scores ~0 (in fact slightly negative). Both are reported: the first is the ceiling,
    the second is the estimate.
    """
    h = (mvt.astype(np.int64) * np.int64(2654435761)) % n_fold
    sse0 = float((resid**2).sum())
    # in-sample
    order = np.argsort(key, kind="stable")
    ks, rs = key[order], resid[order]
    bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    n_g = np.diff(np.r_[bounds, len(ks)]).astype(float)
    s_g = np.add.reduceat(rs, bounds)
    sse_in = sse0 - float((s_g**2 / n_g).sum())
    # cross-fitted
    pred = np.zeros_like(resid)
    uniq, inv = np.unique(key, return_inverse=True)
    for f in range(n_fold):
        other = h != f
        cnt = np.bincount(inv[other], minlength=len(uniq)).astype(float)
        tot = np.bincount(inv[other], weights=resid[other], minlength=len(uniq))
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
        pred[~other] = mean[inv[~other]]
    sse_cf = float(((resid - pred) ** 2).sum())
    n = len(resid)
    return {
        "n": n,
        "n_levels": len(uniq),
        "rmse_base": float(np.sqrt(sse0 / n)),
        "rmse_insample_ceiling": float(np.sqrt(sse_in / n)),
        "rmse_crossfitted": float(np.sqrt(sse_cf / n)),
        "delta_rmse_ceiling": float(np.sqrt(sse_in / n) - np.sqrt(sse0 / n)),
        "delta_rmse_crossfitted": float(np.sqrt(sse_cf / n) - np.sqrt(sse0 / n)),
    }


def s6_residual_recovery(oof_path: Path, seed: int = 0) -> dict:
    """What is left for the airport qualifier to take, on the production model's own residuals.

    The model already carries ADEP_mvt as its own categorical and four label-derived tables keyed on it. If it
    has already recovered the airport distinction, the mean residual inside each (ADEP, STAND) cell of a
    colliding name is ~0 and a per-qualified-level intercept buys nothing. The comparison that matters is
    qualified-level gain MINUS plain-name-level gain: the plain level is already in the model, so only the
    difference is new. A random qualifier with the same level count is carried alongside as the noise floor.
    """
    from prc.train import reference_table

    cols = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", LABEL, "mvt_minus_aobt3"]
    tr = pl.scan_parquet(TRAIN).select(cols).collect()
    ref = reference_table(tr.filter(pl.col(LABEL).is_not_null()), 30).select(
        ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "ref_n"]
    )
    tr = tr.join(ref, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left", maintain_order="left")
    oof = pl.read_parquet(oof_path).with_columns(pl.col("MVT_ID_mvt").cast(tr["MVT_ID_mvt"].dtype))
    j = tr.join(oof, on="MVT_ID_mvt", how="inner", maintain_order="left").filter(
        pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite() & pl.col("oof").is_finite()
    )
    rng = np.random.default_rng(seed)
    out = {}
    for lane, f in (
        ("all", j),
        ("A_matched", j.filter(pl.col("mvt_minus_aobt3").is_not_null())),
        # The only rows where the pooled code is the model's ONLY handle on the stand: the airport-keyed
        # reference table did not clear reference_min_count for them, so nothing else in the feature set
        # separates this airport's A12 from the other one's.
        ("ref_missing", j.filter(pl.col("ref_n").is_null())),
    ):
        resid = (f[LABEL] - f["oof"]).to_numpy().astype(float)
        mvt = f["MVT_ID_mvt"].cast(pl.Int64).to_numpy()
        apt = f["ADEP_mvt"].cast(pl.Utf8).fill_null("__NA__").to_numpy().astype(object)
        lane_out = {"n": len(resid), "rmse": float(np.sqrt((resid**2).mean()))}
        for col in QUAL:
            nm = f[col].cast(pl.Utf8).fill_null("__NA__").to_numpy().astype(object)
            qual = np.char.add(np.char.add(apt.astype(str), SEP), nm.astype(str))
            g_plain = _cross_fitted_sse_gain(resid, nm.astype(str), mvt)
            g_qual = _cross_fitted_sse_gain(resid, qual, mvt)
            # Noise qualifier, the P11 "shuffled" control applied to levels: each plain name is split into
            # exactly as many pseudo-levels as it has real airports, membership drawn from a hash of the row
            # alone. Identical level count and near-identical size profile, zero airport information. It is
            # what isolates "more levels" from "the right levels"; it moves only if splitting a level at all
            # helps, which is the thing the candidate must beat.
            k_true = (
                pl.DataFrame({"nm": nm.astype(str), "apt": apt.astype(str)})
                .group_by("nm")
                .agg(pl.col("apt").n_unique().alias("k"))
            )
            kmap = dict(zip(k_true["nm"].to_list(), k_true["k"].to_list()))
            kv = np.array([kmap[x] for x in nm.astype(str)], dtype=np.int64)
            noise = np.char.add(
                np.char.add(nm.astype(str), "#"), (rng.integers(0, 1 << 30, len(resid)) % kv).astype(str)
            )
            g_noise = _cross_fitted_sse_gain(resid, noise, mvt)
            lane_out[col] = {
                "plain": g_plain,
                "qualified": g_qual,
                "noise_same_level_count": g_noise,
                "new_from_qualifier_ceiling_s": g_qual["delta_rmse_ceiling"] - g_plain["delta_rmse_ceiling"],
                "new_from_qualifier_crossfitted_s": g_qual["delta_rmse_crossfitted"]
                - g_plain["delta_rmse_crossfitted"],
            }
        out[lane] = lane_out
    return out


if __name__ == "__main__":
    main()

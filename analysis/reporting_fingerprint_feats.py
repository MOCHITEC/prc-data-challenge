"""Ranking-observable 'reporting fingerprint' features (docs/analysis/reporting_fingerprint.md §4).

Everything here uses only columns that survive in ranking.parquet: ARR rows keep BLOCK/SCHED/MVT; DEP rows keep
MVT/SCHED/STAND/operator/NM clocks. DEP BLOCK/TAXITIME are never read. Features are computed per split
(train = 2025, rank = 2026) from that split's own ARR rows, exactly as they would be at scoring time.

Output: CACHE/fp_feats.parquet keyed by MVT_ID_mvt (DEP rows of both splits).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from reporting_fingerprint_common import CACHE, load_all

COPY_TOL = 6  # the copy spike at minute-grid airports is |d| <= 6 s (see §2 of the report)
OP = "AIRCRAFT_OPERATOR_flt"


def _win_rate(anchor: np.ndarray, t: np.ndarray, flag: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray]:
    o = np.argsort(t, kind="stable")
    t, flag = t[o], flag[o]
    cs = np.concatenate([[0.0], np.cumsum(flag)])
    lo = np.searchsorted(t, anchor - w, side="left")
    hi = np.searchsorted(t, anchor + w, side="right")
    n = (hi - lo).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)
    return r, n


def build(refresh: bool = False) -> pl.DataFrame:
    out = CACHE / "fp_feats.parquet"
    if out.exists() and not refresh:
        return pl.read_parquet(out)
    df = load_all()
    ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)  # noqa: E731
    arr = df.filter(pl.col("PHASE_mvt") == "ARR").with_columns(
        ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= COPY_TOL)
        .cast(pl.Float64).fill_null(0.0).alias("a_copy"),
        ep("BLOCK_TIME_UTC_mvt").alias("a_inb"), ep("MVT_TIME_UTC_mvt").alias("a_land"), ep("SCHED_TIME_UTC_mvt").alias("a_sched"),
        (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("a_delay"),
        pl.col("TAXITIME_SEC_mvt").alias("a_taxi"),
    )
    dep = df.filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", "split", "apt", "date", OP, "STAND_mvt", "FLIGHT_mvt",
        ep("SCHED_TIME_UTC_mvt").alias("d_sched"), ep("MVT_TIME_UTC_mvt").alias("d_mvt"),
    )
    # day-level and day x operator ARR copy rates (whole calendar day, all ARR rows of that day are observable)
    g1 = arr.group_by("split", "apt", "date").agg(pl.col("a_copy").mean().alias("fa_day_rate"), pl.len().alias("fa_day_n"))
    g2 = arr.group_by("split", "apt", "date", OP).agg(pl.col("a_copy").mean().alias("fa_dayop_rate"), pl.len().alias("fa_dayop_n"))
    g3 = arr.group_by("split", "apt", OP).agg(pl.col("a_copy").mean().alias("fa_op_rate_split"), pl.len().alias("fa_op_n_split"))
    g4 = arr.group_by("split", "apt", "date", "STAND_mvt").agg(pl.col("a_copy").mean().alias("fa_daystand_rate"))
    feats = (dep.join(g1, on=["split", "apt", "date"], how="left")
             .join(g2, on=["split", "apt", "date", OP], how="left")
             .join(g3, on=["split", "apt", OP], how="left")
             .join(g4, on=["split", "apt", "date", "STAND_mvt"], how="left"))
    # windows (+-2h at airport, +-3h same operator) and turnaround pairing on the same stand
    parts = []
    for (sp, apt), d in feats.group_by(["split", "apt"], maintain_order=True):
        a = arr.filter((pl.col("split") == sp) & (pl.col("apt") == apt))
        anc = d["d_sched"].to_numpy()
        r, n = _win_rate(anc, a["a_sched"].to_numpy(), a["a_copy"].to_numpy(), 7200)
        cols = {"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy(), "fa_win2h_rate": r, "fa_win2h_n": n}
        # same operator window
        wr = np.full(len(d), np.nan); wn = np.zeros(len(d))
        dop = d[OP].to_numpy()
        for op, idx in d.with_row_index("_i").group_by(OP):
            if op[0] is None:
                continue
            aa = a.filter(pl.col(OP) == op[0])
            if aa.height == 0:
                continue
            ii = idx["_i"].to_numpy()
            rr, nn = _win_rate(anc[ii], aa["a_sched"].to_numpy(), aa["a_copy"].to_numpy(), 3 * 3600)
            wr[ii], wn[ii] = rr, nn
        cols["fa_opwin3h_rate"], cols["fa_opwin3h_n"] = wr, wn
        # turnaround: latest ARR in-block at the same stand before (DEP take-off - 120 s), within 18 h
        ta = {k: np.full(len(d), np.nan) for k in ("ta_copy", "ta_gap_sched", "ta_gap_mvt", "ta_delay", "ta_same_op", "ta_taxi")}
        nx_gap = np.full(len(d), np.nan)
        dm = d["d_mvt"].to_numpy()
        for st, idx in d.with_row_index("_i").group_by("STAND_mvt"):
            if st[0] is None:
                continue
            aa = a.filter(pl.col("STAND_mvt") == st[0]).sort("a_inb")
            if aa.height == 0:
                continue
            ii = idx["_i"].to_numpy()
            tin = aa["a_inb"].to_numpy()
            j = np.searchsorted(tin, dm[ii] - 120, side="right") - 1
            ok = j >= 0
            jj = np.clip(j, 0, None)
            gap_m = dm[ii] - tin[jj]
            ok &= gap_m < 18 * 3600
            sel = ii[ok]; js = jj[ok]
            ta["ta_copy"][sel] = aa["a_copy"].to_numpy()[js]
            ta["ta_gap_sched"][sel] = anc[sel] - tin[js]
            ta["ta_gap_mvt"][sel] = dm[sel] - tin[js]
            ta["ta_delay"][sel] = aa["a_delay"].to_numpy()[js]
            ta["ta_taxi"][sel] = aa["a_taxi"].to_numpy()[js]
            aop = aa[OP].to_numpy()[js]
            ta["ta_same_op"][sel] = (aop == dop[sel]).astype(float)
            # next arrival in-block at this stand after the departure's scheduled off-block: upper bound for off-block
            k = np.searchsorted(tin, anc[ii], side="right")
            okk = k < len(tin)
            nx_gap[ii[okk]] = tin[k[okk]] - anc[ii[okk]]
        cols.update(ta)
        cols["ta_next_inb_minus_sched"] = nx_gap
        parts.append(pl.DataFrame(cols))
    win = pl.concat(parts)
    feats = feats.join(win, on="MVT_ID_mvt", how="left").drop("split", "apt", "date", OP, "STAND_mvt", "FLIGHT_mvt", "d_sched", "d_mvt")
    feats = feats.with_columns([pl.col(c).cast(pl.Float32) for c in feats.columns if c != "MVT_ID_mvt"])
    feats.write_parquet(out)
    return feats


if __name__ == "__main__":
    f = build(refresh=True)
    print(f.shape)
    print(f.describe())


def build_ff(refresh: bool = False) -> pl.DataFrame:
    """Flight-number format fingerprints (report §5). Observable in ranking: FLIGHT_mvt, FLIGHT_ID_mvt null-ness."""
    out = CACHE / "ff_feats.parquet"
    if out.exists() and not refresh:
        return pl.read_parquet(out)
    from reporting_fingerprint_flightfmt import fmt_expr
    df = load_all().filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", "split", "apt", "FLIGHT_mvt", "FLIGHT_ID_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt")
    df = df.with_columns(fmt_expr(), pl.col("FLIGHT_mvt").str.slice(0, 3).alias("fl_pfx"),
                         pl.col("FLIGHT_mvt").str.len_chars().cast(pl.Float32).alias("fl_len"),
                         pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Float64).alias("_m"))
    # same flight number, same airport, same split, leave-self-out: how often is it NM-matched?
    g = pl.col("_m").sum().over("split", "apt", "FLIGHT_mvt") - pl.col("_m")
    n = pl.len().over("split", "apt", "FLIGHT_mvt") - 1
    df = df.with_columns(n.cast(pl.Float32).alias("fl_n_other"),
                         pl.when(n > 0).then(g / n).otherwise(None).cast(pl.Float32).alias("fl_nm_rate_other"))
    df = df.with_columns(pl.col("fl_fmt").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("k_fl_fmt"),
                         pl.col("fl_pfx").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("k_fl_pfx"))
    # duplicate records: same airport, flight number and scheduled time, filed twice (report §6.2)
    key = ["split", "apt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"]
    grp = pl.col("FLIGHT_mvt").is_not_null()
    unm = 1.0 - pl.col("_m")
    df = df.with_columns(
        pl.when(grp).then(pl.len().over(key)).otherwise(1).cast(pl.Float32).alias("dup_n"),
        pl.when(grp).then(unm.sum().over(key) - unm).otherwise(0).cast(pl.Float32).alias("dup_n_unm_twins"),
        pl.when(grp).then(pl.col("_m").sum().over(key) - pl.col("_m")).otherwise(0).cast(pl.Float32).alias("dup_n_m_twins"),
        pl.when(grp).then((pl.col("MVT_TIME_UTC_mvt").max().over(key) - pl.col("MVT_TIME_UTC_mvt").min().over(key)).dt.total_seconds()).otherwise(None).cast(pl.Float32).alias("dup_mvt_span"),
    )
    df = df.select("MVT_ID_mvt", "k_fl_fmt", "k_fl_pfx", "fl_len", "fl_n_other", "fl_nm_rate_other", "dup_n", "dup_n_unm_twins", "dup_n_m_twins", "dup_mvt_span")
    df.write_parquet(out)
    return df

"""Three feature levers from the public-code deep dive, as model columns behind TrainConfig flags.

Pre-registration and results: docs/competitor_research_2026-09-21/results/feature-levers-L1-L4.md. The
definitions are implemented from the written descriptions in
docs/competitor_research_2026-09-21/results/public-code-deep-dive-2026-09-30.md (§1.6, §2.6, §4) and
21-repo-merry-quicksand.md §4.1. No competitor code was read or copied.

  L1  plan_nm_taxi              (TrainConfig.fe_plan_nm_taxi, model A only)
      ARVT_1 - EOBT_1 (the filed gate-to-gate plan) minus its median over (ADEP, ADES, aircraft type), clipped
      to +-3,600 s. The median table is label-free and is fitted once on the training AND ranking departures
      together (the same rule as prc.features.proxy_mode_table: a per-frame table would be a different lookup
      behind the same column name). A group needs >= PLAN_MIN_N rows; otherwise the (ADEP, ADES) median with the
      same minimum; otherwise null.
  L2  fw_{apt,rwy}{15,30}_{mean,n}  (fe_fwd_nm -> model A, fe_fwd_nm_unmatched -> model B)
      Mean and count of OTHER departures' MVT - AOBT_3 (each clipped to +-7,200 s) whose take-off is in
      (t, t + 15 min] and (t, t + 30 min], t = this row's take-off, per departure airport and per departure
      airport x runway. Departures without AOBT_3 are not counted. Mean null when the count is 0.
  L3  qp_*                      (fe_push_queue, model A only). a = AOBT_3, t = MVT_TIME, same departure airport,
      j != i. Null when this row's AOBT_3 is null.
        qp_apt_at_push          #{j: a_j < a_i < t_j}                 pushed before me, not airborne when I push
        qp_rwy_at_push          the same, same runway
        qp_pushed_after_me      #{j: a_i < a_j < t_i}                 pushed while I was taxiing
        qp_rwy_tko_in_taxi      #{j same runway: a_i < t_j < t_i}     take-offs on my runway during my taxi
                                                                      (every departure, with or without AOBT_3)
        qp_rwy_push_pre20       #{j same runway: a_i - 1200 <= a_j < a_i}
        qp_arr_taxiing_at_push  arrivals at this airport with landing <= a_i < in-block (0 < in-block -
                                landing < 7,200 s)

Parity. Only columns that survive in ranking.parquet are read: departures' MVT_TIME, AOBT_3, EOBT_1, ARVT_1,
runway and route; arrivals' landing and in-block times. No departure BLOCK_TIME or TAXITIME is read. Each frame's
context (L2, L3) comes from its own rows, as for prc.features.context_features.

Like prc.arrival_fill, this module is deliberately outside features.py / features_extra.py, which are hashed into
the feature-cache fingerprint: a recipe with every flag off reuses its cache and builds a byte-identical design
matrix."""

from __future__ import annotations

import numpy as np
import polars as pl

PLAN_CLIP_S = 3600.0
PLAN_MIN_N = 5
FW_CLIP_S = 7200.0
FW_WINDOWS_MIN = (15, 30)
PUSH_PRE_S = 1200.0
ARR_TAXI_MAX_S = 7200.0

L1_COLS = ("plan_nm_taxi",)
L2_COLS = tuple(f"fw_{g}{w}_{s}" for g in ("apt", "rwy") for w in FW_WINDOWS_MIN for s in ("mean", "n"))
L3_COLS = (
    "qp_apt_at_push",
    "qp_rwy_at_push",
    "qp_pushed_after_me",
    "qp_rwy_tko_in_taxi",
    "qp_rwy_push_pre20",
    "qp_arr_taxiing_at_push",
)
LEVER_COLS = L1_COLS + L2_COLS + L3_COLS


def _ep(s: pl.Series) -> np.ndarray:
    return s.dt.epoch("s").cast(pl.Float64).to_numpy()


def _departures(raw: pl.DataFrame) -> pl.DataFrame:
    return raw.filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt",
        pl.col("ADEP_mvt").cast(pl.Utf8),
        pl.col("ADES_mvt").cast(pl.Utf8),
        pl.col("AIRCRAFT_TYPE_mvt").cast(pl.Utf8),
        pl.col("RUNWAY_mvt").cast(pl.Utf8),
        "MVT_TIME_UTC_mvt",
        "AOBT_3_flt",
        (pl.col("ARVT_1_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("_plan"),
    )


# ---------- L1 ----------


def plan_median_tables(deps: list[pl.DataFrame]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The (ADEP, ADES, type) and (ADEP, ADES) medians of ARVT_1 - EOBT_1 over the union of `deps`. Label-free."""
    both = pl.concat([d.select("ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "_plan") for d in deps]).drop_nulls("_plan")
    fine = (
        both.drop_nulls(["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt"])
        .group_by("ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt")
        .agg(pl.col("_plan").median().alias("_med3"), pl.len().alias("_n3"))
        .filter(pl.col("_n3") >= PLAN_MIN_N)
        .drop("_n3")
        .sort("ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt")
    )
    coarse = (
        both.drop_nulls(["ADEP_mvt", "ADES_mvt"])
        .group_by("ADEP_mvt", "ADES_mvt")
        .agg(pl.col("_plan").median().alias("_med2"), pl.len().alias("_n2"))
        .filter(pl.col("_n2") >= PLAN_MIN_N)
        .drop("_n2")
        .sort("ADEP_mvt", "ADES_mvt")
    )
    return fine, coarse


def plan_nm_taxi(dep: pl.DataFrame, fine: pl.DataFrame, coarse: pl.DataFrame) -> pl.DataFrame:
    out = (
        dep.select("MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "_plan")
        .join(fine, on=["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt"], how="left", maintain_order="left")
        .join(coarse, on=["ADEP_mvt", "ADES_mvt"], how="left", maintain_order="left")
        .select(
            "MVT_ID_mvt",
            (pl.col("_plan") - pl.coalesce("_med3", "_med2")).clip(-PLAN_CLIP_S, PLAN_CLIP_S).alias("plan_nm_taxi"),
        )
    )
    return out


# ---------- L2 ----------


def _forward_mean(anchor: np.ndarray, times: np.ndarray, values: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and count of non-NaN `values` whose time is in (t, t + w] for each anchor t."""
    ok = np.isfinite(times) & np.isfinite(values)
    order = np.argsort(times[ok], kind="stable")
    t, v = times[ok][order], values[ok][order]
    cs = np.concatenate([[0.0], np.cumsum(v)])
    lo = np.searchsorted(t, anchor, side="right")
    hi = np.searchsorted(t, anchor + w, side="right")
    n = (hi - lo).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, (cs[hi] - cs[lo]) / np.where(n > 0, n, 1.0), np.nan)
    bad = ~np.isfinite(anchor)
    mean[bad], n[bad] = np.nan, np.nan
    return mean, n


# ---------- L3 ----------


def _count_open(sorted_ev: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """#{e: lo < e < hi} per row (0 when hi <= lo)."""
    c = np.searchsorted(sorted_ev, hi, side="left") - np.searchsorted(sorted_ev, lo, side="right")
    return np.maximum(c, 0).astype(np.float64)


def _in_system(a: np.ndarray, t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """#{j: a_j < q < t_j} for each query q, over the pairs (a_j, t_j) with both finite. Exact.

    #{a_j < q} - #{t_j <= q} + #{j: t_j <= q <= a_j}. The last term is non-zero only for pairs with t_j <= a_j
    (a non-positive MVT - AOBT_3), which are few, and is added with a difference array. When q is a row's own
    a_i the row's own contribution cancels, so self is excluded without a special case."""
    ok = np.isfinite(a) & np.isfinite(t)
    a, t = a[ok], t[ok]
    qa = np.sort(a)
    qt = np.sort(t)
    out = np.searchsorted(qa, q, side="left").astype(np.float64) - np.searchsorted(qt, q, side="right")
    neg = t <= a
    if neg.any():
        order = np.argsort(q, kind="stable")
        qs = q[order]
        diff = np.zeros(len(q) + 1)
        lo = np.searchsorted(qs, t[neg], side="left")  # q >= t_j
        hi = np.searchsorted(qs, a[neg], side="right")  # q <= a_j
        np.add.at(diff, lo, 1.0)
        np.add.at(diff, hi, -1.0)
        extra = np.empty(len(q))
        extra[order] = np.cumsum(diff)[:-1]
        out += extra
    return out


def _lever_frame(dep: pl.DataFrame, arr: pl.DataFrame, want_l2: bool, want_l3: bool) -> pl.DataFrame:
    """L2 and/or L3 columns for every departure in `dep`, per departure airport, in `dep`'s row order."""
    cols: dict[str, np.ndarray] = {}
    n = dep.height
    names = (L2_COLS if want_l2 else ()) + (L3_COLS if want_l3 else ())
    for c in names:
        cols[c] = np.full(n, np.nan)
    apt = dep["ADEP_mvt"].to_numpy()
    rwy = dep["RUNWAY_mvt"].to_numpy()
    t_all = _ep(dep["MVT_TIME_UTC_mvt"])
    a_all = _ep(dep["AOBT_3_flt"])
    proxy = np.clip(t_all - a_all, -FW_CLIP_S, FW_CLIP_S)
    a_land = _ep(arr["MVT_TIME_UTC_mvt"]) if arr.height else np.array([])
    a_inb = _ep(arr["BLOCK_TIME_UTC_mvt"]) if arr.height else np.array([])
    a_apt = arr["ADES_mvt"].cast(pl.Utf8).to_numpy() if arr.height else np.array([], dtype=object)
    for ap in sorted({x for x in apt if x is not None}):
        ia = np.flatnonzero(apt == ap)
        t, a, v = t_all[ia], a_all[ia], proxy[ia]
        if want_l2:
            for w in FW_WINDOWS_MIN:
                m, k = _forward_mean(t, t, v, w * 60.0)
                cols[f"fw_apt{w}_mean"][ia], cols[f"fw_apt{w}_n"][ia] = m, k
        if want_l3:
            has_a = np.isfinite(a)
            q = np.where(has_a, a, np.nan)
            cols["qp_apt_at_push"][ia] = np.where(has_a, _in_system(a, t, np.where(has_a, a, 0.0)), np.nan)
            a_sorted = np.sort(a[has_a])
            cols["qp_pushed_after_me"][ia] = np.where(has_a, _count_open(a_sorted, q, t), np.nan)
            sel = (a_apt == ap) if len(a_apt) else np.zeros(0, bool)
            land, inb = a_land[sel], a_inb[sel]
            ok = np.isfinite(land) & np.isfinite(inb) & (inb > land) & (inb - land < ARR_TAXI_MAX_S)
            sl, sb = np.sort(land[ok]), np.sort(inb[ok])
            taxiing = np.searchsorted(sl, q, side="right") - np.searchsorted(sb, q, side="right")
            cols["qp_arr_taxiing_at_push"][ia] = np.where(has_a, taxiing, np.nan)
        rw = rwy[ia]
        for r in sorted({x for x in rw if x is not None}):
            ir = np.flatnonzero(rw == r)
            gi = ia[ir]
            tr, ar, vr = t[ir], a[ir], v[ir]
            if want_l2:
                for w in FW_WINDOWS_MIN:
                    m, k = _forward_mean(tr, tr, vr, w * 60.0)
                    cols[f"fw_rwy{w}_mean"][gi], cols[f"fw_rwy{w}_n"][gi] = m, k
            if want_l3:
                has_a = np.isfinite(ar)
                q = np.where(has_a, ar, 0.0)
                cols["qp_rwy_at_push"][gi] = np.where(has_a, _in_system(ar, tr, q), np.nan)
                ts = np.sort(tr[np.isfinite(tr)])
                cols["qp_rwy_tko_in_taxi"][gi] = np.where(has_a, _count_open(ts, q, tr), np.nan)
                asr = np.sort(ar[has_a])
                pre = np.searchsorted(asr, q, side="left") - np.searchsorted(asr, q - PUSH_PRE_S, side="left")
                cols["qp_rwy_push_pre20"][gi] = np.where(has_a, pre, np.nan)
    frame = pl.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"], **{c: cols[c] for c in names}})
    return frame.with_columns([pl.col(c).cast(pl.Float32).fill_nan(None) for c in names])


def attach_levers(
    f_tr: pl.DataFrame,
    f_rk: pl.DataFrame,
    raw_tr: pl.DataFrame,
    raw_rk: pl.DataFrame,
    l1: bool,
    l2: bool,
    l3: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The two feature frames with the requested lever columns appended (in LEVER_COLS order), each frame's
    context taken from its own raw rows and L1's median table fitted on both frames' departures together."""
    if not (l1 or l2 or l3):
        return f_tr, f_rk
    d_tr, d_rk = _departures(raw_tr), _departures(raw_rk)
    fine = coarse = None
    if l1:
        fine, coarse = plan_median_tables([d_tr, d_rk])
    out = []
    for feats, dep, raw in ((f_tr, d_tr, raw_tr), (f_rk, d_rk, raw_rk)):
        if feats["MVT_ID_mvt"].n_unique() != feats.height:
            raise AssertionError("MVT_ID_mvt is not unique in the feature frame")
        clash = [c for c in LEVER_COLS if c in feats.columns]
        if clash:
            raise ValueError(f"the feature frame already has {clash}")
        parts = []
        if l1:
            parts.append(plan_nm_taxi(dep, fine, coarse))
        if l2 or l3:
            arr = raw.filter(pl.col("PHASE_mvt") == "ARR").select("ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt")
            parts.append(_lever_frame(dep, arr, l2, l3))
        g = feats.select("MVT_ID_mvt")
        for p in parts:
            g = g.join(p, on="MVT_ID_mvt", how="left", maintain_order="left")
        if g.height != feats.height or not g["MVT_ID_mvt"].equals(feats["MVT_ID_mvt"]):
            raise AssertionError("attach_levers changed the row count or order")
        g = g.select([c for c in LEVER_COLS if c in g.columns])
        out.append(feats.hstack(g))
    return out[0], out[1]

"""Movement (token) tables and neighbour indices for NbrNet (PREREG.md §1-§2).

Leakage rules enforced here:
  R1  the tables are built from `prc.features.mask_departures(raw)`: every departure's BLOCK_TIME / TAXITIME is
      null before any field is computed, and no departure field reads them.
  R2  arrival tokens read the arrival's own in-block time and taxi-in time (both present in ranking.parquet).
  R3  a departure's own table row is never in its departure index.
  R4  every token field is a row-local clock difference, flag or one-hot: nothing fitted, nothing label-derived.
  R6  neighbours come from the same airport AND the same source file (training month); `cov` is the share of the
      window inside that file's movement-time span at that airport.
  R7  window [t0 - 60 min, t0 + 20 min] on the neighbour's movement time (departures: take-off MVT; arrivals:
      in-block time, landing time when in-block is null).
  R8  only the organisers' movement columns are read.

Times are int32 seconds since 2025-01-01T00:00Z (exact; float32 would round epoch seconds to 128 s). Index matrices
are int32, padded with -1, and ordered nearest-first by |key - t0|, so truncating columns keeps the K nearest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from prc.features import mask_departures

BASE = datetime(2025, 1, 1, tzinfo=UTC)
WIN_BEFORE_S = 3600  # t0 - 60 min
WIN_AFTER_S = 1200  # t0 + 20 min
K_DEP, K_ARR = 64, 48
NULL_T = np.int32(-(2**31))
SLOG_SCALE = 8.0  # slog(seconds) / 8 keeps an hour's offset near 1
DT_SCALE = 1800.0
WAKE = ("L", "M", "H", "J")  # + one "other or null" column

DEP_STATIC = (
    "proxy_slog",  # slog(MVT - AOBT_3)
    "aobt_eobt_slog",  # slog(AOBT_3 - EOBT_1)
    "mvt_sched_slog",  # slog(MVT - SCHED)
    "mvt_lobt_slog",  # slog(MVT - LOBT)
    *[f"wake_{w}" for w in WAKE],
    "wake_other",
    "matched",
    "null_aobt",
    "null_eobt",
    "null_lobt",
    "null_sched",
    "null_rwy",
    "null_stand",
)
DEP_DYNAMIC = ("dt", "d_aobt_slog", "aobt_order", "same_rwy", "same_stand", "same_prefix")
ARR_STATIC = (
    "taxi_in_slog",  # slog(TAXITIME) of the arrival = in-block - landing
    "inblock_delay_slog",  # slog(BLOCK - SCHED)
    "land_sched_slog",  # slog(MVT - SCHED)
    *[f"wake_{w}" for w in WAKE],
    "wake_other",
    "null_block",
    "null_sched",
    "null_rwy",
    "null_stand",
)
ARR_DYNAMIC = ("dt_land", "dt_inblock", "same_rwy", "same_stand", "same_prefix")
N_DEP_FIELDS = len(DEP_STATIC) + len(DEP_DYNAMIC)
N_ARR_FIELDS = len(ARR_STATIC) + len(ARR_DYNAMIC)


def slog(x: np.ndarray) -> np.ndarray:
    return (np.sign(x) * np.log1p(np.abs(x)) / SLOG_SCALE).astype(np.float32)


def load_movements(files: list[Path]) -> pl.DataFrame:
    """The raw files concatenated, each row tagged with the index of its source file (`src`)."""
    frames = [pl.read_parquet(p).with_columns(pl.lit(i, dtype=pl.Int16).alias("src")) for i, p in enumerate(files)]
    return pl.concat(frames, how="diagonal")


def _secs(col: str) -> pl.Expr:
    """int32 seconds since BASE, null kept."""
    return ((pl.col(col) - pl.lit(BASE)).dt.total_seconds()).cast(pl.Int64)


def stand_prefix_expr(col: str = "STAND_mvt") -> pl.Expr:
    """Leading run of letters; the first character when the name starts with a digit (PREREG §2)."""
    s = pl.col(col).cast(pl.Utf8)
    letters = s.str.extract(r"^([A-Za-z]+)", 1)
    return pl.when(s.is_null()).then(None).otherwise(pl.coalesce([letters, s.str.slice(0, 1)])).alias("_prefix")


@dataclass
class Tables:
    # departures (sorted by airport, src, MVT)
    dep_id: np.ndarray  # float64 MVT_ID_mvt
    dep_t: np.ndarray  # int32 MVT
    dep_aobt: np.ndarray  # int32 AOBT_3 (NULL_T if null)
    dep_static: np.ndarray  # float32 [n, len(DEP_STATIC)]
    dep_rwy: np.ndarray  # int32 codes, -1 null
    dep_stand: np.ndarray
    dep_prefix: np.ndarray
    dep_group: np.ndarray  # int32 (airport, src) group id
    # arrivals (sorted by airport, src, key time)
    arr_t_land: np.ndarray
    arr_t_inblock: np.ndarray  # NULL_T if null
    arr_key: np.ndarray  # in-block, landing when in-block is null
    arr_static: np.ndarray
    arr_rwy: np.ndarray
    arr_stand: np.ndarray
    arr_prefix: np.ndarray
    arr_group: np.ndarray
    groups: list  # [(airport, src)] by group id
    span: np.ndarray  # int64 [n_groups, 2]: min/max movement time of the group (deps and arrivals)

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                np.save(d / f"{k}.npy", v)
        (d / "groups.json").write_text(json.dumps(self.groups))

    @classmethod
    def load(cls, d: Path, mmap: bool = False) -> Tables:
        kw = {}
        for k in cls.__dataclass_fields__:
            if k == "groups":
                kw[k] = [tuple(g) for g in json.loads((d / "groups.json").read_text())]
            else:
                kw[k] = np.load(d / f"{k}.npy", mmap_mode="r" if mmap else None)
        return cls(**kw)


def _codes(values: list[pl.Series]) -> list[np.ndarray]:
    """One string -> int32 dictionary over all given series (sorted levels: a function of the data alone)."""
    levels = sorted(set().union(*[set(s.drop_nulls().cast(pl.Utf8).unique().to_list()) for s in values]))
    lut = {v: i for i, v in enumerate(levels)}
    return [s.cast(pl.Utf8).replace_strict(lut, default=-1, return_dtype=pl.Int32).fill_null(-1).to_numpy() for s in values]


def _wake(df: pl.DataFrame) -> np.ndarray:
    w = df["WK_TBL_CAT_flt"].cast(pl.Utf8)
    cols = [(w == c).fill_null(False).to_numpy() for c in WAKE]
    other = ~np.any(np.stack(cols, 1), axis=1)
    return np.stack([*cols, other], 1).astype(np.float32)


def build_tables(raw: pl.DataFrame) -> Tables:
    """Token tables from the raw movements (with a `src` column). Departure labels are masked first (R1)."""
    m = mask_departures(raw)  # R1: departures' BLOCK_TIME / TAXITIME are null from here on
    tcols = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt"]
    m = m.with_columns([_secs(c).alias(f"_s_{c}") for c in tcols])
    dep = (
        m.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("MVT_TIME_UTC_mvt").is_not_null() & pl.col("ADEP_mvt").is_not_null())
        .with_columns(pl.col("ADEP_mvt").alias("_apt"))
        .sort(["_apt", "src", "_s_MVT_TIME_UTC_mvt", "MVT_ID_mvt"])
    )
    arr = (
        m.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("MVT_TIME_UTC_mvt").is_not_null() & pl.col("ADES_mvt").is_not_null())
        .with_columns(
            pl.col("ADES_mvt").alias("_apt"),
            pl.coalesce([pl.col("_s_BLOCK_TIME_UTC_mvt"), pl.col("_s_MVT_TIME_UTC_mvt")]).alias("_key"),
        )
        .sort(["_apt", "src", "_key", "MVT_ID_mvt"])
    )
    groups = sorted(set(zip(dep["_apt"].to_list(), dep["src"].to_list())) | set(zip(arr["_apt"].to_list(), arr["src"].to_list())))
    gid = {g: i for i, g in enumerate(groups)}

    def group_ids(df):
        return np.array([gid[g] for g in zip(df["_apt"].to_list(), df["src"].to_list())], dtype=np.int32)

    def t32(df, c):
        s = df[c]
        ok = s.is_not_null().to_numpy()
        v = s.fill_null(0).to_numpy().astype(np.int64)
        if np.any(np.abs(v[ok]) > 2**31 - 2):
            raise ValueError(f"{c} out of int32 range")
        return np.where(ok, v, NULL_T).astype(np.int32), ok

    # ---- departures (R1: only masked frame columns; BLOCK/TAXITIME are never read here) ----
    d_t, _ = t32(dep, "_s_MVT_TIME_UTC_mvt")
    d_aobt, _ = t32(dep, "_s_AOBT_3_flt")
    s = {c: dep[f"_s_{c}"].to_numpy().astype(np.float64) for c in ("MVT_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "SCHED_TIME_UTC_mvt")}
    s = {k: np.where(dep[f"_s_{k}"].is_null().to_numpy(), np.nan, v) for k, v in s.items()}

    def diff(a, b):
        x = s[a] - s[b]
        return np.where(np.isfinite(x), slog(np.nan_to_num(x)), 0.0).astype(np.float32)

    null = {k: np.isnan(v) for k, v in s.items()}
    d_static = np.column_stack(
        [
            diff("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
            diff("AOBT_3_flt", "EOBT_1_flt"),
            diff("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
            diff("MVT_TIME_UTC_mvt", "LOBT_flt"),
            _wake(dep),
            (~null["AOBT_3_flt"]).astype(np.float32),
            null["AOBT_3_flt"].astype(np.float32),
            null["EOBT_1_flt"].astype(np.float32),
            null["LOBT_flt"].astype(np.float32),
            null["SCHED_TIME_UTC_mvt"].astype(np.float32),
            dep["RUNWAY_mvt"].is_null().to_numpy().astype(np.float32),
            dep["STAND_mvt"].is_null().to_numpy().astype(np.float32),
        ]
    ).astype(np.float32)
    assert d_static.shape[1] == len(DEP_STATIC)

    # ---- arrivals (R2: their own in-block and taxi-in are allowed) ----
    a_land, _ = t32(arr, "_s_MVT_TIME_UTC_mvt")
    a_inb, _ = t32(arr, "_s_BLOCK_TIME_UTC_mvt")
    a_key, _ = t32(arr, "_key")
    sa = {c: arr[f"_s_{c}"].to_numpy().astype(np.float64) for c in ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")}
    sa = {k: np.where(arr[f"_s_{k}"].is_null().to_numpy(), np.nan, v) for k, v in sa.items()}
    taxi_in = arr["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    taxi_in = np.where(np.isfinite(taxi_in), taxi_in, np.nan)

    def adiff(x):
        return np.where(np.isfinite(x), slog(np.nan_to_num(x)), 0.0).astype(np.float32)

    a_static = np.column_stack(
        [
            adiff(taxi_in),
            adiff(sa["BLOCK_TIME_UTC_mvt"] - sa["SCHED_TIME_UTC_mvt"]),
            adiff(sa["MVT_TIME_UTC_mvt"] - sa["SCHED_TIME_UTC_mvt"]),
            _wake(arr),
            np.isnan(sa["BLOCK_TIME_UTC_mvt"]).astype(np.float32),
            np.isnan(sa["SCHED_TIME_UTC_mvt"]).astype(np.float32),
            arr["RUNWAY_mvt"].is_null().to_numpy().astype(np.float32),
            arr["STAND_mvt"].is_null().to_numpy().astype(np.float32),
        ]
    ).astype(np.float32)
    assert a_static.shape[1] == len(ARR_STATIC)

    d_rwy, a_rwy = _codes([dep["RUNWAY_mvt"], arr["RUNWAY_mvt"]])
    d_st, a_st = _codes([dep["STAND_mvt"], arr["STAND_mvt"]])
    d_px, a_px = _codes([dep.select(stand_prefix_expr())["_prefix"], arr.select(stand_prefix_expr())["_prefix"]])
    dg, ag = group_ids(dep), group_ids(arr)
    span = np.zeros((len(groups), 2), dtype=np.int64)
    span[:, 0], span[:, 1] = np.iinfo(np.int64).max, np.iinfo(np.int64).min
    for g_arr, times in ((dg, d_t), (ag, a_land), (ag, a_key)):
        np.minimum.at(span[:, 0], g_arr, times.astype(np.int64))
        np.maximum.at(span[:, 1], g_arr, times.astype(np.int64))
    return Tables(
        dep_id=dep["MVT_ID_mvt"].to_numpy().astype(np.float64),
        dep_t=d_t,
        dep_aobt=d_aobt,
        dep_static=d_static,
        dep_rwy=d_rwy,
        dep_stand=d_st,
        dep_prefix=d_px,
        dep_group=dg,
        arr_t_land=a_land,
        arr_t_inblock=a_inb,
        arr_key=a_key,
        arr_static=a_static,
        arr_rwy=a_rwy,
        arr_stand=a_st,
        arr_prefix=a_px,
        arr_group=ag,
        groups=[list(g) for g in groups],
        span=span,
    )


def _nearest_in_window(
    q_t: np.ndarray, q_self: np.ndarray | None, times: np.ndarray, offset: int, k: int, chunk: int = 32768
) -> np.ndarray:
    """For sorted `times` (one group), the indices (+offset) of the k entries nearest to each query time inside
    [t - WIN_BEFORE_S, t + WIN_AFTER_S], nearest first (ties: earlier entry first), excluding `q_self` (the
    query's own position in `times`, R3). Padded with -1."""
    n_q = len(q_t)
    out = np.full((n_q, k), -1, dtype=np.int32)
    if n_q == 0 or len(times) == 0:
        return out
    t64 = times.astype(np.int64)
    lo_all = np.searchsorted(t64, q_t.astype(np.int64) - WIN_BEFORE_S, side="left")
    hi_all = np.searchsorted(t64, q_t.astype(np.int64) + WIN_AFTER_S, side="right")
    for s in range(0, n_q, chunk):
        e = min(n_q, s + chunk)
        lo, hi = lo_all[s:e], hi_all[s:e]
        w = int((hi - lo).max()) if e > s else 0
        if w == 0:
            continue
        cand = lo[:, None] + np.arange(w)[None, :]
        valid = cand < hi[:, None]
        if q_self is not None:
            valid &= cand != q_self[s:e, None]
        cc = np.minimum(cand, len(t64) - 1)
        dist = np.abs(t64[cc] - q_t[s:e, None].astype(np.int64)).astype(np.float64)
        # tie-break towards the earlier entry, deterministically: position adds a sub-second epsilon
        dist = np.where(valid, dist + np.arange(w)[None, :] * 1e-7, np.inf)
        kk = min(k, w)
        part = np.argpartition(dist, kk - 1, axis=1)[:, :kk] if w > kk else np.tile(np.arange(w), (e - s, 1))
        pd = np.take_along_axis(dist, part, 1)
        order = np.argsort(pd, axis=1, kind="stable")
        part = np.take_along_axis(part, order, 1)
        pd = np.take_along_axis(pd, order, 1)
        idx = np.take_along_axis(cand, part, 1) + offset
        out[s:e, :kk] = np.where(np.isfinite(pd), idx, -1).astype(np.int32)
    return out


def build_index(tb: Tables, k_dep: int = K_DEP, k_arr: int = K_ARR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(idx_d [n_dep, k_dep], idx_a [n_dep, k_arr], cov [n_dep]) for every departure-table row as the query."""
    n = len(tb.dep_t)
    idx_d = np.full((n, k_dep), -1, dtype=np.int32)
    idx_a = np.full((n, k_arr), -1, dtype=np.int32)
    cov = np.zeros(n, dtype=np.float32)
    d_bounds = _group_bounds(tb.dep_group, len(tb.groups))
    a_bounds = _group_bounds(tb.arr_group, len(tb.groups))
    for g in range(len(tb.groups)):
        ds, de = d_bounds[g]
        if de <= ds:
            continue
        q_t = tb.dep_t[ds:de]
        idx_d[ds:de] = _nearest_in_window(q_t, np.arange(de - ds), tb.dep_t[ds:de], ds, k_dep)
        as_, ae = a_bounds[g]
        if ae > as_:
            idx_a[ds:de] = _nearest_in_window(q_t, None, tb.arr_key[as_:ae], as_, k_arr)
        lo = np.maximum(q_t.astype(np.int64) - WIN_BEFORE_S, tb.span[g, 0])
        hi = np.minimum(q_t.astype(np.int64) + WIN_AFTER_S, tb.span[g, 1])
        cov[ds:de] = np.clip((hi - lo) / float(WIN_BEFORE_S + WIN_AFTER_S), 0.0, 1.0)
    return idx_d, idx_a, cov


def _group_bounds(g: np.ndarray, n_groups: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) per group id of a group-sorted array."""
    if len(g) and np.any(np.diff(g) < 0):
        raise AssertionError("table is not sorted by group")
    starts = np.searchsorted(g, np.arange(n_groups), side="left")
    ends = np.searchsorted(g, np.arange(n_groups), side="right")
    return list(zip(starts.tolist(), ends.tolist()))


def gather(tb: Tables, q: np.ndarray, idx_d: np.ndarray, idx_a: np.ndarray, k_dep: int, k_arr: int):
    """Token tensors for the query departures `q` (departure-table rows): (dep [B,k_dep,F_d], mask_d [B,k_dep],
    arr [B,k_arr,F_a], mask_a [B,k_arr]), float32/bool. Padded tokens are all-zero."""
    Id = np.asarray(idx_d[q, :k_dep])
    Ia = np.asarray(idx_a[q, :k_arr])
    md, ma = Id >= 0, Ia >= 0
    Idc, Iac = np.where(md, Id, 0), np.where(ma, Ia, 0)
    t0 = tb.dep_t[q].astype(np.int64)[:, None]
    # departures
    dt = (tb.dep_t[Idc].astype(np.int64) - t0) / DT_SCALE
    aj, aq = tb.dep_aobt[Idc].astype(np.int64), tb.dep_aobt[q].astype(np.int64)[:, None]
    aok_j, aok_q = aj != NULL_T, aq != NULL_T
    d_aobt = np.where(aok_j, slog((aj - t0).astype(np.float64)), 0.0)
    order = np.where(aok_j & aok_q, np.sign(aq - aj), 0)
    rq, sq, pq = tb.dep_rwy[q][:, None], tb.dep_stand[q][:, None], tb.dep_prefix[q][:, None]
    same = lambda col, qv, J: ((col[J] == qv) & (qv >= 0)).astype(np.float32)
    dyn_d = np.stack(
        [dt, d_aobt, order, same(tb.dep_rwy, rq, Idc), same(tb.dep_stand, sq, Idc), same(tb.dep_prefix, pq, Idc)], -1
    ).astype(np.float32)
    dep = np.concatenate([tb.dep_static[Idc], dyn_d], -1) * md[..., None]
    # arrivals
    tl = (tb.arr_t_land[Iac].astype(np.int64) - t0) / DT_SCALE
    ib = tb.arr_t_inblock[Iac].astype(np.int64)
    ti = np.where(ib != NULL_T, (ib - t0) / DT_SCALE, 0.0)
    dyn_a = np.stack(
        [tl, ti, same(tb.arr_rwy, rq, Iac), same(tb.arr_stand, sq, Iac), same(tb.arr_prefix, pq, Iac)], -1
    ).astype(np.float32)
    arr = np.concatenate([tb.arr_static[Iac], dyn_a], -1) * ma[..., None]
    return dep.astype(np.float32), md, arr.astype(np.float32), ma


def main(argv=None) -> int:
    """Build and save the tables and the index for the 12 training files (S5)."""
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/nbrnet/tokens"))
    a = ap.parse_args(argv)
    t = time.time()
    files = sorted(a.raw_dir.glob("training_2025-*.parquet"))
    raw = load_movements(files)
    tb = build_tables(raw)
    del raw
    print(f"tables: {len(tb.dep_t):,} departures, {len(tb.arr_key):,} arrivals, {len(tb.groups)} groups ({time.time() - t:.0f}s)", flush=True)
    idx_d, idx_a, cov = build_index(tb)
    print(f"index built ({time.time() - t:.0f}s)", flush=True)
    tb.save(a.out)
    np.save(a.out / "idx_d.npy", idx_d)
    np.save(a.out / "idx_a.npy", idx_a)
    np.save(a.out / "cov.npy", cov)
    nd, na = (idx_d >= 0).sum(1), (idx_a >= 0).sum(1)
    stats = {
        "n_dep": len(tb.dep_t),
        "n_arr": len(tb.arr_key),
        "n_groups": len(tb.groups),
        "files": [p.name for p in files],
        "dep_tokens_mean": float(nd.mean()),
        "dep_tokens_p50_p90_p99": np.percentile(nd, [50, 90, 99]).tolist(),
        "dep_full_share": float((nd == idx_d.shape[1]).mean()),
        "arr_tokens_mean": float(na.mean()),
        "arr_tokens_p50_p90_p99": np.percentile(na, [50, 90, 99]).tolist(),
        "arr_full_share": float((na == idx_a.shape[1]).mean()),
        "cov_lt1_share": float((cov < 1).mean()),
        "seconds": round(time.time() - t, 1),
    }
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

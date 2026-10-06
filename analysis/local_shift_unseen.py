"""Shift worker, tasks 1-2: unseen / rare categories in 2026 and their simulated cost in the 2025 OOF (read-only).

1. For each 2026 departure (ranking.parquet = feat_rank), count of its key in all 2025 departures, for the keys
   STAND, (ADEP,STAND), (ADEP,RUNWAY), (ADEP,STAND,RUNWAY), AIRCRAFT_TYPE, OPERATOR, (ADEP,FLIGHT); bucketed
   unseen / rare (1-19) / seen, per airport and lane (AOBT_3 present or not).
2. The same counts for every OOF row (cv_rows_months7_v2) against ITS fold's training months
   (Jan: Feb-Dec, Jul: all but Jul, Aug-Oct: the other nine, Nov-Dec: Jan-Oct), and the OOF error e = y - (b + c_C)
   per bucket (n, mean e, RMSE, excess SSE / N over the overall MSE).
3. Fallback corrections for unseen/rare (ADEP,STAND) rows, fitted leave-one-month-out on the seven OOF months:
   F0  per-airport mean residual of unseen/rare rows (pure bias)
   F1  slope * d_grp,  d_grp = med y(ADEP, stand-group, RWY) - med y(ADEP, RWY) over the fold's training months
   F2  slope * d_nb,   d_nb  = med y(ADEP, nearest seen stand in group, RWY) - med y(ADEP, RWY)
   Gain reported as SSE reduction / N_all (MSE units), per held-out month.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_shift_unseen.py
Outputs: $SCRATCH/agents/shift/unseen_*.parquet / .txt
"""
import os
import re

import numpy as np
import polars as pl

SCR = os.environ["SCRATCH"]
OUT = os.path.join(SCR, "agents", "shift")
os.makedirs(OUT, exist_ok=True)
RARE = 20
COLS = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
        "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "TAXITIME_SEC_mvt"]
KEYS = {
    "stand": ["STAND_mvt"],
    "apt_stand": ["ADEP_mvt", "STAND_mvt"],
    "apt_rwy": ["ADEP_mvt", "RUNWAY_mvt"],
    "apt_stand_rwy": ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
    "actype": ["AIRCRAFT_TYPE_mvt"],
    "operator": ["AIRCRAFT_OPERATOR_flt"],
    "apt_flight": ["ADEP_mvt", "FLIGHT_mvt"],
}
# held-out months of each OOF fold -> training months are the complement within 2025
HELD = {1: [1], 7: [7], 8: [8, 9, 10], 9: [8, 9, 10], 10: [8, 9, 10], 11: [11, 12], 12: [11, 12]}


def stand_group(s: str | None) -> str | None:
    if s is None:
        return None
    m = re.match(r"^([A-Z]*)(\d*)", s.upper())
    a, d = m.group(1), m.group(2)
    if a:
        return a
    if d:
        return f"#{d[0]}{len(d)}"
    return s


def stand_num(s: str | None) -> float:
    if s is None:
        return np.nan
    m = re.search(r"\d+", s)
    return float(m.group(0)) if m else np.nan


def load(path: str) -> pl.DataFrame:
    d = pl.read_parquet(path, columns=COLS)
    return d.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"),
                          pl.col("mvt_minus_aobt3").is_not_null().alias("laneA"))


def bucket(n: pl.Expr) -> pl.Expr:
    return pl.when(n.is_null() | (n == 0)).then(pl.lit("unseen")).when(n < RARE).then(pl.lit("rare")).otherwise(pl.lit("seen"))


def main() -> None:
    tr = load("data/work_flseq/feat_train.parquet")
    rk = load("data/work_flseq/feat_rank.parquet")
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(str(s))

    # ---------- task 1: 2026 counts ----------
    R = rk.select("MVT_ID_mvt", "ADEP_mvt", "laneA")
    for k, ks in KEYS.items():
        cnt = tr.drop_nulls(ks).group_by(ks).len().rename({"len": f"n_{k}"})
        R = R.with_columns(rk.join(cnt, on=ks, how="left", maintain_order="left")[f"n_{k}"].fill_null(0).alias(f"n_{k}"))
        # a null key is not "unseen": keep it separate
        R = R.with_columns(pl.when(pl.any_horizontal([rk[c].is_null() for c in ks])).then(pl.lit("null"))
                           .otherwise(bucket(pl.col(f"n_{k}"))).alias(f"b_{k}"))
    R.write_parquet(os.path.join(OUT, "unseen_rank_counts.parquet"))
    log(f"2026 departures {R.height:,}")
    for k in KEYS:
        t = R.group_by(f"b_{k}").len().sort(f"b_{k}")
        log(f"  {k:14s} " + "  ".join(f"{b}={n:,}" for b, n in t.rows()))
    log("\nper airport x lane, unseen | rare row counts:")
    agg = [pl.len().alias("n")]
    for k in KEYS:
        agg += [(pl.col(f"b_{k}") == "unseen").sum().alias(f"U_{k}"), (pl.col(f"b_{k}") == "rare").sum().alias(f"R_{k}")]
    tab = R.group_by("ADEP_mvt", "laneA").agg(agg).sort("ADEP_mvt", "laneA")
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=250):
        log(tab)

    # ---------- task 2: OOF simulation ----------
    oof = pl.read_parquet(os.path.join(SCR, "learn", "cv_rows_months7_v2.parquet"))
    oof = oof.with_columns((pl.col("y") - pl.col("b") - pl.col("c_C")).alias("e"),
                           pl.col("ym").str.slice(5, 2).cast(pl.Int32).alias("m"))
    O = oof.join(tr.drop("m", "TAXITIME_SEC_mvt", "ADEP_mvt"), on="MVT_ID_mvt", how="left")
    assert O["MVT_TIME_UTC_mvt"].null_count() == 0
    N = O.height
    mse_all = float((O["e"] ** 2).mean())
    log(f"\nOOF rows {N:,}  MSE {mse_all:,.0f}  RMSE {mse_all ** 0.5:.2f}")
    parts = []
    for held in ({1}, {7}, {8, 9, 10}, {11, 12}):
        sub = O.filter(pl.col("m").is_in(list(held)))
        trn = tr.filter(~pl.col("m").is_in(list(held)))
        for k, ks in KEYS.items():
            cnt = trn.drop_nulls(ks).group_by(ks).len().rename({"len": f"n_{k}"})
            sub = sub.join(cnt, on=ks, how="left", maintain_order="left").with_columns(pl.col(f"n_{k}").fill_null(0))
            sub = sub.with_columns(pl.when(pl.any_horizontal([pl.col(c).is_null() for c in ks])).then(pl.lit("null"))
                                   .otherwise(bucket(pl.col(f"n_{k}"))).alias(f"b_{k}"))
        # stand-group / nearest-stand references from the fold's training months
        lab = trn.filter(pl.col("TAXITIME_SEC_mvt").is_between(1, 7199)).with_columns(
            pl.col("STAND_mvt").map_elements(stand_group, return_dtype=pl.Utf8).alias("grp"))
        ar = lab.group_by("ADEP_mvt", "RUNWAY_mvt").agg(pl.col("TAXITIME_SEC_mvt").median().alias("med_ar"))
        gr = lab.group_by("ADEP_mvt", "grp", "RUNWAY_mvt").agg(pl.col("TAXITIME_SEC_mvt").median().alias("med_gr"),
                                                              pl.len().alias("n_gr"))
        sr = lab.group_by("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt").agg(pl.col("TAXITIME_SEC_mvt").median().alias("med_sr"),
                                                                    pl.len().alias("n_sr"))
        st = lab.group_by("ADEP_mvt", "STAND_mvt", "grp").len().filter(pl.col("len") >= RARE).with_columns(
            pl.col("STAND_mvt").map_elements(stand_num, return_dtype=pl.Float64).alias("num"))
        sub = sub.with_columns(pl.col("STAND_mvt").map_elements(stand_group, return_dtype=pl.Utf8).alias("grp"),
                               pl.col("STAND_mvt").map_elements(stand_num, return_dtype=pl.Float64).alias("num"))
        # nearest seen (>= RARE rows) stand of the same group by stand number, for the unseen/rare rows only
        need = sub.filter(pl.col("b_apt_stand").is_in(["unseen", "rare"])).select("ADEP_mvt", "STAND_mvt", "grp", "num").unique()
        nb = []
        stl = {(a, g): (np.array(nums), names) for a, g, nums, names in
               st.drop_nulls("num").group_by("ADEP_mvt", "grp").agg(pl.col("num"), pl.col("STAND_mvt")).rows()}
        for a, s, g, x in need.rows():
            v = stl.get((a, g))
            if v is None or x is None or np.isnan(x):
                continue
            nums, names = v
            nb.append((a, s, names[int(np.argmin(np.abs(nums - x)))]))
        nbt = pl.DataFrame(nb, schema=["ADEP_mvt", "STAND_mvt", "nb_stand"], orient="row")
        sub = (sub.join(nbt, on=["ADEP_mvt", "STAND_mvt"], how="left")
               .join(ar, on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
               .join(gr, on=["ADEP_mvt", "grp", "RUNWAY_mvt"], how="left")
               .join(sr.rename({"STAND_mvt": "nb_stand", "med_sr": "med_nb", "n_sr": "n_nb"}),
                     on=["ADEP_mvt", "nb_stand", "RUNWAY_mvt"], how="left")
               .with_columns((pl.col("med_gr") - pl.col("med_ar")).alias("d_grp"),
                             (pl.col("med_nb") - pl.col("med_ar")).alias("d_nb")))
        parts.append(sub)
    O = pl.concat(parts, how="diagonal_relaxed")
    O.write_parquet(os.path.join(OUT, "unseen_oof.parquet"))

    log("\nOOF error by bucket (counts against the fold's training months); exSSE/N = sum(e^2 - MSE_all)/N_all")
    for k in KEYS:
        t = (O.group_by(f"b_{k}").agg(pl.len().alias("n"), pl.col("e").mean().alias("mean_e"),
                                     (pl.col("e") ** 2).mean().sqrt().alias("rmse"),
                                     ((pl.col("e") ** 2 - mse_all).sum() / N).alias("exSSE_N"))
             .sort(f"b_{k}"))
        log(f"  {k}")
        for r in t.rows():
            log(f"    {r[0]:7s} n={r[1]:>9,}  mean_e={r[2]:+8.1f}  rmse={r[3]:7.1f}  exSSE/N={r[4]:+8.1f}")
    log("\nunseen|rare (ADEP,STAND) rows per OOF month: n, mean e, rmse")
    t = (O.filter(pl.col("b_apt_stand").is_in(["unseen", "rare"])).group_by("ym", "b_apt_stand")
         .agg(pl.len(), pl.col("e").mean().alias("me"), (pl.col("e") ** 2).mean().sqrt().alias("rm")).sort("ym", "b_apt_stand"))
    for r in t.rows():
        log(f"    {r[0]} {r[1]:6s} n={r[2]:>6,} mean_e={r[3]:+7.1f} rmse={r[4]:7.1f}")
    log("\nunseen|rare (ADEP,STAND) rows per airport (all OOF months)")
    t = (O.filter(pl.col("b_apt_stand").is_in(["unseen", "rare"])).group_by("ADEP_mvt")
         .agg(pl.len(), pl.col("e").mean().alias("me"), (pl.col("e") ** 2).mean().sqrt().alias("rm"), pl.col("laneA").mean()).sort("ADEP_mvt"))
    for r in t.rows():
        log(f"    {r[0]} n={r[1]:>6,} mean_e={r[2]:+7.1f} rmse={r[3]:7.1f} laneA={r[4]:.2f}")

    # ---------- fallback corrections, leave-one-month-out ----------
    mask = pl.col("b_apt_stand").is_in(["unseen", "rare"])
    S = O.filter(mask)
    months = sorted(O["ym"].unique().to_list())
    log(f"\nfallbacks on unseen|rare (ADEP,STAND) rows ({S.height:,}); gain = SSE reduction / N_all, + = better")
    for arm in ("F0_apt_bias", "F1_grp_slope", "F2_nb_slope", "F12_both"):
        gains = []
        for m in months:
            fit, ho = S.filter(pl.col("ym") != m), S.filter(pl.col("ym") == m)
            e_ho = ho["e"].to_numpy()
            if arm == "F0_apt_bias":
                bias = fit.group_by("ADEP_mvt").agg(pl.col("e").mean().alias("c"), pl.len().alias("n"))
                # shrink toward 0 with 200 pseudo-rows
                bias = bias.with_columns((pl.col("c") * pl.col("n") / (pl.col("n") + 200)).alias("c"))
                c = ho.join(bias, on="ADEP_mvt", how="left", maintain_order="left")["c"].fill_null(0).to_numpy()
            else:
                feats = {"F1_grp_slope": ["d_grp"], "F2_nb_slope": ["d_nb"], "F12_both": ["d_grp", "d_nb"]}[arm]
                Xf = fit.select([pl.col(f).fill_null(0) for f in feats]).to_numpy()
                beta = np.linalg.lstsq(Xf.T @ Xf + 1e3 * np.eye(len(feats)), Xf.T @ fit["e"].to_numpy(), rcond=None)[0]
                c = ho.select([pl.col(f).fill_null(0) for f in feats]).to_numpy() @ beta
            gains.append((m, ho.height, float((e_ho ** 2).sum() - ((e_ho - c) ** 2).sum()) / N))
        tot = sum(g for _, _, g in gains)
        log(f"  {arm:14s} total {tot:+7.2f}  " + "  ".join(f"{m[2:]}:{g:+.2f}" for m, _, g in gains))
    with open(os.path.join(OUT, "unseen_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

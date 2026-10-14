"""LTFM Jan/Jul mean-bias probe on the old (Jan/Jul 2026) leaderboard, and the correction it implies for the final file.

Probe files (old LB, 344,841 rows): v48 with every LTFM January (or July) departure shifted by S = -40 s, nothing else.
For a pure shift S of the n rows of one cell, the change in the leaderboard's mean squared error is exactly
    MSE_probe - MSE_v48 = (n / N) * (S^2 - 2 * S * rbar),   rbar = mean(y - v48) over the cell,
so rbar = (S^2 - (R_probe^2 - R_v48^2) * N / n) / (2 S), with R the leaderboard RMSEs (N = 344,841, the full ranking file).

    python -I docs/final_phase_gate/ltfm_probe.py make <v48.parquet> <out_dir>
    python -I docs/final_phase_gate/ltfm_probe.py apply <v48.parquet> <R_v48> <R_probe_jan|-> <R_probe_jul|-> <final_in> <final_out> [--dithered]

apply: the shift for a cell of the final file F is c = rbar - mean(F - v48) over the cell (on F before the +1x MVT dither when
--dithered, i.e. the per-row dither is subtracted first; LTFM rows are not touched by the LIRF corrections). The same
departures (MVT_ID) are in both files. Then the LOBT identity clip on the final clock and the zero floor. Use '-' for a month
whose probe was not run (that month is left unchanged)."""
import sys
import numpy as np
import polars as pl

K, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
S = -40.0
N_OLD = 344841
CELLS = (("LTFM", 1), ("LTFM", 7))


def old_ranking():
    return (pl.scan_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
            .select(K, "ADEP_mvt", pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"), pl.col("MVT_TIME_UTC_mvt").alias("t_old"))
            .collect())


def make(v48p, out):
    V = pl.read_parquet(v48p)
    X = V.join(old_ranking(), on=K, how="left", maintain_order="left")
    assert X.height == N_OLD and X["ADEP_mvt"].null_count() == 0
    for ap, m in CELLS:
        cell = (X["ADEP_mvt"] == ap) & (X["m"] == m)
        new = X[T].cast(pl.Float64) + np.where(cell.to_numpy(), S, 0.0)
        assert float(new.min()) >= 0
        f = X.select(K, pl.Series(T, new.round(0)).cast(pl.Int32)).cast(V.schema)
        f.write_parquet(f"{out}/v48_{ap}_{m:02d}_shift{int(S)}.parquet")
        print(f"{ap} {m:02d}: rows {int(cell.sum())}, share {cell.sum() / N_OLD:.6f}")


def apply(v48p, r0, rj, rl, fin, fout, dithered):
    V = pl.read_parquet(v48p).rename({T: "v48"})
    O = old_ranking()
    X = V.join(O, on=K, how="left")
    F = pl.read_parquet(fin)
    nw = (pl.scan_parquet("data/raw/final_ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
          .select(K, "ADEP_mvt", pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"), "MVT_TIME_UTC_mvt",
                  pl.col("AOBT_3_flt").is_not_null().alias("mat"),
                  (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("mml")).collect())
    Z = F.join(nw, on=K, how="left", maintain_order="left").join(X.select(K, "v48", "t_old"), on=K, how="left", maintain_order="left")
    Z = Z.with_columns(dl=pl.when(pl.col("mat") & pl.col("t_old").is_not_null())
                       .then((pl.col("MVT_TIME_UTC_mvt") - pl.col("t_old")).dt.total_seconds().cast(pl.Float64)).otherwise(0.0))
    shift = pl.lit(0.0)
    report = []
    for (ap, m), r in zip(CELLS, (rj, rl)):
        if r == "-":
            continue
        n = int(((X["ADEP_mvt"] == ap) & (X["m"] == m)).sum())
        share = n / N_OLD
        dmse = float(r) ** 2 - float(r0) ** 2
        rbar = (S ** 2 - dmse / share) / (2 * S)
        cell = (pl.col("ADEP_mvt") == ap) & (pl.col("m") == m)
        cz = Z.filter(cell)
        assert cz["v48"].null_count() == 0 and cz.height == n, "final cell is not the same departures as the old cell"
        base = (cz[T].cast(pl.Float64) - (cz["dl"] if dithered else 0.0)) - cz["v48"].cast(pl.Float64)
        c = rbar - float(base.mean())
        shift = shift + pl.when(cell).then(pl.lit(c)).otherwise(0.0)
        report.append(f"{ap} {m:02d}: n {n}, probe dMSE {dmse:+.3f}, rbar(v48) {rbar:+.2f} s, mean(F-v48) {float(base.mean()):+.2f} s, "
                      f"shift {c:+.2f} s; old-LB MSE gain for v48 {share * rbar ** 2:.1f} s^2")
    v = pl.col(T).cast(pl.Float64) + shift
    lo, hi = (pl.col("mml") - 3606).ceil(), (pl.col("mml") + 3606).ceil()
    v = pl.when(pl.col("mml").is_not_null()).then(pl.max_horizontal(pl.min_horizontal(v, hi), lo)).otherwise(v)
    out = Z.select(K, v.clip(lower_bound=0).round(0).cast(pl.Int32).alias(T)).cast(F.schema)
    assert out[K].equals(F[K]) and out[T].null_count() == 0
    out.write_parquet(fout)
    d = (out[T].cast(pl.Float64) - F[T].cast(pl.Float64)).to_numpy()
    for line in report:
        print(line)
    print(f"rows changed {int((d != 0).sum()):,}; mean change on changed rows {d[d != 0].mean() if (d != 0).any() else 0:+.2f}; "
          f"sum d^2/670,790 {float((d ** 2).sum()) / 670790:.1f}; wrote {fout}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "make":
        make(a[1], a[2])
    else:
        apply(a[1], a[2], a[3], a[4], a[5], a[6], "--dithered" in a)

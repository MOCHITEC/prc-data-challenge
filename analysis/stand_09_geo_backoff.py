"""Does OSM stand->threshold distance help for thin / unseen stand x runway cells?

Month-blocked (fit on 2025 months != m, score m in {1, 7}); thin = stand x runway cell with < 30 training rows.
Predictors for the thin rows (clean y in [30, 7200]):
  apt_rwy        median y of airport x runway
  pfx10_rwy      median of airport x prefix x number-decade x runway (n>=30) -> else apt_rwy (best name-based)
  geo_lin        airport x runway weighted linear fit of cell medians on distance (cells n>=30)
  geo_knn        median y of the 5 geographically nearest stands (same runway, cells n>=30), by stand x/y
  blend          mean of pfx10_rwy and geo_knn where both exist
"""

from __future__ import annotations

import numpy as np
import polars as pl

from stand_common import SCRATCH, load_all, stand_number, stand_prefix

pl.Config.set_tbl_width_chars(200)


def main() -> None:
    geo = pl.read_parquet(f"{SCRATCH}/stand_geo_table.parquet")
    df = load_all(keep_stand_str=True).filter(pl.col("PHASE_mvt") == "DEP").select(
        ["MVT_ID_mvt", pl.col("APT").cast(pl.String), "STAND_ok", pl.col("RUNWAY_mvt").cast(pl.String), "y", "split", "t_mvt"])
    df = df.filter((pl.col("split") == "train") & (pl.col("y") >= 30) & (pl.col("y") <= 7200)).with_columns(
        pl.from_epoch(pl.col("t_mvt").cast(pl.Int64), time_unit="s").dt.month().alias("month"),
        stand_prefix(pl.col("STAND_ok")).alias("pfx"), (stand_number(pl.col("STAND_ok")) // 10).alias("num10"))
    df = df.join(geo.select(["APT", "STAND_ok", "RUNWAY_mvt", "geo_x_m", "geo_y_m", "geo_dist_thr_m"]),
                 on=["APT", "STAND_ok", "RUNWAY_mvt"], how="left")
    rows = []
    for m in (1, 7):
        tr, va = df.filter(pl.col("month") != m), df.filter(pl.col("month") == m)
        cell = tr.group_by(["APT", "STAND_ok", "RUNWAY_mvt"]).agg(
            pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("geo_x_m").first(),
            pl.col("geo_y_m").first(), pl.col("geo_dist_thr_m").first())
        ar = tr.group_by(["APT", "RUNWAY_mvt"]).agg(pl.col("y").median().alias("p_ar"))
        p10 = tr.group_by(["APT", "pfx", "num10", "RUNWAY_mvt"]).agg(pl.col("y").median().alias("p_p10"), pl.len().alias("n_p10"))
        v = va.join(cell.select(["APT", "STAND_ok", "RUNWAY_mvt", "n"]), on=["APT", "STAND_ok", "RUNWAY_mvt"], how="left")
        thin = v.filter(pl.col("n").fill_null(0) < 30).join(ar, on=["APT", "RUNWAY_mvt"], how="left").join(
            p10, on=["APT", "pfx", "num10", "RUNWAY_mvt"], how="left")
        good = cell.filter((pl.col("n") >= 30) & pl.col("geo_dist_thr_m").is_not_null())
        # linear fit per airport x runway
        coef = {}
        for (apt, rwy), g in good.group_by(["APT", "RUNWAY_mvt"]):
            if g.height < 8:
                continue
            x, y, w = g["geo_dist_thr_m"].to_numpy(), g["y_med"].to_numpy(), g["n"].to_numpy().astype(float)
            A = np.vstack([np.ones_like(x), x]).T * np.sqrt(w)[:, None]
            b = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
            coef[(apt, rwy)] = (b, g)
        p_lin, p_knn = [], []
        for apt, rwy, d, gx, gy in thin.select(["APT", "RUNWAY_mvt", "geo_dist_thr_m", "geo_x_m", "geo_y_m"]).iter_rows():
            c = coef.get((apt, rwy))
            if c is None or d is None:
                p_lin.append(None)
                p_knn.append(None)
                continue
            b, g = c
            p_lin.append(float(b[0] + b[1] * d))
            dd = np.hypot(g["geo_x_m"].to_numpy() - gx, g["geo_y_m"].to_numpy() - gy)
            k = np.argsort(dd)[:5]
            p_knn.append(float(np.median(g["y_med"].to_numpy()[k])))
        thin = thin.with_columns(pl.Series("p_lin", p_lin, dtype=pl.Float64), pl.Series("p_knn", p_knn, dtype=pl.Float64))
        thin = thin.with_columns(pl.when(pl.col("n_p10") >= 30).then(pl.col("p_p10")).otherwise(pl.col("p_ar")).alias("p_name"))
        thin = thin.with_columns(
            pl.coalesce(pl.col("p_lin"), pl.col("p_ar")).alias("p_lin_fb"),
            pl.coalesce(pl.col("p_knn"), pl.col("p_ar")).alias("p_knn_fb"),
            pl.coalesce((pl.col("p_knn") + pl.col("p_name")) / 2, pl.col("p_name")).alias("p_blend"),
        )
        yy = thin["y"].to_numpy()
        r = lambda c: float(np.sqrt(np.nanmean((yy - thin[c].fill_null(np.nan).to_numpy()) ** 2)))  # noqa: E731
        cov = thin.select(pl.col("p_knn").is_not_null().mean()).item()
        rows.append((m, thin.height, cov, r("p_ar"), r("p_name"), r("p_lin_fb"), r("p_knn_fb"), r("p_blend")))
    print(pl.DataFrame(rows, schema=["month", "n_thin", "geo_coverage", "apt_rwy", "pfx10_rwy", "geo_lin", "geo_knn5",
                                     "blend(pfx10,knn)"], orient="row"))


if __name__ == "__main__":
    main()

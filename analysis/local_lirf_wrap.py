"""LIRF unmatched departures delayed > ~14 h: the "one day + taxi" label class (local).

When the off-block of a long-delayed LIRF flight without NM AOBT_3 is entered, its clock time is put on the date nearest
the schedule, so a delay of more than ~13.5 h yields BLOCK one day early and y = 86400 * k + taxi (k = round(delay/1 d)).
Otherwise BLOCK stays a schedule copy (y = MVT - SCHED). In 2025 no such row had a normal taxi.
Blend: p' = q * D + (1 - q) * ms on rows with ms >= MS_MIN, D = 86400 * k + T, T = median (y - 86400 k) of the class;
q = (wraps + 1) / (rows + 2) of 2025 in two bands, MS_MIN <= ms < MS_MID and ms >= MS_MID (2025: 1/5 and 10/15).

    .venv/bin/python analysis/local_lirf_wrap.py check
    .venv/bin/python analysis/local_lirf_wrap.py apply <in.parquet> <out.parquet>
"""
import glob
import sys

import numpy as np
import polars as pl

MS_MIN = 50400
MS_MID = 56000
DAY = 86400


def rows(df):
    return (df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("AOBT_3_flt").is_null())
            .with_columns((pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("ms")))


def train():
    T = rows(pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_*.parquet"))])).filter(pl.col("ms") >= 30000)
    k = ((T["ms"] - 1150) / DAY).round(0).clip(1)
    y = T["TAXITIME_SEC_mvt"]
    return T.with_columns(pl.Series("k", k), pl.when((y - T["ms"]).abs() <= 12).then(pl.lit("copy"))
                          .when((y - k * DAY).is_between(0, 4000)).then(pl.lit("wrap")).otherwise(pl.lit("other")).alias("cls"))


def check():
    T = train()
    print(T.with_columns((pl.col("ms") // 3600).alias("ms_h")).group_by(pl.col("ms_h").cut([44, 50, 56])).agg(
        (pl.col("cls") == "copy").sum().alias("copy"), (pl.col("cls") == "wrap").sum().alias("wrap"), (pl.col("cls") == "other").sum().alias("other")))
    W = T.filter(pl.col("ms") >= MS_MIN)
    tw = W.filter(pl.col("cls") == "wrap")
    taxi = (tw["TAXITIME_SEC_mvt"] - tw["k"] * DAY)
    print(f"rows ms >= {MS_MIN}: {W.height}, wrap {tw.height}; taxi part median {taxi.median():.0f} sd {taxi.std():.0f}; months {sorted(tw['MVT_TIME_UTC_mvt'].dt.month().to_list())}")
    # leave-one-out: p and T from the other rows; gain vs predicting the copy (ms)
    y, ms, k, w = (W[c].to_numpy().astype(float) for c in ("TAXITIME_SEC_mvt", "ms", "k")), None, None, None
    y, ms, k = W["TAXITIME_SEC_mvt"].to_numpy().astype(float), W["ms"].to_numpy().astype(float), W["k"].to_numpy().astype(float)
    isw = (W["cls"] == "wrap").to_numpy()
    for shrink in (0, 2, 5):
        g = 0.0
        for i in range(len(y)):
            o = np.arange(len(y)) != i
            p = (isw[o].sum() + shrink * 0.5) / (o.sum() + shrink)
            Tm = float(np.median((y - k * DAY)[o & isw]))
            x = p * (k[i] * DAY + Tm) + (1 - p) * ms[i]
            g += (ms[i] - y[i]) ** 2 - (x - y[i]) ** 2
        print(f"LOO (prior strength {shrink}): sum gain {g:.3e} s^2 over {len(y)} rows = {g / 4.17e6:.0f} MSE on a 4.17M-row year, "
              f"{g / len(y):.3e} per row")


def apply(src, dst):
    T = train()
    tw = T.filter((pl.col("ms") >= MS_MIN) & (pl.col("cls") == "wrap"))
    Tm = float((tw["TAXITIME_SEC_mvt"] - tw["k"] * DAY).median())
    W = T.filter(pl.col("ms") >= MS_MIN).with_columns((pl.col("ms") >= MS_MID).alias("hi"))
    q = {h: (float((W.filter(pl.col("hi") == h)["cls"] == "wrap").sum()) + 1) / (W.filter(pl.col("hi") == h).height + 2) for h in (False, True)}
    print("q by band (lo, hi):", q)
    R = rows(pl.read_parquet("data/raw/ranking.parquet")).filter(pl.col("ms") >= MS_MIN)
    sub = pl.read_parquet(src)
    G = (R.select("MVT_ID_mvt", "FLIGHT_mvt", "ms").join(sub.rename({"TAXITIME_SEC_mvt": "p"}), on="MVT_ID_mvt")
         .with_columns(((pl.col("ms") - Tm) / DAY).round(0).clip(1).alias("k"),
                       pl.when(pl.col("ms") >= MS_MID).then(q[True]).otherwise(q[False]).alias("q"))
         .with_columns((pl.col("q") * (pl.col("k") * DAY + Tm) + (1 - pl.col("q")) * pl.col("ms")).round(0).cast(pl.Int32).alias("new")))
    N = sub.height
    d = (G["new"] - G["p"]).cast(pl.Float64)
    D = (G["k"] * DAY + Tm - G["p"]).to_numpy()
    print(G)
    gw = D ** 2 - (D - d.to_numpy()) ** 2
    gc = ((G["ms"] - G["p"]) ** 2 - (G["ms"] - G["new"]) ** 2).to_numpy()
    qq = G["q"].to_numpy()
    print(f"expected gain at the 2025 rates {float((qq * gw + (1 - qq) * gc).sum()) / N:+.0f} MSE")
    print(f"T={Tm:.0f}; rows {G.height}; if all wrap: gain {float((D ** 2 - (D - d.to_numpy()) ** 2).sum()) / N:+.0f} MSE; "
          f"if all copy (y=ms): gain {float(((G['ms'] - G['p']) ** 2 - (G['ms'] - G['new']) ** 2).sum()) / N:+.0f} MSE")
    out = (sub.join(G.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left", maintain_order="left")
           .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(sub.columns))
    assert (out["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all() and out.null_count().sum_horizontal().item() == 0
    out.write_parquet(dst)


if __name__ == "__main__":
    check() if sys.argv[1] == "check" else apply(sys.argv[2], sys.argv[3])

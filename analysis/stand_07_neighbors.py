"""Adjacent-stand interaction (push-back conflicts) from ranking-visible times only.

Neighbour of a stand = same airport, same letter prefix and suffix-less number within +-1 or +-2
(e.g. A14 <-> A13, A15, A12, A16; EGLL 511 <-> 510, 512 ...; 221L/221R share number 221 and count as |d|=0).
For each departure with an NM off-block (AOBT_3, ranking-visible):
  nb_dep_aobt_5m   departures from neighbour stands whose AOBT_3 is within +-5 min of ours
  nb_arr_inb_5m    arrivals in-blocking at neighbour stands within +-5 min of our AOBT_3
  same_num_dep_5m  departures from the same number with a different suffix (L/R/C sub-stands)
Also computed around take-off-anchored proxy (MVT - 15 min) for NM-unmatched rows: *_tko variants.
Writes SCRATCH/stand_nbr_feats.parquet and prints residual (y - (MVT-AOBT3)) by count.
"""

from __future__ import annotations

import polars as pl

from stand_common import SCRATCH, load_all, stand_number, stand_prefix

pl.Config.set_tbl_rows(40)
W = 300.0


def count_window(anchor: pl.DataFrame, events: pl.DataFrame, offsets: list[int], name: str) -> pl.DataFrame:
    """anchor: MVT_ID_mvt, G, num, t ; events: G, num, t. Count events with num = anchor.num+o (o in offsets)
    and |t_e - t| <= W."""
    ev = events.sort(["G", "num", "t"]).with_columns(pl.int_range(1, pl.len() + 1).over(["G", "num"]).alias("_k"))
    ev_s = ev.sort("t")
    total = None
    for o in offsets:
        a = anchor.with_columns((pl.col("num") + o).alias("numo"))
        res = []
        for side, shift in (("hi", W), ("lo", -W - 1e-3)):
            q = a.with_columns((pl.col("t") + shift).alias("tq")).sort("tq")
            j = q.join_asof(ev_s.select(["G", pl.col("num").alias("numo"), "t", "_k"]).rename({"t": "te"}),
                            left_on="tq", right_on="te", by=["G", "numo"], strategy="backward")
            res.append(j.select(["MVT_ID_mvt", pl.col("_k").fill_null(0).alias(side)]))
        c = res[0].join(res[1], on="MVT_ID_mvt").select(["MVT_ID_mvt", (pl.col("hi") - pl.col("lo")).alias("c")])
        total = c if total is None else total.join(c, on="MVT_ID_mvt").select(
            ["MVT_ID_mvt", (pl.col("c") + pl.col("c_right")).alias("c")])
    return total.rename({"c": name})


def main() -> None:
    df = load_all(keep_stand_str=True).filter(pl.col("STAND_ok").is_not_null())
    df = df.with_columns(
        (pl.col("APT").cast(pl.String) + "|" + stand_prefix(pl.col("STAND_ok"))).alias("G"),
        stand_number(pl.col("STAND_ok")).alias("num"),
    ).filter(pl.col("num").is_not_null())
    dep = df.filter(pl.col("PHASE_mvt") == "DEP")
    arr = df.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("t_blk").is_not_null())
    anc = dep.filter(pl.col("t_aobt").is_not_null()).select(["MVT_ID_mvt", "G", "num", pl.col("t_aobt").alias("t")])
    ev_dep = dep.filter(pl.col("t_aobt").is_not_null()).select(["G", "num", pl.col("t_aobt").alias("t")])
    ev_arr = arr.select(["G", "num", pl.col("t_blk").alias("t")])
    f1 = count_window(anc, ev_dep, [-2, -1, 1, 2], "nb_dep_aobt_5m")
    f2 = count_window(anc, ev_arr, [-2, -1, 1, 2], "nb_arr_inb_5m")
    f3 = count_window(anc, ev_dep, [0], "same_num_dep_5m")  # includes self
    out = f1.join(f2, on="MVT_ID_mvt").join(f3, on="MVT_ID_mvt").with_columns(pl.col("same_num_dep_5m") - 1)
    # take-off anchored (all rows), using take-off times of neighbour departures
    anc2 = dep.select(["MVT_ID_mvt", "G", "num", pl.col("t_mvt").alias("t")])
    ev_dep2 = dep.select(["G", "num", pl.col("t_mvt").alias("t")])
    g1 = count_window(anc2, ev_dep2, [-2, -1, 1, 2], "nb_dep_tko_5m")
    out = anc2.select("MVT_ID_mvt").join(out, on="MVT_ID_mvt", how="left").join(g1, on="MVT_ID_mvt", how="left")
    out.write_parquet(f"{SCRATCH}/stand_nbr_feats.parquet")

    tr = dep.filter(pl.col("split") == "train").select(["MVT_ID_mvt", "APT", "y", "t_mvt", "t_aobt"]).join(out, on="MVT_ID_mvt")
    tr = tr.with_columns((pl.col("y") - (pl.col("t_mvt") - pl.col("t_aobt"))).alias("resid")).filter(
        (pl.col("y") >= 30) & (pl.col("y") <= 7200))
    base = tr.select(pl.col("resid").mean()).item()
    for c in ("nb_dep_aobt_5m", "nb_arr_inb_5m", "same_num_dep_5m", "nb_dep_tko_5m"):
        print(f"## {c}: residual y-(MVT-AOBT3) by count (overall mean {base:.1f})")
        print(tr.group_by(pl.col(c).clip(upper_bound=4)).agg(pl.len(), pl.col("resid").mean().alias("resid_mean"),
                                                            pl.col("y").mean().alias("y_mean")).sort(c))


if __name__ == "__main__":
    main()

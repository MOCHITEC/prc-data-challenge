"""Stand-occupancy features and diagnostics.

For each departure (training and ranking alike) and using only ranking-visible information:
  * ARR in-block times at the same (airport, stand)  -> a_prev1/a_prev2 (latest before take-off), a_next1
  * DEP take-off times at the same stand               -> d_prev / d_next
  * counts of arrivals between the previous departure's take-off and this take-off

Label-side diagnostics (training only; NEVER features):
  * how many arrivals in-blocked at the same stand while this departure was taxiing, i.e. in (BLOCK, MVT)
  * the resulting lower bound  y >= MVT - A  and its tightness
  * stand-log consistency: ARR->ARR / DEP->DEP adjacency in the true (in-block, off-block) sequence

Writes SCRATCH/stand_feats.parquet (one row per departure) and prints the tables used in docs/analysis/stand.md.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from stand_common import SCRATCH, load_all

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)


def asof_idx(left: pl.DataFrame, lcol: str, right: pl.DataFrame, rcol: str, strategy: str, name: str) -> pl.DataFrame:
    """Attach to `left` the right-row (index, time) nearest in `strategy` direction within the same SKEY."""
    r = right.select(["SKEY", pl.col(rcol).alias("_rt"), pl.col("_ridx")]).sort(["_rt"])
    l2 = left.select(["MVT_ID_mvt", "SKEY", pl.col(lcol).alias("_lt")]).filter(pl.col("_lt").is_not_null()).sort("_lt")
    j = l2.join_asof(r, left_on="_lt", right_on="_rt", by="SKEY", strategy=strategy, allow_exact_matches=True)
    return j.select(["MVT_ID_mvt", pl.col("_ridx").alias(f"{name}_idx"), pl.col("_rt").alias(f"{name}_t")])


def main() -> None:
    df = load_all(extra=("EOBT_1_flt", "LOBT_flt"), keep_stand_str=True)
    df = df.with_columns(
        pl.col("EOBT_1_flt").dt.epoch("s").cast(pl.Float64).alias("t_eobt1"),
        pl.col("LOBT_flt").dt.epoch("s").cast(pl.Float64).alias("t_lobt"),
    ).drop(["EOBT_1_flt", "LOBT_flt"])
    print("rows", df.height)
    arr = df.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("SKEY").is_not_null() & pl.col("t_blk").is_not_null())
    dep = df.filter(pl.col("PHASE_mvt") == "DEP")
    # stand-name vocabulary overlap ARR vs DEP per airport
    ov = (
        df.filter(pl.col("SKEY").is_not_null())
        .group_by(["APT", "PHASE_mvt"])
        .agg(pl.col("STAND_ok").unique().alias("s"))
        .pivot(on="PHASE_mvt", index="APT", values="s")
        .with_columns(
            pl.col("DEP").list.len().alias("n_dep_stands"),
            pl.col("ARR").list.len().alias("n_arr_stands"),
            pl.col("DEP").list.set_intersection("ARR").list.len().alias("n_common"),
        )
        .drop(["DEP", "ARR"])
        .sort("APT")
    )
    print("\n## stand vocabulary ARR vs DEP\n", ov)
    df = df.drop(["SKEY_str", "STAND_ok"])

    # arrival sequence per stand
    arr = arr.sort(["SKEY", "t_blk"]).with_columns(pl.int_range(pl.len()).over("SKEY").alias("_ridx"))
    arr_t = arr.select(["SKEY", "_ridx", "t_blk"])
    depk = dep.filter(pl.col("SKEY").is_not_null())

    # latest arrival in-block <= take-off; next arrival in-block > take-off
    j_prev = asof_idx(depk, "t_mvt", arr, "t_blk", "backward", "ap")
    feats = depk.select(["MVT_ID_mvt", "SKEY", "t_mvt", "t_blk", "t_aobt", "t_sched", "t_eobt1", "t_lobt", "y", "split", "APT"]).join(
        j_prev, on="MVT_ID_mvt", how="left"
    )
    # second latest
    feats = feats.join(
        arr_t.select(["SKEY", (pl.col("_ridx") + 1).alias("ap_idx"), pl.col("t_blk").alias("ap2_t")]),
        on=["SKEY", "ap_idx"],
        how="left",
    )
    # next arrival after take-off: ridx = ap_idx + 1 (or 0 if none before)
    feats = feats.with_columns(pl.coalesce(pl.col("ap_idx") + 1, pl.lit(0)).alias("an_idx")).join(
        arr_t.select(["SKEY", pl.col("_ridx").alias("an_idx"), pl.col("t_blk").alias("an_t")]),
        on=["SKEY", "an_idx"],
        how="left",
    )
    feats = feats.with_columns(pl.coalesce(pl.col("ap_idx") + 1, pl.lit(0)).alias("n_arr_upto_t"))
    # departures sequence per stand (by take-off, known in ranking)
    feats = feats.sort(["SKEY", "t_mvt"]).with_columns(
        pl.col("t_mvt").shift(1).over("SKEY").alias("dp_t"),
        pl.col("t_mvt").shift(-1).over("SKEY").alias("dn_t"),
        pl.col("n_arr_upto_t").shift(1).over("SKEY").alias("_n_prev"),
        pl.col("t_aobt").shift(-1).over("SKEY").alias("dn_aobt"),
        pl.col("t_aobt").shift(1).over("SKEY").alias("dp_aobt"),
    )
    feats = feats.with_columns(
        (pl.col("t_mvt") - pl.col("ap_t")).alias("g_arr_prev1"),
        (pl.col("t_mvt") - pl.col("ap2_t")).alias("g_arr_prev2"),
        (pl.col("an_t") - pl.col("t_mvt")).alias("g_arr_next1"),
        (pl.col("t_mvt") - pl.col("dp_t")).alias("g_dep_prev"),
        (pl.col("dn_t") - pl.col("t_mvt")).alias("g_dep_next"),
        (pl.col("n_arr_upto_t") - pl.col("_n_prev")).alias("n_arr_since_prev_dep"),
        # NM-clock-based: the next departure from this stand pushed back (AOBT) - our take-off
        (pl.col("dn_aobt") - pl.col("t_mvt")).alias("g_next_dep_aobt"),
        (pl.col("t_aobt") - pl.col("ap_t")).alias("aobt_minus_arr_prev1"),
        (pl.col("t_aobt") - pl.col("ap2_t")).alias("aobt_minus_arr_prev2"),
    )

    # ---------------- label-side diagnostics (training) ----------------
    tr = feats.filter(pl.col("split") == "train")
    j_blk = asof_idx(tr, "t_blk", arr, "t_blk", "backward", "ab")
    tr = tr.join(j_blk, on="MVT_ID_mvt", how="left").with_columns(
        pl.coalesce(pl.col("ab_idx") + 1, pl.lit(0)).alias("n_arr_upto_blk")
    )
    tr = tr.with_columns((pl.col("n_arr_upto_t") - pl.col("n_arr_upto_blk")).alias("n_arr_during_taxi"))
    # latest arrival in-blocked during taxi = ap_t if n_arr_during_taxi>=1
    tr = tr.with_columns(
        pl.when(pl.col("n_arr_during_taxi") >= 1).then(pl.col("t_mvt") - pl.col("ap_t")).alias("lb_true"),
        (pl.col("t_blk") - pl.col("ab_t")).alias("turn_s"),  # off-block - latest in-block before it
    )
    print("\n## arrivals in-blocked at the same stand during taxi-out (training, deps with valid stand)")
    s = (
        tr.group_by("APT")
        .agg(
            pl.len().alias("n"),
            (pl.col("n_arr_during_taxi") >= 1).mean().alias("share_ge1"),
            (pl.col("n_arr_during_taxi") >= 2).mean().alias("share_ge2"),
            (pl.col("y") - pl.col("lb_true")).median().alias("slack_med"),
            ((pl.col("y") - pl.col("lb_true")) < 120).mean().alias("slack_lt120_of_all"),
            pl.col("turn_s").median().alias("turn_med"),
        )
        .sort("APT")
    )
    print(s)
    tot = tr.select(
        pl.len(),
        (pl.col("n_arr_during_taxi") >= 1).mean().alias("share_ge1"),
        (pl.col("y") - pl.col("lb_true")).median().alias("slack_med"),
        (pl.col("y") - pl.col("lb_true")).quantile(0.1).alias("slack_p10"),
    )
    print(tot)

    # how predictable is "arrival during taxi" from ranking-visible gaps?  lb candidate = g_arr_prev1
    print("\n## g_arr_prev1 (take-off minus latest same-stand in-block) buckets vs truth (training)")
    b = (
        tr.with_columns(
            pl.col("g_arr_prev1")
            .cut([0, 120, 300, 600, 900, 1200, 1800, 2700, 3600, 7200, 14400], left_closed=True)
            .alias("bucket")
        )
        .group_by("bucket")
        .agg(
            pl.len().alias("n"),
            (pl.col("n_arr_during_taxi") >= 1).mean().alias("p_arr_during_taxi"),
            pl.col("y").mean().alias("y_mean"),
            pl.col("y").median().alias("y_med"),
            (pl.col("y") >= pl.col("g_arr_prev1")).mean().alias("p_y_ge_g"),
        )
        .sort("bucket")
    )
    print(b)

    # clock source of the reported BLOCK (docs/06 section 2, tolerance 60 s, simplified priority)
    d = lambda c: (pl.col("t_blk") - pl.col(c)).abs() <= 60  # noqa: E731
    tr = tr.with_columns(
        pl.when(d("t_aobt")).then(pl.lit("aobt3")).when(d("t_sched")).then(pl.lit("sched"))
        .when(d("t_eobt1")).then(pl.lit("eobt1")).when(d("t_lobt")).then(pl.lit("lobt")).otherwise(pl.lit("genuine"))
        .alias("src")
    )
    near = tr.filter(pl.col("g_arr_prev1") < 1500)
    print("\n## rows with a same-stand in-block < 25 min before take-off: violation y < g by clock source")
    print(
        near.group_by("src")
        .agg(pl.len().alias("n"), (pl.col("y") < pl.col("g_arr_prev1")).mean().alias("viol_rate"),
             ((pl.col("y") - pl.col("g_arr_prev1"))).median().alias("y_minus_g_med"))
        .sort("n", descending=True)
    )
    print(
        tr.group_by("src").agg(pl.len().alias("n_all"), (pl.col("g_arr_prev1") < 1500).mean().alias("share_near"))
        .sort("n_all", descending=True)
    )
    # would the bound catch the copy-clock error?  candidate y_c = MVT - candidate clock; violation if y_c < g
    for c, name in (("t_sched", "sched"), ("t_aobt", "aobt3"), ("t_eobt1", "eobt1")):
        yc = pl.col("t_mvt") - pl.col(c)
        r = near.select(
            (yc < pl.col("g_arr_prev1")).mean().alias("cand_violates"),
            ((yc < pl.col("g_arr_prev1")) & (pl.col("src") == name)).sum().alias("viol_and_is_src"),
            ((yc < pl.col("g_arr_prev1"))).sum().alias("viol_n"),
        )
        print(name, r.rows())

    # parity: distribution of the ranking-visible gap in training vs ranking departures
    cuts = [0, 300, 900, 1500, 2700, 3600, 7200]
    par = (
        feats.with_columns(pl.col("g_arr_prev1").cut(cuts, left_closed=True).alias("bucket"))
        .group_by(["split", "bucket"])
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum().over("split")).alias("share"))
        .pivot(on="split", index="bucket", values="share")
        .sort("bucket")
    )
    print("\n## parity of g_arr_prev1 buckets (share of departures)\n", par)

    # stand-log consistency in the true sequence: merge ARR in-block and DEP off-block per stand
    ev = pl.concat(
        [
            arr.filter(pl.col("split") == "train").select(["SKEY", "APT", pl.col("t_blk").alias("t"), pl.lit("A").alias("k")]),
            tr.filter(pl.col("t_blk").is_not_null()).select(["SKEY", "APT", pl.col("t_blk").alias("t"), pl.lit("D").alias("k")]),
        ]
    ).sort(["SKEY", "t"])
    ev = ev.with_columns(pl.col("k").shift(1).over("SKEY").alias("kp"), (pl.col("t") - pl.col("t").shift(1).over("SKEY")).alias("dt"))
    ev = ev.filter(pl.col("kp").is_not_null())
    adj = ev.group_by(["APT"]).agg(
        pl.len().alias("n_pairs"),
        ((pl.col("kp") == "A") & (pl.col("k") == "D")).mean().alias("A->D"),
        ((pl.col("kp") == "D") & (pl.col("k") == "A")).mean().alias("D->A"),
        ((pl.col("kp") == "A") & (pl.col("k") == "A")).mean().alias("A->A"),
        ((pl.col("kp") == "D") & (pl.col("k") == "D")).mean().alias("D->D"),
        (((pl.col("kp") == "A") & (pl.col("k") == "A")) & (pl.col("dt") < 1800)).mean().alias("A->A<30m"),
    ).sort("APT")
    print("\n## stand event alternation (true in-block/off-block sequence, training)\n", adj)

    keep = [
        "MVT_ID_mvt",
        "g_arr_prev1",
        "g_arr_prev2",
        "g_arr_next1",
        "g_dep_prev",
        "g_dep_next",
        "n_arr_since_prev_dep",
        "g_next_dep_aobt",
        "aobt_minus_arr_prev1",
        "aobt_minus_arr_prev2",
    ]
    out = feats.select(keep).join(
        tr.select(["MVT_ID_mvt", "n_arr_during_taxi", "lb_true", "turn_s"]), on="MVT_ID_mvt", how="left"
    )
    out.write_parquet(f"{SCRATCH}/stand_feats.parquet")
    print("wrote", out.shape)


if __name__ == "__main__":
    main()

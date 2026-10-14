"""Compare airport BLOCK, NM AOBT_3 and ADS-B motion references, per airport. Writes aggregate CSVs + figure.

References (all label-free, from detect.py):
  P  (precise, pushback rows): stand run found, first motion backwards (push share >= 0.5), exit resolution
     <= 30 s, no later stop >= 15 min. t_onset = first motion of the pushback (gs >= 1 kt walked back).
     t_taxi = start of sustained forward taxi (end of pushback + engine start).
  UB (upper bound, any row with ground fixes): the aircraft was demonstrably moving under its own power at
     UB = t_taxi if found, else the first departure fix if it already moves (gs > 3 kt). True off-block <= UB.
     BLOCK > UB + 60 s is physically impossible (late BLOCK).
"""
import glob
import os

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

A = os.path.dirname(os.path.abspath(__file__))
J = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(A, "*_det.parquet")))], how="diagonal_relaxed")
days = sorted(J["day"].unique().to_list())
out = []


def show(name, df):
    with pl.Config(tbl_rows=60, tbl_cols=40, tbl_width_chars=260, float_precision=3):
        print(f"\n=== {name}\n{df}")
    df.write_csv(os.path.join(A, f"{name}.csv"))


print("days", days, "rows", J.height)
J = J.with_columns(
    res_s=pl.col("t_move") - pl.col("t_move_lo"),
    c_sched=(pl.col("t_block") - pl.col("t_sched")).abs() <= 12,
    c_eobt=(pl.col("t_block") - pl.col("t_eobt1")).abs() <= 12,
    c_aobt3=(pl.col("t_block") - pl.col("t_aobt3")).abs() <= 60,
    sec0=(pl.col("t_block") % 60 == 0),
)
J = J.with_columns(
    blk_class=pl.when(pl.col("c_sched")).then(pl.lit("=SCHED"))
    .when(pl.col("c_eobt")).then(pl.lit("=EOBT_1"))
    .when(pl.col("c_aobt3")).then(pl.lit("~AOBT_3"))
    .otherwise(pl.lit("other")),
    P=(pl.col("t_onset").is_not_null() & (pl.col("res_s") <= 30) & (pl.col("post_dwell").fill_null(0) < 900)
       & (pl.col("push").fill_null(0) >= 0.5) & (pl.col("t_to") - pl.col("t_onset") < 5400)).fill_null(False),
    # continuity guard: no silence > 180 s between the reference fix and take-off (else it may be the arrival)
    UB=pl.coalesce(pl.when(pl.col("gap_t_taxi") <= 180).then(pl.col("t_taxi")),
                   pl.when((pl.col("dep1_gs").fill_null(0) > 3) & (pl.col("gap_t_dep1") <= 180)).then(pl.col("t_dep1"))),
)
J = J.with_columns(
    d_blk=pl.col("t_block") - pl.col("t_onset"), d_a3=pl.col("t_aobt3") - pl.col("t_onset"),
    d_blk_tx=pl.col("t_block") - pl.col("t_taxi"), d_a3_tx=pl.col("t_aobt3") - pl.col("t_taxi"),
    push_dur=pl.col("t_taxi") - pl.col("t_onset"),
    ub_blk=pl.col("t_block") - pl.col("UB"), ub_a3=pl.col("t_aobt3") - pl.col("UB"),
)
J.write_parquet(os.path.join(A, "all_rows_local.parquet"))  # row level, local only (competition data)

FIVE = ["EDDM", "EGLL", "EHAM", "LEBL", "LSZH"]
ORDER = FIVE + ["EDDF", "LEMD", "LIRF", "LFPG", "LTFM"]

cov = J.group_by("ADEP_mvt").agg(
    pl.len().alias("deps"), pl.col("joined").fill_null(False).mean().alias("joined_cs_to2min"),
    pl.col("ground_fix").fill_null(False).mean().alias("ground"), pl.col("t_move").is_not_null().mean().alias("stand_exit"),
    pl.col("P").mean().alias("P_push_ref"), pl.col("UB").is_not_null().mean().alias("UB_ref"),
    pl.col("res_s").filter(pl.col("P")).median().alias("P_res_med_s"),
    pl.col("stand_dur").filter(pl.col("P")).median().alias("P_stand_dur_med_s")).sort("ADEP_mvt")
show("coverage", cov)


def dist(c, lab):
    e = pl.col(c).drop_nulls()
    return [e.len().alias(f"{lab}_n"), e.median().alias(f"{lab}_med"), e.quantile(.25).alias(f"{lab}_q25"),
            e.quantile(.75).alias(f"{lab}_q75"), (e.abs() <= 60).mean().alias(f"{lab}_w60"),
            (e.abs() > 180).mean().alias(f"{lab}_gt180")]


P = J.filter(pl.col("P"))
show("offsets_push_ref", P.group_by("ADEP_mvt").agg(*dist("d_blk", "BLK"), *dist("d_a3", "A3")).sort("ADEP_mvt"))
show("offsets_push_ref_taxi", P.group_by("ADEP_mvt").agg(*dist("d_blk_tx", "BLKtx"), *dist("d_a3_tx", "A3tx"),
                                                          pl.col("push_dur").median().alias("pushdur_med"),
                                                          pl.col("push_dur").quantile(.25).alias("pushdur_q25"),
                                                          pl.col("push_dur").quantile(.75).alias("pushdur_q75")).sort("ADEP_mvt"))
# per day stability
show("per_day_push_ref", P.group_by("ADEP_mvt", "day").agg(pl.len(), pl.col("d_blk").median().alias("BLK_med"),
                                                           (pl.col("d_blk").quantile(.75) - pl.col("d_blk").quantile(.25)).alias("BLK_iqr"),
                                                           pl.col("d_a3").median().alias("A3_med"),
                                                           (pl.col("d_a3").quantile(.75) - pl.col("d_a3").quantile(.25)).alias("A3_iqr")).sort("ADEP_mvt", "day"))
# whole-second vs minute grid (anonymisation / precision) on P rows
show("push_ref_by_secgrid", P.group_by("ADEP_mvt").agg(pl.col("sec0").mean().alias("blk_on_minute_grid"),
                                                       ((pl.col("t_block") % 60 - 30).abs() > 24).mean().alias("blk_within6s_of_minute")).sort("ADEP_mvt"))
# BLOCK class x physical reference
show("copy_classes_push_ref", P.group_by("ADEP_mvt", "blk_class").agg(
    pl.len().alias("n"), pl.col("d_blk").median().alias("med"), pl.col("d_blk").quantile(.25).alias("q25"),
    pl.col("d_blk").quantile(.75).alias("q75"), (pl.col("d_blk").abs() > 180).mean().alias("gt180"),
    (pl.col("d_blk").abs() > 300).mean().alias("gt300")).sort("ADEP_mvt", "blk_class"))
U = J.filter(pl.col("UB").is_not_null())
show("late_block_upper_bound", U.group_by("ADEP_mvt").agg(
    pl.len().alias("n_ub"), (pl.col("ub_blk") > 60).mean().alias("BLK_after_taxi_60"),
    (pl.col("ub_blk") > 300).mean().alias("BLK_after_taxi_300"),
    (pl.col("ub_a3") > 60).mean().alias("A3_after_taxi_60"), (pl.col("ub_a3") > 300).mean().alias("A3_after_taxi_300"),
    pl.col("ub_blk").median().alias("ub_blk_med"), pl.col("ub_a3").median().alias("ub_a3_med")).sort("ADEP_mvt"))
show("late_block_by_class", U.group_by("ADEP_mvt", "blk_class").agg(
    pl.len().alias("n"), (pl.col("ub_blk") > 60).mean().alias("BLK_after_taxi_60"),
    (pl.col("ub_blk") > 300).mean().alias("BLK_after_taxi_300"),
    (pl.col("ub_blk") < -1800).mean().alias("BLK_30min_before_taxi")).sort("ADEP_mvt", "blk_class"))
show("class_shares", J.group_by("ADEP_mvt").agg(pl.len(), *[(pl.col("blk_class") == k).mean().alias(k) for k in
                                                           ["=SCHED", "=EOBT_1", "~AOBT_3", "other"]],
                                                (pl.col("t_aobt3").is_null()).mean().alias("no_AOBT3")).sort("ADEP_mvt"))
# pooled across airports: is the BLOCK-AOBT_3 agreement a copy? where BLOCK ~ AOBT_3, how close is each to push onset
show("aobt3_equal_rows_push_ref", P.filter(pl.col("blk_class") == "~AOBT_3").group_by("ADEP_mvt").agg(
    pl.len(), (pl.col("t_block") == pl.col("t_aobt3")).mean().alias("exact_equal"),
    (pl.col("t_aobt3") - pl.col("t_block")).median().alias("a3_minus_blk_med"), pl.col("d_blk").median().alias("blk_on_med")).sort("ADEP_mvt"))

# figure: BLOCK - onset and AOBT_3 - onset on pushback rows, per airport
apts = [a for a in ORDER if P.filter(pl.col("ADEP_mvt") == a).height >= 20]
fig, axes = plt.subplots(1, 2, figsize=(12, 0.6 * len(apts) + 1.8), sharey=False)
for ax, col, ttl in [(axes[0], "d_blk", "airport BLOCK - ADS-B pushback onset"), (axes[1], "d_a3", "NM AOBT_3 - ADS-B pushback onset")]:
    data = [P.filter(pl.col("ADEP_mvt") == a)[col].drop_nulls().to_numpy() for a in apts]
    ax.boxplot(data, vert=False, tick_labels=[f"{a} (n={len(d)})" for a, d in zip(apts, data)] if ax is axes[0] else apts, whis=(10, 90),
               showfliers=False, medianprops={"color": "#c0392b"})
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlim(-400, 400)
    ax.set_xlabel("seconds (box = IQR, whiskers = P10-P90)")
    ax.set_title(ttl, fontsize=10)
    ax.invert_yaxis()
fig.suptitle(f"Pushback rows, {len(days)} days of 2025 ({', '.join(days)}); ADS-B: adsb.lol (ODbL)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(A, "fig_offsets_push.png"), dpi=130)

# figure 2: histogram of BLOCK - onset at the five airports (fine bins)
fig, axes = plt.subplots(len(FIVE), 1, figsize=(7, 9), sharex=True)
for ax, a in zip(axes, FIVE):
    x = P.filter(pl.col("ADEP_mvt") == a)["d_blk"].drop_nulls().to_numpy()
    ax.hist(np.clip(x, -300, 300), bins=np.arange(-300, 305, 5), color="#2c7fb8")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_ylabel(a)
axes[-1].set_xlabel("BLOCK - ADS-B pushback onset (s), 5 s bins, clipped at +-300")
fig.tight_layout()
fig.savefig(os.path.join(A, "fig_hist_blk_onset.png"), dpi=130)
print("figures written")

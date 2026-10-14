"""Apply v47's LIRF label-artefact overlays (and optionally the lane-B NM-blind / ucmix deltas and the LOBT clip) on
top of a base submission file. Rows that no layer touches keep their base value bit for bit.

ASSUMED ORDER (the v47 order): base blend -> ADS-B residual layer -> [THIS SCRIPT: lane-B deltas + LIRF overlays]
-> LOBT clip (here only with --lobt-clip; otherwise run it afterwards). So --base must already carry the ADS-B
residual and must NOT carry any of the layers below.

Layers (all reproduce the v47 production scripts in /home/user/prc-data-challenge/analysis/):
  M  LIRF matched copy overlay (local_lirf_copy_overlay.py apply): q = P(schedule copy) from a LightGBM classifier
     (binary, 63 leaves, 1 thread, seed 1) refit on all 12 months of 2025 LIRF matched rows with
     int(99 * 1.1) = 108 rounds; p' = round(p + 0.4 q (MVT_minus_SCHED - p)), kept at p if negative.
     MVT_minus_SCHED = feature `mvt_minus_sched`. Applied to every LIRF matched ranking row (AOBT_3 present).
  U  lane-B deltas (with --laneB): ucmix delta on LIRF unmatched rows in its gate (0 < ms < 8 h), except the rows
     that rule B or C below claims (v47: "the 6 time-zone-class rows and the 3 copy-band rows keep our values");
     NM-blind delta on unmatched rows outside LIRF. Deltas = laneB_deltas.py outputs (blend - plain).
  C  LIRF unmatched copy band (local_lirf_artefacts_apply.py rule C): 7 <= (ms - 1100)/3600 <= 13.5 -> ms.
  B  LIRF unmatched time-zone class (rule B): 0 <= ms <= 3599 and the FLIGHT prefix had >= 1 such 2025 row:
     q (Rome UTC offset at SCHED + 865) + (1 - q) p, q = tz / (n + 1) per prefix; p = base value (pre-ucmix).
  (rule C7r, the "one-day" class, is NOT applied: dropped after v41.)
  Here ms = MVT - SCHED from the raw ranking file; unmatched = PHASE DEP and AOBT_3_flt null.
  L  LOBT clip (with --lobt-clip; local_lobt_clip.py): clip to [ceil(mvt_minus_lobt - 3606), ceil(mvt_minus_lobt + 3606)].

Computation is in float; the result is rounded once at the end when the base column is an integer type.

    /home/user/prc-work/.venv/bin/python /home/user/prc-final/overlays/apply_overlays.py --base <in.parquet> \\
        --out <out.parquet> [--laneB] [--lobt-clip] [--copy-feat fixed|unfixed] [--raw-dir ..] [--feat-dir ..]
"""
import argparse
import glob
import os

import lightgbm as lgb
import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
A = 0.4
ROUNDS = int(99 * 1.1)  # production: copy_LIRF.json best_iter 99, apply() uses int(best_iter * 1.1) = 108
DAY, T_TZ = 86400, 865
DROP = {"TAXITIME_SEC_mvt", "copy", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"}
P = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=1, seed=1)


def copy_q(feat_train, feat_rank, apt="LIRF"):
    """q = P(copy) for the LIRF matched ranking rows; identical to local_lirf_copy_overlay.frames() + apply()."""
    T = (pl.scan_parquet(feat_train).filter((pl.col("ADEP_mvt") == apt) & pl.col("mvt_minus_aobt3").is_not_null()).collect()
         .with_columns(((pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_sched")).abs() <= 12).alias("copy")))
    R = pl.scan_parquet(feat_rank).filter((pl.col("ADEP_mvt") == apt) & pl.col("mvt_minus_aobt3").is_not_null()).collect()
    num = [c for c, t in T.schema.items() if c not in DROP and t.is_numeric() and c in R.columns]
    cat = [c for c, t in T.schema.items() if c not in DROP and t == pl.String and c in R.columns]
    codes = {c: {v: i for i, v in enumerate(sorted(set(T[c].drop_nulls().to_list()) | set(R[c].drop_nulls().to_list())))}
             for c in cat}

    def X(df):
        return df.select(num + [pl.col(c).replace_strict(codes[c], default=None, return_dtype=pl.Int32).alias(c) for c in cat]
                         ).to_numpy().astype(np.float64)
    mdl = lgb.train(P, lgb.Dataset(X(T), T["copy"].to_numpy().astype(int), feature_name=num + cat, categorical_feature=cat), ROUNDS)
    q = mdl.predict(X(R))
    return pl.DataFrame({"MVT_ID_mvt": R["MVT_ID_mvt"], "q": q, "ms_feat": R["mvt_minus_sched"].cast(pl.Float64)}), float(T["copy"].mean())


def lirf_unmatched(df):
    off = 3600 + pl.col("SCHED_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.dst_offset().dt.total_seconds()
    return (df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("AOBT_3_flt").is_null())
            .with_columns(ms=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                          pre=pl.col("FLIGHT_mvt").str.slice(0, 3).fill_null("?"), off=off))


def unmatched_rules(raw_dir):
    """Rule B / C targets for LIRF unmatched ranking rows: MVT_ID_mvt, rule, ms, off, qtz (rule B), FLIGHT_mvt."""
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "FLIGHT_mvt", "TAXITIME_SEC_mvt"]
    tr = lirf_unmatched(pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{raw_dir}/training_*.parquet"))]))
    small = tr.filter(pl.col("ms").is_between(0, 3599)).with_columns(
        tz=((pl.col("TAXITIME_SEC_mvt") - pl.col("off")).is_between(240, 2700)) & ((pl.col("TAXITIME_SEC_mvt") - pl.col("ms")).abs() > 12))
    qp = small.group_by("pre").agg(t=pl.col("tz").sum(), n=pl.len()).filter(pl.col("t") >= 1)
    R = lirf_unmatched(pl.read_parquet(f"{raw_dir}/ranking.parquet", columns=cols[:-1])).join(qp, on="pre", how="left")
    R = R.with_columns(dh=(pl.col("ms") - 1100) / 3600, qtz=(pl.col("t") / (pl.col("n") + 1)).fill_null(0.0))
    rule = (pl.when(pl.col("ms") >= 50400).then(pl.lit("-"))  # C7r slot: dropped after v41, keeps base
            .when(pl.col("dh").is_between(7, 13.5)).then(pl.lit("C"))
            .when(pl.col("ms").is_between(0, 3599) & (pl.col("qtz") > 0)).then(pl.lit("B"))
            .otherwise(pl.lit("-")))
    return R.with_columns(rule=rule).select("MVT_ID_mvt", "FLIGHT_mvt", "rule", pl.col("ms").cast(pl.Float64), "off", "qtz", "t", "n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-dir", default="/home/user/pb-work/data/raw_final")
    ap.add_argument("--feat-dir", default="/home/user/pb-work/data/work_final")
    ap.add_argument("--copy-feat", choices=["fixed", "unfixed"], default="fixed",
                    help="ranking features for the copy classifier: feat_rank.parquet (union-fixed fl_/seq_, default) "
                         "or feat_rank_unfixed.parquet (v47 used the unfixed Jan+Jul frame; q corr 0.9994 between the two)")
    ap.add_argument("--laneB", action="store_true", help="also add the NM-blind and ucmix lane-B deltas")
    ap.add_argument("--nmb", default=os.path.join(HERE, "out", "nmb_laneB_delta_final.parquet"))
    ap.add_argument("--ucmix", default=os.path.join(HERE, "out", "ucmix_laneB_delta_final.parquet"))
    ap.add_argument("--lobt-clip", action="store_true", help="apply the LOBT clip as the last step")
    ap.add_argument("--no-copy", action="store_true", help="skip the matched copy overlay (M)")
    ap.add_argument("--q-cache", default=None, help="parquet with MVT_ID_mvt, q, ms_feat to reuse instead of refitting")
    ap.add_argument("--q-out", default=None, help="write the copy q table here")
    ap.add_argument("--report", default=None, help="write the per-row change table (all touched rows) here")
    a = ap.parse_args()

    base = pl.read_parquet(a.base)
    assert base["MVT_ID_mvt"].is_unique().all() and base["TAXITIME_SEC_mvt"].null_count() == 0
    int_out = base.schema["TAXITIME_SEC_mvt"].is_integer()
    W = base.select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("p0")).with_columns(p=pl.col("p0"),
                                                                                                   layer=pl.lit(""))

    # ---- lane B: unmatched LIRF rules (targets), computed first so ucmix can skip the rows they claim
    U = unmatched_rules(a.raw_dir)
    claimed = U.filter(pl.col("rule") != "-")
    if a.laneB:
        dn = pl.read_parquet(a.nmb).select("MVT_ID_mvt", pl.col("delta").alias("d_nmb"), "ADEP_mvt")
        du = pl.read_parquet(a.ucmix).select("MVT_ID_mvt", pl.col("delta").alias("d_uc"))
        D = dn.join(du, on="MVT_ID_mvt", how="full", coalesce=True).with_columns(
            pl.col("d_nmb").fill_null(0.0), pl.col("d_uc").fill_null(0.0))
        assert D.filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("d_nmb") != 0)).height == 0
        D = D.with_columns(pl.when(pl.col("MVT_ID_mvt").is_in(claimed["MVT_ID_mvt"].implode())).then(0.0)
                           .otherwise(pl.col("d_uc")).alias("d_uc"))
        D = D.with_columns(d=pl.col("d_nmb") + pl.col("d_uc"))
        miss = D.join(W, on="MVT_ID_mvt", how="anti").height
        W = W.join(D.select("MVT_ID_mvt", "d", "d_nmb", "d_uc"), on="MVT_ID_mvt", how="left", maintain_order="left").with_columns(
            p=pl.col("p") + pl.col("d").fill_null(0.0),
            layer=pl.when(pl.col("d_nmb").fill_null(0) != 0).then(pl.lit("nmb"))
            .when(pl.col("d_uc").fill_null(0) != 0).then(pl.lit("ucmix")).otherwise(pl.col("layer"))).drop("d", "d_nmb", "d_uc")
        print(f"lane-B deltas: nmb rows {int((D['d_nmb'] != 0).sum())}, ucmix rows {int((D['d_uc'] != 0).sum())} "
              f"(ucmix skipped on {claimed.height} rule B/C rows), delta rows not in base: {miss}")

    # ---- rules C and B on LIRF unmatched rows; B blends from the base value (pre-ucmix), as in v47
    W = W.join(claimed.select("MVT_ID_mvt", "rule", "ms", "off", "qtz"), on="MVT_ID_mvt", how="left", maintain_order="left")
    newu = (pl.when(pl.col("rule") == "C").then(pl.col("ms"))
            .when(pl.col("rule") == "B").then(pl.col("qtz") * (pl.col("off") + T_TZ) + (1 - pl.col("qtz")) * pl.col("p0"))
            .otherwise(pl.col("p")))
    W = W.with_columns(p=newu, layer=pl.when(pl.col("rule").is_not_null()).then(pl.lit("rule_") + pl.col("rule"))
                       .otherwise(pl.col("layer"))).drop("rule", "ms", "off", "qtz")
    print(f"LIRF unmatched rules in ranking: C {int((claimed['rule'] == 'C').sum())}, B {int((claimed['rule'] == 'B').sum())}; "
          f"in base: {W.filter(pl.col('layer').str.starts_with('rule_')).height}")

    # ---- matched LIRF copy overlay
    if not a.no_copy:
        if a.q_cache:
            Q = pl.read_parquet(a.q_cache)
        else:
            fr = os.path.join(a.feat_dir, "feat_rank.parquet" if a.copy_feat == "fixed" else "feat_rank_unfixed.parquet")
            Q, rate = copy_q(os.path.join(a.feat_dir, "feat_train.parquet"), fr)
            print(f"copy classifier: {ROUNDS} rounds, ranking features {fr}, LIRF matched rows {Q.height}, "
                  f"mean q {Q['q'].mean():.4f} (2025 copy rate {rate:.4f})")
            if a.q_out:
                Q.write_parquet(a.q_out)
        W = W.join(Q, on="MVT_ID_mvt", how="left", maintain_order="left")
        newm = pl.col("p") + A * pl.col("q") * (pl.col("ms_feat") - pl.col("p"))
        W = W.with_columns(p=pl.when(pl.col("q").is_null()).then(pl.col("p")).when(newm < 0).then(pl.col("p")).otherwise(newm),
                           layer=pl.when(pl.col("q").is_not_null()).then(pl.lit("copy")).otherwise(pl.col("layer"))).drop("q", "ms_feat")

    # ---- LOBT clip (last)
    if a.lobt_clip:
        f = pl.read_parquet(os.path.join(a.feat_dir, "feat_rank.parquet"), columns=["MVT_ID_mvt", "mvt_minus_lobt"])
        W = W.join(f, on="MVT_ID_mvt", how="left", maintain_order="left")
        lo, hi = (pl.col("mvt_minus_lobt") - 3606).ceil(), (pl.col("mvt_minus_lobt") + 3606).ceil()
        W = W.with_columns(pr=pl.col("p").round(0) if int_out else pl.col("p"))
        clipped = pl.max_horizontal(pl.min_horizontal(pl.col("pr"), hi), lo)
        W = W.with_columns(pc=pl.when(pl.col("mvt_minus_lobt").is_null()).then(pl.col("pr")).otherwise(clipped))
        moved = pl.col("pc") != pl.col("pr")
        nclip = W.filter(moved).height
        W = W.with_columns(layer=pl.when(moved).then(pl.col("layer") + pl.lit("+lobt")).otherwise(pl.col("layer")),
                           p=pl.when(moved).then(pl.col("pc")).otherwise(pl.col("p"))).drop("pc", "pr", "mvt_minus_lobt")
        print(f"LOBT clip: {nclip} rows clipped")

    # ---- write; untouched rows keep the base value exactly
    val = pl.col("p").round(0) if int_out else pl.col("p")
    W = W.with_columns(new=pl.when(pl.col("layer") == "").then(pl.col("p0")).otherwise(val))
    out = base.with_columns(W["new"].cast(base.schema["TAXITIME_SEC_mvt"]).alias("TAXITIME_SEC_mvt"))
    assert (out["MVT_ID_mvt"] == base["MVT_ID_mvt"]).all() and out.null_count().sum_horizontal().item() == 0
    out.write_parquet(a.out)
    d = (W["new"] - W["p0"])
    print(f"{a.out}: {out.height} rows, {int((d != 0).sum())} changed, sum d^2/N {float((d ** 2).sum()) / out.height:.2f}")
    print(W.with_columns(d=d).filter(pl.col("layer") != "").group_by("layer").agg(
        n=pl.len(), changed=(pl.col("d") != 0).sum(), mean_d=pl.col("d").mean().round(1),
        rms_d=(pl.col("d") ** 2).mean().sqrt().round(1)).sort("layer"))
    if a.report:
        W.with_columns(d=d).filter(pl.col("layer") != "").select("MVT_ID_mvt", "layer", "p0", "new", "d").write_parquet(a.report)


if __name__ == "__main__":
    main()

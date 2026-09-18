"""Data audit 2026-09-18 -- lens: the sched-filled BLOCK phenomenon, everywhere it occurs.

No model training. Group means, correlations and rank statistics only.
Run with POLARS_MAX_THREADS=1.

Inputs (read-only):
  data/work/feat_train.parquet          2,085,047 training departures
  data/work/feat_rank.parquet             344,841 ranking departures
  experiments/oof_main-48c19bc830.parquet  OOF predictions (672,428 non-null)

Definitions (plan.md 2026-09-16 s0):
  y      = TAXITIME_SEC_mvt = MVT - BLOCK
  ms     = mvt_minus_sched  = MVT - SCHED
  proxy  = MVT - AOBT_3
  matched = AOBT_3 non-null (routing definition)
  fill (f10) = |BLOCK - SCHED| <= 10 s
  far        = |AOBT_3 - SCHED| > 600 s          (matched rows only)
  dfill      = fill AND |BLOCK - AOBT_3| > 600 s (matched rows only; "distorted fill")
"""

import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(200)

ROOT = "/home/user/prc-data-challenge"
TRAIN = f"{ROOT}/data/work/feat_train.parquet"
RANK = f"{ROOT}/data/work/feat_rank.parquet"
OOF = f"{ROOT}/experiments/oof_main-48c19bc830.parquet"


def hdr(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)
    sys.stdout.flush()


def show(name, df):
    print(f"--- {name}")
    print(df)
    sys.stdout.flush()


def secdiff(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds()


def build(path, is_train):
    """Narrow frame with the fill predicates and every candidate driver."""
    cols = [
        "MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
        "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt",
        "FLIGHT_mvt", "FLIGHT_RULE_mvt",
        "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt",
        "AOBT_3_flt", "ARVT_1_flt",
        "mvt_minus_aobt3", "mvt_minus_sched", "month", "hour_utc", "dow",
        "planned_flight_s", "dest_distance_km", "ADES_mvt",
    ]
    if is_train:
        cols += ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
    lf = pl.scan_parquet(path).select(cols)

    e = [
        pl.col("MVT_ID_mvt").alias("mid"),
        pl.col("ADEP_mvt").alias("adep"),
        pl.col("ADES_mvt").alias("ades"),
        pl.col("RUNWAY_mvt").alias("rwy"),
        pl.col("STAND_mvt").alias("stand"),
        pl.col("AIRCRAFT_OPERATOR_flt").alias("op"),
        pl.col("MARKET_SEGMENT_flt").alias("seg"),
        pl.col("FLIGHT_TYPE_flt").alias("ftype"),
        pl.col("WK_TBL_CAT_flt").alias("wtc"),
        pl.col("FLIGHT_mvt").str.slice(0, 3).alias("fpfx"),
        pl.col("FLIGHT_RULE_mvt").alias("frule"),
        pl.col("month").alias("month"),
        pl.col("hour_utc").alias("hour"),
        pl.col("dow").alias("dow"),
        pl.col("mvt_minus_sched").alias("ms"),
        pl.col("mvt_minus_aobt3").alias("proxy"),
        pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("actype_null"),
        pl.col("mvt_minus_aobt3").is_null().cast(pl.Int8).alias("unm"),
        # driver candidates that survive into the ranking set
        secdiff("SCHED_TIME_UTC_mvt", "EOBT_1_flt").alias("sched_eobt"),
        secdiff("SCHED_TIME_UTC_mvt", "LOBT_flt").alias("sched_lobt"),
        secdiff("SCHED_TIME_UTC_mvt", "IOBT_flt").alias("sched_iobt"),
        secdiff("LOBT_flt", "IOBT_flt").alias("lobt_iobt"),
        secdiff("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("as_"),
        pl.col("MVT_TIME_UTC_mvt").dt.second().alias("mvt_sec"),
        pl.col("SCHED_TIME_UTC_mvt").dt.second().alias("sched_sec"),
        pl.col("EOBT_1_flt").dt.second().alias("eobt_sec"),
        pl.col("LOBT_flt").dt.second().alias("lobt_sec"),
        pl.col("SCHED_TIME_UTC_mvt").dt.minute().alias("sched_min"),
        pl.col("planned_flight_s").alias("pfs"),
        pl.col("ARVT_1_flt").is_null().cast(pl.Int8).alias("arvt1_null"),
        pl.col("EOBT_1_flt").is_null().cast(pl.Int8).alias("eobt_null"),
        pl.col("LOBT_flt").is_null().cast(pl.Int8).alias("lobt_null"),
        pl.col("IOBT_flt").is_null().cast(pl.Int8).alias("iobt_null"),
    ]
    if is_train:
        e += [
            pl.col("TAXITIME_SEC_mvt").alias("y"),
            secdiff("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("bs"),
            secdiff("BLOCK_TIME_UTC_mvt", "AOBT_3_flt").alias("ba"),
            pl.col("BLOCK_TIME_UTC_mvt").dt.second().alias("block_sec"),
        ]
    lf = lf.select(e)
    if is_train:
        lf = lf.with_columns(
            (pl.col("bs").abs() <= 10).cast(pl.Int8).alias("f10"),
            (pl.col("bs").abs() <= 6).cast(pl.Int8).alias("f6"),
            (pl.col("bs").abs() <= 120).cast(pl.Int8).alias("f120"),
        )
    lf = lf.with_columns((pl.col("as_").abs() > 600).cast(pl.Int8).alias("far"))
    return lf.collect()


def rate(df, by, flag="f10", minn=1):
    return (
        df.group_by(by)
        .agg(pl.len().alias("n"), pl.col(flag).mean().round(4).alias(flag))
        .filter(pl.col("n") >= minn)
        .sort(flag, descending=True)
    )


def r2_groups(df, by, flag="f10"):
    """Between-group variance share of a 0/1 flag using in-sample group means."""
    p = df[flag].mean()
    g = df.group_by(by).agg(pl.len().alias("n"), pl.col(flag).mean().alias("p"))
    num = (g["n"] * (g["p"] - p) ** 2).sum()
    den = df.height * p * (1 - p)
    return len(g), float(num / den)


def auc(score, label):
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=float)
    ok = ~np.isnan(score)
    score, label = score[ok], label[ok]
    n1, n0 = label.sum(), (1 - label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan"), len(score)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r1 = ranks[label[order] == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0)), len(score)


def lomo_score(df, by, flag="f10", prior_w=20.0):
    """Leave-one-month-out smoothed group mean -> out-of-sample R2 and AUC.

    Pure group means with a Beta-prior shrinkage (no model fit, no optimisation).
    """
    key = by if isinstance(by, list) else [by]
    out_pred, out_lab = [], []
    gp = df[flag].mean()
    for m in sorted(df["month"].unique().to_list()):
        tr = df.filter(pl.col("month") != m)
        te = df.filter(pl.col("month") == m)
        if te.height == 0 or tr.height == 0:
            continue
        g = tr.group_by(key).agg(pl.len().alias("gn"), pl.col(flag).sum().alias("gs"))
        g = g.with_columns(((pl.col("gs") + prior_w * gp) / (pl.col("gn") + prior_w)).alias("phat"))
        j = te.join(g.select(key + ["phat"]), on=key, how="left").with_columns(
            pl.col("phat").fill_null(gp)
        )
        out_pred.append(j["phat"].to_numpy())
        out_lab.append(j[flag].to_numpy())
    pred = np.concatenate(out_pred)
    lab = np.concatenate(out_lab).astype(float)
    sse = float(((lab - pred) ** 2).sum())
    sst = float(((lab - lab.mean()) ** 2).sum())
    a, _ = auc(pred, lab)
    return 1 - sse / sst, a, len(pred)


def main():
    hdr("A. Frames, sanity, definitions")
    tr = build(TRAIN, True)
    print("train rows", tr.height, "| unmatched", int(tr['unm'].sum()),
          "| f10", int(tr['f10'].sum()), "| f6", int(tr['f6'].sum()), "| f120", int(tr['f120'].sum()))
    print("null bs/ba:", tr["bs"].null_count(), tr["ba"].null_count())
    # label identity re-check (cheap)
    print("rows where (y - ms) != -(bs):", int(((tr["y"] - tr["ms"]) + tr["bs"]).abs().sum()))
    tr = tr.with_columns(
        pl.when(pl.col("unm") == 0)
        .then(((pl.col("f10") == 1) & (pl.col("ba").abs() > 600)).cast(pl.Int8))
        .otherwise(None)
        .alias("dfill")
    )
    print("matched dfill total:", int(tr["dfill"].sum()))

    hdr("B. Fill rate by airport x matched x month (stability over the 12 training months)")
    m = tr.filter(pl.col("unm") == 0)
    u = tr.filter(pl.col("unm") == 1)
    show("B1 airport x matched", tr.group_by(["adep", "unm"]).agg(
        pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10"),
        pl.col("f6").mean().round(4).alias("f6"), pl.col("f120").mean().round(4).alias("f120"),
        pl.col("dfill").mean().round(4).alias("dfill"),
        pl.col("y").median().alias("y_med"),
    ).sort(["unm", "f10"], descending=[False, True]))

    show("B2 matched fill by airport x month (pivot)",
         m.group_by(["adep", "month"]).agg(pl.col("f10").mean().round(4).alias("f"))
          .pivot(on="month", index="adep", values="f").sort("adep"))
    show("B3 matched dfill by airport x month (pivot)",
         m.group_by(["adep", "month"]).agg(pl.col("dfill").mean().round(4).alias("f"))
          .pivot(on="month", index="adep", values="f").sort("adep"))
    show("B4 LIRF unmatched by month",
         u.filter(pl.col("adep") == "LIRF").group_by("month").agg(
             pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10"),
             pl.col("ms").median().alias("ms_med"), pl.col("y").median().alias("y_med")).sort("month"))
    show("B5 non-LIRF unmatched by month",
         u.filter(pl.col("adep") != "LIRF").group_by("month").agg(
             pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10")).sort("month"))

    hdr("C. Fill rate by operator / segment / hour / stand / runway (matched rows)")
    show("C1 segment, all matched", rate(m, "seg").join(
        rate(m.filter(pl.col("adep") == "LIRF"), "seg").rename({"n": "n_lirf", "f10": "f10_lirf"}),
        on="seg", how="left"))
    show("C2 matched fill by hour, LIRF vs rest", m.group_by("hour").agg(
        pl.len().alias("n"),
        pl.col("f10").mean().round(4).alias("f10_all"),
        (pl.col("f10").filter(pl.col("adep") == "LIRF")).mean().round(4).alias("f10_lirf"),
        (pl.col("f10").filter(pl.col("adep") != "LIRF")).mean().round(4).alias("f10_other"),
        (pl.col("dfill").filter(pl.col("adep") != "LIRF")).mean().round(4).alias("dfill_other"),
    ).sort("hour"))
    show("C3 top-25 operators by fill (matched, n>=500, all airports)",
         rate(m, "op", minn=500).head(25).join(
             m.group_by("op").agg(pl.col("adep").mode().first().alias("adep_mode"),
                                  pl.col("seg").mode().first().alias("seg"),
                                  pl.col("adep").n_unique().alias("n_adep"),
                                  pl.col("dfill").mean().round(4).alias("dfill")), on="op"))
    show("C4 operator spread: how many matched rows sit in fill>=0.5 / <=0.02 operators",
         rate(m, "op", minn=1).with_columns(
             pl.when(pl.col("f10") >= 0.5).then(pl.lit("hi>=.50"))
             .when(pl.col("f10") >= 0.2).then(pl.lit("mid.20-.50"))
             .when(pl.col("f10") >= 0.05).then(pl.lit("lo.05-.20"))
             .otherwise(pl.lit("vlo<.05")).alias("bucket"))
         .group_by("bucket").agg(pl.len().alias("n_ops"), pl.col("n").sum().alias("rows")).sort("bucket"))
    for ap in ["LIRF", "LEBL", "LTFM", "EDDM"]:
        sub = m.filter(pl.col("adep") == ap)
        show(f"C5 {ap} stand fill (n>=300, top10 / bottom5)",
             pl.concat([rate(sub, "stand", minn=300).head(10), rate(sub, "stand", minn=300).tail(5)]))
        show(f"C6 {ap} runway fill (n>=300)", rate(sub, "rwy", minn=300))
    show("C7 LIRF unmatched: fill by runway / stand / flight prefix",
         pl.concat([
             rate(u.filter(pl.col("adep") == "LIRF"), "rwy", minn=30).with_columns(pl.lit("rwy").alias("dim")).rename({"rwy": "key"}),
             rate(u.filter(pl.col("adep") == "LIRF"), "stand", minn=30).with_columns(pl.lit("stand").alias("dim")).rename({"stand": "key"}),
             rate(u.filter(pl.col("adep") == "LIRF"), "fpfx", minn=30).with_columns(pl.lit("fpfx").alias("dim")).rename({"fpfx": "key"}),
         ], how="vertical_relaxed"))

    hdr("D. The 9 non-LIRF airports: are their filled matched rows the same phenomenon?")
    show("D1 filled matched rows by airport: is |AOBT3-SCHED| small (on-time coincidence) or large (fill)?",
         m.filter(pl.col("f10") == 1).group_by("adep").agg(
             pl.len().alias("n_fill"),
             pl.col("far").mean().round(4).alias("far_share"),
             pl.col("as_").abs().median().alias("abs_as_med"),
             pl.col("as_").abs().quantile(0.9).alias("abs_as_p90"),
             pl.col("ba").abs().median().alias("abs_ba_med"),
             pl.col("ba").abs().quantile(0.99).alias("abs_ba_p99"),
             (pl.col("ba").abs() > 600).mean().round(4).alias("ba_gt600"),
             pl.col("y").median().alias("y_med"),
             pl.col("y").std().round(1).alias("y_sd"),
             (pl.col("y") - pl.col("proxy")).std().round(1).alias("resid_sd"),
         ).sort("ba_gt600", descending=True))
    show("D2 control: NOT-filled matched rows, same statistics",
         m.filter(pl.col("f10") == 0).group_by("adep").agg(
             pl.len().alias("n"), pl.col("far").mean().round(4).alias("far_share"),
             pl.col("ba").abs().median().alias("abs_ba_med"),
             (pl.col("ba").abs() > 600).mean().round(4).alias("ba_gt600"),
             pl.col("y").median().alias("y_med"),
             (pl.col("y") - pl.col("proxy")).std().round(1).alias("resid_sd")).sort("adep"))
    show("D3 dfill (fill AND |BLOCK-AOBT3|>600) by airport",
         m.group_by("adep").agg(
             pl.len().alias("n_matched"),
             pl.col("f10").sum().alias("n_fill"),
             pl.col("dfill").sum().alias("n_dfill"),
             pl.col("dfill").mean().round(5).alias("dfill_rate"),
             (pl.col("dfill").sum() / pl.col("f10").sum()).round(4).alias("dfill_of_fill"),
         ).sort("dfill_rate", descending=True))
    show("D4 dfill rows: label distortion size by airport",
         m.filter(pl.col("dfill") == 1).group_by("adep").agg(
             pl.len().alias("n"),
             (pl.col("y") - pl.col("proxy")).median().alias("resid_med"),
             (pl.col("y") - pl.col("proxy")).quantile(0.9).alias("resid_p90"),
             pl.col("y").median().alias("y_med"), pl.col("proxy").median().alias("proxy_med"),
             (pl.col("y") >= 7200).mean().round(4).alias("y_ge_2h"),
         ).sort("n", descending=True))
    show("D5 non-LIRF dfill by operator (n>=30)",
         rate(m.filter(pl.col("adep") != "LIRF"), "op", flag="dfill", minn=30).head(20).join(
             m.filter(pl.col("adep") != "LIRF").group_by("op").agg(
                 pl.col("adep").mode().first().alias("adep_mode"),
                 pl.col("seg").mode().first().alias("seg"),
                 pl.col("f10").mean().round(4).alias("f10")), on="op"))
    show("D6 fill rate by |AOBT3-SCHED| band x airport (matched) -- recomputed with wider bands",
         m.with_columns(pl.col("as_").abs().cut([60, 300, 600, 1800, 3600, 7200],
                                                labels=["<=60", "60-300", "300-600", "600-1800",
                                                        "1800-3600", "3600-7200", ">7200"]).alias("band"))
          .group_by(["adep", "band"]).agg(pl.col("f10").mean().round(4).alias("f"))
          .pivot(on="band", index="adep", values="f").sort("adep"))

    hdr("E. Is the fill predictable from ranking-visible columns? (matched rows)")
    dims = {
        "adep": ["adep"], "op": ["op"], "seg": ["seg"], "hour": ["hour"], "stand": ["adep", "stand"],
        "rwy": ["adep", "rwy"], "far": ["far"], "month": ["month"], "ades": ["ades"], "wtc": ["wtc"],
        "ftype": ["ftype"],
        "adep+op": ["adep", "op"], "adep+op+far": ["adep", "op", "far"],
        "adep+op+hour": ["adep", "op", "hour"],
        "adep+op+hour+far": ["adep", "op", "hour", "far"],
        "adep+stand": ["adep", "stand"], "adep+op+stand": ["adep", "op", "stand"],
        "adep+op+stand+far": ["adep", "op", "stand", "far"],
    }
    rows = []
    for name, by in dims.items():
        for scope, d in (("all-matched", m), ("LIRF-matched", m.filter(pl.col("adep") == "LIRF")),
                         ("other-matched", m.filter(pl.col("adep") != "LIRF"))):
            ng, r2 = r2_groups(d, by, "f10")
            rows.append({"scope": scope, "dim": name, "groups": ng, "r2_insample": round(r2, 4)})
    show("E1 in-sample between-group R2 of the fill flag", pl.DataFrame(rows).pivot(
        on="scope", index=["dim"], values="r2_insample", aggregate_function="first"))

    rows = []
    for name in ["op", "adep+op", "adep+op+far", "adep+op+hour+far", "adep+op+stand", "adep+op+stand+far", "far", "adep+stand"]:
        by = dims[name]
        for scope, d in (("all-matched", m), ("LIRF-matched", m.filter(pl.col("adep") == "LIRF")),
                         ("other-matched", m.filter(pl.col("adep") != "LIRF"))):
            r2o, a, n = lomo_score(d, by, "f10")
            rows.append({"scope": scope, "dim": name, "n": n, "r2_oos": round(r2o, 4), "auc_oos": round(a, 4)})
    show("E2 leave-one-month-out group-mean score (honest)", pl.DataFrame(rows).sort(["scope", "auc_oos"], descending=[False, True]))

    # continuous drivers: point-biserial correlation with the fill flag
    conts = ["ms", "proxy", "as_", "sched_eobt", "sched_lobt", "sched_iobt", "lobt_iobt",
             "mvt_sec", "sched_min", "pfs", "hour", "month"]
    rows = []
    for c in conts:
        for scope, d in (("LIRF-matched", m.filter(pl.col("adep") == "LIRF")),
                         ("other-matched", m.filter(pl.col("adep") != "LIRF")),
                         ("LIRF-unmatched", u.filter(pl.col("adep") == "LIRF"))):
            dd = d.select([pl.col(c).cast(pl.Float64), pl.col("f10").cast(pl.Float64)]).drop_nulls()
            r = float(np.corrcoef(dd[c].to_numpy(), dd["f10"].to_numpy())[0, 1]) if dd.height > 10 else float("nan")
            rows.append({"col": c, "scope": scope, "corr_f10": round(r, 4), "n": dd.height})
    show("E3 point-biserial corr of candidate continuous drivers with the fill flag",
         pl.DataFrame(rows).pivot(on="scope", index="col", values="corr_f10", aggregate_function="first"))

    show("E4 LIRF-matched: how concentrated is the fill? rows by operator-fill decile (LOMO-free, in-sample)",
         m.filter(pl.col("adep") == "LIRF").join(
             rate(m.filter(pl.col("adep") == "LIRF"), "op").rename({"f10": "op_fill", "n": "op_n"}), on="op")
          .with_columns(pl.col("op_fill").cut([0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
                                              labels=["<.02", ".02-.05", ".05-.10", ".10-.20",
                                                      ".20-.40", ".40-.80", ">=.80"]).alias("b"))
          .group_by("b").agg(pl.len().alias("rows"), pl.col("f10").sum().alias("filled"),
                             pl.col("f10").mean().round(4).alias("rate")).sort("b"))

    hdr("F. Airport-free signals: EOBT/LOBT/IOBT and seconds-of-minute")
    show("F1 equality structure of the schedule stamps (share equal to SCHED, by airport)",
         tr.group_by("adep").agg(
             pl.len().alias("n"),
             (pl.col("sched_eobt") == 0).mean().round(4).alias("sched==eobt1"),
             (pl.col("sched_lobt") == 0).mean().round(4).alias("sched==lobt"),
             (pl.col("sched_iobt") == 0).mean().round(4).alias("sched==iobt"),
             (pl.col("lobt_iobt") == 0).mean().round(4).alias("lobt==iobt"),
             pl.col("eobt_null").mean().round(4).alias("eobt_null"),
             pl.col("sched_sec").eq(0).mean().round(4).alias("sched_sec0"),
             pl.col("block_sec").eq(0).mean().round(4).alias("block_sec0"),
             pl.col("mvt_sec").eq(0).mean().round(4).alias("mvt_sec0"),
         ).sort("adep"))
    show("F2 fill rate conditional on schedule-stamp flags (matched)",
         m.group_by([(pl.col("sched_lobt") == 0).alias("s_eq_lobt"),
                     (pl.col("sched_eobt") == 0).alias("s_eq_eobt"),
                     (pl.col("lobt_iobt") == 0).alias("lobt_eq_iobt")])
          .agg(pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10"),
               pl.col("dfill").mean().round(4).alias("dfill")).sort("n", descending=True))
    show("F3 same, LIRF only (matched)",
         m.filter(pl.col("adep") == "LIRF").group_by([(pl.col("sched_lobt") == 0).alias("s_eq_lobt"),
                                                      (pl.col("sched_eobt") == 0).alias("s_eq_eobt")])
          .agg(pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10")).sort("n", descending=True))
    show("F4 LIRF unmatched: schedule-stamp flags vs fill",
         u.filter(pl.col("adep") == "LIRF").group_by([(pl.col("sched_lobt") == 0).alias("s_eq_lobt"),
                                                      pl.col("eobt_null"), pl.col("lobt_null"),
                                                      pl.col("iobt_null")])
          .agg(pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10"),
               pl.col("ms").median().alias("ms_med")).sort("n", descending=True))
    show("F5 MVT seconds-of-minute (ranking-visible) vs fill, by airport",
         tr.with_columns(pl.min_horizontal(pl.col("mvt_sec"), 60 - pl.col("mvt_sec")).alias("mvt_off"))
           .group_by("adep").agg(
               (pl.col("mvt_off") <= 6).mean().round(4).alias("mvt_near_min"),
               pl.col("f10").mean().round(4).alias("f10"),
               (pl.col("f10").filter(pl.col("mvt_off") <= 6)).mean().round(4).alias("f10_near"),
               (pl.col("f10").filter(pl.col("mvt_off") > 6)).mean().round(4).alias("f10_far"),
           ).sort("adep"))
    show("F6 SCHED minute-of-hour granularity vs fill (matched, all airports)",
         m.with_columns((pl.col("sched_min") % 5 == 0).alias("sched_min_mult5"))
          .group_by(["sched_min_mult5"]).agg(pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10")))
    show("F7 airport-free candidate: fill rate by (sched_min%5==0) x far x seg, matched non-LIRF",
         m.filter(pl.col("adep") != "LIRF").group_by(["far", "seg"]).agg(
             pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10"),
             pl.col("dfill").mean().round(4).alias("dfill")).sort("dfill", descending=True).head(15))

    hdr("G. Ranking set: what survives, and composition Jan/Jul 2026 vs training")
    rk = build(RANK, False)
    print("rank rows", rk.height, "| unmatched", int(rk["unm"].sum()))
    show("G1 ranking null rates of the driving columns",
         rk.select([pl.col(c).null_count().alias(c) for c in
                    ["op", "seg", "stand", "rwy", "as_", "ms", "sched_eobt", "sched_lobt", "sched_iobt", "proxy"]]))
    show("G2 ranking rows by airport x month x matched, with far share",
         rk.group_by(["adep", "month", "unm"]).agg(
             pl.len().alias("n"), pl.col("far").mean().round(4).alias("far")).sort(["adep", "unm", "month"]))
    # expected filled rows in ranking under training cell rates
    for label, by, scope_tr, scope_rk in [
        ("LIRF matched, op x far", ["op", "far"], m.filter(pl.col("adep") == "LIRF"), rk.filter((pl.col("adep") == "LIRF") & (pl.col("unm") == 0))),
        ("all matched, adep x op x far", ["adep", "op", "far"], m, rk.filter(pl.col("unm") == 0)),
        ("all matched, adep x seg x far", ["adep", "seg", "far"], m, rk.filter(pl.col("unm") == 0)),
    ]:
        g = scope_tr.group_by(by).agg(pl.len().alias("gn"), pl.col("f10").mean().alias("p"),
                                      pl.col("dfill").mean().alias("pd"))
        j = scope_rk.join(g, on=by, how="left")
        print(f"--- G3 expected filled ranking rows [{label}]: rows {j.height}, "
              f"cell hit {int(j['p'].is_not_null().sum())}, "
              f"E[filled] {round(float(j['p'].fill_null(scope_tr['f10'].mean()).sum()), 1)}, "
              f"E[dfill] {round(float(j['pd'].fill_null(scope_tr['dfill'].mean()).sum()), 1)}, "
              f"flat rate {round(float(scope_tr['f10'].mean()), 4)} -> {round(j.height * float(scope_tr['f10'].mean()), 1)}")
    show("G4 ranking vs training composition on the drivers (matched), LIRF",
         pl.concat([
             m.filter(pl.col("adep") == "LIRF").group_by("seg").agg(pl.len().alias("n_train")).with_columns(
                 (pl.col("n_train") / m.filter(pl.col("adep") == "LIRF").height).round(4).alias("p_train")),
         ]).join(
             rk.filter((pl.col("adep") == "LIRF") & (pl.col("unm") == 0)).group_by("seg").agg(pl.len().alias("n_rank")).with_columns(
                 (pl.col("n_rank") / rk.filter((pl.col("adep") == "LIRF") & (pl.col("unm") == 0)).height).round(4).alias("p_rank")),
             on="seg", how="full", coalesce=True).sort("n_train", descending=True))
    show("G5 schedule-stamp flags: training vs ranking (all rows, by airport)",
         pl.concat([
             tr.group_by("adep").agg((pl.col("sched_lobt") == 0).mean().round(4).alias("s_eq_lobt"),
                                     (pl.col("sched_eobt") == 0).mean().round(4).alias("s_eq_eobt"),
                                     pl.col("eobt_null").mean().round(4).alias("eobt_null")).with_columns(pl.lit("train").alias("set")),
             rk.group_by("adep").agg((pl.col("sched_lobt") == 0).mean().round(4).alias("s_eq_lobt"),
                                     (pl.col("sched_eobt") == 0).mean().round(4).alias("s_eq_eobt"),
                                     pl.col("eobt_null").mean().round(4).alias("eobt_null")).with_columns(pl.lit("rank").alias("set")),
         ]).pivot(on="set", index="adep", values=["s_eq_lobt", "s_eq_eobt", "eobt_null"]).sort("adep"))
    show("G6 LIRF unmatched ranking vs training Jan/Jul: runway / actype-null / ms bands",
         pl.concat([
             u.filter((pl.col("adep") == "LIRF") & pl.col("month").is_in([1, 7])).group_by("month").agg(
                 pl.len().alias("n"), pl.col("actype_null").mean().round(3).alias("actype_null"),
                 (pl.col("ms") >= 7200).mean().round(3).alias("ms_ge_2h"),
                 pl.col("rwy").mode().first().alias("rwy_mode")).with_columns(pl.lit("train2025").alias("set")),
             rk.filter((pl.col("adep") == "LIRF") & (pl.col("unm") == 1)).group_by("month").agg(
                 pl.len().alias("n"), pl.col("actype_null").mean().round(3).alias("actype_null"),
                 (pl.col("ms") >= 7200).mean().round(3).alias("ms_ge_2h"),
                 pl.col("rwy").mode().first().alias("rwy_mode")).with_columns(pl.lit("rank2026").alias("set")),
         ]).sort(["month", "set"]))
    del rk

    hdr("H. What the fill costs on experiments/oof_main-48c19bc830.parquet")
    oof = pl.scan_parquet(OOF).collect()
    j = tr.select(["mid", "adep", "unm", "f10", "f120", "dfill", "far", "y", "ms"]).join(
        oof, left_on="mid", right_on="MVT_ID_mvt", how="inner").filter(pl.col("oof").is_not_null())
    j = j.with_columns(((pl.col("oof") - pl.col("y")) ** 2).alias("se"))
    tot = float(j["se"].sum())
    print("scored rows", j.height, "total SSE", f"{tot:.4e}", "RMSE", round((tot / j.height) ** 0.5, 2))
    show("H1 SSE share by fill x matched",
         j.group_by(["unm", "f10"]).agg(pl.len().alias("n"), pl.col("se").sum().alias("sse"),
                                        (pl.col("se").mean() ** 0.5).round(1).alias("rmse"))
          .with_columns((pl.col("sse") / tot).round(4).alias("sse_share")).sort(["unm", "f10"]))
    show("H2 SSE share of filled rows by airport x matched",
         j.filter(pl.col("f10") == 1).group_by(["adep", "unm"]).agg(
             pl.len().alias("n"), pl.col("se").sum().alias("sse"),
             (pl.col("se").mean() ** 0.5).round(1).alias("rmse"))
          .with_columns((pl.col("sse") / tot).round(4).alias("sse_share")).sort("sse", descending=True))
    show("H3 SSE share of NOT-filled rows by airport x matched (control)",
         j.filter(pl.col("f10") == 0).group_by(["adep", "unm"]).agg(
             pl.len().alias("n"), (pl.col("se").mean() ** 0.5).round(1).alias("rmse"),
             (pl.col("se").sum() / tot).round(4).alias("sse_share")).sort("sse_share", descending=True).head(12))
    show("H4 dfill rows (matched only) by airport",
         j.filter(pl.col("dfill") == 1).group_by("adep").agg(
             pl.len().alias("n"), (pl.col("se").mean() ** 0.5).round(1).alias("rmse"),
             (pl.col("se").sum() / tot).round(4).alias("sse_share"),
             (pl.col("oof") - pl.col("y")).median().alias("err_med")).sort("sse_share", descending=True))
    show("H5 f120-only rows (10 < |BLOCK-SCHED| <= 120): are they the same?",
         j.filter((pl.col("f120") == 1) & (pl.col("f10") == 0)).group_by("unm").agg(
             pl.len().alias("n"), (pl.col("se").mean() ** 0.5).round(1).alias("rmse"),
             (pl.col("se").sum() / tot).round(4).alias("sse_share")))
    # oracle: how much SSE would vanish if filled rows were predicted exactly
    for nm, cond in [("f10 all", pl.col("f10") == 1),
                     ("f10 matched", (pl.col("f10") == 1) & (pl.col("unm") == 0)),
                     ("f10 unmatched", (pl.col("f10") == 1) & (pl.col("unm") == 1)),
                     ("dfill matched", pl.col("dfill") == 1)]:
        rem = tot - float(j.filter(cond)["se"].sum())
        print(f"--- H6 oracle-exact on [{nm}]: n {j.filter(cond).height}, "
              f"total RMSE {round((tot / j.height) ** 0.5, 2)} -> {round((rem / j.height) ** 0.5, 2)}")
    del oof, j

    hdr("I. Unmatched rows: what is left to predict the fill with, and how separable are they?")
    show("I1 null rate of every flt-side column on unmatched rows (training)",
         u.select([pl.col(c).mean().alias(c) for c in ["eobt_null", "lobt_null", "iobt_null", "arvt1_null"]]
                  + [pl.col("op").is_null().mean().alias("op_null"),
                     pl.col("seg").is_null().mean().alias("seg_null"),
                     pl.col("stand").is_null().mean().alias("stand_null"),
                     pl.col("rwy").is_null().mean().alias("rwy_null"),
                     pl.col("fpfx").is_null().mean().alias("fpfx_null")]))
    ul = u.filter(pl.col("adep") == "LIRF").with_columns(
        pl.col("ms").cut([0, 3600, 7200, 14400, 43200],
                         labels=["<0", "0-1h", "1-2h", "2-4h", "4-12h", ">=12h"]).alias("msb"),
        pl.col("fpfx").fill_null("(none)"))
    show("I2 LIRF unmatched: fill by flight prefix x ms band (n>=20)",
         ul.group_by(["fpfx", "msb"]).agg(pl.len().alias("n"), pl.col("f10").mean().round(3).alias("f10"))
           .filter(pl.col("n") >= 20).sort("f10", descending=True))
    rows = []
    for name, by in [("fpfx", ["fpfx"]), ("msb", ["msb"]), ("fpfx+msb", ["fpfx", "msb"]),
                     ("rwy", ["rwy"]), ("stand", ["stand"]), ("hour", ["hour"]), ("fpfx+hour", ["fpfx", "hour"])]:
        ng, r2i = r2_groups(ul, by, "f10")
        r2o, a, n = lomo_score(ul, by, "f10")
        rows.append({"dim": name, "groups": ng, "r2_insample": round(r2i, 4),
                     "r2_lomo": round(r2o, 4), "auc_lomo": round(a, 4), "n": n})
    show("I3 LIRF unmatched (1,488 rows): separability of the fill from MVT-side columns only",
         pl.DataFrame(rows).sort("auc_lomo", descending=True))
    a_ms, _ = auc(ul["ms"].to_numpy(), ul["f10"].to_numpy())
    print(f"--- I4 LIRF unmatched: AUC of the single feature ms (mvt_minus_sched) = {round(a_ms, 4)}")
    uo = u.with_columns(pl.col("fpfx").fill_null("(none)"),
                        pl.col("ms").cut([0, 3600, 7200, 14400, 43200],
                                         labels=["<0", "0-1h", "1-2h", "2-4h", "4-12h", ">=12h"]).alias("msb"))
    rows = []
    for name, by in [("adep", ["adep"]), ("adep+msb", ["adep", "msb"]), ("adep+fpfx", ["adep", "fpfx"]),
                     ("adep+fpfx+msb", ["adep", "fpfx", "msb"])]:
        ng, r2i = r2_groups(uo, by, "f10")
        r2o, a, n = lomo_score(uo, by, "f10")
        rows.append({"dim": name, "groups": ng, "r2_insample": round(r2i, 4),
                     "r2_lomo": round(r2o, 4), "auc_lomo": round(a, 4), "n": n})
    show("I5 all 22,470 unmatched rows: separability of the fill", pl.DataFrame(rows).sort("auc_lomo", descending=True))
    a_ms2, _ = auc(uo["ms"].to_numpy(), uo["f10"].to_numpy())
    print(f"--- I6 all unmatched: AUC of ms alone = {round(a_ms2, 4)}; "
          f"AUC of (adep=='LIRF') alone = {round(auc((uo['adep'] == 'LIRF').cast(pl.Int8).to_numpy(), uo['f10'].to_numpy())[0], 4)}")

    hdr("J. Cross-checks: dfill vs fill AND far; BLOCK minute grid; schedule-stamp lift per airport")
    show("J1 dfill vs (fill AND far) cross-tab, matched rows",
         m.group_by([pl.col("dfill"), ((pl.col("f10") == 1) & (pl.col("far") == 1)).alias("fill_and_far")])
          .agg(pl.len().alias("n")).sort("n", descending=True))
    show("J2 BLOCK / MVT seconds within 6 s of a minute, by airport x matched",
         tr.with_columns(pl.min_horizontal(pl.col("block_sec"), 60 - pl.col("block_sec")).alias("bo"),
                         pl.min_horizontal(pl.col("mvt_sec"), 60 - pl.col("mvt_sec")).alias("mo"))
           .group_by(["adep", "unm"]).agg(pl.len().alias("n"),
                                          (pl.col("bo") <= 6).mean().round(4).alias("block_near"),
                                          (pl.col("mo") <= 6).mean().round(4).alias("mvt_near")).sort(["adep", "unm"]))
    show("J3 schedule-stamp flag lift on the fill, per airport x far (matched)",
         m.with_columns(((pl.col("sched_lobt") == 0) & (pl.col("sched_eobt") == 0)).alias("nosupd"))
          .group_by(["adep", "far", "nosupd"]).agg(pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10"))
          .pivot(on=["far", "nosupd"], index="adep", values="f10").sort("adep"))
    show("J4 same, but the cell counts",
         m.with_columns(((pl.col("sched_lobt") == 0) & (pl.col("sched_eobt") == 0)).alias("nosupd"))
          .group_by(["adep", "far", "nosupd"]).agg(pl.len().alias("n"))
          .pivot(on=["far", "nosupd"], index="adep", values="n").sort("adep"))
    rows = []
    mm = m.with_columns(((pl.col("sched_lobt") == 0) & (pl.col("sched_eobt") == 0)).cast(pl.Int8).alias("nosupd"))
    for name, by in [("far+nosupd", ["far", "nosupd"]), ("adep+far+nosupd", ["adep", "far", "nosupd"]),
                     ("adep+op+far+nosupd", ["adep", "op", "far", "nosupd"]),
                     ("op+far+nosupd", ["op", "far", "nosupd"])]:
        for scope, d in (("all-matched", mm), ("LIRF-matched", mm.filter(pl.col("adep") == "LIRF")),
                         ("other-matched", mm.filter(pl.col("adep") != "LIRF"))):
            r2o, a, n = lomo_score(d, by, "f10")
            rows.append({"scope": scope, "dim": name, "r2_lomo": round(r2o, 4), "auc_lomo": round(a, 4)})
    show("J5 LOMO group-mean score including the no-schedule-update flag",
         pl.DataFrame(rows).pivot(on="scope", index="dim", values=["r2_lomo", "auc_lomo"]))

    hdr("K. Ranking set: expected fill per month with the drivers that survive")
    rk = build(RANK, False).with_columns(
        ((pl.col("sched_lobt") == 0) & (pl.col("sched_eobt") == 0)).cast(pl.Int8).alias("nosupd"),
        pl.col("fpfx").fill_null("(none)"))
    g = mm.group_by(["adep", "op", "far", "nosupd"]).agg(pl.len().alias("gn"), pl.col("f10").mean().alias("p"),
                                                         pl.col("dfill").mean().alias("pd"))
    gb = mm.group_by(["adep", "far", "nosupd"]).agg(pl.col("f10").mean().alias("pb"), pl.col("dfill").mean().alias("pdb"))
    rkm = rk.filter(pl.col("unm") == 0).join(g, on=["adep", "op", "far", "nosupd"], how="left").join(
        gb, on=["adep", "far", "nosupd"], how="left").with_columns(
        pl.coalesce(["p", "pb"]).alias("p"), pl.coalesce(["pd", "pdb"]).alias("pd"))
    show("K1 expected filled / fill-and-far ranking matched rows by airport x month",
         rkm.group_by(["adep", "month"]).agg(pl.len().alias("n"),
                                             pl.col("p").sum().round(0).alias("E_filled"),
                                             (pl.col("p").mean()).round(4).alias("E_rate"),
                                             pl.col("pd").sum().round(0).alias("E_dfill")).sort(["adep", "month"]))
    print("--- K2 totals: ranking matched rows", rkm.height,
          "| E[filled]", round(float(rkm["p"].sum()), 0),
          "| E[fill and far]", round(float(rkm["pd"].sum()), 0),
          "| training flat rate", round(float(m["f10"].mean()), 4))
    ur = rk.filter((pl.col("unm") == 1) & (pl.col("adep") == "LIRF"))
    gu = ul.group_by("fpfx").agg(pl.len().alias("gn"), pl.col("f10").mean().alias("p"))
    ju = ur.join(gu, on="fpfx", how="left")
    print("--- K3 ranking LIRF unmatched", ur.height, "rows; flight-prefix seen in training",
          int(ju["p"].is_not_null().sum()), "; E[filled] under prefix rates",
          round(float(ju["p"].fill_null(float(ul["f10"].mean())).sum()), 1),
          "; flat-rate expectation", round(ur.height * float(ul["f10"].mean()), 1))
    show("K4 ranking LIRF unmatched by flight prefix vs its training fill rate",
         ju.group_by("fpfx").agg(pl.len().alias("n_rank"), pl.col("p").first().round(3).alias("train_fill"),
                                 pl.col("gn").first().alias("n_train")).sort("n_rank", descending=True).head(20))

    print("\nDONE")


if __name__ == "__main__":
    main()

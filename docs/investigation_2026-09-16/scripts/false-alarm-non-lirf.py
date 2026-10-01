"""Verification of investigation item 7 `false-alarm-non-lirf` (plan.md §1, source M).

Hypothesis: model B fires 15-20k predictions on large-ms rows at non-LIRF airports whose labels are
normal (sched-filled fraction ~ 0); the ranking set (EHAM) contains many such rows.

Everything below is re-derived from the data files; nothing is copied from the hypothesis reports.
No training. POLARS_MAX_THREADS=1. Lazy scans, only the needed columns.

Definitions (plan §0 / §4), all computed side by side:
  unmatched:  unm_routing = mvt_minus_aobt3 null (model-B routing rule; H/P/M)
              unm_fid     = FLIGHT_ID_mvt null (D)
  sched-filled: f10 = |BLOCK - SCHED| <= 10 s (D); f6 = <= 6 s (M); f120 = <= 120 s (M);
              r1800 = |y - ms| < 1800 s (M).  (y - ms == SCHED - BLOCK identically, item 1.)
  ms = feature mvt_minus_sched = MVT - SCHED.
  ms bins requested by the task: 2-4 h [7200, 14400), 4-12 h [14400, 43200), >= 12 h [43200, inf).
  M's bins, also reported: (14400, 43200] and (43200, inf).

Submission files (version -> file), fixed from experiments/submissions.jsonl and identity checks run in
this script:
  v2  outputs/v2_two_clock.parquet (repo) and scratchpad uploaded/v2.parquet (bucket copy)
  v3  scratchpad uploaded/v3.parquet (bucket copy; the repo file outputs/v3_two_clock_weather.parquet no longer exists)
  v4  scratchpad uploaded/v4.parquet (bucket copy) == outputs/probe_unmatched_from_v4.parquet on the 5,290 unmatched rows
  v5  outputs/v5_unmatched_capacity.parquet -- candidate only; the uploaded outputs/main.parquet of 09-14 was overwritten
  v7  outputs/main.parquet (mtime 09-15 15:52; uploaded 15:53)
  v8  outputs/det_a.parquet (== probe on all 339,551 matched rows)
  v9  outputs/probe_unmatched_from_v4.parquet (v8 matched + v4 unmatched)
"""
import os

os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl

pl.Config.set_tbl_rows(400)
pl.Config.set_tbl_cols(60)
pl.Config.set_tbl_width_chars(300)
pl.Config.set_fmt_str_lengths(80)
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)

ROOT = "/home/user/prc-data-challenge"
FT = f"{ROOT}/data/work/feat_train.parquet"
FR = f"{ROOT}/data/work/feat_rank.parquet"
OOF = f"{ROOT}/experiments/oof_main-48c19bc830.parquet"
SUB = f"{ROOT}/data/raw/submitting.parquet"
UP = "/tmp/claude-0/-home-user-prc-data-challenge/ca703460-62c0-5c84-91fe-d76c3d0e1e20/scratchpad/uploaded/"

H2, H4, H12 = 7200, 14400, 43200
Q = [0.1, 0.5, 0.9]


def hdr(s):
    print("\n" + "=" * 110 + f"\n{s}\n" + "=" * 110)


def sec(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds()


def msbin():
    return (
        pl.when(pl.col("ms") < H2).then(pl.lit("0:<2h"))
        .when(pl.col("ms") < H4).then(pl.lit("1:2-4h"))
        .when(pl.col("ms") < H12).then(pl.lit("2:4-12h"))
        .otherwise(pl.lit("3:>=12h"))
        .alias("msbin")
    )


def msbin_M():
    return (
        pl.when(pl.col("ms") <= H4).then(pl.lit("0:<=4h"))
        .when(pl.col("ms") <= H12).then(pl.lit("1:(4h,12h]"))
        .otherwise(pl.lit("2:(12h,inf)"))
        .alias("msbin_M")
    )


def qcols(c, prefix):
    return [pl.col(c).quantile(p, "linear").round(0).alias(f"{prefix}_p{int(p * 100)}") for p in Q] + [
        pl.col(c).max().alias(f"{prefix}_max")
    ]


# ----------------------------------------------------------------------------------------------------
# training frame
# ----------------------------------------------------------------------------------------------------
tr = (
    pl.scan_parquet(FT)
    .select(
        "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "mvt_minus_sched",
    )
    .with_columns(
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
        pl.col("mvt_minus_sched").cast(pl.Float64).alias("ms"),
        sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("bs"),
        pl.col("mvt_minus_aobt3").is_null().alias("unm_routing"),
        pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),
        (pl.col("ADEP_mvt") == "LIRF").alias("lirf"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
    )
    .with_columns(
        (pl.col("bs").abs() <= 10).alias("f10"),
        (pl.col("bs").abs() <= 6).alias("f6"),
        (pl.col("bs").abs() <= 120).alias("f120"),
        ((pl.col("y") - pl.col("ms")).abs() < 1800).alias("r1800"),
        msbin(), msbin_M(),
    )
    .select(pl.exclude("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "FLIGHT_ID_mvt", "mvt_minus_aobt3", "TAXITIME_SEC_mvt"))
    .collect()
)
hdr("A. training frame, unmatched definitions")
print(
    tr.select(
        pl.len().alias("n"), pl.col("unm_routing").sum().alias("unm_routing"), pl.col("unm_fid").sum().alias("unm_fid"),
        (pl.col("unm_routing") & ~pl.col("unm_fid")).sum().alias("routing_only"),
        (~pl.col("unm_routing") & pl.col("unm_fid")).sum().alias("fid_only"),
        (pl.col("unm_routing") & ~pl.col("unm_fid") & ~pl.col("lirf") & (pl.col("ms") >= H2)).sum().alias("routing_only_nonLIRF_ms_ge2h"),
        pl.col("ms").is_null().sum().alias("ms_null"),
    )
)
print("routing-only rows (46) by ADEP x msbin:")
print(tr.filter(pl.col("unm_routing") & ~pl.col("unm_fid")).group_by("ADEP_mvt", "msbin").agg(pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("y").max().alias("y_max")).sort("ADEP_mvt", "msbin"))

# ----------------------------------------------------------------------------------------------------
# B. training labels, non-LIRF unmatched, by ms bin (task bins and M's bins), per airport
# ----------------------------------------------------------------------------------------------------
for defn in ("unm_routing", "unm_fid"):
    hdr(f"B. training: unmatched ({defn}) rows with ms >= 2h, by LIRF / non-LIRF x ms bin: n, sched-filled fractions, label quantiles")
    print(
        tr.filter(pl.col(defn) & (pl.col("ms") >= H2))
        .group_by("lirf", "msbin")
        .agg(
            pl.len().alias("n"),
            pl.col("f6").mean().round(4).alias("f6"), pl.col("f10").mean().round(4).alias("f10"),
            pl.col("f120").mean().round(4).alias("f120"), pl.col("r1800").mean().round(4).alias("r1800"),
            *qcols("y", "y"),
            (pl.col("y") >= H2).mean().round(4).alias("y_ge2h"), (pl.col("y") >= H4).mean().round(4).alias("y_ge4h"),
            pl.col("ms").median().alias("ms_med"),
        )
        .sort("lirf", "msbin")
    )
    print("-- M's bins, non-LIRF unmatched:")
    print(
        tr.filter(pl.col(defn) & ~pl.col("lirf") & (pl.col("ms") > H4))
        .group_by("msbin_M")
        .agg(pl.len().alias("n"), pl.col("f6").mean().round(4).alias("f6"), pl.col("f10").mean().round(4).alias("f10"),
             pl.col("f120").mean().round(4).alias("f120"), pl.col("r1800").mean().round(4).alias("r1800"),
             pl.col("r1800").sum().alias("n_r1800"), pl.col("y").median().alias("y_med"), pl.col("y").max().alias("y_max"))
        .sort("msbin_M")
    )
    if defn == "unm_routing":
        print("-- per airport x ms bin, non-LIRF unmatched (routing):")
        print(
            tr.filter(pl.col(defn) & ~pl.col("lirf") & (pl.col("ms") >= H2))
            .group_by("ADEP_mvt", "msbin")
            .agg(pl.len().alias("n"), pl.col("f10").sum().alias("n_f10"), pl.col("f120").sum().alias("n_f120"),
                 pl.col("r1800").sum().alias("n_r1800"), *qcols("y", "y"),
                 (pl.col("y") >= H2).sum().alias("n_y_ge2h"), (pl.col("y") >= H4).sum().alias("n_y_ge4h"))
            .sort("ADEP_mvt", "msbin")
        )
        print("-- non-LIRF unmatched rows with ms >= 4h AND y >= 2h (the exceptions to 'labels are normal'):")
        print(tr.filter(pl.col(defn) & ~pl.col("lirf") & (pl.col("ms") >= H4) & (pl.col("y") >= H2)).select("ADEP_mvt", "mo", "ms", "y", "bs", "f10", "r1800").sort("y", descending=True))
        print("-- non-LIRF unmatched rows with ms >= 4h AND r1800 (|y - ms| < 1800, M's definition of sched-filled):")
        print(tr.filter(pl.col(defn) & ~pl.col("lirf") & (pl.col("ms") >= H4) & pl.col("r1800")).select("ADEP_mvt", "mo", "ms", "y", "bs").sort("ms"))

# airport-level medians used in the counterfactual (D)
hdr("B2. airport medians of the label (training). Variants: all rows; unmatched rows (routing); unmatched rows with ms >= 4h")
med_all = tr.group_by("ADEP_mvt").agg(pl.col("y").median().alias("med_all"), pl.len().alias("n_all"))
med_unm = tr.filter(pl.col("unm_routing")).group_by("ADEP_mvt").agg(pl.col("y").median().alias("med_unm"), pl.col("y").mean().round(1).alias("mean_unm"), pl.len().alias("n_unm"))
med_bin = tr.filter(pl.col("unm_routing") & (pl.col("ms") >= H4)).group_by("ADEP_mvt").agg(pl.col("y").median().alias("med_unm_ms_ge4h"), pl.col("y").mean().round(1).alias("mean_unm_ms_ge4h"), pl.len().alias("n_unm_ms_ge4h"))
meds = med_all.join(med_unm, on="ADEP_mvt", how="left").join(med_bin, on="ADEP_mvt", how="left").sort("ADEP_mvt")
print(meds)

# leave-one-month-out median of unmatched labels per airport (fold-honest version of med_unm)
lomo = []
for mo in range(1, 13):
    d = tr.filter(pl.col("unm_routing") & (pl.col("mo") != mo)).group_by("ADEP_mvt").agg(pl.col("y").median().alias("med_unm_lomo")).with_columns(pl.lit(mo).cast(pl.Int8).alias("mo"))
    lomo.append(d)
lomo = pl.concat(lomo)
print("-- LOMO airport median of unmatched labels: range over months per airport")
print(lomo.group_by("ADEP_mvt").agg(pl.col("med_unm_lomo").min().alias("min"), pl.col("med_unm_lomo").max().alias("max")).sort("ADEP_mvt"))

# ----------------------------------------------------------------------------------------------------
# C. OOF main-48c19bc830
# ----------------------------------------------------------------------------------------------------
oof_all = pl.read_parquet(OOF)
oof = oof_all.filter(pl.col("oof").is_not_null() & pl.col("oof").is_not_nan())
j = oof.join(tr, on="MVT_ID_mvt", how="left")
assert j["y"].null_count() == 0
j = j.with_columns(((pl.col("oof") - pl.col("y")) ** 2).alias("se"))
N_OOF = j.height
TOT = j["se"].sum()
hdr(f"C. OOF: file rows {oof_all.height}, scored rows {N_OOF}, total RMSE {(TOT / N_OOF) ** 0.5:.2f}, SSE {TOT:.4e}")

for defn in ("unm_routing", "unm_fid"):
    u = j.filter(pl.col(defn))
    USSE = u["se"].sum()
    print(f"\n-- def {defn}: unmatched scored rows {u.height}, unmatched RMSE {(USSE / u.height) ** 0.5:.1f}, share of total SSE {USSE / TOT:.4f}")
    print("   non-LIRF unmatched, ms >= 2h pooled:", u.filter(~pl.col("lirf") & (pl.col("ms") >= H2)).select(pl.len().alias("n"), (pl.col("se").sum() / TOT).round(4).alias("sse_share_total"), (pl.col("se").sum() / USSE).round(4).alias("sse_share_unm"), (pl.col("se").mean() ** 0.5).round(1).alias("rmse")).to_dicts())
    print("   LIRF unmatched, ms >= 2h pooled:    ", u.filter(pl.col("lirf") & (pl.col("ms") >= H2)).select(pl.len().alias("n"), (pl.col("se").sum() / TOT).round(4).alias("sse_share_total"), (pl.col("se").sum() / USSE).round(4).alias("sse_share_unm"), (pl.col("se").mean() ** 0.5).round(1).alias("rmse")).to_dicts())
    print("   by LIRF / non-LIRF x ms bin: n, label quantiles, OOF pred quantiles, RMSE, SSE shares")
    print(
        u.filter(pl.col("ms") >= H2)
        .group_by("lirf", "msbin")
        .agg(
            pl.len().alias("n"), *qcols("y", "y"), *qcols("oof", "pred"),
            (pl.col("se").mean() ** 0.5).round(1).alias("rmse"),
            (pl.col("se").sum() / TOT).round(4).alias("sse_share_total"),
            (pl.col("se").sum() / USSE).round(4).alias("sse_share_unm"),
            (pl.col("oof") >= H2).sum().alias("n_pred_ge2h"), (pl.col("oof") >= H4).sum().alias("n_pred_ge4h"),
            (pl.col("y") >= H2).sum().alias("n_y_ge2h"), (pl.col("y") >= H4).sum().alias("n_y_ge4h"),
            pl.col("f10").sum().alias("n_f10"), pl.col("r1800").sum().alias("n_r1800"),
        )
        .sort("lirf", "msbin")
    )
    if defn == "unm_routing":
        print("   per airport x ms bin, non-LIRF unmatched (routing):")
        print(
            u.filter(~pl.col("lirf") & (pl.col("ms") >= H2))
            .group_by("ADEP_mvt", "msbin")
            .agg(pl.len().alias("n"), *qcols("y", "y"), *qcols("oof", "pred"), (pl.col("se").mean() ** 0.5).round(1).alias("rmse"),
                 (pl.col("se").sum() / TOT).round(4).alias("sse_share_total"), (pl.col("se").sum() / USSE).round(4).alias("sse_share_unm"),
                 (pl.col("oof") >= H4).sum().alias("n_pred_ge4h"), (pl.col("y") >= H2).sum().alias("n_y_ge2h"))
            .sort("ADEP_mvt", "msbin")
        )
        print("   per airport, all unmatched OOF rows (routing): SSE share, RMSE")
        print(u.group_by("ADEP_mvt").agg(pl.len().alias("n"), (pl.col("se").sum() / TOT).round(4).alias("sse_share_total"), (pl.col("se").mean() ** 0.5).round(1).alias("rmse"), (pl.col("se").filter(pl.col("ms") >= H4).sum() / TOT).round(4).alias("sse_share_total_ms_ge4h"), (pl.col("ms") >= H4).sum().alias("n_ms_ge4h")).sort("sse_share_total", descending=True))

u = j.filter(pl.col("unm_routing"))
USSE = u["se"].sum()

hdr("C2. 'fires 15-20k': OOF predictions >= 4h on unmatched rows (routing), by LIRF / non-LIRF")
print(
    u.filter(pl.col("oof") >= H4)
    .group_by("lirf")
    .agg(pl.len().alias("n_pred_ge4h"), *qcols("oof", "pred"), *qcols("y", "y"), (pl.col("y") >= H2).sum().alias("n_y_ge2h"),
         (pl.col("y") >= H4).sum().alias("n_y_ge4h"), pl.col("r1800").sum().alias("n_r1800"), (pl.col("se").sum() / TOT).round(4).alias("sse_share_total"),
         ((pl.col("oof") >= 15000) & (pl.col("oof") <= 20000)).sum().alias("n_pred_15k_20k"))
    .sort("lirf")
)
print("-- non-LIRF unmatched rows with OOF pred >= 4h (all of them):")
print(u.filter(~pl.col("lirf") & (pl.col("oof") >= H4)).sort("oof", descending=True).select("ADEP_mvt", "mo", "ms", "y", pl.col("oof").round(0).alias("pred"), pl.col("se").round(0).alias("se"), "r1800"))
print("-- non-LIRF unmatched rows with OOF pred in [2h, 4h): count by ADEP, label quantiles")
print(u.filter(~pl.col("lirf") & (pl.col("oof") >= H2) & (pl.col("oof") < H4)).group_by("ADEP_mvt").agg(pl.len().alias("n"), *qcols("y", "y"), pl.col("ms").median().alias("ms_med"), (pl.col("y") >= H2).sum().alias("n_y_ge2h")).sort("n", descending=True))
print("-- matched OOF rows with pred >= 2h, for scale:", j.filter(~pl.col("unm_routing") & (pl.col("oof") >= H2)).height)

hdr("C3. top-25 SSE rows among all scored OOF rows")
top = j.sort("se", descending=True).head(25).with_columns(pl.int_range(1, 26).alias("rank"))
print(top.select("rank", "ADEP_mvt", "unm_routing", "unm_fid", "mo", "ms", "y", pl.col("oof").round(0).alias("pred"), pl.col("se").round(0).alias("se"), (pl.col("se") / TOT * 100).round(2).alias("pct_total_sse"), "f10", "r1800"))
print("non-LIRF unmatched rows among top-25:", top.filter(~pl.col("lirf") & pl.col("unm_routing")).height, "; their SSE share of total:", round(top.filter(~pl.col("lirf") & pl.col("unm_routing"))["se"].sum() / TOT, 4))
print("top-25 SSE share of total:", round(top["se"].sum() / TOT, 4))

# ----------------------------------------------------------------------------------------------------
# D. counterfactual: replace OOF predictions of non-LIRF unmatched rows with ms >= 4h by an airport median
# ----------------------------------------------------------------------------------------------------
hdr("D. counterfactual OOF RMSE: non-LIRF unmatched predictions replaced by an airport-median label")
jj = j.join(meds.select("ADEP_mvt", "med_all", "med_unm", "med_unm_ms_ge4h", "mean_unm_ms_ge4h"), on="ADEP_mvt", how="left").join(lomo, on=["ADEP_mvt", "mo"], how="left")


def cf(frame, mask, repl_col, label):
    p2 = pl.when(mask).then(pl.col(repl_col)).otherwise(pl.col("oof"))
    f = frame.with_columns(((p2 - pl.col("y")) ** 2).alias("se2"))
    tot2 = f["se2"].sum()
    un = f.filter(pl.col("unm_routing"))
    nrep = frame.filter(mask).height
    sse_rows_before = frame.filter(mask)["se"].sum()
    sse_rows_after = f.filter(mask)["se2"].sum()
    print(f"{label:<75} rows replaced {nrep:>5} | total RMSE {(TOT / N_OOF) ** 0.5:.2f} -> {(tot2 / N_OOF) ** 0.5:.2f} | "
          f"unmatched RMSE {(USSE / un.height) ** 0.5:.1f} -> {(un['se2'].sum() / un.height) ** 0.5:.1f} | "
          f"SSE on replaced rows {sse_rows_before:.3e} -> {sse_rows_after:.3e} (share of total {sse_rows_before / TOT:.4f} -> {sse_rows_after / TOT:.4f})")


for defn in ("unm_routing", "unm_fid"):
    print(f"\n-- unmatched definition: {defn}")
    base = pl.col(defn) & ~pl.col("lirf")
    for thr, name in ((H4, "ms >= 4h"), (H2, "ms >= 2h"), (H12, "ms >= 12h")):
        m = base & (pl.col("ms") >= thr)
        cf(jj, m, "med_unm", f"[{name}] pred := airport median of unmatched training labels")
        cf(jj, m, "med_unm_lomo", f"[{name}] pred := LOMO airport median of unmatched training labels")
        cf(jj, m, "med_all", f"[{name}] pred := airport median of all training labels")
        cf(jj, m, "med_unm_ms_ge4h", f"[{name}] pred := airport median of unmatched labels with ms >= 4h")
        cf(jj, m, "mean_unm_ms_ge4h", f"[{name}] pred := airport MEAN of unmatched labels with ms >= 4h")
        cf(jj, m, "y", f"[{name}] oracle: pred := y")
        if defn == "unm_fid":
            break  # the fid definition differs only on rows that are not in these masks (checked above); one line suffices
    if defn == "unm_routing":
        print("   per-airport contribution to the ms >= 4h replacement (median of unmatched labels):")
        m = base & (pl.col("ms") >= H4)
        f = jj.filter(m).with_columns(((pl.col("med_unm") - pl.col("y")) ** 2).alias("se2"))
        print(f.group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("se").sum().round(0).alias("sse_before"), pl.col("se2").sum().round(0).alias("sse_after"), ((pl.col("se").sum() - pl.col("se2").sum()) / TOT).round(4).alias("gain_share_total")).sort("gain_share_total", descending=True))
        print("   same replacement applied to LIRF unmatched ms >= 4h rows, for contrast:")
        cf(jj, pl.col("unm_routing") & pl.col("lirf") & (pl.col("ms") >= H4), "med_unm", "[LIRF, ms >= 4h] pred := airport median of unmatched training labels")
        cf(jj, pl.col("unm_routing") & (pl.col("ms") >= H4), "med_unm", "[ALL airports, ms >= 4h] pred := airport median of unmatched training labels")

# ----------------------------------------------------------------------------------------------------
# E. ranking set and submission files
# ----------------------------------------------------------------------------------------------------
hdr("E. ranking set: non-LIRF unmatched rows by ms bin; submission predictions")
rk = (
    pl.scan_parquet(FR)
    .select("MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "mvt_minus_aobt3", "mvt_minus_sched", "MVT_TIME_UTC_mvt")
    .with_columns(
        pl.col("mvt_minus_aobt3").is_null().alias("unm_routing"), pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),
        (pl.col("ADEP_mvt") == "LIRF").alias("lirf"), pl.col("mvt_minus_sched").cast(pl.Float64).alias("ms"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
    )
    .with_columns(msbin())
    .select(pl.exclude("FLIGHT_ID_mvt", "mvt_minus_aobt3", "mvt_minus_sched", "MVT_TIME_UTC_mvt"))
    .collect()
)
sub = pl.read_parquet(SUB).select("MVT_ID_mvt")
print("ranking rows", rk.height, "| submitting rows", sub.height, "| MVT_ID intersection", rk.join(sub, on="MVT_ID_mvt").height,
      "| unmatched routing", rk["unm_routing"].sum(), "| fid", rk["unm_fid"].sum(), "| routing-only non-LIRF ms>=2h",
      rk.filter(pl.col("unm_routing") & ~pl.col("unm_fid") & ~pl.col("lirf") & (pl.col("ms") >= H2)).height)

files = {
    "v2": f"{ROOT}/outputs/v2_two_clock.parquet",
    "v2up": UP + "v2.parquet",
    "v3": UP + "v3.parquet",
    "v4": UP + "v4.parquet",
    "v5cand": f"{ROOT}/outputs/v5_unmatched_capacity.parquet",
    "v7": f"{ROOT}/outputs/main.parquet",
    "v8": f"{ROOT}/outputs/det_a.parquet",
    "v9": f"{ROOT}/outputs/probe_unmatched_from_v4.parquet",
}
S = rk
present = []
for k, f in files.items():
    if not os.path.exists(f):
        print("MISSING", k, f)
        continue
    d = pl.read_parquet(f)
    assert d.height == sub.height and (d["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all(), k
    S = S.join(d.rename({"TAXITIME_SEC_mvt": k}).with_columns(pl.col(k).cast(pl.Float64)), on="MVT_ID_mvt", how="left")
    present.append(k)
print("submissions joined:", present, "| null preds:", {k: int(S[k].null_count()) for k in present})
U = S.filter(pl.col("unm_routing"))
M_ = S.filter(~pl.col("unm_routing"))
print("identity checks: v9==v4 on unmatched:", bool((U["v9"] == U["v4"]).all()), "| v9==v8 on matched:", bool((M_["v9"] == M_["v8"]).all()),
      "| v2 repo vs v2 bucket copy: n differing rows", int((S["v2"] != S["v2up"]).sum()), "RMS diff", round(((S["v2"] - S["v2up"]) ** 2).mean() ** 0.5, 2),
      "| v7 vs v5cand RMS diff unmatched", round(((U["v7"] - U["v5cand"]) ** 2).mean() ** 0.5, 1))

print("\n-- ranking non-LIRF unmatched (routing) counts by ADEP x ms bin; matching training Jan+Jul 2025 counts")
cnt_r = U.filter(~pl.col("lirf") & (pl.col("ms") >= H2)).group_by("ADEP_mvt", "msbin").agg(pl.len().alias("n_rank"), (pl.col("mo") == 1).sum().alias("n_jan"), (pl.col("mo") == 7).sum().alias("n_jul"))
cnt_t = tr.filter(pl.col("unm_routing") & ~pl.col("lirf") & (pl.col("ms") >= H2) & pl.col("mo").is_in([1, 7])).group_by("ADEP_mvt", "msbin").agg(pl.len().alias("n_train_JanJul2025"))
print(cnt_r.join(cnt_t, on=["ADEP_mvt", "msbin"], how="full", coalesce=True).fill_null(0).sort("ADEP_mvt", "msbin"))
print("EHAM unmatched ranking: ms >= 2h", U.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("ms") >= H2)).height, "| >= 4h", U.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("ms") >= H4)).height, "| >= 12h", U.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("ms") >= H12)).height,
      "| fid def:", S.filter(pl.col("unm_fid") & (pl.col("ADEP_mvt") == "EHAM") & (pl.col("ms") >= H2)).height)
print("EHAM unmatched TRAINING: ms >= 2h", tr.filter(pl.col("unm_routing") & (pl.col("ADEP_mvt") == "EHAM") & (pl.col("ms") >= H2)).height, "| >= 4h", tr.filter(pl.col("unm_routing") & (pl.col("ADEP_mvt") == "EHAM") & (pl.col("ms") >= H4)).height,
      "| Jan+Jul 2025 >= 2h", tr.filter(pl.col("unm_routing") & (pl.col("ADEP_mvt") == "EHAM") & (pl.col("ms") >= H2) & pl.col("mo").is_in([1, 7])).height)

print("\n-- prediction quantiles per submission, non-LIRF unmatched ranking rows, by ms bin (pooled over airports)")
rows = []
for b in ["1:2-4h", "2:4-12h", "3:>=12h"]:
    g = U.filter(~pl.col("lirf") & (pl.col("msbin") == b))
    for k in present:
        c = g[k]
        rows.append({"msbin": b, "n": g.height, "sub": k, "p10": round(c.quantile(0.1), 0), "p50": c.median(), "p90": round(c.quantile(0.9), 0), "max": c.max(),
                     "n_ge2h": int((c >= H2).sum()), "n_ge4h": int((c >= H4).sum()), "n_ge12h": int((c >= H12).sum())})
print(pl.DataFrame(rows))
print("-- same for LIRF unmatched ranking rows, for contrast")
rows = []
for b in ["1:2-4h", "2:4-12h", "3:>=12h"]:
    g = U.filter(pl.col("lirf") & (pl.col("msbin") == b))
    for k in present:
        c = g[k]
        rows.append({"msbin": b, "n": g.height, "sub": k, "p10": round(c.quantile(0.1), 0), "p50": c.median(), "p90": round(c.quantile(0.9), 0), "max": c.max(), "n_ge4h": int((c >= H4).sum()), "n_ge12h": int((c >= H12).sum())})
print(pl.DataFrame(rows))

print("\n-- per airport x ms bin, non-LIRF unmatched ranking rows: median prediction per submission")
print(
    U.filter(~pl.col("lirf") & (pl.col("ms") >= H2))
    .group_by("ADEP_mvt", "msbin")
    .agg(pl.len().alias("n"), *[pl.col(k).median().alias(f"{k}_med") for k in present], *[pl.col(k).max().alias(f"{k}_max") for k in ("v4", "v5cand", "v8")])
    .sort("ADEP_mvt", "msbin")
)
print("-- EHAM unmatched ranking rows with ms >= 12h (all):")
print(U.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("ms") >= H12)).select("mo", "ms", *present).sort("ms"))
print("-- EHAM unmatched ranking rows with 4h <= ms < 12h: prediction quantiles per submission")
g = U.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("msbin") == "2:4-12h"))
print(pl.DataFrame([{"sub": k, "n": g.height, "p10": round(g[k].quantile(0.1), 0), "p50": g[k].median(), "p90": round(g[k].quantile(0.9), 0), "max": g[k].max()} for k in present]))
print("-- all non-LIRF unmatched ranking rows where any submission predicts >= 4h:")
anyge4 = pl.any_horizontal([pl.col(k) >= H4 for k in present])
print(U.filter(~pl.col("lirf") & anyge4).select("ADEP_mvt", "mo", "ms", *present).sort("ms"))

# ----------------------------------------------------------------------------------------------------
# F. expected LB cost of these predictions IF ranking labels in each (ADEP, ms bin) cell follow the training
#    label distribution of the same cell (non-LIRF unmatched). E[(p - y)^2] = (p - mean_y)^2 + var_y.
#    This is a conditional computation, not a measurement of LB.
# ----------------------------------------------------------------------------------------------------
hdr("F. conditional expected SSE on non-LIRF unmatched ranking rows with ms >= 4h, under the training label distribution per (ADEP, ms bin)")
stats = tr.filter(pl.col("unm_routing") & ~pl.col("lirf") & (pl.col("ms") >= H4)).group_by("ADEP_mvt", "msbin").agg(pl.col("y").mean().alias("mu"), pl.col("y").var(ddof=0).alias("var"), pl.len().alias("n_train"))
G = U.filter(~pl.col("lirf") & (pl.col("ms") >= H4)).join(stats, on=["ADEP_mvt", "msbin"], how="left")
print("cells without training stats:", G.filter(pl.col("mu").is_null()).select("ADEP_mvt", "msbin").unique().to_dicts())
G = G.filter(pl.col("mu").is_not_null())
NR = sub.height
rows = []
for k in present:
    e = ((pl.col(k) - pl.col("mu")) ** 2 + pl.col("var"))
    tot_e = G.select(e.sum()).item()
    rows.append({"sub": k, "n_rows": G.height, "E_SSE": round(tot_e, 0), "E_SSE/N_rank (adds to LB MSE)": round(tot_e / NR, 2), "sqrt(E_SSE/N_rank)": round((tot_e / NR) ** 0.5, 1)})
irreducible = G.select(pl.col("var").sum()).item()
rows.append({"sub": "pred := cell mean (irreducible)", "n_rows": G.height, "E_SSE": round(irreducible, 0), "E_SSE/N_rank (adds to LB MSE)": round(irreducible / NR, 2), "sqrt(E_SSE/N_rank)": round((irreducible / NR) ** 0.5, 1)})
print(pl.DataFrame(rows))
lb = {"v2": 312.02, "v3": 306.29, "v4": 305.03, "v7": 342.59, "v8": 343.80, "v9": 306.46}
for a, b in (("v4", "v8"), ("v4", "v7"), ("v5cand", "v8")):
    ea = G.select(((pl.col(a) - pl.col("mu")) ** 2 + pl.col("var")).sum()).item()
    eb = G.select(((pl.col(b) - pl.col("mu")) ** 2 + pl.col("var")).sum()).item()
    line = f"E[SSE_{a}] - E[SSE_{b}] on these rows = {ea - eb:.3e}"
    if a in lb and b in lb:
        gap = (lb[b] ** 2 - lb[a] ** 2) * NR
        line += f" | measured total SSE gap (LB_{b}^2 - LB_{a}^2)*N = {gap:.3e} | ratio {(eb - ea) / gap:.4f}"
    print(line)
print("RMS(v4 - v8) on non-LIRF unmatched ranking rows: ms >= 4h", round(((G["v4"] - G["v8"]) ** 2).mean() ** 0.5, 1),
      "| all non-LIRF unmatched", round(((U.filter(~pl.col("lirf"))["v4"] - U.filter(~pl.col("lirf"))["v8"]) ** 2).mean() ** 0.5, 1),
      "| LIRF unmatched", round(((U.filter(pl.col("lirf"))["v4"] - U.filter(pl.col("lirf"))["v8"]) ** 2).mean() ** 0.5, 1))
print("sum of (v4 - v8)^2 over non-LIRF unmatched ranking rows / N_rank:", round((((U.filter(~pl.col("lirf"))["v4"] - U.filter(~pl.col("lirf"))["v8"]) ** 2).sum()) / NR, 2),
      "| over LIRF unmatched:", round((((U.filter(pl.col("lirf"))["v4"] - U.filter(pl.col("lirf"))["v8"]) ** 2).sum()) / NR, 2))
print("\ndone")

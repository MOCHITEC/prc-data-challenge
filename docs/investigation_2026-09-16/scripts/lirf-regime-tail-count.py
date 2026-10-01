"""Item 3 `lirf-regime-tail-count` -- independent re-derivation (no training).

Run:  POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/lirf-regime-tail-count.py
Inputs: data/work/feat_train.parquet, data/work/feat_rank.parquet, data/raw/submitting.parquet,
        experiments/oof_main-48c19bc830.parquet, outputs/*.parquet, scratchpad uploaded v3/v4.
Everything is column-selected via scan_parquet; the 132-column frames are never collected whole.
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"
import math
import polars as pl

pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260)
pl.Config.set_fmt_str_lengths(40)

B = "/home/user/prc-data-challenge/"
UP = "/tmp/claude-0/-home-user-prc-data-challenge/ca703460-62c0-5c84-91fe-d76c3d0e1e20/scratchpad/uploaded/"
H2, H4, H12, DAY = 7200, 14400, 43200, 86400

def sec(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds()

def hdr(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)

# ----------------------------------------------------------------------------------------------
# S1. training profile of LIRF unmatched rows vs everything else
# ----------------------------------------------------------------------------------------------
base_cols = ["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "AOBT_3_flt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt",
             "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "RUNWAY_mvt",
             "mvt_minus_sched"]

tr = (pl.scan_parquet(B + "data/work/feat_train.parquet").select(base_cols)
      .with_columns([
          pl.col("AOBT_3_flt").is_null().alias("unm"),             # routing definition (model B)
          pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),      # D's definition
          pl.col("AIRCRAFT_TYPE_mvt").is_null().alias("actype_null"),
          sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("blk_minus_sched"),
          sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("ms"),
          pl.col("TAXITIME_SEC_mvt").cast(pl.Int64).alias("y"),
          pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
      ])
      .with_columns([
          (pl.col("blk_minus_sched").abs() <= 10).alias("fill10"),
          (pl.col("blk_minus_sched").abs() <= 120).alias("fill120"),
          ((pl.col("y") - pl.col("ms")).abs() <= 60).alias("y_eq_ms60"),
      ])
      .collect())

hdr("S1. training: row counts under both unmatched definitions")
print(tr.select([pl.len().alias("n_train"), pl.col("unm").sum().alias("unm_routing(AOBT_3 null)"),
                 pl.col("unm_fid").sum().alias("unm_fid(FLIGHT_ID null)"),
                 (pl.col("unm") & ~pl.col("unm_fid")).sum().alias("aobt_null_but_fid_present"),
                 (~pl.col("unm") & pl.col("unm_fid")).sum().alias("fid_null_but_aobt_present"),
                 (pl.col("ms") != pl.col("mvt_minus_sched")).sum().alias("ms_ne_feature_mvt_minus_sched"),
                 pl.col("mvt_minus_sched").is_null().sum().alias("feature_ms_null")]))

prof = (tr.group_by(["ADEP_mvt", "unm"]).agg([
    pl.len().alias("n"),
    pl.col("actype_null").mean().round(4).alias("actype_null"),
    pl.col("fill10").mean().round(3).alias("fill10"),
    pl.col("fill120").mean().round(3).alias("fill120"),
    pl.col("y").median().alias("y_med"),
    (pl.col("y") >= H2).mean().round(4).alias("y>=2h"),
    (pl.col("y") >= H2).sum().alias("n_y>=2h"),
    (pl.col("y") >= H4).sum().alias("n_y>=4h"),
    (pl.col("y") >= H12).sum().alias("n_y>=12h"),
    (pl.col("ms") >= H2).sum().alias("n_ms>=2h"),
    (pl.col("FLIGHT_RULE_mvt") == "I").mean().round(3).alias("rule_I"),
]).sort(["unm", "ADEP_mvt"]))
print(prof)

hdr("S1b. LIRF unmatched training rows: detail")
lu = tr.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF"))
print(lu.select([pl.len().alias("n"), pl.col("actype_null").sum().alias("actype_null_n"),
                 pl.col("FLIGHT_RULE_mvt").n_unique().alias("rule_nuniq"),
                 pl.col("FLIGHT_RULE_mvt").first().alias("rule_first"),
                 pl.col("y").median().alias("y_med"), (pl.col("y") >= H2).mean().round(4).alias("share_y>=2h"),
                 pl.col("fill10").mean().round(4).alias("fill10"), pl.col("fill120").mean().round(4).alias("fill120"),
                 pl.col("y_eq_ms60").mean().round(4).alias("y_eq_ms60")]))
print("LIRF unmatched, ms >= 2h:")
print(lu.filter(pl.col("ms") >= H2).select([pl.len().alias("n"), pl.col("actype_null").sum().alias("actype_null_n"),
                                             pl.col("fill10").mean().round(4).alias("fill10"),
                                             pl.col("y_eq_ms60").mean().round(4).alias("y_eq_ms60"),
                                             (pl.col("RUNWAY_mvt") == "25").sum().alias("rwy25_n"),
                                             pl.col("y").median().alias("y_med")]))
# share of all-training tail rows that are LIRF unmatched
for thr, name in [(H2, "y>=2h"), (H4, "y>=4h"), (H12, "y>=12h"), (72000, "y>20h")]:
    t = tr.filter(pl.col("y") >= thr)
    print(f"{name}: all {t.height}, unmatched {t['unm'].sum()}, LIRF-unmatched {(t['unm'] & (t['ADEP_mvt']=='LIRF')).sum()}, "
          f"LIRF-matched {(~t['unm'] & (t['ADEP_mvt']=='LIRF')).sum()}")

# ----------------------------------------------------------------------------------------------
# S2. per-month counts: training (2025) vs ranking (2026)
# ----------------------------------------------------------------------------------------------
rk = (pl.scan_parquet(B + "data/work/feat_rank.parquet")
      .select(["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "AOBT_3_flt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt",
               "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "mvt_minus_sched", "PHASE_mvt"])
      .with_columns([
          pl.col("AOBT_3_flt").is_null().alias("unm"),
          pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),
          pl.col("AIRCRAFT_TYPE_mvt").is_null().alias("actype_null"),
          sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("ms"),
          pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
          pl.col("BLOCK_TIME_UTC_mvt").is_null().alias("block_null"),
      ]).collect())

hdr("S2. ranking: row counts under both definitions, per month")
print(rk.select([pl.len().alias("n_rank"), pl.col("PHASE_mvt").n_unique().alias("phase_nuniq"),
                 pl.col("unm").sum().alias("unm_routing"), pl.col("unm_fid").sum().alias("unm_fid"),
                 (pl.col("unm") & ~pl.col("unm_fid")).sum().alias("aobt_null_but_fid_present"),
                 pl.col("block_null").mean().round(4).alias("block_null_rate"),
                 pl.col("mvt_minus_sched").is_null().sum().alias("feature_ms_null"),
                 (pl.col("ms") != pl.col("mvt_minus_sched")).sum().alias("ms_ne_feature")]))
print(rk.group_by("mo").agg([pl.len().alias("n_dep"), pl.col("unm").sum().alias("unm_routing"),
                             pl.col("unm_fid").sum().alias("unm_fid")]).sort("mo"))

def month_table(df, label, has_y):
    aggs = [pl.len().alias("lirf_unm"),
            pl.col("actype_null").sum().alias("actype_null_n"),
            (pl.col("ms") >= H2).sum().alias("ms>=2h"), (pl.col("ms") >= H4).sum().alias("ms>=4h"),
            (pl.col("ms") >= H12).sum().alias("ms>=12h"), pl.col("ms").max().alias("ms_max")]
    if has_y:
        aggs += [pl.col("fill10").mean().round(3).alias("fill10"), (pl.col("y") >= H2).sum().alias("y>=2h"),
                 (pl.col("y") >= H4).sum().alias("y>=4h"), (pl.col("y") >= H12).sum().alias("y>=12h"),
                 (pl.col("y") >= H2).mean().round(3).alias("share_y>=2h"), pl.col("y").median().alias("y_med")]
    lirf_dep = df.filter(pl.col("ADEP_mvt") == "LIRF").group_by("mo").agg([pl.len().alias("lirf_dep")])
    all_unm = df.filter(pl.col("unm")).group_by("mo").agg([pl.len().alias("all_unm")])
    t = (df.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF")).group_by("mo").agg(aggs)
         .join(lirf_dep, on="mo").join(all_unm, on="mo")
         .with_columns([(pl.col("lirf_unm") / pl.col("lirf_dep") * 100).round(2).alias("lirf_unm_%_of_lirf_dep"),
                        (pl.col("lirf_unm") / pl.col("all_unm") * 100).round(1).alias("lirf_%_of_all_unm")])
         .sort("mo"))
    print(f"--- {label} ---"); print(t)
    return t

hdr("S2b. LIRF unmatched per month: training 2025 vs ranking 2026 (routing definition)")
tt = month_table(tr, "training 2025", True)
rt = month_table(rk, "ranking 2026", False)
print("training totals:", tr.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF")).select([
    pl.len(), (pl.col("ms") >= H2).sum().alias("ms>=2h"), (pl.col("ms") >= H4).sum().alias("ms>=4h"), (pl.col("ms") >= H12).sum().alias("ms>=12h")]))
print("ranking totals:", rk.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF")).select([
    pl.len(), (pl.col("ms") >= H2).sum().alias("ms>=2h"), (pl.col("ms") >= H4).sum().alias("ms>=4h"), (pl.col("ms") >= H12).sum().alias("ms>=12h"),
    pl.col("ms").max().alias("ms_max"), pl.col("ms").filter(pl.col("ms") >= H2).median().alias("ms_med_given>=2h")]))
print("Jan+Jul 2025 training LIRF unmatched:", tr.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("mo").is_in([1, 7])).select([
    pl.len(), (pl.col("ms") >= H2).sum().alias("ms>=2h"), (pl.col("ms") >= H4).sum().alias("ms>=4h"), (pl.col("ms") >= H12).sum().alias("ms>=12h"),
    (pl.col("y") >= H2).sum().alias("y>=2h"), (pl.col("y") >= H4).sum().alias("y>=4h"), (pl.col("y") >= H12).sum().alias("y>=12h")]))
# same under D's definition
print("ranking LIRF unmatched (FLIGHT_ID-null) per month:",
      rk.filter(pl.col("unm_fid") & (pl.col("ADEP_mvt") == "LIRF")).group_by("mo").agg([pl.len(), (pl.col("ms") >= H2).sum().alias("ms>=2h")]).sort("mo"))

hdr("S2c. unmatched per airport x month, training vs ranking (all airports, routing definition)")
print(tr.filter(pl.col("unm")).group_by(["ADEP_mvt", "mo"]).len().pivot(on="mo", index="ADEP_mvt", values="len").sort("ADEP_mvt"))
print(rk.filter(pl.col("unm")).group_by(["ADEP_mvt", "mo"]).len().pivot(on="mo", index="ADEP_mvt", values="len").sort("ADEP_mvt"))
print("ranking: unmatched ms>=2h per airport:",
      rk.filter(pl.col("unm") & (pl.col("ms") >= H2)).group_by("ADEP_mvt").agg([pl.len().alias("n_ms>=2h"), (pl.col("ms") >= H12).sum().alias("n_ms>=12h")]).sort("n_ms>=2h", descending=True))

# ----------------------------------------------------------------------------------------------
# S3. OOF SSE share by ADEP x matched/unmatched (oof_main-48c19bc830)
# ----------------------------------------------------------------------------------------------
hdr("S3. OOF oof_main-48c19bc830: SSE share by ADEP x unmatched")
oof = pl.scan_parquet(B + "experiments/oof_main-48c19bc830.parquet").filter(pl.col("oof").is_not_null()).collect()
o = oof.join(tr.select(["MVT_ID_mvt", "ADEP_mvt", "unm", "y", "ms", "mo", "fill10"]), on="MVT_ID_mvt", how="inner")
o = o.with_columns(((pl.col("oof") - pl.col("y")) ** 2).alias("se"))
SSE = o["se"].sum(); N = o.height
print(f"OOF rows {N}, RMSE {math.sqrt(SSE/N):.2f}, months present {sorted(o['mo'].unique().to_list())}")
print(f"unmatched rows {o['unm'].sum()}, unmatched RMSE {math.sqrt(o.filter(pl.col('unm'))['se'].sum()/o['unm'].sum()):.1f}, "
      f"matched RMSE {math.sqrt(o.filter(~pl.col('unm'))['se'].sum()/(~o['unm']).sum()):.1f}")
g = (o.group_by(["ADEP_mvt", "unm"]).agg([pl.len().alias("n"), pl.col("se").sum().alias("sse"),
                                          (pl.col("y") >= H2).sum().alias("n_y>=2h")])
     .with_columns([(pl.col("sse") / SSE * 100).round(2).alias("sse_%"), (pl.col("sse") / pl.col("n")).sqrt().round(0).alias("rmse")])
     .sort("sse_%", descending=True))
print(g)
print("unmatched-side SSE share of total: %.1f %%" % (o.filter(pl.col("unm"))["se"].sum() / SSE * 100))

hdr("S3b. OOF top-k rows by squared error: share of total SSE and composition")
os_ = o.sort("se", descending=True)
for k in [10, 20, 50, 100, 200, 2000]:
    top = os_.head(k)
    print(f"top-{k}: SSE share {top['se'].sum()/SSE*100:.1f} %, LIRF-unm {int((top['unm'] & (top['ADEP_mvt']=='LIRF')).sum())}, "
          f"other-unm {int((top['unm'] & (top['ADEP_mvt']!='LIRF')).sum())}, matched {int((~top['unm']).sum())}, "
          f"y>=2h {int((top['y']>=H2).sum())}, y>=12h {int((top['y']>=H12).sum())}")
print("top-20 rows:")
print(os_.head(20).select(["ADEP_mvt", "unm", "mo", "y", "ms", "fill10", pl.col("oof").round(0), pl.col("se").round(0)]))
print("RMSE without top-20 rows: %.1f; without top-2000: %.1f" % (
    math.sqrt(os_.slice(20)["se"].sum() / (N - 20)), math.sqrt(os_.slice(2000)["se"].sum() / (N - 2000))))
lu_o = o.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF"))
print("unmatched RMSE without LIRF-unmatched rows: %.1f (n=%d); LIRF-unmatched RMSE %.1f (n=%d)" % (
    math.sqrt(o.filter(pl.col("unm") & (pl.col("ADEP_mvt") != "LIRF"))["se"].sum() / (o["unm"].sum() - lu_o.height)),
    o["unm"].sum() - lu_o.height, math.sqrt(lu_o["se"].sum() / lu_o.height), lu_o.height))

hdr("S3c. OOF per month: RMSE, LIRF-unmatched count and SSE share (CV-set-to-set variation)")
pm = (o.group_by("mo").agg([pl.len().alias("n"), (pl.col("se").sum() / pl.len()).sqrt().round(1).alias("rmse"),
                            (pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF")).sum().alias("lirf_unm"),
                            (pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("y") >= H2)).sum().alias("lirf_unm_y>=2h"),
                            (pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("y") >= H12)).sum().alias("lirf_unm_y>=12h"),
                            pl.col("se").filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF")).sum().alias("sse_lirf_unm"),
                            pl.col("se").sum().alias("sse")])
      .with_columns([(pl.col("sse_lirf_unm") / pl.col("sse") * 100).round(1).alias("lirf_unm_sse_%"),
                     ((pl.col("sse") - pl.col("sse_lirf_unm")) / pl.col("n")).sqrt().round(1).alias("rmse_wo_lirf_unm")])
      .drop(["sse_lirf_unm", "sse"]).sort("mo"))
print(pm)

# ----------------------------------------------------------------------------------------------
# S4. submission files on the 5,290 unmatched ranking rows
# ----------------------------------------------------------------------------------------------
hdr("S4. submission files joined to submitting.parquet on MVT_ID_mvt")
sub = pl.read_parquet(B + "data/raw/submitting.parquet").select(["MVT_ID_mvt"])
print("submitting rows", sub.height, "unique MVT_ID", sub["MVT_ID_mvt"].n_unique(),
      "| ranking DEP rows", rk.height, "unique", rk["MVT_ID_mvt"].n_unique(),
      "| intersection", sub.join(rk.select("MVT_ID_mvt"), on="MVT_ID_mvt").height)

files = {
    "v2": B + "outputs/v2_two_clock.parquet",
    "v3": UP + "v3.parquet",
    "v4": UP + "v4.parquet",
    "v7": B + "outputs/main.parquet",           # timestamp 09-15 15:52 == pipeline entry oof_main-48c19bc830 -> uploaded as v7
    "v8": B + "outputs/det_a.parquet",
    "v9": B + "outputs/probe_unmatched_from_v4.parquet",
    "v5cand": B + "outputs/v5_unmatched_capacity.parquet",   # 09-14 02:04 file; NOT verified to be the uploaded v5 (outputs/main.parquet overwritten)
    "ab_split": B + "outputs/ab_split.parquet",
}
avail = {}
for k, f in files.items():
    if os.path.exists(f):
        avail[k] = pl.read_parquet(f).rename({"TAXITIME_SEC_mvt": k}).with_columns(pl.col(k).cast(pl.Float64))
    else:
        print("MISSING:", k, f)
S = rk.select(["MVT_ID_mvt", "ADEP_mvt", "unm", "unm_fid", "actype_null", "ms", "mo", "RUNWAY_mvt"])
for k, d in avail.items():
    S = S.join(d, on="MVT_ID_mvt", how="left")
print("null predictions after join:", {k: int(S[k].null_count()) for k in avail})

# probe identity check
print("v9 == v8 on matched rows:", bool((S.filter(~pl.col("unm"))["v9"] == S.filter(~pl.col("unm"))["v8"]).all()),
      "| v9 == v4 on unmatched rows:", bool((S.filter(pl.col("unm"))["v9"] == S.filter(pl.col("unm"))["v4"]).all()),
      "| n rows where v9 != v8:", int((S["v9"] != S["v8"]).sum()), "| n rows where v9 != v4:", int((S["v9"] != S["v4"]).sum()))

U = S.filter(pl.col("unm"))
print(f"\nUNMATCHED {U.height} rows (routing definition)")
rows = []
for k in avail:
    c = U[k]
    rows.append({"sub": k, "med": c.median(), "mean": round(c.mean(), 1), "p99": round(c.quantile(0.99), 0), "max": c.max(),
                 "n>7200": int((c > H2).sum()), "n>14400": int((c > H4).sum()), "n>43200": int((c > H12).sum()),
                 "n>=7200": int((c >= H2).sum()), "n>=43200": int((c >= H12).sum()),
                 "n>43200_LIRF": int(((c > H12) & (U["ADEP_mvt"] == "LIRF")).sum()),
                 "n>7200_LIRF": int(((c > H2) & (U["ADEP_mvt"] == "LIRF")).sum())})
print(pl.DataFrame(rows))
print("\nALL 344,841 rows: predictions >= 2h / >= 12h per submission")
rows = []
for k in avail:
    c = S[k]
    rows.append({"sub": k, "n>=7200_all": int((c >= H2).sum()), "n>=43200_all": int((c >= H12).sum()),
                 "n>=7200_matched": int(((c >= H2) & ~S["unm"]).sum()), "n>=43200_matched": int(((c >= H12) & ~S["unm"]).sum())})
print(pl.DataFrame(rows))
# training-frequency expectations
r2_u = (tr.filter(pl.col("unm"))["y"] >= H2).mean(); r12_u = (tr.filter(pl.col("unm"))["y"] >= H12).mean()
r2_a = (tr["y"] >= H2).mean(); r12_a = (tr["y"] >= H12).mean()
print(f"training label rates: unmatched y>=2h {r2_u:.5f} (x5290 = {r2_u*U.height:.1f}), y>=12h {r12_u:.5f} (x5290 = {r12_u*U.height:.1f}); "
      f"all rows y>=2h {r2_a:.6f} (x344841 = {r2_a*S.height:.1f}), y>=12h {r12_a:.7f} (x344841 = {r12_a*S.height:.1f})")
print("training unmatched: y>=2h", int((tr.filter(pl.col("unm"))["y"] >= H2).sum()), "y>43200", int((tr.filter(pl.col("unm"))["y"] > H12).sum()),
      "y>=43200", int((tr.filter(pl.col("unm"))["y"] >= H12).sum()), "| all rows y>=43200", int((tr["y"] >= H12).sum()))

hdr("S4b. pairwise RMS differences and Σgap² by airport (unmatched rows)")
pairs = [("v8", "v4"), ("v7", "v4"), ("v2", "v4"), ("v3", "v4"), ("v8", "v2"), ("v7", "v8"), ("v8", "v5cand")]
for a, b in pairs:
    if a not in avail or b not in avail:
        continue
    d = (U[a] - U[b])
    g2 = (d ** 2)
    tot = g2.sum()
    lirf = g2.filter(U["ADEP_mvt"] == "LIRF").sum()
    ms2 = g2.filter(U["ms"] >= H2).sum()
    top20 = g2.sort(descending=True).head(20).sum()
    dm = (S.filter(~pl.col("unm"))[a] - S.filter(~pl.col("unm"))[b])
    print(f"{a}-{b}: unmatched RMS {math.sqrt(tot/U.height):.1f} | LIRF share of Σgap² {lirf/tot*100:.1f} % | ms>=2h rows share {ms2/tot*100:.1f} % "
          f"| top-20 share {top20/tot*100:.1f} % | RMS on ms>=2h rows {math.sqrt(ms2/(U['ms']>=H2).sum()):.0f} (n={(U['ms']>=H2).sum()}), "
          f"elsewhere {math.sqrt((tot-ms2)/(U['ms']<H2).sum()):.0f} | matched RMS {math.sqrt((dm**2).sum()/dm.len()):.1f}")
print("\nΣ(v8−v4)² by airport (unmatched):")
print(U.with_columns(((pl.col("v8") - pl.col("v4")) ** 2).alias("g2")).group_by("ADEP_mvt").agg([
    pl.len().alias("n"), pl.col("g2").sum().alias("sum_g2"), (pl.col("g2").sum() / pl.len()).sqrt().round(0).alias("rms")])
      .with_columns((pl.col("sum_g2") / pl.col("sum_g2").sum() * 100).round(1).alias("share_%")).sort("share_%", descending=True))

hdr("S4c. top-20 disagreement rows v8 vs v4 (unmatched), with all submissions")
top = (U.with_columns(((pl.col("v8") - pl.col("v4")) ** 2).alias("g2")).sort("g2", descending=True).head(20)
       .select(["MVT_ID_mvt", "ADEP_mvt", "mo", "ms", "actype_null", "RUNWAY_mvt", "v2", "v3", "v4", "v7", "v8"] + (["v5cand"] if "v5cand" in avail else []) + ["g2"]))
print(top)
print("of top-20: LIRF", int((top["ADEP_mvt"] == "LIRF").sum()), "| actype_null", int(top["actype_null"].sum()),
      "| ms>=2h", int((top["ms"] >= H2).sum()), "| ms>=12h", int((top["ms"] >= H12).sum()),
      "| Jan", int((top["mo"] == 1).sum()), "Jul", int((top["mo"] == 7).sum()))
print("v4 - ms on these rows (median):", (top["v4"] - top["ms"]).median(), "| v8 - ms median:", (top["v8"] - top["ms"]).median(),
      "| v4/ms median:", (top["v4"] / top["ms"]).median(), "| v8/ms median:", (top["v8"] / top["ms"]).median())

hdr("S4d. ranking LIRF unmatched ms>=2h rows: what each submission predicts")
L = U.filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("ms") >= H2))
print("n =", L.height, "| ms median", L["ms"].median(), "| ms>=12h", int((L["ms"] >= H12).sum()))
rows = []
for k in avail:
    rows.append({"sub": k, "pred_med": L[k].median(), "pred_p10": round(L[k].quantile(0.1), 0), "pred_p90": round(L[k].quantile(0.9), 0),
                 "n_pred>=2h": int((L[k] >= H2).sum()), "n_pred>=12h": int((L[k] >= H12).sum()),
                 "pred/ms_med": round((L[k] / L["ms"]).median(), 3)})
print(pl.DataFrame(rows))
print("the 11 LIRF unmatched rows with ms>=12h:")
print(L.filter(pl.col("ms") >= H12).select(["mo", "ms", "RUNWAY_mvt", "v2", "v3", "v4", "v7", "v8"] + (["v5cand"] if "v5cand" in avail else [])).sort("ms", descending=True))

# ----------------------------------------------------------------------------------------------
# S5. LB-based decomposition using v4 / v8 / v9 (exact identities, no assumption) and implied unmatched RMSE
# ----------------------------------------------------------------------------------------------
hdr("S5. leaderboard identities from v4 (305.027), v8 (343.7964), v9 (306.4555)")
N_all, N_u, N_m = S.height, U.height, S.height - U.height
LB = {"v1": 345.352, "v2": 312.0237, "v3": 306.2942, "v4": 305.027, "v5": 325.2236, "v6": 350.6249, "v7": 342.5933, "v8": 343.7964, "v9": 306.4555}
sse = {k: v ** 2 * N_all for k, v in LB.items()}
# v9 = v8 matched + v4 unmatched  =>  SSE9 - SSE4 = SSE_m(v8) - SSE_m(v4);  SSE8 - SSE9 = SSE_u(v8) - SSE_u(v4)
d_m = sse["v9"] - sse["v4"]; d_u = sse["v8"] - sse["v9"]
print(f"N_all {N_all}, N_unmatched {N_u}, N_matched {N_m}")
print(f"SSE(v8)-SSE(v4) total = {sse['v8']-sse['v4']:.3e}; matched part (v9-v4) = {d_m:.3e} ({d_m/(sse['v8']-sse['v4'])*100:.1f} %), "
      f"unmatched part (v8-v9) = {d_u:.3e} ({d_u/(sse['v8']-sse['v4'])*100:.1f} %)")
print(f"if matched RMSE(v4)=218 -> matched RMSE(v8)= {math.sqrt((218**2*N_m + d_m)/N_m):.2f}")
g2_u = ((U["v8"] - U["v4"]) ** 2).sum()
print(f"local Σ(v8−v4)² on unmatched = {g2_u:.3e}; LB unmatched SSE gap = {d_u:.3e}; ratio gap/Σgap² = {d_u/g2_u:.2f}")
print("(SSE_u(v8) − SSE_u(v4) = Σgap² + 2·Σ gap·(v4 − y); the cross term needs labels)")
print("\nimplied unmatched RMSE per submission if matched RMSE is fixed at 218 (and at 226 for sensitivity):")
rows = []
for k, v in LB.items():
    for mrm in (218.0, 226.0):
        val = (v ** 2 * N_all - mrm ** 2 * N_m) / N_u
        rows.append({"sub": k, "LB": v, "matched_fixed": mrm, "implied_unmatched_rmse": round(math.sqrt(val), 0) if val > 0 else None})
print(pl.DataFrame(rows).pivot(on="matched_fixed", index=["sub", "LB"], values="implied_unmatched_rmse"))
# how big a per-row error would 20 LIRF rows need to carry to explain the v8-v4 unmatched SSE gap?
print(f"unmatched SSE gap v8−v4 spread over 20 rows -> RMS per row {math.sqrt(d_u/20):.0f} s; over 181 rows -> {math.sqrt(d_u/181):.0f} s; over 5290 -> {math.sqrt(d_u/5290):.0f} s")
# hypothetical: if the 11 LIRF ms>=12h rows have y = ms (M's supposition) or y = ms + 86400 style (P's supposition), what would Σ(pred−y)² be for v4 and v8?
for lab, yfun in [("y=ms", lambda d: d["ms"]), ("y=86400+1200 (rollover)", lambda d: pl.Series([DAY + 1200] * d.height))]:
    yy = yfun(L)
    s4 = ((L["v4"] - yy) ** 2).sum(); s8 = ((L["v8"] - yy) ** 2).sum()
    print(f"hypothetical on the {L.height} LIRF unm ms>=2h rows under {lab}: SSE v4 {s4:.3e}, v8 {s8:.3e}, diff v8−v4 {s8-s4:.3e} "
          f"(= {(s8-s4)/d_u*100:.0f} % of LB unmatched gap)")
L11 = L.filter(pl.col("ms") >= H12)
yy = L11["ms"]
print(f"hypothetical on the {L11.height} LIRF unm ms>=12h rows under y=ms: SSE v4 {((L11['v4']-yy)**2).sum():.3e}, v8 {((L11['v8']-yy)**2).sum():.3e}, "
      f"diff {((L11['v8']-yy)**2).sum()-((L11['v4']-yy)**2).sum():.3e} (= {(((L11['v8']-yy)**2).sum()-((L11['v4']-yy)**2).sum())/d_u*100:.0f} % of LB unmatched gap)")
print("\nDONE")

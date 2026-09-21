"""P1 §5 Stage 3 [S-new]: merry-quicksand's LIRF schedule-floor invariant, verified on our
training data, plus the ranking rows it would touch. Read-only; writes nothing."""

import polars as pl, numpy as np
tr = pl.read_parquet("data/work/feat_train.parquet",
    columns=["MVT_ID_mvt","ADEP_mvt","TAXITIME_SEC_mvt","mvt_minus_aobt3","mvt_minus_sched","MVT_TIME_UTC_mvt"])
u = tr.filter(pl.col("mvt_minus_aobt3").is_null())
print("train unmatched", u.height)
li = u.filter(pl.col("ADEP_mvt")=="LIRF"); nl = u.filter(pl.col("ADEP_mvt")!="LIRF")
print("LIRF unm", li.height, "non-LIRF unm", nl.height)
# merry-quicksand schedule_floor claim: every LIRF unmatched row with ms >= 24000 has y >= ms - 60
for lo in (14400, 18000, 24000, 28800, 36000):
    s = li.filter(pl.col("mvt_minus_sched")>=lo)
    if s.height==0: continue
    viol = s.filter(pl.col("TAXITIME_SEC_mvt") < pl.col("mvt_minus_sched")-60)
    print(f"LIRF unm ms>={lo}: n={s.height}  violations(y < ms-60)={viol.height}  min(y-ms)={ (s['TAXITIME_SEC_mvt']-s['mvt_minus_sched']).min() }")
# same at non-LIRF
for lo in (14400, 24000):
    s = nl.filter(pl.col("mvt_minus_sched")>=lo)
    viol = s.filter(pl.col("TAXITIME_SEC_mvt") < pl.col("mvt_minus_sched")-60)
    print(f"nonLIRF unm ms>={lo}: n={s.height} violations={viol.height}")
# ranking counts
rk = pl.read_parquet("data/work/feat_rank.parquet", columns=["MVT_ID_mvt","ADEP_mvt","mvt_minus_aobt3","mvt_minus_sched"])
ru = rk.filter(pl.col("mvt_minus_aobt3").is_null())
rli = ru.filter(pl.col("ADEP_mvt")=="LIRF")
for lo in (14400,24000,40000,43200,86400):
    print(f"rank LIRF unm ms>={lo}: {rli.filter(pl.col('mvt_minus_sched')>=lo).height}")
# cap arithmetic on v10 non-LIRF
sub = pl.read_parquet("outputs/v10_probe_reparam.parquet").rename({"TAXITIME_SEC_mvt":"pred"}).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
j = rk.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).join(sub, on="MVT_ID_mvt")
nlr = j.filter(pl.col("mvt_minus_aobt3").is_null() & (pl.col("ADEP_mvt")!="LIRF"))
p = nlr["pred"].to_numpy().astype(float)
N=344841.0; base=284.3584; baseSSE=N*base**2
for c in (1800,2400,3000,3600):
    d = np.maximum(p-c,0.0)
    # best case: those rows' true labels are all <= c, so SSE falls by sum((pred-y)^2-(c-y)^2) >= sum(d^2) is NOT a bound;
    # report the row count and the shift magnitude instead
    print(f"cap {c}: rows moved {int((p>c).sum())}, total shift {d.sum():.0f}s, max shift {d.max():.0f}s, sum(shift^2)={np.sum(d**2):.3e}")
# LIRF floor arithmetic on v10
rlij = j.filter(pl.col("mvt_minus_aobt3").is_null() & (pl.col("ADEP_mvt")=="LIRF"))
for lo in (24000,):
    s = rlij.filter((pl.col("mvt_minus_sched")>=lo) & (pl.col("mvt_minus_sched")<86400))
    below = s.filter(pl.col("pred") < pl.col("mvt_minus_sched"))
    print(f"rank LIRF unm {lo}<=ms<86400: n={s.height}, pred<ms on {below.height} rows, median gap {(below['mvt_minus_sched']-below['pred']).median() if below.height else None}")

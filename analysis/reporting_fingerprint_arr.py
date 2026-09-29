"""ARR-side fingerprints (visible in ranking) and their correlation with DEP clock source at LIRF (report §4).
Part 1: ARR in-block copy excess per airport/split. Part 2: LIRF DEP vs ARR sched-copy rates by operator/day/stand."""
import sys
from reporting_fingerprint_common import *
import numpy as np
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
df = load_all().filter(pl.col("PHASE_mvt")=="ARR")
rows=[]
for sp in ["train","rank"]:
  for ref,name,base in [("SCHED_TIME_UTC_mvt","sched","BLOCK_TIME_UTC_mvt"),("ARVT_3_flt","arvt3","BLOCK_TIME_UTC_mvt"),("ARVT_1_flt","arvt1","BLOCK_TIME_UTC_mvt"),("ARVT_3_flt","arvt3_mvt","MVT_TIME_UTC_mvt"),("SCHED_TIME_UTC_mvt","sched_mvt","MVT_TIME_UTC_mvt")]:
    x = df.filter(pl.col("split")==sp).with_columns((pl.col(base)-pl.col(ref)).dt.total_seconds().alias("dd"))
    for apt in sorted(x["apt"].unique()):
        v = x.filter(pl.col("apt")==apt)["dd"].drop_nulls().to_numpy()
        n = {k: int(((v>=60*k-6)&(v<=60*k+6)).sum()) for k in (-1,0,1)}
        ex=n[0]-(n[-1]+n[1])/2
        rows.append(dict(split=sp,cmp=f"{base[:5]}-{name}", apt=apt, n=len(v), km1=n[-1],k0=n[0],kp1=n[1], copy_rate=round(ex/len(v),4), purity=round(ex/max(n[0],1),3)))
t=pl.DataFrame(rows)
print(t.filter(pl.col("split")=="train"))
print(t.filter(pl.col("split")=="rank").select("cmp","apt","copy_rate","purity"))

# ---- part 2
df = load_all().filter(pl.col("apt")=="LIRF").with_columns(
    ((pl.col("BLOCK_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs()<=6).alias("copy_s"),
    ((pl.col("BLOCK_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()).alias("b_s"),
    ((pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()).alias("m_s"),
)
dep=df.filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("split")=="train")); arr=df.filter((pl.col("PHASE_mvt")=="ARR")&(pl.col("split")=="train"))
print("DEP copy_s rate", dep["copy_s"].mean(), "ARR", arr["copy_s"].mean())
# by month
print(pl.concat([dep.group_by("month").agg(pl.col("copy_s").mean().alias("dep")).sort("month"), arr.group_by("month").agg(pl.col("copy_s").mean().alias("arr")).sort("month").drop("month")],how="horizontal_extend"))
# by op
g=dep.group_by("AIRCRAFT_OPERATOR_flt").agg(pl.len().alias("nd"),pl.col("copy_s").mean().alias("dep")).join(arr.group_by("AIRCRAFT_OPERATOR_flt").agg(pl.len().alias("na"),pl.col("copy_s").mean().alias("arr")),on="AIRCRAFT_OPERATOR_flt",how="full",coalesce=True).sort("nd",descending=True,nulls_last=True)
print(g.head(25).with_columns(pl.col("AIRCRAFT_OPERATOR_flt").str.slice(0,8)))
print("corr op-level (nd>=200):", g.filter((pl.col("nd")>=200)&(pl.col("na")>=200)).select(pl.corr("dep","arr")))
# market segment
print(dep.group_by("MARKET_SEGMENT_flt").agg(pl.len(),pl.col("copy_s").mean()).sort("len",descending=True))
# delay dependency: copy rate vs MVT-SCHED
print(dep.with_columns((pl.col("m_s")/600).floor().clip(-3,24).alias("bin")).group_by("bin").agg(pl.len(),pl.col("copy_s").mean()).sort("bin"))
# hour local
print(dep.with_columns(pl.col("SCHED_TIME_UTC_mvt").dt.hour().alias("h")).group_by("h").agg(pl.len(),pl.col("copy_s").mean()).sort("h"))
tr=df.filter(pl.col("split")=="train")
dep=tr.filter(pl.col("PHASE_mvt")=="DEP"); arr=tr.filter(pl.col("PHASE_mvt")=="ARR")
op="AIRCRAFT_OPERATOR_flt"
for keys in [["date"],["date",op],["date",op,"STAND_mvt"],["date","STAND_mvt"]]:
    a=arr.group_by(keys).agg(pl.col("copy_s").mean().alias("arr"),pl.len().alias("na"))
    d=dep.group_by(keys).agg(pl.col("copy_s").mean().alias("dep"),pl.len().alias("nd"))
    j=d.join(a,on=keys,how="inner").filter((pl.col("na")>=3)&(pl.col("nd")>=3))
    # remove static operator mean if op in keys
    if op in keys:
        j=j.with_columns((pl.col("dep")-pl.col("dep").mean().over(op)).alias("dep_c"),(pl.col("arr")-pl.col("arr").mean().over(op)).alias("arr_c"))
    else:
        j=j.with_columns((pl.col("dep")-pl.col("dep").mean()).alias("dep_c"),(pl.col("arr")-pl.col("arr").mean()).alias("arr_c"))
    w=j["nd"].to_numpy().astype(float); x=j["arr_c"].to_numpy(); y=j["dep_c"].to_numpy()
    c=np.sum(w*x*y)/np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))
    print(keys, "groups", j.height, "raw corr", round(j.select(pl.corr("dep","arr")).item(),3), "within-op/centered weighted corr", round(c,3))
# daily time series
dd=dep.group_by("date").agg(pl.col("copy_s").mean().alias("dep")).join(arr.group_by("date").agg(pl.col("copy_s").mean().alias("arr")),on="date").sort("date")
print(dd.select(pl.corr("dep","arr")), dd.describe())
print(dd.sort("dep",descending=True).head(10)); print(dd.sort("dep").head(5))

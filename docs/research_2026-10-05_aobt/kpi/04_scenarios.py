# Step 4: recompute the ATXOT KPI excluding each suspect class (a), all classes (b), reference-only / measurement-only (c),
# plus a repair scenario (fills replaced by NM AOBT_3) and a timestamp-precision simulation. Bias in min/flight and in ranking.
import polars as pl, numpy as np, sys
from common import *
pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_rows(300)
DEICE = (sys.argv[1] if len(sys.argv)>1 else "deice")=="deice"
TAG = "deice" if DEICE else "nodeice"
df = load()
f = base_filter(df, deice=DEICE).with_columns(
    t_rep = pl.when(pl.col("F_PLANFILL") & pl.col("AOBT_3_flt").is_not_null())
              .then((pl.col("MVT_TIME_UTC_mvt")-pl.col("AOBT_3_flt")).dt.total_seconds()).otherwise(pl.col("TAXITIME_SEC_mvt")),
    t_round = ((pl.col("MVT_TIME_UTC_mvt").dt.round("1m")-pl.col("BLOCK_TIME_UTC_mvt").dt.round("1m")).dt.total_seconds()),
    t_floor = ((pl.col("MVT_TIME_UTC_mvt").dt.truncate("1m")-pl.col("BLOCK_TIME_UTC_mvt").dt.truncate("1m")).dt.total_seconds()),
)
c = pl.col
CLASSES = {
  "A3_5":   c("F_A3_5"), "A3_60": c("F_A3_60"),
  "SCHED_6": c("F_SCHED_6"), "SCHED_60": c("F_SCHED_60"), "EOBT1_6": c("F_EOBT1_6"),
  "PLAN_6": c("F_PLAN_6"), "PLAN_60": c("F_PLAN_60"),
  "PLANFILL": c("F_PLANFILL"), "PLAN_NMEARLY": c("F_PLAN_NMEARLY"),
  "NEG_SHORT": c("F_NEG")|c("F_SHORT"), "ROLL_LONG": c("F_ROLL")|c("F_LONG"),
  "UNMATCHED": c("unmatched"),
  "ALL_strict": c("F_A3_5")|c("F_PLAN_6")|c("F_NEG")|c("F_SHORT"),
  "ALL_fill": c("F_PLANFILL")|c("F_PLAN_NMEARLY")|c("F_NEG")|c("F_SHORT"),
  "ALL_loose": c("F_A3_60")|c("F_PLAN_60")|c("F_NEG")|c("F_SHORT"),
}
base = kpi(f, window="FY")
def summarize(res, name, mode):
    an = annual(res)
    return res.with_columns(scen=pl.lit(name), mode=pl.lit(mode)), an.with_columns(scen=pl.lit(name), mode=pl.lit(mode))
M=[];A=[]
m,a = summarize(base,"BASE","-"); M.append(m); A.append(a)
for k,e in CLASSES.items():
    for mode,(rm,mm) in {"both":(~e,~e),"ref_only":(~e,None),"meas_only":(None,~e)}.items():
        m,a = summarize(kpi(f, ref_mask=rm, meas_mask=mm, window="FY"), k, mode); M.append(m); A.append(a)
for name,tc in [("REPAIR_fill_with_AOBT3","t_rep"),("PREC_round_min","t_round"),("PREC_floor_min","t_floor")]:
    m,a = summarize(kpi(f, window="FY", tcol=tc), name, "both"); M.append(m); A.append(a)
M=pl.concat(M); A=pl.concat(A)
b = A.filter(c("scen")=="BASE").select("ADEP_mvt", c("add").alias("add0"), c("ref").alias("ref0"), c("n_ref").alias("n0"))
A = A.join(b, on="ADEP_mvt").with_columns(d_add=c("add")-c("add0"), d_ref=c("ref")-c("ref0"), keep=c("n_ref")/c("n0"),
        rank=c("add").rank("ordinal",descending=True).over(["scen","mode"]), rank0=c("add0").rank("ordinal",descending=True).over(["scen","mode"]))
A.write_csv(f"out/scenarios_annual_{TAG}.csv"); M.write_csv(f"out/scenarios_monthly_{TAG}.csv")
# monthly deltas
bm = M.filter(c("scen")=="BASE").select("ADEP_mvt","lmon", c("add").alias("add0"))
Md = M.join(bm, on=["ADEP_mvt","lmon"]).with_columns(d=c("add")-c("add0"))
mr = M.with_columns(rk=c("add").rank("ordinal",descending=True).over(["scen","mode","lmon"]))
mr0 = mr.filter(c("scen")=="BASE").select("ADEP_mvt","lmon",c("rk").alias("rk0"))
mr = mr.join(mr0,on=["ADEP_mvt","lmon"]).group_by(["scen","mode"]).agg(month_rank_changes=(c("rk")!=c("rk0")).sum())
def kendall(x,y):
    n=len(x); s=0
    for i in range(n):
        for j in range(i+1,n): s+=np.sign(x[i]-x[j])*np.sign(y[i]-y[j])
    return s/(n*(n-1)/2)
rows=[]
for (s,mo),g in A.group_by(["scen","mode"]):
    g=g.sort("ADEP_mvt"); md=Md.filter((c("scen")==s)&(c("mode")==mo))
    order = "-".join(g.sort("rank")["ADEP_mvt"].to_list())
    rows.append(dict(scen=s, mode=mo, max_abs_d_add=float(g["d_add"].abs().max()), apt_max=g.sort(c("d_add").abs(),descending=True)["ADEP_mvt"][0],
        mean_d_add=float(g["d_add"].mean()), LIRF_d=float(g.filter(c("ADEP_mvt")=="LIRF")["d_add"][0]),
        max_abs_d_ref=float(g["d_ref"].abs().max()), n_apt_gt03=int((g["d_add"].abs()>0.3).sum()),
        max_abs_month_d=float(md["d"].abs().max()), n_apt_months_gt03=int((md["d"].abs()>0.3).sum()),
        rank_changes=int((g["rank"]!=g["rank0"]).sum()), kendall=kendall(g["add"].to_numpy(), g["add0"].to_numpy()), min_keep=float(g["keep"].min()), order=order))
S = pl.DataFrame(rows).join(mr, on=["scen","mode"]).sort(["scen","mode"])
S.write_csv(f"out/scenarios_summary_{TAG}.csv")
print("BASE order:", "-".join(A.filter(c("scen")=="BASE").sort("rank")["ADEP_mvt"].to_list()))
print(S.drop("order").with_columns(pl.col(pl.Float64).round(3)))
print(S.filter(c("rank_changes")>0).select("scen","mode","order"))
piv = A.filter(c("mode").is_in(["both","-"])).pivot(on="ADEP_mvt", index="scen", values="d_add").with_columns(pl.col(pl.Float64).round(3))
print(piv)
piv2 = A.filter(c("mode")=="ref_only").pivot(on="ADEP_mvt", index="scen", values="d_ref").with_columns(pl.col(pl.Float64).round(3))
print("P10 reference change (min/flight), ref_only:"); print(piv2)

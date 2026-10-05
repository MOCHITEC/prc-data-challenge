"""Clocks angle: NM AOBT_3 vs airport BLOCK, derived-taxi signatures. Outputs CSV tables in ./out"""
import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, numpy as np, json
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260)
os.makedirs("out",exist_ok=True)
D=pl.read_parquet("slim.parquet")
dep=D.filter(pl.col("PHASE_mvt")=="DEP")
GRID={"EDDM","EGLL","LFPG","LIRF","LTFM"}

def spikes(vals_min, n_min_rw=1000):
    """vals_min: np array of rounded minutes. return dict bin->(count,excess) for flagged bins"""
    u,c=np.unique(vals_min,return_counts=True); cnt=dict(zip(u.tolist(),c.tolist())); n=len(vals_min)
    out={}
    for k,v in cnt.items():
        bg=np.median([cnt.get(k+j,0) for j in (-4,-3,-2,2,3,4)])
        ex=v-bg
        if ex>0.5*v and ex/n>0.02: out[k]=(v,ex)
    # pair adjacent spikes (T,T+1): allow neighbour of flagged bin if its own excess vs bg is >0.5
    return out,n

# ---------- 1. derived-signature detection per (APT,RUNWAY) on MVT-AOBT3 ----------
m=dep.filter(pl.col("d_mt").is_not_null()&(pl.col("d_mt").abs()<3*3600)).with_columns(mm=(pl.col("d_mt")/60).round().cast(pl.Int32))
rows=[]; flags=[]
for (apt,rw),g in m.group_by(["APT","RUNWAY_mvt"]):
    if g.height<1000: continue
    sp,n=spikes(g["mm"].to_numpy())
    for k,(v,ex) in sp.items(): rows.append(dict(APT=apt,RUNWAY=rw,n_rw=n,T_min=k,count=v,excess=ex,share=v/n,excess_share=ex/n))
    flags.append(pl.DataFrame({"APT":[apt]*len(sp),"RUNWAY_mvt":[rw]*len(sp),"mm":list(sp.keys())},schema={"APT":pl.Utf8,"RUNWAY_mvt":pl.Utf8,"mm":pl.Int32}))
const=pl.DataFrame(rows).sort("APT","RUNWAY","T_min"); const.write_csv("out/derived_constants_by_runway.csv"); print(const)
fl=pl.concat(flags).with_columns(flag=pl.lit(True))
m=m.join(fl,on=["APT","RUNWAY_mvt","mm"],how="left").with_columns(pl.col("flag").fill_null(False))
# also same-detector on airport y (MVT-BLOCK) to see if airport BLOCK carries constants
yrows=[]
for (apt,rw),g in dep.filter(pl.col("y").is_not_null()&(pl.col("y").abs()<3*3600)).group_by(["APT","RUNWAY_mvt"]):
    if g.height<1000: continue
    sp,n=spikes(np.round(g["y"].to_numpy()/60).astype(int))
    for k,(v,ex) in sp.items(): yrows.append(dict(APT=apt,RUNWAY=rw,n_rw=n,T_min=k,share=v/n,excess_share=ex/n))
print("airport-y spikes:",yrows)
pl.DataFrame(yrows if yrows else [dict(APT=None)]).write_csv("out/airport_y_spikes.csv")

# y-independence estimator: among rows with |y - T| > 8 min, share with d_mt in a flagged bin
m=m.with_columns(far=( (pl.col("y")-pl.col("mm")*60).abs()>480))
per_apt=m.group_by("APT").agg(n=pl.len(),flag_share=pl.col("flag").mean(),
    corr_mt_y=pl.corr("d_mt","y"),
    corr_mt_y_unflag=pl.corr("d_mt","y").filter(~pl.col("flag")),
    ba_med_flag=pl.col("d_ba").filter(pl.col("flag")).median(),ba_med_unflag=pl.col("d_ba").filter(~pl.col("flag")).median(),
    ba_iqr_unflag=(pl.col("d_ba").filter(~pl.col("flag")).quantile(.75)-pl.col("d_ba").filter(~pl.col("flag")).quantile(.25)),
    ).sort("APT")
exc=const.group_by("APT").agg(excess_share_total=(pl.col("excess")).sum()).join(m.group_by("APT").len(),on="APT").with_columns(derived_excess_share=pl.col("excess_share_total")/pl.col("len")).select("APT","derived_excess_share")
per_apt=per_apt.join(exc,on="APT",how="left").with_columns(pl.col("derived_excess_share").fill_null(0))
print(per_apt); per_apt.write_csv("out/derived_share_by_airport.csv")
# conditional flatness: flagged share by decile of airport taxi y
m=m.with_columns(ydec=pl.col("y").rank().over("APT")*10//(pl.len().over("APT")+1))
flat=m.group_by("APT","ydec").agg(flag=pl.col("flag").mean(),y_med=pl.col("y").median()).sort("APT","ydec")
flat.write_csv("out/flag_share_by_taxi_decile.csv")
print(flat.pivot(on="ydec",index="APT",values="flag").sort("APT"))
# by hour and month
byh=m.group_by("APT","hour").agg(n=pl.len(),flag=pl.col("flag").mean()).sort("APT","hour"); byh.write_csv("out/flag_share_by_hour.csv")
print(byh.pivot(on="hour",index="APT",values="flag").sort("APT"))
bym=m.group_by("APT","month").agg(flag=pl.col("flag").mean()).sort("APT","month"); bym.write_csv("out/flag_share_by_month.csv")
print(bym.pivot(on="month",index="APT",values="flag").sort("APT"))
# stand-group association: stand first char
m=m.with_columns(sg=pl.col("STAND_mvt").str.slice(0,1))
sg=m.group_by("APT","sg").agg(n=pl.len(),flag=pl.col("flag").mean(),mm_mode=pl.col("mm").mode().first(),
     mm_mode_share=(pl.col("mm")==pl.col("mm").mode().first()).mean()).filter(pl.col("n")>2000).sort("APT","n",descending=[False,True])
sg.write_csv("out/flag_share_by_standgroup.csv"); print(sg)
m.select("APT","RUNWAY_mvt","month","hour","mm","flag","d_mt","d_ba","y","sec_a").write_parquet("out_m.parquet")

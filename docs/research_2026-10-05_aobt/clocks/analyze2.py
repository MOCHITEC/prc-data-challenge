"""Derived-share estimation via y-independence. needs out/derived_constants_by_runway.csv from analyze.py"""
import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, numpy as np
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260)
D=pl.read_parquet("slim.parquet").filter(pl.col("PHASE_mvt")=="DEP")
const=pl.read_csv("out/derived_constants_by_runway.csv")
m=D.filter(pl.col("d_mt").is_not_null()&pl.col("y").is_not_null()&(pl.col("d_mt").abs()<3*3600)&(pl.col("y").abs()<3*3600)).with_columns(mm=(pl.col("d_mt")/60).round().cast(pl.Int32),ym=(pl.col("y")/60).round().cast(pl.Int32))
res=[]; parts=[]
for (apt,rw),g in m.group_by(["APT","RUNWAY_mvt"]):
    S=sorted(const.filter((pl.col("APT")==apt)&(pl.col("RUNWAY")==rw))["T_min"].to_list())
    mm=g["mm"].to_numpy(); ym=g["ym"].to_numpy(); n=len(mm)
    if not S or n<1000:
        parts.append(g.with_columns(flag=pl.lit(False))); continue
    S=np.array(S)
    inS=np.isin(mm,S)
    def pfar(Sx):
        far=np.min(np.abs(ym[:,None]-Sx[None,:]),axis=1)>=8
        return np.isin(mm[far],Sx).mean(), far.mean()
    p,fs=pfar(S); p0a,_=pfar(S+6); p0b,_=pfar(S-6) if S.min()-6>0 else (np.nan,0)
    plac=np.nanmean([p0a,p0b])
    res.append(dict(APT=apt,RUNWAY=rw,n=n,S=",".join(map(str,S)),flag_share=inS.mean(),p_far=p,placebo=plac,derived_est=max(p-plac,0),far_frac=fs))
    parts.append(g.with_columns(flag=pl.Series(inS)))
r=pl.DataFrame(res).sort("APT","n",descending=[False,True]); r.write_csv("out/derived_est_by_runway.csv"); print(r)
m=pl.concat(parts)
cnt=m.group_by("APT").len()
a=r.with_columns(w=pl.col("derived_est")*pl.col("n"),wf=pl.col("flag_share")*pl.col("n")).group_by("APT").agg(pl.col("w").sum(),pl.col("wf").sum()).join(cnt,on="APT",how="right")
a=a.with_columns(derived_est=(pl.col("w").fill_null(0)/pl.col("len")),flag_share=pl.col("wf").fill_null(0)/pl.col("len")).select("APT",pl.col("len").alias("n"),"flag_share","derived_est")
g2=m.group_by("APT").agg(corr_mt_y=pl.corr("d_mt","y"),
    slope_mt_on_y=pl.cov("d_mt","y")/pl.col("y").var(),
    ba_med=pl.col("d_ba").median(),ba_iqr=pl.col("d_ba").quantile(.75)-pl.col("d_ba").quantile(.25),
    ba_med_unflag=pl.col("d_ba").filter(~pl.col("flag")).median(),ba_med_flag=pl.col("d_ba").filter(pl.col("flag")).median(),
    mt_s0=(pl.col("sec_a")==0).mean())
a=a.join(g2,on="APT").sort("APT")
# corr on unflagged only
cu=m.filter(~pl.col("flag")).group_by("APT").agg(corr_unflag=pl.corr("d_mt","y"),ba_iqr_unflag=pl.col("d_ba").quantile(.75)-pl.col("d_ba").quantile(.25))
a=a.join(cu,on="APT").sort("APT"); print(a); a.write_csv("out/derived_share_by_airport.csv")
# flatness: share of rows in S by decile of airport taxi
m=m.with_columns(ydec=(pl.col("y").rank("ordinal").over("APT")*10/(pl.len().over("APT")+1)).floor().cast(pl.Int32))
flat=m.group_by("APT","ydec").agg(flag=pl.col("flag").mean(),y_med=pl.col("y").median()).sort("APT","ydec"); flat.write_csv("out/flag_share_by_taxi_decile.csv")
print(flat.pivot(on="ydec",index="APT",values="flag").sort("APT"))
byh=m.group_by("APT","hour").agg(n=pl.len(),flag=pl.col("flag").mean()).sort("APT","hour"); byh.write_csv("out/flag_share_by_hour.csv")
print(byh.pivot(on="hour",index="APT",values="flag").sort("APT"))
bym=m.group_by("APT","month").agg(flag=pl.col("flag").mean(),s0=(pl.col("sec_a")==0).mean(),ba_med=pl.col("d_ba").median()).sort("APT","month"); bym.write_csv("out/flag_by_month.csv")
print(bym.pivot(on="month",index="APT",values="flag").sort("APT"))
sg=m.with_columns(sg=pl.col("STAND_mvt").str.slice(0,1)).group_by("APT","sg").agg(n=pl.len(),flag=pl.col("flag").mean(),mm_mode=pl.col("mm").mode().first()).filter(pl.col("n")>3000).sort("APT","n",descending=[False,True])
sg.write_csv("out/flag_by_standgroup.csv"); print(sg)
m.select("APT","RUNWAY_mvt","STAND_mvt","month","hour","mm","ym","flag","d_mt","d_ba","y","sec_a").write_parquet("out_m.parquet")

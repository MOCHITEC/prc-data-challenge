"""Cell-mode concentration: within (APT,STAND,RUNWAY[,month]) cells, share of rows whose minute value falls in the
cell's modal 2-minute pair, for NM taxi (MVT-AOBT3) vs airport taxi (MVT-BLOCK). Derived NM => concentrated, y-independent."""
import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260)
m=pl.read_parquet("out_m.parquet")
def pairshare(df,val,keys,minn=20):
    # count per (cell,val); pair count c_k + c_{k+1}; pick max pair per cell
    c=df.group_by(keys+[val]).len()
    c2=c.with_columns((pl.col(val)-1).alias(val)).rename({"len":"len_next"})
    p=c.join(c2,on=keys+[val],how="left").with_columns(pair=pl.col("len")+pl.col("len_next").fill_null(0))
    best=p.sort("pair",descending=True).group_by(keys).first().select(keys+[pl.col(val).alias(val+"_k"),"pair"])
    n=df.group_by(keys).len().rename({"len":"ncell"})
    best=best.join(n,on=keys).filter(pl.col("ncell")>=minn)
    return best
out=[]
for keys,lab in [(["APT","STAND_mvt","RUNWAY_mvt"],"stand_rwy"),(["APT","STAND_mvt","RUNWAY_mvt","month"],"stand_rwy_month"),(["APT","RUNWAY_mvt"],"rwy")]:
    for val in ["mm","ym"]:
        b=pairshare(m,val,keys)
        a=b.group_by("APT").agg(share=pl.col("pair").sum()/pl.col("ncell").sum(),cover=pl.col("ncell").sum())
        out.append(a.with_columns(level=pl.lit(lab),clock=pl.lit("NM" if val=="mm" else "APT")))
r=pl.concat(out).pivot(on=["level","clock"],index="APT",values="share").sort("APT")
print(r); r.write_csv("out/cellmode_share.csv")
# row-level flag: NM value in modal pair of stand_rwy_month cell
keys=["APT","STAND_mvt","RUNWAY_mvt","month"]
b=pairshare(m,"mm",keys).select(keys+["mm_k",pl.col("pair").alias("pairn"),"ncell"])
by=pairshare(m,"ym",keys).select(keys+["ym_k",pl.col("pair").alias("pairy")])
mm=m.join(b,on=keys,how="inner").join(by,on=keys,how="left").with_columns(
    inNM=(pl.col("mm")==pl.col("mm_k"))|(pl.col("mm")==pl.col("mm_k")+1),
    inAPT=(pl.col("ym")==pl.col("ym_k"))|(pl.col("ym")==pl.col("ym_k")+1))
# within cells: y-independence: corr(d_mt,y) after demeaning by cell
mm=mm.with_columns(dmt_c=pl.col("d_mt")-pl.col("d_mt").mean().over(keys),y_c=pl.col("y")-pl.col("y").mean().over(keys))
a=mm.group_by("APT").agg(n=pl.len(),inNM=pl.col("inNM").mean(),inAPT=pl.col("inAPT").mean(),
    within_corr=pl.corr("dmt_c","y_c"),within_sd_NM=pl.col("dmt_c").std(),within_sd_APT=pl.col("y_c").std(),
    within_slope=pl.cov("dmt_c","y_c")/pl.col("y_c").var()).with_columns(excess=pl.col("inNM")-pl.col("inAPT")).sort("APT")
print(a); a.write_csv("out/cell_signature_by_airport.csv")
mo=mm.group_by("APT","month").agg(inNM=pl.col("inNM").mean(),inAPT=pl.col("inAPT").mean(),within_corr=pl.corr("dmt_c","y_c")).sort("APT","month")
mo.write_csv("out/cell_signature_by_month.csv"); print(mo.pivot(on="month",index="APT",values="within_corr"))
ho=mm.group_by("APT","hour").agg(inNM=pl.col("inNM").mean(),within_corr=pl.corr("dmt_c","y_c")).sort("APT","hour"); ho.write_csv("out/cell_signature_by_hour.csv")
print(ho.pivot(on="hour",index="APT",values="within_corr"))
mm.select("APT","month","hour","inNM","inAPT","dmt_c","y_c","d_ba","y","d_mt").write_parquet("out_cell.parquet")

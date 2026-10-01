import polars as pl
S="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
o=pl.read_parquet(S+"oof_matched_janjul.parquet")
f=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").select("MVT_ID_mvt","RUNWAY_mvt","STAND_mvt","MVT_TIME_UTC_mvt","hour_local","WK_TBL_CAT_flt","mvt_minus_aobt3").collect()
d=o.join(f,on="MVT_ID_mvt").filter("genuine")
tot=(d["err"]**2).sum()
print("genuine rows",d.height,"rmse",(d["err"]**2).mean()**.5)
d=d.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),pl.col("MVT_TIME_UTC_mvt").dt.day().alias("dd"),
   pl.col("STAND_mvt").str.replace(r"\d+.*$","").alias("sp"))
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
def tab(x,keys,minn=200):
    t=x.group_by(keys).agg(pl.len().alias("n"),(pl.col("err")**2).sum().alias("sse"),pl.col("err").mean().round(0).alias("bias"),
        (pl.col("err")**2).mean().sqrt().round(0).alias("rmse")).with_columns((pl.col("sse")/tot*100).round(2).alias("sse%")).filter(pl.col("n")>=minn).sort("sse",descending=True).drop("sse")
    print(t.head(25))
for a in ["LIRF","LTFM","LFPG","EGLL"]:
    x=d.filter(pl.col("ADEP_mvt")==a)
    print("=====",a,x.height,"rmse",round((x["err"]**2).mean()**.5,1))
    tab(x,["mo","RUNWAY_mvt"]); tab(x,["RUNWAY_mvt","sp"])
    tab(x,["mo"]+([] if a!="LFPG" else []) ,0)
# LFPG Jul by closure period
x=d.filter((pl.col("ADEP_mvt")=="LFPG")&(pl.col("mo")==7)).with_columns((pl.col("dd")>=14).alias("closure"))
tab(x,["closure","RUNWAY_mvt"],50)

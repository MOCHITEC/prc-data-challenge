import polars as pl
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_rows(60)
df=pl.read_parquet('dep2025.parquet')
B=pl.col("BLOCK_TIME_UTC_mvt")
def d(c): return ((B-pl.col(c)).dt.total_seconds())
out=[]
for c in ["AOBT_3_flt","SCHED_TIME_UTC_mvt","EOBT_1_flt","LOBT_flt","IOBT_flt"]:
    t=df.with_columns(x=d(c)).group_by("ADEP_mvt").agg(
        n=pl.col("x").is_not_null().sum(),
        k0=(pl.col("x")==0).mean(), k6=(pl.col("x").abs()<=6).mean(), k60=(pl.col("x").abs()<=60).mean(),
        # neighbor density: |x| in (60,240] per 60s window  -> expected count in 13-s window ~ density*13
        nb=((pl.col("x").abs()>60)&(pl.col("x").abs()<=240)).mean()/6,  # per minute
        nb_m1=((pl.col("x")>-90)&(pl.col("x")<=-30)).mean(),
        nb_p1=((pl.col("x")>30)&(pl.col("x")<=90)).mean(),
        b0=(pl.col("x").abs()<=30).mean(),
    ).with_columns(src=pl.lit(c)).sort("ADEP_mvt")
    out.append(t)
print(pl.concat(out))

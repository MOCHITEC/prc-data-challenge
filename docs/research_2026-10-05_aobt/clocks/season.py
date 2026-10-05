import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260)
c=pl.read_csv("out/cell_signature_by_month.csv")
print(c.pivot(on="month",index="APT",values="inNM").sort("APT"))
m=pl.read_parquet("out_cell.parquet").with_columns(win=pl.col("month").is_in([1,2,3,11,12]))
# winter vs summer; within corr excluding rows with long airport taxi (y_c > 300)
r=m.group_by("APT","win").agg(n=pl.len(),inNM=pl.col("inNM").mean(),corr=pl.corr("dmt_c","y_c"),
   
   share_inNM_when_ylong=pl.col("inNM").filter(pl.col("y_c")>600).mean(),
   share_ylong=(pl.col("y_c")>600).mean()).sort("APT","win")
print(r); r.write_csv("out/season_split.csv")
# hour-of-day x season corr for LSZH, EDDF
for a in ["LSZH","EDDF","EHAM"]:
    x=m.filter(pl.col("APT")==a).group_by("win","hour").agg(corr=pl.corr("dmt_c","y_c"),inNM=pl.col("inNM").mean()).sort("win","hour")
    print(a); print(x.pivot(on="hour",index="win",values="inNM"))

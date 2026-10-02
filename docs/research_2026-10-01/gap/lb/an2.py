import polars as pl, datetime as dt
df=pl.read_parquet("lb_raw.parquet").sort("processedAt")
# all improving steps, all teams
rows=[]
for t,s in df.group_by("teamName"):
    b=None
    for r in s.sort("processedAt").iter_rows(named=True):
        if b is not None and r["score"]<b: rows.append(dict(team=t[0],date=r["processedAt"],before=b,after=r["score"],step=b-r["score"]))
        b=r["score"] if b is None else min(b,r["score"])
st=pl.DataFrame(rows).with_columns(pl.col("date").dt.date().alias("day"))
st.write_parquet("improving_steps_all.parquet")
big=st.filter((pl.col("before")<300)&(pl.col("step")>=10))
pl.Config.set_tbl_rows(80);pl.Config.set_tbl_width_chars(200)
print("big steps (>=10 s, from <300) per day")
print(big.group_by("day").agg(pl.len().alias("n"),pl.col("team").str.join(",").alias("teams")).sort("day"))
print(big.filter(pl.col("day")>=dt.date(2026,9,28)).sort("date").select("team","date","before","after","step"))
# subs per day
print("all subs per day:",df.group_by(pl.col("processedAt").dt.date()).len().sort("processedAt").tail(12))
mx=df.group_by("teamName",pl.col("processedAt").dt.date().alias("d")).len().select(pl.col("len").max());print("max subs/team/day",mx)
# rank-k score over time (daily end)
days=[dt.datetime(2026,9,d,23,59,59,tzinfo=dt.timezone.utc) for d in range(10,31)]+[dt.datetime(2026,10,d,23,59,59,tzinfo=dt.timezone.utc) for d in (1,)]+[df["processedAt"].max()]
out=[]
for T in days:
    b=df.filter(pl.col("processedAt")<=T).group_by("teamName").agg(pl.col("score").min()).sort("score")["score"].to_list()
    us=df.filter((pl.col("processedAt")<=T)&(pl.col("teamName")=="resilient-hedgehog"))["score"].min()
    rk=sum(1 for x in b if us is not None and x<us)+1 if us else None
    out.append(dict(t=T,n=len(b),r1=b[0],r3=b[2],r5=b[4],r10=b[9],r20=b[19],r30=b[29],us=us,us_rank=rk))
o=pl.DataFrame(out);print(o);o.write_parquet("rank_curves_daily.parquet")

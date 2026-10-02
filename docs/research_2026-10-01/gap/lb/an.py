import polars as pl
df=pl.read_parquet("lb_raw.parquet").sort("processedAt")
best=df.group_by("teamName").agg(pl.col("score").min().alias("best"),pl.len().alias("n"),pl.col("processedAt").min().alias("first"),pl.col("processedAt").max().alias("last")).sort("best")
best=best.with_row_index("rank",1)
pl.Config.set_tbl_rows(60);pl.Config.set_tbl_width_chars(250);pl.Config.set_fmt_str_lengths(40)
print(best.head(35))
print(best.filter(pl.col("teamName")=="resilient-hedgehog"))
top=best.head(15)["teamName"].to_list()+["resilient-hedgehog"]
rows=[]
for t in top:
    s=df.filter(pl.col("teamName")==t).sort("processedAt")
    b=None
    for r in s.iter_rows(named=True):
        if b is None: rows.append(dict(team=t,date=r["processedAt"],before=None,after=r["score"],step=None,file=r["filename"])); b=r["score"]; continue
        if r["score"]<b:
            rows.append(dict(team=t,date=r["processedAt"],before=b,after=r["score"],step=b-r["score"],file=r["filename"])); b=r["score"]
steps=pl.DataFrame(rows)
steps.write_parquet("improving_steps_top15_us.parquet")
for t in top:
    s=steps.filter(pl.col("team")==t)
    print("\n==",t, "n_sub",df.filter(pl.col("teamName")==t).height)
    print(s.filter((pl.col("step")>=2)|pl.col("step").is_null()).select("date","before","after","step","file"))
    print("  small steps(<2s):",s.filter(pl.col("step")<2).height,"sum",round(s.filter(pl.col("step")<2)["step"].sum(),2))

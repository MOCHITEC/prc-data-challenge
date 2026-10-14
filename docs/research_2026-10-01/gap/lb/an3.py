import polars as pl, math, datetime as dt
N=344841
st=pl.read_parquet("improving_steps_top15_us.parquet").filter(pl.col("step")>=2)
st=st.with_columns(dMSE=(pl.col("before")**2-pl.col("after")**2)).with_columns(eq_row_err=(pl.col("dMSE")*N).sqrt(), frac_86400=pl.col("dMSE")/(86400**2/N))
st=st.filter(pl.col("before")<320)
pl.Config.set_tbl_rows(200);pl.Config.set_tbl_width_chars(220);pl.Config.set_fmt_str_lengths(30)
print(st.select("team",pl.col("date").dt.strftime("%m-%d %H:%M"),pl.col("before").round(1),pl.col("after").round(1),pl.col("step").round(1),pl.col("dMSE").round(0),pl.col("eq_row_err").round(0),pl.col("frac_86400").round(2)))
st.write_csv("steps_ge2_top15_us_with_mse.csv")
g=251.7333**2-219.6399**2;print("gap MSE",g,"rows@86400",g/(86400**2/N),"rows@3600",g/(3600**2/N),"rows@1800",g/(1800**2/N))
for f,lab in ((5290/N,"unmatched 5290"),(1-5290/N,"matched")):
    need=g/f;print(lab,"lane MSE drop needed",round(need))
for r in (180,200,210,220,230):
    print("matched RMSE",r,"->",round(math.sqrt(r*r-g/(1-5290/N)),1))

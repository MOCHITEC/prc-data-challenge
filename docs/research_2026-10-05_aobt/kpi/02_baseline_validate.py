# Step 1-2: replicate PRU ATXOT on 2025 departures and validate against published PRU monthly values.
import polars as pl, numpy as np
from common import *
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(130)
df = load()
pru = pl.read_csv("pru_published_2024_2025.csv").filter(pl.col("YEAR")==2025).select(
    pl.col("APT_ICAO").alias("ADEP_mvt"), pl.col("MONTH_NUM").cast(pl.Int32).alias("lmon"), pl.col("TF").alias("pru_TF"),
    pl.col("VALID_FL").alias("pru_valid"), pl.col("TOTAL_REF_NB_FL").alias("pru_nref"),
    (pl.col("TOTAL_ADD_TIME_MIN")/pl.col("TOTAL_REF_NB_FL")).alias("pru_add"), (pl.col("TOTAL_REF_TIME_MIN")/pl.col("TOTAL_REF_NB_FL")).alias("pru_ref"))
tf = df.group_by(["ADEP_mvt","lmon"]).agg(our_TF=pl.len())
variants = {}
f0 = base_filter(df); f1 = base_filter(df, deice=True)
variants["FY"] = kpi(f0, window="FY")
variants["ROLL"] = kpi(f0, window="ROLL")
variants["FY_deice"] = kpi(f1, window="FY")
variants["ROLL_deice"] = kpi(f1, window="ROLL")
rows=[]; allm=[]
for v,res in variants.items():
    m = res.join(pru, on=["ADEP_mvt","lmon"]).join(tf, on=["ADEP_mvt","lmon"])
    allm.append(m.with_columns(variant=pl.lit(v)))
    for a in APTS:
        s = m.filter(pl.col("ADEP_mvt")==a).sort("lmon")
        x, yv = s["add"].to_numpy(), s["pru_add"].to_numpy()
        sw = s.filter(pl.col("lmon").is_between(4,10))
        rows.append(dict(variant=v, apt=a, r=np.corrcoef(x,yv)[0,1], mad=np.mean(np.abs(x-yv)), bias=np.mean(x-yv),
            mad_AprOct=float((sw["add"]-sw["pru_add"]).abs().mean()), bias_AprOct=float((sw["add"]-sw["pru_add"]).mean()),
            ref_bias=float((s["ref"]-s["pru_ref"]).mean()), our_add=np.mean(x), pru_add=np.mean(yv),
            nref_ratio=s["n_ref"].sum()/s["pru_nref"].sum(), TF_ratio=s["our_TF"].sum()/s["pru_TF"].sum(),
            pru_valid_share=s["pru_valid"].sum()/s["pru_TF"].sum(), our_valid_share=s["n"].sum()/s["our_TF"].sum()))
val = pl.DataFrame(rows)
pl.concat(allm).write_csv("out/replication_monthly.csv")
val.write_csv("out/validation_per_airport.csv")
print(val.with_columns(pl.col(pl.Float64).round(3)))
# pooled
for v in variants:
    m = pl.concat(allm).filter(pl.col("variant")==v)
    print(v, "pooled r=%.3f MAD=%.3f bias=%.3f"%(np.corrcoef(m["add"],m["pru_add"])[0,1], float((m["add"]-m["pru_add"]).abs().mean()), float((m["add"]-m["pru_add"]).mean())))
    an = annual(m.select(variants[v].columns)).join(m.group_by("ADEP_mvt").agg(pru=(pl.col("pru_add")*pl.col("pru_nref")).sum()/pl.col("pru_nref").sum()), on="ADEP_mvt")
    an = an.with_columns(rk_our=pl.col("add").rank(descending=True), rk_pru=pl.col("pru").rank(descending=True))
    print(an.with_columns(pl.col(pl.Float64).round(2)))

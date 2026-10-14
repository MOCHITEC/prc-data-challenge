# Step 3: explicit suspect-AOBT flags; counts per airport x month; excess-mass (copy) estimator; precision attribute.
import polars as pl, numpy as np
from common import *
pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_rows(200)
df = load()
FLAGS = ["F_A3_5","F_A3_60","F_SCHED_6","F_SCHED_60","F_EOBT1_6","F_EOBT1_60","F_LOBT_6","F_IOBT_6","F_PLAN_6","F_PLAN_60",
         "F_SCHEDFILL","F_PLANFILL","F_PLAN_NMEARLY","F_ROLL","F_NEG","F_SHORT","F_LONG"]
agg = [pl.len().alias("n")] + [pl.col(f).sum().alias(f) for f in FLAGS] + [
       (pl.col("BLOCK_TIME_UTC_mvt").dt.second().is_in([0,1,2,3,4,5,6,54,55,56,57,58,59])).mean().alias("blk_ongrid6_share"),
       pl.col("heli").sum().alias("heli"), pl.col("unmatched").sum().alias("unmatched")]
pm = df.group_by(["ADEP_mvt","lmon"]).agg(agg).sort(["ADEP_mvt","lmon"])
pm.write_csv("out/flag_counts_airport_month.csv")
pa = df.group_by("ADEP_mvt").agg(agg).sort("ADEP_mvt")
pa_sh = pa.select(["ADEP_mvt","n"]+[(pl.col(f)/pl.col("n")*100).round(2).alias(f) for f in FLAGS]+["blk_ongrid6_share","heli","unmatched"])
pa.write_csv("out/flag_counts_airport.csv"); pa_sh.write_csv("out/flag_share_pct_airport.csv")
print(pa_sh)
tot = df.select([pl.len().alias("n")]+[pl.col(f).mean().mul(100).round(2).alias(f) for f in FLAGS]); print(tot)

# Excess-mass estimator: share of BLOCK-X in the 0-minute bin (|d|<=30 s) minus mean of the two adjacent 1-min bins.
# A pure copy rule creates a spike at 0; natural agreement (on-time push) produces a smooth density across bins.
rows=[]
for k,c in [("AOBT_3","dA3"),("SCHED","dSCHED"),("EOBT1","dEOBT1"),("LOBT","dLOBT"),("IOBT","dIOBT")]:
    t = df.filter(pl.col(c).is_not_null()).group_by("ADEP_mvt").agg(
        b0=(pl.col(c).abs()<=30).mean(), bm=((pl.col(c)>-90)&(pl.col(c)<-30)).mean(), bp=((pl.col(c)>30)&(pl.col(c)<90)).mean(),
        k6=(pl.col(c).abs()<=6).mean(), k60=(pl.col(c).abs()<=60).mean())
    rows.append(t.with_columns(src=pl.lit(k), excess=pl.col("b0")-(pl.col("bm")+pl.col("bp"))/2))
ex = pl.concat(rows).with_columns(pl.col(pl.Float64).mul(100).round(2)).sort(["src","ADEP_mvt"])
ex.write_csv("out/copy_excess_mass_pct.csv"); print(ex)
# excess per month for SCHED at LIRF / LEBL
print(df.filter(pl.col("ADEP_mvt").is_in(["LIRF","LEBL"])).group_by(["ADEP_mvt","lmon"]).agg(
    b0=(pl.col("dSCHED").abs()<=30).mean(), nb=(((pl.col("dSCHED")>-90)&(pl.col("dSCHED")<-30))|((pl.col("dSCHED")>30)&(pl.col("dSCHED")<90))).mean()/2,
    fill=pl.col("F_SCHEDFILL").mean(), unm=pl.col("unmatched").mean()).with_columns(exc=pl.col("b0")-pl.col("nb")).sort(["ADEP_mvt","lmon"]))
# direction: among filtered PRU population, taxi-out of flagged vs unflagged and vs NM-based taxi (MVT-AOBT_3)
f = base_filter(df).with_columns(yA3=(pl.col("MVT_TIME_UTC_mvt")-pl.col("AOBT_3_flt")).dt.total_seconds())
rows=[]
for fl in ["F_A3_5","F_A3_60","F_SCHED_6","F_SCHEDFILL","F_PLAN_6","F_PLANFILL","F_PLAN_NMEARLY","F_SHORT"]:
    t = f.group_by(["ADEP_mvt",fl]).agg(n=pl.len(), y=pl.col("TAXITIME_SEC_mvt").mean()/60, yA3=pl.col("yA3").mean()/60,
          d_vs_A3=(pl.col("TAXITIME_SEC_mvt")-pl.col("yA3")).median()/60, dep_delay=(pl.col("dSCHED")).median()/60)
    rows.append(t.rename({fl:"flag"}).with_columns(fl=pl.lit(fl)))
dirn = pl.concat(rows).sort(["fl","ADEP_mvt","flag"]).with_columns(pl.col(pl.Float64).round(2))
dirn.write_csv("out/flag_direction.csv"); print(dirn)

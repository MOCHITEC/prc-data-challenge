import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, numpy as np
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
d=pl.read_parquet("slim.parquet").filter((pl.col("PHASE_mvt")=="DEP")&pl.col("d_mt").is_not_null())
d=d.with_columns(mm=(pl.col("d_mt")/60).round().cast(pl.Int32))
for apt in sorted(d["APT"].unique()):
    x=d.filter(pl.col("APT")==apt)
    c=x.group_by("mm").len().sort("mm")
    arr=dict(zip(c["mm"].to_list(),c["len"].to_list()))
    n=x.height
    # excess mass: count - median of neighbours k-3..k+3 excluding k
    ex={k:v-np.median([arr.get(k+j,0) for j in (-3,-2,-1,1,2,3)]) for k,v in arr.items()}
    top=sorted(ex.items(),key=lambda t:-t[1])[:5]
    print(apt,n,"top excess (min,excess share,raw share):",[(k,round(e/n,4),round(arr[k]/n,4)) for k,e in top])
    rw=x.group_by("RUNWAY_mvt").agg(n=pl.len(),mode=pl.col("mm").mode().first()).sort("n",descending=True).head(6)
    for r in rw.iter_rows(named=True):
        xr=x.filter(pl.col("RUNWAY_mvt")==r["RUNWAY_mvt"])
        cc=xr.group_by("mm").len().sort("len",descending=True).head(4)
        print("   rw",r["RUNWAY_mvt"],r["n"],[(a,round(b/xr.height,3)) for a,b in zip(cc["mm"],cc["len"])])

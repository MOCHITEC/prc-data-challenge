import polars as pl, numpy as np
d=pl.read_parquet("joined.parquet").select("MVT_ID_mvt","ADEP_mvt","err","pred","TAXITIME_SEC_mvt","band","MVT_TIME_UTC_mvt")
d=d.with_columns(mo=pl.col("MVT_TIME_UTC_mvt").dt.month())
print(d.group_by("mo","band").agg(n=pl.len(),bias=pl.col("err").mean()).sort("mo","band").pivot(on="mo",index="band",values="bias"))
# cross-month band-wise linear recalibration y = a + b*pred within (band) and (band x airport); fit on other month, genuine rows
out={}
for key in (["band"],["band","ADEP_mvt"]):
    e1=np.zeros(d.height); 
    for m in (1,7):
        tr=d.filter(pl.col("mo")!=m)
        co={}
        for k,g in tr.group_by(key):
            x=g["pred"].to_numpy(); y=g["TAXITIME_SEC_mvt"].to_numpy()
            if len(x)<300: continue
            b,a=np.polyfit(x,y,1); co[k]=(a,b)
        te_idx=np.where((d["mo"]==m).to_numpy())[0]
        te=d[te_idx]
        ks=list(zip(*[te[c].to_list() for c in key]))
        p=te["pred"].to_numpy()
        newp=np.array([co[k][0]+co[k][1]*pi if k in co else pi for k,pi in zip(ks,p)])
        e1[te_idx]=te["TAXITIME_SEC_mvt"].to_numpy()-newp
    e0=d["err"].to_numpy()
    dl=d["band"].is_in(["(1800, 3600]","(3600, 10800]"]).to_numpy() if d["band"].dtype!=pl.Enum else d["band"].cast(pl.Utf8).is_in(["30..60m","1..3h"]).to_numpy()
    r=lambda e,mk:np.sqrt((e[mk]**2).mean())
    allm=np.ones(len(e0),bool)
    print(key,"genuine all %.2f->%.2f"%(r(e0,allm),r(e1,allm)),"delayed %.2f->%.2f"%(r(e0,dl),r(e1,dl)))
    # only bias-removed (global, other month) comparison
e2=np.zeros(d.height)
for m in (1,7):
    mb=d.filter(pl.col("mo")!=m)["err"].mean(); idx=(d["mo"]==m).to_numpy(); e2[idx]=d["err"].to_numpy()[idx]-mb
print("global other-month bias shift: %.2f -> %.2f"%(np.sqrt((d['err']**2).mean()),np.sqrt((e2**2).mean())))

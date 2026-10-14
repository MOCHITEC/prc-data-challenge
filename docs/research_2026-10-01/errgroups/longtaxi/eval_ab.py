import numpy as np, polars as pl, sys, os
E="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
o=pl.read_parquet(E+"oof_matched_janjul.parquet").select("MVT_ID_mvt","ADEP_mvt","TAXITIME_SEC_mvt","genuine","is_rollover")
tx=pl.read_parquet(E+"longtaxi/tx_feats.parquet").select("MVT_ID_mvt","lday","tx_day_late3h","tx_day_tir","tx_mma_c60_ratio")
fl=pl.read_csv(E+"longtaxi/flagged_days_2026.csv")  # not used for 2025
d=o.join(tx,on="MVT_ID_mvt")
vs=sys.argv[1:] or ["base","tx_all","tx_row","tx_day"]; vs=[v for v in vs if os.path.exists(E+f"longtaxi/pred_{v}.parquet")]
for v in vs: d=d.join(pl.read_parquet(E+f"longtaxi/pred_{v}.parquet").rename({"pred":v}),on="MVT_ID_mvt")
y=d["TAXITIME_SEC_mvt"].to_numpy().astype(float); gen=d["genuine"].to_numpy()
key=(d["ADEP_mvt"]+"_"+d["lday"].cast(pl.Utf8)).to_numpy(); u,inv=np.unique(key,return_inverse=True)
rng=np.random.default_rng(1); W=[np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u)) for _ in range(500)]
def comp(a,b,mk):
    sa=np.bincount(inv[mk],(y-a)[mk]**2,len(u)); sb=np.bincount(inv[mk],(y-b)[mk]**2,len(u)); n=np.bincount(inv[mk],None,len(u))
    pt=np.sqrt(sb.sum()/n.sum())-np.sqrt(sa.sum()/n.sum())
    bs=[np.sqrt((w*sb).sum()/(w*n).sum())-np.sqrt((w*sa).sum()/(w*n).sum()) for w in W]
    return round(pt,3), np.round(np.percentile(bs,[2.5,97.5]),3)
dis=((d["tx_day_late3h"]>0.02)|(d["tx_day_tir"]>150)).to_numpy()
masks={"all_matched":np.ones(len(y),bool),"all_excl_rollover":~d["is_rollover"].to_numpy(),"genuine":gen,"gen_long>=30m":gen&(y>=1800),"gen_short":gen&(y<1800),
       "gen_disrupted_day":gen&dis,"gen_normal_day":gen&~dis,"gen_hi_cong(c60>1.3)":gen&(d["tx_mma_c60_ratio"]>1.3).fill_null(False).to_numpy()}
b=d[vs[0]].to_numpy()
for mn,mk in masks.items():
    print(f"{mn:24s} n={mk.sum():7d} base={np.sqrt(np.mean((y-b)[mk]**2)):.2f}", {v:comp(b,d[v].to_numpy(),mk) for v in vs if v!=vs[0]})
for apt in sorted(set(d["ADEP_mvt"].to_list())):
    mk=gen&(d["ADEP_mvt"]==apt).to_numpy()
    print(apt, f"{np.sqrt(np.mean((y-b)[mk]**2)):.1f}", {v:round(np.sqrt(np.mean((y-d[v].to_numpy())[mk]**2))-np.sqrt(np.mean((y-b)[mk]**2)),2) for v in vs if v!=vs[0]})

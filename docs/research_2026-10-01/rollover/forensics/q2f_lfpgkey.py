# COMPLIANCE-SENSITIVE MEASUREMENT: LFPG MVT_ID local ordering key among SCHED/BLOCK/MVT/LOBT/IOBT/EOBT_1/AOBT_3
import polars as pl
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP',pl.col('ADEP_mvt')=='LFPG').sort('MVT_ID_mvt')
K={'S':'SCHED_TIME_UTC_mvt','B':'BLOCK_TIME_UTC_mvt','M':'MVT_TIME_UTC_mvt','L':'LOBT_flt','I':'IOBT_flt','E':'EOBT_1_flt','A':'AOBT_3_flt'}
df=df.with_columns([(pl.col(v).dt.epoch('s').diff()).alias('d'+k) for k,v in K.items()]+[pl.col('MVT_ID_mvt').diff().alias('gid')])
for rk in [False,True]:
    d=df.filter(pl.col('gid')==1, pl.col('dM').abs()<6*3600, (pl.col('file')=='ranking.parquet')==rk)
    print('ranking' if rk else 'training','local id-adjacent pairs',len(d))
    print({k:(round(float((d.filter(pl.col('d'+k).is_not_null())['d'+k]>=0).mean()),4), d['d'+k].is_not_null().sum()) for k in K})
    # strict violations count per key
    print('pairs with key decreasing:',{k:int((d['d'+k]<0).sum()) for k in K})
    # unmatched rows (no LOBT) only
    u=d.filter(pl.col('dL').is_null())
    print('pairs touching unmatched rows',len(u),{k:(round(float((u.filter(pl.col('d'+k).is_not_null())['d'+k]>=0).mean()),4) if u['d'+k].is_not_null().sum() else None, int(u['d'+k].is_not_null().sum())) for k in K})

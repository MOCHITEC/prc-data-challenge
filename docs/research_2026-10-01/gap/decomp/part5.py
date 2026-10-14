import polars as pl, numpy as np
R='/home/user/prc-data-challenge/'
fr=pl.read_parquet(R+'data/work_flseq/feat_rank.parquet',columns=['MVT_ID_mvt','ADEP_mvt','mvt_minus_aobt3','mvt_minus_sched','mvt_minus_lobt','mvt_minus_eobt1','month'])
v32=pl.read_parquet(R+'outputs/lb/resilient-hedgehog_v32.parquet').rename({'TAXITIME_SEC_mvt':'p32'})
v23=pl.read_parquet(R+'outputs/lb/resilient-hedgehog_v23.parquet').rename({'TAXITIME_SEC_mvt':'p23'})
d=fr.join(v32,on='MVT_ID_mvt').join(v23,on='MVT_ID_mvt'); N=d.height; print('N',N)
d=d.with_columns(m=pl.col('mvt_minus_aobt3').is_not_null())
print(d.group_by('m').agg(pl.len(), pl.col('p32').mean(), (pl.col('p32')-pl.col('mvt_minus_aobt3')).std().alias('g_sd')))
u=d.filter(~pl.col('m'))
print('unmatched by airport', u.group_by('ADEP_mvt').len().sort('ADEP_mvt').to_dicts())
for lo,hi in [(4,8),(8,12),(12,16),(16,30),(30,1e9)]:
    s=d.filter((pl.col('mvt_minus_sched')>=lo*3600)&(pl.col('mvt_minus_sched')<hi*3600))
    su=s.filter(~pl.col('m'))
    print(f'ms {lo}-{hi}h: n={s.height} (unm {su.height}, LIRF unm {su.filter(pl.col("ADEP_mvt")=="LIRF").height}); p32 median {s["p32"].median()}, unm p32 med {su["p32"].median()}, unm ms med {su["mvt_minus_sched"].median()}')
# risk proxy: rows where p32 differs from ms by > 3h and ms>4h
r=d.filter((pl.col('mvt_minus_sched')>4*3600)).with_columns(dev=(pl.col('mvt_minus_sched')-pl.col('p32')))
print('rows ms>4h:',r.height,' sum (ms-p32)^2 /N =', (r['dev']**2).sum()/N)
print(r.sort('dev',descending=True).select('ADEP_mvt','m','mvt_minus_sched','mvt_minus_aobt3','mvt_minus_lobt','p32','p23').head(25))
# max predictions
print(d.sort('p32',descending=True).select('ADEP_mvt','m','mvt_minus_sched','mvt_minus_aobt3','mvt_minus_lobt','p32').head(15))

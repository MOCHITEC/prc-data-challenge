import polars as pl
R='/home/user/prc-data-challenge/'
fr=pl.read_parquet(R+'data/work_flseq/feat_rank.parquet',columns=['MVT_ID_mvt','ADEP_mvt','mvt_minus_aobt3','mvt_minus_sched','mvt_minus_lobt','mvt_minus_eobt1'])
v32=pl.read_parquet(R+'outputs/lb/resilient-hedgehog_v32.parquet').rename({'TAXITIME_SEC_mvt':'p32'})
d=fr.join(v32,on='MVT_ID_mvt'); N=d.height
u=d.filter(pl.col('mvt_minus_aobt3').is_null()&(pl.col('mvt_minus_sched')>8*3600)).sort('mvt_minus_sched')
pl.Config.set_tbl_rows(80)
print(u.select('ADEP_mvt','mvt_minus_sched','mvt_minus_lobt','mvt_minus_eobt1','p32'))
# swing if wrong branch: if p32 ~ ms, alt = ~1500 (genuine) ; if p32 small, alt = ms
u=u.with_columns(alt=pl.when((pl.col('p32')-pl.col('mvt_minus_sched')).abs()<5000).then(pl.lit(1500.0)).otherwise(pl.col('mvt_minus_sched')))
u=u.with_columns(sw=((pl.col('alt')-pl.col('p32'))**2/N))
print('rows',u.height,'swing per row (s2 of LB MSE) median',u['sw'].median(),' sum',u['sw'].sum())
print('predictions > 7200 s overall:', d.filter(pl.col('p32')>7200).height, ' unmatched:', d.filter(pl.col('mvt_minus_aobt3').is_null()&(pl.col('p32')>7200)).height)

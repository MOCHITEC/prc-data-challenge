import polars as pl, numpy as np
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'
K='MVT_ID_mvt'
W=pl.read_parquet('data/ref/metar_hourly.parquet').select(pl.col('station').alias('ADEP_mvt'), pl.col('hour_utc').dt.replace_time_zone('UTC').alias('h'), 'deicing_gate','frozen_wx','temp_c')
F=(pl.scan_parquet('data/raw/final_ranking.parquet').filter(pl.col('PHASE_mvt')=='DEP')
   .select(K,'ADEP_mvt','MVT_TIME_UTC_mvt',pl.col('MVT_TIME_UTC_mvt').dt.month().alias('m'),
           (pl.col('MVT_TIME_UTC_mvt')-pl.col('AOBT_3_flt')).dt.total_seconds().alias('prox')).collect())
F=F.with_columns(h=pl.col('MVT_TIME_UTC_mvt').dt.truncate('1h')).join(W,on=['ADEP_mvt','h'],how='left')
F=F.with_columns(de=pl.col('deicing_gate').fill_null(0)>0)
# sanity vs old-ranking features for Jan/Jul (same flights, MVT re-dithered by <=12 s)
rk=pl.scan_parquet('data/work_flseq/feat_rank.parquet').filter(pl.col('PHASE_mvt')=='DEP').select(K,pl.col('deicing_gate').alias('dg_old')).collect()
c=F.join(rk,on=K,how='inner').with_columns(old=pl.col('dg_old').cast(pl.Float64).fill_null(0)>0)
print('Jan/Jul agreement with old features: rows', c.height, 'agree share', float((c['de']==c['old']).mean()), 'old de', int(c['old'].sum()), 'new de', int(c['de'].sum()))
print('de-icing rows by airport x month (final set):')
print(F.filter(pl.col('de')).group_by('ADEP_mvt','m').agg(n=pl.len(), days=pl.col('MVT_TIME_UTC_mvt').dt.date().n_unique()).sort('ADEP_mvt','m'))
v6=pl.read_parquet(f'{S}/v6/final_v6_all5.parquet').rename({'TAXITIME_SEC_mvt':'v6'})
x=F.join(v6,on=K)
g=x.filter(pl.col('m').is_in([1,2])&pl.col('ADEP_mvt').is_in(['EDDF','EDDM','EHAM','LFPG','LSZH','LTFM','EGLL','LEMD'])).group_by('ADEP_mvt','m','de').agg(n=pl.len(), v6=pl.col('v6').mean().round(0), prox_med=pl.col('prox').median(), v6_minus_prox_med=(pl.col('v6')-pl.col('prox')).median()).sort('ADEP_mvt','m','de')
print(g)
F.select(K,'ADEP_mvt','m','de').write_parquet(f'{S}/deice/final_deice_flags.parquet')

import polars as pl, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'
L=f'{S}/adsb/learn/'; K='MVT_ID_mvt'
cols=[K,'ADEP_mvt','t_mvt','t_move','t_push','ground_fix']
F25=pl.concat([pl.read_parquet(L+f, columns=cols) for f in ('feat_2025.parquet','feat_2025m8.parquet','feat_2025nd.parquet')]).unique(K)
A25=pl.concat([pl.read_parquet(L+f) for f in ('appear_fill_2025.parquet','appear_fill_2025m8.parquet','appear_fill_2025nd.parquet')]).unique(K).select(K,'own_ok','t_leave_own')
tr=pl.scan_parquet('data/work_flseq/feat_train.parquet').filter(pl.col('PHASE_mvt')=='DEP').select(K,'deicing_gate').collect()
d=pl.read_parquet(f'{S}/ideas/ht/d25.parquet').select(K,'month','y','final_12','prox').join(tr,on=K).join(F25,on=K).join(A25,on=K,how='left')
d=d.with_columns(de=pl.col('deicing_gate').cast(pl.Float64).fill_null(0)>0, a_leave=(pl.col('t_mvt')-pl.col('t_leave_own')), a_move=(pl.col('t_mvt')-pl.col('t_move')))
APS=['EDDF','EDDM','EHAM','LFPG','LSZH']
x=d.filter(pl.col('ADEP_mvt').is_in(APS)&(pl.col('own_ok').cast(pl.Float64).fill_null(0)>0)&pl.col('a_leave').is_not_null())
g=x.group_by('ADEP_mvt','de').agg(n=pl.len(), y_leave_med=(pl.col('y')-pl.col('a_leave')).median(), y_leave_tmean=(pl.col('y')-pl.col('a_leave')).filter((pl.col('y')-pl.col('a_leave')).abs()<1800).mean().round(0),
     f_leave_med=(pl.col('final_12')-pl.col('a_leave')).median(), r_mean=(pl.col('y')-pl.col('final_12')).mean().round(1)).sort('ADEP_mvt','de')
print('2025 rows with own-stand leave time (own_ok): label minus (MVT - leave-own-stand)'); print(g)
# 2026 Jan de-icing rows
F26=pl.read_parquet('handoff/2026-10-11/adsb/learn/feat_2026.parquet', columns=cols)
A26=pl.read_parquet('handoff/2026-10-11/adsb/learn/appear_fill_2026.parquet').select(K,'own_ok','t_leave_own')
fl=pl.read_parquet(f'{S}/deice/final_deice_flags.parquet')
v48=pl.read_parquet(f'{S}/ideas/cal/v48.parquet').rename({'TAXITIME_SEC_mvt':'v48'})
z=fl.filter(pl.col('m')==1).join(F26,on=K).join(A26,on=K,how='left').join(v48,on=K)
z=z.with_columns(a_leave=(pl.col('t_mvt')-pl.col('t_leave_own')))
zz=z.filter(pl.col('ADEP_mvt').is_in(APS)&(pl.col('own_ok').cast(pl.Float64).fill_null(0)>0)&pl.col('a_leave').is_not_null())
h=zz.group_by('ADEP_mvt','de').agg(n=pl.len(), v_leave_med=(pl.col('v48')-pl.col('a_leave')).median(), v_leave_tmean=(pl.col('v48')-pl.col('a_leave')).filter((pl.col('v48')-pl.col('a_leave')).abs()<1800).mean().round(0)).sort('ADEP_mvt','de')
print('2026 Jan (v48) on rows with own-stand leave time:'); print(h)
cov=z.filter(pl.col('ADEP_mvt').is_in(APS)&pl.col('de')).group_by('ADEP_mvt').agg(n=pl.len(), with_leave=((pl.col('own_ok').cast(pl.Float64).fill_null(0)>0)&pl.col('a_leave').is_not_null()).sum())
print('coverage of Jan 2026 de-icing rows:', cov.sort('ADEP_mvt').to_dicts())

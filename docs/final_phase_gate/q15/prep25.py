import polars as pl, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'
q=pl.read_parquet(f'{S}/ideas/ht/copyq_lomo_2025.parquet').select('MVT_ID_mvt','q','ms_feat')
d=(pl.scan_parquet(f'{S}/ideas/ht/d25.parquet').select('MVT_ID_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','y','matched','ms','mlobt','prox','month','b','final_12','P12')
   .filter(pl.col('matched')&(pl.col('ADEP_mvt')=='LIRF')).collect().join(q,on='MVT_ID_mvt',how='inner'))
d=d.with_columns(imp=((pl.col('ms')-pl.col('mlobt')).abs()>3606), day=pl.col('MVT_TIME_UTC_mvt').dt.date(), copy=((pl.col('y')-pl.col('ms')).abs()<=12))
print('rows',d.height,'imp',d['imp'].sum(),'copy rate',d['copy'].mean(), 'ms_feat==ms',(d['ms_feat']==d['ms']).mean())
print(d.group_by('month').agg(n=pl.len(), copy=pl.col('copy').mean(), qmean=pl.col('q').mean(), rmse_f12=((pl.col('final_12')-pl.col('y'))**2).mean().sqrt(), rmse_b=((pl.col('b')-pl.col('y'))**2).mean().sqrt()).sort('month'))
# calibration table
d=d.with_columns(qb=pl.col('q').cut([0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]))
print(d.filter(~pl.col('imp')).group_by('qb').agg(n=pl.len(), qmean=pl.col('q').mean(), copy=pl.col('copy').mean(), near30=((pl.col('y')-pl.col('ms')).abs()<=30).mean(), near60=((pl.col('y')-pl.col('ms')).abs()<=60).mean(), mean_ms_minus_p=(pl.col('ms')-pl.col('final_12')).mean(), rms_ms_minus_p=((pl.col('ms')-pl.col('final_12'))**2).mean().sqrt()).sort('qb'))
d.select('MVT_ID_mvt','y','ms','mlobt','prox','month','day','b','final_12','P12','q','imp','copy').write_parquet('lirf25.parquet')

import polars as pl, numpy as np
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'
# 1) base used by the integrator's copy_lomo vs d25.final_12
try:
    R=pl.read_parquet(f'{S}/adsb/learn/cv_months12_rows_a2.parquet').select('MVT_ID_mvt','y','b','final_12')
    d=pl.read_parquet('lirf25.parquet').join(R.rename({'y':'y2','b':'b2','final_12':'f2'}),on='MVT_ID_mvt',how='inner')
    print('cv_months12_rows_a2 join',d.height,'y==y2',(d['y']==d['y2']).mean(),'rms(final_12-f2)',float(((d['final_12']-d['f2'])**2).mean()**.5),'rms(b-b2)',float(((d['b']-d['b2'])**2).mean()**.5))
    print('rms(f2-ms) vs rms(final_12-ms) on q>0.8 rows:', float(((d.filter(pl.col('q')>0.8)['f2']-d.filter(pl.col('q')>0.8)['ms'])**2).mean()**.5), float(((d.filter(pl.col('q')>0.8)['final_12']-d.filter(pl.col('q')>0.8)['ms'])**2).mean()**.5))
except Exception as e: print('cv rows check failed', e)
# 2) deliverable sanity: rows26_B3 within LOBT window, disjoint from gate rows, delta column
f=pl.read_parquet('lirf26.parquet'); r=pl.read_parquet('rows26_B3.parquet').join(f.select('MVT_ID_mvt','mlobt','imp','qu','ms','v4'),on='MVT_ID_mvt')
G=pl.read_parquet(f'{S}/ideas/ht/lirf_copy_gate_rows.parquet')
print('B3 rows',r.height,'imp rows among them',r['imp'].sum(),'overlap with gate rows',r.join(G,on='MVT_ID_mvt').height,'new<0',(r['new']<0).sum(),'outside window',((r['new']<(r['mlobt']-3606).ceil())|(r['new']>(r['mlobt']+3606).ceil())).sum())
r=r.with_columns(delta=(pl.col('new')-pl.col('old')).cast(pl.Int32))
print(r.group_by('month').agg(n=pl.len(), mean_delta=pl.col('delta').mean(), rms_delta=(pl.col('delta').cast(pl.Float64)**2).mean().sqrt(), max_abs=pl.col('delta').abs().max(), toward_ms=((pl.col('new')-pl.col('ms')).abs()<(pl.col('old')-pl.col('ms')).abs()).mean()).sort('month'))
r.select('MVT_ID_mvt','month','old','new','delta').write_parquet('lirf_overlay_q15_rows_v4.parquet')
print(r.sort(pl.col('delta').abs(),descending=True).select('month','old','new','delta','qu','ms').head(8))

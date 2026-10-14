"""Two remaining-lever checks on the v23-equivalent OOF (cv_rows_cb_keep): (A) unmatched lane, model vs raw ADS-B;
(B) LIRF schedule-copy labels, cross-month airline-level blend toward the schedule value + oracle bound."""
import polars as pl, numpy as np, os
S=os.environ['SCRATCH']
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(20)
cv=pl.read_parquet(S+'/learn/cv_rows_cb_keep.parquet').with_columns((pl.col('b')+pl.col('c_learned')).alias('p'))
ft=pl.read_parquet('data/work_flseq/feat_train.parquet',columns=['MVT_ID_mvt','mvt_minus_aobt3','mvt_minus_sched','month','AIRCRAFT_OPERATOR_flt','TAXITIME_SEC_mvt','ADEP_mvt'])
st=pl.read_parquet(S+'/learn/feat_2025.parquet',columns=['MVT_ID_mvt','t_mvt','ground_fix','t_move','t_push'])
D=cv.join(ft.drop('ADEP_mvt','TAXITIME_SEC_mvt'),on='MVT_ID_mvt',how='left').join(st,on='MVT_ID_mvt',how='left')
SSE=float(((D['p']-D['y'])**2).sum()); N=D.height
def r(e): return float(np.sqrt(np.mean(e**2))) if len(e) else float('nan')
print('=== A. unmatched lane')
U=D.filter(pl.col('mvt_minus_aobt3').is_null()).with_columns(
  pl.when(~pl.col('has')).then(pl.lit('0 none')).when(~pl.col('ground_fix').fill_null(False)).then(pl.lit('1 air'))
  .when(pl.col('t_move').is_null()).then(pl.lit('2 gnd')).when(pl.col('t_push').is_null()).then(pl.lit('3 stand')).otherwise(pl.lit('4 push')).alias('k'),
  (pl.col('t_mvt')-pl.col('t_push')).alias('a_push'),(pl.col('t_mvt')-pl.col('t_move')).alias('a_move'))
rows=[]
for k in sorted(U['k'].unique()):
    g=U.filter(pl.col('k')==k); y=g['y'].to_numpy(); p=g['p'].to_numpy(); ok=y<4*3600
    row=dict(k=k,n=g.height,pct_sse=round(((p-y)**2).sum()/SSE*100,2),rmse=round(r(p-y)),rmse_lt4h=round(r((p-y)[ok])),medAE=round(float(np.median(np.abs(p-y)))))
    for c in ['a_push','a_move']:
        a=g[c].to_numpy(); m=~np.isnan(a)
        if m.any():
            row[c+'_n']=int(m.sum()); row[c+'_medAE']=round(float(np.median(np.abs(a[m]-y[m])))); row[c+'_model_medAE']=round(float(np.median(np.abs(p[m]-y[m]))))
    rows.append(row)
print(pl.DataFrame(rows))
big=U.filter(pl.col('y')>=4*3600); sm=U.filter(pl.col('y')<4*3600)
print('unmatched y>=4h: n=%d, SSE share %.1f%%'%(big.height,float(((big['p']-big['y'])**2).sum())/SSE*100))
print('unmatched y<4h : n=%d, SSE share %.1f%%, RMSE %.0f, bias %+.0f'%(sm.height,float(((sm['p']-sm['y'])**2).sum())/SSE*100,r((sm['p']-sm['y']).to_numpy()),float((sm['p']-sm['y']).mean())))
print('\n=== B. LIRF schedule-copy labels')
T=ft.filter((pl.col('ADEP_mvt')=='LIRF') & pl.col('mvt_minus_aobt3').is_not_null()).with_columns(((pl.col('TAXITIME_SEC_mvt')-pl.col('mvt_minus_sched')).abs()<=12).alias('copy'))
print('copy rate by month (2025 training rows, matched):', T.group_by('month').agg(pl.col('copy').mean().round(3)).sort('month').rows())
L=D.filter((pl.col('ADEP_mvt')=='LIRF') & pl.col('mvt_minus_aobt3').is_not_null()).with_columns(((pl.col('y')-pl.col('mvt_minus_sched')).abs()<=12).alias('copy'))
print(L.group_by('AIRCRAFT_OPERATOR_flt').agg(pl.len().alias('n'),pl.col('copy').mean().round(3).alias('copy_rate')).sort('n',descending=True).head(12))
L=L.with_columns((pl.col('mvt_minus_sched')-pl.col('p')).alias('dsp'),(pl.col('y')-pl.col('p')).alias('res'))
out=[]
for m_fit,m_ap in [(1,7),(7,1)]:
    F=L.filter(pl.col('month')==m_fit).group_by('AIRCRAFT_OPERATOR_flt').agg(((pl.col('dsp')*pl.col('res')).sum()/((pl.col('dsp')**2).sum()+1e9)).alias('w'))
    out.append(L.filter(pl.col('month')==m_ap).join(F,on='AIRCRAFT_OPERATOR_flt',how='left').with_columns(pl.col('w').fill_null(0).clip(0,1)))
A=pl.concat(out)
old=float(((A['p']-A['y'])**2).sum()); new=float(((A['p']+A['w']*A['dsp']-A['y'])**2).sum())
print('cross-month airline blend toward schedule: total RMSE %.2f -> %.2f'%(np.sqrt(SSE/N),np.sqrt((SSE-old+new)/N)))
orc=float(((L.select(pl.when(pl.col('copy')).then(pl.col('mvt_minus_sched')).otherwise(pl.col('p'))).to_series()-L['y'])**2).sum())
print('oracle (copies known exactly): total RMSE -> %.2f'%np.sqrt((SSE-float(((L['p']-L['y'])**2).sum())+orc)/N))

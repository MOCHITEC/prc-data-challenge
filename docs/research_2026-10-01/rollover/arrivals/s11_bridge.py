import polars as pl, numpy as np
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
x=pl.read_parquet('xapt_best.parquet')
x=x.with_columns(link=(pl.col('lag').abs()<1800)&pl.col('amatched')&pl.col('aobt').is_not_null(),
  t_aobt=(pl.col('MVT_TIME_UTC_mvt')-pl.col('aobt')).dt.total_seconds(),
  t_own=(pl.col('MVT_TIME_UTC_mvt')-pl.col('AOBT_3_flt')).dt.total_seconds())
t=x.filter(pl.col('src')=='train')
m=t.filter(pl.col('matched')&pl.col('t_own').is_not_null()&(pl.col('y')<7200))
print('matched rows: corr y vs MVT-AOBT_3 own', np.corrcoef(m['y'],m['t_own'])[0,1], 'rmse(y - t_own - median bias)', float(np.sqrt(((m['y']-m['t_own'])-(m['y']-m['t_own']).median())**2).mean()) if False else None)
def stats(s,label):
  r=(s['y']-s['t_aobt']).to_numpy().astype(float); b=float(np.median(r))
  print(label,'n',s.height,'bias',b,'rmse after bias',float(np.sqrt(np.mean((r-b)**2))),'rmse of airport-median', float(np.sqrt(np.mean(((s['y']-s['ymed']).to_numpy().astype(float))**2))))
u=t.filter(~pl.col('matched'))
med=u.filter(pl.col('y')<7200).group_by('APT').agg(ymed=pl.col('y').median())
u=u.join(med,on='APT')
ul=u.filter(pl.col('link'))
print(ul.group_by('APT').agg(n=pl.len(), fill=((pl.col('y')-pl.col('ms')).abs()<=60).sum(), big=(pl.col('y')>7200).sum()).sort('n',descending=True))
stats(ul.filter(pl.col('y')<7200),'unmatched linked, y<2h')
stats(ul.filter((pl.col('y')<7200)&(pl.col('t_aobt').abs()<7200)),'unmatched linked, y<2h, |t|<2h')
# own matched for comparison (same estimator but own AOBT)
mm=t.filter(pl.col('link')&pl.col('matched')&(pl.col('y')<7200)).join(med,on='APT')
stats(mm,'matched linked (ARR AOBT), y<2h')
print('rank unmatched linked by APT'); print(x.filter((pl.col('src')=='rank')&~pl.col('matched')&pl.col('link')).group_by('APT').len().sort('len',descending=True))
x.write_parquet('bridge.parquet')

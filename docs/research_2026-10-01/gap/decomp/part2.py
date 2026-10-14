import polars as pl, numpy as np
S='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/gap/decomp/'
d=pl.read_parquet(S+'cv_rows.parquet'); N=d.height; SSE=d['se'].sum()
d=d.with_columns(cls=pl.when(pl.col('rollover')).then(pl.lit('roll')).when(~pl.col('matched')).then(pl.lit('unm')).when(pl.col('copy_any')).then(pl.lit('copy')).otherwise(pl.lit('gen')))
t=d.group_by('ADEP_mvt','cls').agg(n=pl.len(),mse_c=pl.col('se').sum()/N,rmse=pl.col('se').mean().sqrt(),
   big=(pl.col('e').abs()>1800).sum(), sse_big_c=pl.col('se').filter(pl.col('e').abs()>1800).sum()/N).sort('ADEP_mvt','cls')
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
print(t)
# matched rows: tail structure
m=d.filter(pl.col('matched')&~pl.col('rollover'))
for thr in [600,1200,1800,3600]:
    s=m.filter(pl.col('e').abs()>thr); print(f'matched |e|>{thr}: n={s.height} mse_contrib={s["se"].sum()/N:.0f} share_of_matched={s["se"].sum()/m["se"].sum()*100:.1f}%')
se=np.sort(m['se'].to_numpy())[::-1]; ms=m['se'].sum()
for k in [10,100,1000,3000,10000]:
    print(f'matched top{k}: {se[:k].sum()/ms*100:.1f}% of matched SSE, RMSE w/o {((ms-se[:k].sum())/(m.height-k))**.5:.1f}')
# LIRF matched top rows
l=d.filter((pl.col('ADEP_mvt')=='LIRF')&pl.col('matched')).sort('se',descending=True)
print('LIRF matched top 30 share of LIRF matched SSE', l.head(30)['se'].sum()/l['se'].sum(), 'top300', l.head(300)['se'].sum()/l['se'].sum())
print('LIRF matched w/o top 30 RMSE', (l['se'][30:].mean())**.5, ' w/o top300', (l['se'][300:].mean())**.5)
# per-airport after ADS-B CV gains (doc adsb-learned-residual §2), gain assumed on matched rows
gain={'EDDF':13059,'EDDM':13114,'EGLL':8447,'EHAM':15145,'LEBL':11657,'LEMD':3383,'LFPG':832,'LIRF':14404,'LSZH':14641,'LTFM':2}
a=d.filter(~pl.col('rollover')).group_by('ADEP_mvt').agg(n=pl.len(),nm=pl.col('matched').sum(),sse_m=pl.col('se').filter(pl.col('matched')).sum(),sse_u=pl.col('se').filter(~pl.col('matched')).sum())
nall=d.group_by('ADEP_mvt').len().rename({'len':'nall'})
a=a.join(nall,on='ADEP_mvt').with_columns(G=pl.col('ADEP_mvt').replace_strict(gain,default=0)*pl.col('nall'))
a=a.with_columns(rm_pre=(pl.col('sse_m')/pl.col('nm')).sqrt(),rm_post=((pl.col('sse_m')-pl.col('G')).clip(0)/pl.col('nm')).sqrt(),
    c_m_pre=pl.col('sse_m')/N, c_m_post=(pl.col('sse_m')-pl.col('G'))/N, c_u=pl.col('sse_u')/N).sort('ADEP_mvt')
print(a.select('ADEP_mvt','nm','rm_pre','rm_post','c_m_pre','c_m_post','c_u'))
print('totals pre matched',a['c_m_pre'].sum(),'post',a['c_m_post'].sum(),'unm',a['c_u'].sum())

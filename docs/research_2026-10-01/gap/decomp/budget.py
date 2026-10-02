import polars as pl, numpy as np, os
R='/home/user/prc-data-challenge/'
cols=['MVT_ID_mvt','ADEP_mvt','TAXITIME_SEC_mvt','mvt_minus_aobt3','mvt_minus_sched','mvt_minus_eobt1','mvt_minus_lobt','mvt_minus_iobt','month','STAND_mvt','PHASE_mvt']
ft=pl.read_parquet(R+'data/work_flseq/feat_train.parquet',columns=cols)
folds=pl.read_parquet(R+'experiments/oof_main_flseq_osm_nolirf-926741d779_folds.parquet')
oof=pl.read_parquet(R+'experiments/oof_main_flseq_osm_nolirf-926741d779.parquet')
print('oof ids unique', oof['MVT_ID_mvt'].n_unique(), oof.height)
for fold in ['jan','jul','forward','jul_fwd']:
    pass
d=(folds.filter(pl.col('fold').is_in(['jan','jul'])).join(oof,on='MVT_ID_mvt',how='left').join(ft,on='MVT_ID_mvt',how='left'))
print(d.height, d['oof'].null_count(), d['TAXITIME_SEC_mvt'].null_count())
y=pl.col('TAXITIME_SEC_mvt').cast(pl.Float64)
d=d.with_columns(e=(y-pl.col('oof')), matched=pl.col('mvt_minus_aobt3').is_not_null())
d=d.with_columns(se=pl.col('e')**2)
# clock copy: BLOCK = MVT - y ; clock X = MVT - mvt_minus_X ; |BLOCK - X| = |mvt_minus_X - y|
cc=[]
for c in ['mvt_minus_aobt3','mvt_minus_sched','mvt_minus_eobt1','mvt_minus_lobt','mvt_minus_iobt']:
    d=d.with_columns(((pl.col(c)-y).abs()<=60).fill_null(False).alias('cc_'+c))
d=d.with_columns(copy_any=pl.any_horizontal([pl.col('cc_'+c) for c in ['mvt_minus_aobt3','mvt_minus_sched','mvt_minus_eobt1','mvt_minus_lobt','mvt_minus_iobt']]),
                 rollover=(y>72000))
N=d.height; SSE=d['se'].sum(); MSE=SSE/N
print(f'N={N} RMSE={MSE**.5:.2f} MSE={MSE:.0f}')
def show(name,mask):
    s=d.filter(mask); n=s.height; sse=s['se'].sum()
    print(f'{name:55s} n={n:7d} ({n/N*100:5.2f}%) RMSE={((sse/n)**.5 if n else 0):8.1f} SSEshare={sse/SSE*100:5.1f}% MSEcontrib={sse/N:8.0f}')
m=pl.col('matched'); ltfm=pl.col('ADEP_mvt')=='LTFM'; lirf=pl.col('ADEP_mvt')=='LIRF'
show('ALL',pl.lit(True))
show('matched',m)
show('matched rollover y>72000',m&pl.col('rollover'))
show('matched clock-copy (any clock +-60) excl rollover',m&pl.col('copy_any')&~pl.col('rollover'))
show('matched copy of AOBT_3',m&pl.col('cc_mvt_minus_aobt3'))
show('matched copy of SCHED (not AOBT3)',m&pl.col('cc_mvt_minus_sched')&~pl.col('cc_mvt_minus_aobt3'))
show('matched genuine',m&~pl.col('copy_any')&~pl.col('rollover'))
show('matched genuine LTFM',m&~pl.col('copy_any')&~pl.col('rollover')&ltfm)
show('matched genuine non-LTFM',m&~pl.col('copy_any')&~pl.col('rollover')&~ltfm)
show('matched LTFM all',m&ltfm)
show('matched non-LTFM all',m&~ltfm)
show('unmatched',~m)
show('unmatched LIRF',~m&lirf)
show('unmatched LIRF rollover',~m&lirf&pl.col('rollover'))
show('unmatched LIRF sched-fill copy',~m&lirf&pl.col('cc_mvt_minus_sched')&~pl.col('rollover'))
show('unmatched LIRF other',~m&lirf&~pl.col('cc_mvt_minus_sched')&~pl.col('rollover'))
show('unmatched non-LIRF',~m&~lirf)
show('unmatched non-LIRF rollover',~m&~lirf&pl.col('rollover'))
show('unmatched non-LIRF sched copy',~m&~lirf&pl.col('cc_mvt_minus_sched')&~pl.col('rollover'))
show('ALL rollover',pl.col('rollover'))
print('--- top-k')
se=np.sort(d['se'].to_numpy())[::-1]
for k in [1,5,10,20,50,100,200,500,1000,2000,5000,10000,34484]:
    print(f'top{k:6d} share={se[:k].sum()/SSE*100:5.1f}%  RMSE w/o = {((SSE-se[:k].sum())/(N-k))**.5:7.2f}  MSE removed/row={se[:k].sum()/N:7.0f}')
print('--- |e| buckets')
ae=d['e'].abs()
for lo,hi in [(0,60),(60,120),(120,300),(300,600),(600,1200),(1200,3600),(3600,1e9)]:
    show(f'|e| in [{lo},{hi})',(pl.col('e').abs()>=lo)&(pl.col('e').abs()<hi))
print('--- per airport matched genuine / all')
g=d.group_by('ADEP_mvt').agg(n=pl.len(),sse=pl.col('se').sum(),nm=m.sum(),sse_m=pl.col('se').filter(m).sum(),sse_u=pl.col('se').filter(~m).sum(),nu=(~m).sum()).sort('ADEP_mvt')
g=g.with_columns(rmse=(pl.col('sse')/pl.col('n')).sqrt(),rmse_m=(pl.col('sse_m')/pl.col('nm')).sqrt(),rmse_u=(pl.col('sse_u')/pl.col('nu')).sqrt(),share=pl.col('sse')/SSE*100,share_u=pl.col('sse_u')/SSE*100)
pl.Config.set_tbl_rows(20); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
print(g)
d.select('MVT_ID_mvt','ADEP_mvt','fold','matched','copy_any','rollover','e','se').write_parquet(os.path.dirname(__file__)+'/cv_rows.parquet')
# forward fold for comparison
f2=folds.filter(pl.col('fold')=='forward').join(oof,on='MVT_ID_mvt').join(ft.select('MVT_ID_mvt','TAXITIME_SEC_mvt','mvt_minus_aobt3'),on='MVT_ID_mvt')
f2=f2.with_columns(se=(pl.col('TAXITIME_SEC_mvt')-pl.col('oof'))**2,mm=pl.col('mvt_minus_aobt3').is_not_null())
print('forward: all',(f2['se'].mean())**.5,'matched',f2.filter(pl.col('mm'))['se'].mean()**.5,'unm',f2.filter(~pl.col('mm'))['se'].mean()**.5, 'unm SSE share', f2.filter(~pl.col('mm'))['se'].sum()/f2['se'].sum())

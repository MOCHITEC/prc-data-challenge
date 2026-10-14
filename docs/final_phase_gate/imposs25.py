import polars as pl, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'
q=pl.read_parquet(f'{S}/ideas/ht/copyq_lomo_2025.parquet').select('MVT_ID_mvt','q','ms_feat')
d=pl.read_parquet(f'{S}/ideas/ht/d25.parquet').filter(pl.col('matched')&(pl.col('ADEP_mvt')=='LIRF')).join(q,on='MVT_ID_mvt',how='inner')
print('rows',d.height,'ms_feat==ms', (d['ms_feat']==d['ms']).mean())
d=d.with_columns(imp=((pl.col('ms')-pl.col('mlobt')).abs()>3606), day=pl.col('MVT_TIME_UTC_mvt').dt.date())
for base in ['final_12','b']:
    x=d.with_columns(p=pl.col(base))
    ov=pl.col('p')+0.4*pl.col('q')*(pl.col('ms')-pl.col('p'))
    x=x.with_columns(ov=pl.when(ov<0).then(pl.col('p')).otherwise(ov))
    lo=pl.col('mlobt')-3606; hi=pl.col('mlobt')+3606
    x=x.with_columns(A=pl.col('ov').clip(lo,hi), G=pl.when(pl.col('imp')).then(pl.col('p')).otherwise(pl.col('ov')).clip(lo,hi))
    x=x.with_columns(dd=(pl.col('G')-pl.col('y'))**2-(pl.col('A')-pl.col('y'))**2)
    z=x.filter(pl.col('imp'))
    print(f'base {base}: impossible rows {z.height}, copies among them {int(((z["y"]-z["ms"]).abs()<=12).sum())}, mean q {z["q"].mean():.4f}, rows q>0.1 {int((z["q"]>0.1).sum())}')
    print('  dSSE gated-overlay on impossible rows: total %.1f M s^2 = %.2f s^2 of 2025 MSE' % (z['dd'].sum()/1e6, z['dd'].sum()/2085047))
    bym=z.group_by('month').agg(dd=pl.col('dd').sum()/1e6).sort('month')
    print('  by month (M s^2):', {r['month']:round(r['dd'],2) for r in bym.to_dicts()})
    dy=z.group_by('day').agg(dd=pl.col('dd').sum())['dd'].to_numpy()
    rng=np.random.default_rng(0); bs=np.array([rng.choice(dy,len(dy)).sum() for _ in range(10000)])
    print('  day-block bootstrap 95%% CI (M s^2): [%.2f, %.2f]  P(>=0) %.4f' % (np.percentile(bs,2.5)/1e6, np.percentile(bs,97.5)/1e6, np.mean(bs>=0)))
    srt=np.sort(z['dd'].to_numpy()); print('  top-10 rows share of gain', srt[:10].sum()/srt.sum(), ' top-50', srt[:50].sum()/srt.sum())
    # also sanity: whole LIRF matched effect of overlay vs none (A vs p clipped) on possible rows
    w=x.filter(~pl.col('imp')).with_columns(n0=(pl.col('p').clip(lo,hi)-pl.col('y'))**2, n1=(pl.col('A')-pl.col('y'))**2)
    print('  (info) overlay effect on possible rows: %.1f M s^2' % ((w['n1']-w['n0']).sum()/1e6))
    print('  sum (G-A)^2 on impossible rows: %.1f M s^2 -> ratio gain/sumsq = %.2f' % (((z['G']-z['A'])**2).sum()/1e6, -z['dd'].sum()/((z['G']-z['A'])**2).sum()))
    fj=z.filter(pl.col('month').is_in([2,6])); print('  Feb+Jun 2025: rows %d dSSE %.1f M' % (fj.height, fj['dd'].sum()/1e6))

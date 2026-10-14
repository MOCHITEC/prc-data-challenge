import polars as pl
R='/home/user/prc-data-challenge/'; S='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/gap/decomp/'
d=pl.read_parquet(S+'cv_rows.parquet'); N=d.height
ft=pl.read_parquet(R+'data/work_flseq/feat_train.parquet',columns=['MVT_ID_mvt','TAXITIME_SEC_mvt','mvt_minus_aobt3','mvt_minus_sched','mvt_minus_eobt1','mvt_minus_lobt'])
d=d.join(ft,on='MVT_ID_mvt')
y=pl.col('TAXITIME_SEC_mvt')
for ap in ['LIRF','LEBL','LTFM','LFPG','EGLL']:
    x=d.filter((pl.col('ADEP_mvt')==ap)&pl.col('matched')&~pl.col('rollover'))
    x=x.with_columns(cs=((pl.col('mvt_minus_sched')-y).abs()<=60), ca=((pl.col('mvt_minus_aobt3')-y).abs()<=60), ce=((pl.col('mvt_minus_eobt1')-y).abs()<=60), cl=((pl.col('mvt_minus_lobt')-y).abs()<=60),
                     gap=(pl.col('mvt_minus_sched')-pl.col('mvt_minus_aobt3')))
    for nm,mask in [('sched copy, not aobt',pl.col('cs')&~pl.col('ca')),('sched copy & |sched-aobt|>900',pl.col('cs')&~pl.col('ca')&(pl.col('gap').abs()>900)),('eobt/lobt copy only',(pl.col('ce')|pl.col('cl'))&~pl.col('ca')&~pl.col('cs')),('aobt copy',pl.col('ca'))]:
        s=x.filter(mask)
        print(f'{ap} {nm:32s} n={s.height:6d} mse_contrib={s["se"].sum()/N:7.0f} rmse={s["se"].mean()**.5 if s.height else 0:6.0f}')
    # among rows with |sched-aobt|>900, share that are sched copies
    big=x.filter(pl.col('gap').abs()>900)
    print(f'  {ap} rows with |SCHED-AOBT3|>900: {big.height}, sched-copy frac {big["cs"].mean():.3f}, mse_contrib {big["se"].sum()/N:.0f}')

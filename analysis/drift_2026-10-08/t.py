import pandas as pd, numpy as np, glob
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','EOBT_1_flt','AOBT_3_flt','TAXITIME_SEC_mvt']
tr=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob('data/raw/training_*.parquet'))])
r=pd.read_parquet('data/raw/ranking.parquet',columns=cols); d=pd.concat([tr,r]); d=d[(d.PHASE_mvt=='DEP')]
T=lambda c: pd.to_datetime(d[c])
d['ym']=T('MVT_TIME_UTC_mvt').dt.strftime('%y%m')
d['le']=(T('LOBT_flt')-T('EOBT_1_flt')).dt.total_seconds()
d['es']=(T('EOBT_1_flt')-T('SCHED_TIME_UTC_mvt')).dt.total_seconds()
d['ls']=(T('LOBT_flt')-T('SCHED_TIME_UTC_mvt')).dt.total_seconds()
d['ae']=(T('AOBT_3_flt')-T('EOBT_1_flt')).dt.total_seconds(); d['al']=(T('AOBT_3_flt')-T('LOBT_flt')).dt.total_seconds()
d['be']=(T('BLOCK_TIME_UTC_mvt')-T('EOBT_1_flt')).dt.total_seconds(); d['bl']=(T('BLOCK_TIME_UTC_mvt')-T('LOBT_flt')).dt.total_seconds()
for ap in ['LFPG','EGLL','LTFM']:
    x=d[d.ADEP_mvt==ap]
    g=x.groupby('ym').agg(le_neg=('le',lambda s:(s<0).mean()),le_pos=('le',lambda s:(s>0).mean()),es_neg=('es',lambda s:(s<0).mean()),es_pos=('es',lambda s:(s>0).mean()),ls_neg=('ls',lambda s:(s<0).mean()),ls_pos=('ls',lambda s:(s>0).mean()),
       ae_med=('ae','median'),al_med=('al','median'))
    print(ap); print(g.round(2).T.to_string())
# 2025: does BLOCK follow LOBT or EOBT_1 where they differ?
y=d[(d.ym<'2601')&(d.le!=0)&d.le.abs().lt(7200)].copy()
y['res']=((T('AOBT_3_flt')-T('BLOCK_TIME_UTC_mvt')).dt.total_seconds())[y.index] if False else None

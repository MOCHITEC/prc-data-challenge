import pandas as pd, numpy as np
cols=['MVT_ID_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','aobt3_minus_eobt1','aobt3_minus_sched','mvt_minus_aobt3','TAXITIME_SEC_mvt']
a=pd.read_parquet('data/work_flseq/feat_train.parquet',columns=cols,filters=[('ADEP_mvt','=','LFPG')])
b=pd.read_parquet('data/work_flseq/feat_rank.parquet',columns=cols,filters=[('ADEP_mvt','=','LFPG')])
sub=pd.read_parquet('/home/user/prc-data-challenge/outputs/c15_v42_flfixm_seeds_ucmix.parquet').rename(columns={'TAXITIME_SEC_mvt':'pred'})
b=b.drop(columns='TAXITIME_SEC_mvt').merge(sub,on='MVT_ID_mvt')
o=pd.concat([pd.read_parquet(f'experiments/oof_main_flseq_osm_nolirf_{t}.parquet').merge(pd.read_parquet(f'experiments/oof_main_flseq_osm_nolirf_{t}_folds.parquet')[['MVT_ID_mvt']],on='MVT_ID_mvt') for t in ['jj-c96d0c096c','aso-3a1c6221ff','fmj-c953f6665a']]).drop_duplicates('MVT_ID_mvt')
a=a.merge(o,on='MVT_ID_mvt',how='left')
for d in (a,b): d['m']=pd.to_datetime(d.MVT_TIME_UTC_mvt).dt.month; d['eob']=d.aobt3_minus_sched-d.aobt3_minus_eobt1  # eobt-sched
a['res']=(a.TAXITIME_SEC_mvt-a.mvt_minus_aobt3).clip(-1200,1200); a['ores']=(a.oof-a.mvt_minus_aobt3).clip(-1200,1200)
b['pres']=(b.pred-b.mvt_minus_aobt3).clip(-1200,1200)
bins=[-1e9,-600,-300,-120,-60,0,60,120,300,600,1200,3600,1e9]
a['bk']=pd.cut(a.aobt3_minus_eobt1,bins); b['bk']=pd.cut(b.aobt3_minus_eobt1,bins)
t=pd.DataFrame({'n25':a.groupby('bk').size(),'res25':a.groupby('bk').res.mean(),'oofres25':a.groupby('bk').ores.mean(),
  'n26jul':b[b.m==7].groupby('bk').size(),'pres26jul':b[b.m==7].groupby('bk').pres.mean(),'n26jan':b[b.m==1].groupby('bk').size(),'pres26jan':b[b.m==1].groupby('bk').pres.mean()})
print(t.round(0).to_string())
print('EOBT-SCHED median by month 2025:',a.groupby('m').eob.median().to_dict(),' 2026:',b.groupby('m').eob.median().to_dict())
print('share eobt1-sched==0', a.groupby('m').eob.apply(lambda s:(s==0).mean()).round(2).to_dict(), b.groupby('m').eob.apply(lambda s:(s==0).mean()).round(2).to_dict())

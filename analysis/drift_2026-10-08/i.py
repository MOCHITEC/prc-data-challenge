import pandas as pd, numpy as np, glob
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','TAXITIME_SEC_mvt']
tr=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob('data/raw/training_*.parquet'))]); tr=tr[tr.PHASE_mvt=='DEP']
t=pd.to_datetime(tr.MVT_TIME_UTC_mvt); tr['date']=t.dt.date; tr['m']=t.dt.month
tr['proxy']=(t-pd.to_datetime(tr.AOBT_3_flt)).dt.total_seconds()
parts=[]
for tag in ['jj-c96d0c096c','aso-3a1c6221ff','fmj-c953f6665a']:
    o=pd.read_parquet(f'experiments/oof_main_flseq_osm_nolirf_{tag}.parquet'); fo=pd.read_parquet(f'experiments/oof_main_flseq_osm_nolirf_{tag}_folds.parquet')
    o=o.merge(fo[['MVT_ID_mvt','fold']],on='MVT_ID_mvt'); o['src']=tag[:3]; parts.append(o)
o=pd.concat(parts).drop_duplicates('MVT_ID_mvt')
d=tr.merge(o,on='MVT_ID_mvt'); d=d[d.proxy.notna()&(d.proxy.abs()<7200)&(d.TAXITIME_SEC_mvt<7200)]
d['err']=d.oof-d.TAXITIME_SEC_mvt
print(d.groupby('ADEP_mvt').m.unique())
# daily aggregates
g=d.groupby(['ADEP_mvt','m','date']).agg(n=('err','size'),proxy=('proxy','mean'),err=('err','mean'),y=('TAXITIME_SEC_mvt','mean'),oof=('oof','mean')).reset_index()
g['proxy_dev']=g.proxy-g.groupby('ADEP_mvt').proxy.transform('mean')
res=[]
for ap,x in g.groupby('ADEP_mvt'):
    b=np.polyfit(x.proxy_dev,x.err,1,w=np.sqrt(x.n))
    by=np.polyfit(x.proxy_dev,x.y-x.y.mean(),1,w=np.sqrt(x.n)); bo=np.polyfit(x.proxy_dev,x.oof-x.oof.mean(),1,w=np.sqrt(x.n))
    res.append((ap,len(x),round(b[0],3),round(by[0],3),round(bo[0],3),round(x.proxy_dev.std(),0)))
print(pd.DataFrame(res,columns=['apt','days','slope_err_vs_proxydev','slope_y','slope_oof','sd_proxy_dev']).to_string())
# monthly
mm=d.groupby(['ADEP_mvt','m']).agg(proxy=('proxy','mean'),err=('err','mean'),n=('err','size')).reset_index()
print(mm[mm.ADEP_mvt=='LFPG'].round(1).to_string())
g.to_parquet('/tmp/claude-0/-home-user-prc-data-challenge/7ae154e2-3baf-5cb4-beae-354c18126ab1/scratchpad/agents/drift/daily_oof.parquet')

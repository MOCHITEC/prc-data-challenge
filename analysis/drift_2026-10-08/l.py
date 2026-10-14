import pandas as pd, numpy as np, lightgbm as lgb
flt=[('ADEP_mvt','=','LFPG')]
a=pd.read_parquet('data/work_flseq/feat_train.parquet',filters=flt); b=pd.read_parquet('data/work_flseq/feat_rank.parquet',filters=flt)
for d in (a,b): d['m']=pd.to_datetime(d.MVT_TIME_UTC_mvt).dt.month
a=a[a.mvt_minus_aobt3.notna()]; b=b[b.mvt_minus_aobt3.notna()]
a['res']=(a.TAXITIME_SEC_mvt-a.mvt_minus_aobt3).clip(-1200,1200)
bad={'MVT_ID_mvt','FLIGHT_ID_mvt','TAXITIME_SEC_mvt','m','res'}
feats=[c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and c not in bad and not c.startswith('y_') and 'taxitime' not in c.lower()]
print(len(feats))
tr=a[a.m!=7]; te=a[a.m==7]; b7=b[b.m==7]
p=dict(objective='l2',learning_rate=0.05,num_leaves=127,min_data_in_leaf=50,feature_fraction=0.5,bagging_fraction=0.7,bagging_freq=1,verbose=-1,num_threads=4)
mdl=lgb.train(p,lgb.Dataset(tr[feats],tr.res),600)
pa=mdl.predict(te[feats]); pb=mdl.predict(b7[feats])
print('Jul25 truth res',te.res.mean().round(1),'surrogate',pa.mean().round(1),' Jul26 surrogate',pb.mean().round(1))
ca=mdl.predict(te[feats],pred_contrib=True).mean(0); cb=mdl.predict(b7[feats],pred_contrib=True).mean(0)
diff=pd.Series(cb[:-1]-ca[:-1],index=feats).sort_values()
print(diff.head(20).round(1).to_string()); print(diff.tail(8).round(1).to_string())
# also other months 2026 jan
b1=b[b.m==1]; te1=a[a.m==1]
print('Jan25 truth',te1.res.mean().round(1),'Jan26 surrogate(in-sample model)',mdl.predict(b1[feats]).mean().round(1))

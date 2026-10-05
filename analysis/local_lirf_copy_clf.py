"""Can LIRF schedule-copy labels (|y - mvt_minus_sched| <= 12 s) be predicted from the pipeline features?
Classifier fit on LIRF matched rows of the 10 months other than Jan/Jul 2025, scored on the Jan+Jul OOF rows;
then mix the v23-equivalent prediction toward the schedule value: p' = p + a*q*(s - p), a fit on the training months'
own in-sample q is avoided by choosing a on a held-out month (Dec) of the training side."""
import polars as pl, numpy as np, os, lightgbm as lgb
from sklearn.metrics import roc_auc_score
S=os.environ['SCRATCH']
lf=pl.scan_parquet('data/work_flseq/feat_train.parquet').filter((pl.col('ADEP_mvt')=='LIRF') & pl.col('mvt_minus_aobt3').is_not_null())
T=lf.collect()
T=T.with_columns(((pl.col('TAXITIME_SEC_mvt')-pl.col('mvt_minus_sched')).abs()<=12).alias('copy'))
drop={'TAXITIME_SEC_mvt','copy','MVT_ID_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt'}
feats=[c for c,t in T.schema.items() if c not in drop and t.is_numeric()]
cats=[c for c,t in T.schema.items() if c not in drop and t==pl.String]
X=T.select(feats+[pl.col(c).cast(pl.Categorical).to_physical().alias(c) for c in cats])
fn=feats+cats
m=T['month'].to_numpy(); y=T['copy'].to_numpy().astype(int)
tr=~np.isin(m,[1,7,12]); va=m==12; te=np.isin(m,[1,7])
Xn=X.to_numpy().astype(np.float64)
P=dict(objective='binary',learning_rate=0.05,num_leaves=63,min_data_in_leaf=100,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=2)
mdl=lgb.train(P,lgb.Dataset(Xn[tr],y[tr],feature_name=fn,categorical_feature=cats),2000,valid_sets=[lgb.Dataset(Xn[va],y[va])],callbacks=[lgb.early_stopping(100,verbose=False)])
q=mdl.predict(Xn,num_iteration=mdl.best_iteration)
print('copy rate train/dec/test: %.3f %.3f %.3f'%(y[tr].mean(),y[va].mean(),y[te].mean()))
print('AUC dec %.3f | Jan+Jul %.3f | best_iter %d'%(roc_auc_score(y[va],q[va]),roc_auc_score(y[te],q[te]),mdl.best_iteration))
for thr in (0.5,0.7,0.9):
    s=q[te]>=thr; print(f'  q>={thr}: {s.sum()} rows, precision {y[te][s].mean() if s.any() else float("nan"):.3f}, recall {(y[te]&s).sum()/y[te].sum():.3f}')
imp=sorted(zip(fn,mdl.feature_importance('gain')),key=lambda t:-t[1])[:10]; tot=sum(mdl.feature_importance('gain'))
print('top features:',[(f,round(g/tot,3)) for f,g in imp])
Q=pl.DataFrame({'MVT_ID_mvt':T['MVT_ID_mvt'],'q':q,'s':T['mvt_minus_sched']})
cv=pl.read_parquet(S+'/learn/cv_rows_cb_keep.parquet').with_columns((pl.col('b')+pl.col('c_learned')).alias('p'))
SSE=float(((cv['p']-cv['y'])**2).sum()); N=cv.height
L=cv.join(Q,on='MVT_ID_mvt',how='inner')
p,yy,qq,ss=[L[c].to_numpy() for c in ('p','y','q','s')]
base=((p-yy)**2).sum()
# choose a on one month and evaluate on the other (cross-month), plus report the a grid
mon=L.join(T.select('MVT_ID_mvt','month'),on='MVT_ID_mvt')['month'].to_numpy()
grid=np.linspace(0,1.5,31)
def sse(a,msk): return (((p+a*qq*(ss-p))-yy)[msk]**2).sum()
tot_new=0
for fit,ap in [(1,7),(7,1)]:
    a=grid[np.argmin([sse(g,mon==fit) for g in grid])]; tot_new+=sse(a,mon==ap); print(f'a fit on month {fit}: {a:.2f}')
print('LIRF rows %d | total RMSE %.2f -> %.2f (cross-month a)'%(len(p),np.sqrt(SSE/N),np.sqrt((SSE-base+tot_new)/N)))
for g in (0.25,0.5,1.0): print(f'  a={g}: total RMSE {np.sqrt((SSE-base+sse(g,np.ones(len(p),bool)))/N):.2f}')

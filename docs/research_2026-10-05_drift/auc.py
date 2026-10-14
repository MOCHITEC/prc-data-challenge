"""Single-column adversarial AUC, Jan+Jul 2025 (feat_train) vs Jan+Jul 2026 (feat_rank), for every numeric frame
column and the lever columns L1/L2/L3/L5. Label-free; nothing reads TAXITIME or BLOCK of the ranking set."""
import glob, numpy as np, polars as pl, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from prc.levers import attach_levers, LEVER_COLS
R='/home/user/prc-data-challenge/data/'
mon=lambda c: pl.col(c).dt.month().is_in([1,7])
ftr=pl.read_parquet(R+'work/feat_train.parquet').filter(mon('MVT_TIME_UTC_mvt'))
frk=pl.read_parquet(R+'work/feat_rank.parquet')
raw_tr=pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(R+'raw/training_2025-0[17]-*.parquet'))+sorted(glob.glob(R+'raw/training_2025-0[268]-*.parquet'))],how='diagonal')
raw_rk=pl.read_parquet(R+'raw/ranking.parquet')
ftr,frk=attach_levers(ftr,frk,raw_tr,raw_rk,True,True,True,True)
del raw_tr,raw_rk
skip={'TAXITIME_SEC_mvt','MVT_ID_mvt','FLIGHT_ID_mvt','month','year'}
num=[c for c,d in frk.schema.items() if d.is_numeric() and c not in skip and c in ftr.columns]
rng=np.random.default_rng(0)
a=ftr.sample(min(150000,ftr.height),seed=0); b=frk.sample(min(150000,frk.height),seed=1)
y=np.r_[np.zeros(a.height),np.ones(b.height)]; fold=rng.integers(0,2,len(y))
P=dict(objective='binary',num_leaves=15,min_data_in_leaf=200,learning_rate=0.1,verbose=-1,num_threads=1)
out=[]
for c in num:
    x=np.r_[a[c].cast(pl.Float64).to_numpy(),b[c].cast(pl.Float64).to_numpy()].reshape(-1,1)
    p=np.empty(len(y))
    for k in (0,1):
        m=lgb.train(P,lgb.Dataset(x[fold!=k],y[fold!=k]),60); p[fold==k]=m.predict(x[fold==k])
    na=(np.isnan(x[y==0]).mean(),np.isnan(x[y==1]).mean())
    out.append((c,roc_auc_score(y,p),na[0],na[1],c in LEVER_COLS))
t=pl.DataFrame(out,schema=['col','auc','null_2025','null_2026','lever'],orient='row').sort('auc',descending=True)
t.write_csv('/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/measure/drift/auc.csv')
pl.Config.set_tbl_rows(60); print(t.head(50))

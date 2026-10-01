import polars as pl, numpy as np, lightgbm as lgb, json, sys
d=pl.read_parquet("resid_frame.parquet")
f=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").collect_schema()
num=[c for c,t in f.items() if t in (pl.Float64,pl.Float32,pl.Int64,pl.Int32,pl.Int8,pl.Int16,pl.UInt32,pl.UInt8) and c not in ("TAXITIME_SEC_mvt","MVT_ID_mvt","FLIGHT_ID_mvt","month")]
cand=['ma_rel','l_i','e_l','fltd','air_rel','day_short','day_dly','day_ma_rel','nb_ma_rel_30','nb_ma_rel_60','nb_short_30','nb_short_60','nb_dly_30','nb_dly_60','nb_rev_30','nb_rev_60','nb_a_e_30','nb_a_e_60','op3h_ma_rel','op3h_n','arr_ti_rel_30','arr_ti_rel_60']
sets={"A0":["pred","ap"]+num, "A1":["pred","ap"]+num+cand}
P=dict(objective="regression",learning_rate=0.03,num_leaves=31,min_data_in_leaf=300,feature_fraction=1.0,lambda_l2=10,verbose=-1,num_threads=1,seed=0)
tr_mask=~d["is_rollover"].to_numpy()
mo=d["mo"].to_numpy(); err=d["err"].to_numpy()
out={}
for name,cols in sets.items():
    X=d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    p=np.zeros(len(d))
    for m in (1,7):
        tr=(mo!=m)&tr_mask; te=mo==m
        b=lgb.train(P,lgb.Dataset(X[tr],err[tr]),int(sys.argv[1]) if len(sys.argv)>1 else 400)
        p[te]=b.predict(X[te])
        if name=="A1" and m==7:
            imp=sorted(zip(b.feature_importance("gain"),cols),reverse=True)[:25]; print(imp)
    out[name]=p
np.save("resid_preds.npy",np.vstack([out["A0"],out["A1"]]))

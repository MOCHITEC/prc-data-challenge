"""Same recipe as ../make_oof.py (train Feb-Jun+Aug-Dec 2025, predict Jan+Jul, target y-(MVT-AOBT3)), num_threads=1,
baseline vs +tx feature groups. Saves preds per variant."""
import numpy as np, polars as pl, lightgbm as lgb, sys, gc
E="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
sch=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").collect_schema()
num=[c for c,dt in sch.items() if dt in (pl.Float64,pl.Float32,pl.Int64,pl.Int32,pl.Int8,pl.Int16,pl.UInt32,pl.UInt8)
     and c not in ("TAXITIME_SEC_mvt","MVT_ID_mvt","FLIGHT_ID_mvt","month","mo")]
f=(pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet")
   .filter(pl.col("TAXITIME_SEC_mvt").is_not_null() & pl.col("mvt_minus_aobt3").is_not_null())
   .select(["MVT_ID_mvt","TAXITIME_SEC_mvt",pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo")]+[pl.col(c).cast(pl.Float32) for c in num]).collect())
tx=pl.read_parquet(E+"longtaxi/tx_feats.parquet")
TX=[c for c in tx.columns if c.startswith("tx_")]
f=f.join(tx.select(["MVT_ID_mvt"]+TX),on="MVT_ID_mvt",how="left").with_columns([pl.col(c).fill_nan(None) for c in TX])
te_mask=f["mo"].is_in([1,7]).to_numpy()
y=f["TAXITIME_SEC_mvt"].to_numpy().astype(np.float64); px=f["mvt_minus_aobt3"].to_numpy().astype(np.float64)
ids=f.filter(pl.Series(te_mask))["MVT_ID_mvt"]
groups={"base":[],
        "tx_row":[c for c in TX if "day" not in c],
        "tx_day":[c for c in TX if "day" in c and c!="tx_day_n_ratio"],
        "tx_all":[c for c in TX if c!="tx_day_n_ratio"],
        "tx_mma":[c for c in TX if c.startswith(("tx_mma","tx_unm","tx_late3h","tx_tko"))],
        "tx_tir":[c for c in TX if c.startswith("tx_tir")]}
variants=sys.argv[1:] or list(groups)
params=dict(objective="regression",learning_rate=0.08,num_leaves=127,min_data_in_leaf=50,feature_fraction=0.8,
            bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=1,seed=0)
for vv in variants:
    v,_,sd=vv.partition(':'); params['seed']=int(sd or 0); cols=num+groups[v]; v=vv.replace(':','_s')
    X=f.select(cols).to_numpy()
    bst=lgb.train(params,lgb.Dataset(X[~te_mask],(y-px)[~te_mask],free_raw_data=True),600)
    p=px[te_mask]+bst.predict(X[te_mask])
    pl.DataFrame({"MVT_ID_mvt":ids,"pred":p}).write_parquet(E+f"longtaxi/pred_{v}.parquet")
    imp=sorted(zip(bst.feature_importance("gain"),cols),reverse=True)
    open(E+f"longtaxi/gain_{v}.txt","w").write("\n".join(f"{g:.3e} {n}" for g,n in imp[:60]))
    print(v,"rmse",np.sqrt(np.mean((y[te_mask]-p)**2)),flush=True)
    del X,bst; gc.collect()

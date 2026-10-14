import polars as pl, numpy as np, lightgbm as lgb, sys, time
FR=sys.argv[2] if len(sys.argv)>2 else "resid_frame3.parquet"; TAG=sys.argv[3] if len(sys.argv)>3 else ""
d=pl.read_parquet(FR)
f=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").collect_schema()
num=[c for c,t in f.items() if t in (pl.Float64,pl.Float32,pl.Int64,pl.Int32,pl.Int8,pl.Int16,pl.UInt32,pl.UInt8) and c not in ("TAXITIME_SEC_mvt","MVT_ID_mvt","FLIGHT_ID_mvt","month")]
v23=["ref_p10","ref_p20","ref_med","ref_n","arh_med","apt_med","stand_arrival_gap","stand_aobt_turn","stand_arrival_type_match","proxy_minus_mode_min","proxy_at_mode"]
cats=["ap","cat_STAND_mvt","cat_RUNWAY_mvt","cat_AIRCRAFT_OPERATOR_flt","cat_AIRCRAFT_TYPE_mvt"]
G={"ma_rel":["ma_rel"],"rev":["l_i","e_l"],"flt":["fltd","air_rel"],"day":["day_short","day_dly","day_ma_rel"],
   "nb":['nb_ma_rel_30','nb_ma_rel_60','nb_short_30','nb_short_60','nb_dly_30','nb_dly_60','nb_rev_30','nb_rev_60','nb_a_e_30','nb_a_e_60'],
   "op3h":["op3h_ma_rel","op3h_n"],"arr":["arr_ti_rel_30","arr_ti_rel_60"],"gref":["gref_med"],"enc":["fl_gmed","fl_early","fl_n","op_early","op_gmed","st_early"],"fl":["fl_gmed","fl_early","fl_n"],"opst":["op_early","op_gmed","st_early"]}
base=["pred"]+cats+num+v23
sets={"B0":base,"B1":base+sum(G.values(),[])}
which=sys.argv[1].split(",")
for g in G:
    sets["B0+"+g]=base+G[g]
    sets["B1-"+g]=base+[c for k,v in G.items() if k!=g for c in v]
P=dict(objective="regression",learning_rate=0.03,num_leaves=31,min_data_in_leaf=300,feature_fraction=1.0,lambda_l2=10,verbose=-1,num_threads=1,seed=0,max_cat_to_onehot=4,cat_smooth=50,min_data_per_group=200)
tr_mask=~d["is_rollover"].to_numpy(); mo=d["mo"].to_numpy(); err=d["err"].to_numpy()
for name in which:
    cols=sets[name]; t0=time.time()
    X=d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    p=np.zeros(len(d))
    for m in (1,7):
        tr=(mo!=m)&tr_mask; te=mo==m
        b=lgb.train(P,lgb.Dataset(X[tr],err[tr],categorical_feature=[cols.index(c) for c in cats],free_raw_data=True),400)
        p[te]=b.predict(X[te])
    np.save(f"p_{name}{TAG}.npy",p); print(name,len(cols),round(time.time()-t0),flush=True)

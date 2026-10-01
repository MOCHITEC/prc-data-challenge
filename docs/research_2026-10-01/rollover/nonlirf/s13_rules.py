# Candidate rules: training precision (in-sample + leave-one-month-out shift evaluation) and 2026 ranking DEP flag counts
import polars as pl, numpy as np, sys
sys.path.insert(0,'/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf')
from s6_fingerprint import prep
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
tr=pl.read_parquet(O+'dep_train_feat.parquet').join(pl.read_parquet(O+'standgap_train.parquet'),on='MVT_ID_mvt',how='left')
rk=prep(O+'rank.parquet').join(pl.read_parquet(O+'standgap_rank.parquet'),on='MVT_ID_mvt',how='left')
tr=tr.with_columns(mon=pl.col('SCHED_TIME_UTC_mvt').dt.month())
rk=rk.with_columns(mon=pl.col('SCHED_TIME_UTC_mvt').dt.month())
reg=(pl.col('n_occ')>=5)&(pl.col('other_match_share')>=0.8)
nonL=pl.col('APT')!='LIRF'
band=pl.col('ms').is_between(900,2100)
rules={
 'R0 unm & |ms|<2h (all apt)': pl.col('unm')&(pl.col('ms').abs()<7200),
 'R1 unm & ms 900-2100 (all apt)': pl.col('unm')&band,
 'R2 R1 & nonLIRF': pl.col('unm')&band&nonL,
 'R2b unm & ms 600-2400 & nonLIRF': pl.col('unm')&pl.col('ms').is_between(600,2400)&nonL,
 'R3 R2 & regular-matched flight': pl.col('unm')&band&nonL&reg,
 'R3b R3 & no prev-day occurrence': pl.col('unm')&band&nonL&reg&~pl.col('prev_day'),
 'R4 R2 & stand gap>=24h no dep between': pl.col('unm')&band&nonL&(pl.col('gap_h')>=24)&~pl.col('dep_between'),
 'R5 R2 & apt in LFPG,EGLL,EDDF,EDDM,LEBL,LEMD': pl.col('unm')&band&pl.col('APT').is_in(['LFPG','EGLL','EDDF','EDDM','LEBL','LEMD']),
 'R6 R2 & LFPG': pl.col('unm')&band&(pl.col('APT')=='LFPG'),
 'R7 unm & ms 900-2100 & LSZH': pl.col('unm')&band&(pl.col('APT')=='LSZH'),
}
BASE=1000.0
rows=[]
for name,r in rules.items():
    t=tr.filter(r).with_columns(tail=pl.col('y')>14400, roll=pl.col('y')>72000)
    k=rk.filter(r&(pl.col('PHASE_mvt')=='DEP'))
    n=t.height; nt=int(t['tail'].sum()); nr=int(t['roll'].sum())
    # LOMO soft shift: s_m = (sum over other months of max(y-BASE,0) on tail rows)/(rows other months)
    y=t['y'].to_numpy().astype(float); m=t['mon'].to_numpy(); tail=t['tail'].to_numpy()
    D=np.where(tail,y-BASE,0.0)
    dsse=0.0
    for mm in range(1,13):
        o=m!=mm; i=m==mm
        if o.sum()==0 or i.sum()==0: continue
        s=D[o].sum()/o.sum()
        dsse+= (i.sum()*s*s - 2*s*D[i].sum())
    s_all=D.sum()/max(n,1)
    # beta(0.5,0.5) posterior 90% interval for tail rate
    from scipy.stats import beta
    lo,hi=beta.ppf([0.05,0.95],nt+0.5,n-nt+0.5) if n>0 else (np.nan,np.nan)
    rows.append(dict(rule=name,train_n=n,tail4h=nt,roll20h=nr,p_tail=nt/max(n,1),p90lo=lo,p90hi=hi,shift_s=s_all,LOMO_dSSE_train=dsse,
                     rank_dep_n=k.height,rank_jan=int((k['mon']==1).sum()),rank_jul=int((k['mon']==7).sum())))
res=pl.DataFrame(rows)
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(50)
print(res)
res.write_csv(O+'s13_rules.csv')
# list the 2026 ranking DEP rows of the narrow rules
for name in ['R3b R3 & no prev-day occurrence','R4 R2 & stand gap>=24h no dep between','R6 R2 & LFPG']:
    k=rk.filter(rules[name]&(pl.col('PHASE_mvt')=='DEP'))
    print('==',name,k.height)
    print(k.select('MVT_ID_mvt','APT','FLIGHT_mvt','AIRCRAFT_TYPE_mvt','STAND_mvt',pl.col('SCHED_TIME_UTC_mvt').dt.strftime('%Y-%m-%d %H:%M'),pl.col('MVT_TIME_UTC_mvt').dt.strftime('%m-%d %H:%M:%S'),'ms','n_occ','other_match_share','prev_day','gap_h','dep_between').sort('SCHED_TIME_UTC_mvt').to_pandas().to_string(index=False))
# training positives under each rule
print(tr.filter(pl.col('unm')&(pl.col('y')>14400)&(pl.col('ms').abs()<7200)).select('APT','FLIGHT_mvt','ms','y','n_occ','other_match_share','prev_day','gap_h','dep_between').to_pandas().to_string(index=False))

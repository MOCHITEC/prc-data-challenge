# COMPLIANCE-SENSITIVE MEASUREMENT: for id-adjacent DEP pairs where the candidate time keys disagree in order,
# which key is non-decreasing? (tells whether MVT_ID order encodes BLOCK)
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(260)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP').with_columns(rank=pl.col('file')=='ranking.parquet')
df=df.with_columns(tm=pl.col('MVT_TIME_UTC_mvt').dt.epoch('s'), ts=pl.col('SCHED_TIME_UTC_mvt').dt.epoch('s'), tb=pl.col('BLOCK_TIME_UTC_mvt').dt.epoch('s'),
                   ta=pl.col('AOBT_3_flt').dt.epoch('s'), tl=pl.col('LOBT_flt').dt.epoch('s'))
df=df.sort('ADEP_mvt','MVT_ID_mvt')
d=df.with_columns([(pl.col(c)-pl.col(c).shift(1)).over('ADEP_mvt').alias('d'+c) for c in ['tm','ts','tb','ta','tl']]+[pl.col('MVT_ID_mvt').diff().over('ADEP_mvt').alias('gid')]).filter(pl.col('gid')<=50, ~pl.col('rank'))
# only "local" pairs: within 6h on all keys present (exclude group boundaries)
d=d.filter(pl.col('dtm').abs()<6*3600, pl.col('dts').abs()<6*3600)
res=[]
for a in ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']:
    x=d.filter(pl.col('ADEP_mvt')==a)
    def share(cond, key): 
        y=x.filter(cond); return len(y), (round(float((y[key]>=0).mean()),3) if len(y) else None)
    n1,bm=share((pl.col('dtb')>=0)!=(pl.col('dtm')>=0),'dtb')   # BLOCK vs MVT disagree: share where BLOCK is the nondecreasing one
    n2,bs=share((pl.col('dtb')>=0)!=(pl.col('dts')>=0),'dtb')
    n3,ms=share((pl.col('dtm')>=0)!=(pl.col('dts')>=0),'dtm')
    n4,ba=share(((pl.col('dtb')>=0)!=(pl.col('dta')>=0)) & pl.col('dta').is_not_null(),'dtb')
    n5,bl=share(((pl.col('dtb')>=0)!=(pl.col('dtl')>=0)) & pl.col('dtl').is_not_null(),'dtb')
    res.append(dict(ADEP=a,n_local=len(x),nd_blk=round(float((x['dtb']>=0).mean()),3),nd_mvt=round(float((x['dtm']>=0).mean()),3),nd_sch=round(float((x['dts']>=0).mean()),3),
        B_vs_M_n=n1,B_wins_vs_M=bm,B_vs_S_n=n2,B_wins_vs_S=bs,M_vs_S_n=n3,M_wins_vs_S=ms,B_vs_AOBT_n=n4,B_wins_vs_AOBT=ba,B_vs_LOBT_n=n5,B_wins_vs_LOBT=bl))
print(pl.DataFrame(res))

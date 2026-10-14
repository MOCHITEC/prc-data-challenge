# "Duplicate MVT" fingerprint: another row (same airport, same phase, same normalized flight no.) with MVT within 600 s.
# Visible at prediction time for DEP (MVT is not blanked). Tail rates among rows that have such a twin.
import polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
def load(name):
    df=pl.read_parquet(O+f'{name}.parquet').with_columns(
        key=pl.col('FLIGHT_mvt').str.slice(0,2)+pl.lit('|')+pl.col('FLIGHT_mvt').str.extract(r'(\d+)',1).str.strip_chars_start('0'),
        unm=pl.col('FLIGHT_ID_mvt').is_null(),
        ms=(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds())
    if name=='train':
        df=df.with_columns(y=pl.when(pl.col('PHASE_mvt')=='DEP').then((pl.col('MVT_TIME_UTC_mvt')-pl.col('BLOCK_TIME_UTC_mvt')).dt.total_seconds())
              .otherwise((pl.col('BLOCK_TIME_UTC_mvt')-pl.col('MVT_TIME_UTC_mvt')).dt.total_seconds()))
    return df
for name in ['train','rank']:
    df=load(name).filter(pl.col('key').is_not_null())
    a=df.select('MVT_ID_mvt','APT','PHASE_mvt','key','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt').sort('MVT_TIME_UTC_mvt')
    # nearest other row of same (APT,PHASE,key) by MVT, both directions
    b=a.rename({'MVT_ID_mvt':'id2','MVT_TIME_UTC_mvt':'t2','SCHED_TIME_UTC_mvt':'s2'})
    out=[]
    for strat in ['backward','forward']:
        j=a.join_asof(b.with_columns(tt=pl.col('t2')),left_on='MVT_TIME_UTC_mvt',right_on='tt',by=['APT','PHASE_mvt','key'],strategy=strat,allow_exact_matches=True)
        out.append(j)
    # exact matches allowed may hit self; redo excluding self by shifting
    a2=a.with_columns(idx=pl.int_range(pl.len()))
    res=[]
    for strat,off in [('backward',-1),('forward',1)]:
        b2=a.rename({'MVT_ID_mvt':'id2','MVT_TIME_UTC_mvt':'t2','SCHED_TIME_UTC_mvt':'s2'})
        g=a.sort('APT','PHASE_mvt','key','MVT_TIME_UTC_mvt').with_columns(
            pl.col('MVT_TIME_UTC_mvt').shift(-off).over(['APT','PHASE_mvt','key']).alias('t2'),
            pl.col('SCHED_TIME_UTC_mvt').shift(-off).over(['APT','PHASE_mvt','key']).alias('s2'),
            pl.col('MVT_ID_mvt').shift(-off).over(['APT','PHASE_mvt','key']).alias('id2'))
        res.append(g.select('MVT_ID_mvt',pl.col('t2').alias(f't2{strat[0]}'),pl.col('s2').alias(f's2{strat[0]}')))
    f=res[0].join(res[1],on='MVT_ID_mvt')
    df=df.join(f,on='MVT_ID_mvt',how='left').with_columns(
        dprev=(pl.col('MVT_TIME_UTC_mvt')-pl.col('t2b')).dt.total_seconds(), dnext=(pl.col('t2f')-pl.col('MVT_TIME_UTC_mvt')).dt.total_seconds(),
        sdiff_prev=(pl.col('SCHED_TIME_UTC_mvt')-pl.col('s2b')).dt.total_seconds(), sdiff_next=(pl.col('s2f')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds())
    df=df.with_columns(twin=((pl.col('dprev')<=600)&(pl.col('sdiff_prev').abs()>=43200))|((pl.col('dnext')<=600)&(pl.col('sdiff_next').abs()>=43200)))
    df.select('MVT_ID_mvt','twin','dprev','dnext','sdiff_prev','sdiff_next').write_parquet(O+f'twin_{name}.parquet')
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(300)
    if name=='train':
        print(df.filter(pl.col('twin')).group_by('PHASE_mvt','APT','unm').agg(pl.len(),(pl.col('y')>14400).sum().alias('tail4h'),(pl.col('y')>72000).sum().alias('roll')).sort('PHASE_mvt','APT','unm'))
        tl=df.filter((pl.col('y')>72000))
        print(tl.select('PHASE_mvt','APT','FLIGHT_mvt','unm','y','ms','twin','dprev','dnext','sdiff_prev','sdiff_next').to_pandas().to_string(index=False))
    else:
        print(df.filter(pl.col('twin')).group_by('PHASE_mvt','APT','unm').agg(pl.len()).sort('PHASE_mvt','APT','unm'))

import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP')
df=df.with_columns(train=(pl.col('file')!='ranking.parquet'),
  y=pl.col('TAXITIME_SEC_mvt'),
  ms=(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
  bs=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
  unm=pl.col('FLIGHT_ID_mvt').is_null())
df=df.with_columns(cls=pl.when(~pl.col('train')).then(pl.lit('RANK'))
  .when(pl.col('y')>72000).then(pl.lit('R'))
  .when(pl.col('unm') & (pl.col('bs').abs()<=6)).then(pl.lit('SF'))
  .when(pl.col('unm')).then(pl.lit('UN')).otherwise(pl.lit('M')))
tcols=['MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','AOBT_3_flt','LOBT_flt','IOBT_flt','EOBT_1_flt','ARVT_3_flt']
ex=[]
for c in tcols:
    s=pl.col(c)
    ex+= [ (s.dt.microsecond()!=0).alias(c[:5]+'_us'),
           ((s.dt.second()+30)%60-30).alias(c[:5]+'_off') ]
df=df.with_columns(ex)
# 1. microsecond nonzero anywhere?
print('nonzero microseconds per column:',{c[:5]:int(df[c[:5]+'_us'].sum()) for c in tcols})
# 2. per ADEP x cls: share of |off|<=6, share off==0 for each col
agg=[pl.len().alias('n')]
for c in ['MVT_T','BLOCK','SCHED','AOBT_','LOBT_','IOBT_','EOBT_']:
    agg+=[(pl.col(c+'_off').abs()<=6).mean().round(3).alias(c+'_in6'), (pl.col(c+'_off')==0).mean().round(3).alias(c+'_s0')]
print(df.group_by('ADEP_mvt','cls').agg(agg).sort('ADEP_mvt','cls'))
# 3. rollover rows detail
print(df.filter(pl.col('cls')=='R').select('ADEP_mvt','FLIGHT_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','MVT_T_off','BLOCK_off','y','ms'))
# 4. LIRF: distribution of MVT offset and BLOCK offset by class (train) and MVT offset rank unmatched
for a in ['LIRF']:
    sub=df.filter(pl.col('ADEP_mvt')==a)
    print(a,'MVT off hist by cls')
    print(sub.group_by('cls',pl.col('MVT_T_off')).len().pivot(on='cls',index='MVT_T_off',values='len').sort('MVT_T_off'))
    print(a,'BLOCK off hist by cls')
    print(sub.filter(pl.col('train')).group_by('cls',pl.col('BLOCK_off')).len().pivot(on='cls',index='BLOCK_off',values='len').sort('BLOCK_off'))
    # joint of y mod 60 offset = MVT_off - BLOCK_off
    print(sub.filter(pl.col('train')).group_by((pl.col('MVT_T_off')-pl.col('BLOCK_off')).alias('d')).len().sort('d'))

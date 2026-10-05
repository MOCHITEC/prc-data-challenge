import polars as pl, numpy as np, glob
R='/home/user/prc-data-challenge/data/raw/'
o=pl.read_parquet('/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/oof_matched_janjul.parquet')
tr=pl.concat([pl.read_parquet(f,columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt','PHASE_mvt']) for f in [R+'training_2025-01-01_2025-02-01.parquet',R+'training_2025-07-01_2025-08-01.parquet']])
tr=tr.with_columns(pl.col('MVT_TIME_UTC_mvt').cast(pl.Datetime('us')) if tr['MVT_TIME_UTC_mvt'].dtype!=pl.Datetime else pl.col('MVT_TIME_UTC_mvt'))
d=o.join(tr.filter(pl.col('PHASE_mvt')=='DEP'),on='MVT_ID_mvt',how='left').with_columns(pl.col('MVT_TIME_UTC_mvt').dt.date().alias('date'),pl.col('MVT_TIME_UTC_mvt').dt.hour().alias('hr'))
print(d.shape, d['date'].null_count())
for g in [True,None]:
    x=d if g is None else d.filter(pl.col('genuine'))
    mse=(x['err']**2).mean()
    for keys in [['ADEP_mvt','date'],['ADEP_mvt','date','hr']]:
        m=x.with_columns(pl.col('err').mean().over(keys).alias('mu'))
        resid=((m['err']-m['mu'])**2).mean()
        print('genuine' if g else 'all', keys, 'rmse',round(mse**.5,1),'after removing group mean',round(resid**.5,1))
# per airport daily-mean residual SD (genuine)
x=d.filter(pl.col('genuine'))
day=x.group_by('ADEP_mvt','date').agg(pl.col('err').mean().alias('mu'),pl.len().alias('n'))
print(day.group_by('ADEP_mvt').agg(pl.col('mu').std().alias('sd_daymean'),pl.col('mu').abs().max().alias('max'),pl.col('n').mean()).sort('ADEP_mvt'))
day.write_parquet('daymean.parquet')
# join PRU features
dly=pl.read_parquet('apt_dly.parquet'); sa=pl.read_parquet('slot.parquet'); atc=pl.read_parquet('atc.parquet'); tr2=None
f=dly.select('APT_ICAO','date',pl.col('DLY_APT_ARR_1').cast(pl.Float64).alias('arr_dly'),pl.col('DLY_APT_ARR_W_1').cast(pl.Float64).alias('arr_dly_W'),pl.col('FLT_ARR_1').cast(pl.Float64).alias('arr_n'))
f=f.join(sa.select('APT_ICAO','date',pl.col('FLT_DEP_1').cast(pl.Float64).alias('dep_n'),pl.col('FLT_DEP_REG_1').cast(pl.Float64).alias('dep_reg'),pl.col('FLT_DEP_OUT_LATE_1').cast(pl.Float64).alias('late'),pl.col('FLT_DEP_OUT_EARLY_1').cast(pl.Float64).alias('early')),on=['APT_ICAO','date'],how='left')
f=f.join(atc.select('APT_ICAO','date',pl.col('FLT_DEP_IFR_2').cast(pl.Float64).alias('apdf_n'),pl.col('DLY_ATC_PRE_2').cast(pl.Float64).alias('atcpre')),on=['APT_ICAO','date'],how='left')
f=f.with_columns((pl.col('arr_dly')/pl.col('arr_n')).alias('arr_dly_pf'),(pl.col('dep_reg')/pl.col('dep_n')).alias('reg_share'),(pl.col('late')/pl.col('dep_reg')).alias('late_share'),(pl.col('atcpre')/pl.col('apdf_n')).alias('atcpre_pf'))
f.write_parquet('pru_daily.parquet')
j=day.join(f,left_on=['ADEP_mvt','date'],right_on=['APT_ICAO','date'],how='left')
print('joined null arr', j['arr_dly'].null_count(), len(j))
cols=['arr_dly_pf','reg_share','late_share','atcpre_pf','dep_n']
# within-airport correlations (demeaned)
jj=j.with_columns([(pl.col(c)-pl.col(c).mean().over('ADEP_mvt')).alias(c) for c in cols+['mu']])
for c in cols:
    s=jj.select('mu',c).drop_nulls(); print(c, 'within-apt corr with daily mean resid', round(np.corrcoef(s['mu'],s[c])[0,1],3), len(s))
for a in sorted(j['ADEP_mvt'].unique()):
    s=j.filter(pl.col('ADEP_mvt')==a)
    print(a, {c: round(np.corrcoef(*s.select('mu',c).drop_nulls().to_numpy().T)[0,1],2) if s[c].drop_nulls().len()>5 else None for c in cols})

import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP', pl.col('file')!='ranking.parquet')
df=df.with_columns(y=pl.col('TAXITIME_SEC_mvt'),
  bs=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
  unm=pl.col('FLIGHT_ID_mvt').is_null(),
  mo=((pl.col('MVT_TIME_UTC_mvt').dt.second()+30)%60-30),
  bo=((pl.col('BLOCK_TIME_UTC_mvt').dt.second()+30)%60-30))
df=df.with_columns(cls=pl.when(pl.col('y')>72000).then(pl.lit('R'))
  .when(pl.col('unm') & (pl.col('bs').abs()<=6)).then(pl.lit('SF'))
  .when(pl.col('unm')).then(pl.lit('UN')).otherwise(pl.lit('M')), ssum=pl.col('mo')+pl.col('bo'))
cl=['EDDM','EGLL','LFPG','LIRF','LTFM']
sub=df.filter(pl.col('ADEP_mvt').is_in(cl))
print('share of rows with mo+bo in {-1,0}, by ADEP x cls')
print(sub.group_by('ADEP_mvt','cls').agg(pl.len(), pl.col('ssum').is_in([-1,0]).mean().round(4).alias('sum_in_m1_0'),
   (pl.col('ssum')==0).mean().round(4).alias('sum0'),(pl.col('ssum')==-1).mean().round(4).alias('summ1')).sort('ADEP_mvt','cls'))
print('LIRF M: histogram of mo+bo')
print(sub.filter(pl.col('ADEP_mvt')=='LIRF').group_by('cls','ssum').len().pivot(on='cls',index='ssum',values='len').sort('ssum'))
# As function of y: maybe sum relates to y? check y mod 60 vs sum on LIRF M
s=sub.filter(pl.col('ADEP_mvt')=='LIRF',pl.col('cls')=='M').with_columns(ym=((pl.col('y')+30)%60-30))
print(s.group_by('ym').agg(pl.len(),pl.col('ssum').is_in([-1,0]).mean()).sort('ym'))

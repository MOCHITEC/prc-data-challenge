# COMPLIANCE-SENSITIVE MEASUREMENT: at LFPG (and others), for rows where BLOCK and SCHED differ by > 1 h, does the row's
# MVT_ID position bracket its BLOCK or its SCHED (using the id-neighbours' BLOCK and SCHED as reference)?
import polars as pl
pl.Config.set_tbl_rows(40)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP',pl.col('file')!='ranking.parquet').sort('ADEP_mvt','MVT_ID_mvt')
e=lambda c: pl.col(c).dt.epoch('s')
df=df.with_columns(ts=e('SCHED_TIME_UTC_mvt'),tb=e('BLOCK_TIME_UTC_mvt'),tm=e('MVT_TIME_UTC_mvt'))
df=df.with_columns(
  pts=pl.col('ts').shift(1).over('ADEP_mvt'), nts=pl.col('ts').shift(-1).over('ADEP_mvt'),
  ptb=pl.col('tb').shift(1).over('ADEP_mvt'), ntb=pl.col('tb').shift(-1).over('ADEP_mvt'),
  pg=pl.col('MVT_ID_mvt').diff().over('ADEP_mvt'), ng=(-pl.col('MVT_ID_mvt').diff(-1)).over('ADEP_mvt'))
x=df.filter((pl.col('tb')-pl.col('ts')).abs()>3600, pl.col('pg')==1, pl.col('ng')==1)
# neighbour reference: midpoint of neighbours' BLOCK (= their off-block, near their SCHED for normal rows)
x=x.with_columns(lo=pl.min_horizontal('ptb','pts'), hi=pl.max_horizontal('ntb','nts'))
x=x.with_columns(B_in=(pl.col('tb')>=pl.col('lo')-1800)&(pl.col('tb')<=pl.col('hi')+1800),
                 S_in=(pl.col('ts')>=pl.col('lo')-1800)&(pl.col('ts')<=pl.col('hi')+1800),
                 M_in=(pl.col('tm')>=pl.col('lo')-1800)&(pl.col('tm')<=pl.col('hi')+1800),
                 span_h=(pl.col('hi')-pl.col('lo'))/3600)
print(x.group_by('ADEP_mvt').agg(pl.len(),pl.col('B_in').mean().round(3),pl.col('S_in').mean().round(3),pl.col('M_in').mean().round(3),
   (pl.col('B_in')&~pl.col('S_in')).sum().alias('B_only'),(pl.col('S_in')&~pl.col('B_in')).sum().alias('S_only'),pl.col('span_h').median()).sort('ADEP_mvt'))

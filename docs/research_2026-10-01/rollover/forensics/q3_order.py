# What is the row order of each raw file?
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet', columns=['file','rowpos','MVT_ID_mvt','FLIGHT_ID_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt'])
df=df.sort('file','rowpos')
def mono(e): return (e.diff()>=0).mean()
print(df.group_by('file').agg(pl.len(),
   mono(pl.col('MVT_ID_mvt')).alias('id_nondec'),
   mono(pl.col('MVT_TIME_UTC_mvt').cast(pl.Int64)).alias('mvt_nondec'),
   mono(pl.col('SCHED_TIME_UTC_mvt').cast(pl.Int64)).alias('sched_nondec'),
   mono(pl.col('BLOCK_TIME_UTC_mvt').cast(pl.Int64)).alias('block_nondec'),
   (pl.col('ADEP_mvt')!=pl.col('ADEP_mvt').shift(1)).sum().alias('adep_changes'),
   (pl.col('PHASE_mvt')!=pl.col('PHASE_mvt').shift(1)).sum().alias('phase_changes'),
   pl.col('MVT_ID_mvt').is_null().sum().alias('id_null'), pl.col('MVT_ID_mvt').n_unique().alias('id_nuniq')).sort('file'))
f=df.filter(pl.col('file')=='training_2025-07-01')
print(f.head(30)); print(f.slice(100000,20))

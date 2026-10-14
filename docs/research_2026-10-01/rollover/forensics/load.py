# Build compact parquet of training+ranking rows (DEP+ARR) with file row index
import polars as pl, glob, os
D='/home/user/prc-data-challenge/data/raw/'
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
cols=['MVT_ID_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt',
 'SCHED_TIME_UTC_mvt','AIRCRAFT_TYPE_mvt','RUNWAY_mvt','STAND_mvt','TAXITIME_SEC_mvt','LOBT_flt','IOBT_flt','EOBT_1_flt','AOBT_3_flt','ARVT_3_flt']
fr=[]
files=sorted(glob.glob(D+'training_*.parquet'))+[D+'ranking.parquet']
for f in files:
    lf=pl.scan_parquet(f).select(cols).with_row_index('rowpos').with_columns(pl.lit(os.path.basename(f)[:19]).alias('file'))
    fr.append(lf.collect())
df=pl.concat(fr)
df=df.with_columns(pl.col('rowpos').cast(pl.Int64))
df.write_parquet(OUT+'all.parquet')
print(df.shape, df.estimated_size()/1e6)
print(df.group_by('file','PHASE_mvt').len().sort('file','PHASE_mvt'))

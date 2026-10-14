import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(260)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP',pl.col('file')!='ranking.parquet').sort('ADEP_mvt','MVT_ID_mvt')
for fl,a in [('EJU983W','LFPG'),('NJE389D','LSZH')]:
    g=df.filter(pl.col('ADEP_mvt')==a).with_row_index('i').with_columns(pl.col('i').cast(pl.Int64))
    i=g.filter(pl.col('FLIGHT_mvt')==fl, pl.col('TAXITIME_SEC_mvt')>72000)['i'][0]
    print(g.filter((pl.col('i')-i).abs()<=6).select('MVT_ID_mvt','FLIGHT_mvt','ADES_mvt','FLIGHT_ID_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','TAXITIME_SEC_mvt'))
# LFPG ranking: id-adjacent pair structure
r=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP',pl.col('file')=='ranking.parquet',pl.col('ADEP_mvt')=='LFPG').sort('MVT_ID_mvt')
print(r.select('MVT_ID_mvt','FLIGHT_mvt','ADES_mvt','SCHED_TIME_UTC_mvt','MVT_TIME_UTC_mvt').slice(3000,25))

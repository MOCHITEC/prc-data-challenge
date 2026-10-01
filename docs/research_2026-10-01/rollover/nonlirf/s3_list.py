# List all non-LIRF tail rows (|y|>4h DEP/ARR) + LIRF ARR >20h, with every clock expressed relative to BLOCK
import polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
df=pl.read_parquet(O+'train.parquet').with_columns(
    y=pl.when(pl.col('PHASE_mvt')=='DEP').then((pl.col('MVT_TIME_UTC_mvt')-pl.col('BLOCK_TIME_UTC_mvt')).dt.total_seconds())
      .otherwise((pl.col('BLOCK_TIME_UTC_mvt')-pl.col('MVT_TIME_UTC_mvt')).dt.total_seconds()))
clocks={'SCHED':'SCHED_TIME_UTC_mvt','MVT':'MVT_TIME_UTC_mvt','AOBT3':'AOBT_3_flt','EOBT1':'EOBT_1_flt','LOBT':'LOBT_flt','ARVT1':'ARVT_1_flt','ARVT3':'ARVT_3_flt'}
t=df.filter(((pl.col('APT')!='LIRF')&(pl.col('y').abs()>14400))|((pl.col('APT')=='LIRF')&(pl.col('PHASE_mvt')=='ARR')&(pl.col('y')>72000)))
t=t.with_columns([(pl.col('BLOCK_TIME_UTC_mvt')-pl.col(c)).dt.total_seconds().alias('B-'+k) for k,c in clocks.items()])
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(400); pl.Config.set_tbl_cols(40); pl.Config.set_fmt_str_lengths(30)
print(t.select('PHASE_mvt','APT','ADEP_mvt','ADES_mvt','FLIGHT_mvt','CALLSIGN_flt','MARKET_SEGMENT_flt','AIRCRAFT_TYPE_mvt','STAND_mvt','RUNWAY_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','y',*['B-'+k for k in clocks]).sort('PHASE_mvt','APT','y'))

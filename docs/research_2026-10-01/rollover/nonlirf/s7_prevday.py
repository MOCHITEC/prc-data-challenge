# Item 3: previous-day context for EJU983W (LFPG 2025-01-19), EJU42AY (LFPG 01-23), NJE389D (LSZH 05-14)
import polars as pl
from datetime import datetime, timezone
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
df=pl.read_parquet(O+'train.parquet').with_columns(
    y=pl.when(pl.col('PHASE_mvt')=='DEP').then((pl.col('MVT_TIME_UTC_mvt')-pl.col('BLOCK_TIME_UTC_mvt')).dt.total_seconds())
      .otherwise((pl.col('BLOCK_TIME_UTC_mvt')-pl.col('MVT_TIME_UTC_mvt')).dt.total_seconds()))
sel=['PHASE_mvt','APT','ADEP_mvt','ADES_mvt','FLIGHT_mvt','CALLSIGN_flt','AIRCRAFT_TYPE_mvt','STAND_mvt','RUNWAY_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','y']
def show(x,title):
    print('---',title,x.height)
    x=x.select(sel).sort('BLOCK_TIME_UTC_mvt').with_columns([pl.col(c).dt.strftime('%m-%d %H:%M:%S') for c in ['SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt']])
    print(x.to_pandas().to_string(index=False))
def T(s): return pl.lit(datetime.fromisoformat(s).replace(tzinfo=timezone.utc))
B=pl.col('BLOCK_TIME_UTC_mvt')
show(df.filter((pl.col('APT')=='LFPG')&(pl.col('STAND_mvt')=='C03')&(B>=T('2025-01-18 06:00'))&(B<=T('2025-01-20 06:00'))),'LFPG stand C03 01-18..01-20')
show(df.filter((pl.col('APT')=='LFPG')&(pl.col('PHASE_mvt')=='ARR')&pl.col('FLIGHT_mvt').str.starts_with('EJU')&(B>=T('2025-01-18 12:00'))&(B<=T('2025-01-19 20:00'))),'LFPG EJU ARR 01-18 12:00..01-19 20:00')
show(df.filter((pl.col('APT')=='LFPG')&(pl.col('PHASE_mvt')=='DEP')&pl.col('FLIGHT_mvt').str.starts_with('EJU')&(B>=T('2025-01-18 17:00'))&(B<=T('2025-01-18 23:59'))),'LFPG EJU DEP 01-18 17:00..24:00')
show(df.filter((pl.col('APT')=='LFPG')&(pl.col('STAND_mvt')=='D06')&(B>=T('2025-01-22 12:00'))&(B<=T('2025-01-23 18:00'))),'LFPG stand D06 01-22..01-23')
show(df.filter((pl.col('APT')=='LSZH')&(pl.col('STAND_mvt')=='103')&(B>=T('2025-05-12 12:00'))&(B<=T('2025-05-15 00:00'))),'LSZH stand 103 05-12..05-15')
show(df.filter((pl.col('APT')=='LSZH')&pl.col('FLIGHT_mvt').str.starts_with('NJE')&(B>=T('2025-05-12 00:00'))&(B<=T('2025-05-15 23:00'))),'LSZH NJE rows 05-12..05-15')
show(df.filter(pl.col('FLIGHT_mvt').str.starts_with('NJE389')),'all NJE389*')

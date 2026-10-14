# For each anomalous row: any training row (same airport, DEP or ARR) whose MVT/BLOCK/SCHED is within 180 s of the anomalous timestamp,
# plus same FLIGHT/callsign rows within +-8 days.
import polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
df=pl.read_parquet(O+'train.parquet').with_columns(
    y=pl.when(pl.col('PHASE_mvt')=='DEP').then((pl.col('MVT_TIME_UTC_mvt')-pl.col('BLOCK_TIME_UTC_mvt')).dt.total_seconds())
      .otherwise((pl.col('BLOCK_TIME_UTC_mvt')-pl.col('MVT_TIME_UTC_mvt')).dt.total_seconds()))
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(400); pl.Config.set_tbl_cols(40); pl.Config.set_fmt_str_lengths(30)
from datetime import datetime, timezone
def T(s): return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
targets=[('LSZH','NJE389D','2025-05-13 14:39:58'),('LFPG','EJU983W','2025-01-18 19:30:00'),('LFPG','EJU42AY','2025-01-22 22:58:57'),
         ('LSZH','HBCYH','2025-02-08 10:51:47'),('LIRF','IBEE0657','2025-04-23 21:58:57'),('LIRF','WMTMT993','2025-05-18 22:40:58'),('LSZH','JPV291','2025-08-12 10:29:01')]
sel=['PHASE_mvt','APT','ADEP_mvt','ADES_mvt','FLIGHT_mvt','CALLSIGN_flt','AIRCRAFT_TYPE_mvt','STAND_mvt','RUNWAY_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','y']
def show(x):
    x=x.select(sel).sort('MVT_TIME_UTC_mvt').with_columns([pl.col(c).dt.strftime('%m-%d %H:%M:%S') for c in ['SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt']]).rename({'SCHED_TIME_UTC_mvt':'SCHED','BLOCK_TIME_UTC_mvt':'BLOCK','MVT_TIME_UTC_mvt':'MVT','AIRCRAFT_TYPE_mvt':'AC','STAND_mvt':'STD','RUNWAY_mvt':'RWY','PHASE_mvt':'PH'})
    print(x.to_pandas().to_string(index=False))
for apt,fl,ts in targets:
    t=T(ts)
    print('=====',apt,fl,ts)
    m=df.filter((pl.col('APT')==apt)&pl.any_horizontal([((pl.col(c)-pl.lit(t)).dt.total_seconds().abs()<=180) for c in ['MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt']]))
    print('rows at same airport with a clock within 180 s of anomalous ts:'); show(m)
    core=fl.rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ') if not fl.startswith('HB') else fl
    m2=df.filter((pl.col('FLIGHT_mvt').str.contains(fl[:3]+'.*'+''.join(ch for ch in fl[3:] if ch.isdigit()) if any(ch.isdigit() for ch in fl) else fl)) &
                 ((pl.col('MVT_TIME_UTC_mvt')-pl.lit(t)).dt.total_seconds().abs()<=8*86400))
    print('same-flight-string-ish rows +-8 days (any airport):'); show(m2)

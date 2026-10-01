# Visible stand-occupancy feature: for each DEP row, last ARR in-block at the same (airport, stand) before the DEP's MVT,
# and whether another DEP (by MVT) left that stand in between. Uses only columns visible in ranking (ARR BLOCK, DEP MVT).
import polars as pl, sys
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
def build(name):
    df=pl.read_parquet(O+f'{name}.parquet').filter(pl.col('STAND_mvt').is_not_null())
    arr=df.filter(pl.col('PHASE_mvt')=='ARR').select('APT','STAND_mvt',pl.col('BLOCK_TIME_UTC_mvt').alias('t_arr'),pl.col('FLIGHT_mvt').alias('arr_flight'),pl.col('AIRCRAFT_TYPE_mvt').alias('arr_ac')).drop_nulls('t_arr').sort('t_arr')
    dep=df.filter(pl.col('PHASE_mvt')=='DEP').with_columns(t=pl.col('MVT_TIME_UTC_mvt')-pl.duration(minutes=3)).sort('t')
    other=df.filter(pl.col('PHASE_mvt')=='DEP').select('APT','STAND_mvt',(pl.col('MVT_TIME_UTC_mvt')-pl.duration(minutes=3)).alias('t_prevdep'),pl.col('MVT_ID_mvt').alias('pid')).sort('t_prevdep')
    j=dep.join_asof(arr,left_on='t',right_on='t_arr',by=['APT','STAND_mvt'],strategy='backward')
    # previous DEP from same stand strictly before this DEP
    j=j.sort('t').join_asof(other.with_columns(t2=pl.col('t_prevdep')+pl.duration(seconds=1)).sort('t2'),left_on='t',right_on='t2',by=['APT','STAND_mvt'],strategy='backward',allow_exact_matches=False)
    j=j.with_columns(gap_h=(pl.col('MVT_TIME_UTC_mvt')-pl.col('t_arr')).dt.total_seconds()/3600,
                     dep_between=(pl.col('t_prevdep')>pl.col('t_arr')).fill_null(False))
    return j
for name in ['train','rank']:
    j=build(name)
    j.select('MVT_ID_mvt','gap_h','dep_between','arr_flight','arr_ac','t_arr').write_parquet(O+f'standgap_{name}.parquet')
    print(name,j.height,'gap null',j['gap_h'].is_null().sum())

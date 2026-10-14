# LFPG/LSZH/LIRF DEP rows with date(BLOCK) < date(SCHED): dates, y, unmatched, BLOCK-SCHED
import polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
dep=pl.read_parquet(O+'dep_train_feat.parquet').join(pl.read_parquet(O+'standgap_train.parquet'),on='MVT_ID_mvt',how='left')
dep=dep.with_columns(dSB=(pl.col('SCHED_TIME_UTC_mvt').dt.date()-pl.col('BLOCK_TIME_UTC_mvt').dt.date()).dt.total_days(),
                     BmS=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds())
x=dep.filter((pl.col('dSB')>=1)&(pl.col('APT').is_in(['LFPG','LSZH','LIRF'])))
print(x.select('APT','FLIGHT_mvt','unm','STAND_mvt',pl.col('SCHED_TIME_UTC_mvt').dt.strftime('%m-%d %H:%M'),pl.col('BLOCK_TIME_UTC_mvt').dt.strftime('%m-%d %H:%M:%S'),'BmS','ms','y','gap_h','dep_between').sort('APT','SCHED_TIME_UTC_mvt').to_pandas().to_string(index=False))
# all DEP rows with BLOCK earlier than SCHED by > 6h, nonLIRF
z=dep.filter((pl.col('APT')!='LIRF')&(pl.col('BmS')< -6*3600))
print('nonLIRF DEP with BLOCK < SCHED-6h:',z.height); print(z.select('APT','FLIGHT_mvt','unm','BmS','ms','y').to_pandas().to_string(index=False))

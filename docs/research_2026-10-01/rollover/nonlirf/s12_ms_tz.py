# Is the unmatched ms offset a timezone artefact? ms - y (= BLOCK - SCHED) by airport x summer for unmatched vs matched
import polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
dep=pl.read_parquet(O+'dep_train_feat.parquet').with_columns(summer=pl.col('SCHED_TIME_UTC_mvt').dt.month().is_between(4,10),
   BmS=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds())
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(300)
t=dep.filter(pl.col('BmS').abs()<6*3600).group_by('APT','unm','summer').agg(pl.len(),pl.col('BmS').median().alias('BmS_med'),pl.col('y').median().alias('y_med'),
   ((pl.col('BmS')-3600).abs()<=900).mean().alias('BmS~1h'),((pl.col('BmS')-7200).abs()<=900).mean().alias('BmS~2h'),(pl.col('BmS').abs()<=900).mean().alias('BmS~0')).sort('APT','unm','summer')
print(t)

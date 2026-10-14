import polars as pl, numpy as np
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300)
d=pl.read_parquet('dep2.parquet').filter(pl.col('src')=='train')
# MVT_ID ordering vs times (compliance check)
for c in ['MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt']:
    x=d.select(pl.col('MVT_ID_mvt').rank(), pl.col(c).rank()).to_numpy()
    print(c,'spearman w/ MVT_ID', np.corrcoef(x[:,0],x[:,1])[0,1])
# within a month: is MVT_ID sorted by any time? check rollover rows' ID rank among same-month rows
d=d.with_columns(mon=pl.col('SCHED_TIME_UTC_mvt').dt.month())
d=d.sort('MVT_ID_mvt').with_columns(id_rank=pl.col('MVT_ID_mvt').rank().over('mon')/pl.len().over('mon'),
   mvt_rank=pl.col('MVT_TIME_UTC_mvt').rank().over('mon')/pl.len().over('mon'),
   blk_rank=pl.col('BLOCK_TIME_UTC_mvt').rank().over('mon')/pl.len().over('mon'),
   sch_rank=pl.col('SCHED_TIME_UTC_mvt').rank().over('mon')/pl.len().over('mon'))
print(d.filter(pl.col('cls')=='roll').select('FLIGHT_mvt','MVT_ID_mvt','id_rank','mvt_rank','blk_rank','sch_rank'))
print(d.select('MVT_ID_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt').head(5))
print(d.select(pl.col('MVT_ID_mvt').diff().describe()))

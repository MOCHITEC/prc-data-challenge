import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300)
d=pl.read_parquet('dep5.parquet'); a=pl.read_parquet('arr.parquet')
key=[pl.col('src'),pl.col('SCHED_TIME_UTC_mvt').dt.month().alias('m')]
print(d.group_by(key).agg(pl.len().alias('dep'),pl.col('unm').sum().alias('unm'),(pl.col('unm')&(pl.col('ms')>=28800)).sum().alias('unm_ms8h'),
   (pl.col('unm')&(pl.col('ms')>=7200)).sum().alias('unm_ms2h'),
   (pl.col('unm')&pl.col('glitch')).sum().alias('unm_glitch'),(~pl.col('unm')&pl.col('glitch')).sum().alias('mat_glitch'),
   (pl.col('unm')&(pl.col('cls')=='fill')).sum().alias('unm_fill'),(pl.col('unm')&(pl.col('cls')=='roll')).sum().alias('roll'),
   (~pl.col('unm')&(pl.col('cls')=='fill')).sum().alias('mat_fill')).sort('src','m'))
print(a.group_by(key).agg(pl.len().alias('arr'),pl.col('unm').sum().alias('unm'),(pl.col('unm')&(pl.col('cls')=='fill')).sum().alias('unm_fill'),
   (~pl.col('unm')&(pl.col('cls')=='fill')).sum().alias('mat_fill'),(pl.col('ms').abs()>=28800).sum().alias('absms8h')).sort('src','m'))

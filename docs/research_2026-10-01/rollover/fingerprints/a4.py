import polars as pl
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(30); pl.Config.set_tbl_width_chars(250)
d=pl.read_parquet('dep.parquet')
d=d.with_columns(cls=pl.when(pl.col('y')>72000).then(pl.lit('roll')).when((pl.col('y')-pl.col('ms')).abs()<=60).then(pl.lit('fill')).otherwise(pl.lit('norm')))
L=lambda c: pl.col(c).dt.convert_time_zone('Europe/Rome')
d=d.with_columns(sl=L('SCHED_TIME_UTC_mvt'),ml=L('MVT_TIME_UTC_mvt'),bl=L('BLOCK_TIME_UTC_mvt'))
d=d.with_columns(dcross_loc=(pl.col('ml')-pl.duration(seconds=900)).dt.date()!=pl.col('sl').dt.date(),
                 dcross_loc0=(pl.col('ml').dt.date()-pl.col('sl').dt.date()).dt.total_days(),
                 dcross_utc=(pl.col('MVT_TIME_UTC_mvt').dt.date()-pl.col('SCHED_TIME_UTC_mvt').dt.date()).dt.total_days(),
                 bl_eq_sl_date=pl.col('bl').dt.date()==pl.col('sl').dt.date())
t=d.filter(pl.col('src')=='train')
for nm,df in [('unmatched',t.filter(pl.col('unm'))),('matched',t.filter(~pl.col('unm')))]:
    print(nm); print(df.group_by(['cls','dcross_loc0']).agg(pl.len(), pl.col('bl_eq_sl_date').mean().alias('blk_date_eq_sched_date_local')).sort(['cls','dcross_loc0']))
# non-fill unmatched with local date crossing: list
nf=t.filter(pl.col('unm')&(pl.col('cls')!='fill')&(pl.col('dcross_loc0')!=0))
print(nf.select('FLIGHT_mvt','sl','bl','ml','ms','y','cls').sort('ms'))
# local time-of-day of BLOCK vs MVT for rollovers
r=t.filter(pl.col('cls')=='roll').with_columns(tod_gap=((pl.col('ml')-pl.col('bl')).dt.total_seconds()-86400))
print(r.select('FLIGHT_mvt','sl','bl','ml','tod_gap'))
d.write_parquet('dep2.parquet')

import polars as pl
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(30); pl.Config.set_tbl_width_chars(250)
d=pl.read_parquet('dep2.parquet')
u=d.filter(pl.col('unm'))
u=u.with_columns(mb=pl.col('ms').cut([3600,7200,14400,28800,43200,72000]))
t=u.filter(pl.col('src')=='train')
print(t.group_by(['mb','dcross_loc0']).agg(pl.len(),(pl.col('cls')=='roll').sum().alias('roll'),(pl.col('cls')=='fill').sum().alias('fill'),(pl.col('cls')=='norm').sum().alias('norm')).sort(['mb','dcross_loc0']))
print(u.filter(pl.col('src')=='rank').group_by(['mb','dcross_loc0']).len().sort(['mb','dcross_loc0']))
print(t.filter(pl.col('ms')>=43200).select('FLIGHT_mvt','scls','sl','ml','ms','y','cls','RUNWAY_mvt','STAND_mvt','ADES_mvt').sort('cls','ms'))

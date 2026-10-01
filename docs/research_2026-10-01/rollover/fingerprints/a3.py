import polars as pl
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(30); pl.Config.set_tbl_width_chars(250)
d=pl.read_parquet('dep.parquet')
d=d.with_columns(cls=pl.when(pl.col('y')>72000).then(pl.lit('roll')).when((pl.col('y')-pl.col('ms')).abs()<=60).then(pl.lit('fill')).otherwise(pl.lit('norm')),
  glitch=pl.col('scls').is_in(['dup1','dup2']))
t=d.filter((pl.col('src')=='train'))
print('ALL train LIRF DEP rows y>72000 incl matched:', t.filter(pl.col('y')>72000).select('FLIGHT_mvt','unm','scls','ms','y').sort('ms'))
u=t.filter(pl.col('unm'))
u=u.with_columns(mb=pl.col('ms').cut([7200,43200]))
print(u.group_by(['mb','cls']).agg(pl.len(),pl.col('glitch').sum().alias('glitch')).sort(['mb','cls']))
print(u.group_by(['mb','scls']).agg(pl.len(),(pl.col('cls')=='roll').sum().alias('roll'),(pl.col('cls')=='fill').sum().alias('fill')).sort(['mb','scls']))

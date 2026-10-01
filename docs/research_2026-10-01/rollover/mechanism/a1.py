import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(30); pl.Config.set_tbl_width_chars(250)
df=pl.read_parquet('train_ext.parquet')
print(df.group_by('PHASE_mvt','ADEP_mvt').len().sort('len'))
print(df.filter(pl.col('PHASE_mvt')=='ARR').group_by('ADES_mvt').len().sort('len'))

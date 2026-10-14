import polars as pl
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(30); pl.Config.set_tbl_width_chars(250)
d=pl.read_parquet('lirf.parquet').filter(pl.col('PHASE_mvt')=='DEP')
d=d.with_columns(ms=(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),y=pl.col('TAXITIME_SEC_mvt'),
   unm=pl.col('FLIGHT_ID_mvt').is_null())
f=pl.col('FLIGHT_mvt')
def strcls(df):
    return df.with_columns(
      p3=f.str.slice(0,3),
      scls=pl.when(f.str.contains(r'^[A-Z]{3}[0-9]')).then(pl.lit('icao_digit'))
       .when(f.str.slice(3,2)==f.str.slice(1,2)).then(pl.lit('dup2'))
       .when(f.str.slice(3,1)==f.str.slice(2,1)).then(pl.lit('dup1'))
       .when(f.str.contains(r'^[A-Z]{3}[A-Z]')).then(pl.lit('icao_letter'))
       .otherwise(pl.lit('other')),
      flen=f.str.len_chars(), ndig=f.str.count_matches(r'[0-9]'), suff_letter=f.str.contains(r'[A-Z]$'))
d=strcls(d)
print(d.group_by(['src','unm','scls']).len().sort(['src','unm','scls']))
# examples
for c in ['dup1','dup2','icao_letter','other']:
    print(c, d.filter(pl.col('scls')==c).group_by('unm').agg(pl.col('FLIGHT_mvt').unique().head(15)).to_dicts())
d.write_parquet('dep.parquet')

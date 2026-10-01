import polars as pl
pl.Config.set_tbl_rows(250); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(16); pl.Config.set_tbl_width_chars(320)
j=pl.read_parquet('lirf_elig_rot.parquet').with_columns(
  inb1=(pl.col('ib_dd')==1)&(pl.col('gap_h')<12),
  msb=pl.col('ms').cut([3*3600,6*3600,12*3600,20*3600,28*3600]),
  ipair=(pl.col('iflt').str.slice(0,3)==pl.col('FLIGHT_mvt').str.slice(0,3)))
t=j.filter(pl.col('src')=='train')
print(t.group_by('msb','inb1','cls').len().pivot(on='cls',index=['msb','inb1'],values='len').sort('msb','inb1'))
print(t.filter(pl.col('ms')>=43200).select('cls','FLIGHT_mvt','SCHED_TIME_UTC_mvt','MVT_TIME_UTC_mvt','ms_h','iflt','iadep','ib','gap_h','ib_dd','ipair','y').sort('ms_h'))
r=j.filter(pl.col('src')=='rank')
print('RANK'); print(r.group_by('msb','inb1').len().sort('msb','inb1'))
print(r.filter(pl.col('ms')>=10800).select('MVT_ID_mvt','FLIGHT_mvt','ADES_mvt','STAND_mvt','SCHED_TIME_UTC_mvt','MVT_TIME_UTC_mvt','ms_h','iflt','iadep','ib','gap_h','ib_dd','ipair').sort('ms_h'))

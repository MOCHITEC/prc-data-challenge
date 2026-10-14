import polars as pl
pl.Config.set_tbl_rows(250); pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(14); pl.Config.set_tbl_width_chars(320)
df=pl.read_parquet('all.parquet')
d=pl.read_parquet('dep_dd.parquet').filter((pl.col('APT')=='LIRF')&(~pl.col('matched'))&(pl.col('dd_loc')==1))
x=pl.read_parquet('xapt_best.parquet').select('MVT_ID_mvt','src','aflt','amatched','am','aobt','lag')
d=d.join(x,on=['MVT_ID_mvt','src'],how='left')
d=d.with_columns(cls=pl.when(pl.col('y')>72000).then(pl.lit('ROLL')).when((pl.col('y')-pl.col('ms')).abs()<=60).then(pl.lit('fill')).when(pl.col('y')<7200).then(pl.lit('ok')).otherwise(pl.lit('other')),
  link=pl.col('lag').abs()<1800, tod_m=pl.col('MVT_TIME_UTC_mvt').dt.convert_time_zone('Europe/Rome').dt.hour(),
  ms_h=(pl.col('ms')/3600).round(1), mon=pl.col('MVT_TIME_UTC_mvt').dt.month(),
  bs=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds())
d.write_parquet('lirf_elig.parquet')
t=d.filter(pl.col('src')=='train')
print(t.group_by('cls').len())
print(t.group_by('ADES_mvt').agg(pl.len()).sort('len',descending=True).head(5))
print('dest in 10 apts:', t.filter(pl.col('ftmed').is_not_null()).group_by('cls').len() if 'ftmed' in t.columns else '')
print(t.with_columns(dest10=pl.col('aflt').is_not_null()).group_by('cls','link').len().sort('cls'))
print(t.filter(pl.col('link')).select('FLIGHT_mvt','ADES_mvt','aflt','amatched','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','aobt','lag','y','ms','cls'))
# ms bucket x class
print(t.with_columns(msb=pl.col('ms').cut([3*3600,6*3600,12*3600,20*3600,28*3600])).group_by('msb','cls').len().pivot(on='cls',index='msb',values='len').sort('msb'))
print(t.group_by('tod_m','cls').len().pivot(on='cls',index='tod_m',values='len').sort('tod_m'))
r=d.filter(pl.col('src')=='rank')
print('rank elig', r.height, r.with_columns(msb=pl.col('ms').cut([3*3600,6*3600,12*3600,20*3600,28*3600])).group_by('msb').len().sort('msb'))

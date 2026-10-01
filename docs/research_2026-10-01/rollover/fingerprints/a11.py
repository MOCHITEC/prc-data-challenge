import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(40)
d=pl.read_parquet('dep3.parquet')
f=pl.col('FLIGHT_mvt')
d=d.with_columns(fn=pl.when(pl.col('scls')=='dup2').then(f.str.slice(0,3)+f.str.slice(5)).when(pl.col('scls')=='dup1').then(f.str.slice(0,3)+f.str.slice(4)).otherwise(f))
d=d.with_columns(fnum=pl.col('fn').str.slice(0,3)+pl.col('fn').str.slice(3).str.extract(r'^0*([0-9]+)',1))
print(d.filter(pl.col('scls').is_in(['dup1','dup2'])).select('FLIGHT_mvt','fn','fnum').unique().head(15))
q=d.filter(pl.col('unm')&(pl.col('ms')>=43200))
base=d.select('fnum','MVT_ID_mvt','SCHED_TIME_UTC_mvt','MVT_TIME_UTC_mvt','unm','FLIGHT_mvt','src').rename(lambda c:c+'_o')
j=q.join(base,left_on='fnum',right_on='fnum_o').filter(pl.col('MVT_ID_mvt')!=pl.col('MVT_ID_mvt_o'))
j=j.with_columns(dS=(pl.col('SCHED_TIME_UTC_mvt_o')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_minutes()/60,
                 msO=(pl.col('MVT_TIME_UTC_mvt_o')-pl.col('SCHED_TIME_UTC_mvt_o')).dt.total_minutes()/60).filter(pl.col('dS').abs()<=50)
summ=j.group_by('MVT_ID_mvt').agg(pl.format('{}:{}/{}{}',pl.col('FLIGHT_mvt_o'),pl.col('dS').round(1),pl.col('msO').round(1),pl.when(pl.col('unm_o')).then(pl.lit('u')).otherwise(pl.lit('m'))).str.join(' ').alias('others'),
     ((pl.col('dS')>12)&(pl.col('dS')<36)).any().alias('nextday'),((pl.col('dS')<-12)&(pl.col('dS')>-36)).any().alias('prevday'))
o=q.join(summ,on='MVT_ID_mvt',how='left').select('src','cls','FLIGHT_mvt','fnum','ms','nextday','prevday','others').sort('src','cls','ms')
print(o)
print(o.group_by('src','cls').agg(pl.len(),pl.col('nextday').fill_null(False).sum(),pl.col('prevday').fill_null(False).sum()))
d.write_parquet('dep4.parquet')

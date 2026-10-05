import polars as pl, glob, os
os.environ['POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR']='load_as_storage'
A=['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
def rd(f):
    d=pl.read_csv(f,infer_schema_length=0,encoding='utf8-lossy')
    return d.filter(pl.col('APT_ICAO').is_in(A)).with_columns(pl.col('FLT_DATE').str.slice(0,10).str.to_date().alias('date'))
dly=pl.concat([rd(f) for f in ['apt_dly_2025.csv','apt_dly_2026.csv']],how='diagonal')
sa=pl.concat([rd(f) for f in ['atfm_slot_adherence_2025.csv','atfm_slot_adherence_2026.csv']])
atc=pl.concat([rd(f) for f in ['atc_pre_departure_delays_2025.csv','atc_pre_departure_delays_2026.csv']])
allp=pl.concat([rd(f) for f in ['all_pre_departure_delays_2025.csv','all_pre_departure_delays_2026.csv']])
for name,d,col in [('apt_dly',dly,'DLY_APT_ARR_1'),('slot',sa,'FLT_DEP_REG_1'),('atcpre',atc,'DLY_ATC_PRE_2'),('allpre',allp,'DLY_ALL_PRE_2')]:
    d=d.with_columns(pl.col('date').dt.strftime('%Y-%m').alias('ym'))
    t=d.filter(pl.col('ym').is_in(['2025-01','2025-07','2026-01','2026-07'])).group_by('ym','APT_ICAO').agg(pl.len().alias('n'),pl.col(col).is_not_null().sum().alias('nn'))
    print(name, 'max date',d['date'].max()); print(t.pivot(on='ym',index='APT_ICAO',values='nn').sort('APT_ICAO'))
# cause codes for 2026 test months
d26=dly.filter(pl.col('date').dt.month().is_in([1,7]))
cc=[c for c in dly.columns if c.startswith('DLY_APT_ARR_') and c!='DLY_APT_ARR_1']
s=d26.with_columns([pl.col(c).cast(pl.Float64) for c in cc+['DLY_APT_ARR_1']]).group_by(pl.col('date').dt.year().alias('y'),pl.col('date').dt.month().alias('m')).agg([pl.col(c).sum() for c in ['DLY_APT_ARR_1']+cc]).sort('y','m')
print(s.transpose(include_header=True))
# per airport arrival delay totals 2026 test months
print(d26.with_columns(pl.col('DLY_APT_ARR_1').cast(pl.Float64),pl.col('DLY_APT_ARR_W_1').cast(pl.Float64),pl.col('DLY_APT_ARR_D_1').cast(pl.Float64)).group_by('APT_ICAO',pl.col('date').dt.strftime('%Y-%m').alias('ym')).agg(pl.col('DLY_APT_ARR_1').sum(),pl.col('DLY_APT_ARR_W_1').sum(),pl.col('DLY_APT_ARR_D_1').sum(),(pl.col('DLY_APT_ARR_1')>0).sum().alias('days_reg')).sort('APT_ICAO','ym'))
dly.write_parquet('apt_dly.parquet'); sa.write_parquet('slot.parquet'); atc.write_parquet('atc.parquet'); allp.write_parquet('allp.parquet')

import polars as pl
W='/home/user/prc-data-challenge/data/work_flseq/'
def load(f):
    d=pl.read_parquet(f).with_columns((pl.col('first_seen').str.to_datetime(strict=False) if pl.read_parquet_schema(f)['first_seen']==pl.String else pl.col('first_seen').cast(pl.Datetime('us'))),pl.col('flt_id').str.strip_chars().alias('cs'))
    return d.select('adep','cs','first_seen')
for tag,f,src,lab in [('2025-07','fl_202507.parquet','feat_train.parquet',True),('2026-07','fl_202607.parquet','feat_rank.parquet',False)]:
    fl=load(f)
    cols=['MVT_ID_mvt','ADEP_mvt','CALLSIGN_flt','MVT_TIME_UTC_mvt']+(['TAXITIME_SEC_mvt'] if lab else [])
    d=pl.read_parquet(W+src,columns=cols).with_columns(pl.col('MVT_TIME_UTC_mvt').dt.replace_time_zone(None).alias('t'),pl.col('CALLSIGN_flt').cast(pl.Utf8).str.strip_chars().alias('cs'))
    d=d.filter(pl.col('t').dt.strftime('%Y-%m')==tag)
    j=d.join(fl,left_on=['ADEP_mvt','cs'],right_on=['adep','cs']).with_columns(((pl.col('t')-pl.col('first_seen')).dt.total_seconds()).alias('dt')).filter((pl.col('dt')>-120)&(pl.col('dt')<7200))
    j=j.sort('dt').unique('MVT_ID_mvt',keep='first')
    agg=[pl.len().alias('n'),pl.col('dt').median().alias('med_dt'),(pl.col('dt')>300).mean().alias('gt300')]
    if lab:
        agg+= [pl.corr('dt','TAXITIME_SEC_mvt').alias('corr'), (pl.col('TAXITIME_SEC_mvt')-pl.col('dt')).filter(pl.col('dt')>300).median().alias('bias_gt300'),
               ((pl.col('TAXITIME_SEC_mvt')-pl.col('dt')).filter(pl.col('dt')>300).abs()<120).mean().alias('w120_gt300')]
    tot=d.group_by('ADEP_mvt').len().rename({'len':'N'})
    print(tag); print(j.group_by('ADEP_mvt').agg(agg).join(tot,on='ADEP_mvt').with_columns((pl.col('n')/pl.col('N')).round(3).alias('link')).sort('ADEP_mvt'))

import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(40); pl.Config.set_fmt_str_lengths(30); pl.Config.set_tbl_width_chars(300)
TZ={'LSZH':'Europe/Zurich','LIRF':'Europe/Rome','EDDM':'Europe/Berlin','EDDF':'Europe/Berlin','LEBL':'Europe/Madrid','LEMD':'Europe/Madrid','EGLL':'Europe/London','LFPG':'Europe/Paris','EHAM':'Europe/Amsterdam','LTFM':'Europe/Istanbul'}
def load(path, phase='DEP'):
    df=pl.read_parquet(path).filter(pl.col('PHASE_mvt')==phase)
    apt='ADEP_mvt' if phase=='DEP' else 'ADES_mvt'
    df=df.with_columns(pl.col(apt).alias('APT'))
    out=[]
    for a,tz in TZ.items():
        d=df.filter(pl.col('APT')==a)
        d=d.with_columns([pl.col(c).dt.convert_time_zone(tz).dt.replace_time_zone(None).alias(c+'_L') for c in ['MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt']])
        out.append(d)
    df=pl.concat(out)
    df=df.with_columns(
        M=pl.col('MVT_TIME_UTC_mvt'), B=pl.col('BLOCK_TIME_UTC_mvt'), S=pl.col('SCHED_TIME_UTC_mvt'),
        y=pl.col('TAXITIME_SEC_mvt'),
        ms=(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
        unm=pl.col('AOBT_3_flt').is_null() if phase=='DEP' else pl.col('ARVT_3_flt').is_null(),
        month=pl.col('MVT_TIME_UTC_mvt').dt.month(),
    )
    return df

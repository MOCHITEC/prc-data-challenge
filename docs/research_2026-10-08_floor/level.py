import polars as pl

def live_level(frame: pl.DataFrame) -> pl.DataFrame:
    """MVT_ID_mvt -> level: median(MVT-AOBT_3) of matched departures at the airport, 60-min window ending t+30m."""
    m = (frame.filter(pl.col('mvt_minus_aobt3').is_not_null())
         .select('ADEP_mvt', pl.col('MVT_TIME_UTC_mvt').alias('t'), 'mvt_minus_aobt3')
         .sort('ADEP_mvt', 't'))
    m = m.with_columns(pl.col('mvt_minus_aobt3').rolling_median_by('t', window_size='60m').over('ADEP_mvt').alias('level'))
    m = m.group_by('ADEP_mvt', 't', maintain_order=True).agg(pl.col('level').last()).sort('t')
    q = frame.select('MVT_ID_mvt', 'ADEP_mvt', (pl.col('MVT_TIME_UTC_mvt') + pl.duration(minutes=30)).alias('t')).sort('t')
    out = q.join_asof(m, on='t', by='ADEP_mvt', strategy='backward', tolerance='30m')
    return out.select('MVT_ID_mvt', 'level')

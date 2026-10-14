import polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
dep=pl.read_parquet(O+'dep_train_feat.parquet').join(pl.read_parquet(O+'standgap_train.parquet'),on='MVT_ID_mvt',how='left')
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(40)
big=dep.filter(pl.col('y')>14400)
print(big.select('APT','FLIGHT_mvt','unm','STAND_mvt','ms','y','gap_h','dep_between','arr_flight','arr_ac','AIRCRAFT_TYPE_mvt').sort('APT','y').to_pandas().to_string(index=False))
d=dep.with_columns(gb=pl.when(pl.col('gap_h').is_null()).then(pl.lit('null')).when(pl.col('dep_between')).then(pl.lit('dep_between'))
     .when(pl.col('gap_h')<6).then(pl.lit('<6h')).when(pl.col('gap_h')<16).then(pl.lit('6-16h')).when(pl.col('gap_h')<24).then(pl.lit('16-24h')).when(pl.col('gap_h')<48).then(pl.lit('24-48h')).otherwise(pl.lit('>=48h')),
     roll=pl.col('y')>72000, tail4=pl.col('y')>14400)
for lab,flt in [('nonLIRF all',pl.col('APT')!='LIRF'),('nonLIRF unmatched',(pl.col('APT')!='LIRF')&pl.col('unm')),('nonLIRF unmatched |ms|<2h',(pl.col('APT')!='LIRF')&pl.col('unm')&(pl.col('ms').abs()<7200)),('LIRF unmatched',(pl.col('APT')=='LIRF')&pl.col('unm'))]:
    print(lab); print(d.filter(flt).group_by('gb').agg(pl.len(),pl.col('roll').sum(),pl.col('tail4').sum()).sort('gb'))
# per airport for nonLIRF unmatched |ms|<2h with gap>=24h and no dep between
c=d.filter((pl.col('APT')!='LIRF')&pl.col('unm')&(pl.col('ms').abs()<7200)&(pl.col('gap_h')>=24)&~pl.col('dep_between'))
print(c.group_by('APT').agg(pl.len(),pl.col('roll').sum(),pl.col('tail4').sum()).sort('APT'))
# regular-airline (ICAO fmt, n_occ>=5, other_match_share>=0.8) subset
c2=c.filter((pl.col('fmt')=='ICAO')&(pl.col('n_occ')>=5)&(pl.col('other_match_share')>=0.8))
print('gap>=24h & regular matched airline flight:',c2.height,'roll',c2['roll'].sum())
print(c2.select('APT','FLIGHT_mvt','STAND_mvt','ms','y','gap_h','arr_flight','SCHED_TIME_UTC_mvt').sort('gap_h').head(40).to_pandas().to_string(index=False))

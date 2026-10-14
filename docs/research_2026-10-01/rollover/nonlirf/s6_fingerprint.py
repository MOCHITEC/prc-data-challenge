# Fingerprint: non-LIRF DEP tail rows vs NM-unmatched |ms|<2h controls
import polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
def prep(path):
    df=pl.read_parquet(path)
    df=df.with_columns(
        ms=(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
        unm=pl.col('FLIGHT_ID_mvt').is_null(),
        sd=pl.col('SCHED_TIME_UTC_mvt').dt.date(), md=pl.col('MVT_TIME_UTC_mvt').dt.date(),
        mvt_sec=pl.col('MVT_TIME_UTC_mvt').dt.second(), mvt_us=pl.col('MVT_TIME_UTC_mvt').dt.microsecond(),
        sch_sec=pl.col('SCHED_TIME_UTC_mvt').dt.second(), sch_min=pl.col('SCHED_TIME_UTC_mvt').dt.minute(),
        hour=pl.col('SCHED_TIME_UTC_mvt').dt.hour(), wd=pl.col('SCHED_TIME_UTC_mvt').dt.weekday(),
        fmt=pl.when(pl.col('FLIGHT_mvt').str.contains(r'^[A-Z]{3}\d{1,4}[A-Z]{0,2}$')).then(pl.lit('ICAO'))
            .when(pl.col('FLIGHT_mvt').str.contains(r'^[A-Z]{2}\d{1,4}[A-Z]?$')).then(pl.lit('IATA'))
            .when(pl.col('FLIGHT_mvt').str.contains(r'^[A-Z]{5}$|^[A-Z]{1,2}[A-Z0-9]{3,5}$')).then(pl.lit('REG?')).otherwise(pl.lit('other')))
    if 'TAXITIME_SEC_mvt' in df.columns:
        df=df.with_columns(y=pl.when(pl.col('PHASE_mvt')=='DEP').then((pl.col('MVT_TIME_UTC_mvt')-pl.col('BLOCK_TIME_UTC_mvt')).dt.total_seconds()).otherwise(None))
    dep=df.filter(pl.col('PHASE_mvt')=='DEP')
    arr=df.filter(pl.col('PHASE_mvt')=='ARR')
    # regularity of (ADEP,FLIGHT) among DEP rows in same file
    reg=dep.group_by('ADEP_mvt','FLIGHT_mvt').agg(n_occ=pl.len(), n_match=(~pl.col('unm')).sum())
    dep=dep.join(reg,on=['ADEP_mvt','FLIGHT_mvt'],how='left').with_columns(
        other_match_share=(pl.col('n_match')-(~pl.col('unm')).cast(pl.Int64))/(pl.col('n_occ')-1))
    days=dep.select('ADEP_mvt','FLIGHT_mvt',pl.col('sd').alias('d')).unique()
    dep=dep.join(days.with_columns(prev_day=pl.lit(True),d=pl.col('d')+pl.duration(days=1)),left_on=['ADEP_mvt','FLIGHT_mvt','sd'],right_on=['ADEP_mvt','FLIGHT_mvt','d'],how='left')
    dep=dep.join(days.with_columns(next_day=pl.lit(True),d=pl.col('d')-pl.duration(days=1)),left_on=['ADEP_mvt','FLIGHT_mvt','sd'],right_on=['ADEP_mvt','FLIGHT_mvt','d'],how='left')
    # same-day multiple DEP rows of same flight
    dup=dep.group_by('ADEP_mvt','FLIGHT_mvt','sd').agg(n_same_day=pl.len())
    dep=dep.join(dup,on=['ADEP_mvt','FLIGHT_mvt','sd'],how='left')
    # ARR of same FLIGHT string at same airport on same/prev day
    ad=arr.select(pl.col('ADES_mvt').alias('ADEP_mvt'),'FLIGHT_mvt',pl.col('md').alias('d')).unique()
    dep=dep.join(ad.with_columns(arr_same_day=pl.lit(True)),left_on=['ADEP_mvt','FLIGHT_mvt','sd'],right_on=['ADEP_mvt','FLIGHT_mvt','d'],how='left')
    dep=dep.join(ad.with_columns(arr_prev_day=pl.lit(True),d=pl.col('d')+pl.duration(days=1)),left_on=['ADEP_mvt','FLIGHT_mvt','sd'],right_on=['ADEP_mvt','FLIGHT_mvt','d'],how='left')
    return dep.with_columns([pl.col(c).fill_null(False) for c in ['prev_day','next_day','arr_same_day','arr_prev_day']])
if __name__=='__main__':
    dep=prep(O+'train.parquet')
    dep.write_parquet(O+'dep_train_feat.parquet')
    pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(40)
    F=['APT','FLIGHT_mvt','AIRCRAFT_TYPE_mvt','STAND_mvt','RUNWAY_mvt','FLIGHT_RULE_mvt','fmt','ms','y','mvt_sec','mvt_us','sch_sec','sch_min','hour','wd','n_occ','other_match_share','prev_day','next_day','n_same_day','arr_same_day','arr_prev_day']
    pos=dep.filter((pl.col('APT')!='LIRF')&(pl.col('y')>14400))
    print(pos.select(F).to_pandas().to_string(index=False))
    ctl=dep.filter((pl.col('APT')!='LIRF')&pl.col('unm')&(pl.col('ms').abs()<7200)&(pl.col('y')<14400))
    print('controls',ctl.height)
    for c in ['fmt','FLIGHT_RULE_mvt','prev_day','next_day','arr_same_day','arr_prev_day','n_same_day']:
        print(ctl[c].value_counts(sort=True).head(8).to_pandas().to_string(index=False))
    print('AC null share',ctl['AIRCRAFT_TYPE_mvt'].is_null().mean(),' mvt_sec==0',(ctl['mvt_sec']==0).mean(),' mvt_us!=0',(ctl['mvt_us']!=0).mean(), 'sch_sec!=0',(ctl['sch_sec']!=0).mean())
    print('other_match_share>=0.8 & n_occ>=5:',ctl.filter((pl.col('other_match_share')>=0.8)&(pl.col('n_occ')>=5)).height)
    print('  & prev_day False:',ctl.filter((pl.col('other_match_share')>=0.8)&(pl.col('n_occ')>=5)&~pl.col('prev_day')).height)
    print('mvt_sec by apt (share==0) controls:'); print(ctl.group_by('APT').agg(pl.len(),(pl.col('mvt_sec')==0).mean().alias('sec0'),((pl.col('mvt_sec')<=5)|(pl.col('mvt_sec')>=55)).mean().alias('sec_near0')).sort('APT'))
    print('ms<0 share controls:',(ctl['ms']<0).mean(),' ms quantiles',ctl['ms'].quantile(0.1),ctl['ms'].quantile(0.5),ctl['ms'].quantile(0.9))

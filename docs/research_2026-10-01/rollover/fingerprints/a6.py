import polars as pl
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_cols(40); pl.Config.set_fmt_str_lengths(14); pl.Config.set_tbl_width_chars(300)
d=pl.read_parquet('dep2.parquet')
allr=pl.read_parquet('lirf.parquet')
arr=allr.filter(pl.col('PHASE_mvt')=='ARR')
dep=allr.filter(pl.col('PHASE_mvt')=='DEP')
q=d.filter(pl.col('unm')&(pl.col('ms')>=43200)).sort('src','cls','ms')
rows=[]
for r in q.iter_rows(named=True):
    S,M,st,fl=r['SCHED_TIME_UTC_mvt'],r['MVT_TIME_UTC_mvt'],r['STAND_mvt'],r['FLIGHT_mvt']
    a=arr.filter((pl.col('STAND_mvt')==st))
    inb=a.filter((pl.col('BLOCK_TIME_UTC_mvt')>S-pl.duration(hours=24))&(pl.col('BLOCK_TIME_UTC_mvt')<M)).sort('BLOCK_TIME_UTC_mvt')
    od=dep.filter((pl.col('STAND_mvt')==st)&(pl.col('MVT_TIME_UTC_mvt')>S-pl.duration(hours=24))&(pl.col('MVT_TIME_UTC_mvt')<M+pl.duration(hours=6))&(pl.col('MVT_ID_mvt')!=r['MVT_ID_mvt'])).sort('MVT_TIME_UTC_mvt')
    # events in (S, M-40min)
    n_arr_mid=inb.filter((pl.col('BLOCK_TIME_UTC_mvt')>S)&(pl.col('BLOCK_TIME_UTC_mvt')<M-pl.duration(minutes=40))).height
    n_dep_mid=od.filter((pl.col('MVT_TIME_UTC_mvt')>S)&(pl.col('MVT_TIME_UTC_mvt')<M-pl.duration(minutes=40))).height
    last_inb=inb['BLOCK_TIME_UTC_mvt'].max()
    gap_h=None if last_inb is None else (M-last_inb).total_seconds()/3600
    last_before_S=inb.filter(pl.col('BLOCK_TIME_UTC_mvt')<S)['BLOCK_TIME_UTC_mvt'].max()
    gS=None if last_before_S is None else (S-last_before_S).total_seconds()/3600
    # same flight string other departures within -2..+2 days
    sf=dep.filter((pl.col('FLIGHT_mvt')==fl)&(pl.col('MVT_ID_mvt')!=r['MVT_ID_mvt'])&((pl.col('SCHED_TIME_UTC_mvt')-S).dt.total_hours().abs()<=50))
    sfs=';'.join(f"{(x['SCHED_TIME_UTC_mvt']-S).total_seconds()/3600:+.1f}/{(x['MVT_TIME_UTC_mvt']-x['SCHED_TIME_UTC_mvt']).total_seconds()/3600:.1f}{'' if x['FLIGHT_ID_mvt'] is None else 'm'}" for x in sf.iter_rows(named=True))
    rows.append(dict(src=r['src'],cls=r['cls'],fl=fl,st=st,ms_h=round(r['ms']/3600,1),n_arr_mid=n_arr_mid,n_dep_mid=n_dep_mid,gapM_lastinb_h=None if gap_h is None else round(gap_h,1),gapS_lastinb_h=None if gS is None else round(gS,1),same_fl=sfs))
o=pl.DataFrame(rows)
print(o)
o.write_parquet('stand_feats_12h.parquet')

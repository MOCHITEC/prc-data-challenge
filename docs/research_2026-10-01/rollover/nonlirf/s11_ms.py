# ms distribution for unmatched vs matched DEP (by airport, winter/summer) ; family "prev-attempt BLOCK" = unmatched, ms in normal band, y>4h
import polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
dep=pl.read_parquet(O+'dep_train_feat.parquet').with_columns(summer=pl.col('MVT_TIME_UTC_mvt').dt.month().is_between(4,10))
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(40)
print(dep.filter(pl.col('ms').abs()<7200*2).group_by('APT','unm').agg(pl.len(),*[pl.col('ms').quantile(q).alias(f'q{int(q*100)}') for q in [0.1,0.25,0.5,0.75,0.9]],
      ((pl.col('ms')>=900)&(pl.col('ms')<=2100)).mean().alias('band900_2100')).sort('APT','unm'))
pos=dep.filter(pl.col('unm')&(pl.col('ms').abs()<7200)&(pl.col('y')>14400))
print(pos.select('APT','FLIGHT_mvt','ms','y','SCHED_TIME_UTC_mvt').to_pandas().to_string(index=False))
c=dep.filter(pl.col('unm')&(pl.col('ms').abs()<7200))
print('all-airport unmatched |ms|<2h:',c.height,' with ms in [900,2100]:',c.filter(pl.col('ms').is_between(900,2100)).height,' tail>4h in band:',c.filter(pl.col('ms').is_between(900,2100)&(pl.col('y')>14400)).height)
for a in ['LIRF','nonLIRF']:
    cc=c.filter((pl.col('APT')=='LIRF') if a=='LIRF' else (pl.col('APT')!='LIRF'))
    print(a,cc.height,'band',cc.filter(pl.col('ms').is_between(900,2100)).height,'tail band',cc.filter(pl.col('ms').is_between(900,2100)&(pl.col('y')>14400)).height,'y>20h band',cc.filter(pl.col('ms').is_between(900,2100)&(pl.col('y')>72000)).height)

# Day-difference families for tail rows (|y|>4h) by phase/airport/month + control base rates
import polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
df=pl.read_parquet(O+'train.parquet').with_columns(
    y=pl.when(pl.col('PHASE_mvt')=='DEP').then((pl.col('MVT_TIME_UTC_mvt')-pl.col('BLOCK_TIME_UTC_mvt')).dt.total_seconds())
      .otherwise((pl.col('BLOCK_TIME_UTC_mvt')-pl.col('MVT_TIME_UTC_mvt')).dt.total_seconds()),
    ms=(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
    unm=pl.col('FLIGHT_ID_mvt').is_null(),
    dMB=(pl.col('MVT_TIME_UTC_mvt').dt.date()-pl.col('BLOCK_TIME_UTC_mvt').dt.date()).dt.total_days(),
    dSB=(pl.col('SCHED_TIME_UTC_mvt').dt.date()-pl.col('BLOCK_TIME_UTC_mvt').dt.date()).dt.total_days(),
    mon=pl.col('MVT_TIME_UTC_mvt').dt.month())
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30)
t=df.filter(pl.col('y').abs()>14400)
print(t.group_by('PHASE_mvt','APT','unm','dMB','dSB').agg(pl.len(),pl.col('y').median(),pl.col('mon').unique().sort().cast(pl.String).str.join(',')).sort('PHASE_mvt','APT','dMB','dSB'))
# whole-population: rows with dSB != 0 (BLOCK date differs from SCHED date), by phase/apt, and their y distribution
print('rows with date(SCHED)!=date(BLOCK):')
u=df.filter(pl.col('dSB')!=0).with_columns(big=pl.col('y').abs()>14400, Bm=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds())
print(u.group_by('PHASE_mvt','APT','dSB').agg(pl.len(),pl.col('big').sum(),(pl.col('Bm').abs()<4*3600).sum().alias('B_within4h_of_S')).sort('PHASE_mvt','APT','dSB'))
# controls: non-LIRF DEP unmatched with |ms|<2h
c=df.filter((pl.col('PHASE_mvt')=='DEP')&(pl.col('APT')!='LIRF')&pl.col('unm'))
print('nonLIRF DEP unmatched n=',c.height,' |ms|<2h n=',c.filter(pl.col('ms').abs()<7200).height,
      ' of which y>20h:',c.filter((pl.col('ms').abs()<7200)&(pl.col('y')>72000)).height,' y>4h:',c.filter((pl.col('ms').abs()<7200)&(pl.col('y')>14400)).height)
print(c.with_columns(msb=pl.col('ms').abs()<7200).group_by('APT','msb').agg(pl.len(),(pl.col('y')>14400).sum().alias('y>4h'),(pl.col('y')>72000).sum().alias('y>20h')).sort('APT','msb'))
d=df.filter((pl.col('PHASE_mvt')=='DEP')&(pl.col('APT')!='LIRF'))
print('nonLIRF DEP all n=',d.height,' y>4h',(d['y']>14400).sum(),' y>20h',(d['y']>72000).sum())

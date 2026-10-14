# Label-tail families: y histogram in 1h bins beyond normal range, by phase/airport
import polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
df=pl.read_parquet(O+'train.parquet').with_columns(
    y=pl.when(pl.col('PHASE_mvt')=='DEP').then((pl.col('MVT_TIME_UTC_mvt')-pl.col('BLOCK_TIME_UTC_mvt')).dt.total_seconds())
      .otherwise((pl.col('BLOCK_TIME_UTC_mvt')-pl.col('MVT_TIME_UTC_mvt')).dt.total_seconds()))
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30)
for ph in ['DEP','ARR']:
    s=df.filter(pl.col('PHASE_mvt')==ph)
    print(ph,'quantiles', s['y'].quantile(0.001), s['y'].quantile(0.999), 'neg',(s['y']<0).sum())
    t=s.filter((pl.col('y')<-600)|(pl.col('y')>7200)).with_columns(hb=(pl.col('y')/3600).floor().cast(pl.Int32))
    print(t.pivot(on='APT',index='hb',values='y',aggregate_function='len',sort_columns=True).sort('hb'))

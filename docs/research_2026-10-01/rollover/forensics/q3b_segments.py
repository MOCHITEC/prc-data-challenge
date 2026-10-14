# Row order: segments; and position of rollover rows relative to neighbours (MVT vs BLOCK).
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet', columns=['file','rowpos','MVT_ID_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','TAXITIME_SEC_mvt'])
AP10=['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
df=df.sort('file','rowpos').with_columns(AP=pl.when(pl.col('PHASE_mvt')=='DEP').then(pl.col('ADEP_mvt')).otherwise(pl.col('ADES_mvt')),
   t=pl.col('MVT_TIME_UTC_mvt').dt.epoch('s'))
# airport-of-interest consistency (DEP between two of the 10 airports appears twice?)
print('AP in 10:', df['AP'].is_in(AP10).mean())
df=df.with_columns(newseg=((pl.col('AP')!=pl.col('AP').shift(1).over('file')) | (pl.col('t')<pl.col('t').shift(1).over('file'))).fill_null(True))
df=df.with_columns(seg=pl.col('newseg').cast(pl.Int32).cum_sum().over('file'))
sg=df.group_by('file','seg').agg(pl.len().alias('n'),pl.col('AP').first(),pl.col('t').min().alias('t0'),pl.col('t').max().alias('t1'),
    pl.col('MVT_TIME_UTC_mvt').min().alias('tmin'))
sg=sg.with_columns(span_h=(pl.col('t1')-pl.col('t0'))/3600)
print(sg.group_by(pl.col('file')=='ranking.parquet').agg(pl.len().alias('nseg'),pl.col('n').median().alias('n_med'),pl.col('n').quantile(0.9).alias('n_p90'),
   pl.col('span_h').median().alias('span_med_h'),pl.col('span_h').max().alias('span_max_h'),
   (pl.col('tmin').dt.minute()+pl.col('tmin').dt.hour()*60).median().alias('start_minofday_med')))
# are segment starts on fixed boundaries? look at start hour/minute distribution
print(sg.filter(pl.col('file')=='training_2025-07-01').sort('seg').head(15))
# neighbour test for training DEP rows: does own MVT lie between prev and next row MVT within the same segment?
df=df.with_columns(prev_t=pl.col('t').shift(1).over('file'), next_t=pl.col('t').shift(-1).over('file'),
   prev_seg=pl.col('seg').shift(1).over('file'), next_seg=pl.col('seg').shift(-1).over('file'),
   blk=pl.col('BLOCK_TIME_UTC_mvt').dt.epoch('s'))
dep=df.filter(pl.col('PHASE_mvt')=='DEP')
dep=dep.with_columns(roll=pl.col('TAXITIME_SEC_mvt')>72000,
   # median time of the 10 surrounding rows (5 each side), regardless of segment
   )
r=dep.filter(pl.col('roll'))
out=[]
for row in r.iter_rows(named=True):
    f=df.filter(pl.col('file')==row['file'], (pl.col('rowpos')-row['rowpos']).abs()<=5).sort('rowpos')
    nb=f.filter(pl.col('rowpos')!=row['rowpos'])
    out.append(dict(ADEP=row['ADEP_mvt'],FL=row['FLIGHT_mvt'],MVT=row['MVT_TIME_UTC_mvt'],BLOCK=row['BLOCK_TIME_UTC_mvt'],
        nb_mvt_min=nb['MVT_TIME_UTC_mvt'].min(), nb_mvt_max=nb['MVT_TIME_UTC_mvt'].max(), seg_n=int(sg.filter(pl.col('file')==row['file'],pl.col('seg')==row['seg'])['n'][0]),
        nb_ap=','.join(sorted(set(nb['AP'].to_list())))))
print(pl.DataFrame(out))
# global: share of training DEP rows whose own MVT within [min,max] of its 2 nearest same-file neighbours' MVT (in-sequence by MVT)
dep=dep.with_columns(inseq=((pl.col('t')>=pl.col('prev_t'))|(pl.col('prev_seg')!=pl.col('seg'))) & ((pl.col('t')<=pl.col('next_t'))|(pl.col('next_seg')!=pl.col('seg'))))
print(dep.group_by(pl.col('file')=='ranking.parquet','roll').agg(pl.len(),pl.col('inseq').mean()))

# MVT_ID / FLIGHT_ID ordering key per airport, and where rollover rows sit in ID order.
import polars as pl, numpy as np
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(260)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet', columns=['file','rowpos','MVT_ID_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','TAXITIME_SEC_mvt','LOBT_flt'])
df=df.filter(pl.col('PHASE_mvt')=='DEP').with_columns(rank=pl.col('file')=='ranking.parquet',
   tm=pl.col('MVT_TIME_UTC_mvt').dt.epoch('s'), ts=pl.col('SCHED_TIME_UTC_mvt').dt.epoch('s'), tb=pl.col('BLOCK_TIME_UTC_mvt').dt.epoch('s'),
   unm=pl.col('FLIGHT_ID_mvt').is_null(), y=pl.col('TAXITIME_SEC_mvt'))
df=df.with_columns(bs=pl.col('tb')-pl.col('ts'))
df=df.with_columns(cls=pl.when(pl.col('rank')).then(pl.lit('RANK')).when(pl.col('y')>72000).then(pl.lit('R'))
  .when(pl.col('unm') & (pl.col('bs').abs()<=6)).then(pl.lit('SF')).when(pl.col('unm')).then(pl.lit('UN')).otherwise(pl.lit('M')))
df=df.sort('ADEP_mvt','MVT_ID_mvt')
df=df.with_columns(gap=pl.col('MVT_ID_mvt').diff().over('ADEP_mvt'))
# run = consecutive ids with gap <= 50 within airport
df=df.with_columns(run=(pl.col('gap').is_null() | (pl.col('gap')>50)).cast(pl.Int32).cum_sum().over('ADEP_mvt'))
def nd(c): return (pl.col(c).diff()>=0).mean()
# within-run ordering keys (pairs adjacent in id): share non-decreasing
print('share of id-adjacent DEP pairs (same airport, same run) non-decreasing in key; train vs rank')
print(df.with_columns([ (pl.col(c).diff().over('ADEP_mvt','run')>=0).alias('nd_'+c) for c in ['tm','ts','tb']])
  .group_by('ADEP_mvt','rank').agg(pl.len(), pl.col('nd_tm').mean().round(3), pl.col('nd_ts').mean().round(3), pl.col('nd_tb').mean().round(3), pl.col('run').n_unique().alias('nruns')).sort('ADEP_mvt','rank'))
# k-neighbour medians in id order (same airport, +-5 by id, excluding self)
K=5
parts=[]
for (a,), g in df.group_by(['ADEP_mvt']):
    g=g.sort('MVT_ID_mvt')
    tm=g['tm'].to_numpy().astype(float); ts=g['ts'].to_numpy().astype(float); tb=g['tb'].fill_null(np.nan).to_numpy().astype(float) if g['tb'].null_count()<len(g) else np.full(len(g),np.nan)
    tb=np.array([np.nan if v is None else v for v in g['tb'].to_list()],dtype=float)
    n=len(g); idx=np.arange(n)
    def nbmed(arr):
        out=np.full(n,np.nan)
        stack=[]
        for d in list(range(-K,0))+list(range(1,K+1)):
            j=idx+d; ok=(j>=0)&(j<n)
            v=np.full(n,np.nan); v[ok]=arr[j[ok]]; stack.append(v)
        return np.nanmedian(np.vstack(stack),axis=0)
    parts.append(g.with_columns(nb_tm=pl.Series(nbmed(tm)), nb_ts=pl.Series(nbmed(ts)), nb_tb=pl.Series(nbmed(tb))))
d2=pl.concat(parts)
d2=d2.with_columns(dm=(pl.col('tm')-pl.col('nb_tm'))/3600, dsch=(pl.col('ts')-pl.col('nb_ts'))/3600, dblk=(pl.col('tb')-pl.col('nb_tb'))/3600,
                   blk_vs_nbm=(pl.col('tb')-pl.col('nb_tm'))/3600)
print('rollover rows: own time minus median of 10 id-neighbours (hours)')
print(d2.filter(pl.col('cls')=='R').select('ADEP_mvt','FLIGHT_mvt','MVT_ID_mvt','MVT_TIME_UTC_mvt','dm','dsch','dblk','blk_vs_nbm').sort('ADEP_mvt'))
print('distribution of |dm|>12h and |dsch|>12h by cls/airport')
print(d2.group_by('ADEP_mvt','cls').agg(pl.len(), (pl.col('dm').abs()>12).mean().round(4).alias('dm_gt12'), (pl.col('dsch').abs()>12).mean().round(4).alias('dsch_gt12'),
   pl.col('dm').abs().median().round(3).alias('dm_med'), pl.col('dsch').abs().median().round(3).alias('dsch_med'), pl.col('dblk').abs().median().round(3).alias('dblk_med')).sort('ADEP_mvt','cls'))
d2.select('ADEP_mvt','MVT_ID_mvt','file','rowpos','cls','dm','dsch','dblk','blk_vs_nbm','run').write_parquet(OUT+'idnb.parquet')

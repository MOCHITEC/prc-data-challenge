# Identify the MVT_ID ordering key at LFPG/LSZH/EHAM/LTFM (non-SCHED-ordered airports) + tie-break at SCHED-ordered airports
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(260)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP').with_columns(rank=pl.col('file')=='ranking.parquet')
df=df.sort('ADEP_mvt','MVT_ID_mvt')
keys={'dM':pl.col('MVT_TIME_UTC_mvt').dt.date(),'dS':pl.col('SCHED_TIME_UTC_mvt').dt.date(),'dB':pl.col('BLOCK_TIME_UTC_mvt').dt.date(),
      'hM':pl.col('MVT_TIME_UTC_mvt').dt.truncate('1h'),'hS':pl.col('SCHED_TIME_UTC_mvt').dt.truncate('1h'),'hB':pl.col('BLOCK_TIME_UTC_mvt').dt.truncate('1h'),
      'fl':pl.col('FLIGHT_mvt'),'ades':pl.col('ADES_mvt'),'fid':pl.col('FLIGHT_ID_mvt'),'stand':pl.col('STAND_mvt'),'rwy':pl.col('RUNWAY_mvt'),
      'mvt':pl.col('MVT_TIME_UTC_mvt'),'blk':pl.col('BLOCK_TIME_UTC_mvt'),'sch':pl.col('SCHED_TIME_UTC_mvt'),'lobt':pl.col('LOBT_flt')}
df=df.with_columns([v.alias(k) for k,v in keys.items()])
ex=[]
for k in keys:
    ex.append((pl.col(k)>=pl.col(k).shift(1)).over('ADEP_mvt').mean().round(3).alias(k))
print('share id-adjacent pairs non-decreasing, per airport (train)')
print(df.filter(~pl.col('rank')).group_by('ADEP_mvt').agg(ex).sort('ADEP_mvt'))
print('ranking')
print(df.filter(pl.col('rank')).group_by('ADEP_mvt').agg([e for e in ex if e.meta.output_name() not in ('dB','hB','blk')]).sort('ADEP_mvt'))
# Tie-break at SCHED-ordered airports: pairs adjacent in id with identical SCHED -> non-decreasing in MVT? BLOCK? FLIGHT?
t=df.filter(~pl.col('rank')).with_columns(same=(pl.col('sch')==pl.col('sch').shift(1)).over('ADEP_mvt'),
   ndm=(pl.col('mvt')>=pl.col('mvt').shift(1)).over('ADEP_mvt'), ndb=(pl.col('blk')>=pl.col('blk').shift(1)).over('ADEP_mvt'),
   ndf=(pl.col('fl')>=pl.col('fl').shift(1)).over('ADEP_mvt'), ndfid=(pl.col('fid')>=pl.col('fid').shift(1)).over('ADEP_mvt'),
   ndl=(pl.col('lobt')>=pl.col('lobt').shift(1)).over('ADEP_mvt'))
print('tie pairs (same SCHED) non-decreasing in key')
print(t.filter(pl.col('same')).group_by('ADEP_mvt').agg(pl.len(),pl.col('ndm').mean().round(3),pl.col('ndb').mean().round(3),pl.col('ndf').mean().round(3),pl.col('ndfid').mean().round(3),pl.col('ndl').mean().round(3)).sort('ADEP_mvt'))

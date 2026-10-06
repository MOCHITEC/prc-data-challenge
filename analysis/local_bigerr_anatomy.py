"""Big-error anatomy of the matched lane (AOBT_3 present) in the 7-month OOF (p = b + c_C), local, read-only.

Needs $SCRATCH/agents/matched/bigerr_base.parquet from analysis/local_bigerr_prep.py. Prints: |e| band x lane, airport,
sign, BLOCK-AOBT_3 / MVT-SCHED / MVT-LOBT bands, regime flags (BLOCK == SCHED/LOBT/IOBT/EOBT_1/AOBT_3 within 12/60 s,
whole-minute / 5-minute BLOCK, AOBT_3 +- whole hours), ADS-B agreement (push/move/gs3 near BLOCK vs near AOBT_3),
regime categories per month and airport, the copy rate as a function of LOBT - p, and truncation near the LOBT window.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_bigerr_anatomy.py
"""
import polars as pl, os
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30)
S=os.environ['SCRATCH']; D=pl.read_parquet(S+'/agents/matched/bigerr_base.parquet')
N=D.height; SSE=float((D['e']**2).sum())
D=D.with_columns(pl.col('AOBT_3_flt').is_not_null().alias('m'),
  pl.col('e').abs().cut([300,600,1200,1800,3600],left_closed=True).alias('band'),
  (pl.col('y')-pl.col('d_aobt')).alias('blk_aobt'),(pl.col('has')&pl.col('ground_fix').fill_null(False)).alias('gf'))
def g(df,by,sort=None):
    r=df.group_by(by).agg(pl.len().alias('n'),((pl.col('e')**2).sum()/SSE*100).round(2).alias('pctSSE'),((pl.col('e')**2).sum()/N).round(1).alias('mseN'),pl.col('e').mean().round(0).alias('bias'),pl.col('has').mean().round(2).alias('has_r'),pl.col('gf').mean().round(2).alias('gf_r'))
    return r.sort(sort or by, descending=bool(sort))
q=lambda c,br: pl.col(c).cut(br).alias(c+'_b')
M=D.filter(pl.col('m')); B=M.filter(pl.col('e').abs()>600)

# ---- a2
print(g(B,['gf','band']))
print(g(B.with_columns(q('blk_aobt',[-3600,-1800,-600,-120,-12,12,120,600,1800,3600])),['blk_aobt_b']))
print(g(M.with_columns(q('blk_aobt',[-3600,-1800,-600,-120,-12,12,120,600,1800,3600])),['blk_aobt_b']))
print(g(B.with_columns(q('d_aobt',[0,300,600,1200,1800,3600,7200])),['d_aobt_b']))
print(g(B.with_columns(q('d_sched',[-3600,0,600,1200,1800,3600,7200,14400])),['d_sched_b']))
print(g(B.with_columns(q('d_lobt',[-3600,0,600,1200,1800,3600,7200,14400])),['d_lobt_b']))
# relation to ADS-B move: y - d_move = BLOCK error vs move
print(g(B.filter(pl.col('gf')).with_columns((pl.col('y')-pl.col('d_move')).alias('blk_move')).with_columns(q('blk_move',[-1800,-600,-120,-30,30,120,600,1800])),['blk_move_b']))
print(g(B.filter(pl.col('gf')).with_columns((pl.col('d_aobt')-pl.col('d_move')).alias('aobt_move')).with_columns(q('aobt_move',[-1800,-600,-120,-30,30,120,600,1800])),['aobt_move_b']))

# ---- a3
blk=pl.col('BLOCK_TIME_UTC_mvt')
def near(c,tol): return ((blk-pl.col(c)).dt.total_seconds().abs()<=tol)
X=M.with_columns(
  near('SCHED_TIME_UTC_mvt',12).alias('eqS12'),near('SCHED_TIME_UTC_mvt',60).alias('eqS60'),
  near('LOBT_flt',12).alias('eqL12'),near('LOBT_flt',60).alias('eqL60'),
  near('IOBT_flt',12).alias('eqI12'),near('EOBT_1_flt',12).alias('eqE12'),near('EOBT_1_flt',60).alias('eqE60'),
  near('AOBT_3_flt',12).alias('eqA12'),
  (blk.dt.second()==0).alias('sec0'),((blk.dt.second()==0)&(blk.dt.minute()%5==0)).alias('min5'),
  (pl.col('e').abs()>600).alias('big'),(pl.col('e').abs()>1800).alias('big18'))
# AOBT +- whole hours
hh=((pl.col('blk_aobt')/3600).round(0))
X=X.with_columns(((pl.col('blk_aobt')-hh*3600).abs()<=60).and_(hh!=0).alias('hshift'),hh.alias('hk'))
flags=['eqS12','eqS60','eqL12','eqL60','eqI12','eqE12','eqE60','eqA12','sec0','min5','hshift']
agg=[pl.len().alias('n')]+[pl.col(f).mean().round(3).alias(f) for f in flags]
print(X.group_by('ADEP_mvt').agg(agg).sort('ADEP_mvt'))
print(X.filter('big').group_by('ADEP_mvt').agg(agg).sort('ADEP_mvt'))
print(X.filter('big18').group_by('ADEP_mvt').agg(agg).sort('ADEP_mvt'))
X.select(['MVT_ID_mvt']+flags+['hk','big','big18']).write_parquet(S+'/agents/matched/flags.parquet')

# ---- a5
F=pl.read_parquet(S+"/agents/matched/flags.parquet")
X=M.join(F,on='MVT_ID_mvt')
# offsets BLOCK - SCHED etc (= d_sched - y)
X=X.with_columns((pl.col('d_sched')-pl.col('y')).alias('bs'),(pl.col('d_lobt')-pl.col('y')).alias('bl'),(pl.col('d_eobt')-pl.col('y')).alias('be'))
Bx=X.filter(pl.col('e').abs()>600)
for c in ['bs','bl','be']:
    print(c, g(Bx.with_columns(q(c,[-1800,-600,-300,-120,-60,-12,12,60,120,300,600,1800])),[c+'_b']))
# per airport: copy any
X=X.with_columns((pl.col('eqS12')|pl.col('eqL12')|pl.col('eqE12')|pl.col('eqI12')).alias('cp'))
# oracle gain: best of targets
X=X.with_columns(pl.when(pl.col('eqS12')).then(pl.col('d_sched')).when(pl.col('eqL12')).then(pl.col('d_lobt')).when(pl.col('eqE12')).then(pl.col('d_eobt')).otherwise(pl.col('d_iobt')).alias('tgt'))
print(X.group_by('ADEP_mvt').agg(pl.len(),pl.col('cp').mean().round(3).alias('cprate'),
   (pl.when(pl.col('cp')).then(pl.col('e')**2-(pl.col('tgt')-pl.col('y'))**2).otherwise(0).sum()/N).round(1).alias('oracle_gain'),
   (pl.when(pl.col('cp')).then(pl.col('e')**2).otherwise(0).sum()/N).round(1).alias('cp_sseN'),
   (pl.when(pl.col('cp')&(pl.col('e').abs()>600)).then(1).otherwise(0).sum()).alias('cp_big'),
   ).sort('ADEP_mvt'))
X.select('MVT_ID_mvt','cp','tgt','bs','bl','be').write_parquet(S+'/agents/matched/flags2.parquet')

# ---- a9
F=pl.read_parquet(S+"/agents/matched/flags.parquet")
X=M.join(F,on='MVT_ID_mvt')
# ADS-B event nearest to BLOCK vs AOBT: use d_push (gated) and d_move
for ev in ['d_push','d_move','d_gs3']:
    Y=X.filter(pl.col(ev).is_not_null()).with_columns((pl.col(ev)-pl.col('y')).abs().alias('eb'),(pl.col(ev)-pl.col('d_aobt')).abs().alias('ea'))
    Y=Y.with_columns(((pl.col('ea')>600)&(pl.col('eb')<=180)).alias('aobt_wrong'),((pl.col('eb')>600)&(pl.col('ea')<=180)).alias('block_wrong'))
    print(ev, Y.height, g(Y,['aobt_wrong','block_wrong']))
    print(Y.filter('aobt_wrong').group_by('ADEP_mvt').agg(pl.len(),((pl.col('e')**2).sum()/N).round(1).alias('mseN'),pl.col('e').mean().round(0)).sort('ADEP_mvt'))
# hour shifts
print(g(X.filter(pl.col('hshift')),['hk']))
print(g(X.filter(pl.col('hshift')),['ADEP_mvt']))
# disruption days among big rows
Bx=X.filter(pl.col('e').abs()>600)
dd=Bx.group_by('ADEP_mvt','day').agg(pl.len(),((pl.col('e')**2).sum()/SSE*100).round(2).alias('pctSSE')).sort('pctSSE',descending=True)
print(dd.head(25)); print('top25 days share', dd.head(25)['pctSSE'].sum(), 'top100', dd.head(100)['pctSSE'].sum(), 'ndays', dd.height)

# ---- b4
F=pl.read_parquet(S+"/agents/matched/flags.parquet")
X=M.join(F,on='MVT_ID_mvt')
ev=['d_push','d_move','d_gs3','d_first']
X=X.with_columns(pl.min_horizontal([(pl.col(c)-pl.col('y')).abs() for c in ['d_push','d_move','d_gs3']]).alias('adsb_to_blk'),
                 pl.min_horizontal([(pl.col(c)-pl.col('d_aobt')).abs() for c in ['d_push','d_move','d_gs3']]).alias('adsb_to_aobt'))
cat=(pl.when((pl.col('ADEP_mvt')=='LIRF')&(pl.col('eqS12')|pl.col('eqL12'))).then(pl.lit('a_LIRF_copy'))
 .when(pl.col('eqS12')|pl.col('eqL12')|pl.col('eqE12')).then(pl.lit('b_copy_other'))
 .when(pl.col('hshift')).then(pl.lit('c_hour_shift'))
 .when(pl.col('adsb_to_blk').is_null()).then(pl.lit('d_no_adsb_event'))
 .when(pl.col('adsb_to_blk')<=180).then(pl.lit('e_adsb_agrees_BLOCK'))
 .when(pl.col('adsb_to_aobt')<=300).then(pl.lit('f_adsb_agrees_AOBT_not_BLOCK'))
 .otherwise(pl.lit('g_adsb_neither'))).alias('cat')
Bx=X.filter(pl.col('e').abs()>600).with_columns(cat,(pl.col('e')>0).alias('pos'))
print(g(Bx,['cat']))
print(g(Bx.filter(pl.col('e').abs()>1800),['cat']))
print(g(Bx,['cat','pos']))
Bx.select('MVT_ID_mvt','cat','ym','ADEP_mvt').write_parquet(S+'/agents/matched/bigcat.parquet')
# category e per month
print(Bx.filter(pl.col('cat')=='e_adsb_agrees_BLOCK').group_by('ym').agg(pl.len(),((pl.col('e')**2).sum()/N).round(1)).sort('ym'))

# ---- b5
X=M.with_columns((pl.col('d_lobt')+3606-pl.col('p')).alias('room_up'),(pl.col('p')-(pl.col('d_lobt')-3606)).alias('room_dn'))
for c in ['room_up','room_dn']:
    print(X.group_by(pl.col(c).cut([0,120,300,600,900,1200,1800])).agg(pl.len(),(-pl.col('e')).mean().round(1).alias('mean_y_minus_p'),((pl.col('e')**2).sum()/N).round(1).alias('mseN'),
       (pl.col('e').abs()>600).sum().alias('nbig')).sort(c))

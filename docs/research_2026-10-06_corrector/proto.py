"""Stage-2 corrector prototype on the 4-month OOF (Jan, Jul, Nov, Dec 2025) of a2o+L6.
Target r = y - oof (clipped), Huber loss, leave-one-month-out over the 4 months. Matched rows only.
Inputs: oof, distances from oof to the planned clocks (dist_plano), clocks, L6-like context already in the frame,
categoricals, leave-month-out cell stats of r (airport x stand x runway, airport x hour). Reports dMSE with an
airport-day block bootstrap and the per-month split."""
import sys, numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, '/home/user/prc-data-challenge/src')
from prc.compare import block_bootstrap_rmse_diff
R = '/home/user/prc-data-challenge/'
OOF = sys.argv[1] if len(sys.argv) > 1 else 'main_flseq_osm_nolirf_l2_opt_l6-177174f530'
o = pl.read_parquet(R + f'experiments/oof_{OOF}.parquet')
cols = ['MVT_ID_mvt', 'TAXITIME_SEC_mvt', 'ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt', 'MVT_TIME_UTC_mvt',
        'hour_local', 'dow', 'mvt_minus_aobt3', 'mvt_minus_sched', 'mvt_minus_eobt1', 'mvt_minus_lobt', 'mvt_minus_iobt',
        'aobt3_minus_sched', 'nbr_aobt_delay_mean_60m', 'nbr_mvt_minus_aobt_mean_60m', 'dep_tko_next_30m', 'temp_c', 'vis_km']
f = pl.read_parquet(R + 'data/work/feat_train.parquet', columns=cols)
d = f.join(o, on='MVT_ID_mvt').filter(pl.col('oof').is_not_null() & pl.col('TAXITIME_SEC_mvt').is_not_null() & pl.col('mvt_minus_aobt3').is_not_null())
d = d.with_columns(pl.col('MVT_TIME_UTC_mvt').dt.month().alias('mo'), pl.col('MVT_TIME_UTC_mvt').dt.date().cast(pl.Utf8).alias('day'))
clk = ['mvt_minus_aobt3', 'mvt_minus_sched', 'mvt_minus_eobt1', 'mvt_minus_lobt', 'mvt_minus_iobt']
d = d.with_columns([(pl.col('oof') - pl.col(c)).alias(f'dp_{c}') for c in clk])
d = d.with_columns(pl.min_horizontal([pl.col(f'dp_{c}').abs() for c in clk]).alias('dp_min'),
                   ((pl.col('TAXITIME_SEC_mvt') - pl.col('oof')).clip(-1800, 1800)).alias('r'))
print('rows', d.height, 'months', d['mo'].unique().sort().to_list())
cats = ['ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt']
for c in cats: d = d.with_columns(pl.col(c).cast(pl.Utf8).fill_null('NA'))
num = ['oof', 'hour_local', 'dow', 'aobt3_minus_sched', 'nbr_aobt_delay_mean_60m', 'nbr_mvt_minus_aobt_mean_60m', 'dep_tko_next_30m',
       'temp_c', 'vis_km', 'dp_min'] + clk + [f'dp_{c}' for c in clk]
P = dict(objective='huber', alpha=300.0, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=1, seed=0)
pred = np.zeros(d.height); months = sorted(d['mo'].unique().to_list())
mo = d['mo'].to_numpy()
for m in months:
    tr, te = d.filter(pl.col('mo') != m), d.filter(pl.col('mo') == m)
    # leave-month-out cell stats of r, fitted on the training months only
    def cell(keys, name):
        t = tr.group_by(keys).agg(pl.col('r').mean().alias(name), pl.len().alias(name + '_n')).filter(pl.col(name + '_n') >= 30)
        return t
    extra = []
    for keys, name in ((['ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt'], 'c_srr'), (['ADEP_mvt', 'hour_local'], 'c_ah'), (['ADEP_mvt', 'AIRCRAFT_TYPE_mvt'], 'c_at')):
        t = cell(keys, name); te = te.join(t, on=keys, how='left')
        # training side: nested leave-month-out so the stat never sees its own month
        parts = []
        for m2 in tr['mo'].unique().to_list():
            t2 = tr.filter(pl.col('mo') != m2).group_by(keys).agg(pl.col('r').mean().alias(name), pl.len().alias(name + '_n')).filter(pl.col(name + '_n') >= 30)
            parts.append(tr.filter(pl.col('mo') == m2).join(t2, on=keys, how='left'))
        tr = pl.concat(parts, how='diagonal_relaxed'); extra += [name, name + '_n']
    feats = num + extra + cats
    def X(df):
        x = df.select(feats).to_pandas()
        for c in cats: x[c] = x[c].astype('category')
        return x
    allcat = {c: sorted(set(tr[c].to_list()) | set(te[c].to_list())) for c in cats}
    xa, xb = X(tr), X(te)
    for c in cats:
        xa[c] = xa[c].cat.set_categories(allcat[c]); xb[c] = xb[c].cat.set_categories(allcat[c])
    mdl = lgb.train(P, lgb.Dataset(xa, tr['r'].to_numpy(), categorical_feature=cats), 600)
    idx = np.flatnonzero(mo == m)
    order = te['MVT_ID_mvt'].to_numpy()
    pos = dict(zip(d['MVT_ID_mvt'].to_numpy()[idx], idx))
    pred[[pos[i] for i in order]] = mdl.predict(xb)
y = d['TAXITIME_SEC_mvt'].to_numpy().astype(float); a = d['oof'].to_numpy()
blk = (d['ADEP_mvt'] + d['day']).to_numpy()
for shrink in (1.0, 0.7, 0.5):
    b = a + shrink * pred
    r = block_bootstrap_rmse_diff(y, a, b, blk, n_boot=500)
    per = {m: round(float(np.sqrt(np.mean((y[mo==m]-b[mo==m])**2)) - np.sqrt(np.mean((y[mo==m]-a[mo==m])**2))), 2) for m in months}
    body = y <= 3600
    rb = np.sqrt(np.mean((y[body]-b[body])**2)) - np.sqrt(np.mean((y[body]-a[body])**2))
    print(f'shrink {shrink}: dRMSE {r["diff_rmse"]:+.3f} [{r["ci95"][0]:+.3f},{r["ci95"][1]:+.3f}] per-month {per} body {rb:+.3f}')

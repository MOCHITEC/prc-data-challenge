"""NM-blind direct model for NM-unmatched rows outside LIRF (merry-mushroom idea), blended with current model B.
Folds mirror the OOF: jan (train: other 11 months), jul (other 11), forward (Nov+Dec; train: Jan-Oct).
NM-blind = LightGBM on ALL labelled departures with 0 < y <= 7200, every NM-clock / proxy-derived column removed,
subsampled to 600k rows per fold; evaluated on unmatched non-LIRF rows of the validation months."""
import sys, numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, '/home/user/prc-data-challenge/src')
from prc.compare import block_bootstrap_rmse_diff
from prc.train import _NM_CLOCK_PREFIXES, LANE_A_ONLY_PREFIXES, _LABEL_DERIVED_PREFIXES
R = '/home/user/prc-data-challenge/'
sch = pl.read_parquet_schema(R + 'data/work/feat_train.parquet')
drop_pref = _NM_CLOCK_PREFIXES + LANE_A_ONLY_PREFIXES + _LABEL_DERIVED_PREFIXES + ('aobt3', 'eobt1', 'lobt', 'iobt', 'nbr_', 'fw_', 'tx_')
nm_cols = [c for c in sch if c.endswith('_flt')]
num = [c for c, t in sch.items() if t.is_numeric() and c not in ('TAXITIME_SEC_mvt', 'MVT_ID_mvt', 'FLIGHT_ID_mvt', 'month')
       and not c.startswith(drop_pref) and c not in nm_cols]
cats = ['ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt']
f = pl.read_parquet(R + 'data/work/feat_train.parquet', columns=['MVT_ID_mvt', 'TAXITIME_SEC_mvt', 'MVT_TIME_UTC_mvt', 'mvt_minus_aobt3'] + num + cats)
f = f.with_columns([pl.col(c).cast(pl.Float32) for c in num] + [pl.col(c).cast(pl.Utf8).fill_null('NA') for c in cats]
                   + [pl.col('MVT_TIME_UTC_mvt').dt.month().alias('mo')])
print('features', len(num), '+', len(cats), 'rows', f.height)
o = pl.read_parquet(R + 'experiments/oof_main_flseq_osm_nolirf_l2_opt-7244b3004f.parquet')
f = f.join(o, on='MVT_ID_mvt', how='left')
levels = {c: sorted(f[c].unique().to_list()) for c in cats}
def X(d):
    x = d.select(num + cats).to_pandas()
    for c in cats: x[c] = x[c].astype('category').cat.set_categories(levels[c])
    return x
P = dict(objective='regression', learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=2, seed=0)
folds = {'jan': ([1], [m for m in range(1, 13) if m != 1]), 'jul': ([7], [m for m in range(1, 13) if m != 7]), 'forward': ([11, 12], list(range(1, 11)))}
res = []
for name, (va, trm) in folds.items():
    tr = f.filter(pl.col('mo').is_in(trm) & pl.col('TAXITIME_SEC_mvt').is_between(1, 7200))
    tr = tr.sample(min(600000, tr.height), seed=0)
    te = f.filter(pl.col('mo').is_in(va) & pl.col('mvt_minus_aobt3').is_null() & (pl.col('ADEP_mvt') != 'LIRF')
                  & pl.col('TAXITIME_SEC_mvt').is_not_null() & pl.col('oof').is_not_null())
    m = lgb.train(P, lgb.Dataset(X(tr), tr['TAXITIME_SEC_mvt'].to_numpy(), categorical_feature=cats), 800)
    res.append(te.select('MVT_ID_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'TAXITIME_SEC_mvt', 'oof').with_columns(
        pl.Series('nmb', m.predict(X(te))), pl.lit(name).alias('fold')))
    print(name, 'train', tr.height, 'eval unmatched non-LIRF', te.height, flush=True)
d = pl.concat(res)
d.write_parquet('/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/nmblind/preds.parquet')
y = d['TAXITIME_SEC_mvt'].to_numpy().astype(float); a = d['oof'].to_numpy(); b = d['nmb'].to_numpy()
fo = d['fold'].to_numpy(); blk = (d['ADEP_mvt'] + d['MVT_TIME_UTC_mvt'].dt.date().cast(pl.Utf8)).to_numpy()
r = lambda p, s=slice(None): float(np.sqrt(np.mean((y[s] - p[s]) ** 2)))
print(f'rows {len(y)}; model B {r(a):.1f}; NM-blind alone {r(b):.1f}')
N_ALL = 672428  # rows in the pooled OOF (jan+jul+forward), for whole-frame scaling
for w in (0.25, 0.5, 0.75):
    p = (1 - w) * a + w * b
    x = block_bootstrap_rmse_diff(y, a, p, blk, n_boot=1000)
    per = {k: round(r(p, fo == k) - r(a, fo == k), 1) for k in ('jan', 'jul', 'forward')}
    dmse = np.sum((y - p) ** 2 - (y - a) ** 2) / N_ALL
    print(f'w={w}: unmatched non-LIRF {r(a):.1f} -> {r(p):.1f} ({x["diff_rmse"]:+.1f} [{x["ci95"][0]:+.1f},{x["ci95"][1]:+.1f}]) per fold {per}; whole-frame approx {dmse/(2*271):+.3f} s')

"""LIRF copy-overlay gate (|MVT-SCHED - (MVT-LOBT)| > 3606 s -> q = 0) ported to v4, and the +1x MVT-dither step."""
import polars as pl, numpy as np, sys
SP = '/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'
k, t = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
v2 = pl.read_parquet(f'{SP}/gate/final_from_v48_cb3_appear_cat_c13_v2.parquet').rename({t: 'v2'})
v4 = pl.read_parquet(f'{SP}/gate/final_recommended.parquet')
G = pl.read_parquet(f'{SP}/ideas/ht/lirf_copy_gate_rows.parquet')  # MVT_ID, month, old (v2), new (v2 gated)
R = (pl.scan_parquet('data/raw/final_ranking.parquet').filter(pl.col('PHASE_mvt') == 'DEP')
     .select(k, 'ADEP_mvt', pl.col('AOBT_3_flt').is_not_null().alias('mat'), pl.col('MVT_TIME_UTC_mvt').dt.month().alias('mon'),
             ((pl.col('MVT_TIME_UTC_mvt') - pl.col('LOBT_flt')).dt.total_seconds()).cast(pl.Float64).alias('mml'),
             ((pl.col('MVT_TIME_UTC_mvt') - pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds()).cast(pl.Float64).alias('ms')).collect())
chk = G.join(v2, on=k)
assert chk.height == G.height and (chk['old'] == chk['v2']).all(), 'gate table old != v2'
X = G.join(R, on=k)
assert X.height == G.height and (X['ADEP_mvt'] == 'LIRF').all() and X['mat'].all()
assert ((X['ms'] - X['mml']).abs() > 3606).all(), 'a gate row is not impossible by the identity'
D = G.select(k, 'month', (pl.col('new') - pl.col('old')).cast(pl.Float64).alias('gd'))
lo, hi = (pl.col('mml') - 3606).ceil(), (pl.col('mml') + 3606).ceil()

def apply_gate(sub, months):
    d = D.filter(pl.col('month').is_in(months))
    Z = sub.join(d.select(k, 'gd'), on=k, how='left', maintain_order='left').join(R.select(k, 'mml'), on=k, how='left', maintain_order='left')
    p = pl.col(t).cast(pl.Float64) + pl.col('gd').fill_null(0)
    p = pl.when(pl.col('gd').is_not_null() & pl.col('mml').is_not_null()).then(pl.max_horizontal(pl.min_horizontal(p, hi), lo)).otherwise(p)
    return Z.select(k, p.clip(0).round(0).cast(pl.Int32).alias(t))

def apply_dither(sub):
    old = pl.scan_parquet('data/raw/ranking.parquet').filter(pl.col('PHASE_mvt') == 'DEP').select(k, pl.col('MVT_TIME_UTC_mvt').alias('t_old')).collect()
    nw = pl.scan_parquet('data/raw/final_ranking.parquet').filter(pl.col('PHASE_mvt') == 'DEP').select(k, 'MVT_TIME_UTC_mvt', pl.col('AOBT_3_flt').is_not_null().alias('mat')).collect()
    j = nw.join(old, on=k).with_columns(delta=(pl.col('MVT_TIME_UTC_mvt') - pl.col('t_old')).dt.total_seconds()).filter('mat').select(k, 'delta')
    Z = sub.join(j, on=k, how='left', maintain_order='left').join(R.select(k, 'mml'), on=k, how='left', maintain_order='left')
    p = pl.col(t).cast(pl.Float64) + pl.col('delta').fill_null(0).cast(pl.Float64)
    p = pl.when(pl.col('delta').is_not_null() & pl.col('mml').is_not_null()).then(pl.max_horizontal(pl.min_horizontal(p, hi), lo)).otherwise(p)
    return Z.select(k, p.clip(0).round(0).cast(pl.Int32).alias(t))

# 1) reproduce the integrator's +1x file from v4
opt = pl.read_parquet(f'{SP}/gate/optional_janjul_reanchor_plus1delta.parquet')
mine = apply_dither(v4)
print('dither re-implementation == integrator optional file:', mine.equals(opt))
tpl = pl.read_parquet('data/raw/final_submitting.parquet')
out = {}
for tag, months in (('full', [1, 2, 6, 7]), ('fj', [2, 6])):
    g = apply_gate(v4, months)
    gd = apply_dither(g)
    for name, f in ((f'v4_gate{tag}', g), (f'v4_gate{tag}_plus1delta', gd)):
        f = tpl.select(k).join(f, on=k, how='left', maintain_order='left').cast(tpl.schema)
        assert f.height == tpl.height and f.null_count().sum_horizontal().item() == 0
        f.write_parquet(f'{SP}/gate/{name}.parquet'); out[name] = f
    dd = (g[t].cast(pl.Float64) - v4[t].cast(pl.Float64)).to_numpy()
    mon = v4.join(R.select(k, 'mon'), on=k, how='left', maintain_order='left')['mon'].to_numpy()
    print(f'gate {tag}: rows changed {int((dd != 0).sum())}, sum d^2/N {float((dd**2).sum())/len(dd):.1f}, by month',
          {m: int(((dd != 0) & (mon == m)).sum()) for m in (1, 2, 6, 7)}, 'max|d|', int(np.abs(dd).max()))
    x = (gd[t].cast(pl.Float64) - opt[t].cast(pl.Float64)).to_numpy()
    print(f'  {tag}+1delta vs integrator optional: rows differ {int((x != 0).sum())}')

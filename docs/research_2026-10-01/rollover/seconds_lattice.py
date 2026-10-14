"""Does knowing y mod 60 (from the opposite-sign second jitter on MVT/BLOCK) reduce RMSE?

1. verify the jitter rule per airport (training, labelled)
2. fit a realistic matched-lane LightGBM (target y - proxy) on non-scored months, predict Jan+Jul 2025
3. snap predictions with the Bayes posterior mean over the allowed lattice, using a residual density
   estimated on an inner holdout month (never the evaluation months); measure RMSE change
"""

import numpy as np
import polars as pl
import lightgbm as lgb

GRID = ["EDDM", "EGLL", "LFPG", "LIRF", "LTFM"]
f = pl.read_parquet("data/work/feat_train.parquet")
f = f.filter(pl.col("TAXITIME_SEC_mvt").is_not_null())


def off(col):  # signed second offset in [-30, 29]
    s = pl.col(col).dt.second()
    return pl.when(s >= 30).then(s - 60).otherwise(s)


f = f.with_columns(
    off("MVT_TIME_UTC_mvt").alias("oM"),
    off("BLOCK_TIME_UTC_mvt").alias("oB"),
    pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
)
print("== 1. jitter rule: share of rows with oM + oB in {-1, 0}")
print(
    f.group_by("ADEP_mvt")
    .agg(((pl.col("oM") + pl.col("oB")).is_in([-1, 0])).mean().alias("rule"), pl.len())
    .sort("ADEP_mvt")
)

# lattice: y = MVT - BLOCK; BLOCK sec = -oM or -oM-1 => y ≡ 2*oM or 2*oM+1 (mod 60)
f = f.with_columns(
    ((pl.col("TAXITIME_SEC_mvt") - 2 * pl.col("oM")) % 60).alias("ymod_rel"),
)
g = f.filter(pl.col("ADEP_mvt").is_in(GRID))
print("y - 2*oM mod 60 in {0,1}:", float(g["ymod_rel"].is_in([0, 1]).mean()), "n", g.height)
print("  split 0/1:", g["ymod_rel"].value_counts().sort("count", descending=True).head(4))

# 2. realistic matched-lane model
m = f.filter(pl.col("mvt_minus_aobt3").is_not_null())
num = [c for c, d in m.schema.items() if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.Int16, pl.UInt32, pl.UInt8)
       and c not in ("TAXITIME_SEC_mvt", "MVT_ID_mvt", "FLIGHT_ID_mvt", "month", "mo", "oB", "ymod_rel")]
tr = m.filter(~pl.col("mo").is_in([1, 6, 7]))  # June = inner holdout for the residual density
ho = m.filter(pl.col("mo") == 6)
te = m.filter(pl.col("mo").is_in([1, 7]))


def X(d):
    return d.select([pl.col(c).cast(pl.Float32) for c in num]).to_numpy()


tgt = lambda d: (d["TAXITIME_SEC_mvt"] - d["mvt_minus_aobt3"]).to_numpy().astype(float)  # noqa: E731
params = dict(objective="regression", learning_rate=0.08, num_leaves=127, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
bst = lgb.train(params, lgb.Dataset(X(tr), tgt(tr)), num_boost_round=600)
p_ho = ho["mvt_minus_aobt3"].to_numpy() + bst.predict(X(ho))
p_te = te["mvt_minus_aobt3"].to_numpy() + bst.predict(X(te))
y_ho, y_te = ho["TAXITIME_SEC_mvt"].to_numpy().astype(float), te["TAXITIME_SEC_mvt"].to_numpy().astype(float)
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))  # noqa: E731
print(f"\n== 2. matched-lane model: holdout(Jun) RMSE {rmse(y_ho, p_ho):.2f}, Jan+Jul RMSE {rmse(y_te, p_te):.2f}")

# 3. residual density at 1 s resolution from the June holdout (window +-W)
W = 3000
r = np.clip(np.round(y_ho - p_ho).astype(int), -W, W)
dens = np.bincount(r + W, minlength=2 * W + 1).astype(float) + 0.5
dens /= dens.sum()
# also a residual-*conditional* density: residuals are not iid - near-copy rows have tiny residuals.
# Use density conditional on |p - proxy| band (proxy = mvt_minus_aobt3) as a second variant.


def snap(p, oM, allowed_offsets, density):
    """posterior mean of y given p, y ≡ 2*oM + k (mod 60) for k in allowed_offsets"""
    out = np.empty_like(p)
    base = np.round(p).astype(int)
    ks = np.arange(-W, W + 1)
    for i in range(len(p)):
        cand = base[i] + ks
        ok = np.isin((cand - 2 * oM[i]) % 60, allowed_offsets)
        w = density[(cand - p[i]).round().astype(int).clip(-W, W) + W] * ok
        out[i] = (w * cand).sum() / w.sum() if w.sum() > 0 else p[i]
    return out


te_g = te.with_columns(pl.Series("p", p_te)).filter(pl.col("ADEP_mvt").is_in(GRID))
rng = np.random.default_rng(0)
idx = rng.choice(te_g.height, size=min(60000, te_g.height), replace=False)
s = te_g[idx]
p, oM, y = s["p"].to_numpy(), s["oM"].to_numpy(), s["TAXITIME_SEC_mvt"].to_numpy().astype(float)
p_snap2 = snap(p, oM, [0, 1], dens)  # what the jitter actually reveals (two values)
ymod = ((y - 2 * oM) % 60).astype(int)
p_or = np.array([snap(p[i:i + 1], oM[i:i + 1], [ymod[i]], dens)[0] for i in range(len(p))])  # oracle: exact residue
print(f"\n== 3. grid airports, Jan+Jul sample n={len(p)}")
print(f"  base RMSE            {rmse(y, p):.3f}")
print(f"  snapped (2 residues) {rmse(y, p_snap2):.3f}   diff {rmse(y, p_snap2) - rmse(y, p):+.3f}")
print(f"  oracle exact residue {rmse(y, p_or):.3f}   diff {rmse(y, p_or) - rmse(y, p):+.3f}")
small = np.abs(y - p) < 30
print(f"  rows with |err|<30s: {small.mean():.3%}; RMSE on them base {rmse(y[small], p[small]):.2f} snapped {rmse(y[small], p_snap2[small]):.2f}")
# copy-class rows: BLOCK within 60 s of AOBT_3 -> y close to proxy
cp = np.abs(y - s["mvt_minus_aobt3"].to_numpy()) <= 60
print(f"  aobt3-copy rows {cp.mean():.2%}: base {rmse(y[cp], p[cp]):.2f} snapped {rmse(y[cp], p_snap2[cp]):.2f}")
# If prediction were the copy candidate itself (perfect routing), how much would the lattice add?
pc = s["mvt_minus_aobt3"].to_numpy().astype(float)
pcs = snap(pc[cp], oM[cp], [0, 1], np.exp(-0.5 * (np.arange(-W, W + 1) / 4.0) ** 2) + 1e-12)
print(f"  copy rows routed perfectly: candidate RMSE {rmse(y[cp], pc[cp]):.2f} -> lattice-snapped {rmse(y[cp], pcs):.2f}")

"""Score blends of the heads saved by nn_blend_probe.py on jan+jul (the scored months)."""

import sys
from pathlib import Path

import numpy as np

from prc.clocks import classes, routed_prediction
from prc.cv import paired_bootstrap_rmse_diff, rmse

D = Path(sys.argv[1])
names = classes()
folds = {k: dict(np.load(D / f"{k}.npz", allow_pickle=True)) for k in ("jan", "jul") if (D / f"{k}.npz").exists()}
W = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def routed(d, w_taxi=0.0, a_clf=0.0, other="f_mlp"):
    f = (1 - w_taxi) * d["f_lgb"] + w_taxi * d[other]
    p = (1 - a_clf) * d["p_lgb"] + a_clf * d["p_mlp"]
    return np.clip(routed_prediction(p, d["cand"], f, names), 0, None)


print("== taxi head alone, rows whose BLOCK_TIME is a real off-block (genuine / aobt3), matched to NM ==")
for k, d in folds.items():
    m = np.isin(d["src"], ["genuine", "aobt3"]) & d["matched"]
    y = d["y"][m]
    r = {h: rmse(y, d[h][m]) for h in ("f_lgb", "f_mlp", "f_ridge")}
    e_l, e_m, e_r = (d["f_lgb"][m] - y), (d["f_mlp"][m] - y), (d["f_ridge"][m] - y)
    print(k, {h: round(v, 2) for h, v in r.items()}, "corr(lgb,mlp)=%.3f corr(lgb,ridge)=%.3f" % (
        np.corrcoef(e_l, e_m)[0, 1], np.corrcoef(e_l, e_r)[0, 1]))
    print("   blend lgb/mlp:", {w: round(rmse(y, (1 - w) * d["f_lgb"][m] + w * d["f_mlp"][m]), 2) for w in W})
    print("   blend lgb/ridge:", {w: round(rmse(y, (1 - w) * d["f_lgb"][m] + w * d["f_ridge"][m]), 2) for w in W})
    g = d["src"][m] == "genuine"
    print("   genuine only: lgb %.2f mlp %.2f 50/50 %.2f" % (
        rmse(y[g], d["f_lgb"][m][g]), rmse(y[g], d["f_mlp"][m][g]),
        rmse(y[g], 0.5 * d["f_lgb"][m][g] + 0.5 * d["f_mlp"][m][g])))

print("\n== full routed prediction (what the leaderboard scores) ==")
for k, d in folds.items():
    print(k, "baseline lgb/lgb %.2f" % rmse(d["y"], routed(d)))
    print("   taxi blend (mlp):", {w: round(rmse(d["y"], routed(d, w_taxi=w)), 2) for w in W})
    print("   taxi blend (ridge):", {w: round(rmse(d["y"], routed(d, w_taxi=w, other="f_ridge")), 2) for w in W})
    print("   clf blend:", {a: round(rmse(d["y"], routed(d, a_clf=a)), 2) for a in W})
    print("   classifier mlp-only p_genuine mean %.3f vs lgb %.3f" % (d["p_mlp"][:, 0].mean(), d["p_lgb"][:, 0].mean()))
    ll = lambda p: -np.mean(np.log(np.clip(p[np.arange(len(p)), idx], 1e-9, 1)))  # noqa: E731
    idx = np.array([names.index(s) if s in names else 0 for s in d["src"]])
    print("   clf logloss lgb %.4f mlp %.4f 50/50 %.4f" % (ll(d["p_lgb"]), ll(d["p_mlp"]), ll(0.5 * d["p_lgb"] + 0.5 * d["p_mlp"])))

if len(folds) == 2:
    print("\n== honest selection: pick weights on one month, score on the other ==")
    grid = [(w, a) for w in W for a in W]
    y = np.concatenate([folds[k]["y"] for k in ("jan", "jul")])
    base = np.concatenate([routed(folds[k]) for k in ("jan", "jul")])
    chosen = []
    for pick, score in (("jan", "jul"), ("jul", "jan")):
        best = min(grid, key=lambda g: rmse(folds[pick]["y"], routed(folds[pick], *g)))
        chosen.append((score, best))
        print(f"picked on {pick}: w_taxi={best[0]} a_clf={best[1]} -> {score}: "
              f"{rmse(folds[score]['y'], routed(folds[score], *best)):.2f} vs base {rmse(folds[score]['y'], routed(folds[score])):.2f}")
    new = np.concatenate([routed(folds[s], *g) for s, g in sorted(chosen, key=lambda t: ("jan", "jul").index(t[0]))])
    print("jan+jul pooled, cross-picked:", paired_bootstrap_rmse_diff(y, base, new))
    for g in ((0.3, 0.0), (0.5, 0.0), (0.0, 0.3), (0.3, 0.3)):
        new = np.concatenate([routed(folds[k], *g) for k in ("jan", "jul")])
        print(f"jan+jul pooled fixed w_taxi={g[0]} a_clf={g[1]}:", paired_bootstrap_rmse_diff(y, base, new))

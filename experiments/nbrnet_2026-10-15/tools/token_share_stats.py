"""Per-airport token statistics on a sample of departures: valid tokens, and the share of valid tokens that share
the query's runway / stand / stand prefix (PREREG §2 promises the prefix share)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nbrnet"))
from nbr_tokens import ARR_DYNAMIC, ARR_STATIC, DEP_DYNAMIC, DEP_STATIC, Tables, gather  # noqa: E402

d = Path(sys.argv[1])
tb = Tables.load(d)
idx_d, idx_a = np.load(d / "idx_d.npy", mmap_mode="r"), np.load(d / "idx_a.npy", mmap_mode="r")
rng = np.random.default_rng(0)
q = np.sort(rng.choice(len(tb.dep_t), 200_000, replace=False))
out = {}
apt = np.array([tb.groups[g][0] for g in tb.dep_group[q]])
for a in sorted(set(apt)):
    qq = q[apt == a]
    dep, md, arr, ma = gather(tb, qq, idx_d, idx_a, idx_d.shape[1], idx_a.shape[1])
    off = len(DEP_STATIC)
    r = {"n": int(len(qq)), "dep_tokens": float(md.sum(1).mean()), "arr_tokens": float(ma.sum(1).mean())}
    for j, name in enumerate(DEP_DYNAMIC):
        if name.startswith("same_"):
            r[f"dep_{name}"] = float(dep[..., off + j][md].mean())
    for j, name in enumerate(ARR_DYNAMIC):
        if name.startswith("same_"):
            r[f"arr_{name}"] = float(arr[..., len(ARR_STATIC) + j][ma].mean())
    out[a] = r
print(json.dumps(out, indent=1))

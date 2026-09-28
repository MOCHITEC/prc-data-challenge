"""What is proxy_sec_in_min actually doing? Read-only, no fit (C1-two-column-remeasured.md 3).

The mechanism (delay-mechanisms.md C1): NM sets AOBT_3 to take-off minus a NOMINAL PER-RUNWAY WHOLE-MINUTE
allowance. If that is what is happening, then on the rows where the flag fires (proxy_at_mode == 1) the
offset is a whole number of minutes and proxy_sec_in_min must be ~0 there. If it is uniform on those rows
too, the column is the take-off stamp's seconds and says nothing about provenance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from prc.train import LABEL, PROXY

WORK = Path("data/work_c1c3")
EXP = Path("experiments")
COLS = ["MVT_ID_mvt", LABEL, PROXY, "ADEP_mvt", "proxy_at_mode", "proxy_minus_mode_min", "proxy_sec_in_min"]

ids = {}
for line in (EXP / "experiments.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        ids[r["config"]["name"]] = r["exp_id"]

tr = pl.read_parquet(WORK / "feat_train.parquet", columns=COLS)
rk = pl.read_parquet(WORK / "feat_rank.parquet", columns=[c for c in COLS if c not in (LABEL,)])

print("== M1. the whole-minute prediction, on the rows where the flag fires ==")
print("if AOBT_3 = take-off - (a whole number of minutes), proxy_sec_in_min is 0 on those rows.\n")
for nm, f in (("train", tr), ("rank", rk)):
    d = f.filter(pl.col("proxy_at_mode").is_not_null())
    for lab, sel in (
        ("flag == 1", d.filter(pl.col("proxy_at_mode") == 1)),
        ("flag == 0", d.filter(pl.col("proxy_at_mode") == 0)),
    ):
        s = sel["proxy_sec_in_min"]
        zero = (s == 0).sum()
        print(
            f"  {nm:<5} {lab:<9} n={sel.height:>9,}  proxy_sec_in_min == 0 on {zero:>8,} ({zero / max(1, sel.height):6.2%})  "
            f"mean={s.mean():6.2f}  sd={s.std():5.2f}  (uniform 0-59 would be mean 29.5, sd 17.3, 1.67% zeros)"
        )
print()

print("== M2. is proxy_sec_in_min a whole-minute story anywhere? per airport, flag == 1 rows ==")
d = tr.filter(pl.col("proxy_at_mode") == 1)
g = (
    d.group_by("ADEP_mvt")
    .agg(
        pl.len().alias("n"),
        (pl.col("proxy_sec_in_min") == 0).mean().alias("zero_share"),
        pl.col("proxy_sec_in_min").mean().alias("mean_sec"),
    )
    .sort("n", descending=True)
)
print(f"  {'ADEP':>6} {'n':>9} {'sec==0':>8} {'mean sec':>9}")
for r in g.iter_rows(named=True):
    print(f"  {r['ADEP_mvt']:>6} {r['n']:>9,} {r['zero_share']:8.2%} {r['mean_sec']:9.2f}")
print()

print("== M3. what each column relates to: the base arm's out-of-fold residual ==")
oof = pl.read_parquet(EXP / f"oof_{ids['c13_base']}.parquet")
d = (
    tr.join(oof, on="MVT_ID_mvt", how="inner")
    .filter(pl.col(LABEL).is_not_null() & pl.col("oof").is_not_null())
    .drop_nulls(["proxy_at_mode"])
)
y = d[LABEL].to_numpy().astype(float)
res = y - d["oof"].to_numpy()
print(f"  rows = {d.height:,}")
for c in ("proxy_at_mode", "proxy_minus_mode_min", "proxy_sec_in_min"):
    v = d[c].to_numpy().astype(float)
    print(
        f"  {c:<22} corr(y) = {np.corrcoef(v, y)[0, 1]:+.5f}   corr(base OOF residual) = {np.corrcoef(v, res)[0, 1]:+.5f}"
    )

print("\n  mean base residual by flag value (seconds; the model over/under-predicts by this much):")
for val in (0, 1):
    sel = d.filter(pl.col("proxy_at_mode") == val)
    yv = sel[LABEL].to_numpy().astype(float)
    rv = yv - sel["oof"].to_numpy()
    print(
        f"    proxy_at_mode == {val}: n={sel.height:>8,}  mean residual {rv.mean():+8.2f}  RMSE {np.sqrt((rv**2).mean()):8.2f}"
    )

print("\n  mean base residual by decile of proxy_sec_in_min (a real nuisance signal would trend):")
sec = d["proxy_sec_in_min"].to_numpy().astype(float)
q = np.quantile(sec, np.linspace(0, 1, 11))
for i in range(10):
    m = (sec >= q[i]) & (sec <= q[i + 1] if i == 9 else sec < q[i + 1])
    if m.sum():
        print(f"    [{q[i]:5.1f}, {q[i + 1]:5.1f})  n={m.sum():>8,}  mean residual {res[m].mean():+8.2f}")

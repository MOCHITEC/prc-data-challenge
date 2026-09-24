"""operator_aircraft_route 10: summarise the 06c ablation (paired bootstrap vs base on pooled jan+jul OOF)."""

from __future__ import annotations

import json
import os

import numpy as np
import polars as pl
from operator_aircraft_route_common import INTERIM, load_dep, md, paired_boot


def main() -> None:
    logs = [json.loads(line) for line in open(os.path.join(INTERIM, "oar_ablation.jsonl"))]
    print("## A. fold 別（固定ラウンド・学習行 50% サブサンプル・同一行）\n")
    t = pl.DataFrame([{k: v for k, v in r.items() if not k.startswith("imp")} for r in logs])
    print(md(t.select("arm", "fold", "n", "rmse", "rmse_clean", "clf_logloss", "n_feats", "t_clf", "t_reg"), "{:,.3f}"))
    arms = [a for a in dict.fromkeys(t["arm"].to_list()) if os.path.exists(os.path.join(INTERIM, f"oar_oof_{a}.parquet"))]
    base = pl.read_parquet(os.path.join(INTERIM, "oar_oof_base.parquet"))
    meta = load_dep(["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt"])
    base = base.join(meta, on="MVT_ID_mvt", how="left")
    y, pb = base["y"].to_numpy(), base["pred"].to_numpy()
    nf = pl.read_parquet(os.path.join(INTERIM, "oar_newfeats.parquet"), columns=["MVT_ID_mvt", "stand_lb"])
    base = base.join(nf, on="MVT_ID_mvt", how="left", maintain_order="left")
    segs = {
        "全体": np.ones(len(y), bool),
        "clean (y<7200)": y < 7200,
        "NM 未マッチ": base["FLIGHT_ID_mvt"].is_null().to_numpy(),
        "LIRF": (base["ADEP_mvt"] == "LIRF").to_numpy(),
        "stand_lb あり": base["stand_lb"].is_not_null().to_numpy(),
        "sched クラス": (base["cls"] == "sched").to_numpy(),
        "genuine クラス": (base["cls"] == "genuine").to_numpy(),
    }
    for apt in sorted(base["ADEP_mvt"].unique().to_list()):
        segs[f"clean {apt}"] = (y < 7200) & (base["ADEP_mvt"] == apt).to_numpy()
    for fold in (1, 7):
        segs[f"clean fold{fold}"] = (y < 7200) & (base["MVT_ID_mvt"].is_in(
            load_dep(["MVT_ID_mvt", "ym"]).filter(pl.col("ym") % 100 == fold)["MVT_ID_mvt"].implode())).to_numpy()
    rows = []
    for a in arms:
        if a == "base":
            continue
        o = pl.read_parquet(os.path.join(INTERIM, f"oar_oof_{a}.parquet"))
        o = base.select("MVT_ID_mvt").join(o, on="MVT_ID_mvt", how="left", maintain_order="left")
        pa = o["pred"].to_numpy()
        for s, m in segs.items():
            d, lo, hi = paired_boot(y[m], pb[m], pa[m], n=400)
            rows.append((a, s, int(m.sum()), float(np.sqrt(np.mean((y[m] - pb[m]) ** 2))), float(np.sqrt(np.mean((y[m] - pa[m]) ** 2))),
                         d, lo, hi))
    print("\n## B. arm − base（jan+jul プール, 行ブートストラップ 400 回）\n")
    print(md(pl.DataFrame(rows, schema=["arm", "segment", "n", "rmse_base", "rmse_arm", "delta", "ci_lo", "ci_hi"], orient="row"),
             "{:,.2f}"))
    print("\n## C. 新特徴量の gain 比率（%、fold 平均）\n")
    rows = []
    for r in logs:
        for k, v in r["imp_reg_new"].items():
            rows.append((r["arm"], k, "reg", v))
        for k, v in r["imp_clf_new"].items():
            rows.append((r["arm"], k, "clf", v))
    if rows:
        g = pl.DataFrame(rows, schema=["arm", "feature", "head", "gain%"], orient="row")
        g = g.filter(pl.col("arm") == "all").group_by("feature", "head").agg(pl.col("gain%").mean()).pivot(
            on="head", index="feature", values="gain%").sort("reg", descending=True)
        print(md(g, "{:.3f}"))


if __name__ == "__main__":
    main()

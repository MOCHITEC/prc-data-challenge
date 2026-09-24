"""operator_aircraft_route 01: airline identity (CALLSIGN / FLIGHT_mvt prefix) vs taxi-out level and clock source.

Prints markdown tables (captured into docs/analysis/operator_aircraft_route.md).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from operator_aircraft_route_common import load_dep, md

SRC = ["genuine", "aobt3", "sched", "eobt1", "lobt", "iobt", "rollover_24h"]
HUB = {"EDDF": "DLH", "EDDM": "DLH", "EGLL": "BAW", "EHAM": "KLM", "LEBL": "VLG", "LEMD": "IBE", "LFPG": "AFR",
       "LIRF": "ITY", "LSZH": "SWR", "LTFM": "THY"}


def main() -> None:
    d = load_dep().with_columns(pl.col("FLIGHT_ID_mvt").is_not_null().alias("nm"))
    print("## A. 航空会社の同定率\n")
    cov = (
        d.group_by("split", "nm")
        .agg(
            pl.len().alias("rows"),
            pl.col("cs3").is_not_null().mean().alias("callsign3"),
            pl.col("fl_pfx").is_not_null().mean().alias("flight_prefix"),
            pl.col("airline").is_not_null().mean().alias("airline_any"),
        )
        .sort("split", "nm")
    )
    print(md(cov, "{:.4f}"))
    # is cs3 redundant with the hashed operator?
    t = d.filter(pl.col("nm")).group_by("AIRCRAFT_OPERATOR_flt").agg(pl.col("cs3").n_unique().alias("k"), pl.len())
    print(f"\nNM 行で AIRCRAFT_OPERATOR_flt 1 値あたりの cs3 種類数 = 1 の行比率: "
          f"{t.filter(pl.col('k') == 1)['len'].sum() / t['len'].sum():.4f}\n")

    tr = d.filter(pl.col("split") == "train")
    # --- B. hub vs non-hub
    tr = tr.with_columns(
        pl.struct("ADEP_mvt", "airline")
        .map_elements(lambda s: HUB.get(s["ADEP_mvt"]) == s["airline"], return_dtype=pl.Boolean)
        .alias("is_hub")
    )
    print("## B. ハブキャリア vs その他（学習 2025）\n")
    b = (
        tr.group_by("ADEP_mvt", "is_hub")
        .agg(
            pl.len().alias("n"),
            pl.col("y").median().alias("y_med"),
            pl.col("y").std().alias("y_sd"),
            (pl.col("clock_src") == "sched").mean().mul(100).alias("sched%"),
            (pl.col("clock_src") == "aobt3").mean().mul(100).alias("aobt3%"),
            (pl.col("clock_src") == "eobt1").mean().mul(100).alias("eobt1%"),
            (~pl.col("nm")).mean().mul(100).alias("nm_unmatched%"),
            (pl.col("y") > 3600).mean().mul(1e4).alias("gt1h_per10k"),
        )
        .sort("ADEP_mvt", "is_hub")
    )
    print(md(b, "{:,.2f}"))

    # --- C. airline effect beyond airport x stand (in-sample, top airlines)
    ref = tr.group_by("ADEP_mvt", "STAND_mvt").agg(pl.col("y").median().alias("stand_med"))
    tr = tr.join(ref, on=["ADEP_mvt", "STAND_mvt"], how="left").with_columns(
        (pl.col("y").clip(0, 3600) - pl.col("stand_med")).alias("r_stand")
    )
    print("\n## C. 主要航空会社（全空港合算, 上位 30）: 水準・ばらつき・出所\n")
    print("r_stand = clip(y,0,3600) − 空港×スタンド中央値 の中央値（スタンドで説明できない航空会社効果）\n")
    c = (
        tr.group_by("airline")
        .agg(
            pl.len().alias("n"),
            pl.col("ADEP_mvt").n_unique().alias("apts"),
            pl.col("y").median().alias("y_med"),
            pl.col("y").std().alias("y_sd"),
            pl.col("r_stand").median().alias("r_stand_med"),
            (pl.col("clock_src") == "genuine").mean().mul(100).alias("genuine%"),
            (pl.col("clock_src") == "aobt3").mean().mul(100).alias("aobt3%"),
            (pl.col("clock_src") == "sched").mean().mul(100).alias("sched%"),
            (pl.col("clock_src") == "eobt1").mean().mul(100).alias("eobt1%"),
            (pl.col("y") > 3600).mean().mul(1e4).alias("gt1h_per10k"),
        )
        .sort("n", descending=True)
        .head(30)
    )
    print(md(c, "{:,.1f}"))

    # airline x airport extremes of r_stand
    print("\n## D. 空港×航空会社でスタンド中央値から最も外れる組（n≥1000）\n")
    e = (
        tr.group_by("ADEP_mvt", "airline")
        .agg(pl.len().alias("n"), pl.col("r_stand").median().alias("r_stand_med"),
             pl.col("r_stand").mean().alias("r_stand_mean"),
             (pl.col("clock_src") == "sched").mean().mul(100).alias("sched%"),
             (pl.col("clock_src") == "aobt3").mean().mul(100).alias("aobt3%"))
        .filter(pl.col("n") >= 1000)
        .sort("r_stand_mean")
    )
    print(md(pl.concat([e.head(10), e.tail(12)]), "{:,.1f}"))

    # --- E. clock-source predictability out of time: log loss by grouping hierarchy
    print("\n## E. 出所ラベルの予測力（月ブロック: jan / jul を除いた月で率を fit → jan, jul で評価）\n")
    print("多クラス log loss（7 クラス, 加算平滑化 α=20 で親階層へ縮約）。小さいほど良い。\n")
    tr = tr.with_columns(pl.col("airline").fill_null("?"), pl.col("STAND_mvt").fill_null("?"),
                         pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hr"))
    rows = []
    for fold in (1, 7):
        fit = tr.filter(pl.col("ym") % 100 != fold)
        ev = tr.filter(pl.col("ym") % 100 == fold)
        hier = {
            "airport": [["ADEP_mvt"]],
            "airport>stand": [["ADEP_mvt"], ["ADEP_mvt", "STAND_mvt"]],
            "airport>airline": [["ADEP_mvt"], ["ADEP_mvt", "airline"]],
            "airport>stand>stand×airline": [["ADEP_mvt"], ["ADEP_mvt", "STAND_mvt"], ["ADEP_mvt", "STAND_mvt", "airline"]],
            "airport>airline>airline×stand": [["ADEP_mvt"], ["ADEP_mvt", "airline"], ["ADEP_mvt", "STAND_mvt", "airline"]],
            "airport>airline>airline×hour": [["ADEP_mvt"], ["ADEP_mvt", "airline"], ["ADEP_mvt", "airline", "hr"]],
        }
        for name, levels in hier.items():
            ll = smoothed_logloss(fit, ev, levels)
            rows.append((fold, name, *ll))
    res = pl.DataFrame(rows, schema=["fold_month", "hierarchy", "logloss_all", "logloss_nm_unmatched", "brier_sched"],
                       orient="row")
    print(md(res, "{:.4f}"))

    # --- F. NM-unmatched rows by airline
    print("\n## F. NM 未マッチ行（学習）: 航空会社別\n")
    u = tr.filter(~pl.col("nm"))
    f = (
        u.group_by("airline")
        .agg(pl.len().alias("n"), pl.col("ADEP_mvt").mode().first().alias("main_apt"),
             pl.col("y").median().alias("y_med"), pl.col("y").mean().alias("y_mean"), pl.col("y").std().alias("y_sd"),
             (pl.col("clock_src") == "sched").mean().mul(100).alias("sched%"),
             (pl.col("y") > 3600).sum().alias("n_gt1h"),
             ((pl.col("y") - 7200).clip(0) ** 2).sum().alias("_tail"))
        .sort("n", descending=True)
    )
    tot_tail = f["_tail"].sum()
    f = f.with_columns((pl.col("_tail") / tot_tail * 100).alias("tail_sse%")).drop("_tail")
    print(md(f.head(25), "{:,.1f}"))
    print("\n tail_sse% = Σ max(y−7200,0)² の航空会社別シェア（尾部の所在）\n")
    print(md(f.sort("tail_sse%", descending=True).head(12), "{:,.1f}"))

    rk = d.filter((pl.col("split") == "rank") & ~pl.col("nm"))
    print("\n ranking の NM 未マッチ行: 航空会社上位\n")
    print(md(rk.group_by("airline").agg(pl.len().alias("n"), pl.col("ADEP_mvt").mode().first().alias("main_apt"))
             .sort("n", descending=True).head(15)))


def smoothed_logloss(fit: pl.DataFrame, ev: pl.DataFrame, levels: list[list[str]], alpha: float = 20.0):
    K = len(SRC)
    glob = np.array([(fit["clock_src"] == s).mean() for s in SRC]) + 1e-6
    glob /= glob.sum()
    prior = np.tile(glob, (ev.height, 1))
    for keys in levels:
        cnt = fit.group_by(keys).agg([(pl.col("clock_src") == s).sum().alias(s) for s in SRC])
        j = ev.select(keys).join(cnt, on=keys, how="left", maintain_order="left")
        c = np.nan_to_num(j.select(SRC).to_numpy().astype(float))
        n = c.sum(1, keepdims=True)
        prior = (c + alpha * prior) / (n + alpha)
    yi = np.array([SRC.index(s) for s in ev["clock_src"].to_list()])
    p = np.clip(prior[np.arange(ev.height), yi], 1e-9, 1)
    ll = -np.log(p)
    unm = ev["FLIGHT_ID_mvt"].is_null().to_numpy()
    ps = prior[:, SRC.index("sched")]
    brier = np.mean((ps - (yi == SRC.index("sched"))) ** 2)
    return float(ll.mean()), float(ll[unm].mean()), float(brier)


if __name__ == "__main__":
    main()

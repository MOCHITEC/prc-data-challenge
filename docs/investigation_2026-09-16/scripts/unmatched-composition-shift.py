"""Item 17 `unmatched-composition-shift` (docs/investigation_2026-09-16/plan.md).

No model training. Lazy scans, only the columns needed. Run with the project venv:

    POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/unmatched-composition-shift.py

Both "unmatched" definitions of plan §0 are computed:
  routing  = `mvt_minus_aobt3` null  (rows sent to model B)
  flightid = `FLIGHT_ID_mvt` null    (D's definition)

Sections
  A. row counts / unmatched rate per set (train-all, train Jan+Jul 2025, rank Jan 2026, rank Jul 2026, rank all)
  B. composition of unmatched rows by airport, FLIGHT prefix (2 / 3 chars), FLIGHT_RULE, hour (UTC), "seen among
     matched training flights", AIRCRAFT_TYPE null — train vs rank, with total-variation distance per dimension;
     unmatched rate per airport (unmatched / departures at that airport)
  C. training label (y) median and share y >= 2 h by those subgroups (unmatched rows); composition-implied
     y >= 2 h share under the ranking mix
  D. OOF (experiments/oof_main-48c19bc830.parquet) unmatched SSE per (airport x month) cell, reweighted to the
     ranking (airport x month) composition -> expected unmatched RMSE under ranking composition (assumption:
     per-cell MSE on ranking rows equals per-cell OOF MSE; only the cell weights change)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")
import polars as pl  # noqa: E402

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_str_lengths(40)

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "data/work/feat_train.parquet"
RANK = ROOT / "data/work/feat_rank.parquet"
OOF = ROOT / "experiments/oof_main-48c19bc830.parquet"

LABEL, PROXY, TIME = "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "MVT_TIME_UTC_mvt"
H2 = 7200
DEFS = {"routing": pl.col(PROXY).is_null(), "flightid": pl.col("FLIGHT_ID_mvt").is_null()}

COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", PROXY, "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt",
    "AIRCRAFT_TYPE_mvt", TIME, LABEL,
]


def hdr(s: str) -> None:
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


def base(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .select(COLS)
        .with_columns(
            (pl.col(TIME).dt.year() * 100 + pl.col(TIME).dt.month()).alias("ym"),
            pl.col(TIME).dt.month().alias("mo"),
            pl.col(TIME).dt.hour().alias("hour"),
            pl.col("FLIGHT_mvt").str.slice(0, 2).alias("pre2"),
            pl.col("FLIGHT_mvt").str.slice(0, 3).alias("pre3"),
            pl.col("AIRCRAFT_TYPE_mvt").is_null().alias("actype_null"),
            pl.col(PROXY).is_null().alias("unm_routing"),
            pl.col("FLIGHT_ID_mvt").is_null().alias("unm_flightid"),
        )
    )


# ----------------------------------------------------------------------------------------------------------------
# load (small column subset; ~2.1M + 345k rows)
# ----------------------------------------------------------------------------------------------------------------
tr = base(TRAIN).collect()
rk = base(RANK).collect()

hdr("A. preconditions and row counts")
print("train rows", tr.height, "PHASE values", tr["PHASE_mvt"].value_counts().to_dicts())
print("rank  rows", rk.height, "PHASE values", rk["PHASE_mvt"].value_counts().to_dicts())
print("train ym", sorted(tr["ym"].unique().to_list()))
print("rank  ym", sorted(rk["ym"].unique().to_list()))
print("train MVT_ID unique", tr["MVT_ID_mvt"].n_unique(), "rank MVT_ID unique", rk["MVT_ID_mvt"].n_unique())
print("train label null", tr[LABEL].null_count(), "rank label null", rk[LABEL].null_count(), "(rank labels unavailable if == rows)")
print("train FLIGHT_mvt null", tr["FLIGHT_mvt"].null_count(), "rank FLIGHT_mvt null", rk["FLIGHT_mvt"].null_count())
print("train FLIGHT_RULE null", tr["FLIGHT_RULE_mvt"].null_count(), "rank", rk["FLIGHT_RULE_mvt"].null_count())

# matched training flights (routing definition; also FLIGHT_ID definition) for the "seen" flag
seen_sets = {
    "routing": set(tr.filter(~pl.col("unm_routing"))["FLIGHT_mvt"].drop_nulls().unique().to_list()),
    "flightid": set(tr.filter(~pl.col("unm_flightid"))["FLIGHT_mvt"].drop_nulls().unique().to_list()),
}
print("distinct FLIGHT_mvt among matched training rows: routing", len(seen_sets["routing"]), "flightid", len(seen_sets["flightid"]))

# sets
SETS = {
    "train_all": tr,
    "train_JanJul25": tr.filter(pl.col("mo").is_in([1, 7])),
    "rank_Jan26": rk.filter(pl.col("ym") == 202601),
    "rank_Jul26": rk.filter(pl.col("ym") == 202607),
    "rank_all": rk,
}

for d, expr in DEFS.items():
    rows = []
    for name, df in SETS.items():
        n = df.height
        u = df.filter(expr).height
        rows.append({"def": d, "set": name, "rows": n, "unmatched": u, "pct": round(100 * u / n, 3)})
    print(pl.DataFrame(rows))

# ----------------------------------------------------------------------------------------------------------------
# B. composition of unmatched rows
# ----------------------------------------------------------------------------------------------------------------
DIMS = ["ADEP_mvt", "pre2", "pre3", "FLIGHT_RULE_mvt", "hour", "seen", "actype_null"]


def unm_frames(d: str) -> dict[str, pl.DataFrame]:
    seen = pl.Series(sorted(seen_sets[d]))
    out = {}
    for name, df in SETS.items():
        out[name] = df.filter(DEFS[d]).with_columns(pl.col("FLIGHT_mvt").is_in(seen).alias("seen"))
    return out


def share_table(frames: dict[str, pl.DataFrame], dim: str, top: int | None = None) -> pl.DataFrame:
    tabs = []
    for name, df in frames.items():
        t = df.group_by(dim).agg(pl.len().alias("n")).with_columns((pl.col("n") / df.height * 100).alias(f"{name}"))
        tabs.append(t.select([dim, f"{name}", pl.col("n").alias(f"n_{name}")]))
    out = tabs[0]
    for t in tabs[1:]:
        out = out.join(t, on=dim, how="full", coalesce=True)
    out = out.fill_null(0)
    out = out.with_columns((pl.col("rank_all") - pl.col("train_all")).alias("d_all_pp"), (pl.col("rank_all") - pl.col("train_JanJul25")).alias("d_JJ_pp"))
    out = out.sort("rank_all", descending=True)
    if top:
        out = out.head(top)
    share_cols = [c for c in out.columns if c in SETS]
    return out.with_columns([pl.col(c).round(2) for c in share_cols + ["d_all_pp", "d_JJ_pp"]])


def tv_distance(frames: dict[str, pl.DataFrame], dim: str, a: str, b: str) -> float:
    ta = frames[a].group_by(dim).agg(pl.len().alias("na")).with_columns(pl.col("na") / frames[a].height)
    tb = frames[b].group_by(dim).agg(pl.len().alias("nb")).with_columns(pl.col("nb") / frames[b].height)
    j = ta.join(tb, on=dim, how="full", coalesce=True).fill_null(0)
    return float(0.5 * (j["na"] - j["nb"]).abs().sum())


UF = {d: unm_frames(d) for d in DEFS}

for d in DEFS:
    hdr(f"B. composition of unmatched rows — definition: {d}")
    frames = UF[d]
    for dim, top in [("ADEP_mvt", None), ("pre2", 15), ("pre3", 20), ("FLIGHT_RULE_mvt", None), ("hour", None), ("seen", None), ("actype_null", None)]:
        print(f"\n--- {dim} (share % of unmatched rows in each set; top {top or 'all'} by rank_all) ---")
        print(share_table(frames, dim, top))
    print("\n--- total-variation distance between compositions (0 = identical, 1 = disjoint) ---")
    rows = []
    for dim in DIMS:
        rows.append({
            "dim": dim,
            "rank_all vs train_all": round(tv_distance(frames, dim, "rank_all", "train_all"), 4),
            "rank_all vs train_JanJul25": round(tv_distance(frames, dim, "rank_all", "train_JanJul25"), 4),
            "rank_Jan26 vs train_all": round(tv_distance(frames, dim, "rank_Jan26", "train_all"), 4),
            "rank_Jul26 vs train_all": round(tv_distance(frames, dim, "rank_Jul26", "train_all"), 4),
            "rank_Jan26 vs rank_Jul26": round(tv_distance(frames, dim, "rank_Jan26", "rank_Jul26"), 4),
            "train_JanJul25 vs train_all": round(tv_distance(frames, dim, "train_JanJul25", "train_all"), 4),
        })
    print(pl.DataFrame(rows))

    print("\n--- unmatched rate per airport (unmatched / departures at airport, %) ---")
    rows = []
    for name, df in SETS.items():
        t = df.group_by("ADEP_mvt").agg(pl.len().alias("dep"), DEFS[d].sum().alias("unm")).with_columns((pl.col("unm") / pl.col("dep") * 100).round(3).alias(name))
        rows.append(t.select(["ADEP_mvt", name]))
    out = rows[0]
    for t in rows[1:]:
        out = out.join(t, on="ADEP_mvt", how="full", coalesce=True)
    print(out.sort("ADEP_mvt"))

    # per-airport Jan / Jul 2025 vs 2026 unmatched counts
    print("\n--- unmatched count per airport x month: train Jan25 / Jul25 vs rank Jan26 / Jul26 ---")
    a = frames["train_JanJul25"].group_by(["ADEP_mvt", "mo"]).agg(pl.len().alias("n")).pivot(on="mo", index="ADEP_mvt", values="n").rename({"1": "train_Jan25", "7": "train_Jul25"})
    b = frames["rank_all"].group_by(["ADEP_mvt", "mo"]).agg(pl.len().alias("n")).pivot(on="mo", index="ADEP_mvt", values="n").rename({"1": "rank_Jan26", "7": "rank_Jul26"})
    print(a.join(b, on="ADEP_mvt", how="full", coalesce=True).fill_null(0).with_columns((pl.col("rank_Jan26") / pl.col("train_Jan25")).round(2).alias("Jan_ratio"), (pl.col("rank_Jul26") / pl.col("train_Jul25")).round(2).alias("Jul_ratio")).sort("ADEP_mvt"))

    # "seen" flag: is the ranking set's lower seen-rate explained by new flights in 2026? check "seen among
    # matched ranking flights" as well (flights seen in the ranking set's own matched rows)
    rk_seen = set(rk.filter(~DEFS[d])["FLIGHT_mvt"].drop_nulls().unique().to_list())
    seen_any = seen_sets[d] | rk_seen
    print("\n--- 'seen' variants, ranking unmatched rows ---")
    ru = frames["rank_all"]
    print("seen among matched TRAINING flights:", round(ru["seen"].mean() * 100, 2), "%")
    print("seen among matched RANKING flights:", round(ru["FLIGHT_mvt"].is_in(pl.Series(sorted(rk_seen))).mean() * 100, 2), "%")
    print("seen among matched training OR ranking flights:", round(ru["FLIGHT_mvt"].is_in(pl.Series(sorted(seen_any))).mean() * 100, 2), "%")
    tu = frames["train_all"]
    print("training unmatched: seen among matched TRAINING flights:", round(tu["seen"].mean() * 100, 2), "%")
    print("training unmatched: seen among matched RANKING flights:", round(tu["FLIGHT_mvt"].is_in(pl.Series(sorted(rk_seen))).mean() * 100, 2), "%")
    # per month seen rate (training) — is 'seen' stable across months
    print("\n--- training unmatched 'seen among matched training flights' rate by month; ranking by month ---")
    print(pl.concat([
        tu.group_by("ym").agg(pl.len().alias("n"), (pl.col("seen").mean() * 100).round(2).alias("seen_pct")),
        ru.group_by("ym").agg(pl.len().alias("n"), (pl.col("seen").mean() * 100).round(2).alias("seen_pct")),
    ]).sort("ym"))
    # seen rate by airport, train vs rank
    print("\n--- 'seen' rate by airport (unmatched rows) ---")
    print(tu.group_by("ADEP_mvt").agg((pl.col("seen").mean() * 100).round(2).alias("train_all")).join(
        frames["train_JanJul25"].group_by("ADEP_mvt").agg((pl.col("seen").mean() * 100).round(2).alias("train_JanJul25")), on="ADEP_mvt").join(
        ru.group_by("ADEP_mvt").agg((pl.col("seen").mean() * 100).round(2).alias("rank_all")), on="ADEP_mvt").sort("ADEP_mvt"))
    # seen rate by airport x month: train Jan25 / Jul25 vs rank Jan26 / Jul26; plus 'seen among matched RANKING flights'
    print("\n--- 'seen' rate by airport x month (unmatched rows): seen among matched TRAINING flights; last two cols = seen among matched RANKING flights ---")
    rk_seen_s = pl.Series(sorted(rk_seen))
    a = frames["train_JanJul25"].group_by(["ADEP_mvt", "mo"]).agg((pl.col("seen").mean() * 100).round(1).alias("s")).pivot(on="mo", index="ADEP_mvt", values="s").rename({"1": "train_Jan25", "7": "train_Jul25"})
    b = ru.group_by(["ADEP_mvt", "mo"]).agg((pl.col("seen").mean() * 100).round(1).alias("s")).pivot(on="mo", index="ADEP_mvt", values="s").rename({"1": "rank_Jan26", "7": "rank_Jul26"})
    c = ru.with_columns(pl.col("FLIGHT_mvt").is_in(rk_seen_s).alias("seen_rk")).group_by(["ADEP_mvt", "mo"]).agg((pl.col("seen_rk").mean() * 100).round(1).alias("s")).pivot(on="mo", index="ADEP_mvt", values="s").rename({"1": "rank_Jan26_seenRK", "7": "rank_Jul26_seenRK"})
    print(a.join(b, on="ADEP_mvt", how="full", coalesce=True).join(c, on="ADEP_mvt", how="full", coalesce=True).sort("ADEP_mvt"))
    # is the 'seen' drop a flight-number turnover? share of matched RANKING flight numbers that appear among matched TRAINING flights (all rows, weighted)
    rk_m = rk.filter(~DEFS[d])
    print("matched RANKING rows whose FLIGHT_mvt appears among matched TRAINING flights: %.2f %% (Jan26 %.2f, Jul26 %.2f)" % (
        rk_m["FLIGHT_mvt"].is_in(pl.Series(sorted(seen_sets[d]))).mean() * 100,
        rk_m.filter(pl.col("mo") == 1)["FLIGHT_mvt"].is_in(pl.Series(sorted(seen_sets[d]))).mean() * 100,
        rk_m.filter(pl.col("mo") == 7)["FLIGHT_mvt"].is_in(pl.Series(sorted(seen_sets[d]))).mean() * 100))
    tr_m = tr.filter(~DEFS[d])
    print("matched TRAINING rows whose FLIGHT_mvt appears among matched RANKING flights: %.2f %% (Jan25 %.2f, Jul25 %.2f)" % (
        tr_m["FLIGHT_mvt"].is_in(rk_seen_s).mean() * 100,
        tr_m.filter(pl.col("mo") == 1)["FLIGHT_mvt"].is_in(rk_seen_s).mean() * 100,
        tr_m.filter(pl.col("mo") == 7)["FLIGHT_mvt"].is_in(rk_seen_s).mean() * 100))
    # KL prefix: which airport x month
    print("\n--- rows with FLIGHT prefix 'KL' among unmatched rows, by airport x set ---")
    kl = []
    for name, df in frames.items():
        kl.append(df.filter(pl.col("pre2") == "KL").group_by("ADEP_mvt").agg(pl.len().alias(name)))
    o = kl[0]
    for t in kl[1:]:
        o = o.join(t, on="ADEP_mvt", how="full", coalesce=True)
    print(o.fill_null(0).sort("ADEP_mvt"))
    print("EHAM unmatched Jan 2026: KL share %.1f %% (n=%d); EHAM unmatched Jan 2025: KL share %.1f %% (n=%d); EHAM unmatched all-train: %.1f %%" % (
        (frames["rank_Jan26"].filter(pl.col("ADEP_mvt") == "EHAM")["pre2"] == "KL").mean() * 100, frames["rank_Jan26"].filter(pl.col("ADEP_mvt") == "EHAM").height,
        (frames["train_JanJul25"].filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 1))["pre2"] == "KL").mean() * 100, frames["train_JanJul25"].filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 1)).height,
        (tu.filter(pl.col("ADEP_mvt") == "EHAM")["pre2"] == "KL").mean() * 100))
    print("KL share among EHAM MATCHED departures: train %.1f %%, rank Jan26 %.1f %%, rank Jul26 %.1f %%" % (
        (tr_m.filter(pl.col("ADEP_mvt") == "EHAM")["pre2"] == "KL").mean() * 100,
        (rk_m.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 1))["pre2"] == "KL").mean() * 100,
        (rk_m.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 7))["pre2"] == "KL").mean() * 100))

# ----------------------------------------------------------------------------------------------------------------
# C. training label by subgroup (unmatched rows)
# ----------------------------------------------------------------------------------------------------------------
for d in DEFS:
    hdr(f"C. training label (y) by subgroup of unmatched rows — definition: {d}")
    tu = UF[d]["train_all"]
    ru = UF[d]["rank_all"]
    print("overall unmatched training: n", tu.height, "y median", tu[LABEL].median(), "share y>=2h %", round((tu[LABEL] >= H2).mean() * 100, 3), "mean", round(tu[LABEL].mean(), 1), "RMS", round(float((tu[LABEL].cast(pl.Float64) ** 2).mean() ** 0.5), 1))
    implied = []
    for dim, top in [("ADEP_mvt", None), ("pre2", 15), ("pre3", 20), ("FLIGHT_RULE_mvt", None), ("hour", None), ("seen", None), ("actype_null", None)]:
        g = tu.group_by(dim).agg(
            pl.len().alias("n_train"),
            pl.col(LABEL).median().alias("y_med"),
            (pl.col(LABEL) >= H2).mean().mul(100).round(3).alias("y_ge2h_pct"),
            (pl.col(LABEL) >= H2).sum().alias("n_ge2h"),
            (pl.col(LABEL) >= 14400).sum().alias("n_ge4h"),
            pl.col(LABEL).mean().round(1).alias("y_mean"),
        )
        r = ru.group_by(dim).agg(pl.len().alias("n_rank"))
        g = g.join(r, on=dim, how="full", coalesce=True).fill_null(0).with_columns(
            (pl.col("n_train") / tu.height * 100).round(2).alias("share_train"),
            (pl.col("n_rank") / ru.height * 100).round(2).alias("share_rank"),
        ).sort("n_rank", descending=True)
        # composition-implied y>=2h share and mean under ranking mix (cells with n_train==0 contribute nothing;
        # their ranking weight is reported)
        have = g.filter(pl.col("n_train") > 0)
        w_rank_covered = have["n_rank"].sum() / max(ru.height, 1)
        imp_ge2h = float((have["y_ge2h_pct"] * have["n_rank"]).sum() / max(have["n_rank"].sum(), 1))
        imp_mean = float((have["y_mean"] * have["n_rank"]).sum() / max(have["n_rank"].sum(), 1))
        act_ge2h = float((tu[LABEL] >= H2).mean() * 100)
        implied.append({"dim": dim, "y_ge2h_pct_train_mix": round(act_ge2h, 3), "y_ge2h_pct_rank_mix": round(imp_ge2h, 3), "y_mean_train_mix": round(float(tu[LABEL].mean()), 1), "y_mean_rank_mix": round(imp_mean, 1), "rank_weight_covered": round(w_rank_covered, 4)})
        print(f"\n--- {dim} ---")
        print(g.head(top) if top else g)
    print("\n--- composition-implied label statistics: training per-cell stats weighted by ranking cell shares ---")
    print(pl.DataFrame(implied))

    # airport x month (Jan/Jul) label stats and implied under ranking (airport x month) mix
    print("\n--- airport x month (Jan, Jul) unmatched training rows: label stats; ranking counts ---")
    tjj = UF[d]["train_JanJul25"]
    g = tjj.group_by(["ADEP_mvt", "mo"]).agg(pl.len().alias("n_train"), pl.col(LABEL).median().alias("y_med"), (pl.col(LABEL) >= H2).mean().mul(100).round(2).alias("y_ge2h_pct"), (pl.col(LABEL) >= H2).sum().alias("n_ge2h"))
    r = ru.group_by(["ADEP_mvt", "mo"]).agg(pl.len().alias("n_rank"))
    g = g.join(r, on=["ADEP_mvt", "mo"], how="full", coalesce=True).fill_null(0).sort(["mo", "ADEP_mvt"])
    print(g)
    print("train Jan+Jul25 unmatched y>=2h share %:", round(float((tjj[LABEL] >= H2).mean() * 100), 3), " implied under rank (airport x month) mix %:", round(float((g["y_ge2h_pct"] * g["n_rank"]).sum() / g["n_rank"].sum()), 3), " expected n(y>=2h) among 5,290/5,283 ranking unmatched:", round(float((g["y_ge2h_pct"] / 100 * g["n_rank"]).sum()), 1))

# ----------------------------------------------------------------------------------------------------------------
# D. OOF unmatched SSE per (airport x month) cell, reweighted to ranking composition
# ----------------------------------------------------------------------------------------------------------------
oof = pl.read_parquet(OOF).drop_nulls("oof")
print("\nOOF rows with prediction:", oof.height)
for d in DEFS:
    hdr(f"D. OOF reweighting — definition: {d}")
    tu = UF[d]["train_all"]
    ru = UF[d]["rank_all"]
    j = tu.join(oof, on="MVT_ID_mvt", how="inner").with_columns(((pl.col(LABEL) - pl.col("oof")) ** 2).alias("se"))
    print("OOF unmatched rows:", j.height, "months:", sorted(j["mo"].unique().to_list()), "RMSE:", round(float(j["se"].mean() ** 0.5), 1), "SSE:", f"{j['se'].sum():.4e}")
    jj = j.filter(pl.col("mo").is_in([1, 7]))
    print("OOF unmatched rows in months 1,7:", jj.height, "RMSE:", round(float(jj["se"].mean() ** 0.5), 1))
    print("OOF unmatched by month:")
    print(j.group_by("mo").agg(pl.len().alias("n"), pl.col("se").mean().sqrt().round(1).alias("rmse"), pl.col("se").sum().alias("sse")).with_columns((pl.col("sse") / pl.col("sse").sum() * 100).round(2).alias("sse_pct")).sort("mo"))

    # (airport x month) cells, months 1 and 7 only (the ranking months); ranking weights = share of ranking unmatched rows
    cell = jj.group_by(["ADEP_mvt", "mo"]).agg(pl.len().alias("n_oof"), pl.col("se").mean().alias("mse"), pl.col("se").sum().alias("sse"), pl.col("se").max().alias("max_se"), (pl.col(LABEL) >= H2).sum().alias("n_ge2h"))
    w = ru.group_by(["ADEP_mvt", "mo"]).agg(pl.len().alias("n_rank"))
    cell = cell.join(w, on=["ADEP_mvt", "mo"], how="full", coalesce=True).fill_null(0)
    cell = cell.with_columns(
        (pl.col("n_oof") / cell["n_oof"].sum()).alias("w_oof"),
        (pl.col("n_rank") / cell["n_rank"].sum()).alias("w_rank"),
        (pl.col("max_se") / pl.col("sse")).alias("top1_share_in_cell"),
        pl.col("mse").sqrt().alias("rmse_cell"),
    ).with_columns((pl.col("mse") * pl.col("w_rank")).alias("mse_x_wrank"), (pl.col("mse") * pl.col("w_oof")).alias("mse_x_woof"))
    print("\n(airport x month) cells — OOF months 1 and 7 vs ranking months Jan 2026 / Jul 2026:")
    print(cell.sort(["mo", "ADEP_mvt"]).with_columns(pl.col("mse").round(0), pl.col("rmse_cell").round(1), pl.col("w_oof").round(4), pl.col("w_rank").round(4), pl.col("top1_share_in_cell").round(3), pl.col("mse_x_wrank").round(0), pl.col("mse_x_woof").round(0)).select(["mo", "ADEP_mvt", "n_oof", "n_rank", "w_oof", "w_rank", "rmse_cell", "n_ge2h", "top1_share_in_cell", "mse_x_woof", "mse_x_wrank"]))
    rmse_oof_mix = float(cell["mse_x_woof"].sum() ** 0.5)
    rmse_rank_mix = float(cell["mse_x_wrank"].sum() ** 0.5)
    print(f"\nunmatched RMSE, OOF months 1+7, OOF composition (check = {round(float(jj['se'].mean() ** 0.5), 1)}): {rmse_oof_mix:.1f}")
    print(f"expected unmatched RMSE under ranking (airport x month) composition: {rmse_rank_mix:.1f}   ratio {rmse_rank_mix / rmse_oof_mix:.3f}")
    # contribution share by cell under ranking mix
    print("\nshare of expected MSE (ranking mix) by cell, top 8:")
    print(cell.with_columns((pl.col("mse_x_wrank") / cell["mse_x_wrank"].sum() * 100).round(2).alias("share_pct")).sort("share_pct", descending=True).select(["mo", "ADEP_mvt", "n_oof", "n_rank", "rmse_cell", "share_pct"]).head(8))

    # variants
    # (i) airport-only cells using all OOF months (1,7,11,12), weights = ranking airport shares
    ca = j.group_by("ADEP_mvt").agg(pl.len().alias("n_oof"), pl.col("se").mean().alias("mse"))
    wa = ru.group_by("ADEP_mvt").agg(pl.len().alias("n_rank"))
    ca = ca.join(wa, on="ADEP_mvt", how="full", coalesce=True).fill_null(0)
    r_a_oof = float((ca["mse"] * ca["n_oof"]).sum() / ca["n_oof"].sum()) ** 0.5
    r_a_rank = float((ca["mse"] * ca["n_rank"]).sum() / ca["n_rank"].sum()) ** 0.5
    print(f"\nvariant (i) airport-only cells, all 4 OOF months: OOF mix {r_a_oof:.1f} -> ranking airport mix {r_a_rank:.1f} (ratio {r_a_rank / r_a_oof:.3f})")
    # (ii) month-only cells (1,7): the effect of the Jan/Jul balance alone
    cm = jj.group_by("mo").agg(pl.len().alias("n_oof"), pl.col("se").mean().alias("mse")).join(ru.group_by("mo").agg(pl.len().alias("n_rank")), on="mo", how="full", coalesce=True).fill_null(0)
    r_m_oof = float((cm["mse"] * cm["n_oof"]).sum() / cm["n_oof"].sum()) ** 0.5
    r_m_rank = float((cm["mse"] * cm["n_rank"]).sum() / cm["n_rank"].sum()) ** 0.5
    print(f"variant (ii) month-only cells (1,7): OOF mix {r_m_oof:.1f} -> ranking month mix {r_m_rank:.1f} (ratio {r_m_rank / r_m_oof:.3f})")
    # (iii) sensitivity: drop the top-k OOF unmatched rows by se before building cells (months 1,7)
    for k in (10, 20, 50):
        jk = jj.sort("se", descending=True).slice(k)
        ck = jk.group_by(["ADEP_mvt", "mo"]).agg(pl.len().alias("n_oof"), pl.col("se").mean().alias("mse")).join(w, on=["ADEP_mvt", "mo"], how="full", coalesce=True).fill_null(0)
        r_o = float((ck["mse"] * ck["n_oof"]).sum() / ck["n_oof"].sum()) ** 0.5
        r_r = float((ck["mse"] * ck["n_rank"]).sum() / ck["n_rank"].sum()) ** 0.5
        print(f"variant (iii) drop top-{k} OOF unmatched rows (months 1,7): OOF mix {r_o:.1f} -> ranking mix {r_r:.1f} (ratio {r_r / r_o:.3f})")
    # (iv) (airport x month x seen) cells
    cs = jj.group_by(["ADEP_mvt", "mo", "seen"]).agg(pl.len().alias("n_oof"), pl.col("se").mean().alias("mse")).join(ru.group_by(["ADEP_mvt", "mo", "seen"]).agg(pl.len().alias("n_rank")), on=["ADEP_mvt", "mo", "seen"], how="full", coalesce=True).fill_null(0)
    have = cs.filter(pl.col("n_oof") > 0)
    r_s_oof = float((have["mse"] * have["n_oof"]).sum() / have["n_oof"].sum()) ** 0.5
    r_s_rank = float((have["mse"] * have["n_rank"]).sum() / have["n_rank"].sum()) ** 0.5
    print(f"variant (iv) (airport x month x seen) cells: OOF mix {r_s_oof:.1f} -> ranking mix {r_s_rank:.1f} (ratio {r_s_rank / r_s_oof:.3f}); ranking rows in cells with no OOF rows: {int(cs.filter(pl.col('n_oof') == 0)['n_rank'].sum())}")
    # (v) label-only analogue on ALL training unmatched rows (no model): RMS of y per cell (airport x month over
    # 12 months) reweighted — what composition alone does to the label's second moment
    cy = tu.filter(pl.col("mo").is_in([1, 7])).group_by(["ADEP_mvt", "mo"]).agg(pl.len().alias("n_tr"), (pl.col(LABEL).cast(pl.Float64) ** 2).mean().alias("y2"), pl.col(LABEL).var().alias("yvar")).join(w, on=["ADEP_mvt", "mo"], how="full", coalesce=True).fill_null(0)
    sd_tr = float((cy["yvar"] * cy["n_tr"]).sum() / cy["n_tr"].sum()) ** 0.5
    sd_rk = float((cy["yvar"] * cy["n_rank"]).sum() / cy["n_rank"].sum()) ** 0.5
    print(f"variant (v) within-cell label SD pooled (train Jan+Jul25 unmatched, airport x month cells): train mix {sd_tr:.1f} -> ranking mix {sd_rk:.1f} (ratio {sd_rk / sd_tr:.3f})")
    # OOF 'seen' subgroup error
    print("\nOOF unmatched error by 'seen' (all OOF months):")
    print(j.group_by("seen").agg(pl.len().alias("n"), pl.col("se").mean().sqrt().round(1).alias("rmse"), pl.col(LABEL).median().alias("y_med"), pl.col("oof").median().round(0).alias("pred_med")))
    print("OOF unmatched error by actype_null (all OOF months):")
    print(j.group_by("actype_null").agg(pl.len().alias("n"), pl.col("se").mean().sqrt().round(1).alias("rmse"), pl.col(LABEL).median().alias("y_med"), pl.col("oof").median().round(0).alias("pred_med")))
    print("OOF unmatched error by FLIGHT_RULE (all OOF months):")
    print(j.group_by("FLIGHT_RULE_mvt").agg(pl.len().alias("n"), pl.col("se").mean().sqrt().round(1).alias("rmse"), pl.col(LABEL).median().alias("y_med")))

print("\ndone")

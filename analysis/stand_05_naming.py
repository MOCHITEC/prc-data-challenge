"""Stand naming structure, per-stand dispersion, rare/new stands in ranking and back-off for them.

Tables (training = 2025 labelled departures, ranking = Jan/Jul 2026 departures):
  1. per airport: stand-prefix groups (letters before the digits) with n, median y, share of departures
  2. how much of y's variance is explained by airport / +runway / +prefix / +stand / +stand x runway
     (in-sample eta^2 on clean rows 30..7200 s) and the residual (y - (MVT-AOBT3)) version
  3. the most dispersed stands (robust IQR of y) - candidates for towing / remote de-icing / data problems
  4. ranking coverage of stand and stand x runway by training counts
  5. back-off evaluation for thin stand x runway cells, month-blocked (fit on months != m, score m in {1, 7}):
     apt x runway median  vs  apt x prefix x runway  vs  numeric-neighbour stand  vs  apt x stand (any runway)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from stand_common import load_all, stand_number, stand_prefix

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_str_lengths(60)


def main() -> None:
    df = load_all(keep_stand_str=True).filter(pl.col("PHASE_mvt") == "DEP")
    df = df.with_columns(
        stand_prefix(pl.col("STAND_ok")).alias("pfx"),
        stand_number(pl.col("STAND_ok")).alias("num"),
        pl.col("STAND_ok").str.extract(r"\d+([A-Za-z]+)$", 1).fill_null("").alias("sfx"),
        (pl.col("t_mvt") - pl.col("t_aobt")).alias("x_aobt"),
        pl.col("RUNWAY_mvt").cast(pl.String),
        pl.col("APT").cast(pl.String),
        pl.from_epoch(pl.col("t_mvt").cast(pl.Int64), time_unit="s").dt.month().alias("month"),
    ).with_columns(
        # number block of 100 (e.g. EGLL 5xx = T5, LIRF 3xx) and of 10
        (pl.col("num") // 100).alias("num100"),
        (pl.col("num") // 10).alias("num10"),
    )
    tr = df.filter(pl.col("split") == "train")
    rk = df.filter(pl.col("split") == "rank")
    clean = tr.filter((pl.col("y") >= 30) & (pl.col("y") <= 7200))

    # 1. prefix groups
    g = (
        clean.group_by(["APT", "pfx"])
        .agg(pl.len().alias("n"), pl.col("STAND_ok").n_unique().alias("n_stands"), pl.col("y").median().alias("y_med"),
             (pl.col("y") - pl.col("x_aobt")).median().alias("resid_med"),
             pl.col("num").min().alias("num_min"), pl.col("num").max().alias("num_max"))
        .with_columns((pl.col("n") / pl.col("n").sum().over("APT")).alias("share"))
        .filter(pl.col("share") >= 0.005)
        .sort(["APT", "n"], descending=[False, True])
    )
    print("## 1. stand prefix groups (clean training departures, share >= 0.5%)\n", g)
    # for purely numeric airports, 100-blocks
    g2 = (
        clean.filter(pl.col("pfx") == "")
        .group_by(["APT", "num100"])
        .agg(pl.len().alias("n"), pl.col("STAND_ok").n_unique().alias("n_stands"), pl.col("y").median().alias("y_med"))
        .with_columns((pl.col("n") / pl.col("n").sum().over("APT")).alias("share"))
        .filter(pl.col("share") >= 0.01)
        .sort(["APT", "num100"])
    )
    print("## 1b. numeric stands by hundreds block\n", g2)

    # 2. variance explained (in-sample, clean rows)
    def eta2(frame: pl.DataFrame, target: str, keys: list[str]) -> float:
        f = frame.filter(pl.col(target).is_not_null())
        m = f.with_columns(pl.col(target).mean().over(keys).alias("_m"))
        sst = ((f[target] - f[target].mean()) ** 2).sum()
        sse = ((m[target] - m["_m"]) ** 2).sum()
        return 1 - sse / sst

    clean = clean.with_columns((pl.col("y") - pl.col("x_aobt")).alias("resid"))
    rows = []
    for keys in (["APT"], ["APT", "RUNWAY_mvt"], ["APT", "pfx"], ["APT", "pfx", "RUNWAY_mvt"],
                 ["APT", "pfx", "num10", "RUNWAY_mvt"], ["APT", "STAND_ok"], ["APT", "STAND_ok", "RUNWAY_mvt"]):
        rows.append(("x".join(keys), eta2(clean, "y", keys), eta2(clean, "resid", keys)))
    print("## 2. in-sample share of variance explained (clean rows)\n",
          pl.DataFrame(rows, schema=["keys", "eta2_y", "eta2_resid(y-(MVT-AOBT3))"], orient="row"))
    per_apt = []
    for apt in sorted(clean["APT"].unique().to_list()):
        c = clean.filter(pl.col("APT") == apt)
        per_apt.append((apt, eta2(c, "y", ["RUNWAY_mvt"]), eta2(c, "y", ["STAND_ok"]), eta2(c, "y", ["pfx", "RUNWAY_mvt"]),
                        eta2(c, "y", ["STAND_ok", "RUNWAY_mvt"]), eta2(c, "resid", ["STAND_ok", "RUNWAY_mvt"])))
    print(pl.DataFrame(per_apt, schema=["apt", "rwy", "stand", "pfx_rwy", "stand_rwy", "resid_stand_rwy"], orient="row"))

    # 3. most dispersed stands (IQR within stand x runway, n>=200)
    disp = (
        clean.group_by(["APT", "STAND_ok", "RUNWAY_mvt"])
        .agg(pl.len().alias("n"), pl.col("y").median().alias("y_med"),
             (pl.col("y").quantile(0.75) - pl.col("y").quantile(0.25)).alias("iqr"),
             (pl.col("resid").quantile(0.75) - pl.col("resid").quantile(0.25)).alias("resid_iqr"))
        .filter(pl.col("n") >= 200)
    )
    apt_iqr = disp.group_by("APT").agg(pl.col("iqr").median().alias("apt_iqr_med"))
    disp = disp.join(apt_iqr, on="APT").with_columns((pl.col("iqr") / pl.col("apt_iqr_med")).alias("iqr_ratio"))
    print("## 3. most dispersed stand x runway cells (n>=200), top 25 by IQR ratio to airport median IQR\n",
          disp.sort("iqr_ratio", descending=True).head(25))
    # dispersion vs prefix: remote stands?
    print("## 3b. stand-level y IQR by prefix group (median over cells)\n",
          disp.join(clean.select(["APT", "STAND_ok", "pfx"]).unique(), on=["APT", "STAND_ok"])
          .group_by(["APT", "pfx"]).agg(pl.len().alias("cells"), pl.col("iqr").median(), pl.col("y_med").median())
          .filter(pl.col("cells") >= 5).sort(["APT", "iqr"], descending=[False, True]))

    # 4. ranking coverage
    cnt_s = tr.group_by(["APT", "STAND_ok"]).agg(pl.len().alias("n_s"))
    cnt_sr = tr.group_by(["APT", "STAND_ok", "RUNWAY_mvt"]).agg(pl.len().alias("n_sr"))
    rk2 = rk.join(cnt_s, on=["APT", "STAND_ok"], how="left").join(cnt_sr, on=["APT", "STAND_ok", "RUNWAY_mvt"], how="left")
    cov = rk2.group_by("APT").agg(
        pl.len().alias("n_rank"),
        pl.col("STAND_ok").is_null().mean().alias("stand_bad"),
        (pl.col("n_s").is_null() & pl.col("STAND_ok").is_not_null()).mean().alias("stand_unseen"),
        (pl.col("n_s").fill_null(0) < 30).mean().alias("stand_lt30"),
        (pl.col("n_sr").fill_null(0) < 30).mean().alias("stand_rwy_lt30"),
        (pl.col("n_sr").fill_null(0) < 100).mean().alias("stand_rwy_lt100"),
    ).sort("APT")
    print("## 4. ranking departures: training support of their stand / stand x runway\n", cov)
    tot = rk2.select((pl.col("n_sr").fill_null(0) < 30).mean().alias("sr_lt30"),
                     (pl.col("n_s").is_null()).mean().alias("s_unseen"))
    print(tot)
    unseen = rk2.filter(pl.col("n_s").is_null() & pl.col("STAND_ok").is_not_null()).group_by(["APT", "STAND_ok"]).len().sort("len", descending=True)
    print("unseen stands in ranking (top):\n", unseen.head(25))
    # also same in training month-blocked (what jan/jul folds see)
    for m in (1, 7):
        trm, vam = tr.filter(pl.col("month") != m), tr.filter(pl.col("month") == m)
        c = trm.group_by(["APT", "STAND_ok", "RUNWAY_mvt"]).agg(pl.len().alias("n_sr"))
        v = vam.join(c, on=["APT", "STAND_ok", "RUNWAY_mvt"], how="left")
        print(f"fold {m}: stand x runway <30 share", v.select((pl.col("n_sr").fill_null(0) < 30).mean()).item())

    # 5. back-off evaluation on thin cells, target = resid (y - x_aobt) on clean NM rows, and y
    res = []
    for target in ("y", "resid"):
        for m in (1, 7):
            base = clean.filter(pl.col(target).is_not_null())
            trm, vam = base.filter(pl.col("month") != m), base.filter(pl.col("month") == m)
            def tab(keys, name):
                return trm.group_by(keys).agg(pl.col(target).median().alias(name), pl.len().alias(name + "_n"))
            v = vam
            for keys, name in ((["APT", "STAND_ok", "RUNWAY_mvt"], "sr"), (["APT", "RUNWAY_mvt"], "r"),
                               (["APT", "pfx", "RUNWAY_mvt"], "pr"), (["APT", "pfx", "num10", "RUNWAY_mvt"], "p10r"),
                               (["APT", "STAND_ok"], "s"), (["APT", "pfx", "num10"], "p10"), (["APT"], "a")):
                v = v.join(tab(keys, name), on=keys, how="left")
            # numeric-neighbour: same prefix and runway, nearest number with >=30 rows
            nb = trm.group_by(["APT", "pfx", "num", "RUNWAY_mvt"]).agg(pl.col(target).median().alias("nb"), pl.len().alias("nb_n")).filter(
                (pl.col("nb_n") >= 30) & pl.col("num").is_not_null()).sort("num")
            vv = v.filter(pl.col("num").is_not_null()).sort("num").join_asof(
                nb.rename({"num": "num_nb"}), left_on="num", right_on="num_nb", by=["APT", "pfx", "RUNWAY_mvt"], strategy="nearest")
            v = v.join(vv.select(["MVT_ID_mvt", "nb", "num_nb"]), on="MVT_ID_mvt", how="left")
            thin = v.filter(pl.col("sr_n").fill_null(0) < 30)
            def r(col):
                p = thin.select(pl.coalesce([pl.col(col), pl.col("r"), pl.col("a")])).to_series().to_numpy()
                yy = thin[target].to_numpy()
                return float(np.sqrt(np.nanmean((yy - p) ** 2)))
            def r_mix():
                # hierarchical: stand x runway if n>=5 shrunk, else neighbour, else prefix-decade x runway, ...
                p = thin.select(pl.coalesce([
                    pl.when(pl.col("p10r_n") >= 30).then(pl.col("p10r")),
                    pl.when(pl.col("pr_n") >= 30).then(pl.col("pr")),
                    pl.col("r"), pl.col("a")])).to_series().to_numpy()
                yy = thin[target].to_numpy()
                return float(np.sqrt(np.nanmean((yy - p) ** 2)))
            res.append((target, m, thin.height, thin.height / v.height, r("a"), r("r"), r("pr"), r("p10r"), r("nb"), r("s"), r_mix(),
                        r("sr")))
    print("## 5. back-off RMSE on thin stand x runway cells (<30 rows in the training months), clean rows\n",
          pl.DataFrame(res, schema=["target", "month", "n_thin", "share", "apt", "apt_rwy", "pfx_rwy", "pfx10_rwy",
                                    "num_neighbour_rwy", "stand_anyrwy", "hier(p10r>pr>r)", "sr_raw(if any)"], orient="row"))


if __name__ == "__main__":
    main()

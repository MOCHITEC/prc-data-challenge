"""ADS-B override coverage breakdown: why scored departures at the override airports were NOT overridden.

Read-only analysis. Nothing is fitted, nothing is written to `outputs/`, `src/`, `configs/` or `data/`. The
detection is not re-run: every reason is read off the stage-1 columns of `$SCRATCH/<tag>_joined.parquet`
(`scripts/adsb_stage1_feasibility.py analyse`) and the eligibility / guard logic of
`scripts/adsb_override_candidate.py` (copied here, not changed). The raw filtered fixes `$SCRATCH/<tag>.parquet`
are read only to split two stage-1 outcomes that the joined file does not separate (A5, A7) and for part B.

    POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
    .venv/bin/python scripts/adsb_cov_breakdown.py [--base PATH]

Part A (scored 2026 days). Waterfall, first matching reason wins, in this order:
  A1 missing day            day not analysed by adsb_ingest_all.py
  A2 MVT 00:00-02:00 UTC    outside the stage-1 scope [D 02:00, D+1 00:00)
  A3 no CALLSIGN_flt/AOBT_3  lane B (the override requires AOBT_3_flt and a `cs` join)
  A4 not in joined file     other (should be 0)
  A5 join is null           no ADS-B fix at ADEP carries CALLSIGN_flt or FLIGHT_mvt that day. Split by whether an
                            ADS-B take-off at ADEP not claimed by any joined departure of that day lies within
                            +-60 s of MVT_TIME ("unclaimed take-off": candidate callsign mismatch) or not. A placebo
                            count at MVT_TIME +-600 s measures chance matches.
  A6 join fn / fn_notime    CALLSIGN_flt did not match; FLIGHT_mvt did (not used by the rule)
  A7 join cs_notime         CALLSIGN_flt matched an aircraft at ADEP, but no candidate gave a usable phase. Split:
                            no fix of that aircraft within +-3 h of MVT_TIME, or fixes within 3 h but neither a
                            ground fix in the 4 h before MVT+2 min nor an airborne fix within +-30 min.
  A8 cs, no ground fix       points exist (airborne) but no ground segment before take-off
  A9 cs, ground, no stand    ground segment but no stationary stand run: stand_rule moving_when_first_seen /
                            no qualifying run / stand run ends < 60 s before the end of the phase
  A10 stand, no pushback     push is null (no usable step with track in the 180 s after the stand) / push < 0.5
                            (first motion not tail-first) / push >= 0.5 but post_dwell >= 900 s (dwell guard)
  A11 guard trip            |base - y_a| > 900 s
  A12 overridden
Airports: the five included ones and LEMD (as if included; LEMD's bias computed by the same S1 rule for A11).

Part B (the four labelled 2025 days, lane A, join cs, ground fix present, no gated pushback, at the five airports
and LEMD). Alternative events read from the same phase's ground fixes, estimate y_e = MVT - t_event + b, with b the
per-airport median(t_event - BLOCK) over the same group on the other three days (leave one day out). Residual
y_e - y; median, IQR, RMSE; the production OOF RMSE on the same rows; and the 900 s guard applied against the OOF.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_stage1_feasibility as s1  # noqa: E402

ROOT = s1.REPO_DIR
SCR = s1.SCRATCH
APTS5 = ["EDDM", "EGLL", "EHAM", "LEBL", "LSZH"]
APTS = APTS5 + ["LEMD"]
LAB_DAYS = ["2025-07-21", "2025-07-13", "2025-01-22", "2025-11-12"]
GUARD = 900.0
BASE = (sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else
        "/tmp/claude-0/-home-user-prc-data-challenge/ca703460-62c0-5c84-91fe-d76c3d0e1e20/scratchpad/"
        "v16_from_bucket.parquet")
TO_WIN = 60.0
PLACEBO = 600.0
JCOLS = ["MVT_ID_mvt", "ADEP_mvt", "cs", "t_mvt", "join", "icao", "ground_fix", "t_first", "t_dep1", "t_move",
         "t_move_lo", "t_to", "n_ground", "stand_rule", "push", "post_dwell", "AOBT_3_flt"]
FCOLS = ["icao", "apt", "t", "lat", "lon", "ground", "gs", "cs_ff", "cs_bf"]

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(40)
pl.Config.set_float_precision(3)


def tag_of(day):
    return f"v{day.replace('-', '.')}-planes-readsb-prod-0"


def head(s):
    print(f"\n{'=' * 110}\n{s}\n{'=' * 110}", flush=True)


def bias_all_days() -> dict:
    """S1 of adsb_override_candidate.py, for the five airports and LEMD: median(t_push - BLOCK) over covered rows."""
    ft = pl.read_parquet(ROOT + "data/work/feat_train.parquet", columns=["MVT_ID_mvt", "mvt_minus_aobt3"])
    fr = []
    for day in LAB_DAYS:
        J = s1.add_estimators(pl.read_parquet(f"{SCR}/{tag_of(day)}_joined.parquet"))
        fr.append(J.join(ft, on="MVT_ID_mvt", how="left").select("ADEP_mvt", "mvt_minus_aobt3", "join", "t_push",
                                                                  "t_block"))
    C = pl.concat(fr).filter(pl.col("mvt_minus_aobt3").is_not_null() & (pl.col("join") == "cs")
                             & pl.col("t_push").is_not_null())
    bt = C.group_by("ADEP_mvt").agg(pl.len().alias("n"), (pl.col("t_push") - pl.col("t_block")).median().alias("b"))
    print("bias (S1 rule):", bt.filter(pl.col("ADEP_mvt").is_in(APTS)).sort("ADEP_mvt").rows())
    return {r[0]: float(r[2]) for r in bt.rows() if r[0] in APTS}


def load_fixes(day):
    return (pl.scan_parquet(f"{SCR}/{tag_of(day)}.parquet").select(FCOLS)
            .filter(pl.col("apt").is_in(APTS)).collect())


def takeoffs(fx: pl.DataFrame) -> pl.DataFrame:
    """Ground -> airborne transitions per (icao, apt), same definition as stage-1's take-off fix."""
    f = fx.select("icao", "apt", "t", "ground", "cs_ff", "cs_bf").sort("icao", "apt", "t")
    f = f.with_columns(pl.col("ground").shift(1).over("icao", "apt").alias("g_prev"))
    return f.filter(~pl.col("ground") & pl.col("g_prev").fill_null(False)).select(
        "icao", "apt", pl.col("t").alias("t_tk"), pl.coalesce("cs_ff", "cs_bf").alias("adsb_cs"))


def part_a(bias: dict) -> None:
    head("A  scored 2026 days: waterfall of reasons, five included airports and LEMD")
    import adsb_override_candidate as oc  # analysed_days(): the same day list the candidate used
    days, missing = oc.analysed_days()
    print(f"analysed days {len(days)}, missing {missing or 'none'}; base {BASE}")
    tpl = pl.read_parquet(ROOT + "data/raw/submitting.parquet")
    base = pl.read_parquet(BASE)
    assert (base["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
    rk = (pl.scan_parquet(ROOT + "data/raw/ranking.parquet")
          .select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "AOBT_3_flt", "PHASE_mvt").collect())
    S = tpl.select("MVT_ID_mvt").join(rk, on="MVT_ID_mvt", how="left").filter(pl.col("ADEP_mvt").is_in(APTS))
    S = S.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("day"),
                       (pl.col("MVT_TIME_UTC_mvt").dt.hour() < 2).alias("h0_2"))
    print("PHASE of scored rows:", S["PHASE_mvt"].value_counts().rows(),
          "; CALLSIGN present xor AOBT_3 present:",
          S.select((pl.col("CALLSIGN_flt").is_not_null() ^ pl.col("AOBT_3_flt").is_not_null()).sum()).item())
    Js, extra = [], []
    for day in days:
        J = s1.add_estimators(pl.read_parquet(f"{SCR}/{tag_of(day)}_joined.parquet", columns=JCOLS))
        Jd = J.filter(pl.col("ADEP_mvt").is_in(APTS))
        fx = load_fixes(day)
        tk = takeoffs(fx)
        # claimed take-offs: any joined departure of the day (all join types) whose t_to is this take-off
        cl = J.filter(pl.col("icao").is_not_null() & pl.col("t_to").is_not_null()).select("icao", "t_to")
        tk = tk.join(cl, on="icao", how="left").with_columns(
            ((pl.col("t_to") - pl.col("t_tk")).abs() <= TO_WIN).fill_null(False).alias("c")).group_by(
            "icao", "apt", "t_tk", "adsb_cs").agg(pl.col("c").any().alias("claimed"))
        un = tk.filter(~pl.col("claimed")).sort("t_tk")
        # A5 split: nearest unclaimed take-off to MVT (and to MVT +- 600 s placebo)
        J5 = Jd.filter(pl.col("join").is_null()).select("MVT_ID_mvt", "ADEP_mvt", "t_mvt", "cs")
        out = J5
        for name, off in [("to0", 0.0), ("to_p", PLACEBO), ("to_m", -PLACEBO)]:
            q = J5.with_columns((pl.col("t_mvt") + off).alias("tq")).sort("tq")
            m = q.join_asof(un.rename({"apt": "ADEP_mvt"}).select("ADEP_mvt", "t_tk", "adsb_cs"), left_on="tq",
                            right_on="t_tk", by="ADEP_mvt", strategy="nearest")
            m = m.select("MVT_ID_mvt", ((pl.col("t_tk") - pl.col("tq")).abs() <= TO_WIN).fill_null(False).alias(name),
                         pl.col("adsb_cs").alias(f"cs_{name}"))
            out = out.join(m, on="MVT_ID_mvt", how="left")
        # A7 split: cs_notime -> any fix of the callsign's aircraft at ADEP within +-3 h of MVT?
        J7 = Jd.filter(pl.col("join") == "cs_notime").select("MVT_ID_mvt", "ADEP_mvt", "cs", "t_mvt")
        keyed = pl.concat([fx.select("apt", "icao", pl.col("cs_ff").alias("k")),
                           fx.select("apt", "icao", pl.col("cs_bf").alias("k"))]).drop_nulls("k").unique()
        near3 = (J7.join(keyed.rename({"apt": "ADEP_mvt", "k": "cs"}), on=["ADEP_mvt", "cs"], how="left")
                 .join(fx.select(pl.col("apt").alias("ADEP_mvt"), "icao", "t"), on=["ADEP_mvt", "icao"], how="left")
                 .group_by("MVT_ID_mvt").agg(((pl.col("t") - pl.col("t_mvt")).abs() <= 3 * 3600).any()
                                             .fill_null(False).alias("fix3h")))
        Js.append(Jd.with_columns(pl.lit(day).alias("jday")))
        extra.append(out.select("MVT_ID_mvt", "to0", "to_p", "to_m", "cs", "cs_to0").join(
            near3, on="MVT_ID_mvt", how="full", coalesce=True))
        print(f"  {day}: joined rows {Jd.height}, take-offs {tk.height} (unclaimed {un.height}), "
              f"join-null {J5.height}, cs_notime {J7.height}", flush=True)
        del fx, tk, un, keyed
    J = pl.concat(Js, how="diagonal_relaxed")
    X = pl.concat(extra, how="diagonal_relaxed")
    assert J["MVT_ID_mvt"].n_unique() == J.height
    D = S.join(J.drop("ADEP_mvt", "AOBT_3_flt"), on="MVT_ID_mvt", how="left").join(
        X.drop("cs"), on="MVT_ID_mvt", how="left").join(
        base.rename({"TAXITIME_SEC_mvt": "base"}), on="MVT_ID_mvt", how="left").join(
        pl.DataFrame({"ADEP_mvt": list(bias), "b": list(bias.values())}), on="ADEP_mvt", how="left")
    # the candidate script's rule, verbatim in logic
    D = D.with_columns((pl.col("AOBT_3_flt").is_not_null() & (pl.col("join") == "cs")
                        & pl.col("t_push").is_not_null()).fill_null(False).alias("pushback"))
    D = D.with_columns(pl.when(pl.col("pushback")).then(pl.col("t_mvt") - pl.col("t_push") + pl.col("b")).alias("y_a"))
    D = D.with_columns((pl.col("pushback") & ((pl.col("base").cast(pl.Float64) - pl.col("y_a")).abs() > GUARD))
                       .alias("trip"))
    c = pl.col
    reason = (
        pl.when(c("day").is_in(missing)).then(pl.lit("A1 missing day"))
        .when(c("h0_2")).then(pl.lit("A2 MVT 00:00-02:00 UTC"))
        .when(c("CALLSIGN_flt").is_null() | c("AOBT_3_flt").is_null()).then(pl.lit("A3 no CALLSIGN_flt/AOBT_3 (lane B)"))
        .when(c("jday").is_null()).then(pl.lit("A4 other: not in joined file"))
        .when(c("join").is_null() & c("to0").fill_null(False))
        .then(pl.lit("A5a no fix carries callsign; unclaimed ADS-B take-off within 60 s"))
        .when(c("join").is_null()).then(pl.lit("A5b no fix carries callsign; no unclaimed take-off within 60 s"))
        .when(c("join").is_in(["fn", "fn_notime"])).then(pl.lit("A6 callsign join failed (FLIGHT_mvt matched)"))
        .when((c("join") == "cs_notime") & ~c("fix3h").fill_null(False))
        .then(pl.lit("A7a callsign's aircraft has no fix at ADEP within 3 h"))
        .when(c("join") == "cs_notime").then(pl.lit("A7b fixes within 3 h, no ground fix and no air fix within 30 min"))
        .when(~c("ground_fix").fill_null(False)).then(pl.lit("A8 points but no ground segment"))
        .when(c("t_move").is_null() & (c("stand_rule") == "moving_when_first_seen"))
        .then(pl.lit("A9a ground, no stand: moving when first seen"))
        .when(c("t_move").is_null() & c("stand_rule").is_null()).then(pl.lit("A9b ground, no stand: no qualifying run"))
        .when(c("t_move").is_null()).then(pl.lit("A9c ground, stand run ends < 60 s before phase end"))
        .when(c("t_push").is_null() & c("push").is_null())
        .then(pl.lit("A10a stand, no pushback: no usable step with track"))
        .when(c("t_push").is_null() & (c("push") < 0.5)).then(pl.lit("A10b stand, no pushback: first motion forward"))
        .when(c("t_push").is_null()).then(pl.lit("A10c stand, pushback, dwell guard (post_dwell >= 900 s)"))
        .when(c("trip")).then(pl.lit("A11 guard trip"))
        .otherwise(pl.lit("A12 overridden")))
    D = D.with_columns(reason.alias("reason"))
    # checks against the candidate run
    ov5 = D.filter(c("ADEP_mvt").is_in(APTS5) & (c("reason") == "A12 overridden"))
    audit = pl.read_parquet(os.path.join(SCR, "adsb_override_rows.parquet"), columns=["MVT_ID_mvt"])
    print(f"CHECK overridden at 5 airports {ov5.height} (audit file {audit.height}); id sets equal: "
          f"{set(ov5['MVT_ID_mvt'].to_list()) == set(audit['MVT_ID_mvt'].to_list())}; trips at 5 airports "
          f"{D.filter(c('ADEP_mvt').is_in(APTS5) & c('trip')).height}; pushbacks per airport "
          f"{D.filter(c('pushback')).group_by('ADEP_mvt').len().sort('ADEP_mvt').rows()}")
    print("h0_2 rows that are in a joined file (should be 0):", D.filter(c("h0_2") & c("jday").is_not_null()).height)
    T = D.group_by("reason", "ADEP_mvt").len().pivot(on="ADEP_mvt", index="reason", values="len").fill_null(0)
    T = T.select("reason", *APTS).sort("reason")
    T = T.with_columns(pl.sum_horizontal(*APTS5).alias("five"))
    tot = D.group_by("ADEP_mvt").len()
    totd = dict(tot.rows())
    print("counts (scored departures):")
    print(T)
    print("totals:", {a: totd[a] for a in APTS}, "five:", sum(totd[a] for a in APTS5))
    print("share of the airport's scored departures (%):")
    print(T.with_columns([(pl.col(a) / totd[a] * 100).alias(a) for a in APTS]
                         + [(pl.col("five") / sum(totd[a] for a in APTS5) * 100).alias("five")]))
    # A5 placebo and callsign-prefix agreement
    A5 = D.filter(c("reason").str.starts_with("A5"))
    print("A5 (join null, lane A, in scope): unclaimed take-off within 60 s of MVT / of MVT+600 / of MVT-600")
    print(A5.group_by("ADEP_mvt").agg(pl.len().alias("n"), c("to0").sum().alias("at_mvt"),
                                      c("to_p").sum().alias("placebo_p600"), c("to_m").sum().alias("placebo_m600"),
                                      (c("to0") & (c("cs_to0").str.slice(0, 3)
                                                   == c("CALLSIGN_flt").str.strip_chars().str.slice(0, 3)))
                                      .sum().alias("at_mvt_same_3char_prefix")).sort("ADEP_mvt"))
    ex = A5.filter(c("to0")).select("ADEP_mvt", "CALLSIGN_flt", "cs_to0").head(25)
    print("examples CALLSIGN_flt vs ADS-B callsign of the unclaimed take-off:", ex.rows())
    D.select("MVT_ID_mvt", "ADEP_mvt", "day", "reason").write_parquet(os.path.join(SCR, "adsb_cov_reasons.parquet"))


# ============================================================================================ part B
def events(tt, la, lo, gs, t_move):
    """Alternative off-block events on one phase's ground fixes (sorted)."""
    ev = {"move": t_move, "dep1": None}
    if len(tt) == 0:
        return ev
    big = np.where(np.diff(tt) >= s1.GAP_S)[0]
    k1 = int(big[-1] + 1) if len(big) else 0
    ev["dep1"] = float(tt[k1])
    for X in (3.0, 5.0, 10.0):
        v = None
        for i in range(k1, len(tt)):
            if not (gs[i] > X):
                continue
            w = np.where((tt > tt[i]) & (tt <= tt[i] + 30.0))[0]
            if len(w) and np.all(gs[w] > X):
                v = float(tt[i])
                break
        ev[f"gs{int(X)}"] = v
    d = s1.dist_m(la[k1], lo[k1], la[k1:], lo[k1:])
    w = np.where(d > 100.0)[0]
    ev["d100"] = float(tt[k1 + w[0]]) if len(w) else None
    return ev


def part_b() -> None:
    head("B  labelled 2025 days: lane A, cs join, ground segment, no gated pushback -> alternative events")
    ft = pl.read_parquet(ROOT + "data/work/feat_train.parquet",
                         columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3"]).rename({"TAXITIME_SEC_mvt": "y"})
    oof = pl.read_parquet(ROOT + "experiments/oof_main_gap-62dd7c6e88.parquet")
    rows, nmatch = [], []
    for day in LAB_DAYS:
        J = s1.add_estimators(pl.read_parquet(f"{SCR}/{tag_of(day)}_joined.parquet", columns=JCOLS + ["t_block"]))
        J = J.join(ft, on="MVT_ID_mvt", how="left").join(oof, on="MVT_ID_mvt", how="left")
        G = J.filter(pl.col("ADEP_mvt").is_in(APTS) & pl.col("mvt_minus_aobt3").is_not_null()
                     & (pl.col("join") == "cs") & pl.col("ground_fix").fill_null(False) & pl.col("t_push").is_null())
        fx = load_fixes(day).filter(pl.col("ground")).sort("t")
        byk = {k: v for k, v in fx.partition_by(["icao", "apt"], as_dict=True).items()}
        for r in G.iter_rows(named=True):
            f = byk.get((r["icao"], r["ADEP_mvt"]))
            t_end = r["t_to"] if r["t_to"] is not None and r["t_to"] > r["t_first"] else r["t_mvt"] + 120.0
            if f is None:
                tt = la = lo = gs = np.array([])
            else:
                t = f["t"].to_numpy()
                m = (t >= r["t_first"]) & (t < t_end) if r["t_to"] is not None and r["t_to"] > r["t_first"] else \
                    (t >= r["t_first"]) & (t <= t_end)
                tt, la, lo = t[m], f["lat"].to_numpy()[m], f["lon"].to_numpy()[m]
                gs = f["gs"].fill_null(np.nan).to_numpy()[m]
            nmatch.append(len(tt) == r["n_ground"])
            ev = events(tt, la, lo, gs, r["t_move"])
            sub = ("stand, first motion forward" if r["t_move"] is not None and r["push"] is not None and r["push"] < 0.5
                   else "stand, no usable step" if r["t_move"] is not None and r["push"] is None
                   else "stand, dwell guard" if r["t_move"] is not None
                   else "no stand run")
            rows.append({"day": day, "apt": r["ADEP_mvt"], "sub": sub, "t_mvt": r["t_mvt"], "t_block": r["t_block"],
                         "y": r["y"], "oof": r["oof"], **{f"t_{k}": v for k, v in ev.items()}})
        del fx, byk
    print(f"phase reconstruction: fix count equals stage-1 n_ground on {np.mean(nmatch) * 100:.1f} % of "
          f"{len(nmatch)} rows")
    B = pl.DataFrame(rows, infer_schema_length=None)
    print("rows per airport x sub-group:")
    print(B.group_by("apt", "sub").len().pivot(on="sub", index="apt", values="len").fill_null(0).sort("apt"))
    evs = ["move", "dep1", "gs3", "gs5", "gs10", "d100"]
    out = []
    for grp_name, Gf in [("all", B), ("stand, first motion forward", B.filter(pl.col("sub") == "stand, first motion forward")),
                         ("no stand run", B.filter(pl.col("sub") == "no stand run"))]:
        for e in evs:
            E = Gf.filter(pl.col(f"t_{e}").is_not_null()).with_columns(
                (pl.col(f"t_{e}") - pl.col("t_block")).alias("off"))
            parts = []
            for apt in APTS:
                A = E.filter(pl.col("apt") == apt)
                for h in LAB_DAYS:
                    tr = A.filter(pl.col("day") != h)
                    te = A.filter(pl.col("day") == h)
                    if te.height == 0 or tr.height == 0:
                        continue
                    parts.append(te.with_columns(pl.lit(float(tr["off"].median())).alias("b")))
            if not parts:
                continue
            P = pl.concat(parts).with_columns((pl.col("t_mvt") - pl.col(f"t_{e}") + pl.col("b")).alias("y_e"))
            P = P.with_columns((pl.col("y_e") - pl.col("y")).alias("r"), (pl.col("oof") - pl.col("y")).alias("r_oof"),
                               ((pl.col("oof") - pl.col("y_e")).abs() > GUARD).alias("trip"))
            P = P.with_columns(pl.when(pl.col("trip")).then(pl.col("r_oof")).otherwise(pl.col("r")).alias("r_g"))
            for apt in APTS + ["pooled5", "pooled6"]:
                Q = (P.filter(pl.col("apt").is_in(APTS5)) if apt == "pooled5" else P if apt == "pooled6"
                     else P.filter(pl.col("apt") == apt))
                if Q.height == 0:
                    continue
                r = Q["r"]
                out.append({"group": grp_name, "event": e, "apt": apt, "n_group": Gf.filter(
                    pl.col("apt").is_in(APTS5) if apt == "pooled5" else pl.lit(True) if apt == "pooled6"
                    else pl.col("apt") == apt).height, "n_event_loo": Q.height,
                    "med": float(r.median()), "q25": float(r.quantile(.25)), "q75": float(r.quantile(.75)),
                    "iqr": float(r.quantile(.75) - r.quantile(.25)), "rmse_event": float((r ** 2).mean() ** .5),
                    "rmse_oof": float((Q["r_oof"] ** 2).mean() ** .5),
                    "rmse_guarded": float((Q["r_g"] ** 2).mean() ** .5), "trips": int(Q["trip"].sum()),
                    "gain_s2_row": float((Q["r_oof"] ** 2 - Q["r_g"] ** 2).mean())})
    R = pl.DataFrame(out)
    print("residual y_e - y (s), per-airport bias leave-one-day-out; rmse_guarded = 900 s guard against the OOF")
    for g in R["group"].unique(maintain_order=True).to_list():
        print(f"\n--- group: {g}")
        print(R.filter(pl.col("group") == g).drop("group"))
    R.write_csv(os.path.join(SCR, "adsb_cov_partb.csv"))


if __name__ == "__main__":
    b = bias_all_days()
    part_a(b)
    part_b()

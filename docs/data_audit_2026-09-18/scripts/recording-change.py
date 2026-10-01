"""Recording-change audit: is the INSTRUMENT different between 2025 and 2026?

Lens: not "are the flights different" (covered by the 2026-09-16 shift reports) but
"did the way the data is recorded change at the 2025/2026 boundary".

Source: data/raw/*.parquet only (12 monthly training files + ranking.parquet).
24 month-partitions: 2025-01..2025-12 (one file each) and 2026-01, 2026-07
(ranking.parquet split on month(MVT_TIME_UTC_mvt)).

No training. POLARS_MAX_THREADS=1. Month-by-month, lazy, aggregation-only queries.

Stages
  0  file-level parquet/arrow metadata (writer, row groups, encodings, R version)
  1  per column x per month x per phase: nulls, distinct, min, max
  2  timestamp seconds-of-minute + sub-second recording granularity
  3  NM join rate per month, per airport, per phase; _flt column null rates
     conditional on the join succeeding
  4  categorical vocabulary: new/lost values, unseen-in-training ranking rows,
     frequency shift of values present on both sides
  5  identifier columns: MVT_ID / FLIGHT_ID ranges, integrality, overlap, density
  6  arrival rows: do they behave the same in 2026 (they keep label and BLOCK)

Output: plain text to stdout; report written by hand to
docs/data_audit_2026-09-18/results/recording-change.md
"""

from __future__ import annotations

import base64
import glob
import os
import re
import sys
import warnings

import polars as pl
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(60)
pl.Config.set_tbl_width_chars(400)
pl.Config.set_fmt_str_lengths(60)

RAW = "data/raw"
TRAIN_FILES = sorted(glob.glob(f"{RAW}/training_*.parquet"))
RANK_FILE = f"{RAW}/ranking.parquet"

MVT_TS = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
FLT_TS = ["LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]
TS = MVT_TS + FLT_TS

MVT_CAT = [
    "FLIGHT_mvt",
    "FLIGHT_RULE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
]
FLT_CAT = [
    "CALLSIGN_flt",
    "ADEP_flt",
    "ADES_flt",
    "ADES_FILED_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
]
CAT = MVT_CAT + FLT_CAT
NUM = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt"]
ALL_COLS = NUM + CAT + TS
FLT_COLS = FLT_CAT + FLT_TS

# vocabulary columns whose drift is asked about explicitly
VOCAB = [
    "STAND_mvt",
    "RUNWAY_mvt",
    "AIRCRAFT_TYPE_mvt",
    "ADES_mvt",
    "AIRCRAFT_OPERATOR_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_TYPE_flt",
    "ADES_FILED_flt",
    "ADEP_mvt",
    "FLIGHT_RULE_mvt",
]


def months():
    """Yield (label, lazyframe) for the 24 month-partitions."""
    for f in TRAIN_FILES:
        label = os.path.basename(f)[len("training_") : len("training_") + 7]
        yield label, pl.scan_parquet(f)
    for m, label in ((1, "2026-01"), (7, "2026-07")):
        yield label, pl.scan_parquet(RANK_FILE).filter(
            pl.col("MVT_TIME_UTC_mvt").dt.month() == m
        )


def apcol(phase):
    """The audited airport: ADEP for departures, ADES for arrivals."""
    return "ADEP_mvt" if phase == "DEP" else "ADES_mvt"


def hr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


# --------------------------------------------------------------------------- 0
def stage0():
    hr("STAGE 0 - file-level metadata (writer fingerprint)")
    rows = []
    for f in sorted(glob.glob(f"{RAW}/*.parquet")):
        pf = pq.ParquetFile(f)
        m = pf.metadata
        rg = m.row_group(0)
        comp = {rg.column(i).compression for i in range(m.num_columns)}
        enc = {tuple(rg.column(i).encodings) for i in range(m.num_columns)}
        kv = m.metadata or {}
        arrow_schema = kv.get(b"ARROW:schema", b"")
        raw = base64.b64decode(arrow_schema) if arrow_schema else b""
        # The AIRCRAFT_OPERATOR_flt field carries an `arrow.r.vctrs` extension whose
        # ARROW:extension:metadata is an R `serialize(ascii = TRUE)` blob:
        #   "A\n3\n<writer R version>\n<min reader R version>\n5\nUTF-8\n..."
        # with version = 65536*major + 256*minor + patch.  This is the only
        # writer-environment fingerprint the files carry.
        rver = rmin = ""
        mm = re.search(rb"A\n3\n(\d+)\n(\d+)\n", raw)
        if mm:
            def dec(v):
                v = int(v)
                return f"{v // 65536}.{(v % 65536) // 256}.{v % 256}"
            rver, rmin = dec(mm.group(1)), dec(mm.group(2))
        rows.append(
            dict(
                file=os.path.basename(f),
                rows=m.num_rows,
                row_groups=m.num_row_groups,
                cols=m.num_columns,
                created_by=m.created_by,
                fmt=m.format_version,
                comp=",".join(sorted(comp)),
                enc=str(sorted(enc)),
                arrow_schema_len=len(arrow_schema),
                r_writer_version=rver,
                r_min_reader=rmin,
                bytes_on_disk=os.path.getsize(f),
                rg_bytes=rg.total_byte_size,
            )
        )
    df = pl.DataFrame(rows)
    print(
        df.select(
            "file", "rows", "row_groups", "created_by", "fmt", "comp",
            "arrow_schema_len", "r_writer_version", "r_min_reader",
        )
    )
    print("\ndistinct created_by:", df["created_by"].unique().to_list())
    print("distinct encodings:", df["enc"].unique().to_list())
    print("distinct R writer version:", df["r_writer_version"].unique().to_list())

    # schema equality across files
    ref = pq.ParquetFile(TRAIN_FILES[0]).schema_arrow
    for f in sorted(glob.glob(f"{RAW}/*.parquet")):
        s = pq.ParquetFile(f).schema_arrow
        if f.endswith("submitting.parquet"):
            continue
        print(
            f"schema equals Jan25: {os.path.basename(f):45s} "
            f"{s.equals(ref, check_metadata=False)} "
            f"(with metadata {s.equals(ref, check_metadata=True)})"
        )


# --------------------------------------------------------------------------- 1
def stage1():
    hr("STAGE 1 - per column x month x phase: n, null%, n_distinct, min, max")
    out = []
    for label, lf in months():
        for phase in ("DEP", "ARR"):
            q = lf.filter(pl.col("PHASE_mvt") == phase)
            aggs = [pl.len().alias("n")]
            for c in ALL_COLS:
                aggs += [
                    pl.col(c).null_count().alias(f"{c}|null"),
                    pl.col(c).n_unique().alias(f"{c}|nuniq"),
                    pl.col(c).min().cast(pl.String).alias(f"{c}|min"),
                    pl.col(c).max().cast(pl.String).alias(f"{c}|max"),
                ]
            r = q.select(aggs).collect().row(0, named=True)
            n = r["n"]
            for c in ALL_COLS:
                out.append(
                    dict(
                        month=label,
                        phase=phase,
                        col=c,
                        n=n,
                        null_pct=round(100 * r[f"{c}|null"] / n, 4) if n else None,
                        n_distinct=r[f"{c}|nuniq"],
                        vmin=r[f"{c}|min"],
                        vmax=r[f"{c}|max"],
                    )
                )
            print(f"  stage1 {label} {phase} n={n}", flush=True)
    df = pl.DataFrame(out)
    df.write_parquet(f"{TMP}/stage1.parquet")

    for phase in ("DEP", "ARR"):
        for c in ALL_COLS:
            sub = df.filter((pl.col("phase") == phase) & (pl.col("col") == c)).sort("month")
            print(f"\n--- {phase} :: {c} ---")
            print(sub.select("month", "n", "null_pct", "n_distinct", "vmin", "vmax"))


def stage1_summary():
    """Compact: which columns step-change at the boundary."""
    hr("STAGE 1b - boundary step detection (2025 range vs 2026 value)")
    df = pl.read_parquet(f"{TMP}/stage1.parquet")
    for phase in ("DEP", "ARR"):
        print(f"\n### phase {phase}: null% -- 2025 min..max, then 2026-01 / 2026-07")
        rows = []
        for c in ALL_COLS:
            s = df.filter((pl.col("phase") == phase) & (pl.col("col") == c))
            t = s.filter(pl.col("month").str.starts_with("2025"))["null_pct"]
            j = s.filter(pl.col("month") == "2026-01")["null_pct"][0]
            u = s.filter(pl.col("month") == "2026-07")["null_pct"][0]
            lo, hi = t.min(), t.max()
            outside = not (lo <= j <= hi) or not (lo <= u <= hi)
            rows.append(
                dict(col=c, t25_min=lo, t25_max=hi, m2026_01=j, m2026_07=u,
                     outside_2025_range=outside)
            )
        print(pl.DataFrame(rows))

        print(f"\n### phase {phase}: n_distinct per 1000 rows")
        rows = []
        for c in ALL_COLS:
            s = df.filter((pl.col("phase") == phase) & (pl.col("col") == c)).with_columns(
                (1000 * pl.col("n_distinct") / pl.col("n")).round(3).alias("d")
            )
            t = s.filter(pl.col("month").str.starts_with("2025"))["d"]
            rows.append(
                dict(
                    col=c,
                    t25_min=t.min(),
                    t25_max=t.max(),
                    m2026_01=s.filter(pl.col("month") == "2026-01")["d"][0],
                    m2026_07=s.filter(pl.col("month") == "2026-07")["d"][0],
                )
            )
        print(pl.DataFrame(rows))


# --------------------------------------------------------------------------- 2
def stage2():
    hr("STAGE 2 - timestamp recording granularity (seconds-of-minute, sub-second)")
    out = []
    for label, lf in months():
        for phase in ("DEP", "ARR"):
            q = lf.filter(pl.col("PHASE_mvt") == phase)
            aggs = [pl.len().alias("n")]
            for c in TS:
                sec = pl.col(c).dt.second()
                us = pl.col(c).dt.microsecond()
                nn = pl.col(c).is_not_null()
                aggs += [
                    nn.sum().alias(f"{c}|nn"),
                    (sec == 0).sum().alias(f"{c}|s0"),
                    (sec % 30 == 0).sum().alias(f"{c}|s30"),
                    (sec % 10 == 0).sum().alias(f"{c}|s10"),
                    sec.n_unique().alias(f"{c}|snuniq"),
                    (us != 0).sum().alias(f"{c}|usnz"),
                    (pl.col(c).dt.minute() % 5 == 0).sum().alias(f"{c}|m5"),
                ]
            r = q.select(aggs).collect().row(0, named=True)
            for c in TS:
                nn = r[f"{c}|nn"]
                if nn == 0:
                    out.append(dict(month=label, phase=phase, col=c, n_nonnull=0))
                    continue
                out.append(
                    dict(
                        month=label,
                        phase=phase,
                        col=c,
                        n_nonnull=nn,
                        sec0_pct=round(100 * r[f"{c}|s0"] / nn, 3),
                        sec_mod30_pct=round(100 * r[f"{c}|s30"] / nn, 3),
                        sec_mod10_pct=round(100 * r[f"{c}|s10"] / nn, 3),
                        sec_distinct=r[f"{c}|snuniq"],
                        subsec_nonzero=r[f"{c}|usnz"],
                        min_mod5_pct=round(100 * r[f"{c}|m5"] / nn, 3),
                    )
                )
            print(f"  stage2 {label} {phase}", flush=True)
    df = pl.DataFrame(out)
    df.write_parquet(f"{TMP}/stage2.parquet")
    for phase in ("DEP", "ARR"):
        for c in TS:
            print(f"\n--- sec-of-minute {phase} :: {c} ---")
            print(
                df.filter((pl.col("phase") == phase) & (pl.col("col") == c))
                .sort("month")
                .select("month", "n_nonnull", "sec0_pct", "sec_mod10_pct",
                        "sec_mod30_pct", "sec_distinct", "subsec_nonzero",
                        "min_mod5_pct")
            )


def stage2b():
    """Full seconds histogram for MVT_TIME, per month, DEP: chi-square-free look."""
    hr("STAGE 2b - MVT_TIME/BLOCK_TIME seconds histogram shape, DEP")
    for c in ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"):
        rows = []
        for label, lf in months():
            h = (
                lf.filter(pl.col("PHASE_mvt") == "DEP")
                .select(pl.col(c).dt.second().alias("s"))
                .drop_nulls()
                .group_by("s")
                .agg(pl.len().alias("k"))
                .collect()
            )
            tot = h["k"].sum()
            if tot == 0:
                continue
            h = h.with_columns((pl.col("k") / tot).alias("p"))
            p = h.sort("s")["p"].to_list()
            uni = 1.0 / len(p)
            tv = 0.5 * sum(abs(x - uni) for x in p)
            top = h.sort("k", descending=True).head(3)
            rows.append(
                dict(
                    month=label,
                    n=tot,
                    n_sec_values=len(p),
                    tv_vs_uniform=round(tv, 4),
                    top3=", ".join(
                        f"{s}:{100 * pp:.2f}%"
                        for s, pp in zip(top["s"].to_list(), top["p"].to_list())
                    ),
                )
            )
            print(f"  stage2b {c} {label}", flush=True)
        print(f"\n--- {c} DEP ---")
        print(pl.DataFrame(rows))


# --------------------------------------------------------------------------- 3
def stage3():
    hr("STAGE 3 - NM join rate per month / per airport / per phase")
    joined = pl.col("FLIGHT_ID_mvt").is_not_null()
    out = []
    per_ap = []
    for label, lf in months():
        for phase in ("DEP", "ARR"):
            q = lf.filter(pl.col("PHASE_mvt") == phase)
            r = q.select(
                pl.len().alias("n"),
                joined.sum().alias("joined"),
                pl.col("AOBT_3_flt").is_not_null().sum().alias("aobt3_nn"),
                (joined & pl.col("AOBT_3_flt").is_null()).sum().alias("j_no_aobt3"),
                (~joined & pl.col("AOBT_3_flt").is_not_null()).sum().alias("nj_aobt3"),
                (joined & pl.col("CALLSIGN_flt").is_null()).sum().alias("j_no_cs"),
                (~joined & pl.col("CALLSIGN_flt").is_not_null()).sum().alias("nj_cs"),
            ).collect().row(0, named=True)
            out.append(
                dict(
                    month=label,
                    phase=phase,
                    n=r["n"],
                    join_pct=round(100 * r["joined"] / r["n"], 4),
                    unmatched=r["n"] - r["joined"],
                    unmatched_pct=round(100 * (r["n"] - r["joined"]) / r["n"], 4),
                    aobt3_nn_pct=round(100 * r["aobt3_nn"] / r["n"], 4),
                    joined_but_no_aobt3=r["j_no_aobt3"],
                    unjoined_but_aobt3=r["nj_aobt3"],
                    joined_but_no_callsign=r["j_no_cs"],
                    unjoined_but_callsign=r["nj_cs"],
                )
            )
            g = (
                q.group_by(pl.col(apcol(phase)).alias("airport"))
                .agg(pl.len().alias("n"), joined.sum().alias("j"))
                .collect()
                .with_columns(
                    (100 * (pl.col("n") - pl.col("j")) / pl.col("n"))
                    .round(3)
                    .alias("unmatched_pct")
                )
                .with_columns(pl.lit(label).alias("month"), pl.lit(phase).alias("phase"))
            )
            per_ap.append(g)
            print(f"  stage3 {label} {phase}", flush=True)
    df = pl.DataFrame(out)
    df.write_parquet(f"{TMP}/stage3.parquet")
    for phase in ("DEP", "ARR"):
        print(f"\n--- join rate, phase {phase} ---")
        print(df.filter(pl.col("phase") == phase).sort("month"))

    ap = pl.concat(per_ap)
    ap.write_parquet(f"{TMP}/stage3_ap.parquet")
    for phase in ("DEP", "ARR"):
        piv = (
            ap.filter(pl.col("phase") == phase)
            .pivot(on="month", index="airport", values="unmatched_pct")
            .sort("airport")
        )
        print(f"\n--- unmatched % per airport x month, phase {phase} ---")
        print(piv)

    hr("STAGE 3b - _flt column null rate CONDITIONAL on the join succeeding (DEP)")
    out = []
    for label, lf in months():
        q = lf.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("FLIGHT_ID_mvt").is_not_null())
        aggs = [pl.len().alias("n")] + [
            pl.col(c).null_count().alias(c) for c in FLT_COLS
        ]
        r = q.select(aggs).collect().row(0, named=True)
        n = r["n"]
        d = dict(month=label, n_matched=n)
        for c in FLT_COLS:
            d[c] = round(100 * r[c] / n, 3) if n else None
        out.append(d)
        print(f"  stage3b {label}", flush=True)
    print(pl.DataFrame(out).sort("month"))


# --------------------------------------------------------------------------- 4
def stage4():
    hr("STAGE 4 - categorical vocabulary drift")
    # per month value counts for the vocabulary columns, DEP rows
    frames = []
    for label, lf in months():
        for c in VOCAB:
            g = (
                lf.filter(pl.col("PHASE_mvt") == "DEP")
                .group_by(pl.col(c).alias("value"))
                .agg(pl.len().alias("k"))
                .collect()
                .with_columns(pl.lit(label).alias("month"), pl.lit(c).alias("col"))
            )
            frames.append(g)
        # (ADEP, STAND) and (ADEP, RUNWAY) pairs: stands are airport-local
        for c in ("STAND_mvt", "RUNWAY_mvt"):
            g = (
                lf.filter(pl.col("PHASE_mvt") == "DEP")
                .group_by(
                    (pl.col("ADEP_mvt") + pl.lit("/") + pl.col(c).fill_null("<NULL>"))
                    .alias("value")
                )
                .agg(pl.len().alias("k"))
                .collect()
                .with_columns(pl.lit(label).alias("month"), pl.lit(f"ADEP+{c}").alias("col"))
            )
            frames.append(g)
        print(f"  stage4 counts {label}", flush=True)
    vc = pl.concat(frames)
    vc.write_parquet(f"{TMP}/stage4_vc.parquet")

    cols = VOCAB + ["ADEP+STAND_mvt", "ADEP+RUNWAY_mvt"]
    tr = vc.filter(pl.col("month").str.starts_with("2025"))
    print("\n--- vocabulary size and new/lost values (DEP rows) ---")
    rows = []
    for c in cols:
        tset = set(tr.filter(pl.col("col") == c)["value"].drop_nulls().to_list())
        for m in ("2026-01", "2026-07"):
            r = vc.filter((pl.col("col") == c) & (pl.col("month") == m))
            rset = set(r["value"].drop_nulls().to_list())
            new = rset - tset
            n_rows = r["k"].sum()
            new_rows = r.filter(pl.col("value").is_in(list(new)) if new else pl.lit(False))["k"].sum()
            rows.append(
                dict(
                    col=c,
                    month=m,
                    train2025_distinct=len(tset),
                    month_distinct=len(rset),
                    new_values=len(new),
                    rows_on_new_values=new_rows or 0,
                    pct_rows_on_new=round(100 * (new_rows or 0) / n_rows, 4) if n_rows else None,
                    lost_values=len(tset - rset),
                    example_new=", ".join(sorted(map(str, list(new)))[:8]),
                )
            )
    print(pl.DataFrame(rows))

    print("\n--- unseen-in-training rows over the WHOLE ranking set (both months, DEP) ---")
    rows = []
    for c in cols:
        tset = set(tr.filter(pl.col("col") == c)["value"].drop_nulls().to_list())
        r = vc.filter((pl.col("col") == c) & pl.col("month").str.starts_with("2026"))
        agg = r.group_by("value").agg(pl.len(), pl.col("k").sum())
        unseen = agg.filter(~pl.col("value").is_in(list(tset)))
        rows.append(
            dict(
                col=c,
                unseen_values=unseen.height,
                unseen_rows=unseen["k"].sum() or 0,
                rank_rows=r["k"].sum(),
            )
        )
    print(pl.DataFrame(rows))

    print("\n--- per-month new-value arrival WITHIN 2025 (baseline churn) ---")
    # for each 2025 month m, how many values are new relative to all earlier 2025 months
    labels = [os.path.basename(f)[9:16] for f in TRAIN_FILES]
    for c in ("STAND_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
              "RUNWAY_mvt", "ADEP+STAND_mvt"):
        rows = []
        seen: set = set()
        for m in labels + ["2026-01", "2026-07"]:
            r = vc.filter((pl.col("col") == c) & (pl.col("month") == m))
            vals = set(r["value"].drop_nulls().to_list())
            new = vals - seen if seen else set()
            nr = r.filter(pl.col("value").is_in(list(new)) if new else pl.lit(False))["k"].sum() or 0
            tot = r["k"].sum()
            rows.append(
                dict(month=m, distinct=len(vals), new_vs_all_earlier=len(new) if seen else None,
                     rows_on_new=nr, pct=round(100 * nr / tot, 4) if tot else None)
            )
            seen |= vals
        print(f"\n  {c}")
        print(pl.DataFrame(rows))

    print("\n--- frequency shift of values present on BOTH sides (TV distance) ---")
    rows = []
    for c in cols:
        t = (
            tr.filter(pl.col("col") == c)
            .group_by("value")
            .agg(pl.col("k").sum())
            .with_columns((pl.col("k") / pl.col("k").sum()).alias("p"))
        )
        # within-2025 baseline: Jan25 vs Jul25
        for name, a_months, b_months in (
            ("2025 all vs 2026-01", labels, ["2026-01"]),
            ("2025 all vs 2026-07", labels, ["2026-07"]),
            ("2025-01 vs 2025-07 (baseline)", ["2025-01"], ["2025-07"]),
            ("2026-01 vs 2026-07", ["2026-01"], ["2026-07"]),
        ):
            def dist(ms):
                d = (
                    vc.filter((pl.col("col") == c) & pl.col("month").is_in(ms))
                    .group_by("value")
                    .agg(pl.col("k").sum())
                )
                return d.with_columns((pl.col("k") / pl.col("k").sum()).alias("p"))

            A, B = dist(a_months), dist(b_months)
            j = A.join(B, on="value", how="full", coalesce=True).with_columns(
                pl.col("p").fill_null(0.0), pl.col("p_right").fill_null(0.0)
            )
            tv = 0.5 * (j["p"] - j["p_right"]).abs().sum()
            rows.append(dict(col=c, pair=name, tv=round(tv, 4)))
    print(pl.DataFrame(rows).pivot(on="pair", index="col"))
    _ = t


# --------------------------------------------------------------------------- 5
def stage5():
    hr("STAGE 5 - identifier columns: range, integrality, overlap, density")
    out = []
    for label, lf in months():
        r = (
            lf.select(
                pl.len().alias("n"),
                pl.col("MVT_ID_mvt").min().alias("mvt_min"),
                pl.col("MVT_ID_mvt").max().alias("mvt_max"),
                pl.col("MVT_ID_mvt").n_unique().alias("mvt_nuniq"),
                pl.col("MVT_ID_mvt").null_count().alias("mvt_null"),
                (pl.col("MVT_ID_mvt") % 1 != 0).sum().alias("mvt_nonint"),
                pl.col("FLIGHT_ID_mvt").min().alias("fid_min"),
                pl.col("FLIGHT_ID_mvt").max().alias("fid_max"),
                pl.col("FLIGHT_ID_mvt").n_unique().alias("fid_nuniq"),
                (pl.col("FLIGHT_ID_mvt") % 1 != 0).sum().alias("fid_nonint"),
                pl.col("MVT_ID_mvt").quantile(0.5).alias("mvt_med"),
                pl.col("FLIGHT_ID_mvt").quantile(0.5).alias("fid_med"),
            )
            .collect()
            .row(0, named=True)
        )
        r["month"] = label
        r["mvt_span"] = r["mvt_max"] - r["mvt_min"]
        r["mvt_fill_density"] = round(r["n"] / r["mvt_span"], 4) if r["mvt_span"] else None
        r["fid_span"] = (r["fid_max"] - r["fid_min"]) if r["fid_max"] is not None else None
        out.append(r)
        print(f"  stage5 {label}", flush=True)
    df = pl.DataFrame(out).select(
        "month", "n", "mvt_min", "mvt_max", "mvt_span", "mvt_med", "mvt_nuniq",
        "mvt_null", "mvt_nonint", "mvt_fill_density", "fid_min", "fid_max",
        "fid_span", "fid_med", "fid_nuniq", "fid_nonint",
    )
    print(df)

    print("\n--- ID overlap between training and ranking ---")
    tr_ids = pl.concat(
        [pl.scan_parquet(f).select("MVT_ID_mvt") for f in TRAIN_FILES]
    ).collect()["MVT_ID_mvt"]
    rk_ids = pl.scan_parquet(RANK_FILE).select("MVT_ID_mvt").collect()["MVT_ID_mvt"]
    ts, rs = set(tr_ids.to_list()), set(rk_ids.to_list())
    print(f"train MVT_ID distinct {len(ts)} of {len(tr_ids)} rows; "
          f"rank {len(rs)} of {len(rk_ids)}; intersection {len(ts & rs)}")
    print(f"train max {max(ts)}  rank min {min(rs)}  rank max {max(rs)}  "
          f"gap rank_min - train_max = {min(rs) - max(ts)}")
    del ts, rs, tr_ids, rk_ids

    tr_f = pl.concat(
        [pl.scan_parquet(f).select("FLIGHT_ID_mvt") for f in TRAIN_FILES]
    ).collect()["FLIGHT_ID_mvt"].drop_nulls()
    rk_f = pl.scan_parquet(RANK_FILE).select("FLIGHT_ID_mvt").collect()["FLIGHT_ID_mvt"].drop_nulls()
    ts, rs = set(tr_f.to_list()), set(rk_f.to_list())
    print(f"train FLIGHT_ID distinct {len(ts)}; rank {len(rs)}; intersection {len(ts & rs)}")
    print(f"train max {max(ts)}  rank min {min(rs)}  gap = {min(rs) - max(ts)}")

    print("\n--- is MVT_ID monotone in MVT_TIME within a month? (DEP, Spearman-ish) ---")
    for label, lf in months():
        d = (
            lf.filter(pl.col("PHASE_mvt") == "DEP")
            .select("MVT_ID_mvt", "MVT_TIME_UTC_mvt")
            .collect()
        )
        c = d.select(
            pl.corr(
                pl.col("MVT_ID_mvt").rank(),
                pl.col("MVT_TIME_UTC_mvt").cast(pl.Int64).rank(),
            ).alias("r")
        )["r"][0]
        print(f"  {label}: rank-corr(MVT_ID, MVT_TIME) = {c:.4f}")


# --------------------------------------------------------------------------- 6
def stage6():
    hr("STAGE 6 - arrival rows: the unused window onto 2026 recording")
    out = []
    for label, lf in months():
        q = lf.filter(pl.col("PHASE_mvt") == "ARR")
        r = (
            q.select(
                pl.len().alias("n"),
                pl.col("TAXITIME_SEC_mvt").null_count().alias("y_null"),
                pl.col("TAXITIME_SEC_mvt").min().alias("y_min"),
                pl.col("TAXITIME_SEC_mvt").quantile(0.05).alias("y_p05"),
                pl.col("TAXITIME_SEC_mvt").median().alias("y_med"),
                pl.col("TAXITIME_SEC_mvt").quantile(0.95).alias("y_p95"),
                pl.col("TAXITIME_SEC_mvt").quantile(0.999).alias("y_p999"),
                pl.col("TAXITIME_SEC_mvt").max().alias("y_max"),
                (pl.col("TAXITIME_SEC_mvt") <= 0).sum().alias("y_le0"),
                (pl.col("TAXITIME_SEC_mvt") >= 7200).sum().alias("y_ge2h"),
                (
                    (
                        pl.col("BLOCK_TIME_UTC_mvt").cast(pl.Int64)
                        - pl.col("MVT_TIME_UTC_mvt").cast(pl.Int64)
                    )
                    // 1_000_000
                    - pl.col("TAXITIME_SEC_mvt")
                )
                .abs()
                .max()
                .alias("identity_max_abs_err"),
                (
                    (
                        pl.col("BLOCK_TIME_UTC_mvt").cast(pl.Int64)
                        - pl.col("SCHED_TIME_UTC_mvt").cast(pl.Int64)
                    ).abs()
                    <= 10_000_000
                )
                .mean()
                .alias("block_eq_sched_frac"),
                pl.col("FLIGHT_ID_mvt").is_null().mean().alias("unmatched_frac"),
                pl.col("STAND_mvt").is_null().mean().alias("stand_null_frac"),
                pl.col("RUNWAY_mvt").is_null().mean().alias("runway_null_frac"),
                pl.col("AIRCRAFT_TYPE_mvt").is_null().mean().alias("actype_null_frac"),
            )
            .collect()
            .row(0, named=True)
        )
        r["month"] = label
        out.append(r)
        print(f"  stage6 {label}", flush=True)
    df = pl.DataFrame(out)
    print(
        df.select(
            "month", "n", "y_null", "y_min", "y_p05", "y_med", "y_p95", "y_p999",
            "y_max", "y_le0", "y_ge2h", "identity_max_abs_err",
        )
    )
    print(
        df.select(
            "month", "block_eq_sched_frac", "unmatched_frac", "stand_null_frac",
            "runway_null_frac", "actype_null_frac",
        ).with_columns(
            (100 * pl.col(c)).round(4).alias(c)
            for c in ("block_eq_sched_frac", "unmatched_frac", "stand_null_frac",
                      "runway_null_frac", "actype_null_frac")
        )
    )

    print("\n--- ARR taxi-in median per airport x month (recording sanity) ---")
    frames = []
    for label, lf in months():
        g = (
            lf.filter(pl.col("PHASE_mvt") == "ARR")
            .group_by(pl.col("ADES_mvt").alias("airport"))
            .agg(pl.col("TAXITIME_SEC_mvt").median().alias("med"))
            .collect()
            .with_columns(pl.lit(label).alias("month"))
        )
        frames.append(g)
    print(
        pl.concat(frames)
        .pivot(on="month", index="airport", values="med")
        .sort("airport")
    )

    print("\n--- ARR: block_eq_sched (LIRF-style fill) per airport x month ---")
    frames = []
    for label, lf in months():
        g = (
            lf.filter(pl.col("PHASE_mvt") == "ARR")
            .group_by(pl.col("ADES_mvt").alias("airport"))
            .agg(
                (
                    (
                        pl.col("BLOCK_TIME_UTC_mvt").cast(pl.Int64)
                        - pl.col("SCHED_TIME_UTC_mvt").cast(pl.Int64)
                    ).abs()
                    <= 10_000_000
                )
                .mean()
                .mul(100)
                .round(2)
                .alias("fill_pct")
            )
            .collect()
            .with_columns(pl.lit(label).alias("month"))
        )
        frames.append(g)
    print(
        pl.concat(frames)
        .pivot(on="month", index="airport", values="fill_pct")
        .sort("airport")
    )

    print("\n--- DEP counterpart of the same sanity columns (for comparison) ---")
    out = []
    for label, lf in months():
        q = lf.filter(pl.col("PHASE_mvt") == "DEP")
        r = (
            q.select(
                pl.len().alias("n"),
                pl.col("TAXITIME_SEC_mvt").null_count().alias("y_null"),
                pl.col("BLOCK_TIME_UTC_mvt").null_count().alias("block_null"),
                pl.col("SCHED_TIME_UTC_mvt").null_count().alias("sched_null"),
                pl.col("FLIGHT_ID_mvt").is_null().mean().mul(100).round(4).alias("unmatched_pct"),
                pl.col("STAND_mvt").is_null().mean().mul(100).round(4).alias("stand_null_pct"),
                pl.col("RUNWAY_mvt").is_null().mean().mul(100).round(4).alias("runway_null_pct"),
                pl.col("AIRCRAFT_TYPE_mvt").is_null().mean().mul(100).round(4).alias("actype_null_pct"),
                (
                    (
                        pl.col("MVT_TIME_UTC_mvt").cast(pl.Int64)
                        - pl.col("SCHED_TIME_UTC_mvt").cast(pl.Int64)
                    )
                    // 1_000_000
                )
                .median()
                .alias("mvt_minus_sched_med"),
            )
            .collect()
            .row(0, named=True)
        )
        r["month"] = label
        out.append(r)
    print(pl.DataFrame(out))


# --------------------------------------------------------------------------- 7
def stage7():
    hr("STAGE 7 - string format fingerprints (length / charset), DEP")
    cols = ["FLIGHT_mvt", "CALLSIGN_flt", "STAND_mvt", "RUNWAY_mvt",
            "AIRCRAFT_OPERATOR_flt", "ADES_mvt", "AIRCRAFT_TYPE_mvt",
            "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt"]
    out = []
    for label, lf in months():
        q = lf.filter(pl.col("PHASE_mvt") == "DEP")
        aggs = [pl.len().alias("n")]
        for c in cols:
            ln = pl.col(c).str.len_chars()
            aggs += [
                ln.min().alias(f"{c}|lmin"),
                ln.max().alias(f"{c}|lmax"),
                ln.mean().alias(f"{c}|lmean"),
                pl.col(c).str.contains(r"^[A-Z0-9]+$").mean().alias(f"{c}|az09"),
                pl.col(c).str.contains(r"[a-z]").mean().alias(f"{c}|lower"),
                pl.col(c).str.contains(r"\s").mean().alias(f"{c}|space"),
            ]
        r = q.select(aggs).collect().row(0, named=True)
        for c in cols:
            out.append(
                dict(
                    month=label,
                    col=c,
                    len_min=r[f"{c}|lmin"],
                    len_max=r[f"{c}|lmax"],
                    len_mean=round(r[f"{c}|lmean"], 3) if r[f"{c}|lmean"] is not None else None,
                    pct_AZ09=round(100 * r[f"{c}|az09"], 3) if r[f"{c}|az09"] is not None else None,
                    pct_has_lower=round(100 * r[f"{c}|lower"], 4) if r[f"{c}|lower"] is not None else None,
                    pct_has_space=round(100 * r[f"{c}|space"], 4) if r[f"{c}|space"] is not None else None,
                )
            )
        print(f"  stage7 {label}", flush=True)
    df = pl.DataFrame(out)
    for c in cols:
        print(f"\n--- {c} ---")
        print(df.filter(pl.col("col") == c).sort("month").drop("col"))


# --------------------------------------------------------------------------- 8
def stage8():
    hr("STAGE 8 - feed coverage: rows and calendar days per airport x month (DEP)")
    frames = []
    for label, lf in months():
        g = (
            lf.filter(pl.col("PHASE_mvt") == "DEP")
            .group_by("ADEP_mvt")
            .agg(
                pl.len().alias("n"),
                pl.col("MVT_TIME_UTC_mvt").dt.date().n_unique().alias("days"),
            )
            .collect()
            .with_columns(pl.lit(label).alias("month"))
        )
        frames.append(g)
        print(f"  stage8 {label}", flush=True)
    d = pl.concat(frames)
    print("\n--- DEP rows per airport x month ---")
    print(d.pivot(on="month", index="ADEP_mvt", values="n").sort("ADEP_mvt"))
    print("\n--- distinct calendar days with >=1 departure, per airport x month ---")
    print(d.pivot(on="month", index="ADEP_mvt", values="days").sort("ADEP_mvt"))
    print("\n--- airport share of departures (%) per month ---")
    tot = d.group_by("month").agg(pl.col("n").sum().alias("tot"))
    print(
        d.join(tot, on="month")
        .with_columns((100 * pl.col("n") / pl.col("tot")).round(3).alias("pct"))
        .pivot(on="month", index="ADEP_mvt", values="pct")
        .sort("ADEP_mvt")
    )

    hr("STAGE 8b - MVT_TIME seconds-of-minute distribution vs pooled 2025 (TV), DEP")
    hists = {}
    for label, lf in months():
        h = (
            lf.filter(pl.col("PHASE_mvt") == "DEP")
            .select(pl.col("MVT_TIME_UTC_mvt").dt.second().alias("s"))
            .group_by("s")
            .agg(pl.len().alias("k"))
            .collect()
        )
        hists[label] = h.with_columns((pl.col("k") / h["k"].sum()).alias("p"))
    ref = (
        pl.concat([v.select("s", "k") for m, v in hists.items() if m.startswith("2025")])
        .group_by("s")
        .agg(pl.col("k").sum())
    )
    ref = ref.with_columns((pl.col("k") / ref["k"].sum()).alias("p"))
    rows = []
    for m, h in hists.items():
        j = h.join(ref, on="s", how="full", coalesce=True).with_columns(
            pl.col("p").fill_null(0.0), pl.col("p_right").fill_null(0.0)
        )
        rows.append(dict(month=m, tv_vs_pooled2025=round(0.5 * (j["p"] - j["p_right"]).abs().sum(), 5),
                         pct_within_5s_of_minute=round(
                             100 * h.filter((pl.col("s") <= 5) | (pl.col("s") >= 55))["p"].sum(), 3)))
    print(pl.DataFrame(rows))


STAGES = {
    "0": stage0, "1": stage1, "1b": stage1_summary, "2": stage2, "2b": stage2b,
    "3": stage3, "4": stage4, "5": stage5, "6": stage6, "7": stage7, "8": stage8,
}

if __name__ == "__main__":
    TMP = os.environ.get("AUDIT_TMP", "/tmp/recchg")
    os.makedirs(TMP, exist_ok=True)
    want = sys.argv[1:] or list(STAGES)
    for s in want:
        STAGES[s]()

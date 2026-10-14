"""Recompute the seq_* and fl_nm_rate_other columns of every 2025 departure over the ranking's window shape.

    .venv/bin/python docs/gm_research_2026-10-01/build_seq_windows.py   # writes data/work/seq_window.parquet

The ranking frame is two months (Jan + Jul 2026) and prc.features_extra computes these columns over the frame it is
given, so for each month pair {m, m + 6} of 2025 this runs the same code (`mask_departures` + `extra_context_features`
+ `flight_features`) on that pair alone. Edge effects at month boundaries come out the way they do on the ranking
frame. Six pairs, one at a time, to keep memory low. Labels are never read by these functions."""

import gc
from pathlib import Path

import polars as pl

from prc.features import mask_departures
from prc.features_extra import SEQ_COLS, extra_context_features, flight_features

RAW, OUT = Path("data/raw"), Path("data/work/seq_window.parquet")


def main() -> None:
    files = sorted(RAW.glob("training_*.parquet"))
    parts = []
    for p in range(6):
        months = (p + 1, p + 7)
        sel = [f for f in files if int(f.name.split("_")[1][5:7]) in months]
        raw = pl.concat([pl.read_parquet(f) for f in sel], how="diagonal")
        ctx_src = mask_departures(raw)
        dep_ids = raw.filter(pl.col("PHASE_mvt") == "DEP")["MVT_ID_mvt"]
        ctx = extra_context_features(ctx_src, dep_ids).select(["MVT_ID_mvt", *SEQ_COLS])
        fl = flight_features(ctx_src.filter(pl.col("PHASE_mvt") == "DEP")).select("MVT_ID_mvt", "fl_nm_rate_other")
        part = (
            pl.DataFrame({"MVT_ID_mvt": dep_ids})
            .join(ctx, on="MVT_ID_mvt", how="left")
            .join(fl, on="MVT_ID_mvt", how="left")
        )
        print(f"pair {months}: {len(sel)} files, {part.height:,} departures")
        parts.append(part)
        del raw, ctx_src, ctx, fl
        gc.collect()
    out = pl.concat(parts)
    if out["MVT_ID_mvt"].n_unique() != out.height:
        raise AssertionError("a departure appears in two month pairs")
    out = out.rename({c: f"{c}__w" for c in out.columns if c != "MVT_ID_mvt"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)
    print(f"wrote {OUT}: {out.height:,} rows, {len(out.columns) - 1} columns")


if __name__ == "__main__":
    main()

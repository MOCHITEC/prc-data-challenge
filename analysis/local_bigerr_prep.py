"""Big-error anatomy, step 0: join the 7-month OOF (cv_rows_months7_v2) with raw NM/APDF time fields and ADS-B
stage-1 features, all as seconds relative to take-off (MVT). Writes $SCRATCH/agents/matched/bigerr_base.parquet
(train, all 1,233,234 OOF rows) and bigerr_rank.parquet (2026 ranking rows, same columns minus labels, plus v40 pred).

Columns (seconds, all relative so that "x" means MVT - X): d_aobt, d_sched, d_lobt, d_iobt, d_eobt, d_move, d_push,
d_first, d_to; y = MVT - BLOCK (label), p = b + c_C.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_bigerr_prep.py
"""
import os

import polars as pl

S = os.environ["SCRATCH"]
OUT = os.path.join(S, "agents", "matched")
os.makedirs(OUT, exist_ok=True)
RAW = ["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
       "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt", "STAND_mvt", "RUNWAY_mvt", "CALLSIGN_flt",
       "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_mvt", "PHASE_mvt"]
ADS = ["MVT_ID_mvt", "join", "ground_fix", "t_mvt", "t_first", "t_move", "t_move_lo", "t_push", "t_to", "push",
       "post_dwell", "stand_dur", "n_ground", "t_gs3", "t_glast", "seen"]


def rel(df: pl.DataFrame) -> pl.DataFrame:
    m = pl.col("MVT_TIME_UTC_mvt")
    sec = lambda c: (m - pl.col(c)).dt.total_seconds().cast(pl.Float64)  # noqa: E731
    tm = m.dt.epoch("s").cast(pl.Float64)
    return df.with_columns(
        sec("AOBT_3_flt").alias("d_aobt"), sec("SCHED_TIME_UTC_mvt").alias("d_sched"), sec("LOBT_flt").alias("d_lobt"),
        sec("IOBT_flt").alias("d_iobt"), sec("EOBT_1_flt").alias("d_eobt"),
        (tm - pl.col("t_move")).alias("d_move"), (tm - pl.col("t_push")).alias("d_push"),
        (tm - pl.col("t_first")).alias("d_first"), (tm - pl.col("t_to")).alias("d_to"),
        (tm - pl.col("t_gs3")).alias("d_gs3"),
        m.dt.date().cast(pl.String).alias("day"), m.dt.hour().alias("hour_utc"),
    )


def main() -> None:
    cv = pl.read_parquet(os.path.join(S, "learn", "cv_rows_months7_v2.parquet")).with_columns(
        (pl.col("b") + pl.col("c_C")).alias("p"))
    files = sorted(f for f in os.listdir("data/raw") if f.startswith("training_2025"))
    raw = pl.concat([pl.read_parquet(os.path.join("data/raw", f), columns=RAW)
                     .filter(pl.col("PHASE_mvt") == "DEP") for f in files])
    raw = raw.filter(pl.col("MVT_ID_mvt").is_in(cv["MVT_ID_mvt"].implode()))
    F = pl.concat([pl.read_parquet(os.path.join(S, "learn", f), columns=ADS)
                   for f in ("feat_2025.parquet", "feat_2025aso.parquet", "feat_2025novdec.parquet")], how="diagonal_relaxed")
    F = F.unique("MVT_ID_mvt", keep="first")
    D = cv.join(raw.drop("ADEP_mvt"), on="MVT_ID_mvt", how="left").join(F, on="MVT_ID_mvt", how="left")
    D = rel(D).with_columns((pl.col("p") - pl.col("y")).alias("e"),
                            (pl.col("MVT_TIME_UTC_mvt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y_chk"))
    print("train", D.height, "y mismatch", int((D["y"] != D["y_chk"]).sum()), "raw miss", int(D["MVT_TIME_UTC_mvt"].is_null().sum()))
    D.drop("y_chk").write_parquet(os.path.join(OUT, "bigerr_base.parquet"))

    rk = pl.read_parquet("data/raw/ranking.parquet", columns=[c for c in RAW if c != "BLOCK_TIME_UTC_mvt"]).filter(
        pl.col("PHASE_mvt") == "DEP")
    F26 = pl.read_parquet(os.path.join(S, "learn", "feat_2026.parquet"), columns=ADS).unique("MVT_ID_mvt", keep="first")
    v40 = pl.read_parquet("/home/user/prc-data-challenge/outputs/v40_v39_7m_pos_12m.parquet")
    print("v40 cols", v40.columns, v40.height)
    R = rk.join(F26, on="MVT_ID_mvt", how="left").join(v40.rename({"TAXITIME_SEC_mvt": "p"}), on="MVT_ID_mvt", how="inner")
    R = rel(R).with_columns(pl.col("t_mvt").is_not_null().alias("has"))
    print("rank", R.height)
    R.write_parquet(os.path.join(OUT, "bigerr_rank.parquet"))


if __name__ == "__main__":
    main()

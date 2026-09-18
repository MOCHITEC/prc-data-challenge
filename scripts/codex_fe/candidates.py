"""Label-free candidates built outside the production feature cache.

Every source survives on ranking departures; arrival labels are used only after
that arrival's on-block time. Candidate links are not asserted to be real rotations.
"""

import os

os.environ["POLARS_MAX_THREADS"] = "2"
import json
from pathlib import Path

import numpy as np
import polars as pl

ID = "MVT_ID_mvt"
T = "MVT_TIME_UTC_mvt"
DESCRIPTIONS = {}


def build(f, arrivals):
    exprs = []

    def add(name, expr, why):
        DESCRIPTIONS[name] = why
        exprs.append(expr.alias(name))

    def sec(a, b):
        return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)

    for name, a, b in [
        ("arrival_revision", "ARVT_3_flt", "ARVT_1_flt"),
        ("actual_remaining", "ARVT_3_flt", T),
        ("filed_remaining", "ARVT_1_flt", T),
        ("iobt_eobt_revision", "IOBT_flt", "EOBT_1_flt"),
        ("aobt_iobt_revision", "AOBT_3_flt", "IOBT_flt"),
        ("aobt_lobt_revision", "AOBT_3_flt", "LOBT_flt"),
        ("iobt_schedule_revision", "IOBT_flt", "SCHED_TIME_UTC_mvt"),
        ("eobt_schedule_revision", "EOBT_1_flt", "SCHED_TIME_UTC_mvt"),
        ("lobt_iobt_revision", "LOBT_flt", "IOBT_flt"),
    ]:
        add(name, sec(a, b), f"Seconds {a} minus {b}; clock update/consistency information.")
    add(
        "duration_revision",
        sec("ARVT_3_flt", "AOBT_3_flt") - sec("ARVT_1_flt", "EOBT_1_flt"),
        "Change in end-to-end duration, separating off-block and arrival revisions.",
    )
    for short, col in [("mvt", T), ("schedule", "SCHED_TIME_UTC_mvt"), ("aobt", "AOBT_3_flt"), ("arvt", "ARVT_3_flt")]:
        add(short + "_second", pl.col(col).dt.second(), "Timestamp precision/reporting convention.")
        add(short + "_minute5", pl.col(col).dt.minute() % 5, "Five-minute timestamp rounding/reporting convention.")
    for short, col in [("schedule", "SCHED_TIME_UTC_mvt"), ("aobt", "AOBT_3_flt")]:
        add(short + "_hour", pl.col(col).dt.hour(), "Off-block daily phase distinct from take-off hour.")
        add(
            short + "_day_gap",
            (pl.col(T).dt.date() - pl.col(col).dt.date()).dt.total_days(),
            "Known clock date disagreement; not the hidden block-date rollover.",
        )
    add(
        "offblock_clock_spread",
        pl.max_horizontal([sec(c, "SCHED_TIME_UTC_mvt") for c in ["LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]])
        - pl.min_horizontal(
            [sec(c, "SCHED_TIME_UTC_mvt") for c in ["LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]]
        ),
        "Range of NM off-block clock revisions.",
    )
    add(
        "offblock_unique",
        pl.concat_list(["LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]).list.n_unique(),
        "Number of distinct NM off-block clocks.",
    )
    add("flight_rule", pl.col("FLIGHT_RULE_mvt"), "Operational flight rule absent from baseline categoricals.")
    add(
        "flight_prefix",
        pl.col("FLIGHT_mvt").str.extract(r"^([A-Za-z]+)", 1),
        "Flight designator prefix on unmatched as well as matched rows.",
    )
    add(
        "callsign_suffix",
        pl.col("CALLSIGN_flt").str.extract(r"([A-Za-z]+)$", 1),
        "Callsign suffix reporting/flight pattern.",
    )
    add(
        "flight_number_band",
        pl.col("FLIGHT_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64, strict=False) // 100,
        "Coarse flight-number service family.",
    )
    add(
        "stand_prefix",
        pl.col("STAND_mvt").str.extract(r"^([^0-9]*)", 1),
        "Stand region sharing information across individual stands.",
    )
    add(
        "stand_number",
        pl.col("STAND_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64, strict=False),
        "Ordinal stand location proxy.",
    )
    add("stand_suffix", pl.col("STAND_mvt").str.extract(r"([A-Za-z]+)$", 1), "Stand subdivision/remote-stand coding.")
    add(
        "runway_heading",
        pl.col("RUNWAY_mvt").str.extract(r"(\d+)", 1).cast(pl.Float64, strict=False),
        "Share information across runway directions.",
    )
    add(
        "runway_side",
        pl.col("RUNWAY_mvt").str.extract(r"([A-Za-z]+)$", 1),
        "Parallel runway side / helipad distinction.",
    )
    add("same_airport", pl.col("ADEP_mvt") == pl.col("ADES_mvt"), "Local flight indicator.")
    add(
        "dest_region_mismatch",
        pl.col("ADES_mvt").str.slice(0, 2) != pl.col("ADES_FILED_flt").str.slice(0, 2),
        "Large filed/destination mismatch.",
    )
    for c in ["temp_c", "wind_ms", "gust_ms", "vis_km", "precip_mm", "dewp_spread_c"]:
        add(c + "_trend", pl.col(c) - pl.col(c + "_prev1h"), "One-hour weather change rather than level.")
    add("gust_excess", pl.col("gust_ms") - pl.col("wind_ms"), "Gustiness above mean wind.")
    for w in [5, 15, 30]:
        add(
            f"dep_acceleration_{w}",
            pl.col(f"dep_tko_prev_{w}m") * 60 / w - pl.col("dep_tko_prev_60m"),
            "Recent departure rate above trailing hourly rate.",
        )
        add(
            f"arr_acceleration_{w}",
            pl.col(f"arr_land_prev_{w}m") * 60 / w - pl.col("arr_land_prev_60m"),
            "Recent arrival rate above trailing hourly rate.",
        )
    add(
        "proxy_neighbor_deviation",
        pl.col("mvt_minus_aobt3") - pl.col("nbr_mvt_minus_aobt_mean_30m"),
        "Own proxy relative to recent airport reporting state.",
    )
    add(
        "delay_neighbor_deviation",
        pl.col("aobt3_minus_eobt1") - pl.col("nbr_aobt_delay_mean_30m"),
        "Own off-block delay relative to recent airport state.",
    )
    add(
        "runway_headway_asymmetry",
        pl.col("rwy_headway_prev_s") - pl.col("rwy_headway_next_s"),
        "Local runway service interruption asymmetry.",
    )
    out = f.select([ID, *exprs])
    # Most recent completed arrival on the same stand, strictly before the departure take-off.
    # No flight-ID identity assumption, no departure block/label ever passed here.
    a = arrivals.select(
        pl.col("ADES_mvt").alias("ADEP_mvt"),
        "STAND_mvt",
        pl.col("BLOCK_TIME_UTC_mvt").alias("_arr_block"),
        pl.col("TAXITIME_SEC_mvt").alias("_arr_taxi"),
        pl.col("AIRCRAFT_TYPE_mvt").alias("_arr_type"),
        pl.col("SCHED_TIME_UTC_mvt").alias("_arr_sched"),
        pl.col(T).alias("_arr_land"),
    )
    a = a.drop_nulls(["ADEP_mvt", "STAND_mvt", "_arr_block"]).sort("_arr_block")
    d = f.select(ID, "ADEP_mvt", "STAND_mvt", T, "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "AOBT_3_flt").sort(T)
    linked = d.join_asof(
        a,
        left_on=T,
        right_on="_arr_block",
        by=["ADEP_mvt", "STAND_mvt"],
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    linked = linked.with_columns((pl.col(T) - pl.col("_arr_block")).dt.total_seconds().alias("_gap"))
    valid = pl.col("_gap") <= 21600
    specs = {
        "stand_arrival_gap": pl.col("_gap"),
        "stand_arrival_taxi": pl.col("_arr_taxi"),
        "stand_arrival_type_match": (pl.col("AIRCRAFT_TYPE_mvt") == pl.col("_arr_type")).cast(pl.Int8),
        "stand_arrival_delay": (pl.col("_arr_land") - pl.col("_arr_sched")).dt.total_seconds(),
        "stand_sched_turn": (pl.col("SCHED_TIME_UTC_mvt") - pl.col("_arr_block")).dt.total_seconds(),
        "stand_aobt_turn": (pl.col("AOBT_3_flt") - pl.col("_arr_block")).dt.total_seconds(),
    }
    for name in specs:
        DESCRIPTIONS[name] = "Most recent completed same-stand arrival within six hours: " + name
    out = out.join(
        linked.select(ID, *[pl.when(valid).then(e).otherwise(None).alias(n) for n, e in specs.items()]),
        on=ID,
        maintain_order="left",
    )
    # Reporting state specific to a runway or operator, with own row excluded by strict time lag.
    for group, keys in [("runway", ["ADEP_mvt", "RUNWAY_mvt"]), ("operator", ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"])]:
        pieces = []
        for sub in f.select(ID, T, "mvt_minus_aobt3", *keys).partition_by(keys):
            ts = sub[T].dt.epoch("s").to_numpy().astype(float)
            v = sub["mvt_minus_aobt3"].to_numpy()
            order = np.argsort(ts)
            st = ts[order]
            sv = v[order]
            finite = np.isfinite(sv)
            st = st[finite]
            sv = sv[finite]
            lo = np.searchsorted(st, ts - 3600)
            hi = np.searchsorted(st, ts - 60)
            n = hi - lo
            s = np.r_[0, np.cumsum(sv)]
            s2 = np.r_[0, np.cumsum(sv * sv)]
            mean = np.divide(s[hi] - s[lo], n, out=np.full(len(ts), np.nan), where=n > 0)
            var = np.divide(s2[hi] - s2[lo], n, out=np.full(len(ts), np.nan), where=n > 0) - mean * mean
            pieces.append(
                pl.DataFrame(
                    {
                        ID: sub[ID],
                        f"{group}_proxy_mean": mean,
                        f"{group}_proxy_sd": np.sqrt(np.maximum(var, 0)),
                        f"{group}_proxy_n": n,
                    }
                )
            )
        out = out.join(pl.concat(pieces), on=ID, maintain_order="left")
        for suffix in ["mean", "sd", "n"]:
            DESCRIPTIONS[f"{group}_proxy_{suffix}"] = (
                f"Previous 60 minutes (lag 60 seconds), same airport and {group}, known MVT-AOBT proxy {suffix}."
            )
    for seed in range(4):
        name = f"noise_{seed}"
        out = out.with_columns((pl.col(ID).hash(seed=20260918 + seed) % 1000003).cast(pl.Float64).alias(name))
        DESCRIPTIONS[name] = "Deterministic hash negative control."
    assert out.height == f.height and out[ID].equals(f[ID])
    return out


if __name__ == "__main__":
    for split in ["train", "rank"]:
        f = pl.read_parquet(f"data/work/feat_{split}.parquet")
        paths = (
            sorted(Path("data/raw").glob("training_*.parquet"))
            if split == "train"
            else [Path("data/raw/ranking.parquet")]
        )
        a = (
            pl.scan_parquet(paths)
            .filter(pl.col("PHASE_mvt") == "ARR")
            .select(
                "ADES_mvt",
                "STAND_mvt",
                "BLOCK_TIME_UTC_mvt",
                "TAXITIME_SEC_mvt",
                "AIRCRAFT_TYPE_mvt",
                "SCHED_TIME_UTC_mvt",
                T,
            )
            .collect()
        )
        out = build(f, a)
        out.write_parquet(f"outputs/codex_fe/candidates_{split}.parquet")
        print(split, out.shape, flush=True)
        del f, a, out
    Path("docs/codex_fe_results/candidates.json").write_text(json.dumps(DESCRIPTIONS, indent=2))

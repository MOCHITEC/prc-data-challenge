"""Flight-number *format* as a reporting fingerprint (report §5)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from reporting_fingerprint_common import *  # noqa
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(220)


def fmt_expr() -> pl.Expr:
    f = pl.col("FLIGHT_mvt")
    p1, p2, p3 = f.str.slice(0, 1), f.str.slice(1, 1), f.str.slice(2, 1)
    rest = f.str.slice(3)
    dup = (f.str.contains(r"^[A-Z]{3}[A-Z]") &
           (rest.str.starts_with(p1 + p2 + p3) | rest.str.starts_with(p2 + p3) | rest.str.starts_with(p3)))
    return (pl.when(f.is_null()).then(pl.lit("null"))
            .when(dup).then(pl.lit("dup_prefix"))
            .when(f.str.contains(r"^[A-Z]{3}\d")).then(pl.lit("icao"))
            .when(f.str.contains(r"^[A-Z0-9]{2}\d{1,4}[A-Z]?$")).then(pl.lit("iata"))
            .when(f.str.contains(r"^[A-Z]{3}[A-Z]")).then(pl.lit("icao_alpha"))
            .otherwise(pl.lit("other")).alias("fl_fmt"))


if __name__ == "__main__":
    df = load_all().with_columns(fmt_expr())
    print("format share by airport x phase x split")
    t = df.group_by("apt", "PHASE_mvt", "split", "fl_fmt").len().with_columns((pl.col("len") / pl.col("len").sum().over("apt", "PHASE_mvt", "split")).round(4).alias("share"))
    print(t.pivot(on="fl_fmt", index=["apt", "PHASE_mvt", "split"], values="share").sort("apt", "PHASE_mvt", "split"))
    dep = df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("split") == "train"))
    print("DEP train: clock source and tail by format x NM match")
    print(dep.group_by("fl_fmt", pl.col("FLIGHT_ID_mvt").is_null().alias("nm_null")).agg(
        pl.len(), (pl.col("clock_src") == "sched").mean().round(3).alias("sched"), (pl.col("clock_src") == "aobt3").mean().round(3).alias("aobt3"),
        pl.col("TAXITIME_SEC_mvt").mean().round(0).alias("ymean"), pl.col("TAXITIME_SEC_mvt").std().round(0).alias("ysd"),
        (pl.col("TAXITIME_SEC_mvt") > 3600).sum().alias("n_gt1h"), (pl.col("TAXITIME_SEC_mvt") > 3600 * 3).sum().alias("n_gt3h")).sort("fl_fmt", "nm_null"))
    print("LIRF only:")
    print(dep.filter(pl.col("apt") == "LIRF").group_by("fl_fmt", pl.col("FLIGHT_ID_mvt").is_null().alias("nm_null")).agg(
        pl.len(), (pl.col("clock_src") == "sched").mean().round(3).alias("sched"), pl.col("TAXITIME_SEC_mvt").mean().round(0).alias("ymean"),
        pl.col("TAXITIME_SEC_mvt").std().round(0).alias("ysd"), (pl.col("TAXITIME_SEC_mvt") > 3600).sum().alias("n_gt1h"),
        (pl.col("TAXITIME_SEC_mvt") > 3600 * 3).sum().alias("n_gt3h")).sort("fl_fmt", "nm_null"))
    print(dep.filter((pl.col("apt") == "LIRF") & (pl.col("fl_fmt") == "dup_prefix")).select("FLIGHT_mvt", "CALLSIGN_flt", "STAND_mvt", "clock_src", "TAXITIME_SEC_mvt").head(15))
    print("rank DEP rows by format (LIRF):", df.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("split") == "rank") & (pl.col("apt") == "LIRF")).group_by("fl_fmt", pl.col("FLIGHT_ID_mvt").is_null().alias("nm_null")).len().sort("fl_fmt"))

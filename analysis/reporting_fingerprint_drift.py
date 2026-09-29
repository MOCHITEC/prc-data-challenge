"""Month-by-month stability of reporting practices, and the Jan/Jul 2026 ranking-side ARR fingerprint (report §7)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from reporting_fingerprint_common import *  # noqa
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)

df = load_all()


def excess(ref: str, base: str = "BLOCK_TIME_UTC_mvt") -> list[pl.Expr]:
    d = (pl.col(base) - pl.col(ref)).dt.total_seconds()
    k0 = (d.abs() <= 6).cast(pl.Float64)
    k1 = ((d - 60).abs() <= 6).cast(pl.Float64) + ((d + 60).abs() <= 6).cast(pl.Float64)
    return [k0.mean(), (k0.mean() - k1.mean() / 2)]


rows = []
for ph, refs in [("ARR", {"sched": "SCHED_TIME_UTC_mvt"}), ("DEP", {"sched": "SCHED_TIME_UTC_mvt", "aobt3": "AOBT_3_flt", "eobt1": "EOBT_1_flt"})]:
    d = df.filter(pl.col("PHASE_mvt") == ph)
    for name, ref in refs.items():
        e = excess(ref)
        g = d.group_by("apt", "year", "month").agg(e[1].round(4).alias("copy_excess"), pl.len().alias("n"))
        g = g.with_columns(pl.lit(f"{ph}_{name}").alias("what"))
        rows.append(g)
t = pl.concat(rows)
t = t.with_columns((pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Utf8).str.zfill(2)).alias("ym"))
wide = t.filter(pl.col("n") > 0).pivot(on="ym", index=["what", "apt"], values="copy_excess", sort_columns=True).sort("what", "apt")
print("copy excess (k0 - mean(k+-1)) by month; 2026-01/2026-07 = ranking (only ARR has BLOCK there)")
print(wide.filter(pl.col("apt").is_in(["LIRF", "LEBL", "LEMD", "EGLL", "LFPG"]) | (pl.col("what") == "ARR_sched")))
wide.write_csv(CACHE / "drift_copy_excess.csv")
# NM-unmatched rate and malformed-flight share by month
from reporting_fingerprint_flightfmt import fmt_expr
dep = df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(fmt_expr())
print(dep.group_by("apt", "year", "month").agg(pl.col("FLIGHT_ID_mvt").is_null().mean().round(4).alias("nm_null"),
        (pl.col("fl_fmt") == "dup_prefix").mean().round(4).alias("dup")).filter(pl.col("apt").is_in(["LIRF", "EHAM", "LSZH"])).sort("apt", "year", "month")
      .pivot(on="apt", index=["year", "month"], values=["nm_null", "dup"]))

# --- LIRF: predict the ranking-period DEP sched-copy rate from the ARR-side fingerprint (daily linear fit on 2025)
import numpy as np  # noqa: E402
l = df.filter(pl.col("apt") == "LIRF").with_columns(
    ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 6).cast(pl.Float64).alias("c"))
arr_d = l.filter(pl.col("PHASE_mvt") == "ARR").group_by("split", "date", "month").agg(pl.col("c").mean().alias("arr"))
dep_d = l.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("split") == "train")).group_by("date").agg(pl.col("c").mean().alias("dep"))
j = arr_d.filter(pl.col("split") == "train").join(dep_d, on="date")
b, a0 = np.polyfit(j["arr"].to_numpy(), j["dep"].to_numpy(), 1)
print(f"LIRF daily fit: dep_copy = {a0:.4f} + {b:.3f} * arr_copy   (r = {np.corrcoef(j['arr'], j['dep'])[0, 1]:.3f})")
m = arr_d.group_by("split", "month").agg(pl.col("arr").mean()).sort("split", "month").with_columns((a0 + b * pl.col("arr")).round(4).alias("dep_pred"))
m = m.join(dep_d.join(arr_d.select("date", "month"), on="date").group_by("month").agg(pl.col("dep").mean().round(4).alias("dep_obs_2025")), on="month", how="left")
print(m.filter(pl.col("month").is_in([1, 7])))
# NM-unmatched and malformed-flight share by month, training vs ranking
from reporting_fingerprint_flightfmt import fmt_expr  # noqa: E402
dep = df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(fmt_expr())
q = dep.group_by("apt", "year", "month").agg(pl.col("FLIGHT_ID_mvt").is_null().mean().round(4).alias("nm_null"),
        (pl.col("fl_fmt") == "dup_prefix").mean().round(4).alias("dup_fmt"),
        (pl.col("FLIGHT_ID_mvt").is_null() & pl.col("fl_fmt").is_in(["dup_prefix", "null"])).sum().alias("n_unm_badfmt"))
print(q.filter(pl.col("month").is_in([1, 7])).sort("apt", "month", "year"))

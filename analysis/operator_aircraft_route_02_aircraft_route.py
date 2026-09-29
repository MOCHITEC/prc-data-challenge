"""operator_aircraft_route 02: aircraft type / WTC / widebody / runway choice / destination distance / market."""

from __future__ import annotations

import math
import os

import numpy as np
import polars as pl
from operator_aircraft_route_common import ROOT, is_widebody, load_dep, md

HELI = r"^(AS50|AS55|AS65|A109|A119|A139|A169|A189|EC20|EC25|EC30|EC35|EC45|EC55|EC75|H125|H130|H135|H145|H160|H175|B06|B407|B412|B429|S76|S92|R44|R22|AW09|MI8|BK17)"


def airports() -> pl.DataFrame | None:
    p = os.path.join(ROOT, "data", "external", "ourairports_airports.csv")
    if not os.path.exists(p):
        return None
    a = pl.read_csv(p, columns=["ident", "gps_code", "icao_code", "latitude_deg", "longitude_deg", "iso_country",
                                "continent"], infer_schema_length=0)
    a = a.with_columns(pl.coalesce(pl.col("icao_code").replace("", None), pl.col("gps_code").replace("", None),
                                   pl.col("ident")).alias("icao"),
                       pl.col("latitude_deg").cast(pl.Float64), pl.col("longitude_deg").cast(pl.Float64))
    return a.select("icao", "latitude_deg", "longitude_deg", "iso_country", "continent").unique("icao")


def gc_km(lat1, lon1, lat2, lon2):
    r = math.pi / 180
    dlat = (lat2 - lat1) * r
    dlon = (lon2 - lon1) * r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def with_distance(d: pl.DataFrame) -> pl.DataFrame:
    a = airports()
    if a is None:
        return d.with_columns(pl.lit(None, pl.Float64).alias("dist_km"))
    o = a.rename({"icao": "ADEP_mvt", "latitude_deg": "lat0", "longitude_deg": "lon0"}).select("ADEP_mvt", "lat0", "lon0")
    x = a.rename({"icao": "ADES_mvt", "latitude_deg": "lat1", "longitude_deg": "lon1", "continent": "ades_continent"}).select(
        "ADES_mvt", "lat1", "lon1", "ades_continent")
    d = d.join(o, on="ADEP_mvt", how="left").join(x, on="ADES_mvt", how="left")
    km = gc_km(d["lat0"].to_numpy(), d["lon0"].to_numpy(), d["lat1"].to_numpy(), d["lon1"].to_numpy())
    return d.with_columns(pl.Series("dist_km", km)).drop("lat0", "lon0", "lat1", "lon1")


def main() -> None:
    d = load_dep(["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
                  "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "FLIGHT_RULE_mvt", "split", "y",
                  "clock_src", "airline", "FLIGHT_ID_mvt", "SCHED_TIME_UTC_mvt"])
    d = with_distance(d).with_columns(
        is_widebody().alias("wide"),
        pl.col("AIRCRAFT_TYPE_mvt").str.contains(HELI).fill_null(False).alias("heli"),
        (pl.col("SCHED_TIME_UTC_mvt").dt.hour() // 3).alias("hb"),
    )
    os.makedirs(os.path.join(ROOT, "data", "interim"), exist_ok=True)
    d.select("MVT_ID_mvt", "dist_km", "wide", "heli", "ades_continent").write_parquet(
        os.path.join(ROOT, "data", "interim", "oar_route.parquet"))
    tr = d.filter(pl.col("split") == "train")
    ref = tr.group_by("ADEP_mvt", "STAND_mvt").agg(pl.col("y").clip(0, 3600).median().alias("ref"))
    tr = tr.join(ref, on=["ADEP_mvt", "STAND_mvt"], how="left").with_columns((pl.col("y").clip(0, 3600) - pl.col("ref")).alias("r"))

    print("## A. 距離（大圏, OurAirports）の被覆\n")
    print(md(d.group_by("split").agg(pl.len(), pl.col("dist_km").is_not_null().mean().alias("dist_cov")).sort("split"), "{:.4f}"))

    print("\n## B. 目的地距離 × y（学習）。r = clip(y,0,3600) − 空港×スタンド中央値\n")
    b = tr.with_columns(pl.col("dist_km").cut([500, 1000, 2000, 3500, 6000, 9000]).alias("dist")).group_by("dist").agg(
        pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("r").mean().alias("r_mean"),
        pl.col("r").median().alias("r_med"), pl.col("wide").mean().mul(100).alias("wide%"),
        (pl.col("clock_src") == "sched").mean().mul(100).alias("sched%"),
        (pl.col("clock_src") == "aobt3").mean().mul(100).alias("aobt3%")).sort("dist")
    print(md(b, "{:,.1f}"))

    print("\n## C. ワイドボディ vs ナローボディ（学習, 空港別）: 同一スタンド内の差 r\n")
    c = tr.filter(~pl.col("heli")).group_by("ADEP_mvt", "wide").agg(pl.len().alias("n"), pl.col("y").median().alias("y_med"),
                                                                     pl.col("r").mean().alias("r_mean")).sort("ADEP_mvt", "wide")
    print(md(c, "{:,.1f}"))

    print("\n## D. 滑走路の選択: 重量機（WTC H/J）とそれ以外の滑走路構成比（学習, 上位滑走路）\n")
    w = tr.with_columns(pl.col("WK_TBL_CAT_flt").is_in(["H", "J"]).alias("heavy"))
    rs = w.group_by("ADEP_mvt", "RUNWAY_mvt", "heavy").len().with_columns(
        (pl.col("len") / pl.col("len").sum().over("ADEP_mvt", "heavy")).alias("share"))
    piv = rs.pivot(on="heavy", index=["ADEP_mvt", "RUNWAY_mvt"], values="share").rename({"true": "share_heavy", "false": "share_other"})
    piv = piv.with_columns((pl.col("share_heavy").fill_null(0) - pl.col("share_other").fill_null(0)).alias("diff")).filter(
        (pl.col("share_heavy").fill_null(0) > 0.03) | (pl.col("share_other").fill_null(0) > 0.03)).sort("ADEP_mvt", "RUNWAY_mvt")
    print(md(piv, "{:.3f}"))

    print("\n## E. 機種の不一致（AIRCRAFT_TYPE_mvt ≠ _flt）\n")
    e = tr.filter(pl.col("FLIGHT_ID_mvt").is_not_null()).with_columns(
        (pl.col("AIRCRAFT_TYPE_mvt") != pl.col("AIRCRAFT_TYPE_flt")).fill_null(True).alias("mismatch")).group_by("mismatch").agg(
        pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("r").mean().alias("r_mean"),
        (pl.col("clock_src") == "sched").mean().mul(100).alias("sched%"))
    print(md(e, "{:,.1f}"))

    print("\n## F. 市場区分・ヘリ（学習）\n")
    f = tr.group_by("MARKET_SEGMENT_flt").agg(pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("y").std().alias("y_sd"),
                                            pl.col("r").mean().alias("r_mean"),
                                            (pl.col("clock_src") == "sched").mean().mul(100).alias("sched%"),
                                            (pl.col("clock_src") == "aobt3").mean().mul(100).alias("aobt3%"),
                                            (pl.col("y") > 3600).mean().mul(1e4).alias("gt1h_per10k")).sort("n", descending=True)
    print(md(f, "{:,.1f}"))
    h = d.group_by("split", "heli").agg(pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("y").mean().alias("y_mean"),
                                        pl.col("FLIGHT_ID_mvt").is_null().mean().alias("nm_unmatched"),
                                        pl.col("ADEP_mvt").mode().first().alias("main_apt")).sort("split", "heli")
    print()
    print(md(h, "{:,.3f}"))

    print("\n## G. LCC のスタンド: 航空会社別に見た y の水準は空港×スタンドで説明済みか（学習, 市場区分別の r）\n")
    g = tr.group_by("ADEP_mvt", "MARKET_SEGMENT_flt").agg(pl.len().alias("n"), pl.col("y").median().alias("y_med"),
                                                        pl.col("r").mean().alias("r_mean")).filter(pl.col("n") >= 3000).sort(
        "ADEP_mvt", "n", descending=[False, True])
    print(md(g, "{:,.1f}"))


if __name__ == "__main__":
    main()

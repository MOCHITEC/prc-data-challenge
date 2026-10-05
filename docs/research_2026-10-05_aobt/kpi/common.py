"""Shared: load 2025 departures, local month, PRU-style filters, suspect-AOBT flags, ATXOT KPI engine."""
import polars as pl, numpy as np
TZ = {"EDDF":"Europe/Berlin","EDDM":"Europe/Berlin","EGLL":"Europe/London","EHAM":"Europe/Amsterdam","LEBL":"Europe/Madrid",
      "LEMD":"Europe/Madrid","LFPG":"Europe/Paris","LIRF":"Europe/Rome","LSZH":"Europe/Zurich","LTFM":"Europe/Istanbul"}
APTS = sorted(TZ)
MINUTE_GRID_BLOCK = {"EDDM","EGLL","LFPG","LIRF","LTFM","LEBL","LEMD"}   # BLOCK within +-6 s of :00 on ~100% of rows (verified in 03_flags)
HELI = set("""A109 A119 A129 A139 A149 A169 A189 AS32 AS3B AS50 AS55 AS65 AS35 B06 B06T B105 B212 B222 B230 B407 B412 B427 B429 B430 B505
BK17 EC20 EC25 EC30 EC35 EC45 EC55 EC75 EH10 H120 H125 H130 H135 H145 H155 H160 H175 H215 H225 MI8 MI17 NH90 R22 R44 R66 S61 S76 S92
UH1 EXPL MD52 MD60 H500 G2CA LYNX PUMA SUCO GAZL ALO2 ALO3 S330 HELI""".split())
TOL_EXACT = 6     # s; +-6 s = anonymisation jitter on minute-grid airports (so 'exact copy' needs 6, not 5)
TOL_LOOSE = 60
PLAN_SRC = {"SCHED":"SCHED_TIME_UTC_mvt","EOBT1":"EOBT_1_flt","LOBT":"LOBT_flt","IOBT":"IOBT_flt"}

def load():
    df = pl.read_parquet("dep2025.parquet")
    parts=[]
    for a,tz in TZ.items():
        s = df.filter(pl.col("ADEP_mvt")==a).with_columns(
            lt=pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone(tz))
        s = s.with_columns(lmon=pl.col("lt").dt.month().cast(pl.Int32), lyr=pl.col("lt").dt.year().cast(pl.Int32),
                           lhour_utc=pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").dt.replace_time_zone(None)).drop("lt")
        parts.append(s)
    df = pl.concat(parts).filter(pl.col("lyr")==2025)
    B = pl.col("BLOCK_TIME_UTC_mvt"); y = pl.col("TAXITIME_SEC_mvt")
    def d(c): return (B-pl.col(c)).dt.total_seconds()
    ex = [ (pl.col("AIRCRAFT_TYPE_mvt").is_in(list(HELI)) | pl.col("AIRCRAFT_TYPE_flt").is_in(list(HELI))).fill_null(False).alias("heli"),
           d("AOBT_3_flt").alias("dA3")]
    for k,c in PLAN_SRC.items(): ex.append(d(c).alias("d"+k))
    df = df.with_columns(ex)
    fl = [ (pl.col("dA3").abs()<=5).fill_null(False).alias("F_A3_5"),
           (pl.col("dA3").abs()<=TOL_LOOSE).fill_null(False).alias("F_A3_60")]
    for k in PLAN_SRC:
        fl += [(pl.col("d"+k).abs()<=TOL_EXACT).fill_null(False).alias(f"F_{k}_6"),
               (pl.col("d"+k).abs()<=TOL_LOOSE).fill_null(False).alias(f"F_{k}_60")]
    df = df.with_columns(fl)
    df = df.with_columns(
        F_PLAN_6 = pl.any_horizontal([pl.col(f"F_{k}_6") for k in PLAN_SRC]),
        F_PLAN_60 = pl.any_horizontal([pl.col(f"F_{k}_60") for k in PLAN_SRC]),
        F_ROLL = y>72000, F_NEG = y<0, F_SHORT = y<60, F_LONG = y>7200,
    )
    # FILL: BLOCK == plan time (+-6 s) while NM AOBT_3 says the aircraft left > 5 min LATER (or no NM record):
    # the airport wrote the planned time instead of a measured AOBT for a delayed flight -> taxi-out inflated.
    # (the symmetric 'NM earlier' case is kept separately: at LTFM AOBT_3 is itself synthetic, so it is not used.)
    df = df.with_columns(
        F_SCHEDFILL = pl.col("F_SCHED_6") & ((pl.col("dA3")< -300) | pl.col("dA3").is_null()),
        F_PLANFILL = pl.col("F_PLAN_6") & ((pl.col("dA3")< -300) | pl.col("dA3").is_null()),
        F_PLAN_NMEARLY = pl.col("F_PLAN_6") & (pl.col("dA3")> 300),
        unmatched = pl.col("FLIGHT_ID_mvt").is_null(),
    )
    return df

def deice_proxy():
    m = pl.read_parquet("/home/user/prc-data-challenge/data/ref/metar_hourly.parquet")
    # de-icing likely: frozen precip at <=3C, or frost conditions (T<=1C, dew-point spread <=2C)
    m = m.with_columns(di=((pl.col("frozen_wx") & (pl.col("temp_c")<=3)) | ((pl.col("temp_c")<=1) & (pl.col("dewp_spread_c")<=2))).fill_null(False))
    # spread over the following 2 hours (aircraft de-iced after the weather hour)
    m = m.sort(["station","hour_utc"]).with_columns(di=pl.col("di").cast(pl.Int8).rolling_max(3).over("station").cast(pl.Boolean))
    return m.select(pl.col("station").alias("ADEP_mvt"), pl.col("hour_utc").alias("lhour_utc"), "di")

def base_filter(df, deice=False):
    q = df.filter(pl.col("STAND_mvt").is_not_null() & pl.col("RUNWAY_mvt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()
                  & pl.col("TAXITIME_SEC_mvt").is_not_null() & (pl.col("TAXITIME_SEC_mvt")<=7200) & ~pl.col("heli"))
    if deice:
        q = q.join(deice_proxy(), on=["ADEP_mvt","lhour_utc"], how="left").filter(~pl.col("di").fill_null(False)).drop("di")
    return q

def kpi(df, ref_mask=None, meas_mask=None, window="FY", tcol="TAXITIME_SEC_mvt"):
    """df: filtered flights. ref_mask/meas_mask: polars expr; rows where True are KEPT for reference / measurement.
    window FY: one reference from all 2025 (= PRU window for Dec-2025); ROLL: Jan..month i (incremental first year, PRU 6.1).
    Returns per airport-month: n_meas, n_ref_flights (with valid ref), sum_add_min, sum_ref_min."""
    d = df.with_columns(t=pl.col(tcol).cast(pl.Float64))
    R = d if ref_mask is None else d.filter(ref_mask)
    M = d if meas_mask is None else d.filter(meas_mask)
    keys=["ADEP_mvt","STAND_mvt","RUNWAY_mvt"]
    def refs(sub):
        p = sub.with_columns(p10=pl.col("t").quantile(0.1, "linear").over(keys)).group_by(keys).agg(
                p10=pl.col("p10").first(), nle=(pl.col("t")<=pl.col("p10")).sum())
        return p.filter(pl.col("nle")>=10).select(keys+["p10"])
    out=[]
    if window=="FY":
        r = refs(R)
        mm = M.join(r, on=keys, how="left")
    else:
        mm_parts=[]
        for i in range(1,13):
            r = refs(R.filter(pl.col("lmon")<=i))
            mm_parts.append(M.filter(pl.col("lmon")==i).join(r, on=keys, how="left"))
        mm = pl.concat(mm_parts)
    res = mm.group_by(["ADEP_mvt","lmon"]).agg(
        n=pl.len(), n_ref=pl.col("p10").is_not_null().sum(),
        add_min=((pl.col("t")-pl.col("p10"))/60).sum(), ref_min=(pl.col("p10")/60).sum(),
        txo_min=(pl.when(pl.col("p10").is_not_null()).then(pl.col("t")/60)).sum()
    ).with_columns(add=pl.col("add_min")/pl.col("n_ref"), ref=pl.col("ref_min")/pl.col("n_ref")).sort(["ADEP_mvt","lmon"])
    return res

def annual(res):
    return res.group_by("ADEP_mvt").agg(n_ref=pl.col("n_ref").sum(), add=pl.col("add_min").sum()/pl.col("n_ref").sum(),
                                        ref=pl.col("ref_min").sum()/pl.col("n_ref").sum()).sort("ADEP_mvt")

"""Local-date (Europe/Rome etc.) mechanism check for day-shift rows. Training 2025 DEP rows + ranking DEP rows."""
import glob, polars as pl
TZ = {"LIRF":"Europe/Rome","LSZH":"Europe/Zurich","LFPG":"Europe/Paris","EDDF":"Europe/Berlin","EDDM":"Europe/Berlin",
      "EHAM":"Europe/Amsterdam","EGLL":"Europe/London","LEBL":"Europe/Madrid","LEMD":"Europe/Madrid","LTFM":"Europe/Istanbul"}
cols=["MVT_ID_mvt","FLIGHT_ID_mvt","FLIGHT_mvt","ADEP_mvt","PHASE_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","TAXITIME_SEC_mvt","AOBT_3_flt","LOBT_flt","AIRCRAFT_TYPE_mvt"]
files=sorted(glob.glob("/home/user/prc-data-challenge/data/raw/training_*.parquet"))
def prep(lf):
    lf=lf.select(cols).filter(pl.col("PHASE_mvt")=="DEP")
    parts=[]
    for ap,tz in TZ.items():
        p=lf.filter(pl.col("ADEP_mvt")==ap).with_columns(
            [pl.col(c).dt.convert_time_zone(tz).dt.date().alias(c[:4]+"_ld") for c in ["MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt"]]
            +[pl.col(c).dt.date().alias(c[:4]+"_ud") for c in ["MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt"]]
            +[(pl.col("MVT_TIME_UTC_mvt")-pl.duration(seconds=900)).dt.convert_time_zone(tz).dt.date().alias("OFFp_ld")])
        parts.append(p)
    return pl.concat(parts).with_columns(
        ms=(pl.col("MVT_TIME_UTC_mvt")-pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
        unm=pl.col("AOBT_3_flt").is_null(),
        dd_loc=(pl.col("MVT__ld")-pl.col("SCHE_ld")).dt.total_days(),
        dd_utc=(pl.col("MVT__ud")-pl.col("SCHE_ud")).dt.total_days(),
        ddoff_loc=(pl.col("OFFp_ld")-pl.col("SCHE_ld")).dt.total_days(),
    )
tr=prep(pl.scan_parquet(files)).collect()
tr=tr.with_columns(y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).with_columns(
    roll=pl.col("y")>72000, fill=(pl.col("y")-pl.col("ms")).abs()<=60,
    bloc_eq_sched_loc=(pl.col("BLOC_ld")==pl.col("SCHE_ld")), bloc_eq_sched_utc=(pl.col("BLOC_ud")==pl.col("SCHE_ud")))
tr=tr.with_columns(cls=pl.when(pl.col("roll")&~pl.col("fill")).then(pl.lit("roll")).when(pl.col("fill")).then(pl.lit("fill")).otherwise(pl.lit("other")))
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
print("== 16 rows y>72000: local vs UTC date(BLOCK)==date(SCHED)")
print(tr.filter(pl.col("roll")).select("ADEP_mvt","FLIGHT_mvt","SCHED_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","MVT_TIME_UTC_mvt","y","ms","unm","bloc_eq_sched_utc","bloc_eq_sched_loc","dd_utc","dd_loc").sort("y"))
print("== Q1: all DEP rows (all airports) by local-day crossing of take-off-15min vs SCHED; rollover count")
print(tr.group_by("ADEP_mvt","ddoff_loc").agg(n=pl.len(),roll=(pl.col("cls")=="roll").sum(),fill=(pl.col("cls")=="fill").sum(),unm=pl.col("unm").sum()).filter(pl.col("ddoff_loc")!=0).sort("ADEP_mvt","ddoff_loc"))
L=tr.filter(pl.col("ADEP_mvt")=="LIRF")
print("== LIRF crossers (ddoff_loc>=1) by ms band x lane: classes")
L2=L.filter(pl.col("ddoff_loc")>=1).with_columns(band=pl.col("ms").cut([3600,14400,28800,43200,57600,72000],labels=["<1h","1-4h","4-8h","8-12h","12-16h","16-20h",">=20h"]))
print(L2.group_by("unm","band").agg(n=pl.len(),roll=(pl.col("cls")=="roll").sum(),fill=(pl.col("cls")=="fill").sum(),other=(pl.col("cls")=="other").sum()).sort("unm","band"))
print("== LIRF non-crossers (ddoff_loc==0) ms>=12h by lane")
print(L.filter((pl.col("ddoff_loc")==0)&(pl.col("ms")>=43200)).group_by("unm").agg(n=pl.len(),roll=(pl.col("cls")=="roll").sum(),fill=(pl.col("cls")=="fill").sum(),other=(pl.col("cls")=="other").sum()))
print("== LIRF unmatched ms>=12h: crossing (UTC vs local) x class")
print(L.filter(pl.col("unm")&(pl.col("ms")>=43200)).group_by("dd_utc","dd_loc","ddoff_loc").agg(n=pl.len(),roll=(pl.col("cls")=="roll").sum(),fill=(pl.col("cls")=="fill").sum(),other=(pl.col("cls")=="other").sum()).sort("dd_utc","dd_loc"))
# rows with local crossing and ms in 12-16h among unmatched LIRF: listing
print(L.filter(pl.col("unm")&(pl.col("ms")>=43200)).select("FLIGHT_mvt","SCHED_TIME_UTC_mvt","MVT_TIME_UTC_mvt","ms","y","cls","dd_utc","dd_loc","ddoff_loc").sort("ms"))
tr.filter(pl.col("ADEP_mvt")=="LIRF").write_parquet("/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/review/lirf_train.parquet")
# ranking
rk=prep(pl.scan_parquet("/home/user/prc-data-challenge/data/raw/ranking.parquet")).collect()
print("== ranking LIRF unmatched ms>=12h, local/UTC crossing")
print(rk.filter((pl.col("ADEP_mvt")=="LIRF")&pl.col("unm")&(pl.col("ms")>=43200)).select("FLIGHT_mvt","SCHED_TIME_UTC_mvt","MVT_TIME_UTC_mvt","ms","dd_utc","dd_loc","ddoff_loc").sort("ms"))
print("== ranking crossers (ddoff_loc>=1) unmatched by airport x ms band")
print(rk.filter(pl.col("unm")&(pl.col("ddoff_loc")>=1)).with_columns(band=pl.col("ms").cut([3600,14400,43200,57600],labels=["<1h","1-4h","4-12h","12-16h",">=16h"])).group_by("ADEP_mvt","band").agg(n=pl.len()).sort("ADEP_mvt","band"))
print("== ranking matched LIRF rows with mvt-lobt > 72000")
print(rk.filter((pl.col("ADEP_mvt")=="LIRF")&~pl.col("unm")).with_columns(mlobt=(pl.col("MVT_TIME_UTC_mvt")-pl.col("LOBT_flt")).dt.total_seconds()).filter(pl.col("mlobt")>20000).select("FLIGHT_mvt","SCHED_TIME_UTC_mvt","LOBT_flt","AOBT_3_flt","MVT_TIME_UTC_mvt","ms","mlobt"))
print("== training: matched rows with mvt-aobt3 > 72000 (MVT-date-wrong type), all airports")
print(tr.filter(~pl.col("unm")).with_columns(ma=(pl.col("MVT_TIME_UTC_mvt")-pl.col("AOBT_3_flt")).dt.total_seconds()).filter(pl.col("ma")>20000).select("ADEP_mvt","FLIGHT_mvt","SCHED_TIME_UTC_mvt","AOBT_3_flt","BLOCK_TIME_UTC_mvt","MVT_TIME_UTC_mvt","y","ma"))

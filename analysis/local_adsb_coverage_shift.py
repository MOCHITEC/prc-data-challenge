"""ADS-B status of matched departures by airport and month, Jan/Jul 2025 (labelled) vs Jan/Jul 2026 (scored).
Read-only; uses only the stage-1 joined files in $SCRATCH (no label)."""
import glob, os, sys
sys.path.insert(0, 'scripts')
import polars as pl
import adsb_stage1_feasibility as s1
S = os.environ['SCRATCH']
pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(30)
def load(year):
    out = []
    for f in sorted(glob.glob(f"{S}/v{year}.*_joined.parquet")):
        sch = pl.read_parquet_schema(f)
        keep = [c for c in ['MVT_ID_mvt','ADEP_mvt','t_mvt','t_aobt3','join','icao','ground_fix','t_move','t_move_lo','t_dep1',
                            't_to','push','post_dwell','stand_rule','stand_dur','stand_nfix','dep1_gs','gap_before_dep1',
                            'opens_with_landing','seen','n_ground'] if c in sch]
        J = s1.add_estimators(pl.read_parquet(f, columns=keep))
        out.append(J.select('MVT_ID_mvt','ADEP_mvt','t_aobt3','icao','ground_fix','t_move','t_push')
                   .with_columns(pl.lit(os.path.basename(f)[6:8]).alias('mon')))
    D = pl.concat(out, how='diagonal_relaxed')
    return D.filter(pl.col('t_aobt3').is_not_null()).with_columns(
        pl.when(pl.col('icao').is_null()).then(0).when(~pl.col('ground_fix').fill_null(False)).then(1)
        .when(pl.col('t_move').is_null()).then(2).when(pl.col('t_push').is_null()).then(3).otherwise(4).alias('k'))
a, b = load('2025'), load('2026')
for mon in ['01', '07']:
    rows = []
    for lab, D in (('2025', a), ('2026', b)):
        G = D.filter(pl.col('mon') == mon).group_by('ADEP_mvt').agg(
            pl.len().alias(f'n {lab}'),
            (pl.col('k') == 0).mean().round(2).alias(f'none {lab}'), (pl.col('k') == 1).mean().round(2).alias(f'air {lab}'),
            (pl.col('k') >= 2).mean().round(2).alias(f'ground+ {lab}'), (pl.col('k') >= 3).mean().round(2).alias(f'stand+ {lab}'))
        rows.append(G)
    T = rows[0].join(rows[1], on='ADEP_mvt').sort('ADEP_mvt')
    T = T.select('ADEP_mvt', 'n 2025', 'n 2026', 'none 2025', 'none 2026', 'air 2025', 'air 2026', 'ground+ 2025',
                 'ground+ 2026', 'stand+ 2025', 'stand+ 2026')
    print(f'\n=== month {mon}: matched departures by ADS-B status, 2025 vs 2026'); print(T)

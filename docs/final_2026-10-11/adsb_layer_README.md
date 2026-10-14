> **Status (2026-10-14): superseded — see [`docs/FINAL_SUBMISSION.md`](../FINAL_SUBMISSION.md).** The status line below ("June blocked") is from 10-11; June was later ingested for 29 of 30 days. The method description still applies.

# ADS-B residual layer for the final ranking set (Jan, Feb, Jun, Jul 2026)

Status (2026-10-11 17:40 UTC): models fit, Jan/Jul validated against v47, February built and applied on the dev base.
June blocked: the ADS-B ingest died while fetching 2026-06-01 (section 4). Final assembly:

    cd /home/user/prc-final && POLARS_MAX_THREADS=2 .venv/bin/python adsb_out/adsb_layer.py apply \
        --base <pre-residual final blend, 670,790 rows> --out <out.parquet>

(all four months by default; Jan/Jul are recomputed on the final MVT_TIME clock; rows without ADS-B keep the base).
To restart the ingest for the missing days (the default day list of the script is Jan+Jul, so pass --days):

    cd /home/user/prc-final && SCRATCH=/tmp/claude-0/-home-user-prc-data-challenge/7ae154e2-3baf-5cb4-beae-354c18126ab1/scratchpad/adsb \
      POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
      nohup .venv/bin/python scripts/adsb_ingest_all.py run --days $(seq -f "2026-06-%02g" 1 30 | paste -sd,) >> ingest_run1.log 2>&1 &

## 1. What the v47 ADS-B layer is (provenance)

v47 (`prc-data-challenge/outputs/c15_v42_flfixm_seeds_ucmix.parquet`) was never built in one step. Its ADS-B layer is
a chain of deltas applied to successive submission files:

| step | file | what it did to the ADS-B layer | rows |
|---|---|---|---|
| v35 (other session) | — | "their" learned residual (Jan+Jul 2025 callsign rows, original PARAMS) on every callsign ADS-B row, + the v34 positional rule overlay | all callsign rows except LTFM |
| v34 overlay | `local_posjoin_overlay.py` | positional-join rule rows (unmatched, push signature): `b + w (a - b)`, `a = MVT - t_move` (w = 1) or end-of-stop variant (w = 0.75) | 430 (Jan/Jul) |
| C1 / v36 | `local_posjoin_learned_apply.py` | Jan+Jul residual (`final_model_cb_local.txt`) on positional-join rows (not LIRF/LTFM, not rule rows) | 885 |
| C2 / v37 | `local_adsb_apply_months.py` + `local_months_ground_only.py` | swap Jan+Jul (A) -> 4-month (B) residual, ground-fix rows only (LIRF included) | 240,843 |
| C3 / v38 | `local_adsb_apply_swap.py 4:cur 4:v2` | 4-month current params -> 4-month V2 params, ground-fix rows (LIRF included) | 242,813 |
| v40 | `EXCLUDE=LIRF local_adsb_apply_swap.py 4:v2 7:v2` | 4:v2 -> 7:v2 on ground-fix rows, LIRF excluded | 230,380 |
| v40 | `local_posjoin_model_swap.py` | positional rows: Jan+Jul -> 7:v2 (`final_model_7v2.txt`) | 884 |
| v40 | `EXCLUDE=LIRF,LFPG local_adsb_apply_swap.py 7:v2 12:v2` | 7:v2 -> 12:v2 on ground-fix rows, LIRF and LFPG excluded | 206,925 |
| v42 (C11) | `local_adsb_rescat.py apply12 ... yes` | 12:v2 -> 12:v2+apron (LFPG: 7:v2 -> 7:v2+apron), ground-fix rows, LIRF/LTFM excluded | 230,454 |
| v47 | — | no ADS-B change (member deltas, ucmix, LOBT clip only) | |

Every swap used `new = file + c_new(b) - c_old(b)` with b recovered from the file by a fixed point. With the
residuals' dependence on b being weak, the telescoped end state per row is:

| component (this rebuild) | rows | correction |
|---|---|---|
| G12A | callsign ADS-B row, `ground_fix`, airport not LTFM / LIRF / LFPG | 12-month V2 + apron residual (778 rounds) |
| G7A | same at LFPG | 7-month V2 + apron residual (441 rounds) |
| GL4 | same at LIRF | v35 residual + (B - A) + (4:v2 - 4:cur) ~= 4-month V2 residual (521 rounds) |
| AIR | callsign ADS-B row without a ground fix, not LTFM | v35's Jan+Jul residual; proxy here: `final_model_months_A.txt` (local Jan+Jul refit, 386 rounds) |
| POS | positional-join row, not LIRF / LTFM / rule row | 7-month V2 residual (`final_model_7v2.txt`, saved production model) |
| RULE | positional rule rows | `b + w (a - b)` |
| LTFM | — | 0 for every learned component |

c is clipped to +-1800 s, a negative new value reverts to b, and values are rounded.

Model training data (as production): 2025 callsign ADS-B rows of `learn/feat_2025{,novdec,aso,fmj}.parquet`, with b =
the cb hybrid OOF (Jan/Jul folds), the forward-fold OOF (Nov/Dec), the local aso and fmj folds (Aug-Oct, Feb-Jun);
target `clip(y - b, +-3600)`; `adsb_learn_cv.PARAMS` + V2 (255 leaves, min 300, ff 0.7), seed 20260930. The OOF files
now live only in `/home/user/prc-work/experiments` (the main repo's copies are gone), so `fit` reads them from there.
The apron vocabulary is the production one (`$SCRATCH/agents/rescat/vocab.json`, 125 kept codes).

Per-day features: `scripts/adsb_learn_features.py day_features` on the stage-1 `<tag>_joined.parquet` / `<tag>.parquet`
(same code as production; Jan/Jul use the stored `learn/feat_2026.parquet`). Positional join:
`analysis/local_adsb_posjoin.py run` (+ `local_posjoin_learnfeat.py`, `local_dwell_end.py` logic), outputs redirected
to `adsb_out/pos` and `adsb_out/feat`.

Models never saved in production and refit here: 12:v2+apron (`models/m12a.txt`), 7:v2+apron (`models/m7a.txt`),
4:v2 (`models/m4v2.txt`). Reused as saved: `learn/final_model_months_A.txt`, `learn/final_model_7v2.txt`.

## 2. Script

`adsb_out/adsb_layer.py` (Python: `/home/user/prc-final/.venv/bin/python`, POLARS_MAX_THREADS=2, peak RSS ~2.3 GB in `fit`,
< 1 GB otherwise):

```
adsb_layer.py fit [--check-7v2]                      # once; done -> adsb_out/models/{m12a,m7a,m4v2,m7v2_check}.txt, meta.json
adsb_layer.py feat --days 2026-02-01,...             # per day, after the ingest analysed it (feat_loop.sh does this)
adsb_layer.py apply --base <blend.parquet> --out <out.parquet> [--months 2,6] [--tmvt final|stored]
```

`apply` writes `<out>` (MVT_ID_mvt, TAXITIME_SEC_mvt, same rows / order as `--base`; rows without a component keep the
base value) and `<out stem>_detail.parquet` (b, comp, c, new per row). `--base` must be the pre-residual blend.
`--months` restricts the correction to some months (default all four).

**MVT_TIME jitter in the final set.** The final ranking file re-releases the Jan/Jul departures with `MVT_TIME_UTC_mvt`
(and arrivals' BLOCK) shifted by -12..+12 s (median |shift| 4 s, 92 % of rows) against the old ranking file; every other
column is identical. The Jan/Jul ADS-B joined files were built on the old times, the Feb/Jun ones on the final times.
`--tmvt final` (default) re-anchors every ADS-B row on the final MVT_TIME so features, base and the (final) label use
one clock; `--tmvt stored` keeps the old times (used only for the comparison with v47 below). Anyone reusing old-ranking
artefacts for Jan/Jul (e.g. v47 values) should know the label clock moved by up to 12 s.

Reproduction checks of the inputs:
- `day_features('2026-01-15')` rebuilt now equals the stored `learn/feat_2026-01-15.parquet` exactly.
- `local_adsb_posjoin.run` on 2026-01-04 rebuilt now equals the stored `pos/` file exactly.
- Training rows for 12:v2 = 1,645,799 (production log: 1,645,799); 7 months 1,027,067 (same).
- 7:v2 refit (float32 matrix) vs the saved production `final_model_7v2.txt` on 200k training rows: corr 0.997,
  RMS diff 8.2 s against an RMS correction of 104 s.

## 3. Validation on Jan+Jul 2026 (old ranking rows) against v47

Base b0 = v47-era blend rebuilt from the pb-work member files (`validate_janjul.py`: matched lane
((0.5 a2o + 0.5 pao) + p2o + c2 + m4) / 4 with the per-member fl_/seq_ fix, unmatched lane = model B from the a2o file).
v47 - b0 contains the ADS-B layer plus everything else that differs (LIRF overlays, NM-blind, 3-seed deltas, ucmix,
LOBT clip); rows without any ADS-B component show that noise floor. Applied with `--tmvt stored`.

| component | rows | RMS(v47 - b0) | RMS(v47 - b0 - c) | RMS c | corr(c, v47 - b0) | slope |
|---|---|---|---|---|---|---|
| G12A | 216,175 | 145.0 | 52.1 | 136.6 | 0.934 | 0.99 |
| G7A (LFPG) | 22,409 | 105.6 | 53.3 | 85.7 | 0.864 | 1.07 |
| GL4 (LIRF) | 9,750 | 147.5 | 119.8 | 91.1 | 0.585 | 0.95 |
| AIR (not LIRF) | 21,950 | 32.8 | 26.0 | — | 0.558 | 0.95 |
| AIR (LIRF) | 15,377 | 205.9 | 151.7 | — | 0.690 | 1.23 |
| POS | 890 | 471.6 | 161.7 | 399.0 | 0.946 | 1.11 |
| RULE | 430 | 1164.9 | 128.7 | 1129.4 | 0.994 | 1.04 |
| none (not LIRF) | 56,088 | 33.4 | 33.4 | 0 | | |
| none (LIRF) | 1,772 | 324.6 | 324.6 | 0 | | |

All rows: RMS(v47 - b0) 139.5 s -> RMS(v47 - (b0 + c)) 64.6 s; rows with a component and |v47 - b0| < 1800 s: corr
0.913, 138.4 -> 56.4 s. On G12A rows the median |v47 - b0 - c| is 18 s (median |v47 - b0| 68 s). Row selection matches
production exactly where it can be checked: G12A + G7A = 216,175 + 22,409 = 238,584 rows, the same count as the
production apron swap (`rescat/rank_delta_12m.parquet`, 238,584 ground rows, LIRF/LTFM excluded); POS 890 vs 884-885
in production; RULE 430 = the 430 rows of `replacements_v33.csv`.

Reading: the layer is reproduced closely but not exactly. The ground-fix components explain ~87 % of the
(v47 - b0) variance on their rows with slope ~1; the leftover ~45 s RMS above the ~30 s noise floor comes from (a) the
production swaps having been evaluated at a fixed-point b̂ that differs from b0 by ~40 s on average (the audit found the
fixed point does not converge), (b) float32 / refit noise (~8 s), (c) at LIRF / air-only rows, the v35 residual of the
other session, approximated by the local Jan+Jul refit. LIRF rows are dominated by the label-artefact overlays, so their
low correlation says little about the residual.

## 4. Feb + Jun 2026 (in progress)

Features are built by `feat_loop.sh` (polls the ingest state; `adsb_layer.py feat` per analysed day) into
`adsb_out/feat/` and `adsb_out/pos/`. Development base: `pb-work/outputs/final/main_flseq_osm_nolirf_l2_opt_l6_cmix.parquet`
(a2o + model B), output `adsb_out/dev/a2o_dev_layer.parquet` (+ `_detail`). Per-day coverage: `coverage_days.py <detail>`.

### February (28/28 days ingested)

Dev base, `--tmvt final`. Coverage (share of departures with any component) is 0.96-0.98 at the 9 ADS-B airports,
the same as Jan/Jul (departures before 02:00 UTC are never joined, by stage-1 design). LTFM 0 (rule).

| airport | Feb departures | covered | ground-fix learned | air-only | pos + rule | c q10 / q50 / q90 (covered) | RMS c (all rows) | Jan RMS c |
|---|---|---|---|---|---|---|---|---|
| EDDF | 14,654 | 14,262 | 10,157 | 4,030 | 75 | -54 / -8 / 37 | 69.5 | 135.7 |
| EDDM | 10,534 | 10,311 | 10,291 | 19 | 1 | -126 / -2 / 166 | 157.3 | 182.7 |
| EGLL | 17,719 | 17,335 | 16,640 | 682 | 13 | -142 / -9 / 154 | 125.3 | 141.3 |
| EHAM | 16,802 | 16,425 | 16,257 | 25 | 143 | -136 / -5 / 141 | 134.0 | 261.4 |
| LEBL | 12,167 | 11,892 | 11,787 | 104 | 1 | -117 / -13 / 109 | 103.1 | 105.6 |
| LEMD | 16,191 | 15,657 | 15,601 | 45 | 11 | -130 / -17 / 131 | 118.4 | 123.3 |
| LFPG | 16,934 | 16,506 | 7,596 | 8,906 | 4 | -23 / -7 / 25 | 38.0 | 86.2 |
| LIRF | 10,264 | 9,988 | 8,784 | 1,204 | 0 | -79 / -3 / 91 | 91.9 | 94.5 |
| LSZH | 9,374 | 9,108 | 9,028 | 60 | 20 | -139 / -15 / 151 | 131.7 | 162.3 |
| LTFM | 19,519 | 0 | 0 | 0 | 0 | — | 0 | 0 |

Feb correction quantiles by component (Jan in brackets): G12A n 89,761, RMS 125 (150), q01 / q10 / q50 / q90 / q99 =
-293 / -127 / -12 / 137 / 376; G7A n 7,596, RMS 54 (112); GL4 n 8,784, RMS 92 (94); AIR n 15,075, RMS 34 (46);
POS n 219, RMS 254 (507); RULE n 49, RMS 465 (1,359; Jan is dominated by the EHAM snow days).

Coverage shifts against Jan/Jul (receiver coverage, not a pipeline fault; same code reproduces stored Jan days exactly):
- **EDDF**: ground-fix share drops from 0.90-0.93 per day to 0.49-0.80 from 2026-01-29 on and stays there all of
  February (air-only share 0.06 -> 0.28). Ground-fix rows are also sparser: median `post_n` 4 (Jan 12, Jul 8; the 2025
  training rows had a median around 34). EDDF corrections shrink accordingly (RMS 80 on G12A rows vs 139 Jan / 89 Jul).
  This is the regime the audit already flagged as out of the 2025 distribution; expect less gain at EDDF in Feb.
- **LFPG**: ground share 0.45 (Jan 0.56, Jul 0.56), with low days on 02-15..02-17 and 02-20..02-22 (0.25-0.32); Jul has
  the same kind of dip on 07-23..07-31.
- **LIRF**: Feb has ground coverage like Jan (0.86 vs 0.88); July 2026 has none. So Feb LIRF gets the GL4 (4:v2)
  correction on 8,784 rows, as Jan did in v47.
- Other airports: Feb coverage and correction spread match Jan/Jul.

### June
NOT DONE: the ingest process stopped silently at 17:06-17:08 UTC while downloading 2026-06-01 (part `.ab` at 0.95 of
1.50 GB, no error in `ingest.log`, no OOM kill recorded, no ingest process alive since). It was not restarted from here
(instruction: do not start an ingest). It is resumable: restarting `scripts/adsb_ingest_all.py run` from
`/home/user/prc-final` (same SCRATCH env) resumes the partial download. `feat_loop.sh` is still polling every 5 min and
builds each June day's features as soon as the ingest marks it analysed; then rerun `apply`.

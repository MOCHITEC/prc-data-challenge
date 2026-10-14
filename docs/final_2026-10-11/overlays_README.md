# v47 post-processing layers for the final evaluation set (lane-B levers + LIRF overlays)

Status: done 2026-10-11 (lane-B deltas fitted on the final data 20:41-20:57 UTC; validation below).

## What is here

| file | what |
|---|---|
| `laneB_deltas.py` | Lane-B (unmatched, AOBT_3 null) deltas on the final data: NM-blind blend (w 0.25, unmatched rows outside LIRF) and ucmix (LIRF unmatched, 0 < MVT−SCHED < 8 h). Same method as `prc-data-challenge/analysis/local_{nmb,ucmix}_laneB.py`; one lane-B-only fit with both flags (the two blends touch disjoint rows, model B does not depend on either flag). |
| `run_laneB_final.sh` | waits until no `prc.pipeline` fit runs and ≥ 9 GB is free, then runs `laneB_deltas.py` on `data/raw_final` + `data/work_final`. |
| `apply_overlays.py` | applies the LIRF overlays (and optionally the lane-B deltas and the LOBT clip) on top of a base file. |
| `val_copy_q.py`, `val_compose.py`, `val_laneB.py` | Jan+Jul validation against the v47 lineage. |
| `out/` | outputs: `nmb_laneB_delta_final.parquet`, `ucmix_laneB_delta_final.parquet`, `copyq_LIRF_final_fixed.parquet`, logs. |
| `val/` | validation intermediates. |

## Order assumed by `apply_overlays.py`

v47 order: **base blend → ADS-B residual layer → [lane-B deltas + LIRF overlays] → LOBT clip**.

`--base` must already carry the ADS-B residual and must not carry any of the layers below. The LOBT clip is
applied last only with `--lobt-clip` (same rule as `analysis/local_lobt_clip.py`, using `mvt_minus_lobt` from
the final `feat_rank.parquet`); otherwise run it afterwards.

Inside the script (reproduces v47 exactly, see validation C_old):
1. **lane-B deltas** (`--laneB`): NM-blind delta on unmatched non-LIRF rows; ucmix delta on LIRF unmatched rows in
   its gate, **except** the rows that rule B or C claims (v47: "the time-zone-class and copy-band rows keep our values").
2. **Rule C, LIRF unmatched copy band**: 7 ≤ (ms − 1100)/3600 ≤ 13.5 → predict ms (ms = MVT − SCHED from the raw ranking file).
3. **Rule B, LIRF unmatched time-zone class**: 0 ≤ ms ≤ 3599 and the flight prefix had ≥ 1 such 2025 row:
   q·(Rome UTC offset at SCHED + 865) + (1 − q)·p, q = tz/(n + 1) per prefix (2025 rates), p = base value (pre-ucmix).
   The one-day class C7r (ms ≥ 50 400) is **not** applied (dropped after v41).
4. **Matched LIRF copy overlay**: classifier q (LightGBM binary, 63 leaves, 1 thread, seed 1, 108 rounds = int(99·1.1),
   refit on all 2025 LIRF matched rows), p' = p + 0.4·q·(mvt_minus_sched − p) (kept at p if negative), on every LIRF
   matched row.
5. LOBT clip (optional).

Everything is computed in float and rounded once at the end when the base column is an integer. Rows no layer
touches keep the base value exactly.

## Usage

```bash
# (a) lane-B deltas on the final data (~needs 5-6 GB; do not run beside a prc.pipeline fit)
cd /home/user/pb-work && PYTHONPATH=src POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
  /home/user/prc-work/.venv/bin/python /home/user/prc-final/overlays/laneB_deltas.py \
  --raw-dir data/raw_final --work-dir data/work_final --out-dir /home/user/prc-final/overlays/out --tag final

# (b) overlays on a base (base = blend + ADS-B residual), lane-B deltas included, LOBT clip last
POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage /home/user/prc-work/.venv/bin/python \
  /home/user/prc-final/overlays/apply_overlays.py --base <base.parquet> --out <out.parquet> --laneB --lobt-clip \
  [--q-cache /home/user/prc-final/overlays/out/copyq_LIRF_final_fixed.parquet] [--report <rows.parquet>]
```

Defaults: `--raw-dir /home/user/pb-work/data/raw_final`, `--feat-dir /home/user/pb-work/data/work_final`,
`--copy-feat fixed` (classifier on the union-fixed `feat_rank.parquet`; `unfixed` uses `feat_rank_unfixed.parquet`),
`--nmb/--ucmix` = the `out/*_final.parquet` delta files. `--q-cache` skips the ~20 s classifier refit.
`--no-copy` skips the matched copy overlay.

## Outputs (final data, all 670 790 ranking rows; the delta files hold the 9 077 unmatched rows)

| file | content |
|---|---|
| `out/nmb_laneB_delta_final.parquet` | MVT_ID_mvt, plain, blend, delta, ADEP_mvt. NM-blind: 8 458 target rows (non-LIRF unmatched), 8 354 with abs(delta) > 0.5; RMS by month Jan 111 / Feb 75 / Jun 55 / Jul 45 s. Fit: model B 43 rounds, NM-blind 1 130 rounds on 2 084 075 rows. |
| `out/ucmix_laneB_delta_final.parquet` | same columns. ucmix: 582 LIRF rows in the gate (Jan 96, Feb 72, Jun 154, Jul 260), mean delta -515 s. `plain` = model B for every row (re-based, see `laneB_deltas.py`). |
| `out/copyq_LIRF_final_fixed.parquet` | copy q for the 51 527 LIRF matched rows (MVT_ID_mvt, q, ms_feat); pass it as `--q-cache`. |
| `out/laneB_final.log` | fit log. |

Rows touched in the final set: rule C 20 (Jan 7, Feb 2, Jun 1, Jul 10), rule B 8 (2 of them Feb/Jun), copy overlay
51 527 LIRF matched rows (Feb 10 189, Jun 14 821); the ucmix skip covers the rule-B/C rows.

## Validation on Jan+Jul (old ranking set, the v47 lineage)

**Copy classifier** (`val_copy_q.py`): on the old Jan+Jul frame it reproduces v47's q (`copyq_LIRF.parquet` q_orig)
**exactly** (max abs(dq) = 0, 26 516 rows). On the final frame, Jan+Jul ids: corr 0.9981 (fixed features) / 0.9985
(unfixed) vs q_orig, RMS dq 0.014 / 0.012. The cause is the re-drawn ±12 s MVT jitter in the final files and the
4-month ranking frame.

**A. Copy overlay, not circular**: base v33_posjoin → `apply_overlays` (old frame) vs production v35_tmp: **26 516 / 26 516
LIRF matched rows identical**.

**B. Rules B/C, not circular**: base = c10 with LIRF unmatched rows set back to v40 + fl-fix lane-B delta, vs production c10:
rule C 17/17 identical; rule B 4/6 identical; the other 2 differ by 13 s and 3 s, because production applied rule B
before adding the fl-fix delta on those rows. 360 untouched rows identical. With the final raw file: the same 23 rows
are selected; the copy-band values move by the MVT jitter (RMS 5 s).

**C. End to end on v47** (v47 before the overlays rebuilt from c15_pre: LIRF matched inverted with q_orig, old
ucmix/nmb deltas subtracted, rule-B rows inverted):
- *C_old* (old raw, old frame, old delta files): **all 344 841 rows identical to v47**. The composition and row
  selection are right: ucmix on 349 gate rows, the rule-B/C rows skipped, nmb on 4 907 rows.
- *C_final* (final raw, final q, final delta files) vs v47, per layer. "ours" and "v47" are each change from the
  same pre-overlay base.

| layer | rows | corr (ours vs v47) | RMS of change: ours / v47 | RMS difference | Σ diff² / N |
|---|---|---|---|---|---|
| copy overlay (LIRF matched), q fixed | 26 516 | 0.9945 | 58.6 / 59.1 | 6.0 | 2.8 |
| copy overlay, q unfixed | 26 516 | 0.9974 | 59.0 / 59.1 | 4.1 | 1.3 |
| NM-blind (non-LIRF unmatched) | 4 907 | 0.9946 | 85.6 / 85.6 | 8.9 | 1.1 |
| ucmix (LIRF unmatched, gate) | 360 | 0.979 | 986 / 971 | 172 | 31.0 |
| rule B | 6 | 1.0 | identical | 0 | 0 |
| rule C | 17 | – | ms moves by the jitter | 5.4 | 0.001 |

**Lane-B deltas vs the old-ranking files** (`val_laneB.py`, shared Jan+Jul ids): NM-blind corr 0.9946, RMS 85.6 vs
85.6, RMS difference 8.9 s, mean −6.18 vs −6.33. ucmix: same 356 rows non-zero, corr 0.979, RMS 1 012 vs 998,
RMS difference 173 s, mean −548.2 vs −548.2. The plain model-B predictions differ by RMS 30 s (NM-blind rows) and
416 s (LIRF ucmix rows). That comes from the different inputs: the final file has a new MVT jitter draw (633 739
of 689 534 Jan/Jul movements moved by up to ±12 s), the context features are recomputed over the 4-month frame,
and the features are union-fixed. The final ranking frame also changes the 2026-side label encodings and categories.

## Caveats
- The final ranking file is **not** the old one with more months added. MVT_TIME (and so BLOCK/TAXITIME semantics and
  every MVT-based feature) carries a new ±12 s noise draw, so Jan/Jul deltas computed on the final data do not
  equal the v47 ones bit for bit (numbers above). If Jan/Jul are taken from v47, Feb/Jun from this run, the two
  halves are consistent in method, not in noise draw.
- ucmix is the noisiest layer (RMS change about 1 000 s on few rows; Jan/Jul RMS difference 173 s). Its effect
  is dominated by the copy-vs-taxi classifier p on rows 0-8 h late.
- Copy-overlay q default = union-fixed final features (train/test consistent). v47 used the unfixed (Jan+Jul)
  frame. `--copy-feat unfixed` matches v47 slightly better on Jan/Jul (RMS diff 4.1 vs 6.0 s); either is fine.
- v47's base already contains cmix (LIRF copy mixture inside the a2o member, blend weight 1/8), and the 0.4 copy
  overlay sits on top of that full blend. The external-ideas note "cmix instead of the overlay, never both" was
  not followed in v42/v47 (checked: c11 − c10 on LIRF matched rows = the v40 + L6/cmix member deltas). This script
  reproduces v47 as it was.
- Rule B blends from the base value it receives (pre-ucmix). Production C10 applied it before the fl-fix lane-B
  delta on 2 rows (13 s and 3 s difference).
- The LOBT clip only touches rows with a LOBT window (matched rows; `mvt_minus_lobt` is null on all unmatched rows).
  In a dry run on a pb member file, 2 LIRF matched rows were clipped after the copy overlay.
- `laneB_deltas.py` peaks at about 10 GB RSS for about 16 min, so do not run it beside a `prc.pipeline` fit.

# v42 candidate: v40 + (R6-cbopt − mlp) matched-lane delta (2026-10-06)

Not submitted. Nothing uploaded. File: `outputs/v42_candidate_v40_plus_r6cbopt_delta.parquet` (git-ignored, local
only), md5 `d6c780ae4a94aa7379a7cea5690d651c`.

## Inputs

- v40 (board 238.7712): `resilient-hedgehog_v40.parquet`, downloaded read-only from the team bucket. v35–v39 were also
  downloaded read-only, to identify v40's own steps. The bucket v35 equals `outputs/v35_candidate_mlp_adsb_posjoin.parquet`.
- delta = `outputs/hybrid_blend_reweight_cbopt.parquet` − `outputs/hybrid_blend_l2_pa_opt_mlp.parquet`, row-aligned to
  `data/raw/submitting.parquet`. The delta is 0 on every unmatched row. On the 339,551 matched rows
  (`mvt_minus_aobt3` non-null in `data/work_flseq/feat_rank.parquet`) its RMS is 22.1 s and its max |delta| is 1,104 s.

## Rows not changed

- **LTFM (46,860 matched rows): skipped.** None of the v35→v40 steps changes any LTFM row (0 of 47,581 differ), so
  the rule applies. Cost: in R6's OOF the LTFM gain was −0.29 s [−0.49, −0.09] (see the evidence below). That gain is
  not in this candidate.
- **LOBT clip rows (28): excluded.** These are the rows where v40 sits exactly on `ceil(mvt_minus_lobt ± 3606)` (22,
  from v40's final clip), together with the rows where v36 (C1) sits on that bound (19, from C1's clip). Both sets
  follow the logic of `analysis/local_lobt_clip.py` on `claude/laughing-cray-vvwh0k`. All 28 rows are matched, and 26
  of them have a nonzero delta.
- **430 positional-overlay push rows (`docs/posjoin_2026-10-05/replacements_v33.csv`): excluded.** v35 and v40
  both equal the CSV's `new` column on these rows. All of them are unmatched, so their delta is 0 anyway.
- **LIRF matched schedule-copy blend: a partial overwrite, so the delta is scaled.** In v36,
  `analysis/local_lirf_copy_overlay.py` set `p' = p + 0.4·q·(MVT−SCHED − p)` on the LIRF matched rows (26,313 of
  26,516 rows changed). Here `0.4q` is recovered per row as `(v36 − v35)/(mvt_minus_sched − v35)` and clipped to
  [0, 0.4]. Its mean is 0.066. It is set to 0 on the 14 rows where `|mvt_minus_sched − v35| < 1`. The delta applied
  is `(1 − 0.4q)·delta`, which is what re-running the blend on the new base would give.
- **All other v36–v40 steps are residual swaps or deltas**: the posjoin learned residual, the 4m/7m/12m ADS-B swaps
  and the appear tier. The delta is added to them unchanged. These steps' `c(b)` depends on `b`, and they were not
  recomputed on the new base, because their models are not in this container. Treating them as unchanged is an
  approximation.

## Construction and checks

- `cand = rint(v40 + scale·delta)` on the 292,664 matched, non-LTFM, non-overwrite rows. Every other row keeps v40's
  value.
- 25 rows that the delta pushed outside the LOBT box were clipped back into it. This is v40's last step, re-applied.
- 3 negative values were reverted to v40's value.
- Checks: 344,841 rows; schema, ids and order equal `data/raw/submitting.parquet`; 0 nulls; 0 negatives. The minimum
  is 0 and the maximum is 113,641.
- **Versus v40:** 281,181 rows changed. On the changed rows the RMS is 21.9 s (19.8 s over all rows). The max
  |delta| is 1,104 s and sum d²/N is 391.

## OOF evidence for the delta

- `rejected-rereview.md` §4, R6 against `hybrid_blend_l2_pa_opt_mlp` on 672,428 rows: "K=0 **−0.32 [−0.44, −0.24]**
  ... K=200 **−0.36 [−0.48, −0.26]** ... Month bootstrap −0.32 [−0.38, −0.23]. **Verdict: ADOPT.**"
  - Matched lane: −0.43 [−0.55, −0.31].
  - LIRF −0.55 [−0.95, −0.12]; LTFM −0.29 [−0.49, −0.09].
- `catboost-retune.md`, R6 secondary (`prc.compare rr_R6_pa5_lofo rr_R6_pa5_lofo_cbopt`): "K=0 **−0.07 [−0.14,
  −0.01]**, K=20 −0.09 [−0.16, −0.02], K=200 **−0.09 [−0.16, −0.01]**, K=2000 −0.11 [−0.16, −0.06]; month bootstrap
  −0.07 [−0.16, +0.00] ... this is a secondary and is not the decision."
- This evidence is for the bases with no ADS-B residual on top. No OOF exists for "v40 recipe on R6-cbopt"; the
  residual swaps were not re-fitted on the new base.

Build script: session scratchpad `work/build_v42.py` (not committed). Peak RSS was 207 MB.

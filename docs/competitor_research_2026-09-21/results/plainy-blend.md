# Plain-y member blended 50/50 with the anchored model A on the matched lane (2026-09-30)

## 0. Pre-registration (written before any plain-y number was seen)

**Source of the idea.** `public-code-deep-dive-2026-09-30.md` §4 row 2: kind-mango blends a plain-target member
(the clock as a feature) 50/50 with an anchored member. They report hold-out −818 MSE and v74 −3.73 s live, bundled
with other changes. merry-quicksand's own test of that blend reads −9.08 s pooled. The same file lists the question
as open for us (§5): "Whether kind-mango's 50/50 plain + anchored blend beats our anchored-only model on our
features [U]." No competitor code was opened for this item. The implementation below is our own.

**Prior tests in our own docs.** None found. `grep -rin "plain\|direct target\|blend" docs/` finds only competitor
descriptions and the per-airport blend (P11). Our model A has fitted the anchored target since the first trainer
commit (ce9b0f3). The claim that it must is argument, not measurement: `src/prc/train.py` line 4 says "A tree
cannot represent the identity part of y = proxy - f(x), so the reparameterisation is the point." The nearest
recorded row is `docs/README.md` §5, "v1 → v2（2 クロック化・木を増やす） −4.69 / −3.33 / −2.76 採用". It is not a
plain-vs-anchored test: v1's code (e0099d7) already fitted `LABEL − PROXY` on the matched rows, and the change is
bundled with more trees. The deep dive (§4 row 2) records our status as "anchored `g` only".

**Targets as they are in the code.**
- **Model A (matched lane, `mvt_minus_aobt3` non-null), anchored.** Fits `g = TAXITIME_SEC_mvt − mvt_minus_aobt3`
  (= AOBT_3 − BLOCK, anchored to the MVT − AOBT_3 clock) and predicts `mvt_minus_aobt3 + g`.
- **Model B (unmatched lane).** Fits `TAXITIME_SEC_mvt − shift` and predicts `ĝ + shift`. The shift is
  `mvt_minus_sched` on LIRF rows (`unmatched_sched_reparam: true`, airports `[LIRF]`) and 0 elsewhere. So model B
  is plain-y outside LIRF and schedule-anchored at LIRF (`prc.train.unmatched_target`). It is not changed here.

**No existing direct-y option for model A.** The repo had none, so this adds one.
`TrainConfig.matched_target` is `"anchored"` (the default) or `"plain"`. The default is left out of the hash, and
the record key is only written when the value is `"plain"`.

**(1) The plain-y member**, `configs/main_flseq_osm_nolirf_plainy.yaml`. It is `configs/main_flseq_osm_nolirf.yaml`
plus `matched_target: plain`, with a new name. Model A fits `TAXITIME_SEC_mvt` directly and predicts the model's
output with nothing added back. Everything else is the same as the anchored model A of `main_flseq_osm_nolirf`:
- The feature list, with `mvt_minus_aobt3` still in it as the last column.
- The categoricals, the encodings, `osm_stand_rwy_km` (nulled at LIRF), and the LightGBM parameters: lr 0.05,
  127 leaves, min_data 50, ff/bf 0.8, λ2 1, deterministic.
- `num_rounds` 2000, `early_stopping` 100 on the one-month inner holdout, the round-scaling rule, and 3 seeds.
- The zero floor and the LOBT window projection (`lobt_bound`, slack 3606 s).

The plain-y member picks its own round count by the same early-stopping rule. Model B in this run is
`main_flseq_osm_nolirf`'s model B. It is not used.

**(2) The blend.** On the matched lane, `p = 0.5 × anchored + 0.5 × plain`. The anchored member is the OOF and
full-fit output of `main_flseq_osm_nolirf-926741d779`. The plain member is the new run. Both members are already
floored and LOBT-projected. The window is convex, so their average stays inside it. The weight is **fixed at 0.5**.
The unmatched lane is `main_flseq`'s model B (`main_flseq-8615e6a194`, v16), exactly as in H2. The lane is
`mvt_minus_aobt3` non-null: `data/work_flseq/feat_train.parquet` for OOF rows, `data/work_flseq/feat_rank.parquet`
for scored rows. That is H2's construction (`osm-on-flseq.md` §8).

**Secondary only.** The OOF-optimal weight `w` on the anchored member (`w·anchored + (1 − w)·plain`, least squares
on matched rows) is fitted leave-one-fold-out. For each of the three pooled folds (jan, jul, forward), `w` is fitted
on the matched OOF rows of the other two and applied to that fold. The per-fold weights are reported, and the
resulting hybrid is compared with H2 the same way. It does not enter the decision.

**(3) The comparison.** `hybrid_H2_plainy` (fixed 0.5) vs `hybrid_H2` on the 672,428 pooled OOF rows, with
`prc.compare`: full, K = 20 / 200 / 2000, and the month bootstrap. Also reported: the lane split, per calendar
month, per airport, and all rows except LIRF. **The decision uses `prc.compare.verdict` on the fixed-0.5 blend
only.** It needs the full CI and the drop-200 CI to both exclude zero on the same side. ADOPT needs the improving
side.

**Output, regardless of the verdict.** `outputs/hybrid_H2_plainy.parquet`. Matched rows are
`rint(0.5 × outputs/main_flseq_osm_nolirf.parquet + 0.5 × outputs/main_flseq_osm_nolirf_plainy.parquet)`. The
members are already Int32-rounded, so the file can differ from a float blend by at most 1 s per row. Unmatched rows
come from `outputs/main_flseq.parquet`. The file is validated for rows, schema, ids and order against the template,
nulls, and negatives. It is not submitted.

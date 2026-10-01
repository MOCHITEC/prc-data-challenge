# Plain-y member blended 50/50 with the anchored model A on the matched lane: ADOPT by the rule (2026-09-30)

**Verdict (prc.compare.verdict, `hybrid_H2` → `hybrid_H2_plainy`, fixed weight 0.5, 672,428 pooled OOF rows):**
full −0.78 s [−1.00, −0.60], K=200 −0.90 s [−1.09, −0.71]. Both CIs exclude zero on the improving side, so the
verdict is **ADOPT**. The month bootstrap reads −0.78 [−1.08, −0.51]. The plain-y member alone is worse than the
anchored model by +0.93 s [+0.59, +1.31]. The gain comes from the blend, not from the member.
`outputs/hybrid_H2_plainy.parquet` is built and validated. Nothing was submitted.

The pre-registration (§0) was committed as 96258d8 before the training run started (run start 10:34 UTC).

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

## 1. Run

`.venv/bin/python -m prc.pipeline --config configs/main_flseq_osm_nolirf_plainy.yaml --work-dir data/work_flseq`.
This is the same feature cache as the two base runs. The id is `main_flseq_osm_nolirf_plainy-b971f4b81b`.
- It exited with 0. CV took 3,808 s and the full fit followed. No ADS-B job was running.
- The peak cgroup `total_rss` was **10.49 GB** (`logs/peak_plainy.txt`), against a 13.68 GB limit.
- Model B was untouched. Every fold's unmatched RMSE equals `main_flseq_osm_nolirf`'s to the last digit, and all
  5,290 unmatched rows of the full-fit file equal `outputs/main_flseq_osm_nolirf.parquet`.

| fold | matched RMSE, anchored (926741d779) | matched RMSE, plain (b971f4b81b) | rounds anchored / plain |
|---|---:|---:|---|
| jan | 201.52 | 204.36 | 1178 / 955 |
| jul | 241.73 | 242.49 | 1040 / 1042 |
| jul_fwd | 250.24 | 251.22 | 893 / 890 |
| forward | 196.93 | 197.70 | 1125 / 724 |
| pooled matched | 211.49 | 212.71 | full fit 1212 / 942 |

The plain member alone vs the anchored member alone (`prc.compare`, both with `main_flseq_osm_nolirf`'s model B):
- K=0 **+0.93** [+0.59, +1.31] and K=200 +1.24 [+0.94, +1.64].
- **REJECT** ("B is worse on all rows").

This is the first direct measurement of the claim in `train.py` line 4. On our features the anchored target beats
the plain target by about 1 s. It does not beat it by a large margin, because the plain model is given the clock as
a feature.

## 2. Blend vs H2 (the decision)

Construction (`scripts/plainy_blend.py oof`):
- The H2 OOF is rebuilt from the stored files. It reproduces `osm-on-flseq.md` §8 against `main_flseq-8615e6a194`
  exactly: −0.14 [−0.23, −0.03], K=200 −0.14 [−0.25, −0.03].
- Hybrid OOF files are kept in the scratchpad, not in `experiments/`.

`prc.compare hybrid_H2 hybrid_H2_plainy --work-dir data/work_flseq`:

| | K=0 | K=20 | K=200 | K=2000 | month bootstrap | verdict |
|---|---|---|---|---|---|---|
| **fixed 0.5** | **−0.78 [−1.00, −0.60]** | −0.93 [−1.16, −0.72] | **−0.90 [−1.09, −0.71]** | −0.83 [−0.94, −0.70] | −0.78 [−1.08, −0.51] | **ADOPT** |
| LOFO weight (secondary) | −0.81 [−1.00, −0.65] | −0.96 [−1.15, −0.80] | −0.95 [−1.11, −0.80] | −0.85 [−0.96, −0.75] | −0.81 [−1.07, −0.57] | (ADOPT; not the decision) |

RMSE: H2 272.82, blend 272.05, LOFO 272.02. The K=200 drop removes 147 unmatched and 53 matched rows.

- **Lane split (fixed 0.5).** Matched 211.49 → 210.48: −1.02 [−1.28, −0.77]. Unmatched +0.00 (identical by
  construction). All rows except LIRF −0.91 [−1.20, −0.67].
- **Per calendar month**, all rows / matched:
  - 2025-01: −0.37 [−0.87, −0.11] / −0.64 [−1.12, −0.14]
  - 2025-07: −0.88 [−1.26, −0.52] / −1.10 [−1.54, −0.67]
  - 2025-11: −0.87 [−1.24, −0.55] / −1.00 [−1.38, −0.64]
  - 2025-12: −1.21 [−1.72, −0.79] / −1.30 [−1.85, −0.83]
  - Every month improves.
- **Per airport**, all rows, with the share of squared disagreement. The difference comes from matched rows only.
  All ten point estimates are negative, and six CIs exclude zero.

| airport | n | share | diff | CI95 |
|---|---:|---:|---:|---|
| LFPG | 78,036 | 18.2 % | −1.07 | [−2.15, −0.61] |
| EHAM | 79,639 | 15.4 % | −0.51 | [−1.13, +0.10] |
| EGLL | 79,633 | 15.1 % | −1.17 | [−2.03, −0.42] |
| LTFM | 91,338 | 13.1 % | −1.40 | [−1.83, −0.98] |
| LIRF | 49,924 | 12.3 % | −0.22 | [−0.76, +0.39] |
| EDDM | 51,688 | 7.6 % | −0.59 | [−2.22, +0.55] |
| EDDF | 72,276 | 6.1 % | −1.44 | [−1.97, −0.94] |
| LEMD | 70,732 | 4.6 % | −0.05 | [−0.37, +0.25] |
| LSZH | 43,009 | 4.1 % | −0.67 | [−1.33, −0.04] |
| LEBL | 56,153 | 3.5 % | −1.03 | [−1.37, −0.71] |

- **Secondary: OOF-optimal weight on the anchored member, leave-one-fold-out.** jan 0.563, jul 0.606, forward
  0.617 (each fitted on the other two folds). The weight fitted on all three folds is 0.594. The optimum is
  shallow: LOFO beats fixed 0.5 by 0.03 s. This does not enter the decision, and the file uses 0.5.

**Reading.** The rule passes by a wide margin: the K=200 CI upper end is −0.71. The effect is spread over months
and airports. It is larger than H2's own −0.14. It is a matched-lane change on rows with a normal error size,
because K=0, K=200 and K=2000 all read about −0.8 to −0.9. The blend was pre-registered with a fixed weight, so
unlike H2 it was not selected on these rows. The unmatched lane and its choice (v16's model B) are inherited from
H2, which was post hoc. The OOF gain is about a fifth of kind-mango's reported live −3.73 s, and theirs was
bundled with other changes. On the board, expect it to be smaller than the OOF gain, not larger.

## 3. File

`outputs/hybrid_H2_plainy.parquet` (`scripts/plainy_blend.py full`) has 344,841 rows: 339,551 matched and 5,290
unmatched. The lane is `mvt_minus_aobt3` non-null in `data/work_flseq/feat_rank.parquet`.
- Matched rows are `rint(0.5 × main_flseq_osm_nolirf + 0.5 × main_flseq_osm_nolirf_plainy)` from the Int32
  full-fit files. Unmatched rows come from `outputs/main_flseq.parquet` (v16).
- The schema `{MVT_ID_mvt: double, TAXITIME_SEC_mvt: int32}` equals `data/raw/submitting.parquet`, and the ids and
  their order equal the template.
- It has 0 nulls and 0 negatives, with min 0 and max 113,641. `python -m prc.validate_submission` prints OK, with
  only the usual >10,800 s warning.
- Against `outputs/hybrid_H2.parquet`, it differs on 328,497 rows, all of them matched, with max |d| 1,369 s and
  RMS 31.00 s. The two members differ on matched rows by RMS 62.48 s, with max 2,737 s.
- `outputs/main_flseq_osm_nolirf_plainy.parquet` (the plain member's own full fit) also validates OK.
- **No ADS-B override has been applied.** The v20 analogue would be
  `scripts/adsb_override_candidate.py --base outputs/hybrid_H2_plainy.parquet --out ...`. Nothing was submitted.

## 4. Code and tests

- **`TrainConfig.matched_target`** is `"anchored"` by default, or `"plain"`. It is in `_HASH_OPTIONAL`, and
  `predict_fold` writes `info["matched_target"]` only for `"plain"`. Its only effects are model A's target
  (`LABEL` instead of `LABEL − PROXY`) and its add-back (0 instead of `PROXY`). The < 200-row fallback still
  predicts the proxy.
- **`tests/test_matched_target.py`** has 5 tests:
  - The experiment id of every one of the 114 recipes in `configs/` equals the id computed by the code at f67a8c1,
    and each recipe resolves to `"anchored"`.
  - The plainy recipe is its base plus the key.
  - An unknown value is refused.
  - The default equals an explicit `"anchored"` byte for byte: predictions and `info`.
  - `"plain"` changes model A only: the unmatched predictions are equal.
- **Byte-identity checked in-session.** On a synthetic three-month frame, `predict_fold` predictions (two configs,
  including `lobt_bound` and the LIRF reparam) and `info` hash identically under the f67a8c1 code and the new code.
- **Full suite:** 210 passed before the run and 210 passed after it.

Logs (git-ignored):
- `logs/main_flseq_osm_nolirf_plainy.log`, `logs/run_plainy_status.txt` and `logs/peak_plainy.txt`
- `logs/plainy_oof.txt`, `logs/compare_plainy_hybrid_H2_plainy.txt` and `logs/compare_plainy_hybrid_H2_plainy_lofo.txt`
- `logs/compare_plainy_member_plain.txt`, `logs/compare_plainy_detail.txt` and `logs/compare_plainy_lofo_detail.txt`
- `logs/plainy_full.txt`

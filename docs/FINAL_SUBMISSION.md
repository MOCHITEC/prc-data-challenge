# Final submission: the file, how it was built, how to reproduce it

Team **resilient-hedgehog**, PRC Data Challenge 2026. Written 2026-10-14.

This is the one authoritative page about the final file. `README.md` §3, the runbook
(`docs/final_phase_2026-10-11/RUNBOOK_final_submit.md`) and the release checklist point here. The notes under
`handoff/` are dated working notes; they recommend other files, which were not chosen (§7).

Every number below cites a committed path, or a row-level check rerun on 2026-10-14 (§2.4).

## 1. The final file

The file's identity is **defined by this table**. Copies of it:
- the commands on this page (§1, §5) repeat the md5;
- the runbook §0 copies it into shell variables;
- README §3 and the checklist §1.2 name the commit.

If a different file is chosen, change this table first. `grep -rn be8d89dc README.md docs/` then finds every copy.

| | |
|---|---|
| file | `handoff/2026-10-13/final_from_v48.parquet` **as committed in `25f6246`** (`25f624685ec1d26f8309f9d7980588b834c36c0f`, 2026-10-13 02:24Z, branch `claude/laughing-cray-vvwh0k`; contained in `main`) |
| md5 | **`be8d89dcf79bd59a6f43d80193c95685`** |
| size, rows | 2,145,836 bytes; 670,790 rows; schema `MVT_ID_mvt: double, TAXITIME_SEC_mvt: int32` |
| get it | `git show 25f6246:handoff/2026-10-13/final_from_v48.parquet > final_from_v48.parquet` |
| upload | Planned once, as `resilient-hedgehog_final.parquet`, before 2026-10-14T10:00Z. **It had not been uploaded when this page was written.** The upload that was made is recorded in `experiments/submissions.jsonl` (lines with `"kind": "final"`, written by `prc.submit --final`). If the ledger shows a different md5, trust the ledger: this page is then out of date. |

The path holds different content in other commits:
- `558c736` has an earlier version, md5 `c7ffa31e…`. It differs only in June (§2.4).
- The file was deleted at the branch tip (`6adbcac`).

Always take the file from `25f6246`.

**Validation.** `python -m prc.validate_final <file> --template final_submitting.parquet --ranking
final_ranking.parquet --expect-md5 be8d89dcf79bd59a6f43d80193c95685` prints **PASS** and exits 0. It was rerun on
2026-10-14 against the organisers' files (`final_submitting.parquet` md5 `2c1f4072…`, `final_ranking.parquet` md5
`1ce0c6ee…`; the runbook §3 lists both).
- The ids equal the template's in set and order.
- 0 nulls and 0 negatives. Min 0 s, max 112,725 s.
- 66 rows are above 10,800 s. The validator reports this as a warning, not a failure.
- All 10 airports have rows in all 4 months.

| month | rows | p50 (s) | p99 (s) | max (s) | rows > 10,800 s |
|---|---|---|---|---|---|
| Jan | 152,719 | 970 | 3,095 | 57,025 | 15 |
| Feb | 144,158 | 922 | 2,198 | 42,724 | 11 |
| Jun | 181,791 | 926 | 2,066 | 69,165 | 9 |
| Jul | 192,122 | 936 | 2,138 | 112,725 | 31 |

## 2. How each month's rows were produced

| rows | where the values come from | ADS-B residual layer in those rows |
|---|---|---|
| Jan + Jul, 344,841 | copied unchanged from leaderboard submission v48 | v48's: the v33 → v48 delta chain, which includes the 12-month appear tier and the EDDF coverage delta |
| Feb, 144,158 | the chain of §2.2 | our rebuilt layer (`adsb_layer.py`); no appear term |
| Jun except 2026-06-11, 175,692 | the chain of §2.2 | our rebuilt layer; no appear term |
| Jun 2026-06-11, 6,099 | the chain of §2.2 with the A12 layer in step 3 (§2.3) | A12 (`final_adsb.py`) |

### 2.1 January and July: v48, unchanged

- **Source.** `handoff/2026-10-08/C15_appear12_cov26eddf.parquet`, md5 `8ee6679f43ead60db1a1aec7876caf09`,
  344,841 rows. It is leaderboard submission **v48**, uploaded 2026-10-11 14:47Z. Its public leaderboard score is
  **234.4099**, the team's best (`docs/DEVELOPMENT_HISTORY.md` §1).
- **Check.** All 344,841 Jan/Jul rows of the final file equal it; 0 differ (§2.4).
- **What v48 is.** v48 = C15 + the 12-month appear delta + the EDDF coverage delta (`handoff/2026-10-08/ON_C15.md`).
  - C15 is submission v47 (234.4831). It is v42 plus the fl_/seq_ consistency fix, 3-seed L6 members and ucmix.
  - The EDDF delta is explained in `README.md` §3.
  - v48 is the end of a chain of per-submission delta files, v33 → v48. No single script builds it. See
    `docs/SOLUTION_OVERVIEW.md` §7, `docs/batch_2026-10-06/README.md` and `docs/final_2026-10-11/adsb_layer_README.md`
    §1.
- **Clock.** v48 was built on the clock of the old `ranking.parquet`. `final_ranking.parquet` moves the Jan/Jul MVT
  times by up to ±12 s: 316,651 of the 344,841 rows (runbook, "Facts measured on 10-13"). The final file does not
  re-anchor these rows (§3).

### 2.2 February and June: the chain

The recipe is the top of `handoff/2026-10-13/README.md` as of `25f6246` (in Japanese; the rest of that note is
translated in §8). In English:

1. **Members.** Each member was trained on the final raw data:
   `python -m prc.pipeline --config <config> --skip-cv --raw-dir <final raw dir> --work-dir <final work dir>`.
   - The code was peaceful-babbage `6a97bf0`, which `main` contains.
   - The matched rows of the feature cache had their `fl_*` / `seq_*` columns replaced by values built on the
     2025 + 2026 union (`analysis/local_audit_fix2026.py` run on the final frame). The result is committed as
     `handoff/2026-10-13/inputs/feat_rank_extra_union_final.parquet`.

   | member | config (`configs/`) | seeds | committed predictions (`handoff/2026-10-13/members/`) |
   |---|---|---|---|
   | a2o | `main_flseq_osm_nolirf_l2_opt_l6_cmix.yaml` | 3 | `main_flseq_osm_nolirf_l2_opt_l6_cmix.parquet` |
   | pao | `main_flseq_osm_nolirf_l2_pa_opt_l6.yaml` | 3 | `main_flseq_osm_nolirf_l2_pa_opt_l6.parquet` |
   | p2o | `main_flseq_osm_nolirf_plainy_l2_opt_l6.yaml` | 3 | `main_flseq_osm_nolirf_plainy_l2_opt_l6.parquet` |
   | c2 | `main_flseq_osm_nolirf_cb_l2.yaml` | 1 (seed 42) | `main_flseq_osm_nolirf_cb_l2.parquet` |
   | m4 | `main_flseq_osm_nolirf_l2_mlp_l6.yaml` | 3 | `main_flseq_osm_nolirf_l2_mlp_l6.parquet` |

   The training logs are in `docs/final_2026-10-11/logs/`.
2. **Blend.**
   - Matched rows (AOBT_3 present, 661,713 rows): `((a2o + pao)/2 + p2o + c2 + m4) / 4`.
   - Unmatched rows (9,077): model B from the a2o run.
   - The result, for all 4 months, is `handoff/2026-10-13/pre_adsb_base_blend5_final.parquet` (md5
     `38edd5a6a6ab0328de841936cffa907a`, float). It equals the blend of the five committed member files exactly (§2.4).
3. **ADS-B residual layer**, our rebuild of v47's layer: `docs/final_2026-10-11/adsb_layer.py apply --base <blend>
   --out <out>`.
   - Components: G12A, G7A (LFPG), GL4 (LIRF), AIR, POS and RULE (`docs/final_2026-10-11/adsb_layer_README.md` §1).
   - The correction is clipped to ±1800 s and is 0 at LTFM.
   - Models: `docs/final_2026-10-11/adsb_models/`. Per-day features: `handoff/2026-10-13/adsb_store/`.
   - The layer has **no appear term**.
   - It covers all 28 February days and 29 of the 30 June days. The June ingest had no data for 2026-06-11 (the
     02:30 update at the end of `handoff/2026-10-13/README.md` @ `25f6246`).
4. **Overlays and LOBT clip:** `docs/final_2026-10-11/apply_overlays.py --laneB --lobt-clip --copy-feat unfixed`
   (`docs/final_2026-10-11/overlays_README.md`). In order:
   - lane-B deltas: NM-blind at weight 0.25 on unmatched rows outside LIRF, and ucmix on LIRF unmatched rows with
     0 < MVT − SCHED < 8 h. The deltas are `handoff/2026-10-13/inputs/{nmb,ucmix}_laneB_delta_final.parquet`.
   - LIRF unmatched copy band: rows 7 to 13.5 h late are predicted as `MVT − SCHED`.
   - LIRF unmatched time-zone class: a per-prefix q blend.
   - LIRF matched copy overlay: `p' = p + 0.4·q·(MVT − SCHED − p)`. The q used is
     `handoff/2026-10-13/inputs/copyq_LIRF_final_unfixed.parquet`.
   - LOBT clip, last: each prediction is clipped to `[MVT − LOBT − 3606, MVT − LOBT + 3606]` s.
5. **Floor at 0, round, int32.**

The output of steps 1–4, for all 4 months, is committed as
`handoff/2026-10-13/inputs/orig_chain_blend_adsb2_ovl.parquet` (md5 `3fe8fb33f6ef9724558015d90b756cd5`, listed in
`handoff/2026-10-13/MD5SUMS`). Its February rows and its June rows other than 2026-06-11 become the final rows after
step 5.
- Its per-row ADS-B detail is `handoff/2026-10-13/inputs/orig_chain_adsb_detail.parquet` (columns `b`, `comp`, `c`,
  `new`, …). Its column `b` equals `pre_adsb_base_blend5_final.parquet` on all 670,790 rows (rerun 2026-10-14).
- The file was committed in `68e7693` (2026-10-13 20:32Z), after `25f6246`, as the "old chain" input of
  `docs/final_2026-10-11/assemble_step.py`. It still reproduces the final rows exactly (§2.4).

### 2.3 June 11

- 2026-06-11 has 6,099 rows. They went through the same blend, then the **A12 layer** (`docs/appear_tier_2026-10-05/final_adsb.py`, beautiful-davinci `6b9d29e`, with the store `handoff/2026-10-11/adsb`), then the same step 4.
- The output, for all 4 months, is committed as `handoff/2026-10-13/inputs/blend_A12_ovl.parquet` (md5
  `3221ce5bfb468f702575556e183acef7`). It was committed in `e301db5` (2026-10-13 19:47Z), after `25f6246`.
- The file of `558c736` used this A12 path on every June day.
- The A12 layer changes little on this day. A later audit found that the A12 store holds only 78 appear rows for
  2026-06-11, so the layer is "effectively 0" there (`handoff/2026-10-13/README.md`, section dated 2026-10-13 20:15
  UTC, item 1).
- On those 6,099 rows the final file differs from the rounded pre-ADS-B blend on 605 rows. That count includes the
  overlays (§2.4).

### 2.4 Row-level checks (rerun 2026-10-14)

Each line is a direct row-by-row join, on `MVT_ID_mvt`, of the named committed files with the final file. Month and
day come from the DEP rows of `final_ranking.parquet`. These were one-off polars checks made for this page and are
not committed. The full re-assembly can be rerun with the committed script (§5.2).

| comparison | result |
|---|---|
| Jan + Jul vs v48 (`handoff/2026-10-08/C15_appear12_cov26eddf.parquet`) | 0 of 344,841 rows differ |
| Feb vs the `558c736` file (md5 `c7ffa31e…`) | 0 of 144,158 differ |
| Jun vs the `558c736` file | 142,281 of 181,791 differ, at 9 airports (none at LTFM) and on 29 days (none on 06-11). The RMS change over all June rows is 44.6 s, the value in the `25f6246` README update |
| Feb + Jun (not 06-11) vs `inputs/orig_chain_blend_adsb2_ovl.parquet`, floored at 0 | 0 differ. One February LSZH row is −10 in the chain file and 0 in the final file |
| Jun 06-11 vs `inputs/blend_A12_ovl.parquet` | 0 of 6,099 differ |
| Jun 06-11 vs `inputs/orig_chain_blend_adsb2_ovl.parquet` | 78 rows differ, all at EDDM. So 06-11 must be taken from the A12 file |
| `pre_adsb_base_blend5_final.parquet` vs the blend of the five member files (§2.2 step 2) | max abs difference 0 on all 670,790 rows |
| Jun 06-11 vs `round(pre_adsb_base_blend5_final.parquet)` | 605 of 6,099 differ; this count includes the overlays |
| the whole file re-assembled from v48, the chain file and the A12 file (§5.2) | **0 of 670,790 rows differ**; `scripts/final/assemble_final_v48.py` writes a byte-identical file (same md5) |

## 3. What is not in the file

Each item was checked against the file and the dated notes.

- **No "+1×δ" re-anchor of January/July.**
  - The Jan/Jul rows equal v48 (§2.4). v48 was uploaded on 2026-10-11.
  - The re-anchor script `docs/final_2026-10-11/add_delta_janjul.py` was added later, in `0a6e769` (2026-10-13
    15:05Z).
  - `handoff/2026-10-13/CANDIDATES.md` (section dated 23:00) and commit `a94067b` record a decision, attributed to
    the user on 2026-10-13, to apply +1×δ to whichever file was finally chosen. The file chosen on 2026-10-14 does
    not carry it: its Jan/Jul rows equal v48.
- **No February/June appear increment** `c_A12 − c_P12`.
  - February is identical to the `558c736` file. `handoff/2026-10-13/CANDIDATES.md` (section dated 01:40) describes
    that file's February as "our own ADS-B layer only; no appear layer". Its section dated 02:40 describes the
    `25f6246` file as "neither February nor June has the appear layer yet".
  - The increment was first added to a different file, in `a37e5bb` (2026-10-13 14:37Z).
  - In the final file, February and June (except 06-11) equal the chain output, which has no increment (§2.4).
  - 06-11 carries the A12 layer itself, which has appear features, but only 78 appear rows exist for that day
    (§2.3).
- **None of the later candidates.** The file was committed at 02:24Z on 2026-10-13, before any of them. In
  particular CatBoost is one seed (§2.2), not 3 or 5.

  | later candidate (not submitted) | md5 | what it adds | where described |
  |---|---|---|---|
  | `final_from_v48_reweight.parquet` (withdrawn) | `dd361cae…` | a2o:pao 0.2:0.8 outside LIRF | `handoff/2026-10-13/README.md` |
  | `final_from_v48_reweight_ltfm.parquet` (withdrawn) | `155f479a…` | + cmix at LTFM | same |
  | `final_from_v48_cb3.parquet`, `final_from_v48_cb3_appear.parquet` | `6bbcb6c8…`, `6aa805f4…` | 3-seed CatBoost; + Feb/Jun appear increment | same |
  | laughing-cray "v2" `final_from_v48_cb3_appear_cat_c13_v2.parquet` | `ce79384d…` | + cat_tight a2o/pao, exact C13, own layer on 06-11 | same |
  | laughing-cray "v3" `final_v3_cb5.parquet` | `838d5f41…` | + 5-seed CatBoost | same |
  | laughing-cray "v4" `final_v4_cb5_mlp5.parquet` = `final_recommended.parquet` | `d0ce32b6…` | + 5-seed MLP | same, section "★★ 最終版" |
  | `optional_janjul_reanchor_plus1delta.parquet` | `9d0a5bf1…` | v4 + +1×δ | same |
  | beautiful-davinci `final_v5_plus_febjun_appear.parquet` (withdrawn 06:20Z) | `c0d4a143…` | this file + Feb/Jun appear increment | `handoff/2026-10-13/CANDIDATES.md` |
  | beautiful-davinci `final_v4_gatefull_plus1delta.parquet` and 3 variants | `b7e8298d…` etc. | v4 + LIRF copy-overlay gate (+ +1×δ) | `handoff/2026-10-14/README.md` |

**Naming warning.** The labels "v4" and "v5" mean different files in different notes.
- In `handoff/2026-10-13/CANDIDATES.md`, "v5" is **this file** (`25f6246`, `be8d89dc…`). The "v4 base" there is the
  `558c736` file (`c7ffa31e…`), which first appeared as `cand_v4a_…` in `0220c69`.
- In `handoff/2026-10-13/README.md` and `handoff/2026-10-14/README.md`, "v2/v3/v4" are the later laughing-cray
  candidates in the table above.
- Leaderboard numbers (v1–v49) are a third series (`docs/DEVELOPMENT_HISTORY.md`).

This page names files by commit and md5 only.

## 4. Why this file

The team chose the v48-based file on 2026-10-14. The notes do not record a reason; this page does not supply one.
Its January/July rows are the team's best leaderboard file, unchanged. Its February/June rows come from the
documented chain (§2.2), without the later additions of §3.

## 5. Reproduction

### 5.1 Get and check the file (about 2 minutes; the runbook §2–§5 has the full commands)

```bash
git show 25f6246:handoff/2026-10-13/final_from_v48.parquet > final_from_v48.parquet
md5sum final_from_v48.parquet        # be8d89dcf79bd59a6f43d80193c95685
python -m prc.validate_final final_from_v48.parquet \
  --template data/raw/final/final_submitting.parquet --ranking data/raw/final/final_ranking.parquet \
  --expect-md5 be8d89dcf79bd59a6f43d80193c95685
```

The competition files come from the organisers' bucket (`README.md` §5).

### 5.2 Re-assemble the file from committed parts (byte for byte)

`scripts/final/assemble_final_v48.py` rebuilds the final file from three committed files and the organisers'
`final_submitting.parquet` / `final_ranking.parquet`. Run it from the repository root:

```bash
python scripts/final/assemble_final_v48.py --out outputs/final/final_from_v48.parquet \
  --expect-md5 be8d89dcf79bd59a6f43d80193c95685 --reference final_from_v48.parquet   # --reference: the file from 5.1
```

- **Inputs.** By default it reads `data/raw/final/final_submitting.parquet` and `data/raw/final/final_ranking.parquet`.
  It stops if any input md5 differs from the committed value:

  | rows | input | md5 |
  |---|---|---|
  | Jan + Jul, 344,841 | `handoff/2026-10-08/C15_appear12_cov26eddf.parquet` (v48) | `8ee6679f…` |
  | Feb, and Jun except 2026-06-11 (144,158 + 175,692) | `handoff/2026-10-13/inputs/orig_chain_blend_adsb2_ovl.parquet` (own layer) | `3fe8fb33…` |
  | Jun 2026-06-11, 6,099 | `handoff/2026-10-13/inputs/blend_A12_ovl.parquet` (A12 layer) | `3221ce5b…` |

  The month and day of each row come from the UTC date of its DEP row's `MVT_TIME_UTC_mvt` in `final_ranking.parquet`.
- **What it does.** It picks the source per row, floors at 0 and writes Int32 rows in the order of
  `final_submitting.parquet`. The floor moves exactly one row: February, LSZH, `MVT_ID_mvt` 204314027, −10 → 0.
- **Result**, rerun on 2026-10-14: md5 `be8d89dcf79bd59a6f43d80193c95685`, the same bytes as the committed file;
  0 rows differ from it.
- **The earlier version.** `--june-layer a12` writes the `558c736` file instead (md5
  `c7ffa31edc2a4de5f1837682d90e14cb`, same bytes; rerun 2026-10-14).
- **What it does not rebuild.** It does not refit the members, recompute the blend, or rerun the ADS-B layers, the
  overlays or the LOBT clip. Those are already inside the two chain files, and the script takes them as committed
  (§5.3, §6).

The independent check of §2.4 does the same comparison without the script. It was written for this page before
the script existed, and it gave the same result (0 rows differ).

### 5.3 Rebuild the parts from raw data

This means running the chain of §2.2 again, upstream of the files that §5.2 consumes. Two things are needed that
the repository does not hold:
- **The overlay stage needs the organisers' data and a feature cache.** It cannot be rerun from the committed blend
  alone. `docs/final_2026-10-11/apply_overlays.py` reads the organisers' 2025 `training_*.parquet` files (LIRF
  unmatched time-zone rule) from `--raw-dir`. It reads `feat_train.parquet` and `feat_rank*.parquet` (LIRF copy q,
  LOBT clip) from `--feat-dir`. Those are the feature cache that `prc.pipeline` builds; it is not committed. The
  competition files are not redistributed here (`README.md` §5).
- **The members need a fresh fit.** `src/prc` does not save LightGBM or CatBoost models
  (`docs/final_phase_2026-10-11/IMPACT.md` §4(c)). A refit will not give identical numbers; how close it comes was not
  measured. The committed member predictions (§2.2) and the committed blend let the rebuild start at step 2 or 3
  without refitting.

The chain scripts were run on other machines and hard-code those machines' paths. The table maps them to the
committed copies. It translates the path table at the end of `handoff/2026-10-13/README.md`, keeping only the rows this file needs.

| path in the scripts | committed copy |
|---|---|
| `/home/user/pb-work/outputs/final/*.parquet` (member predictions) | `handoff/2026-10-13/members/` (the five files in §2.2) |
| `scratch/agents/audit/feat_rank_extra_union.parquet` (union fl_/seq_ columns) | `handoff/2026-10-13/inputs/feat_rank_extra_union_final.parquet` |
| `/home/user/prc-final/assemble/base_blend.parquet` (blend) | `handoff/2026-10-13/pre_adsb_base_blend5_final.parquet` |
| `adsb_layer.py` `SCR` (`…/scratchpad/adsb`): `learn/feat_2026`, `learn/feat_pos_2026`, `pos/dwell_2026`, `pos/v2026.*_pos`, `agents/rescat/{codes,vocab}` | `handoff/2026-10-13/adsb_store/{learn,pos,rescat}/` |
| `adsb_out/feat`, `adsb_out/pos` (per-day Feb/Jun features) | `handoff/2026-10-13/adsb_store/{feat,posday}/` |
| `adsb_out/models` | `docs/final_2026-10-11/adsb_models/` |
| `overlays/out/{nmb,ucmix}_laneB_delta_final.parquet` | `handoff/2026-10-13/inputs/` |
| `overlays/val/copyq_final_unfixed.parquet` (LIRF copy q) | `handoff/2026-10-13/inputs/copyq_LIRF_final_unfixed.parquet` |
| `/home/user/prc-final/assemble/blend_adsb2_ovl.parquet` (chain output, steps 1–4) | `handoff/2026-10-13/inputs/orig_chain_blend_adsb2_ovl.parquet` |
| `/home/user/prc-final/assemble/blend_A12_ovl.parquet` (A12 path, used for 06-11) | `handoff/2026-10-13/inputs/blend_A12_ovl.parquet` (the name is in `docs/final_2026-10-11/assemble_step.py`; the match is from §2.4) |
| `/home/user/bd-work/handoff/2026-10-08/C15_appear12_cov26eddf.parquet` (v48), `final_adsb.py`, `handoff/2026-10-11/adsb` | the same paths in this repository (beautiful-davinci `6b9d29e`, contained in `main`) |

Notes on the parts:
- `handoff/2026-10-13/adsb_store/feat/` also holds `feat_2026-06-11.parquet`. It was committed later, in `68e7693`
  (2026-10-13 20:32Z), for a later candidate, and the final file does not use it. To rebuild this file, leave it
  out, or take the 06-11 rows from the A12 path.
- `docs/final_2026-10-11/adsb_layer.py`, `apply_overlays.py` and `laneB_deltas.py` are unchanged since `25f6246`.
  `docs/appear_tier_2026-10-05/final_adsb.py` (md5 `2d777569ee77317b640409d39478e53a`) and
  `handoff/2026-10-11/adsb` are unchanged since `6b9d29e`.
- Members: `prc.pipeline.load_raw` matches `ranking.parquet` / `submitting.parquet` by exact name. So the final raw
  directory must hold the final files under those names. The hand-off notes name the directory
  (`--raw-dir data/raw_final`, `handoff/2026-10-13/README.md`, section dated 20:15) but not the commands that filled
  it.
- Rebuilding the ADS-B inputs from scratch means downloading adsb.lol `globe_history` archives (§6).

## 6. Honest limits

- **January/July cannot be regenerated exactly.** They are copied from v48 (§2.1).
  - v48 is the end of a chain of delta files, v33 → v48, each applied to the previous submission file. The bases
    were recovered from those files by a fixed point (`docs/final_2026-10-11/adsb_layer_README.md` §1).
  - No branch holds an assembly script for C11/C12/C15, and several inputs were only on other machines
    (`docs/final_phase_2026-10-11/IMPACT.md` §4(c)).
  - Running the documented chain on Jan/Jul gives values that differ from v48 by RMS 53.2 s, median 16 s
    (`handoff/2026-10-13/README.md`, validation table). The same note attributes the median mainly to the ±12 s MVT
    shift in `final_ranking` and to the approximations of the rebuilt ADS-B layer.
- **The months are not built the same way.** January/July carry the 12-month appear tier and the EDDF coverage
  delta. February/June do not (§3).
  - On held-out Feb + Jun 2025, the appear tier (A12) measured −1.43 s [−2.00, −1.06] against the same residual
    without it (P12) (`docs/appear_tier_2026-10-05/results.md` §21).
  - No measurement says what leaving it out costs in the 2026 February/June rows.
- **The exact reproduction starts from post-layer files.** `scripts/final/assemble_final_v48.py` reproduces the file
  byte for byte (§5.2), but it consumes v48 and the two chain files as committed.
  - Both chain files were committed after `25f6246`: the A12 chain in `e301db5`, the own chain in `68e7693`.
  - Both committed versions of the path rebuild from them byte for byte: `be8d89dc…` and `c7ffa31e…`.
  - The steps inside the chain files (blend → ADS-B layer → overlays → LOBT clip) can be rerun only with the
    organisers' 2025 training files and a rebuilt feature cache (§5.3).
- **Members must be retrained** to rebuild from raw data. No LightGBM or CatBoost model is saved (§5.3). The
  committed member predictions remove the need to refit for every step from the blend onwards.
- **ADS-B download size.** The adsb.lol archives are large.
  - Measured: 194.9 GB for the 57 Jan/Jul 2026 days downloaded, 2.20 to 4.28 GB per day
    (`docs/final_phase_2026-10-11/IMPACT.md` §4(a), from
    `docs/competitor_research_2026-09-21/results/adsb-full-ingest-candidate.md`).
  - Measured: 166.11 GB for Jan/Jul 2025, 1.68 to 3.85 GB per day
    (`docs/competitor_research_2026-09-21/results/adsb-ingest-2025-labelled.md` §3).
  - The residual models use all 12 months of 2025. No total download size for the full rebuild was measured.
- **2026-06-11 has almost no ADS-B correction** (§2.3). adsb.lol has a data gap that day
  (`handoff/2026-10-13/CANDIDATES.md`, part 1).
- **Absolute paths.** The final-chain scripts hard-code the paths of the machines they ran on; §5.3 maps them. The
  scripts were not edited.

## 7. About `handoff/`

`handoff/<date>/` folders are **dated working notes** that parallel sessions passed to each other, mostly in
Japanese. They are kept as written, as the record.
- Several of them name a "recommended" (推奨) file, for example:
  - `handoff/2026-10-13/README.md`: `d0ce32b6…`;
  - `handoff/2026-10-14/README.md` and `handoff/2026-10-13/CANDIDATES.md`: `b7e8298d…`.
- `handoff/2026-10-13/CANDIDATES.md` records a "+1×δ" decision.
- **None of those recommendations was submitted.** The final file is the one in §1.
- Where a hand-off note and this page disagree about the final file, this page is right. Exception: the upload
  ledger `experiments/submissions.jsonl` overrides both (§1).

## 8. English translation of the hand-off note for this file

This is `handoff/2026-10-13/README.md` **as of `25f6246`**: `git show 25f6246:handoff/2026-10-13/README.md`. The
note was written by the laughing-cray thread on 2026-10-13, at 00:50 with an update at 02:30 UTC, and is in
Japanese. The recipe part is in §2.2 above and the path table in §5.3. The rest is translated here. Later sections
of the same file, added after `25f6246`, are about other candidates (§3, §7).

**Header.** It follows the team instruction (`docs/final_phase_2026-10-11/NO_FINAL_UPLOAD.md`). The note says:
- this thread has uploaded nothing to the bucket;
- the bucket's `resilient-hedgehog_final.parquet` (10-09) was not made by this thread;
- v48 and v49 were not submitted by this thread.

**Files.**

| file | md5 | content |
|---|---|---|
| `final_from_v48.parquet` | `be8d89dcf79bd59a6f43d80193c95685` | Jan/Jul = v48's file (old ranking clock). Feb = the chain below (own ADS-B layer). Jun = the chain below, "ADS-B layer = the A12 layer". Validate OK |
| `pre_adsb_base_blend5_final.parquet` | `38edd5a6a6ab0328de841936cffa907a` | the 5-member blend before the ADS-B layer (4 months, float); the base b to pass to `final_adsb.py` and similar |

The June cell was not updated when the md5 was. The 02:30 update below is what the file contains: §2.4 checks it.

**Where it ran.** Working directory `/home/user/prc-final`; member outputs in `/home/user/pb-work/outputs/final`.

**ADS-B layers** (two kinds):
- **Own:** `docs/final_2026-10-11/adsb_layer.py apply`. It rebuilds the composition of v47's layer (G12A / G7A / GL4 /
  AIR / POS / RULE, c = 0 at LTFM). The note lists it for Jan, Feb and Jul.
- **A12:** `docs/appear_tier_2026-10-05/final_adsb.py` from the beautiful-davinci thread, with
  `handoff/2026-10-11/adsb`. It covers all 4 months.

**Overlays.** `apply_overlays.py --laneB --lobt-clip --copy-feat unfixed`, in this order:
- lane-B: NM-blind (w 0.25) and ucmix;
- LIRF: copy band, time-zone class, matched copy overlay (0.4·q);
- the LOBT clip (±3606 s) last.

**Validation** (on Jan/Jul rows, against v47 and v48):

| compared | RMS difference | median difference |
|---|---|---|
| chain (own layer) vs v47 | 52.5 s | 16 s |
| chain (own layer) vs v48 | 53.2 s | 16 s |
| chain (A12 layer only) vs v48 | 68.8 s | 16 s |
| v48 vs v47 | 22.6 s | 6 s |

- "The own layer is closer to v47 / v48." The note's explanation, marked by the note itself as a belief: the A12
  layer alone lacks posjoin, LIRF matched (GL4), air-only and similar parts.
- "The median difference of 16 s comes mainly from the ±12 s shift of MVT in final_ranking" and from the
  approximations of the rebuilt ADS-B layer.
- "The script itself was confirmed to reproduce v47 on every row on the old ranking data." The note does not say
  which script. `docs/final_2026-10-11/overlays_README.md` states that `apply_overlays.py` "reproduces v47 exactly".
- At 00:50 June had no own layer yet, because the adsb.lol June ingest was still running. The note planned a "v5"
  with June on the own layer, "+ (c_A12 − c_P12) if needed".

**Update (2026-10-13 02:30 UTC): June switched to the own ADS-B layer as well.**
- The adsb.lol June data were ingested for 29 of the 30 days; 2026-06-11 had no data.
- The June rows were replaced with the own layer, the same composition as v47 / v48.
- Only the 2026-06-11 rows keep the A12 layer (the beautiful-davinci 4-month version, which has data for that day).
- January, February and July are unchanged. In June, 142,281 rows changed; the RMS of the change is 44.6 s.
- Validate OK. md5 `be8d89dcf79bd59a6f43d80193c95685`.

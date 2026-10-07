# C12 (v42) + 12-month appear delta: candidate, not submitted (2026-10-07)

Current best on the board: v42 = C12 = 236.4369 (top-10 % cutoff 236.0475). This note adds beautiful-davinci's
12-month appear delta to C12. Nothing was uploaded.

## 1. Status: has anyone applied, bundled or submitted it?

No. Checked at 2026-10-07 ~04:15 UTC.

- beautiful-davinci after dd6e60d:
  - `36f49b7` "Stand map from 12 months of 2025 ADS-B: pre-registration (before any CV)"
  - `9f506bd` "Stand map from 12 months: stand_map12.py (leave-own-month-out maps, rand12 control) and cv_standmap12.py"
  - Both are new stand-map work. Neither applies, bundles or submits the appear12 table.
- laughing-cray after 0b16e18:
  - `ace5f58` "Isolation submissions v43 (apron), v44 (C10), v45 (peaceful-babbage member deltas)"
  - `b6c627e` "Board attribution of v42 (236.4369): C10 -1.36 s, PB members -0.42, apron -0.11, NM-blind ~-0.43; encoding cross-fit rejected"
  - `docs/batch_2026-10-06/README.md` does not mention appear12.
- `experiments/submissions.jsonl`, highest version on each branch:
  - laughing-cray: 45 (v43–v45 are isolation files built on v40).
  - model-b-and-arrival-features: 42 (`v40-plus-r6cbopt-delta`). C12 later overwrote this object.
  - beautiful-davinci: 39. Every other branch is at 35 or lower.
  - No entry anywhere contains the appear12 table.
- Leaderboard API (`teamName=resilient-hedgehog`, 44 items):
  - v42 236.4369, v44 237.4070, v45 238.3508, v43 238.6626, v40 238.7712, v41 239.5174.
  - Nothing above v45. v43–v45 contain no appear12.
- The only plan on record is in dd6e60d / APPEAR12.md: "合意どおり他の施策と束ねて出す" (bundle with other items).

## 2. How the delta is defined (dd6e60d, `apply_delta12.py`, `handoff/2026-10-06/APPEAR12.md`)

- **Formula:** `delta = (c_A12 − c_P12)(b) − (c_A4 − c_P4)(b)`. It applies only on rows that have appear columns
  (ADS-B ground fix plus callsign join), and LTFM gets 0. c_A4 and c_P4 are v39's 4-month models; A12 and P12 are
  fit on 12 months.
- **Base:** v39. b was recovered by a fixed point, `b = file − c_P4(b) − (c_A4 − c_P4)(b)`, using v38's 4-month
  plain residual.
- **Shipped form:** a fixed additive per-row table, `appear12_delta.parquet` (MVT_ID_mvt, integer delta).
  - 240,983 rows, no duplicate ids, all ids in the template.
  - Mean −1.43 s, sd 26.2 s, range −625 to +603.
  - Rows include 9,597 at LIRF.
  - Check: v39 + table reproduces `v39_appear12_on_v39.parquet` exactly.
- **Intended use (APPEAR12.md):** add the table as-is to any file that still carries v39's appear delta
  (v39, v40, v41, v42). Do **not** re-run `apply_delta12.py` on v40 or later files: their plain residual is the
  12- or 7-month one, and recovering b with the 4-month one would be wrong.
- **Sensitivity to b:**
  - Switching the b recovery between P4 and P12 moves the delta by RMS 15.7 s (the delta's own RMS is 25.9 s).
  - APPEAR12.md says a b change of "a few to ~10 s" (数秒〜十数秒) has a small effect.

## 3. How it interacts with C12

C12 = v40 + C10 + apron swap + L6/cmix members + NM-blind + MLP-L6, then the LOBT clip
(`docs/batch_2026-10-06` on laughing-cray).

- **v40 residual swaps and apron.** The 4→7→12 swaps and the apron swap (`local_adsb_rescat.py apply12`) are
  additive replacements of c_plain: `new = file + c_new(b) − c_old(b)`. They leave the term (c_A4 − c_P4) in the
  file, so C12 still carries v39's appear delta. The table therefore stays on its intended footing.
  - Caveat: A12 − P12 was fit against plain P12, not P12 + apron. Apron and appear may overlap. Apron alone gained
    only −0.11 s on the board (v43).
- **Base changes from C10, members and NM-blind.** C10's fl_/seq_ fix, the L6/cmix and MLP-L6 members and NM-blind
  all move b.
  - Estimated b shift on table rows: (C12 − v40) − (v43 − v40), using the isolation files.
  - That shift is median 8 s, p90 28 s, p99 85 s, RMS 22.9 s.
  - By part (RMS on table rows): apron 30.3, C10 8.0, PB members 20.5, rest (NM-blind + MLP-L6 + interactions +
    clip) 5.8.
  - So b moved by about as much as the P4/P12 recovery difference behind the 15.7 s figure. The table is evaluated
    at v39's b, not at C12's b.
- **Overlays.** On these rows C12's value comes from an overlay, not the residual chain, so the delta was not added:
  - 2 LIRF copy-band rows (rule C of `local_lirf_artefacts_apply.py`; C12 = MVT − SCHED exactly).
  - 1 LIRF time-zone-class row (rule B, q > 0).
  - 29 rows sitting on the LOBT bound in C12.
- **After the delta:**
  - The LOBT clip was re-applied: 15 rows were pulled back to the bound.
  - 1 row went negative and was reverted to C12 (C12 13, delta −42).
- **Verdict:**
  - Mechanically well-defined: an additive table joined by id. It is also the authors' stated use ("v42 など").
  - Exact evaluation at C12's b is **not recoverable here**. It needs beautiful-davinci's A4/P4/A12/P12 models and
    2026 appear features, which are not on this machine.
  - An approximation is recoverable in that session: b_C12 ≈ b_v39 + shift. b_v39 comes from their fixed point;
    shift = (v42 − v40) − (v43 − v40), using the bucket files.

## 4. Candidate

- **File:** `outputs/cand_c12_appear12.parquet`, built by
  `docs/competitor_research_2026-09-21/scripts/c12_plus_appear12.py <v42> <appear12_delta> <out>`.
- **Inputs:**
  - v42 was read from the team bucket: `resilient-hedgehog_v42.parquet`, etag 8cab2177…, modified
    2026-10-07 02:06:20, which matches C12's upload time.
  - The table came from `origin/claude/beautiful-davinci-ehugk4:handoff/2026-10-06/appear12_delta.parquet`.
- **Validation:** 344,841 rows; schema (MVT_ID_mvt f64, TAXITIME_SEC_mvt i32) equals the template; ids and order
  equal `data/raw/submitting.parquet`; 0 nulls; 0 negatives (1 reverted to C12); min 0.
- **Change vs C12:** 240,950 rows changed, all inside the table. RMS over changed rows 26.24 s; RMS over all rows
  21.93 s; Σd²/N 481; max |d| 625 s; mean −0.99 s.
  - On 240,935 table rows the change equals the table value exactly.
  - The other 48 table rows are the 32 excluded, 15 re-clipped and 1 negative-reverted rows.
- **Expected effect:**
  - Authors' estimate: −0.85 to −0.95 s on LB, from CV gain of 0.83–0.94 × Σd²/N on a v39/v40 base.
  - On C12 the result is less certain, because of the b shift and the possible apron overlap.
  - 236.44 − 0.9 ≈ 235.5 would clear the 236.0475 cutoff only if most of the gain carries over.

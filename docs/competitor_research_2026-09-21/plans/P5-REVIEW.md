# P5 — adversarial review of `P5-physics-lower-bound.md`

Reviewer: adversarial review pass, 2026-09-21. Target: `docs/competitor_research_2026-09-21/plans/P5-physics-lower-bound.md` (829 lines).
Scope: read-only analysis against `data/`, `outputs/`, `experiments/`, `src/`. Nothing under `src/`, `configs/` or `outputs/` was modified. No submission was built or sent. Scratch scripts: `/tmp/rev/r1.py` … `/tmp/rev/r30.py`.

Marking: **[S]** = I measured or read it, with the command; **[X]** = my inference on [S] numbers.

---

## (a) VERDICT — **PROCEED WITH CHANGES**, and the changes are not cosmetic

The *rule* survives every attack I could mount, including three the plan does not make and which strengthen it. The *plan* does not survive: it is built on the wrong file.

**The kill:** `outputs/submit_v10_reparam.parquet` is **not** our 284.3584 submission. The file that scored 284.3584 is `outputs/v10_probe_reparam.parquet`. The two differ on **336,426 of the 339,551 matched rows**, RMS 64.69 s, max 5,623 s [S, `/tmp/rev/r26.py`]. Everything in §4, §5.4 and §6 of the plan — the 47 rows, Σ(move)² = 1.7567e7, the +0.0896 s floor, the 47.2 % top-1 share, the −0.3683 s worst case, the pre-registered price, and the label-free gate's assertions — is computed against a model output that was **never uploaded and has no leaderboard number**.

If the plan were executed as written, we would upload a file that differs from our best scored submission on 336,426 rows. That is the exact failure the plan congratulates itself for avoiding at §1.6 and forbids at §5.4 step 5 ("Never build on an unscored upload").

On the **correct** baseline the arithmetic changes materially and in both directions [S, `/tmp/rev/r28.py`, `/tmp/rev/r29.py`]:

| quantity | plan (wrong file) | **true (scored file)** |
|---|---|---|
| rows moved | 47 (43 up / 4 down) | **34 (30 up / 4 down)** |
| Σ(move)² | 1.7567e7 | **2.3729e7** |
| guaranteed gain | +0.0896 s | **+0.1210 s** |
| max move | 2,881 s | **4,295 s** |
| **top-1 share of the floor** | 47.2 % | **77.7 %** |
| top-3 share | 93.9 % | 89.5 % |
| airports touched | "all 10" | **6** (no EDDF, EDDM, LIRF, LSZH) |
| absolute worst case | +0.3683 s | **+0.3550 s** |
| calibrated expected gain | "+0.3 to +0.6 s" | **+0.36 s** |
| overlap with the plan's 47 rows | — | **17** (30 of the plan's 47 do not move; 17 movers are not on the plan's list) |

The plan's own withdrawal trigger fires on this: §6.3 says withdraw if "the pre-upload gate of §5.4 step 3 reports a row count other than 47". Against the real file it reports 34.

The rule is still worth shipping. It is worth *more* than the plan says (+0.121 s floor, +0.36 s expected). But it now **flatly fails** the single-row-share guard (`.../09-repo-sergiuv11.md:747`, reject where one row supplies > 50 % of the net advantage) at 77.7 %, not "within a rounding error" of failing it. That must be re-argued from scratch, not inherited from §4.6.

---

## (b) Attack by attack

### 1. Re-verify the identity independently — **SURVIVES, and is stronger than the plan claims**

**The count reproduces exactly.** [S] `/tmp/rev/r1.py` over all twelve `data/raw/training_*.parquet`, DEP rows only:

```
DEP rows 2,085,047 | LOBT_flt present 2,062,577 | BLOCK present 2,085,047
|BLOCK - LOBT| > 3606 : 0
min d = -3606   max d = +3606   (both attained)
TAXITIME_SEC_mvt != MVT - BLOCK : 0 violations, 0 nulls
```

**The bound holds on every slice I could cut** [S, `/tmp/rev/r17.py`]: all 10 airports (max |d| 3602–3606, 0 violations each), all 676 `AIRCRAFT_OPERATOR_flt` values (0 violations, max 3606), all 12 months, and on every subpopulation that looks like a bad join:

| subpopulation | n | violations | max \|d\| |
|---|---|---|---|
| `ADES_mvt != ADES_flt` | 29,160 | 0 | 3605 |
| `ADES_mvt` differs from both flt destination cols | 27,763 | 0 | 3605 |
| `FLIGHT_mvt != CALLSIGN_flt` | 560,556 | 0 | 3605 |
| `LOBT != EOBT_1` (plan was updated) | 618,141 | 0 | 3606 |
| `MARKET_SEGMENT_flt == Non-Scheduled` | 14,577 | 0 | 3605 |
| `ADEP_mvt != ADEP_flt` | 0 | — | — |

**LOBT is never nonsensical.** [S] `/tmp/rev/r16.py`: 0 rows with year < 2000, 0 with year > 2027, range 2025-01-01 00:20 → 2025-12-31 23:40 (2025) and 2026-01-01 00:04 → 2026-07-31 23:30 (2026); microsecond component always 0. `LOBT == MVT_TIME` on 278 of 2,062,577 rows (2025) and 31 of 339,551 (2026) — those are rows where the plan's `m_lobt = 0`, so `lo = 0` and the rule is inert on them. Duplicated `(CALLSIGN_flt, LOBT_flt)` keys: 3 pairs in 2025, 1 pair in 2026 — irrelevant at that scale.

**Nulls line up exactly as the plan claims.** [S] `/tmp/rev/r13.py` on the ranking DEP frame: `LOBT_flt` null on 5,290 rows, `AOBT_3_flt` null on 5,290 rows, `LOBT null AND AOBT present` = 0, `AOBT null AND LOBT present` = 0. Row ids in `outputs/*.parquet` and `ranking.parquet` DEP form the same 344,841-element set with no duplicates.

**I found evidence the plan missed that makes the bound much harder than "0 violations in 2.06M rows".** The 3606 edge is not the end of a smooth tail; it is a *truncation*, and the truncation is visible in a variable that has nothing to do with the bound [S, `/tmp/rev/r7.py`, `/tmp/rev/r9.py`].

`BLOCK_TIME_UTC_mvt` carries a sub-minute jitter `j = BLOCK − round_to_minute(BLOCK)` whose support is the full `[-30, +29]`. Define `Dr = round_to_minute(BLOCK) − LOBT`; `Dr` runs exactly `[-3600, +3600]` and `LOBT` is minute-aligned on 2,056,231 of 2,062,577 rows. Then:

```
Dr = +3540 (one minute inside the edge): n=791, jitter spans -30 … +29   (full support)
Dr = +3600 (the edge):                   n=608, jitter spans -30 … +6    (truncated above)
Dr = -3600 (the edge):                   n=107, jitter spans  -6 … +27   (truncated below)
```

That is the signature of a *raw-second* rule `|BLOCK − LOBT| ≤ 3606` slicing through an unrelated, uniformly-supported nuisance variable. It rules out my leading alternative hypothesis — that the real rule is a minute-resolution window (`|Dr| ≤ 3600`) and that 3606 is just the largest jitter that happened to co-occur with the edge, in which case the true bound would be 3630 and clipping at 3606 would be a fit. It is not: under that hypothesis we would expect roughly 11 % of the 608 edge rows (the jitter mass above +6) to land in `d ∈ [3607, 3629]`, and we observe **0** [X]. **The plan's §1.4 conclusion is right; its argument ("a hard edge with mass on it") is weaker than the evidence available. Put the jitter-truncation test in the plan — it is the only measurement here that distinguishes a system constant from a quantile.**

**The one place the same constant does leak.** [S] `/tmp/rev/r3.py`, `/tmp/rev/r10.py`: on ARR rows the analogous edge is `MVT_TIME − ARVT_3_flt`, max exactly **+3606**, min **−3623**. Two 2025 rows (EDDV→LSZH 2025-03-03 at −3621; EDDK→LSZH 2025-04-08 at −3623) and one 2026 row (LEBL→LSZH 2026-01-10 at −3615) sit outside ±3606. So a 3606-second window in this same extract *can* leak, by up to 23 s [X]. The relevant consequence is quantitative and benign: an endpoint error of 23 s on a moved row changes ΔSSE by at most 23² = 529, i.e. ~3e-9 s of RMSE [X]. **This does not threaten the rule, but it does refute the plan's framing that 3606 is exact "with certainty" — the honest statement is "exact on the DEP side in 2.06M rows; the sibling window on the ARR side leaks by ≤ 23 s at a rate of ~1e-6".**

### 2. The 2026 transfer — **SURVIVES; and the plan's only stated drift signal is a seasonality artefact**

The plan is right that `|BLOCK − LOBT|` cannot be checked on 2026 DEP rows: [S] `/tmp/rev/r17.py` confirms `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are null on **all 344,841** 2026 DEP rows.

**What I could test, and did:**

**(i) The label-construction identity still holds in 2026.** [S] `/tmp/rev/r17.py`: on ARR rows, `TAXITIME_SEC_mvt == BLOCK − MVT` on 2,082,749/2,082,750 in 2025 (1 null row) and **344,693/344,693 in 2026, 0 violations, 0 nulls**. The organisers' label construction is bit-for-bit unchanged between years. The plan does not make this check and should.

**(ii) Preserved hard edges.** [S] `/tmp/rev/r11.py`, a full pairwise min/max scan over the eight timestamp columns visible in both years on DEP rows: `LOBT − IOBT` has min **exactly −900 in both 2025 and 2026**. `MVT − LOBT` min is −3,382 (2025) and −3,173 (2026), both inside −3606; **no row in either year has `m_lobt < −3606`** [S, `/tmp/rev/r22.py`], which is the necessary condition implied by `d ≥ −3606` and `y ≥ 0`. Nothing in the observable edge structure moved.

**(iii) The plan's drift signal is a season artefact — this is an error in `p5_drift.py`.** The plan says at `P5-physics-lower-bound.md:773` that `|AOBT_3 − LOBT| > 3606` "rises from 0.194 % to 0.282 % — the two clocks disagree ~45 % more often in 2026", calls it "the one real drift signal", and at :816 makes it a withdrawal trigger ("moves materially past 0.28 %"). That compares 2026's Jan+Jul against **all twelve months** of 2025. Season-matched [S, `/tmp/rev/r22.py`]:

| | unmatched % | \|a\|>3606 % | \|a\|>7212 % | m_lobt>3606 % |
|---|---|---|---|---|
| 2025 all 12 months | 1.078 | 0.1945 | 0.0054 | 1.414 |
| **2025 Jan+Jul** | **1.560** | **0.3280** | **0.0124** | **2.216** |
| **2026 Jan+Jul** | **1.534** | **0.2815** | **0.0080** | **2.110** |
| 2025 Jul alone | 2.065 | 0.4626 | 0.0134 | 2.868 |

Like-for-like the disagreement rate **falls 14 %** (0.328 → 0.2815), the unmatched rate is flat (1.560 → 1.534), and `m_lobt > 3606` is flat (2.216 → 2.110). **There is no drift signal.** The withdrawal trigger at :816 is calibrated against the wrong baseline and would fire spuriously on any normal Jan+Jul extract.

**(iv) The one genuine 2026 anomaly, and its explanation.** Rows with `m_lobt > 8000` — the population where the clip does the most work — rise from 12 (2025 Jan) to 31 (2026 Jan). [S] `/tmp/rev/r23.py` shows 21 of those 31 are **EHAM on 2026-01-02 … 01-07**, a single-airport disruption cluster; 2025 Jan's 12 are spread over 8 airport-days. July is flat (17 → 14). [X] This is an operational event, not an instrument change, and it is the kind of event where multi-hour taxi times are real. None of the EHAM cluster rows is a large mover on the real file (largest EHAM contribution is 1.58 % of the floor).

**(v) What remains unverifiable.** `d ≤ +3606` — the inequality that the 30 upward-moving rows depend on — has **no observable proxy on 2026 DEP rows**. Every test above is either about the machinery around the clamp or about the `d ≥ −3606` side. The plan is honest about this at §3.5 and my attacks did not change it.

### 3. The selection — **the plan's enumeration is against the wrong file; the corrected one fails the single-row guard outright**

[S] `/tmp/rev/r13.py`, `/tmp/rev/r14.py` reproduce the plan's 47 rows exactly on `outputs/submit_v10_reparam.parquet`: 47 moved, 43 up, 4 down, Σ(move)² = 17,566,665, top-1 47.25 %, top-3 93.95 %. So the plan's *arithmetic* is correct. Its *input* is not.

[S] `experiments/submissions.jsonl` entry 15: `{"version": 10, "exp_id": "submit_v10_reparam-77f92c802d", "file": "outputs/v10_probe_reparam.parquet", "object_name": "resilient-hedgehog_v10.parquet", "leaderboard_score": 284.3584}`. The `exp_id` names the training run; the **`file` key names the uploaded artefact**, and it is `v10_probe_reparam.parquet`, not `submit_v10_reparam.parquet`. The note on that entry says why: *"v4's 339,551 matched predictions kept byte-identical, only the 5,290 unmatched rows replaced by model B"*. Cross-check [S, `/tmp/rev/r27.py`]: `v10_probe_reparam.parquet` differs from `v11_rollover_mixture.parquet` on **exactly 11 rows**, matching the v11 log note ("identical on 344,830 rows, 11 rows replaced"). `submit_v10_reparam.parquet` differs from it on 336,426. The chain of custody is unambiguous.

**Corrected enumeration** [S, `/tmp/rev/r28.py`, `/tmp/rev/r29.py`] against `outputs/v10_probe_reparam.parquet`:

- **34 rows move** (30 up to `lo`, 4 down to `hi`); 0 unmatched rows touched; 0 rows with `lo > hi`; 7,166 rows have `lo > 0` (same as the plan, since that depends only on `m_lobt`).
- Σ(move)² = **2.3729e7** ⇒ guaranteed gain **≥ +0.1210 s** (284.3584 → ≤ 284.2374).
- median \|move\| 103 s, **max 4,295 s**.
- Concentration: **top-1 = 77.74 %**, top-3 = 89.5 %, top-5 = 96.3 %.
- Airport concentration: **LFPG 18 rows / 97.98 % of Σ(move)²**; LFPG-July alone is 16 rows and **97.87 %**. Only 6 of 10 airports appear; EDDF, EDDM, LIRF and LSZH do not move at all. The plan's §4.2 claim "all 10 airports and both months are represented" is false on the real file.
- Operator/day concentration: no single operator or day beyond the top row itself; the top row is one flight.

**The top row is 77.7 % of the whole thing.** [S] `/tmp/rev/r29.py`, `MVT_ID_mvt = 212754054.0`:

```
LFPG  2026-07-13  ACI501  Mainline  stand A18  rwy 08L
  MVT_TIME  22:19:57     SCHED 19:30:00
  LOBT      19:30:00  =  IOBT 19:30:00
  EOBT_1    22:10:00     AOBT_3 22:05:00     ARVT_3 2026-07-14 09:20:24
  ADES_mvt  NWWW         ADES_flt VTBS         <-- movement and flight plan disagree on destination
  pred 2296  ->  6591   (lo = m_lobt - 3606 = 10197 - 3606)
```

Read it plainly: `LOBT` is the *initial* filed off-block (it equals `IOBT`), `EOBT_1`/`AOBT_3` say the aircraft actually pushed back at 22:05, fifteen minutes before the movement time. The clip asserts the aircraft was off-block no later than 20:30 and taxied for 1 h 50 m. Every other field on the row says 15 minutes. **Is that anomaly disqualifying? No — and I tried hard to make it so.** [S] `/tmp/rev/r24.py`: 2025 rows with exactly this signature (`LOBT == IOBT`, `EOBT_1 − LOBT > 6000`, `|AOBT_3 − EOBT_1| < 1800`, `m_lobt > 7000`) number six, and on every one of them the true taxi time is enormous while `m_aobt` says twenty minutes:

```
ADEP  MVT_TIME              y      m_lobt  m_aobt   d
LIRF  2025-09-27 10:27:02  10025   10022    1082     -3
LFPG  2025-06-28 10:26:37   9162    9997     817    835
LIRF  2025-07-29 14:02:56   9172    9476     956    304
LIRF  2025-04-14 12:17:58   7676    7678    1318      2
LFPG  2025-02-14 09:58:28   7291    7408     988    117
LIRF  2025-10-12 10:37:02  10925    7322    1082  -3603
   y - lo:  min 2771, median 3547, max 7209, violations 0
```

On this exact population the `AOBT_3` clock is catastrophically wrong and the `LOBT` clock is right, every time, with the truth sitting a median 3,547 s *above* the clip's floor. **The top row is anomalous in precisely the way the rule is designed for. It does not disqualify the row; it is the reason the row is worth 77 % of the gain.** [S] `/tmp/rev/r15.py` generalises: on 2025 rows with `AOBT_3 − LOBT ∈ [8000, 10000)` (n = 47), `g = AOBT_3 − BLOCK` has min 5,095 and median 8,157 — the NM off-block is routinely two hours after the real one on that population.

**But the guard still fails.** Repo A's rule is "reject a candidate where one row supplies > 50 % of the net advantage". 77.7 % is not a rounding error. §4.6's defence ("the guard exists to catch candidates whose *sign* depends on one row; here the sign does not") is sound as far as it goes, but it must be re-written against 77.7 % and against the fact that **a single LFPG flight on 2026-07-13 now carries more than three quarters of the effect, and a single airport-month carries 97.9 %**. The realistic outcome distribution is close to binary: this row is either worth ~+0.28 s on its own or worth ~+0.094 s of damage.

### 4. Interaction with the model — **SURVIVES on mechanics; the plan's own §2.1 argument cuts against it and needs answering**

[S] `src/prc/train.py:379, 398` — routing is on `PROXY = "mvt_minus_aobt3"` null-ness. `src/prc/train.py:385, 396` — model A fits `LABEL − PROXY` and emits `m_tg[PROXY] + g`. [S] `src/prc/train.py:412-414` — the LIRF `sched_reparam` shift is added back to `g` **before** `_assemble`, with the comment at :410-411 stating the output is already in seconds of taxi time. [S] `src/prc/train.py:400-405` — model B's column list explicitly excludes anything starting with `mvt_minus_lobt` or `lobt_minus`, so model B never sees LOBT either. [S] `src/prc/train.py:433` — `_assemble` applies `.fill_null(fallback).fill_nan(fallback).clip(lower_bound=0.0)` after the concat. **§5.2 and §5.3 are correct: the two row sets are disjoint and exhaust the file, the reparameterisation is undone before the clip point, and there is no way for the clip to operate on a reparameterised quantity.** [S] `/tmp/rev/r13.py` confirms 0 unmatched rows touched on either candidate file.

[S] `src/prc/features.py:65` — `mvt_minus_lobt` is `(MVT − LOBT).dt.total_seconds().cast(Float64)` with nulls preserved and no imputation, so a future in-`_assemble` implementation (§5.1) would compute the same bound as a standalone script. That claim checks out.

**Two problems the plan does not address.**

**(a) The plan's rejection of rule A applies verbatim to rule B and is not answered.** §2.1 rejects rule A because *"for us, `pred < m_aobt − 3600` happens only when the model has positively decided ĝ(x) < −3600"*, and §2.2 shows the model is right 82 % of the time on exactly those rows. But `mvt_minus_lobt` **is already a model-A feature** (`src/prc/features.py:65`, and it appears in the feature list at `src/prc/features.py:237`). By the plan's own reasoning, when model A predicts outside the LOBT window it has also "positively decided". The plan never runs the rule-A-style check for rule B and never states why the asymmetry is legitimate. **It is legitimate, and the plan must say so:** model A is parameterised as `m_aobt + g(x)`, so satisfying the window requires `g(x) ≥ a − 3606` where `a = AOBT_3 − LOBT` is a continuous feature — an identity a tree can only approximate in steps, which is the exact reason `src/prc/train.py:3-4` gives for the AOBT reparameterisation in the first place. The difference from rule A is that rule A's floor can be checked against the truth and is wrong 82 % of the time, whereas rule B's window contains the truth by a verified fact. I confirmed the latter directly: [S] `/tmp/rev/r19.py`, `/tmp/rev/r20.py`, **0 of 2,062,577 training rows have `y` outside `[m_lobt − 3606, m_lobt + 3606]`.**

**(b) `max(m_lobt − 3606, 0)` is not the true lower endpoint, and the plan never notices.** The interval that provably contains the truth is `[m_lobt − 3606, m_lobt + 3606]`; the rule uses `max(·, 0)`. [S] `/tmp/rev/r20.py`: 129 training rows have `y < 0` (min −12), and on all 129 `lo = 0 > y`, so the projection argument does **not** hold on them. It is harmless only because `src/prc/train.py:433` already floors every prediction at 0, so `p ≥ 0 > y` already and the extra clip cannot bind: [S] all 129 have `m_lobt ≤ 3577`, and **0 rows have `y < 0` while `m_lobt − 3606 > 0`**. The plan's "ΔSSE ≤ 0 row by row, with certainty" (line 168) is therefore true *given the existing zero floor*, not unconditionally. Say so, or the statement is wrong as written.

### 5. The OOF evidence — **the fold-pooling defect is harmless here, but the gate result is a tautology and is being sold as evidence**

**The pooling defect does not bite.** Both arms of the comparison are the *same* OOF column, one clipped. [S] `/tmp/rev/r18.py`: the OOF files are two columns (`MVT_ID_mvt`, `oof`) with no fold label and different coverage (672,428 non-null on `oof_main-48c19bc830`, 328,009 on `oof_submit_v10_reparam-77f92c802d`). Since no fitted parameter crosses fold boundaries and the paired difference is computed row-wise on identical row sets, the defect described in `EXECUTION-ORDER.md` prerequisite B has no purchase. **The plan's argument on this point holds.** I reproduced the plan's numbers exactly [S, `/tmp/rev/r19.py`]: 85/85 helped on `oof_main`, 28/28 on `oof_submit_v10_reparam`, ΔSSE −4.4674e7 and −2.3988e7, ratios 5.33× and 6.92×, and all four drop levels at P(better) = 1.000 with strictly negative CIs.

**But the measurement contains no information.** Given the verified fact that `y ∈ [lo, hi]` on 2,062,577/2,062,577 training rows, `ΔSSE ≤ 0` holds row-by-row as a theorem. Therefore: every bound row helps; every bootstrap resample has `SSE_B ≤ SSE_A`, hence `RMSE_B ≤ RMSE_A`, hence `b_better_prob = 1.000`; and dropping the worst K rows cannot flip a sign that is nonpositive on every row. **All eight cells of §3.4 are entailed by §1.4 and could have been written down without running `prc.compare`** [X]. The headline at `P5-physics-lower-bound.md:17-20` — *"the first change in this repository's history to pass `prc.compare` at all four drop levels with P(better) = 1.000"* — is therefore not a finding. `prc.compare` has **zero power** against the one failure mode that matters (the bound being invalid on 2026), because it runs on 2025 where the bound is known-true. §3.4 half-concedes this in one clause and §0 then trades on it anyway. Delete the claim or demote it to "as the projection property requires".

**The OOF belongs to the wrong model, too.** `oof_submit_v10_reparam-77f92c802d` is the OOF of the `submit_v10_reparam` training run — the run whose matched predictions were **never uploaded**. The shipped matched lane is v4's (`v4_linear_tree`), and [S] there is no `oof_v4*` file in `experiments/`. So §3.3's bind counts, the 6.92× multiplier and §4.4 estimate 3 all describe a model A we are not clipping. Consistent with that: only 17 of the plan's 47 rows are movers on the real file [S, `/tmp/rev/r28.py`]. The projection argument still guarantees the sign for any model, but **every magnitude transferred from that OOF is unsupported.**

### 6. Things nobody thought of

**Format and gate: passes, but only by luck on one bound.** [S] `/tmp/rev/r25.py`, `/tmp/rev/r30.py` — I built the clipped candidate in memory from the real scored file and ran `prc.validate_submission.validate` against `data/raw/submitting.parquet` with `reference_median = 912.0` (the 2025 DEP label median):

```
scored v10 : ok=True  errors=[]  warnings=['max prediction 112771 exceeds 10800s']  min 0 max 112771 median 959
clipped    : ok=True  errors=[]  warnings=['max prediction 112771 exceeds 10800s']  min 0 max 112771 median 959
schema identical to template: True   ids identical to template: True   rows 344841
```

Template dtype is `TAXITIME_SEC_mvt: int32`, ids `double`. The clip of an int32 prediction to integer bounds stays integral, so `write_submission`'s rounding path is a no-op. No negatives, no NaN, no inf, median drift 5.2 % against a 30 % limit. The `MAX_PLAUSIBLE_SECONDS` warning already fires on the unmodified file and is not an error (`src/prc/validate_submission.py:19, 88-89`).

**The `lo > hi` hazard is real and the margin is 433 seconds.** `hi = m_lobt + 3606` goes negative whenever `m_lobt < −3606`. [S] `/tmp/rev/r22.py`: min `m_lobt` is −3,382 in 2025 and −3,173 in 2026, so the margin on the current extract is 433 s and the case cannot arise. But if it ever did, `clip(p, 0, negative)` is silent in polars and `validate` would then catch it only via the `negative prediction found` error. The plan's §5.1 guard is necessary; make it an assertion in the v12 build script, not a note.

**`MVT_ID_mvt` is float64 with integral values in the template, the scored file and the ranking frame**, and all three id sets are identical with no duplicates [S, `/tmp/rev/r13.py`, `/tmp/rev/r25.py`], so `_align_id_dtype` (`src/prc/predict.py:35-53`) is a no-op on this path.

**The scored file has no OOF and no reproducing config**, which is `EXECUTION-ORDER.md` prerequisite A. Building v12 on it is still correct — it is the only file with a leaderboard number — but it means the offline evidence for the clip and the file being clipped come from two different models. That is worth one sentence in the log entry.

---

## (c) What I could not verify

1. **`d ≤ +3606` on 2026.** No observable proxy exists on 2026 DEP rows. The 30 upward-moving rows — including the one worth 77.7 % — rest entirely on 2025 extrapolation. Everything I tested (label identity, `LOBT − IOBT` edge, `m_lobt` support, season-matched rates) is consistent with the machinery being unchanged, but none of it observes `BLOCK`.
2. **Why the constant is 3606 and not 3600 or 3660.** The jitter-truncation test proves it is a raw-second rule, not a minute-resolution one, but the origin of the 6 seconds is unexplained. The plan's "the tolerance the recording system carries" is a guess; so is mine.
3. **Whether the leak rate on the ARR sibling window (2/2,067,301 in 2025, 1/342,141 in 2026, overshoot ≤ 23 s) transfers to the DEP window.** It does not appear to: 310 DEP rows sit within 6 s of ±3606 with 0 leaks, against 2 leaks among ~12 ARR rows in the same band [S, `/tmp/rev/r6.py`, `/tmp/rev/r10.py`], which is hard to reconcile with a common mechanism [X]. I could not determine what the ARR window is actually applied to.
4. **The plan's §2 numbers on rule A** (the 82 % "truth above the floor" column, the ΔSSE figures) I did not re-derive. I have no reason to doubt them and rule A is being rejected anyway.
5. **Whether `outputs/v10_probe_reparam.parquet` on disk is byte-identical to the uploaded `resilient-hedgehog_v10.parquet`.** The evidence is strong but indirect: the submissions log names it, its mtime (Sep 17 06:15) precedes the recorded upload (06:16:30), and v11 was built from it and differs in exactly the 11 rows the v11 note claims. I did not fetch the object from the team bucket to compare.

---

## (d) Required changes before this is executed

**Blocking — the plan cannot be run as written.**

1. **Replace every occurrence of `outputs/submit_v10_reparam.parquet` with `outputs/v10_probe_reparam.parquet`** as the file to transform, at lines **262, 448, 673, 686** (and fix the description at 224). Add a one-line provenance note citing `experiments/submissions.jsonl` entry with `"leaderboard_score": 284.3584` and its `"file"` key. **Add an assertion to the build script that the input file, byte-compared, differs from `outputs/v11_rollover_mixture.parquet` on exactly 11 rows** — that is the cheapest available proof that you are holding the scored file.

2. **Re-derive the whole of §4 against the correct file.** The replacement numbers are measured and listed in section (a) above: 34 rows, 30 up / 4 down, Σ(move)² = 2.3729e7, guaranteed gain ≥ +0.1210 s, max move 4,295 s, worst case +0.3550 s (y = 0) / +0.2564 s (y = 900) / +0.1210 s (y = p).

3. **Rewrite the pre-registered price at line 684.** My calibrated estimate, built by resampling `d` from 2025 rows matched on `m_lobt` (and on `a = AOBT_3 − LOBT`) and evaluating `ΔSSE = −|move|·(|move| + 2·dist(y, boundary))` exactly, is **+0.36 s** (+0.358 conditioning on `m_lobt` alone, +0.373 adding `a`) [S, `/tmp/rev/r29.py`]. Proposed registration: **guaranteed ≥ +0.121 s; expected +0.36 s; absolute worst case −0.355 s; a result outside [−0.36, +0.90] falsifies the construction, not the rule.**

4. **Rewrite the §5.4 step 3 label-free gate** to assert **34 rows, 30 up / 4 down, max |move| = 4,295 s**, plus: 0 unmatched rows touched, `lo ≤ hi` on every row (hard assertion, not a note), every moved row inside its own window, no negative, no NaN, 344,841 rows, and **336,426-row-difference check against `submit_v10_reparam.parquet` to prove you did not pick up the wrong file again**.

5. **Rewrite §4.6 for a 77.7 % top-1 share.** The single-row-share guard is failed outright, not narrowly. The defence still stands on the sign argument, but the plan must state plainly that **one LFPG flight (ACI501, 2026-07-13, `MVT_ID_mvt` 212754054) carries 77.7 % of the guaranteed gain, LFPG-July carries 97.9 %, and four of the ten airports are untouched** — and that the realised board delta is therefore close to a two-point distribution.

**Required corrections that do not block but must not ship uncorrected.**

6. **Delete or demote the §0 headline claim** (lines 17-20) that this is "the first change to pass `prc.compare` at all four drop levels with P(better) = 1.000". A change with a per-row nonpositive ΔSSE passes that test by construction; the gate has no power against the only failure mode. Keep the numbers, drop the rhetoric.

7. **Fix §3.5 item 4, §6.1 item 2 and the §6.3 withdrawal trigger** (lines 411-413, 770-775, 814-816) with season-matched rates. There is no drift: `|AOBT_3 − LOBT| > 3606` is 0.328 % (2025 Jan+Jul) → 0.2815 % (2026), a 14 % *fall*; `m_lobt > 3606` is 2.216 % → 2.110 %; unmatched is 1.560 % → 1.534 %. Set the withdrawal trigger against the season-matched baseline, not against 0.194 %.

8. **Fix line 456's claim** that "all 10 airports and both months are represented" — 6 airports on the real file.

9. **Add the jitter-truncation evidence to §1.4.** It is the strongest evidence in the whole case and the plan does not have it: at `Dr = round_to_minute(BLOCK) − LOBT = ±3600` the sub-minute jitter of `BLOCK`, whose unconditional support is `[−30, +29]`, is truncated to `[−30, +6]` and `[−6, +27]` respectively, which is only explicable as a raw-second rule at 3606.

10. **Add the two 2026 transfer checks the plan omits**: the ARR label identity (`TAXITIME == BLOCK − MVT`, 344,693/344,693 in 2026) and the preserved `LOBT − IOBT ≥ −900` edge in both years.

11. **Qualify the "ΔSSE ≤ 0 with certainty" claim at line 168.** It is conditional on the pre-existing zero floor at `src/prc/train.py:433`, because `max(m_lobt − 3606, 0)` exceeds the truth on the 129 training rows with `y < 0`.

12. **State the model-decision symmetry and answer it** (§2.1's argument against rule A applies to rule B because `mvt_minus_lobt` is a model-A feature; the answer is the reparameterisation argument at `src/prc/train.py:3-4`, plus the fact that rule B's window provably contains the truth where rule A's floor does not).

13. **Add one sentence to the log entry** noting that the offline evidence (`oof_submit_v10_reparam-77f92c802d`) is the OOF of a different model A than the one being clipped, and that the shipped matched lane (v4) has no OOF at all.

**What survives untouched:** the identity itself (§1.4), the structural projection argument (§1.5), the rejection of rule A (§2), the model-routing and LIRF-reparameterisation analysis (§5.2, §5.3), the "do not tune 3606" and "do not clip the training target" prohibitions (§5.5), and the placement argument for `_assemble` (§5.1). The rule is sound and, on the correct file, worth more than the plan claims. The plan is not executable until items 1–5 are done.

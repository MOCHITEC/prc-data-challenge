# P6 adversarial review — per-airport upper bound from 2025 label support

Reviewed 2026-09-23 against `docs/competitor_research_2026-09-21/plans/P6-per-airport-upper-bound.md`
(written 2026-09-21, before v12 and v13). Read-only. Nothing under `src/`, `configs/` or `outputs/` was
touched; no submission was built or sent. Scratch scripts in `/tmp/p6rev/`.

Marks: **[S]** measured or read by me in this review, with the command; **[X]** my inference.
All monetary figures are seconds of leaderboard RMSE, positive = worse.

---

## (a) VERDICT — **PROCEED WITH CHANGES, and re-register the price at roughly one third of the plan's**

The plan does **not** have a wrong-file defect of the P5 kind. I re-enumerated against
`outputs/v12_lobt_bound.parquet` and the twelve rows, the four caps and the movement arithmetic all survive
the move from v10 to v12 unchanged. The rule is still well defined, still bounded, and still touches exactly
twelve rows.

What does not survive is **the price and the evidence base**:

1. **The pre-registered −0.20 s is arithmetically impossible.** It is computed at a central case of
   `y ≈ 1,000 s` on all twelve rows. The LOBT identity that v12 ships proves `y ≥ 1,722 s` on the weakest of
   the twelve and `y ≥ 3,298 s` on the strongest. The **maximum gain the rule can achieve on any feasible
   label vector is −0.094 s**, and the central case under the plan's own support assumption is **−0.062 s**.
   §(b)6.
2. **The "22 helped / 0 hurt" OOF sign check is a theorem, not a measurement.** A cap set at the 2025
   per-airport maximum cannot hurt any 2025 row, because every 2025 label is ≤ that maximum by construction.
   The count is entailed, exactly as `P5-REVIEW.md:165` showed `P(better) = 1.000` was entailed for the LOBT
   bound. It carries zero information about 2026. §(b)7a.
3. **The OOF rows it was measured on are the wrong class.** 16 of the 18 rows the four-airport rule touches
   on `oof_main-48c19bc830` are **unmatched** (`nm_matched = 0`); all twelve rows it would move on the
   ranking file are **matched**. On the v10-recipe OOF the rule fires on **one** row, at EDDM, and on **zero**
   LEMD/LEBL rows. §(b)7b.
4. **The one label-free channel that looks at 2026 *departures* at LEMD argues against the rule.** The
   LOBT-implied provable lower bound on the departure label reaches **3,298 s at LEMD in 2026 against 3,043 s
   over all of 2025**, and the rate above 3,000 s goes from 0.00 to 0.81 per 10,000. The three rows carrying
   that signal are three of the twelve, and they are among the biggest contributors. The plan's arrival mirror
   (taxi-**in**) cannot see this. §(b)5b.
5. **The interaction section is written against the wrong rule and its safety margin is 10× too loose.** The
   binding constraint is not `m_aobt − 3600 ≤ S_a` (margin 2,884 s at EDDM); it is
   `max(mvt_minus_lobt − 3606, 0) ≤ S_a`, whose margin today is **300 s at LEMD**. And the composition order
   the plan prescribes (§5.2: floor first, then cap) is the order that would violate the proven window;
   the order its §5.1 code placement would actually produce is the safe one. §(b)2.

Corrected pre-registration: **−0.06 s expected, feasible range [−0.094, +0.030], tail risk +0.46 s**, on a
base of 284.045. That is **one fifth of the guaranteed floor v12 already banked** (+0.121 s guaranteed,
+0.3134 s realised). Proceed only if the slot is genuinely idle; it is not worth displacing anything.

---

## (b) Attack lines

### 1. Which file should this be applied to now?

**[S]** `experiments/submissions.jsonl` (last four records): v10 `outputs/v10_probe_reparam.parquet`
284.3584; v11 `outputs/v11_rollover_mixture.parquet` 293.978; **v12 `outputs/v12_lobt_bound.parquet`
284.045, `delta_vs_v10: -0.3134`, `status: uploaded`**; v13 `outputs/v13_single_config_bound.parquet`
uploaded 2026-09-23T06:15Z with **no `leaderboard_score`**. The scored best is therefore
`outputs/v12_lobt_bound.parquet`. I did not trust the plan's path: it names
`outputs/v10_probe_reparam.parquet` at `P6-per-airport-upper-bound.md:169` and `:486`, which is the previous
best, not the current one.

**[S]** `/tmp/p6rev/a1.py`:

```
v10 vs v12 differ: 34
v10 vs v13 differ: 336425
v10 vs submit_v10_reparam differ: 336426      <- the P5 trap, still present on disk
rows above S_a: v10 -> 12 (LEMD 10, LEBL 2)
                v12 -> 12 (LEMD 10, LEBL 2)   -- identical MVT_IDs and identical predictions
                v13 -> 10 (LEMD  8, LEBL 2)
```

**Findings.**

- The LOBT bound **did not touch any of the twelve**. `p12 == p10` on all twelve rows, so the plan's §4.4
  table (predictions, caps, moves 146/552/1,147, months 8 Jan / 4 Jul) transfers verbatim. **[S]**
- The LOBT bound did move **three other rows at the capped airports** — LEMD `213724395` 1,479→1,505, LEBL
  `213798774` 1,317→1,421, LEBL `213805661` 1,192→1,202, all raised, all landing 2,000+ s **below** the cap.
  So the bound neither creates nor destroys a cap candidate. **[S]** `/tmp/p6rev/a2.py`.
- **The trap the plan walks past:** `outputs/submit_v10_reparam.parquet` still sits in `outputs/` and differs
  from the scored file on 336,426 rows. The plan's build step reads a hand-typed path. Require the build
  script to assert the source file's SHA or its 34-row diff against v10, not its name.
- **v13 must not be the source.** It is unscored, the repo's own discipline forbids building on an unscored
  upload (`P6-per-airport-upper-bound.md:158-160` quoting `MODEL_ANALYSIS.md:1050`), and the rule fires on
  **10** rows there rather than 12 — the row set is model-dependent, so a P6 built on v13 would not be
  comparable to one built on v12.

**Verdict on this line: the plan's enumeration is correct; only the source path and the baseline number are
stale.** This is not a P5-grade defect.

### 2. Do the cap and the LOBT bound fight?

The plan's §5.2 analyses P5 **rule A** (the AOBT floor `pred := max(pred, m_aobt − 3600)`). That rule was
rejected — `P5-REVIEW.md:194` "rule A is being rejected anyway", and `P5-REVIEW.md:157` records why: the model
is right 82 % of the time on exactly the rows the floor would move. What shipped is **rule B**, the LOBT
window, `src/prc/train.py:442-474` (`_lobt_bound`), `src/prc/train.py:133-145` (`TrainConfig.lobt_bound`,
`lobt_bound_slack = 3606.0`), applied in `_assemble` at `src/prc/train.py:487-497`. So §5.2 is an analysis of
a rule that does not exist.

Redone from scratch, **[S]** `/tmp/p6rev/a2.py`:

- **No correctness failure today.** `lo := max(mvt_minus_lobt − 3606, 0)` exceeds `S_a` on **0 of 129,317**
  rows at the four capped airports. The cap therefore never pulls a prediction below a lower edge that
  provably contains the truth.
- **But the margin is 300 s, not 2,884 s.** `max(lo − S_a)` per airport: **LEMD −300**, LEBL −1,284,
  EDDF −2,111, EDDM −2,824. The plan's checklist (`:596-597`) asserts the tightest margin is "3,238 s at
  EDDM" — that is the AOBT quantity. The real binding quantity is ten times tighter, and it is tightest at
  the airport where all ten capped rows live. One LEMD flight with `mvt_minus_lobt > 7,204` would make the
  two rules contradict each other outright.
- **On the twelve rows the two bounds stack legally but tightly.** `cap − lo` is 300 / 323 / 588 / 720 / 885
  / … / 1,882 s. On row `203363337` the composed admissible interval for the truth is **[3,298, 3,598]** — a
  300-second box, of which the top 0.01 % of LEMD's entire 2025 distribution (p99.99 = 3,437) occupies half.
- **Order matters, and the plan prescribes the wrong one.** The correct projection onto
  `[lo, hi] ∩ (−∞, S_a]` is `max(lo, min(p, hi, S_a))`, i.e. **cap first, LOBT second**. If `S_a < lo` ever
  occurs, cap-then-LOBT returns `lo` (the cap goes inert, the proven bound wins); LOBT-then-cap returns
  `S_a < lo` — a value the data proves is wrong. §5.2 explicitly prescribes "apply the floor, then the cap"
  (`:533-539`) and justifies it with "the physical observation is the better evidence" — the prescription
  contradicts its own justification. Its §5.1 code placement ("immediately after the existing
  `clip(lower_bound=0.0)`", `:494-495`) happens to land *before* `_lobt_bound` and would therefore produce
  the safe order by accident.

### 3. Re-verify `S_a`

**[S]** `/tmp/p6rev/a3.py`, reading `data/raw/training_*.parquet` directly (13 files, 4,167,797 rows;
2,085,047 `PHASE_mvt == "DEP"`, zero null labels), never `feat_train.parquet`:

| ADEP | n | max | n>3,600 | n>7,200 | top-8 labels |
|---|---|---|---|---|---|
| EDDF | 230,141 | **6,907** | 73 | 0 | 6907, 6566, 6403, 6234, 5990, 5920, 5911, 5699 |
| EDDM | 167,334 | **6,899** | 59 | 0 | 6899, 6850, 6612, 6487, 6305, 6250, 5398, 5332 |
| LEBL | 179,705 | **3,604** | 1 | 0 | 3604, 3599, 3595, 3594, 3584, 3572, 3567, 3552 |
| LEMD | 212,242 | **3,598** | 0 | 0 | 3598, 3567, 3566, 3562, 3549, 3546, 3537, 3522 |

**The four quoted values reproduce exactly.** So do the censoring signatures: the `[3500,3600)` →
`[3600,3800)` histogram step is LEMD 10→0 and LEBL 16→1 while every other airport rises (EDDF 16→24,
EDDM 12→17, EGLL 83→141, EHAM 15→31, LFPG 47→103, LIRF 99→193, LSZH 2→7, LTFM 76→152), and the 24 monthly
maxima at LEMD/LEBL are all in [2,993, 3,604].

**[S]** `/tmp/p6rev/a8.py`: `TAXITIME_SEC_mvt == (MVT_TIME − BLOCK_TIME)` on **all 2,085,047** DEP rows, so
the ceiling lives in the timestamps, not in a post-hoc clip of the label column. That is weaker evidence than
a label clip would be: a label clip would be pipeline machinery certain to be reapplied in 2026, whereas a
timestamp property need not be.

**Two different objects are being capped under one rule.** **[X]**

- LEMD/LEBL: a **censored** distribution. 3,598 and 3,604 are a wall; the next values below are 3,567 and
  3,599; nothing above. A no-margin cap is defensible here.
- EDDF/EDDM: an **uncensored** distribution whose maximum is a single flight. EDDF 6,907 with the runner-up
  at 6,566 (341 s below); EDDM 6,899 with the runner-up at 6,850. The histogram rises straight through 3,600.
  There is no truncation evidence at all at these two airports — the plan's §3.6 evidence (histogram step,
  12 monthly maxima) is presented for LEMD/LEBL only and does not apply here.
  Capping a *sampled* maximum with **no margin** is precisely the case `kind-mango`'s `+600` exists for
  (`postprocess.py:31`), and the plan's argument against the margin (`:395-397`, "the bound is not sampled —
  it is a truncation visible in the histogram") is true at LEMD/LEBL and **false at EDDF/EDDM**.

The cap binds on zero EDDF/EDDM rows today (**[S]** `/tmp/p6rev/a7.py`: max v12 prediction EDDF 6,493,
EDDM 5,031), so this is inert for the upload — but §5.1 step 2 folds the rule permanently into `_assemble`,
where it would bind on every future run.

### 4. Airport selection

**[S]** `/tmp/p6rev/a7.py` — count of 2025 DEP labels above 7,200, and the distance of the maximum to 7,200:

```
EDDF 0 (max 6907, 293 s below)      EHAM 3 (max 11032)
EDDM 0 (max 6899, 301 s below)      LSZH 6      LTFM 21
LEBL 0 (max 3604, 3596 s below)     EGLL 37     LFPG 37     LIRF 480
LEMD 0 (max 3598, 3602 s below)
```

The set is **fragile in both directions and on single-digit row counts**:

- **Closest to joining: EHAM, by three rows.** Delete the three EHAM flights above 7,200 (11,032 / 10,972 /
  8,750) and EHAM satisfies the selection criterion exactly. The plan then vetoes EHAM on a completely
  different basis (the arrival mirror, §3.5) and is right to. **[X] The stated selection criterion is not what
  makes the set correct — the arrival mirror is. Three rows separate the criterion from selecting an airport
  the plan itself calls a real 2026 disruption.** The criterion should be demoted to a filter and the arrival
  mirror promoted to the decision rule, or the rule restricted to the two airports where the censoring
  evidence exists.
- **Closest to leaving: EDDF at 293 s, EDDM at 301 s.** One flight taxiing five minutes longer in 2025 drops
  either airport out of the set. A threshold that a single observation can flip is not a partition of
  airports; it is a partition of one sample.

### 5. 2026 transfer

**(a) The arrival mirror, re-run on all ten airports** — **[S]** `/tmp/p6rev/a4.py`, 344,693 `PHASE_mvt ==
"ARR"` rows in `data/raw/ranking.parquet`, zero null `TAXITIME_SEC_mvt`, months {1, 7}, against 2025 Jan+Jul
arrivals at the same `ADES_mvt`. Ratio = 2026 observed max ÷ 2025 quantile at `q = 1 − 1/n`:

```
EDDF 1.051   EDDM 0.710   LEBL 0.743   LEMD 1.138      <- the four capped airports
EGLL 0.857   LSZH 0.822   LTFM 1.020   LIRF 1.590
EHAM 3.517   LFPG 9.252                                <- inflated; both correctly excluded
```

**It clears all four capped airports, not just the two the plan checked.** Tail rates above 3,600 s per
10,000 arrivals, 2026: EDDM 0.00, LEBL 0.00, LEMD 0.00, EDDF 0.28 — against EHAM 18.88, LIRF 15.65, LFPG
5.54. The plan's §3.5 conclusion is reproduced and extended; the EHAM veto is confirmed.

**(b) The channel the plan could not run, because it predates v12 — and it argues the other way.** **[S]**
`/tmp/p6rev/a9.py`. `lo = max(MVT_TIME − LOBT − 3606, 0)` is a **provable lower bound on the departure taxi
label**, verified on 2,062,577 training rows (`src/prc/train.py:135-137`; I re-confirmed max |BLOCK − LOBT| =
3,606 exactly at LEMD and LEBL). Unlike taxi-in, it speaks about departures, and it is available on 2026.

| ADEP | 2025 Jan+Jul max `lo` | rate >3,000 /10⁴ | 2026 max `lo` | rate >3,000 /10⁴ |
|---|---|---|---|---|
| **LEMD** | 2,742 (all-2025: **3,043**) | 0.00 | **3,298** | **0.81** |
| LEBL | 2,251 (all-2025: 2,641) | 0.00 | 2,320 | 0.00 |
| EDDF | 4,787 | 0.27 | 4,796 | 3.58 |
| EDDM | 3,237 | 0.37 | 4,075 | 0.39 |
| EHAM | 5,270 | 3.91 | 9,808 | 23.05 |

**LEMD 2026 contains a departure whose true taxi time is provably at least 3,298 s — 255 s beyond the largest
provable lower bound anywhere in all of 2025 at that airport, and within 300 s of the ceiling the rule treats
as inviolable.** The rate above 3,000 s goes from zero to 0.81 per 10,000.

**[X]** This does not prove the ceiling was lifted. It does mean the only 2026 evidence that looks at LEMD
*departures* points the opposite way to the arrival mirror, and it lands on exactly the rows the rule is
about: the three LEMD rows with `lo` above 3,000 (`203363337` lo 3,298, `203363474` lo 3,275, `203363319`
lo 3,010) are three of the twelve and together carry **33.9 %** of the rule's best-case gain
(**[S]** `/tmp/p6rev/a10.py`). The plan's §6 item 2 names "the ceiling was lifted" as the single assumption
carrying the rule; this is the first evidence bearing on it, and it is adverse.

LEBL is clean on this channel (2,320 vs 2,251, no change).

### 6. Price against 284.045

**[S]** `/tmp/p6rev/a5.py`, `/tmp/p6rev/a10.py`. Base 284.045, N = 344,841, one second of RMSE = 1.959e8 of
SSE. Applying the cap to `outputs/v12_lobt_bound.parquet`: 12 rows, total movement 7,466 s, `Σ(p−c)² =
5.921e6`, mean 1,039.3865 → 1,039.3649, median 959 and max 112,771 unchanged.

**The plan's central case is infeasible.** Per row, the cap helps iff `y < (p + c)/2`, and `y ≥ lo` is
proven:

```
MVT_ID      ADEP  p      cap    lo      helps iff y <     y is at least
203363474   LEMD  4745   3598   3275    4171.5            3275
203363319   LEMD  4720   3598   3010    4159.0            3010
203363337   LEMD  4599   3598   3298    4098.5            3298
203349361   LEMD  4418   3598   2713    4008.0            2713
213727911   LEMD  4290   3598   2237    3944.0            2237
203363460   LEMD  4289   3598   2487    3943.5            2487
203363459   LEMD  4011   3598   2309    3804.5            2309
203363465   LEMD  4001   3598   2308    3799.5            2308
213732168   LEMD  4000   3598   2878    3799.0            2878
213802815   LEBL  3953   3604   1722    3778.5            1722
203349358   LEMD  3878   3598   2603    3738.0            2603
213791326   LEBL  3750   3604   2320    3677.0            2320
```

| label assumption | ΔRMSE at 284.045 | feasible under the LOBT identity? |
|---|---|---|
| plan's central case, `y = 1,000` on all twelve (`:464`) | −0.2284 | **NO — violated on all 12 rows** |
| `y = lo` on every row (**best feasible case**) | **−0.0938** | yes |
| `y ~ U[lo, S_a]` per row (support holds) | **−0.0620** mean, 5–95 % [−0.0716, −0.0525] | yes |
| `y = S_a` on every row | −0.0302 | yes |
| `y = p` — the model was right | **+0.0302** | yes |
| `y = 7,200` on every row | +0.2442 | yes |
| `y = hi = mvt_minus_lobt + 3606` (**worst feasible case**) | **+0.4555** | yes |
| 10 % / 20 % / 30 % of rows drawn from a lifted ceiling `U[lo, 7200]` | −0.048 / −0.035 / −0.021 mean; P(helps) 0.98 / 0.92 / 0.79 | yes |

**The rule cannot deliver −0.20 s. Its ceiling is −0.094 s and its honest expectation is −0.06 s.** The
plan's own break-even (common `y` = 3,995 s) is unchanged and still correct; what changed is that the
favourable side of it is now bounded at 3,298 rather than unbounded below.

Concentration is better than the plan claims, because it is computed at a feasible `y`: top-1 row **14.0 %**
of the gain, top-3 **38.4 %** (the plan said 16.3 % / 46.1 % at `y = 1,000`).

**Against the slot.** v12 banked +0.3134 s with a *guaranteed* floor of +0.121 s
(`experiments/submissions.jsonl` v12 record). P6's entire best case is −0.094 s with no floor at all, and
it would consume upload v14 (**[S]** `prc.submit._today_count`: 1 of 5 used today, versions 1–13 taken).
**[X]** It is worth the slot only when nothing else is ready; it is not worth an hour of anyone's queue
ahead of P1, P4 or the CatBoost blend.

### 7. The thing nobody thought of

**(a) The OOF sign check is a theorem.** `P6-per-airport-upper-bound.md:380-387` offers "22 rows, 22 helped /
0 hurt, −3.47 s, negative in all four OOF months and at all five affected airports" as the rule's strongest
evidence, and calls it "a stronger object than the 49/4 of the item-13 cap".

It is not an object at all. `c = max{y : 2025 rows at that airport}`, so `y ≤ c` holds on **every** 2025 row
by construction, and `|min(p, c) − y| ≤ |p − y|` follows for every row. **Zero hurt rows is entailed, on any
model, on any data, at any airport.** This is the same defect `P5-REVIEW.md:165` found in the LOBT plan's
`P(better) = 1.000` headline: a test with zero power against the only failure mode that matters, presented
as the headline result.

The leave-one-month-out variant does not fix it at the two airports that matter: LEMD's and LEBL's 2025
maxima both fall in **October**, which is not an OOF month, so the held-out support is 3,598 / 3,604 in every
fold and the tautology stands. **[S]** `/tmp/p6rev/a6.py` — LOO run: 18 rows, still 18/0, −2.799 s.

**(b) The OOF rows are the wrong class.** **[S]** `/tmp/p6rev/a6.py` on `oof_main-48c19bc830` (672,428 scored
rows, RMSE 334.582): the four-airport rule fires on **18** rows, of which **16 have `nm_matched = 0`** —
unmatched predictions of 3,637–16,880 s against labels of 580–1,810 s. Only two are matched. All twelve rows
on the ranking file are matched. On `oof_submit_v10_reparam-77f92c802d` (328,009 rows, the v10 recipe) the
rule fires on **one** row — EDDM, matched, month 12 — and on **zero** LEMD or LEBL rows.

So: the fold evidence is 89 % drawn from the unmatched lane that the v10 reparameterisation eliminated (the
P1 measurement in the brief — 17 non-LIRF unmatched above 3,600 and none above 7,200, from 310/100 at v7 —
is the same fact seen from the other side), and there is **no fold observation of any sign** on the object
actually being shipped. The plan concedes the disjointness at `:210-212` and then reuses the 22/0 count as
evidence at `:386` anyway.

**(c) Gate and file mechanics — clean.** **[S]** `/tmp/p6rev/a7.py`: `outputs/v12_lobt_bound.parquet` has
344,841 rows, schema `{MVT_ID_mvt: double, TAXITIME_SEC_mvt: int32}` **equal** to `data/raw/submitting.parquet`
and `MVT_ID_mvt` in **identical order**. All twelve caps are integers, so the `int32` column is unaffected.
`validate_submission` (`src/prc/validate_submission.py:39-110`) passes: the max (112,771, LIRF) already
exceeds `MAX_PLAUSIBLE_SECONDS = 10800` on v10 and v12 and is a **warning only** (`:91`), and the median is
unchanged at 959.

**(d) No capped row was touched by the LOBT bound.** The 34 rows v12 moved are LFPG 18, EHAM 8, LTFM 3,
EGLL 2, LEBL 2, LEMD 1; the three at LEMD/LEBL are `213724395`, `213798774`, `213805661`, none of them among
the twelve, all raised, all finishing below 1,510 s. **[S]** `/tmp/p6rev/a2.py`.

**(e) The plan's checklist numbers are v10's and would fail on v12.** `:576` asserts "mean shift −0.022" with
"mean 1,039.3554 → 1,039.3337"; on v12 it is **1,039.3865 → 1,039.3649** (same −0.0216 shift, different
absolute values). A checklist that asserts the absolute figures blocks a correct build. `:570-571` asserts
the diff is against "the scored v10 file" — it must be against v12.

---

## (c) What I could not verify

1. **Whether the LEMD/LEBL departure ceiling holds in 2026.** Unverifiable in principle: `BLOCK_TIME_UTC_mvt`
   is null on all 344,841 2026 DEP rows (**[S]** `/tmp/p6rev/a8.py`), and 2026 arrival taxi-in at those
   airports maxes at 3,303 / 2,428 s, so it cannot witness a 3,600 s ceiling. The plan, `fenil264` and I all
   agree. The LOBT lower-bound channel (§5b) is the closest available proxy and it is adverse at LEMD.
2. **The `kind-mango` line numbers.** `/tmp/kind-mango` is not present on this machine; §2 of the plan is
   taken on trust. It is not load-bearing for the verdict.
3. **Whether v13 beats v12.** Unscored at review time. If it does, the base moves and the row set becomes 10,
   not 12 (**[S]** `/tmp/p6rev/a1.py`), and everything in §(b)6 must be recomputed.
4. **The true labels.** Nothing here is a measurement of the outcome; it is arithmetic over twelve known
   predictions under assumptions about twelve unknown labels.
5. **The plan's §3.5 same-day arrival cluster table** (the six LEMD/LEBL days) I did not re-derive
   day-by-day; the airport-level and month-level mirror reproduces its conclusion.

---

## (d) Required changes, and the corrected pre-registration

**Blocking (all five must be done before a build):**

1. **Source file.** Build from `outputs/v12_lobt_bound.parquet`. Assert in the build script that the source
   differs from `outputs/v10_probe_reparam.parquet` on **exactly 34 rows** and from
   `outputs/v11_rollover_mixture.parquet` on 45, and refuse otherwise. Do not identify the source by filename
   — `outputs/submit_v10_reparam.parquet` is still on disk and still 336,426 rows away from the truth.
2. **Re-register the price.** Replace "−0.20 s, range [−0.23, +0.24]" with **expected −0.06 s, feasible range
   [−0.094, +0.030], tail risk +0.46 s** (the LOBT-feasible worst case), on base 284.045 → 283.985 expected.
   State explicitly that `y = 1,000` is infeasible and why. A realised delta outside [−0.10, +0.46]
   falsifies the construction, not the rule.
3. **Rewrite §5.2 against the LOBT window.** Replace the AOBT gate with
   `assert max(mvt_minus_lobt − 3606, 0) ≤ S_a` on every row at the four airports (today: tightest margin
   **300 s at LEMD**, then LEBL 1,284, EDDF 2,111, EDDM 2,824). Fix the order to **cap first, LOBT bound
   second**, so that a future violation makes the cap inert rather than overriding a proven bound, and say so
   in the code comment.
4. **Delete the "22 helped / 0 hurt" claim from §4.1**, or demote it to "entailed by `y ≤ max(y_2025)`;
   carries no information". Replace the sentence "it is the rule I am proposing rather than a proxy for it"
   with the measured fact: 16 of 18 OOF rows are unmatched, all 12 ranking rows are matched, and the v10-recipe
   OOF fires on one EDDM row and zero LEMD/LEBL rows.
5. **Add the LOBT lower-bound transfer check to §3.6 and the checklist**, with the LEMD result stated as an
   adverse finding: 2026 max `lo` 3,298 vs 3,043 over all of 2025, rate above 3,000 s 0.81 vs 0.00 per 10,000,
   and three of the twelve rows carrying it. If LEMD's 2026 max `lo` ever reaches `S_a`, the rule is void at
   that airport.

**Strongly recommended:**

6. **Drop EDDF and EDDM, or give them a margin.** They are selected by a criterion that a single 2025 flight
   can flip (293 s / 301 s from 7,200), their maxima are sampled rather than censored (runner-up 341 s / 49 s
   below), and the rule binds on zero rows there. Shipping them adds no gain, adds two unjustified no-margin
   caps, and — via §5.1 step 2 — makes them permanent. If they are kept for "rule generality", carry
   `kind-mango`'s `+600` at those two airports only, and say in one line why LEMD/LEBL do not need it.
7. **Restate the selection rule.** "Zero 2025 labels above 7,200" is three EHAM rows away from selecting an
   airport the plan itself vetoes. Make the decision rule "censoring evidence (histogram step + 12 monthly
   maxima) **and** a clean 2026 arrival mirror **and** a clean 2026 LOBT lower-bound channel", which selects
   {LEMD, LEBL} and nothing else.
8. **Fix the checklist constants to v12's** (mean 1,039.3865 → 1,039.3649; diff target v12, not v10).
9. **Hold step 2** (folding the cap into `_assemble`) until the upload is scored. A permanently-on cap at four
   airports, two of them with no censoring evidence, is a larger commitment than the upload it is justifying.

**Corrected one-line pre-registration.**

> v14 = `outputs/v12_lobt_bound.parquet` with `pred := min(pred, S_a)` at `a ∈ {LEBL, LEMD}`
> (`S = 3604, 3598`), 12 rows changed, 344,829 bit-identical. Expected **−0.06 s** (284.045 → 283.985);
> feasible range **[−0.094, +0.030]** given `y ≥ max(mvt_minus_lobt − 3606, 0)` on every row; tail risk
> **+0.46 s** if the LEMD/LEBL 2025 departure ceiling was lifted for 2026. No guaranteed floor — unlike v12,
> this is not a projection onto a set proven to contain the truth.

*(Dropping EDDF/EDDM changes no row on this file and therefore no number above.)*

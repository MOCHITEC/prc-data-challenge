# P5 — the physics-identity bound on predictions

Plan, 2026-09-21. Theme P5. One change, pre-priced, effect isolated.

Marking: **[S]** = sourced (a file:line in this tree, or a measurement I ran against
`data/` and `outputs/` with the script named). **[X]** = my inference or arithmetic on
[S] numbers.

Scratch scripts, read-only, written outside the repo:
`/tmp/claude-0/-home-user-prc-data-challenge/ca703460-62c0-5c84-91fe-d76c3d0e1e20/scratchpad/p5_{measure,oof,rank,drift,price,gate,mech,g,worst,ratio}.py`.
No file under `src/` or `configs/` was modified. No submission was built or sent.

---

## 0. The headline, before the argument

**[X] The rule the research round recommended is the wrong one for us, and the measurement
says so unambiguously. The rule it recommended second is the right one, and it is the first
change in this repository's history to pass `prc.compare` at all four drop levels with
P(better) = 1.000.**

| rule | our 3-fold OOF, K=0 | our 3-fold OOF, K=200 | verdict under our rule |
|---|---|---|---|
| **A** — sqcm9896's AOBT floor `m_aobt > 1800 → max(pred, m_aobt − 3600)` | **+0.0985** [+0.0584,+0.1481] | **+0.1465** [+0.0839,+0.2130] | **REJECT** (P(better) = 0.000 at every level) |
| **B** — javidmardanov's LOBT window `clip(pred, m_lobt − 3606, m_lobt + 3606)` | **−0.0993** [−0.2037,−0.0258] | **−0.0941** [−0.1759,−0.0275] | **ADOPT** (P(better) = 1.000 at every level) |

[S] Produced by calling this repo's own `prc.compare.robustness_curve` on
`experiments/oof_main-48c19bc830.parquet` and `experiments/oof_submit_v10_reparam-77f92c802d.parquet`
(`p5_gate.py`). Full four-level tables in §2.4 and §3.4.

The plan is therefore: **ship rule B alone, as a post-prediction clip, as one isolated probe
submission. Do not ship rule A.** Pricing in §4, placement in §5, failure mode in §6.

---

## 1. First principles: what each rule actually asserts

### 1.1 The label identity, re-verified here

[S] `TAXITIME_SEC_mvt == MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt` on **2,085,047 / 2,085,047**
training DEP rows, 0 violations, 0 nulls (`p5_measure.py`). This reproduces
`docs/CODEX_BRIEF.md:11-12`. Write it as

```
y  =  MVT − BLOCK                                                    (I)
```

with `MVT = MVT_TIME_UTC_mvt` and `BLOCK = BLOCK_TIME_UTC_mvt`, the blanked column.

### 1.2 Rule A: what `m_aobt − 3600` asserts

Let `m_aobt = MVT − AOBT_3_flt`. Substituting (I):

```
y − m_aobt  =  (MVT − BLOCK) − (MVT − AOBT_3)  =  AOBT_3 − BLOCK  ≡  g            (II)
```

So `pred ≥ m_aobt − 3600` is **exactly** the assertion

```
g  =  AOBT_3 − BLOCK  ≥  −3600      i.e.   BLOCK  ≤  AOBT_3 + 3600                (III)
```

— "the actual off-block was at most one hour after the Network Manager's recorded off-block".

**[S] This is not an identity. It is false on our data.** Distribution of `g` over the
2,062,577 training rows with `AOBT_3_flt` (`p5_measure.py`):

| quantile | g (s) |
|---|---|
| min | **−9,069** |
| 1e-5 | −5,178 |
| 1e-4 | −3,993 |
| 1e-3 | −2,100 |
| median | −51 |
| 1−1e-3 | +3,001 |
| max | **+20,657** |

mean −17.03, sd 384.47. Violations of (III): **620 rows (0.0301 %)** have `g < −3600`;
14 rows have `g < −5400`; 4 have `g < −7200`. So **3,600 is an empirical quantile of
`AOBT_3 − BLOCK`, sitting at about the 0.03 percentile — not a bound derived from anything.**
[X] The correct label for rule A is "a one-sided winsorisation at a hand-picked quantile",
which is exactly the class of object five of our board measurements refute (§1.5).

### 1.3 Rule A's `1800` gate is inert

[X] `max(pred, m_aobt − 3600)` can only bind when `m_aobt − 3600 > pred`. Our pipeline
already floors every prediction at zero — `src/prc/train.py:433`,
`.clip(lower_bound=0.0)` [S]. So the rule is vacuous whenever `m_aobt ≤ 3600`, because the
floor it proposes is then ≤ 0. The gate at 1,800 therefore does nothing at all: **any gate
value in [−∞, 3600] produces the identical rule.**

[S] Confirmed empirically: on all three OOF files and on the ranking set, dropping the
`m_aobt > 1800` gate entirely changes the bound-row count by **exactly 0**
(`p5_oof.py`, line `[A'] same with NO 1800 gate: binds … (delta vs A: 0)`).

**[X] Verdict on the constants: `1800` is arbitrary and provably inert; `3600` is arbitrary
and is the whole rule. Neither is principled. Re-deriving `3600` on our data would just be
choosing a different quantile of `AOBT_3 − BLOCK` by hand, which is a fit, not a bound.**

### 1.4 Rule B: what `±3606` asserts, and why it *is* an identity

Let `m_lobt = MVT − LOBT_flt`. Then

```
(MVT − LOBT) − y  =  (MVT − LOBT) − (MVT − BLOCK)  =  BLOCK − LOBT  ≡  d           (IV)
```

So `|(MVT − LOBT) − TAXITIME| ≤ 3606` is **exactly** `|BLOCK − LOBT| ≤ 3606`: the actual
off-block never departs from the last filed off-block time by more than one hour and six
seconds, in either direction. [X] That is an operational rule of the Network Manager flight
plan (a slot slipping past an hour forces a refile), and the `+6` is the tolerance the
recording system carries.

**[S] Zero violations, and the bound is exactly tight, on our data** (`p5_measure.py`,
`p5_price.py`):

- `|d| > 3606`: **0 of 2,062,577** rows with `LOBT_flt`.
- `max |d| = 3606.0`, attained. **14 rows sit at exactly d = +3606, 4 rows at exactly
  d = −3606, and 0 rows lie in |d| ∈ [3607, 4000].** The distribution stops dead: 500 rows
  in [3550, 3599], 292 in [3600, 3605], 18 at exactly 3606, then nothing.
- Zero violations in **every one of the 12 training months**; max |d| is 3606 in 9 months and
  3604–3605 in the other 3.
- Zero violations at **every one of the 10 airports**; max |d| is 3602–3606 everywhere.
- **Jan 2025: 0 violations in 152,271 rows. Jul 2025: 0 violations in 186,775 rows** — the
  two calendar months that are actually scored.

[X] A hard edge with mass exactly on it and nothing beyond it is the signature of a clamp
inside the source system, not a heavy-tailed distribution that happens to stop. **3606 is
the system's constant, not a quantile we chose. Re-deriving it on our data returns 3606
exactly.** Copying it is correct; tuning it would be strictly worse, because any wider value
weakens the bound and any narrower value makes it false.

[S] This independently reproduces javidmardanov's claim
(`docs/competitor_research_2026-09-21/results/08-repo-javidmardanov.md:91-93`) on our own
data, and extends it: they verified it on non-{7,11,12} months; **it holds on all twelve of ours.**

### 1.5 Bounding the PREDICTION is a different operation from cleaning the LABEL

[X] This has to be said explicitly, because five board measurements in our research round
point the other way and they are all about something else.

[S] The five: drop training rows > 120 min **+24.1**; drop > 60 min **+36.3**; cap
predictions at 7,200 s **+77.6**; cap at 4,500 s **+96.1**; drop 17 rows with taxi > 6 h
**+13.7** — `docs/competitor_research_2026-09-21/results/09-repo-sergiuv11.md:626, 638, 755-757`.
Plus sqcm9896's global upper clip at **+90.7**
(`.../24-repo-tail-sweep.md:296-297`).

[X] Every one of those six is a **constant, global, label-space** operation: one number
applied to all 344,841 rows (or to the training labels), chosen because large values are
inconvenient. They fail for one reason — the large values are real, and squaring makes
truncating them catastrophic.

Rule B is none of those things:

| axis | the six refuted operations | rule B |
|---|---|---|
| bound value | one global constant | **per row**, `m_lobt ± 3606` — a different interval on every row |
| derivation | chosen to make the tail smaller | derived from `y = MVT − BLOCK` and `\|BLOCK − LOBT\| ≤ 3606` |
| relation to truth | the truth is often outside it | the truth is **provably inside it** on 2,062,577/2,062,577 rows |
| what it does to a tail row | pulls it toward the bulk | leaves it alone unless it is outside that row's own window, and then moves it **to the nearest point of the window, which still contains the truth** |
| direction | one-sided, always toward the bulk | two-sided, and 43 of 47 binding rows move **upward** (§4.2) |

**[X] The decisive structural property.** Clipping onto an interval is the metric projection
onto a convex set. If the truth `y` lies in `[lo, hi]`, then for `r = clip(p, lo, hi)`,
`|r − y| ≤ |p − y|` for **every** row, with equality only where the rule does not bind.
Projections are non-expansive and `y` is a fixed point of the projection. So conditional on
the bound holding, `ΔSSE ≤ 0` **row by row, with certainty** — there is no hit rate to
estimate, no sign to get wrong, no expectation to take. The six refuted operations have no
such property because their intervals do not contain the truth.

[S] Our own OOF confirms the arithmetic: **85 of 85 bound rows helped, 0 hurt** on
`oof_main-48c19bc830`; **28 of 28 helped, 0 hurt** on `oof_submit_v10_reparam-77f92c802d`
(`p5_oof.py`). A 100 % hit rate is not luck; it is the projection property.

### 1.6 Why this is not our v11

[S] Our v11 rollover-mixture overlay scored **293.978** against a **+8.377** floor implied by
applying it to zero rows — i.e. it lost more than doing nothing would have cost
(`docs/competitor_research_2026-09-21/results/15-postprocess-overlays.md:537, 550`; the same
file's diagnosis: "the selection was wrong, not the arithmetic").

[X] Four differences, each checkable before upload:

1. **v11 selected rows by a model-side belief** (which rows look like day-rollover rows).
   Rule B selects rows by an **inequality between two supplied timestamps** — `LOBT_flt`
   and `MVT_TIME_UTC_mvt`, both present on 100 % of the rows it touches. No model output
   enters the selection.
2. **v11 replaced a prediction with a different quantity.** Rule B **projects** a prediction
   onto an interval. It cannot move a row past the interval boundary, so the move size is
   bounded above by `|p − boundary|`, which we can enumerate before uploading (§4.2).
3. **v11's row set could not be priced in advance** — we did not know the truth and the
   replacement was unbounded. Rule B's **worst case and guaranteed case are both computed
   below from the submission file alone**, with no label needed (§4.3, §4.4).
4. **v11 failed our own gate's spirit and was shipped anyway.** Rule B passes `prc.compare`
   at K = 0, 20, 200 and 2000 with P(better) = 1.000 at every level (§3.4).

---

## 2. Rule A, measured on our data

### 2.1 Why it transferred for sqcm9896 and will not for us — the mechanism

[S] sqcm9896 fit the **raw duration**: `src/config.py:14`, `TARGET = "TAXITIME_SEC_mvt"`,
"Model raw target for RMSE" (`.../24-repo-tail-sweep.md:276-278`). `m_aobt` enters their
model only as a *feature*.

[S] We do the opposite. `src/prc/train.py:3-4`:

> Model A (rows with an NM off-block time): learn g(x) = TAXITIME - (MVT_TIME - AOBT_3) and
> predict proxy + g(x). **A tree cannot represent the identity part of y = proxy - f(x), so
> the reparameterisation is the point.**

[S] `src/prc/train.py:40` `PROXY = "mvt_minus_aobt3"`; `src/prc/train.py:385` fits
`LABEL − PROXY`; `src/prc/train.py:396` emits `m_tg[PROXY] + g`.

[X] So for them, `pred < m_aobt − 3600` was mostly a *tree failing to track the identity* —
a modelling defect the floor repaired. For us, `pred = m_aobt + ĝ(x)`, so
`pred < m_aobt − 3600` happens **only when the model has positively decided `ĝ(x) < −3600`**,
i.e. when it has actively concluded the off-block was very late. There is no identity for
the floor to restore; it can only overrule a decision.

[S] The calibration check confirms the model has learned this quantity well. `ĝ = pred − m_aobt`
on the 339,551 matched ranking rows of `outputs/submit_v10_reparam.parquet`, against the true
`g` distribution from §1.2 (`p5_g.py`):

| | model ĝ (2026 ranking) | true g (2025 training) |
|---|---|---|
| mean | −8.04 | −17.03 |
| sd | 346.00 | 384.47 |
| q1e-4 | −3,987 | −3,993 |
| q1e-3 | −3,082 | −2,100 |
| min | −5,497 | −9,069 |

[X] The model's ĝ tracks the real distribution closely, including its left tail. It emits
`ĝ < −3600` on 193 ranking rows against an expected ~102 at the training base rate — it
over-emits by ~1.9×, but flooring those rows is the wrong repair, because **82 % of the rows
it would floor are rows where the truth really is below the floor** (§2.2).

### 2.2 Rule A on our OOF: it moves the wrong rows

[S] `p5_oof.py`, rule A exactly as specified (`m_aobt > 1800 → max(pred, m_aobt − 3600)`):

| OOF file | n | binds | helped | hurt | **truth above the floor** | ΔSSE | ΔRMSE |
|---|---|---|---|---|---|---|---|
| `oof_submit_v10_reparam-77f92c802d` | 328,009 | 90 (0.027 %) | 21 | **69** | **16 / 90** | **+3.255e7** | **+0.2219** |
| `oof_main-48c19bc830` | 672,428 | 166 (0.025 %) | 49 | **117** | **34 / 166** | **+4.432e7** | **+0.0985** |
| `oof_reparam_lirf-41526561b0` | 328,009 | 92 (0.028 %) | 22 | **70** | **17 / 92** | **+3.247e7** | **+0.2213** |

[X] Read the "truth above the floor" column. The base rate of `g < −3600` is 0.030 %
(§1.2). Among the rows where **our** model predicts below the floor, the truth is below the
floor **82 %** of the time. The model is a very good detector of exactly the population the
floor was designed to erase. Rule A is a rule that overrules a correct detector.

[S] Single-row-share check (the Repo A guard,
`.../09-repo-sergiuv11.md:747`): removing the top-1 and top-3 contributing rows makes rule A
**worse**, not better — ΔSSE +3.255e7 → +3.328e7 (excl. top-1) → +3.397e7 (excl. top-3) on the
v10-lineage OOF. There is no single row carrying the loss; the loss is the rule.

### 2.3 Rule A on the ranking set

[S] `p5_rank.py`, against `outputs/submit_v10_reparam.parquet` (our 284.3584 file):

- binds on **193 rows** (0.0560 %), all matched rows, none unmatched;
- moves them **up** by median 169 s, mean 240 s, max 1,897 s;
- Σ(move)² = **2.3687e7** → if the model were right on all 193, **+0.1208 s**.

[X] Given §2.2 says the model is right about 82 % of the time on exactly this row set, the
central estimate for rule A on the leaderboard is roughly **+0.10 to +0.25 s** — a loss of
the same size our OOF measured. There is no scenario in the measurement where rule A helps us.

### 2.4 Rule A under `prc.compare`

[S] `p5_gate.py`, this repo's own `robustness_curve`:

```
oof_main-48c19bc830.parquet  n=672428   [A] AOBT floor
  K=    0  A  334.5821  B  334.6806  diff  +0.0985  CI +0.0584,+0.1481  P(B better) 0.000
  K=   20  A  251.7702  B  251.9011  diff  +0.1309  CI +0.0738,+0.1921  P(B better) 0.000
  K=  200  A  224.9465  B  225.0930  diff  +0.1465  CI +0.0839,+0.2130  P(B better) 0.000
  K= 2000  A  197.8261  B  197.8807  diff  +0.0546  CI +0.0296,+0.0819  P(B better) 0.000
```

**REJECT at every level.** The effect does not vanish at K=200 — it *grows*, which means it
is a broad property of the rule, not a tail lottery. [X] That is the strongest possible
rejection our instrument can produce.

**[X] Conclusion on theme P5's headline item: rule A is refuted on our data, for a stated
mechanism (our model already carries the identity that rule A exists to restore), and the
refutation is robust to every drop level. Do not ship it. This is the correct outcome of
stealing a competitor's rule: we measured it instead of copying it.**

---

## 3. Rule B, measured on our data

### 3.1 Coverage — and the one thing it cannot reach

[S] `p5_rank.py`, `p5_measure.py`:

| | training DEP | ranking DEP |
|---|---|---|
| rows | 2,085,047 | 344,841 |
| `AOBT_3_flt` present | 2,062,577 | 339,551 |
| `LOBT_flt` present | 2,062,577 | 339,551 |
| **`AOBT_3` null AND `LOBT` present** | 0 | **0** |

**[S] `LOBT_flt` is null on exactly the rows `AOBT_3_flt` is null on — all 5,290 unmatched
ranking rows. Rule B touches zero model-B rows.**

[X] This is the single most important limitation and it must be stated up front: model B
carries **48.1 %** of squared error on 1.53 % of rows (`docs/handoff.md` §2), and
rule B cannot reach any of it. This is a matched-rows-only, model-A-only change. It is not
the 20.3-second transfer sqcm9896 got, and nothing in this plan claims it will be.

### 3.2 The mechanism: which rows it acts on, and why the LOBT clock wins there

[X] Rule B can only bind where the model's own anchor disagrees with the window. Since
model A predicts `m_aobt + ĝ` with ĝ tightly centred (sd 346), binding essentially requires
`m_aobt` itself to lie outside `[m_lobt − 3606, m_lobt + 3606]` — i.e. the two clocks
disagree by more than the physical bound.

[S] That population, measured on 2025 (`p5_mech.py`):

- **4,011 of 2,062,577 rows (0.1945 %)** have `|AOBT_3 − LOBT| > 3606`.
- On them, predicting `y` by `m_aobt` gives **RMSE 3,166**; predicting by `m_lobt` gives
  **RMSE 2,572**; predicting by `clip(m_aobt, m_lobt ± 3606)` gives **RMSE 2,269**.
- For contrast, on the whole 2025 set: `m_aobt` **RMSE 384.9**, `m_lobt` **RMSE 740.1**.

**[X] This is the mechanism in one line: `AOBT_3` is by far the better clock overall
(385 vs 740), but on the 0.19 % of rows where it contradicts the physical bound it is
catastrophically worse (3,166), and clipping it into the window cuts that by 28 %.** Rule B
is a targeted repair of the proxy on the rows where the proxy is provably wrong.

[S] The 2026 counterpart of that population: **956 of 339,551 ranking rows (0.28 %)** have
`|AOBT_3 − LOBT| > 3606` — 446 in January, 510 in July (`p5_drift.py`). Slightly denser than
2025's 0.19 %, same shape.

### 3.3 Rule B on our OOF

[S] `p5_oof.py`, `clip(pred, max(m_lobt − 3606, 0), m_lobt + 3606)`:

| OOF file | n | binds (lower / upper) | helped | hurt | ΔSSE | ΔRMSE |
|---|---|---|---|---|---|---|
| `oof_submit_v10_reparam-77f92c802d` | 328,009 | 28 (21 / 7) | **28** | **0** | −2.399e7 | **−0.1637** |
| `oof_main-48c19bc830` | 672,428 | 85 (74 / 11) | **85** | **0** | −4.467e7 | **−0.0993** |
| `oof_reparam_lirf-41526561b0` | 328,009 | 26 (19 / 7) | **26** | **0** | −2.372e7 | **−0.1617** |

Hit rate **100 % on every file**, as §1.5's projection argument requires.

[S] **Realised vs guaranteed** (`p5_ratio.py`). The guaranteed floor on a bound row is
`(r − p)²` (see §4.3); the realised gain is always at least that, and here much more:

| OOF file | Σ(move)² (guaranteed) | realised ΔSSE | ratio | rows exceeding the floor |
|---|---|---|---|---|
| `oof_submit_v10_reparam` | −3.465e6 | −2.399e7 | **6.92×** | 28 / 28 |
| `oof_main-48c19bc830` | −8.379e6 | −4.467e7 | **5.33×** | 85 / 85 |

### 3.4 Rule B under `prc.compare`

[S] `p5_gate.py`:

```
oof_main-48c19bc830.parquet  n=672428   [B] LOBT clip
  K=    0  A  334.5821  B  334.4828  diff  -0.0993  CI -0.2037,-0.0258  P(B better) 1.000
  K=   20  A  251.7702  B  251.6382  diff  -0.1320  CI -0.2455,-0.0419  P(B better) 1.000
  K=  200  A  224.9465  B  224.8523  diff  -0.0941  CI -0.1759,-0.0275  P(B better) 1.000
  K= 2000  A  197.7801  B  197.7605  diff  -0.0196  CI -0.0314,-0.0117  P(B better) 1.000

oof_submit_v10_reparam-77f92c802d.parquet  n=328009   [B] LOBT clip
  K=    0  A  223.5054  B  223.3417  diff  -0.1637  CI -0.3968,-0.0110  P(B better) 1.000
  K=   20  A  207.9572  B  207.8863  diff  -0.0709  CI -0.1671,-0.0060  P(B better) 1.000
  K=  200  A  197.3659  B  197.3571  diff  -0.0088  CI -0.0149,-0.0038  P(B better) 1.000
  K= 2000  A  177.0122  B  177.0041  diff  -0.0082  CI -0.0145,-0.0033  P(B better) 1.000
```

[S] The adopt rule as implemented: `verdict()` reads the K=0 CI first and rejects if it
crosses zero; otherwise both K=0 and K=200 must be on the improving side
(`docs/README.md` §6 訂正, `docs/investigation_2026-09-16/results/decision-rule-blind-to-tail.md` §3.1, §4).

**ADOPT on both files.** [X] All four CIs are strictly negative on both files, and
P(better) = 1.000 at all eight cells — which is what a rule with a 100 % per-row hit rate
must produce, and a property no fitted change in this repository has ever had.

[X] Seasonal arm (the sqcm9896 Jan-AND-Jul conjunction,
`.../24-repo-tail-sweep.md:125-133`): it is satisfied trivially and for the structural
reason. Because the sign is guaranteed per row, **no** subset of rows can regress — not a
month, not an airport, not a fold. There is no month-gate to run.

### 3.5 Can the bound be verified on 2026? No — state this plainly

[S] `p5_drift.py`. `|AOBT_3 − LOBT|` is observable in both years and is **not** bounded at
3606: max 20,421 in 2025, max 11,160 in 2026; 0.194 % of 2025 and 0.282 % of 2026 rows exceed
3606; **0 rows sit at exactly ±3606**. So the clamp lives specifically on
`BLOCK_TIME_UTC_mvt` versus `LOBT_flt`, and `BLOCK_TIME_UTC_mvt` is blanked on all 344,841
scored rows (`docs/CODEX_BRIEF.md:25`, "0 %").

**[X] There is therefore no way to verify the bound on 2026 data.** Arrival rows in the
ranking set keep `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` (`docs/CODEX_BRIEF.md:31`), but
for an arrival `BLOCK` is in-block at the destination while `LOBT` is off-block at the origin,
so no analogous bound exists to check. The evidence for 2026 is entirely:

1. 0 violations in 2,062,577 rows across all 12 months and all 10 airports of 2025, with the
   distribution terminating exactly at the constant (§1.4);
2. javidmardanov's independent verification on their own reconstruction of the same data
   (`.../08-repo-javidmardanov.md:91-93`), and their shipping it as a hard clip in a
   scored pipeline;
3. [S] the recording instrument is otherwise stable 2025 → 2026 — "schema, null rates, ID
   encodings, closed vocabularies, and the NM join rate are all inside their 2025 bands"
   (`docs/CODEX_BRIEF.md`, §8, citing `results/recording-change.md`);
4. [S] `m_lobt = MVT − LOBT` has the same shape in both years: 2025 median 1,237 / q.99 3,840
   / min −3,382; 2026 median 1,263 / q.99 4,138 / min −3,173 (`p5_drift.py`). Rows where the
   floor is positive (`m_lobt > 3606`): 1.41 % of 2025, **2.11 % of 2026**.

[X] That is a strong prior and not a proof. §6 prices the case where it is wrong.

---

## 4. The arithmetic: what this is worth

### 4.1 The exchange rate

[S/X] At RMSE = 284.3584 over N = 344,841 rows (`.../15-postprocess-overlays.md:6-19`;
row count confirmed on every submission by the leaderboard API,
`.../24-repo-tail-sweep.md:15`):

```
MSE  = 284.3584²                     =        80,859.702
SSE  = MSE × 344,841                 =  2.788374e10
```

Cost of one second of RMSE, from 284.3584 down to 283.3584:

```
ΔSSE(1 s)    = (284.3584² − 283.3584²) × 344,841  =  1.9577e8
ΔSSE(0.1 s)  =                                       1.9608e7
ΔSSE(0.01 s) =                                       1.9612e6
```

[X] **One second of RMSE costs 1.9577 × 10⁸ of total squared error.** Equivalently: one row
whose error falls from 13,992 s to 0; or 20 rows each falling from 3,129 s to 0. Any overlay
that moves a few dozen rows by a few hundred seconds is worth thousandths of a second. This
is the number that disciplines everything below.

### 4.2 What rule B does to our 284.3584 file — enumerated, no labels needed

[S] `p5_rank.py`, `p5_break.py`, applying `clip(pred, max(m_lobt−3606,0), m_lobt+3606)` to
`outputs/submit_v10_reparam.parquet`:

- rows where the lower bound is even positive (`m_lobt > 3606`): **7,166**;
- of those, the prediction is **already inside** the window on **7,123**;
- **47 rows move** (0.0136 % of the file): **43 up** to the lower bound, **4 down** to the
  upper bound;
- moves: median **94 s**, mean **248 s**, max **2,881 s**;
- **0 unmatched rows touched** (all 5,290 keep their v10 value bit-for-bit);
- **0 rows where `lo > hi`** (min `m_lobt` = −3,173, so `hi` is always positive);
- **all 10 airports and both months** are represented.

Per airport-month, ordered by Σ(move)²:

| airport | month | rows | Σ(move)² | max move |
|---|---|---|---|---|
| LIRF | 1 | 1 | 8.300e6 | 2,881 |
| LFPG | 7 | 8 | 8.284e6 | 2,631 |
| LTFM | 1 | 2 | 3.507e5 | 571 |
| EHAM | 1 | 10 | 3.089e5 | 455 |
| EGLL | 7 | 2 | 7.38e4 | 221 |
| … (11 more cells: EGLL/1, LEMD/7, LSZH/7, EDDF/1, LSZH/1, EDDM/1, LIRF/7, EDDF/7, …) | | | | |

The twelve largest movers, with both clocks shown (`p5_rank.py`):

```
LIRF mo1 pred 4552 -> 7433   m_lobt 11039   m_aobt  1199
LFPG mo7 pred 3960 -> 6591   m_lobt 10197   m_aobt   897
LFPG mo7 pred    0 -> 1132   m_lobt  4738   m_aobt -3302
EHAM mo1 pred 2239 -> 2694   m_lobt  6300   m_aobt  4140
EGLL mo7 pred 1871 -> 2092   m_lobt  5698   m_aobt  2278
LFPG mo7 pred  884 -> 1076   m_lobt  4682   m_aobt  1142
LSZH mo7 pred  950 -> 1136   m_lobt  4742   m_aobt   482
EHAM mo1 pred 1432 -> 1610   m_lobt  5216   m_aobt  1736
EGLL mo1 pred 2033 -> 2211   m_lobt  5817   m_aobt  2097
LTFM mo1 pred 1102 -> 1259   m_lobt  4865   m_aobt  1445
LFPG mo7 pred  567 ->  718   m_lobt  4324   m_aobt   604
LEMD mo7 pred 2735 -> 2878   m_lobt  6484   m_aobt   964
--- the four downward moves ---
LTFM mo1 pred 1778 -> 1207   m_lobt -2399   m_aobt   901
EHAM mo1 pred 2466 -> 2268   m_lobt -1338   m_aobt -2628
LFPG mo1 pred 1636 -> 1448   m_lobt -2158   m_aobt  3362
EGLL mo7 pred 1604 -> 1446   m_lobt -2160   m_aobt  1500
```

[X] Note the top two rows: `m_aobt` and `m_lobt` disagree by 9,840 s and 9,300 s. These are
exactly the §3.2 population where 2025 says the LOBT clock is the reliable one.

### 4.3 The guaranteed case

[X] For a lower-binding row (`p < lo ≤ y`), with `r = lo`:

```
ΔSSE  =  (lo − y)² − (p − y)²  =  (lo − p)·(lo + p − 2y)   and   lo + p − 2y ≤ p − lo < 0
   ⇒  ΔSSE  ≤  −(lo − p)² ,  with equality exactly at y = lo.
```

So `Σ(move)²` is not the risk — it is the **minimum gain**, attained in the least favourable
admissible position of the truth. Symmetrically for upper binds.

[S] `p5_price.py`: Σ(move)² over the 47 rows = **1.7567e7** (1.7141e7 from the 43 lower,
4.2555e5 from the 4 upper).

```
guaranteed ΔSSE  ≤  −1.7567e7   ⇒   RMSE gain  ≥  0.0896 s
                                     284.3584  →  ≤ 284.2688
```

[X] **Conditional on `|BLOCK − LOBT| ≤ 3606` holding on those 47 rows, we cannot score worse
than 284.2688. There is no distributional assumption in that statement.**

### 4.4 The expected case

Three independent estimates, all [X] on [S] numbers:

1. **Truth uniform-ish within the window.** `p5_price.py` places the truth at a fixed
   fraction of `[lo, hi]` on each bound row:
   - at the quarter point: ΔSSE −5.639e7 → **+0.2877 s**
   - at the midpoint: ΔSSE −9.521e7 → **+0.4859 s**
   - at the far end: ΔSSE −1.728e8 → **+0.8827 s**
2. **Transfer the OOF multiplier.** Our own OOF realised **5.33×** and **6.92×** the
   guaranteed floor (§3.3). Applied to 1.7567e7: ΔSSE −9.4e7 to −1.22e8 → **+0.48 to +0.62 s**.
3. **Scale the OOF gain by bind density.** v10-lineage OOF: −0.1637 s on 328,009 rows with
   28 binds. The ranking has **47 binds on 344,841 rows** (1.6× the density) and larger moves
   (max 2,881 vs 1,422). [X] Naive scaling gives ~**+0.3 s**, but this estimate is the weakest
   of the three because the OOF baseline RMSE is 223.5 against a board 284.4, so a given ΔSSE
   buys different seconds.

**[X] Central estimate: +0.3 to +0.6 s of RMSE. Floor: +0.0896 s. Both are small.** Say so
plainly: this change is worth roughly one-fortieth of the 39-second gap to the leader. It is
worth shipping because it is free of downside, not because it is large.

### 4.5 The worst case, priced three ways

[S] `p5_worst.py`. All assume the bound is invalid on **every one of the 47 rows**:

| assumption about the truth on the 47 moved rows | ΔSSE | ΔRMSE |
|---|---|---|
| the model was exactly right (`y = p`) | +1.757e7 | **+0.0896** |
| `y = 900 s` (the global median label) on all 47 | +5.531e7 | **+0.2819** |
| `y = 0` on all 47 (impossible: min 2025 label is −12) | +7.227e7 | **+0.3683** |

[X] **The absolute worst case in the universe of admissible truths is +0.3683 s.** For
comparison, v11's overlay cost us **+9.62 s** against v10 (293.978 vs 284.3584,
`.../15-postprocess-overlays.md:537`). Rule B's total downside is 26× smaller than the
loss we have already eaten once, and its upside is guaranteed rather than hoped for.

### 4.6 Concentration — the honest weak point

[S] `p5_break.py`, on the guaranteed floor of 1.7567e7:

- **top-1 row supplies 47.2 %**; top-3 supply **93.9 %**;
- gain excluding top-1: **+0.0473 s**; excluding top-3: **+0.0054 s**.

[X] This **fails the spirit of the single-row-share guard** (Repo A: reject a candidate where
one row supplies > 50 % of the net advantage —
`.../09-repo-sergiuv11.md:747`), and it is within a rounding error of failing it literally.
The guard must be answered, not waved away. The answer is that the guard exists to catch
candidates whose *sign* depends on one row. Here the sign does not: **every** one of the 47
rows has ΔSSE ≤ 0 by construction (§4.3), and the OOF measured 113 of 113 bound rows helping
across two files. The concentration affects only **how much** we gain, not **whether**. The
guard's diagnostic reading is therefore: *"the realised number will be dominated by two rows
at LIRF/Jan and LFPG/Jul, so do not over-interpret a 0.05 s outcome as a failure of the rule."*

[X] The same concentration is the whole of the downside in §4.5: the top-1 row alone costs
**+0.1760 s** if its bound is violated and the truth is 0, and the top-2 together cost
+0.3175 s of the +0.3683 s worst case.

---

## 5. The plan

### 5.1 Where a prediction-side bound belongs in this codebase

[S] There is exactly one place today where a post-prediction, prediction-space constraint is
applied: `src/prc/train.py:433`, inside `_assemble()`:

```python
.with_columns(pl.col("pred").fill_null(fallback).fill_nan(fallback).clip(lower_bound=0.0))
```

[S] `_assemble` is the join point for both regimes' predictions (`src/prc/train.py:424, 427`)
and runs on every path — CV folds via `predict_fold`, and the ranking prediction via
`predict_ranking` → `predict_fold` (`src/prc/predict.py:31`). [S] `src/prc/train.py:411`
already names it as the downstream consumer the model-B reparameterisation must hand
seconds-of-taxi-time to: "nothing downstream (`_assemble`'s clip, the metrics, the
submission) knows it."

**[X] The bound belongs in `_assemble`, alongside the existing zero floor, and nowhere else.**
Reasons:

1. It is the only function both CV and ranking pass through, so the OOF numbers in §3 and the
   shipped file are produced by the same code. A bound applied only in `predict.py` would be
   unmeasurable by `make compare`.
2. The zero floor there is the same kind of object — a prediction-space constraint independent
   of the model — so the two compose naturally: `clip(pred, max(m_lobt − 3606, 0), m_lobt + 3606)`
   subsumes the existing `lower_bound=0.0` on rows with LOBT and leaves it in force elsewhere.
3. `_assemble` receives `target_f`, the frame, so `mvt_minus_lobt` is available without a new
   join — it is already a built feature (`src/prc/features.py:65`,
   `_sec("MVT_TIME_UTC_mvt", "LOBT_flt").alias("mvt_minus_lobt")`).

Shape of the change (not applied — this plan writes no code):

- a `TrainConfig` flag, default **off**, e.g. `lobt_bound: bool = False` plus
  `lobt_bound_slack: float = 3606.0`, so the default build stays bit-identical;
- in `_assemble`, when the flag is on and `mvt_minus_lobt` is present: `lo = max(m_lobt − slack, 0)`,
  `hi = m_lobt + slack`, `pred = clip(pred, lo, hi)` on rows where `m_lobt` is non-null,
  `pred = max(pred, 0)` elsewhere (unchanged);
- a guard that refuses `lo > hi` (impossible on current data — min `m_lobt` is −3,173 in 2026
  and −3,382 in 2025 — but a silent `clip` with inverted bounds is a submission-destroying bug);
- `info["lobt_bound_rows"]`, `info["lobt_bound_up"]`, `info["lobt_bound_down"]`,
  `info["lobt_bound_max_move"]` recorded, so the label-free gate of §5.4 reads them;
- a test that the flag off is bit-identical, and a test on synthetic data that a row outside
  the window is moved to the boundary and a row inside is untouched. [S] `docs/guard_audit.md`
  records that all nine existing invariants have a mutation test; a tenth constraint needs one.

### 5.2 Interaction with the model A / model B routing

[S] The routing key is `PROXY = "mvt_minus_aobt3"` null-ness: `src/prc/train.py:379`
(matched) and `:398` (unmatched). [S] `LOBT_flt` is null on exactly the same rows as
`AOBT_3_flt` — 0 exceptions in either direction, both sets (§3.1).

[X] Therefore:

- **Rule B applies to model A's output only, and to 100 % of it.** Every one of the 339,551
  matched ranking rows has a well-defined window; every one of the 5,290 unmatched rows has
  none and keeps its prediction unchanged.
- The two never conflict: `_assemble` runs after the concat, so one expression covers both
  regimes with the null branch doing nothing.
- [X] **Corollary worth naming: the bound cannot help where our error is.** Model B carries
  48.1 % of squared error (`docs/handoff.md` §2) and rule B is blind to it. If we later want a
  bound for model B we need a different anchor; `SCHED_TIME_UTC_mvt` is present on 100 % of
  ranking rows (`docs/CODEX_BRIEF.md:23`) but there is no clamp on `|BLOCK − SCHED|` — that is
  a separate investigation, not this plan.

### 5.3 Interaction with the LIRF reparameterisation

[S] The reparameterisation is `unmatched_sched_reparam` with
`unmatched_sched_reparam_airports: [LIRF]` (`configs/main.yaml`,
`configs/submit_v10_reparam.yaml`); it shifts **model B's** target by `mvt_minus_sched` at
LIRF and adds it back before `_assemble` (`src/prc/train.py:412-414`, and the comment at
`:410-411` stating the result is already back in seconds of taxi time by then).

[X] **Zero interaction, for two independent reasons:**

1. The reparameterisation touches only unmatched rows (`u_tg`, `src/prc/train.py:398`);
   rule B touches only rows with LOBT, which are exactly the matched rows. The two row sets
   are disjoint and together exhaust the file.
2. Even if they overlapped, the reparameterisation is undone before `_assemble`
   (`src/prc/train.py:413`, `g = g + shift_tg`), so the clip would operate on taxi-seconds
   either way — which is the property `src/prc/train.py:410-411` was written to guarantee.

[S] Measured confirmation: rule B's effect on `oof_reparam_lirf-41526561b0` (reparam on) is
−0.1617 s and on `oof_submit_v10_reparam-77f92c802d` (reparam on, full config) is −0.1637 s;
on `oof_main-48c19bc830` (reparam off) it is −0.0993 s. [X] The differences are the model's,
not the rule's: the bind sets differ by two rows.

### 5.4 How to ship it as one isolated, pre-priced submission

[X] The design follows the v9/v10 probe method, which is the one method in this repository's
history that has produced an attributable leaderboard number [S]
(`configs/submit_v10_reparam.yaml` header: "the submission built from this run keeps v4's
339,551 matched predictions untouched and replaces only the 5,290 unmatched ones, so the
difference from v4's 305.027 is the model-B change and nothing else"). Here the isolation is
even cleaner, because no retraining is involved at all.

**Build the candidate by transforming `outputs/submit_v10_reparam.parquet` directly, with no
model run.** The candidate differs from our 284.3584 file in **47 rows and nothing else**; the
other 344,794 rows are bit-identical. Any board delta is therefore those 47 rows, exactly, and
the metric is deterministic — there is no seed, no fold, no retrain variance to confound it.
[S] This also sidesteps the known refit hazard that produced the v2→v8 accident
(`docs/handoff.md` §8.1 訂正: two seed keys in the refit path reproduced v8 on all 5,290 rows).

Order of operations:

1. **Pre-register the price, before uploading** (the `kind-mango` discipline,
   `.../15-postprocess-overlays.md:238-240`, "One pre-registered price per upload"). Written
   here, now: **guaranteed ≥ +0.0896 s; expected +0.3 to +0.6 s; absolute worst case
   −0.3683 s; result outside [−0.37, +0.89] falsifies the construction, not the rule.**
2. **Build the file** with a throwaway script that reads `outputs/submit_v10_reparam.parquet`
   and `data/raw/ranking.parquet`, computes the window, clips, and writes
   `outputs/v12_lobt_bound.parquet`. Route it through `prc.predict.write_submission` so the
   existing gate runs (`src/prc/predict.py:56-114`: template row order, id dtype, id-set
   equality, reference-median drift).
3. **Label-free gate before upload** (`kind-mango` MP7,
   `.../15-postprocess-overlays.md:242-247`), asserted in the script and printed:
   - exactly **47** rows differ from v10, of which **43 up / 4 down**;
   - **max |move| = 2,881 s**, no move exceeds `|p − boundary|` for its own row;
   - **0** rows where `AOBT_3_flt` is null are touched;
   - every moved row lands **inside** `[max(m_lobt − 3606, 0), m_lobt + 3606]`;
   - no prediction becomes negative; no NaN; row count 344,841.
   Block the upload on any mismatch.
4. **Upload as v12.** [S] Five per day, and the result lands as `<name>_result.json` in the
   team bucket within ~15 s (`docs/README.md` §1; `.../09-repo-sergiuv11.md:780-783`).
5. **Only after the board answers**, decide whether to move the rule into `_assemble` behind
   the config flag of §5.1 and make it part of the standing pipeline. [S] "Never build on an
   unscored upload" (`.../15-postprocess-overlays.md:238-239`).
6. **Log it** in `experiments/experiments.jsonl` with the pre-registered price alongside the
   realised one, so the offline→live transfer ratio for a *bound* (as opposed to a *fit*)
   enters our record. [X] We currently have one such ratio in this competition — sqcm9896's
   −23.9 → −20.3 (85 %) — and it is the whole basis of theme P5. A second data point on our
   own data is worth more than the 0.1 s.

**Do not bundle anything else into v12.** [S] `kind-mango`'s own post-mortem on bundling:
"the paired hold-out is poorly calibrated once several base changes stack on each other"
(14 % transfer, `.../15-postprocess-overlays.md:270`).

### 5.5 What NOT to do, recorded so it is not rediscovered

- [S/§2] **Do not ship rule A**, in any form, at any gate value. Measured +0.0985 to
  +0.2219 s on three OOF files, rejected at all four drop levels with P = 0.000.
- [X] **Do not widen or narrow 3606.** Widening weakens a bound that is already exactly tight;
  narrowing makes it false and converts a projection into a fit — i.e. turns rule B into
  rule A.
- [X] **Do not apply the window as a training-target clip.** The label already satisfies it on
  100 % of rows, so it would be vacuous — and [S] sqcm9896 measured train-target clipping as
  exactly that, "vacuous — moves neither median nor mean RMSE"
  (`.../24-repo-tail-sweep.md:799`).
- [X] **Do not drop the upper half.** It binds 4 rows and Σ(move)² = 4.26e5 → at most
  ±0.0022 s either way. It is free, it is derived from the same inequality, and dropping it
  for "never clip from above" reasons would be applying a global-constant heuristic to a
  per-row bound (§1.5). Keep it; note that it is worth essentially nothing.
- [X] **Do not conclude from a small board delta that the bound is wrong.** With 93.9 % of the
  guaranteed floor in three rows, a realised +0.05 s is fully consistent with a correct rule
  whose truths happened to sit near the binding boundary.

### 5.6 The follow-on this measurement opens (out of scope for v12)

[X] §3.2 found something larger than the overlay: on the 4,011 / 2,062,577 rows where
`|AOBT_3 − LOBT| > 3606`, **the model's anchor is provably outside the admissible window**, and
`clip(m_aobt, m_lobt ± 3606)` cuts the anchor's RMSE from 3,166 to 2,269 on that population.
Rule B only repairs this where the final prediction leaves the window (47 ranking rows); the
proxy is wrong on **956** ranking rows. Clipping `PROXY` itself before it enters model A —
or adding `|AOBT_3 − LOBT|` and the signed distance-outside-window as features — is a model
change, needs a full CV run, and belongs in a separate plan. It is the larger prize. Do not
mix it into v12.

---

## 6. The failure mode, stated honestly

### 6.1 What would have to be true for this to cost us

Exactly one thing, and it is not a matter of degree:

**`|BLOCK_TIME_UTC_mvt − LOBT_flt| ≤ 3606 s` would have to fail on at least one of the 47
rows rule B moves.**

Nothing else can hurt us. Not seasonality — the bound holds in all 12 months of 2025
including both scored months. Not an airport — it holds at all 10. Not the tail — the bound
is per row, so a 131,167-second label is bounded exactly as tightly as a 900-second one. Not
model error — the projection argument (§1.5, §4.3) is independent of how good the prediction
was. Not the fold structure our instrument is known to distort (`docs/CODEX_BRIEF.md:44-48`)
— no fold enters the rule.

Three concrete ways the bound could fail in 2026 [X]:

1. **The clamp is a property of the 2025 extract, not of the source system.** If the organisers
   rebuilt `ranking.parquet` under a different pipeline, the constant could move or vanish.
   [S] Against this: the 2026 ranking set *was* replaced on 2026-09-04
   (`.../09-repo-sergiuv11.md`, slot 09 title), and the audit of the replacement found the
   instrument stable — "schema, null rates, ID encodings, closed vocabularies, and the NM
   join rate are all inside their 2025 bands" (`docs/CODEX_BRIEF.md`, §8).
2. **`LOBT_flt` drifts.** [S] Partially checkable: `LOBT == EOBT_1` on 70.03 % of 2025 and
   69.08 % of 2026 rows; `m_lobt` median 1,237 → 1,263, q.99 3,840 → 4,138, min −3,382 →
   −3,173 (`p5_drift.py`). Same distribution, mildly slower. [X] The one real drift signal:
   `|AOBT_3 − LOBT| > 3606` rises from 0.194 % to 0.282 % — the two clocks disagree ~45 %
   more often in 2026. That makes the rule bind more (47 rows vs the OOF's 28 at lower density)
   but says nothing about whether the bound itself still holds.
3. **A small number of genuinely anomalous rows.** [S] Rule-of-three: 0 violations in
   2,062,577 observations gives a 95 % upper bound on the violation rate of **1.454e-6**,
   i.e. **≤ 0.494 expected violations among all 339,551 matched ranking rows**, and
   ≤ 7e-5 among the 47 rows rule B actually moves.

### 6.2 How many rows of wrong-sign movement it would take

[S] `p5_price.py`. To cancel the guaranteed gain of 1.7567e7 SSE:

- at the **median** move (94 s), with the model previously exact: **1,988 rows** of wrong-sign
  movement;
- at the **mean** move (248 s): **286 rows**;
- **but only 47 rows move at all.** A complete reversal from the median-sized movers is
  arithmetically impossible.

[X] So the question collapses to the top three rows, which hold 93.9 % of both the gain and
the exposure. Per-row exposure if the bound is violated on that one row (`p5_worst.py`):

| row | move | if `y = p` (model was exact) | if `y = 900` | if `y = 0` |
|---|---|---|---|---|
| LIRF/Jan 4,552 → 7,433 | 2,881 | +0.0423 s | +0.1496 s | +0.1760 s |
| LFPG/Jul 3,960 → 6,591 | 2,631 | +0.0353 s | +0.1174 s | +0.1415 s |
| LFPG/Jul 0 → 1,132 | 1,132 | +0.0065 s | **−0.0039 s** | +0.0065 s |
| all 47 together | — | **+0.0896 s** | **+0.2819 s** | **+0.3683 s** |

**[X] The honest summary of the bet: we are wagering at most 0.37 s, and realistically at most
0.18 s (the single largest row), against a guaranteed 0.09 s and an expected 0.3–0.6 s, on
the proposition that a constant which holds exactly on 2,062,577 consecutive rows across
twelve months and ten airports also holds on 47 rows in 2026.**

### 6.3 What would make me withdraw this before uploading

[X] Any one of these, checkable without the board:

- the pre-upload gate of §5.4 step 3 reports a row count other than 47, or a move larger than
  `|p − boundary|`, or any unmatched row touched — that is a construction bug, not a rule
  failure;
- a re-run of `p5_measure.py` finds any 2025 row with `|BLOCK − LOBT| > 3606`;
- `data/raw/ranking.parquet` is replaced again before we upload, and `m_lobt`'s distribution
  moves outside the 2025 band, or the `AOBT_3 − LOBT` disagreement rate moves materially past
  0.28 %.

### 6.4 What the board result will and will not settle

[X] It will settle whether the bound holds well enough to be worth its 47 rows — deterministically,
because the file differs in 47 rows and nothing else.

It will **not** settle theme P5's larger claim. sqcm9896 got −20.3 s from a rule in this family
because their model had no identity reparameterisation to begin with; we got ours in v2's
two-clock change (`docs/handoff.md` §4: v1 345.35 → v2 312.02) and again in v10's model-B
reparameterisation (−20.67 s, `docs/CODEX_BRIEF.md:47-51`). **We have already collected the
20 seconds this theme is named after. What is left is the 0.09-to-0.6 s remainder on the rows
where the two clocks contradict each other — plus the §5.6 follow-on, which is where any
remaining size in this theme actually lives.**

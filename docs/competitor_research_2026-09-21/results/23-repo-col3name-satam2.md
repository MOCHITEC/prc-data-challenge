# SLOT 23 — `col3name/air-data` (kind-earthquake) and `satam2/knowledgeable-helicopter`

Investigator note, 2026-09-21. Evidence marking: **[S]** = directly read from source I
cloned / from the official leaderboard API. **[X]** = my inference, not established by
the code. Line numbers refer to the cloned trees at
`/tmp/slot23/repoA` (col3name/air-data, HEAD `c14bea4`) and `/tmp/slot23/repoB`
(satam2/knowledgeable-helicopter, `main` HEAD `9e5b9e6` and branch
`experiments/280-screening` HEAD `88b7de4`).

---

## 0. Headline corrections to the slot brief — read this first

Two of the three premises in the brief are wrong, and one is right in a way that
matters more than stated.

### 0.1 Repo A does NOT contain the rank-21 / 268.52 code. [S]

I pulled the full leaderboard from the official API
(`https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard`,
cursor-paginated, 2,144 submission rows, 168 teams). kind-earthquake's submission
history: [S]

| file | score | processed |
|---|---:|---|
| kind-earthquake_v42.parquet | **268.5216** | 2026-09-16T17:40Z |
| kind-earthquake_v41.parquet | 269.0365 | 2026-09-16T10:19Z |
| kind-earthquake_v45.parquet | 270.2344 | 2026-09-17T00:05Z |
| kind-earthquake_v34.parquet | 270.3062 | 2026-09-14T08:54Z |
| kind-earthquake_v30.parquet | 285.2314 | 2026-09-14T01:14Z |
| kind-earthquake_v10.parquet | 537.9422 | 2026-09-07T23:02Z |
| kind-earthquake_v6.parquet  | 542.1366 | 2026-09-05T02:02Z |
| kind-earthquake_v1.parquet  | 561.5520 | 2026-09-04T20:41Z |
| **kind-earthquake_v2.parquet** | **599.6651** | 2026-09-04T20:55Z |
| **kind-earthquake_v3.parquet** | **731.1525** | 2026-09-04T23:20Z |

The repository's last commit is `c14bea4`, **2026-09-05 02:37 +0300** [S]
(`git log`, `/tmp/slot23/repoA`). Remote has exactly one branch, `main`
(`git ls-remote --heads origin` → only `refs/heads/main`) [S]. The only two
submission artefacts committed are `kind-earthquake_v2.parquet` and
`kind-earthquake_v3.parquet` — which scored **599.67** and **731.15**, i.e.
315 s and 447 s *worse* than our 284.36 [S]. The 268.52 run (`v42`) was submitted
eleven days after the repo stopped being updated, and `col3name` has no other
PRC-related repository (`mcp__github__search_repositories user:col3name`,
70 repos, only `air-data` is aviation) [S].

**Conclusion: the claim "plain CatBoost + compact feature recipe reaches 268.52" is
not supported by any code I can read.** The published code corresponds to the
540–730 s era of this team. Everything Repo A tells us below is about a pipeline
that was *much worse than ours*; it is useful only as a negative control and for
one specific diagnostic (§1.3). Nothing in Repo A should be copied on the strength
of the 268.52 number.

### 0.2 Repo B's published `main` branch is not the 278.01 model either — but its
### `experiments/280-screening` branch *is*, and it is a goldmine. [S]

knowledgeable-helicopter's submissions: [S]

| file | score | processed |
|---|---:|---|
| knowledgeable-helicopter_v3.parquet | **278.0146** | 2026-09-16T23:36Z |
| knowledgeable-helicopter_v2.parquet | 294.6260 | 2026-09-16T08:09Z |
| knowledgeable-helicopter_v1.parquet | 348.0775 | 2026-09-14T22:28Z |

`main` HEAD is 2026-09-12 and describes the "30 features / no external data /
3-way routing" release; that release corresponds to the **348.08** submission,
not 278.01 (`docs/handoff/RESEARCH_HISTORY.md` states this explicitly: "Original
release | 331.175989 s local | A historical earlier submission scored 348.0775") [S].

The remote has a second branch, `experiments/280-screening` (`88b7de4`,
2026-09-16 19:05 −0700, "Preserve V3 research handoff and screening
implementation"), 865 files / 89,686 insertions over `main`, containing
`TEAM_HANDOFF.md`, `docs/handoff/V3_RECIPE.json`,
`docs/handoff/RESEARCH_HISTORY.md` and a 836-file `research_snapshot/review_work/`
source dump [S]. That branch documents the **278.0146** model in full.

### 0.3 The AOBT-proxy residual hypothesis is CONFIRMED for Repo B, REFUTED for Repo A. [S]

- Repo B (both branches) fits **`Y − P` where `P = MVT_TIME_UTC_mvt − AOBT_3_flt`** [S].
- Repo A fits the **raw `TAXITIME_SEC_mvt`** and never forms `MVT − AOBT` at all [S].

Repo A is the counter-example that proves the point: it has a CatBoost with ~80
engineered features, per-airport/runway/stand interactions, seven traffic windows
and leave-one-out target statistics — and it scored 599–731 in the era we can
read, because it never parameterised the target against the clock. Repo B, with a
crude 30-feature CatBoost, got 348 on its *first* try and 278 once it built experts
on top of the same residual.

---

## 1. REPO A — `col3name/air-data` (kind-earthquake)

### 1.1 Required answer: what is the fitting target?

**Raw taxi-out duration in seconds. No residual, no clock proxy.** [S]

```
train3.py:21   TARGET = "TAXITIME_SEC_mvt"
train3.py:456      return X_train, train_df[TARGET], features, cat_features
train3.py:464      train_df[TARGET],
train3.py:613      model.fit(X, y, cat_features=cat_features)
```
`prepare_xy` hands `train_df[TARGET]` — the raw label — straight to
`CatBoostRegressor.fit`. Identical in `train1.py:441,447`. There is no offset,
no `residual`, no `proxy` identifier anywhere in the repo (`grep -n
"AOBT\|LOBT\|proxy\|residual" *.py`) [S].

Training rows are filtered to `30 ≤ TARGET ≤ 7200` (`train3.py:376`,
`train1.py:367`), so extreme labels are *dropped* rather than modelled [S].
Predictions are floored at zero only (`makeSubmit.py:378,531`, `np.maximum`) [S].

### 1.2 The near-miss: they have both halves of the proxy and never subtract them

This is the single most striking thing in Repo A. [S]

```
train3.py:218-220   if "EOBT_1_flt" in df.columns:
                        d = minutes_diff(df[TIME_COL], df["EOBT_1_flt"])
                        df["MVT_MINUS_EOBT_MIN"] = d
train3.py:224-227   if "AOBT_3_flt" in df.columns and "EOBT_1_flt" in df.columns:
                        df["AOBT_MINUS_EOBT_MIN"] = minutes_diff(
                            df["AOBT_3_flt"], df["EOBT_1_flt"])
```

`MVT_MINUS_EOBT_MIN − AOBT_MINUS_EOBT_MIN = (MVT − EOBT) − (AOBT − EOBT) = MVT − AOBT`
— exactly merry-quicksand's and javidmardanov's proxy. Both operands are features
of the model. But an axis-aligned tree cannot form a difference of two columns; it
would need a staircase of splits to approximate `P`, and it can never *offset* the
target by it. **The proxy is present in Repo A's feature matrix as two separate
columns and is worth nothing there.** `AOBT_3_flt` appears nowhere else in the
repo [S].

`LOBT_flt` is parsed into `TIMESTAMP_COLUMNS` (`train3.py:34`) and then never
used for any feature [S]. No `MVT − LOBT` feature, no clipping window.

### 1.3 Complete feature list (train3.py — the newest submission-producing recipe)

`FEATURES` list, `train3.py:64-92`, 79 names + 6 historical = **85 columns**
(`get_features` filters to those present) [S]:

**Categorical base (11)** — `AIRPORT` (=`ADEP_mvt`, upper/strip/`__MISSING__`,
`train3.py:139`), `AIRLINE` (=`AIRCRAFT_OPERATOR_flt`, :150), `AIRCRAFT`
(=`AIRCRAFT_TYPE_mvt` fallback `AIRCRAFT_TYPE_flt`, :142-148), `DESTINATION`
(=`ADES_mvt`), `RUNWAY_mvt`, `STAND_mvt`, `FLIGHT_RULE_mvt`,
`MARKET_SEGMENT_flt`, `FLIGHT_RULE_flt`, `FLIGHT_TYPE_flt`, `WK_TBL_CAT_flt`.

**Calendar (11)** `train3.py:168-203` — `MONTH`, `DAY_OF_YEAR`, `WEEKDAY`, `HOUR`,
`MINUTE`, `IS_WEEKEND`, `MINUTE_OF_DAY`, `HOUR_SIN`, `HOUR_COS`, `WEEKDAY_SIN`,
`WEEKDAY_COS` (sin/cos on minute-of-day/1440 and weekday/7). All UTC; **no local
time, no DST** [S].

**Time buckets (4)** — `TIME_15M` = `str(HOUR*4 + MINUTE//15)`, `TIME_30M` =
`str(HOUR*2 + MINUTE//30)`, `TIME_60M` = `str(HOUR)`, `TRAFFIC_BANK` = hour→
{EARLY_MORNING 5-8, MORNING_BANK 8-11, MIDDAY 11-14, AFTERNOON 14-18,
EVENING_BANK 18-22, NIGHT else} (`train3.py:185-202`).

**Schedule/clock deltas (9)** `train3.py:210-240`, all in **minutes** —
`MVT_MINUS_SCHED_MIN` = MVT−`SCHED_TIME_UTC_mvt`; `SCHED_DELAY_ABS` = |that|;
`SCHED_DELAY_POS` = clip(that, ≥0); `MVT_MINUS_EOBT_MIN` = MVT−`EOBT_1_flt`;
`EOBT_DELAY_ABS`; `EOBT_DELAY_POS`; `AOBT_MINUS_EOBT_MIN` = `AOBT_3_flt`−`EOBT_1_flt`;
`EOBT_MINUS_IOBT_MIN` = `EOBT_1_flt`−`IOBT_flt`; `DELAY_REGIME` =
`pd.cut(MVT_MINUS_EOBT_MIN, [-inf,-30,-5,5,15,30,60,inf])` →
VERY_EARLY/EARLY/ON_TIME/SMALL_DELAY/DELAY/BIG_DELAY/SEVERE_DELAY.

**Categorical interactions (9)** `train3.py:251-267`, string-concatenated with
`"__"` — `AIRPORT_AIRLINE`, `AIRPORT_AIRCRAFT`, `AIRPORT_DESTINATION`,
`AIRPORT_RUNWAY`, `AIRPORT_STAND`, `AIRPORT_STAND_RUNWAY`, `AIRPORT_TIME15`,
`AIRPORT_TIME30`, `AIRPORT_WEEKDAY_TIME` (=AIRPORT×WEEKDAY×TIME_30M).

**Traffic counts (21)** `train3.py:275-306` — for m ∈ {5,10,15,30,60,90,120}:
`DEP_COUNT_{m}M`, `ARR_COUNT_{m}M` = time-based rolling sum of `IS_DEPARTURE` /
`IS_ARRIVAL` per `AIRPORT`, `closed="left"` (strictly prior events);
`TOTAL_TRAFFIC_{m}M` = sum of the two. Only 5/15/30/60/120 totals are kept in
`FEATURES`.

**Traffic derivatives (5)** — `DEP_RATE_15M` = DEP_COUNT_15M/0.25; `DEP_RATE_60M`
= DEP_COUNT_60M; `TRAFFIC_ACCELERATION` = TOTAL_15M/15 − TOTAL_60M/60;
`DEP_ARR_RATIO_15M`, `DEP_ARR_RATIO_60M` = (DEP+1)/(ARR+1) (`train3.py:294-305`).

**Runway load (6)** `train3.py:309-340` — grouped by (`AIRPORT`,`RUNWAY_mvt`),
rows with missing/UNKNOWN runway excluded: `RUNWAY_DEP_{15}M`, `RUNWAY_ARR_{15}M`,
`RUNWAY_LOAD_{5,15,30,60}M` (= DEP+ARR), all `closed="left"`.

**Regimes (2)** `train3.py:343-355` — `TRAFFIC_REGIME` =
cut(TOTAL_TRAFFIC_30M, [-inf,5,15,30,50,inf]) → QUIET/NORMAL/BUSY/VERY_BUSY/
SATURATED; `RUNWAY_LOAD_REGIME` = cut(RUNWAY_LOAD_15M, [-inf,2,5,8,inf]) →
LOW/NORMAL/HIGH/SATURATED.

**Historical target statistics (6)** `train3.py:385-432`, `HISTORICAL_GROUPS`
:55-62 — smoothed leave-one-out means of `TAXITIME_SEC_mvt`:
`HIST_AIRPORT_MEAN` (key AIRPORT, α=20), `HIST_AIRPORT_MEDIAN_PROXY`
(same key/α — **a duplicate of the previous column; the "MEDIAN" name is a bug,
it computes the same LOO mean**) [S], `HIST_AIRPORT_AIRLINE_MEAN` (α=10),
`HIST_AIRPORT_RUNWAY_MEAN` (α=10), `HIST_AIRPORT_TIME30_MEAN` (α=10),
`HIST_AIRPORT_WEEKDAY_TIME_MEAN` (α=10).
Train-mode formula (`:419-425`): `(sum − y + α·global_mean) / (count − 1 + α)`;
apply-mode (`:428-430`): plain group mean, `fillna(global_mean)`.

### 1.4 `train1.py` extras that `train3.py` dropped (and why it matters)

`train1.py:270-319` `add_queue_features` builds `PREV_TAXI_MEAN_{15,30,60}M`,
`PREV_TAXI_MEDIAN_30M`, `PREV_TAXI_P90_60M`, `PREV_TAXI_COUNT_30M` — time-rolling
statistics of the *label itself* over prior departures at the airport, plus
`QUEUE_REGIME` = cut(PREV_TAXI_MEDIAN_30M, [-inf,300,600,900,1200,inf]) and
`QUEUE_TREND` = cut(PREV_TAXI_MEAN_15M − PREV_TAXI_MEAN_60M,
[-inf,-120,-30,30,120,inf]) (`train1.py:335-345`) [S].

**These are broken at inference.** `makeSubmit.py:201-202` sets
`df[TARGET] = np.nan` for the ranking file, so every `PREV_TAXI_*` column is NaN
and `QUEUE_REGIME`/`QUEUE_TREND` are the string `"nan"` for all 344,841 ranking
rows, while in training they carried real signal [S]. `train2.2.py` imports
exactly this `build_features`/`create_model` from `train1` (`train2.2.py:10-18`)
and produced `kind-earthquake_v2.parquet` — which scored **599.67** [S]. This is a
textbook train/serve skew and is the most likely single cause of that score. [X:
causal attribution is mine; the repo contains no diagnostic.]

`train3.py` replaced `PREV_TAXI_*` with the `HIST_*` maps (§1.3) precisely because
those are computable at inference from saved dictionaries
(`train3.py:629-642` pickles `train3_historical_stats.pkl`) [S]. Yet
`kind-earthquake_v3.parquet` still scored **731.15** [S] — worse. I cannot
determine from the repo why; `makeSubmit.py` imports `build_features` from
**`train1`**, not `train3` (`makeSubmit.py:10-13`), so the inference script in the
repo is not matched to `train3`'s feature list at all. [S for the import; [X] for
this being the cause of 731.]

### 1.5 Model family, full hyperparameters, CV design

**`train3.py:487-505`** (the deeper recipe): [S]
```
CatBoostRegressor(
  loss_function="RMSE", eval_metric="RMSE",
  iterations=1400, learning_rate=0.035, depth=8, l2_leaf_reg=10,
  random_seed=42, random_strength=0.35,
  bootstrap_type="Bernoulli", subsample=0.85,
  thread_count=-1, allow_writing_files=False, verbose=100,
  od_type="Iter", od_wait=150)          # early stopping only in CV
```
**`train1.py:411-429`** (the recipe actually wired to `makeSubmit.py`): identical
shape but `iterations=800, learning_rate=0.05, depth=7, l2_leaf_reg=8,
random_strength=0.5, subsample=0.8, od_wait=100` [S].

**CV design** — expanding-window month-forward, `CV_MONTHS = [9,10,11,12]`
(`train3.py:26`): `temporal_split(df, m)` = train on `MVT month < m`, validate on
`month == m` (`train3.py:472-474`) [S]. Four folds, each with
`use_best_model=True` and `eval_set` = the validation month (`:536-542`).
Predictions floored at 0 before scoring (`:544`). Per-airport RMSE printed
(`:508-517`). Final model: refit on **all** 2025 departures with
`iterations = max(300, median(best_iterations))` and **no early stopping**
(`:592-613`) [S]. Feature importance and a features.json are dumped
(`:618-651`).

Note the fold leakage in the `HIST_*` features: `enrich_fold` (`:477-484`) builds
LOO stats on `train_df` only and maps them onto `valid_df` — correct. But the
final model calls `add_historical_target_features(df, train_mode=True)` on the
whole year (`:599`), i.e. LOO over all 12 months, and then `makeSubmit`-style
inference would use full-year group means. That is legitimate here (no future
leak into 2026) [S].

### 1.6 Required answer: what does the CatBoost setup do with categoricals?

**Nothing beyond defaults.** [S]

`create_model` (`train3.py:487-505`, `train1.py:411-429`) sets **no**
`one_hot_max_size`, **no** `max_ctr_complexity`, **no** `simple_ctr` /
`combinations_ctr`, **no** `has_time`, **no** `boosting_type`, **no**
`ctr_target_border_count`, **no** `cat_features` ordering hints [S]. The only
categorical handling is:

- `get_cat_features` (`train3.py:439-441`) returns the intersection of the model's
  feature list with `CATEGORICAL_COLUMNS ∪ DERIVED_CATEGORICAL_COLUMNS` — so
  `AIRPORT, AIRLINE, AIRCRAFT, DESTINATION, RUNWAY_mvt, STAND_mvt,
  FLIGHT_RULE_mvt, MARKET_SEGMENT_flt, FLIGHT_RULE_flt, FLIGHT_TYPE_flt,
  WK_TBL_CAT_flt, TIME_15M, TIME_30M, TIME_60M, TRAFFIC_BANK, DELAY_REGIME,
  AIRPORT_AIRLINE, AIRPORT_AIRCRAFT, AIRPORT_DESTINATION, AIRPORT_RUNWAY,
  AIRPORT_STAND, AIRPORT_STAND_RUNWAY, AIRPORT_TIME15, AIRPORT_TIME30,
  AIRPORT_WEEKDAY_TIME, TRAFFIC_REGIME, RUNWAY_LOAD_REGIME` (27 of them) [S].
  In `train1.py:397-403` it is instead *dtype-sniffed* (any string/object/category
  column), which is why `train1` also passes the numeric-looking `DELAY_REGIME`
  etc. [S]
- values normalised by `safe_str` = `fillna("__MISSING__").astype(str).str.strip()
  .str.upper()` (`train3.py:95-96`) — so missing is its own level, and case/space
  variants collapse [S].
- Interaction categories are **pre-crossed as strings** (`_interaction`,
  `train3.py:244-248`) rather than left for CatBoost's CTR combinations [S].

So: ordered target statistics are on (CatBoost computes CTRs by default), but with
library defaults `max_ctr_complexity = 4` (CatBoost docs: "In all other cases: 4"
for CPU non-multiclass) [S, catboost.ai/docs/en/references/training-parameters/ctr].
`one_hot_max_size` and `has_time` defaults are not stated in that page; CatBoost's
CPU default for `one_hot_max_size` is small (2) and `has_time` is `False`, meaning
the CTR permutations are **random, not chronological** [X — I did not verify these
two defaults from a primary source].

**The concrete gap:** with `has_time=False`, CatBoost's ordered target statistics
for `AIRPORT_STAND_RUNWAY` etc. are computed over random permutations of a
year-long temporally ordered dataset. Repo A pre-crossed 9 interactions by hand
*and* let CTR build up to complexity-4 combinations on top of 27 categoricals —
a large, unregularised CTR surface. Nothing in the repo evaluates that choice.

### 1.7 Experiment log reconstructed from Repo A

There is **no** experiment log, README results table, notes file or CV output
committed [S]. `models/cv_scores_train3.csv` is written at runtime
(`train3.py:581`) but is `.gitignore`d. The only reconstructable history is:

| artefact | commit | date | leaderboard score |
|---|---|---|---:|
| `train.py`/`train0.py` (initial) | `0ab941c`/`c8f4107` | 09-04 18:38 | — |
| `train1.py`, `train2.py` | `ef21109` | 09-04 22:12 | v1 561.55 (20:41) |
| `train2.1.py`, `submit.py` | `6d18ecc` | 09-04 23:10 | v2 **599.67** (20:55) |
| `train2.2.py` (+ fix) | `1b214b6`,`0e99dc3` | 09-04 23:18/23:24 | — |
| **`kind-earthquake_v2.parquet`** | `7fec16b` | 09-04 23:54 | **599.67** |
| `train3.py`, `makeSubmit.py` | `6e47b4a` | 09-05 00:38 | v3 **731.15** (23:20 prev day) |
| **`kind-earthquake_v3.parquet`** | `c5f30de` | 09-05 02:18 | **731.15** |
| `train4.py` | `c14bea4` | 09-05 02:37 | v4 609.94 / v5 586.15 / v6 542.14 |

[S: commit dates from `git log --name-status`; scores from the leaderboard API.]

`train2.2.py` is a September-only experiment: `TRAIN_FILE =
data/training_2025-09-01_2025-10-01.parquet`, `TRAIN_MONTH = 9`,
`VALIDATION_FRACTION = 0.15` (last 15 % of September) (`train2.2.py:33,57,60`) [S].
`train4.py` is a schema-adaptive rewrite ("Resourceful-Quiver-inspired", :68-73)
with column-name guessing (`TIME_CANDIDATES` etc., :78-92), exponentially-decayed
historical target statistics (`add_leakage_safe_decay_features`, :319-…) and a
self-test asserting the current target is excluded (:18-30). It never produced a
committed submission [S].

The submission parquets themselves: both 344,841 × 2, `TAXITIME_SEC_mvt` as
**float64** (not int32); v2 mean 914.3 s / min 100.2 / max 4412.8; v3 mean 1103.9
/ min 398.4 / max 3638.0; RMSE between the two files 459.5 s [S, computed from the
committed parquets].

### 1.8 What Repo A is actually good for

One transferable idea, one warning:

- **Warning (high value):** any feature derived from the *label* of neighbouring
  rows (rolling mean taxi-time, queue regime) silently becomes NaN on the ranking
  file. If our pipeline has anything of that shape, it is a 300-second-class bug.
- **Idea:** `HIST_*` smoothed leave-one-out target means keyed on
  AIRPORT / AIRPORT×AIRLINE / AIRPORT×RUNWAY / AIRPORT×TIME30 /
  AIRPORT×WEEKDAY×TIME30 with α = 10–20 are a cheap, inference-safe substitute
  (`train3.py:385-432`). Repo B explicitly *rejected* passing such means as model
  inputs (`reports/implementation_decisions.md`: "Custom hierarchical means are
  standalone baselines. They are not passed as in-sample target features to
  CatBoost") [S] — so this is contested, not a proven win.

---

## 2. REPO B — `satam2/knowledgeable-helicopter`, `main` (the 348.08 / 30-feature release)

### 2.1 Required answer: what is the fitting target?

**A residual against the AOBT clock proxy, for the majority route.** [S]

Proxy definition — two places, identical:
```
src/taxiout/cache.py:75    meta["proxy_sec"] = duration(dep[MOVEMENT], dep["AOBT_3_flt"]).to_numpy()
src/taxiout/audit.py:45    dep["proxy_sec"] = duration(dep[MOVEMENT], dep["AOBT_3_flt"])
src/taxiout/schema.py:43-44  def duration(end, start): return (utc(end) - utc(start)).dt.total_seconds()
src/taxiout/schema.py:6      MOVEMENT = "MVT_TIME_UTC_mvt"
```
→ `proxy_sec = MVT_TIME_UTC_mvt − AOBT_3_flt`. Identical to merry-quicksand's
`proxy` and javidmardanov's base.

Target:
```
src/taxiout/models/residual.py:9-10
    def residual_target(y, proxy):
        return np.asarray(y, dtype=float) - np.asarray(proxy, dtype=float)

src/taxiout/train.py:148
    target = residual_target(y, meta["proxy_sec"]) if is_residual else y
src/taxiout/train.py:214   (final fit)
    target = residual_target(y, meta["proxy_sec"]) if name == "residual" else y
```

Reconstruction:
```
src/taxiout/models/residual.py:13-17
    def combine(proxy, correction, fallback, status):
        usable = np.asarray(status) == "present"
        prediction = np.asarray(fallback, dtype=float).copy()
        prediction[usable] = np.asarray(proxy)[usable] + np.asarray(correction)[usable]
        return prediction, np.where(usable, "residual", np.char.add("direct_", ...))
```

The `model_card.md` states the semantics outright: "Define proxy as actual takeoff
minus the observed NM off-block timestamp. The supervised residual is disagreement
between NM and airport off-block timestamps; this is partly a
recording/source-reliability problem, not simply a physical taxi-congestion
problem" (`TEAM_HANDOFF.md`) [S].

**No LOBT-based window.** `LOBT_flt` is used only as
`takeoff_minus_LOBT_flt` and `last_minus_initial = LOBT − IOBT`
(`features/clocks.py:9-11`). There is **no** `|(MVT−LOBT) − TAXITIME| ≤ 3606`
clip anywhere — in fact `README.md` and `model_card.md` state "Predictions are not
clipped" / "There is no output cap or nonnegative clamp", and the reproduced
submission's range is `minimum_sec: -268, maximum_sec: 87035`
(`reports/reproduction.json`) [S]. So javidmardanov's clipping trick is *not*
shared by Repo B.

### 2.2 Required answer: reconstruct the 3-way proxy regime split exactly

**It is a 4-way split at inference, defined purely on `proxy_sec`, with 3 trained
models.** [S]

Router:
```
src/taxiout/models/residual.py:4-6
def proxy_status(proxy, config):
    proxy = np.asarray(proxy, dtype=float)
    return np.where(~np.isfinite(proxy), "missing",
           np.where((proxy < config["proxy_min"]) | (proxy > config["proxy_max"]),
                    "invalid", "present"))
```
with `proxy_min: 0`, `proxy_max: 7200` (`configs/release.yaml:19-20`) [S].

Applied in `src/taxiout/predict.py:55-84`:

| # | regime | routing rule (code) | prediction |
|---|---|---|---|
| 1 | **ordinary / "present"** | `0 ≤ proxy ≤ 7200`, finite | `proxy + residual_model.predict(x)` (`predict.py:64-65` → `combine`) |
| 2 | **long positive** | finite, `proxy > 7200` | `proxy + residual_model.predict(x)` — the *same* correction, deliberately extrapolated (`predict.py:67-70`, gated on `long_proxy_correction: true`, `release.yaml:26`); route label `residual_long_proxy` |
| 3 | **negative** | finite, `proxy < 0` | `direct_model.predict(x)` — a raw-target CatBoost (`predict.py:61`, status `"invalid"`) |
| 4 | **missing** | `AOBT_3_flt` absent → `proxy` NaN | `missing_model.predict(x)` — a **dedicated raw-target CatBoost trained only on missing-proxy rows** (`predict.py:71-74`, `train.py:115-156`, gated on `missing_specialist: true`) |

Three models are trained (`train.py:109, 149-153`), named `direct`, `residual`,
`missing`, all CatBoost, all on the same 30-column feature matrix:

- **`direct`**: target = raw `y`; fit on the seeded 250,000-row sample
  (`train.py:109, 208`). Serves regime 3 and is the mandatory fallback
  (`predict.py:50-51`).
- **`residual`**: target = `y − proxy`; fit **only on `status == "present"` rows**
  (`train.py:118-121`: `subset = status == "present"`), i.e. only on
  `0 ≤ proxy ≤ 7200`, within the 250k sample → 247,127 rows
  (`model_card.md`) [S]. Serves regimes 1 and 2.
- **`missing`**: target = raw `y`; fit on **every** missing-proxy row, not the
  sample (`train.py:121`: `component_periods = [idx["fit"], tune, idx["refit"]]`
  for the non-residual route; `train.py:213`: `component_positions =
  np.arange(len(x))` when `name == "missing"`) → **22,470 rows**
  (`model_card.md`, `README.md`) [S].

**Against our two-way split:** our 2,062,577 matched / 22,470 unmatched is the
*same* partition as Repo B's finite / missing. `22,470` matches exactly
(`README.md`: "all 22,470 eligible missing-proxy rows"; `V3_RECIPE.json`:
`"new_missing_fit_rows": 22470`) and `2,062,577` matches exactly
(`TEAM_HANDOFF.md`: "All new finite-clock fits used 2,062,577 eligible 2025 rows")
[S]. So the third regime is **carved out of our 2,062,577 matched rows**: it is the
subset whose `MVT − AOBT` is negative or above 7,200 s. On the 2026 ranking file
the counts are `ordinary 339,377`, `finite_nonordinary 174`, `missing 5,290`
(`V3_RECIPE.json`) [S] — so the "invalid finite" regime is **174 rows out of
344,841 (0.05 %)**. It is a robustness guard, not a source of score.

The score-relevant split is therefore **two-way: finite-proxy residual vs.
missing-proxy dedicated model** — with the missing 1.5 % of rows carrying
**31.42 % of the remaining MSE** (`RESEARCH_HISTORY.md`) [S].

### 2.3 Required answer: complete 30-feature list, transcribed

Built in `src/taxiout/features/pipeline.py:29-57` with `release.yaml` settings
(`clocks: true, traffic: true, quality: false, local_calendar: false,
legacy: false`, `coverage: month-isolated`). In construction order [S]:

**1–9. Raw categoricals** (`schema.py:10-11` `CATEGORIES`, tokenised by
`pipeline.py:11-13` `token()` = `"s" + len(value) + ":" + value`, NA → `"m:"` —
a collision-safe length-prefixed token, then `.astype("category")`):
`ADEP_mvt`, `RUNWAY_mvt`, `STAND_mvt`, `ADES_mvt`, `AIRCRAFT_TYPE_mvt`,
`AIRCRAFT_OPERATOR_flt`, `WK_TBL_CAT_flt`, `MARKET_SEGMENT_flt`,
`FLIGHT_TYPE_flt`.

**10. `airport_stand`** = `token(ADEP_mvt) + token(STAND_mvt)`, categorical
(`pipeline.py:41-43`).
**11. `airport_runway`** = `token(ADEP_mvt) + token(RUNWAY_mvt)`, categorical.

**12–15. Calendar** (`features/calendar.py:10`, from `MVT_TIME_UTC_mvt`, UTC):
`utc_hour`, `weekday` (0=Mon), `month`, `minute`.

**16–20. Take-off minus each clock** (`features/clocks.py:8-9`, seconds, float32):
`takeoff_minus_AOBT_3_flt` ← **this is `proxy_sec` itself, also a feature**;
`takeoff_minus_EOBT_1_flt`; `takeoff_minus_IOBT_flt`; `takeoff_minus_LOBT_flt`;
`takeoff_minus_SCHED_TIME_UTC_mvt`.

**21. `nm_actual_minus_estimated`** = `AOBT_3_flt − EOBT_1_flt` (`clocks.py:10`).
**22. `last_minus_initial`** = `LOBT_flt − IOBT_flt` (`clocks.py:11`).

**23–26. Prior traffic counts** (`features/traffic.py:31-39`), half-open
`[t − w, t)` windows, **within the same actual UTC month**, keyed on airport
(departures by `ADEP_mvt`, arrivals by `ADES_mvt`, `traffic.py:22`), simultaneous
events excluded (`side="left"` on both ends):
`dep_prior_15m`, `dep_prior_60m`, `arr_prior_15m`, `arr_prior_60m`.

**27–28. Window-coverage** (`traffic.py:44-45`): `observed_window_15m_sec`,
`observed_window_60m_sec` = `min(seconds since month start, w·60)` — tells the
model when a count is truncated by the month boundary.
**29. `month_edge`** (`traffic.py:46`) = 1 if within 3,600 s of month start.

**30. `proxy_missing`** (`clocks.py:12`, appended last by `pipeline.py:52`) =
`takeoff_minus_AOBT_3_flt.isna()` as float — an explicit missing-clock flag.

All numeric columns: `±inf → NaN → −999999`, cast float32 (`pipeline.py:53-54`) [S].
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are physically dropped from the frame
before any feature code sees them (`availability.py:13`, asserted by
`assert_observations`, :20-23) [S].

The rejected `quality: true` block (`clocks.py:14-26`) adds ~29 more: per-clock
`_missing`, `_invalid_parse`, `_second`, `_after_takeoff`; `ADEP/ADES/
AIRCRAFT_TYPE` `_source_missing` and `_source_disagrees` between the `_mvt` and
`_flt` copies; `planned_duration_sec` = `ARVT_1_flt − EOBT_1_flt`;
`proxy_negative`; `proxy_over_two_hours`. The `local_calendar: true` block
(`calendar.py:11-23`) adds `local_hour`, `local_weekday`, `utc_offset_hours`,
`dst` via a hard-coded ICAO→tz map (`calendar.py:4-6`). Both were **measured and
rejected** — see §2.5.

### 2.4 Model family, hyperparameters, CV design

**CatBoost only** (`models/direct.py:5,36`). `configs/release.yaml`: [S]
```
iterations: 180      depth: 6       learning_rate: 0.075
l2_leaf_reg: 8       loss: RMSE     eval_metric: RMSE (direct.py:32)
early_stopping_rounds: 40           seed: 20260910   threads: 4
train_sample: 250000                verbose: False   allow_writing_files: False
```
`cat_features` = every `category`-dtype column (`direct.py:38`). **No
`one_hot_max_size`, no `max_ctr_complexity`, no `has_time`** — same defaults as
Repo A [S]. The docstring is explicit about the philosophy
(`pipeline.py:17-21`): "CatBoost receives raw categorical tokens and learns its
own ordered statistics. No upstream target-encoded feature or ranking distribution
fitting is used."

Frozen final tree counts: `direct 178, residual 180, missing 131`
(`reports/freeze.json` `final_iterations`), chosen as the per-component **median of
the F1/F2/F3 early-stopped counts**, fixed before the confirmation run
(`freeze.json` `iteration_rule`) [S].

**CV design** — `configs/folds.yaml`, four-stage per fold
(*fit → tune → refit → score*), all calendar-month boundaries, start-inclusive /
end-exclusive: [S]

| fold | fit | tune | refit | score | role |
|---|---|---|---|---|---|
| F1 | Jan–May | Jun | Jan–Jun | **Jul 2025** | summer dev |
| F2 | Jan–Aug | Sep | Jan–Sep | **Oct 2025** | autumn dev |
| F3 | Jan–Sep | Oct | Jan–Oct | **Nov 2025** | winter dev |
| G1 | Jan–Feb | Mar | Jan–Mar | **Oct 2025** | 6-month-gap stress |
| C1 | Jan–Oct | Nov | Jan–Nov | **Dec 2025** | reused confirmation |

`splits.py:18-38` enforces: stages must not overlap; `refit == fit ∪ tune`; and —
importantly — **flight-ID purging**: any row in an earlier stage sharing a
`FLIGHT_ID_mvt` with a scoring row is removed (`splits.py:26-31`) [S].

**Seasonal selection metric** (`metrics.py:23-28`): not a mean of RMSEs but
`sqrt( w_jul · MSE(F1) + w_jan · MSE(F3) )` with weights = the actual ranking
template proportions `192122/344841` (July) and `152719/344841` (January),
hard-asserted (`metrics.py:24-26`) [S].

Submission: `numpy.rint` nearest-even → checked int32, template order preserved,
**no clipping** (`README.md`, `reports/reproduction.json`) [S].

### 2.5 Required answer: the experiment log (main branch) — scores and failures

`reports/experiments.csv`, 34 completed runs [S]. Seasonal column from
`reports/selection_decision.md`:

| candidate | F1 (Jul) | F2 (Oct) | F3 (Nov) | seasonal | G1 |
|---|---:|---:|---:|---:|---:|
| `direct` (raw target, shared) | 583.742 | 276.681 | 341.204 | **491.329** | 305.358 |
| `residual` (+ direct fallback) | 579.440 | 274.438 | 333.856 | **486.230** | — |
| `direct_specialist` (raw + missing specialist) | 423.666 | 272.518 | 284.459 | **368.560** | — |
| `residual_specialist` | 417.718 | 270.240 | 275.603 | **361.735** | 286.413 |
| `residual_all_finite_specialist` | 371.489 | 268.397 | 277.561 | **333.175** | 286.541 |
| **`residual_long_proxy_specialist` (selected)** | 369.370 | 269.313 | 275.712 | **331.176** | 285.575 |
| C1 (Dec 2025) for the selected recipe | — | — | — | — | **266.406** |

Read the deltas: [S]

- **residual vs direct: only −5.1 s seasonal (491.3 → 486.2).** On its own the
  AOBT residual is worth almost nothing.
- **missing specialist: −122.8 s seasonal (486.2 → 361.7 on top of residual;
  491.3 → 368.6 on top of direct).** This is by far the largest single win in the
  entire log — a dedicated model for the 22,470 rows where `AOBT_3_flt` is absent.
- long-proxy correction (applying the residual model above 7,200 s):
  361.7 → 331.2, i.e. −30.6 s, but `selection_decision.md` warns "The July
  improvement is concentrated in an extreme day-long observation on 15 July …
  the airport-day bootstrap interval spans a small regression" [S].

Per-fold missing-proxy RMSE (`experiments.csv` `missing_rmse_sec`) collapses from
**3186.77 → 1531.60** (F1) and **2606.64 → 1595.10** (F3) when the specialist is
added [S]. `model_card.md` gives the missing cohort's SSE share: F1 35.5 %,
F2 12.5 %, F3 28.0 %, C1 20.9 % [S].

**Documented failures** (`selection_decision.md` "Rejected Additions") [S]:

1. **quality + local-calendar features** (the ~33 extra columns of §2.3):
   F1 585.548, F3 330.534, seasonal 489.292 vs 491.329 matched direct — a 2.04 s
   gain, **below their declared 2.457 s added-complexity threshold. Not promoted.**
2. **Huber loss** `Huber:delta=300` with default Newton leaves: "numerically
   unstable one/three-tree models" — `experiments.csv` shows F1 **19609.445** and
   F3 **4947.192** [S]. With `leaf_estimation_method: Gradient` (`configs/robust.yaml`)
   it was stable but worse: F1 667.908, F3 424.990. **Rejected.**
3. **Traffic ablation** (`configs/no_traffic.yaml`): F1 372.834, F3 278.355 — both
   worse than the retained 369.370 / 275.712. **Traffic counts kept.**
4. **Clock ablation** (legacy diagnostic): 443.973 s without clocks vs 339.050 s
   with clocks on 328,009 Nov/Dec departures — a ~105 s effect [S].
5. Not attempted at all on `main`: "Weather, layout, learned blending, rotations
   and additional tail architectures" [S].

Promotion gates, for reference (`docs/SCREENING.md`): ≥ max(2 s, 0.5 %) seasonal
improvement; per-fold regression ≤ max(5 s, 2 %); missing-clock RMSE regression
≤ 5 %; paired leave-one-day-out checks [S].

---

## 3. REPO B — `experiments/280-screening`: the actual 278.01 recipe

Everything in this section is from the branch `88b7de4`. This is the model that
beat us by 6 s.

### 3.1 Official-vs-local record [S]

`TEAM_HANDOFF.md` + `docs/handoff/RESEARCH_HISTORY.md`:

| recipe | local seasonal dev RMSE | official RMSE |
|---|---:|---:|
| original 30-feature release (`main`) | 331.176 | 348.0775 |
| V2 | 292.813 | 294.626 |
| **V3 (retained)** | **268.663** | **278.0146** |

The local→official penalty for V3 is **+9.35 s** (official MSE +7.08 %), and the
team explicitly declines to call it an overfit estimate [S].

### 3.2 The V3 routing — same 3 regimes, much bigger experts [S]

`docs/handoff/V3_RECIPE.json` and
`research_snapshot/review_work/tail240_20260916/final_submission/compose.py:33-44`:

```python
finite  = np.isfinite(proxy)
regular = finite & (proxy >= 0) & (proxy <= 7200)     # 339,377 ranking rows
```

| regime | model |
|---|---|
| ordinary (`0 ≤ P ≤ 7200`), 339,377 rows | weighted blend of **five** experts: `lgb387` 0.4386, `ple387` 0.2662, `cat225` 0.1536, `tabm225` 0.1062, `lgb449` 0.0355 |
| finite non-ordinary (`P<0` or `P>7200`), **174** rows | single `lgb225` |
| missing proxy, 5,290 rows | fixed nested blend `0.5625·V2 + 0.1875·ExtraTrees + 0.25·normalized` |

All ordinary/non-ordinary experts fit the **same residual `Y − P`**:
`final_ordinary/run.py:104` — `'policy': 'Inherited int(median(F1,F3
tune-selected steps)); all eligible 2025 finite NM rows; raw residual Y-P; no
tuning, sampling, clipping or ranking labels.'` [S]. The formulation is named
`aobt_allfinite` and defined in
`breakthrough_20260916/models/run_full.py:38-40`:
```python
if formulation=="aobt_allfinite":
    offset = meta.proxy_sec.to_numpy(float)
    return x, offset, np.isfinite(offset), {"rule":"observed actual-NM proxy;
        every finite value including negative and >7200; missing rows retain submitted reference"}
```
So **training** uses *every* finite proxy row (2,062,577) and only **routing** at
inference splits on `[0, 7200]` [S].

Full-year training lengths (`V3_RECIPE.json`): `lgb387` 2,250 trees, `lgb449`
2,256, `lgb225` 2,117, `cat225` 4,594, `tabm225` 22 epochs, `ple387` 19 epochs,
missing-normalized 111 rounds, missing-ExtraTrees 300 [S].

Mixture weights are **not** fitted on the ranking set: they are the average of
per-fold calibration weights, renormalised after zeroing anything < 1e-12
(`compose.py:23-29`) [S]. XGBoost was in the candidate pool and got weight ≈ 0
(`RESEARCH_HISTORY.md`) [S].

### 3.3 Where the 30 → 225 → 387 → 449 feature growth comes from [S]

This is the part the brief calls "30 features". It is 30 on `main` only. On the
V3 branch:

- **225 base** = the `main` feature pipeline (with `quality`/`local_calendar` on)
  plus augmentation blocks. `sequence_leaf63.py:51-53`:
  ```python
  x, receipts = run_augmented.augment(x, ['source_past','source_twosided','surface_T','trajectory'])
  x, extra    = run_information.additional(x, ['conventions','geometry','weather_T'])
  ```
  I **cannot** reconstruct the exact 225 column names: they live in
  `manifest['feature_columns']` inside the private run directory, which is not in
  Git (`final_ordinary/run.py:92-94` reads it from
  `private_runs/...` ). [S that it is private; the list itself is undeterminable
  from the repo.]
- **+112 = 337** — "sequence flatten": the **last 4 departures and last 4
  completed arrivals** before the query, same airport, same UTC month, strictly
  `< t`, within 60 min, self/same-flight excluded
  (`sequence_context/cache.py:179`: `'last16DEP bytakeoff and16completedARR
  byinblock, strict<queryT, within60m, sameairport/UTCmovementmonth, excludes
  self/sameflight'`; `missing/sequence_flatten/build.py:98` keeps the nearest 4
  per phase). 14 channels per neighbour
  (`build.py:21-23`, `cache.py:24`):
  `offset_AOBT_3_flt`, `offset_EOBT_1_flt`, `offset_IOBT_flt`, `offset_LOBT_flt`,
  `offset_SCHED_TIME_UTC_mvt`, `aobt_second`, `movement_second`,
  `arrival_duration`, `age_sec`, `landing_age_sec`, `same_runway`, `same_stand`,
  `same_operator`, `padding_missing`. 14 × 2 phases × 4 ranks = **112**.
- **+50 = 387** — `deeper_lgb/context_union.py:38`: `'Original225 +
  nearest4DEP/4completedARR112 + past/future/day50 = 387 columns'`. The 50
  "retrospective" columns are a past/future/same-day context cache; its builder is
  not in the snapshot (only the consumer). [S for the composition; the 50 names
  are undeterminable.]
- **449** = 225 base + **224** = the same flatten with **8** neighbours per phase
  instead of 4 (`sequence_leaf63_eight.py:42`: `assert len(manifest['features'])
  == 224`) [S].

**Measured value of this expansion** (`RESEARCH_HISTORY.md`) [S]:
> "Expanding LightGBM from **30 to 225** observed fields improved seasonal RMSE by
> about **10.87 s** at the smaller capacity setting and **11.38** at the larger."
> "Removing 24 peer departure-clock channels eliminated about **98.7 %** of the
> observed improvement in one matched 600-tree context experiment."

i.e. almost the entire value of the ordered-neighbour block comes from the
**clock channels of neighbouring departures**, not from counts.

### 3.4 External data — the brief's "no external data" is wrong for V3 [S]

`main`'s README says no external data, and that is true of the 30-feature release.
The V3 experts eat two external sources:

- **METAR** from Iowa State Mesonet ASOS:
  `breakthrough_20260916/weather/download_public.py:38` builds
  `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?...` for the exact 10
  competition ICAOs (`STATIONS` :16) for all of 2025 plus 2026-01 and 2026-07.
  `weather/build_features.py:17-43` derives 16 columns: `temperature_c`,
  `dewpoint_depression_c`, `wind_east_kt`, `wind_north_kt`,
  `wind_direction_missing`, `visibility_reported_miles`, `pressure_inhg`,
  `gust_reported_kt`, `wind_speed_kt`, `humidity_pct`,
  `lowest_cloud_reported_ft`, and binary `rain/snow/fog/thunder/freezing_reported`
  parsed from `wxcodes` (`RA|DZ`, `SN|SG|PL`, `FG|BR`, `TS`, `FZ`). Joined as
  **prior-issued only** with `STALE_SECONDS = 10800` (:18) — i.e. the most recent
  METAR strictly before the movement, plus an age feature.
- **OpenStreetMap airport geometry** via Overpass with a historical snapshot date
  `2024-12-31T23:59:59Z`: `geometry/fetch_public_geometry.py:12-21`
  (`aeroway=parking_position`, `aeroway=taxiway`, `aeroway=runway`), bounded to
  LIRF and LFPG in that script.

The V3 LGB/PLE experts use `weather_T` and `geometry` blocks
(`sequence_leaf63.py:53`) [S].

### 3.5 The missing-proxy route in V3 — the highest-leverage piece [S]

`tail240_20260916/forensics/final_missing/run.py:78-80`:
```
forest      = 'Frozen ExtraTrees300/minleaf1/maxfeatures.7/seed20260916/2CPU;
               earliermonth crossfit templates on train, full2025template for 2026 ranking'
normalized  = 'Frozen scale sqrt3600^2+(schedule-900)^2; Z=(Y-900)/scale;
               weights=scale^2/fitmean; 111 rounds, no evalset; raw reconstruction 900+scale*z'
composition = 'missing only: V4=.75V2+.25ET; selected=.75V4+.25normalized
               = .5625V2+.1875ET+.25normalized; no weight fitting'
```
ExtraTrees params verbatim (`run.py:54`):
`n_estimators=300, max_features=0.7, min_samples_leaf=1,
criterion='squared_error', n_jobs=2, bootstrap=False` [S].

The **"normalized" trick** (`tail240_20260916/models/normalized_missing_tune.py:
22, 27-42`) is a heteroscedastic reparameterisation that *provably preserves raw
MSE*:
```python
PARAMS = dict(objective='regression', metric='rmse', learning_rate=.05, num_leaves=31, ...)

def scale_of(x):
    schedule = x.schedule_proxy_sec.to_numpy(float)
    valid = np.isfinite(schedule) & (schedule != -999999)
    return np.sqrt(3600.**2 + np.where(valid, schedule - 900., 0.)**2)

def transformed(y, scale):  return (y - 900.) / scale
def reconstruct(v, scale):  return 900. + scale * v
```
fitted with LightGBM sample weights `scale**2 / mean(scale**2)`
(`final_missing/run.py:200`). Their own justification (`:55`): "RawY = 900 + s(X)·Z;
Z = (Y−900)/s(X); weights = s(X)²/fitmean. Therefore weighted scaled squared loss
equals unmodified raw squared loss divided by fitmean." Missing schedule → `s = 3600`.
No threshold was chosen from labels.

Measured missing-route ablations on the complete score cohorts (`RESEARCH_HISTORY.md`) [S]:

| arm | seasonal RMSE |
|---|---:|
| V3 as shipped | **268.663** |
| remove `normalized` | 271.074 |
| remove ExtraTrees | 271.185 |
| remove both | 275.704 |

So the two missing-proxy correctors are worth **7.04 s seasonal together**, on
1.52 % of rows. The team flags both as tail-sensitive ("Removing five
component-supporting rows per fold leaves only about 0.011 s of normalized benefit
and reverses the ExtraTrees benefit") [S].

An ExtraTrees `min_samples_leaf=20` variant **lost on all six chronological
panels** vs `min_samples_leaf=1` [S].

### 3.6 The `MVT_ID` record-order features — an unusual, cheap idea [S]

`tail240_20260916/forensics/final_missing/features.py:9-35` `record_peers`:
within each (`ADEP_mvt`, year-month) group, **sort rows by `MVT_ID_mvt`** (not by
time), take the ±w ID-neighbours for w ∈ {2, 8, 32} excluding self, and emit
`id_peer_median_time_minus_query_w{w}` (median neighbour movement time minus own
time) and `id_peer_time_spread_w{w}` (p90 − p10 of neighbour times). Six columns.

This treats `MVT_ID_mvt` as a **record-insertion-order key** and measures how
temporally coherent a row's ID-neighbourhood is — a direct probe of
data-provenance/source-batching, which is exactly the "disagreement between
off-block sources" story. There is a whole `missing/record_order/` and
`missing/nm_id_context/` directory devoted to this family [S].

### 3.7 V3 experiment log — incremental gains and named failures [S]

Incremental build-up (`RESEARCH_HISTORY.md`):

| composition | seasonal RMSE | gain |
|---|---:|---:|
| frozen V4 reference | 272.264 | — |
| + fixed 25 % `normalized` missing blend | 269.863 | 2.400 |
| replace PLE225 with PLE387 | **268.663** | 1.200 |

Matched effects [S]:
- 30 → 225 fields: **−10.87 / −11.38 s** (two capacity settings).
- capacity package change: −2.31 / −2.82 s.
- removing 24 peer departure-clock channels removes 98.7 % of the sequence gain.
- learning curve: 816,060 fitting rows beat a deterministic half-sample by
  **6.119 s** on the same June finite-clock cohort.
- CatBoost 225 → 387: +4.755 s standalone but only **+0.463 s inside the
  ensemble** → below materiality, rejected.

Named failures worth not repeating [S]:
- 32-member PLE ensemble: **−0.392 s** (worse) on the June endpoint.
- Neural relation-attention candidate: **−1.017 s**.
- Nonlinear expert gating / "current-nine" selector: **−1.119 s**.
- Additive calibration on current experts: +0.157 s, reversed by removing
  influential rows → not promoted.
- Three-expert mixture with 50 % shrinkage toward equal weights: gains of
  +0.043/+0.026/+0.005 s on three panels and losses of −0.129/−0.021/−0.018 on
  the others → **failed both frozen advancement gates**.
- Chronology-clean full-cohort rebuild scored **275.356** vs V3's 268.663 —
  6.693 s *worse* [S].
- "An analogous finite-clock formulation [of the normalized target] was negative."
- XGBoost weight ≈ 0 in the blend.
- Per-airport capacity models and linear leaves: standalone gains failed the
  ensemble endpoint.

---

## 4. Things these repos do that a single-LightGBM-on-raw-target pipeline does NOT

Ordered roughly by measured effect size, each one named and located.

| # | Thing | Where | Measured effect |
|---|---|---|---|
| 1 | **Dedicated CatBoost trained only on the 22,470 missing-`AOBT_3_flt` rows**, routed by `np.isfinite(proxy_sec)` | `train.py:115-121, 213`; `predict.py:71-74` | **−122.8 s seasonal** (486.2 → 361.7) [S] |
| 2 | **Fit `y − (MVT_TIME_UTC_mvt − AOBT_3_flt)` instead of `y`** on all finite-proxy rows | `models/residual.py:9-10`; `train.py:148`; `run_full.py:38-40` | −5.1 s alone; but it is the substrate everything else sits on [S] |
| 3 | **Expand 30 → 225 observed fields** (quality flags, source-disagreement flags, geometry, weather, batch-source context) | `sequence_leaf63.py:51-53` | **−10.9 / −11.4 s** [S] |
| 4 | **Ordered-neighbour clock channels**: for each departure, the 4 (or 8) preceding departures and 4 completed arrivals at the same airport/month, each contributing its 5 raw clock offsets + 2 age fields + same-runway/stand/operator flags + padding flag | `sequence_context/cache.py:24,179`; `missing/sequence_flatten/build.py:21-23,98` | 98.7 % of the sequence-block gain comes from the 24 peer *departure clock* channels [S] |
| 5 | **Two missing-route correctors**: ExtraTrees(300, leaf 1, max_features 0.7) + a LightGBM on `Z=(Y−900)/sqrt(3600²+(sched−900)²)` with weights `scale²`, blended `0.5625 V2 + 0.1875 ET + 0.25 normalized` | `final_missing/run.py:54,78-80`; `normalized_missing_tune.py:22-42` | **−7.04 s seasonal** [S] |
| 6 | **Long-proxy extrapolation route**: apply the residual correction to `proxy > 7200` instead of falling back to the direct model | `predict.py:67-70`; `configs/release.yaml:26` | −30.6 s seasonal, but tail-driven and bootstrap-unstable [S] |
| 7 | **Negative-proxy route to a separate raw-target model** | `residual.py:4-6`; `predict.py:61` | only 174 of 344,841 ranking rows — robustness, not score [S] |
| 8 | **Month-isolated traffic windows + explicit coverage features** `observed_window_15m_sec`, `observed_window_60m_sec`, `month_edge` so the model knows when a count is truncated | `features/traffic.py:40-46` | ablating all traffic costs +3.5/+2.6 s per fold [S] |
| 9 | **Flight-ID purging between CV stages** — remove any fit/tune row sharing `FLIGHT_ID_mvt` with a scoring row | `splits.py:26-31` | not quantified [S] |
| 10 | **Four-stage folds** (fit → tune iterations → refit on fit∪tune → score) with the final tree count fixed as the **median of per-fold early-stopped counts** | `folds.yaml`; `freeze.json` `iteration_rule`; `train.py:105-109` | not quantified [S] |
| 11 | **Ranking-proportion-weighted seasonal metric** `sqrt(w_jul·MSE_Jul + w_jan·MSE_Nov)`, w = 192122/344841 and 152719/344841 | `metrics.py:23-28` | changes which candidate wins [S] |
| 12 | **No clipping, no non-negative clamp, extreme labels retained**; nearest-even int32 rounding only | `README.md`; `reproduction.json` (range −268 … 87035) | contrast with Repo A's `between(30,7200)` filter and `np.maximum(·,0)` [S] |
| 13 | **Collision-safe length-prefixed category tokens** `"s"+len+":"+value`, missing → `"m:"`, instead of raw strings or `fillna("MISSING")` | `pipeline.py:11-13,39` | not quantified [S] |
| 14 | **`MVT_ID_mvt` record-order peer features** (`id_peer_median_time_minus_query_w{2,8,32}`, `id_peer_time_spread_w{...}`) | `final_missing/features.py:9-35` | not isolated [S] |
| 15 | **External METAR (prior-issued, 3 h staleness cap) + OSM airport geometry** | `weather/download_public.py`, `weather/build_features.py`, `geometry/fetch_public_geometry.py` | not isolated from the 225-field gain [S] |
| 16 | **Multi-family blend with weights averaged across folds, not fitted on the test set**; zeroed below 1e-12 and renormalised | `compose.py:23-29` | XGBoost got 0; lgb449 got 0.035 [S] |
| 17 | **Explicit removal of `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` from the frame before feature code runs**, with an assertion | `availability.py:13,20-23` | prevents the Repo A class of bug [S] |
| 18 | *(Repo A only, cautionary)* smoothed **leave-one-out** target means on AIRPORT / ×AIRLINE / ×RUNWAY / ×TIME30 / ×WEEKDAY×TIME30, α=10–20 | `train3.py:385-432` | unmeasured; Repo B deliberately refused this [S] |

Things **neither** repo does, contra the other investigators' repos: no LOBT-based
clipping window (`|(MVT−LOBT) − y| ≤ 3606`) exists in either [S].

---

## 5. What cannot be determined from the code

- **Repo A**: the code that scored 268.52 (v42) is not published anywhere I can
  find. Nothing about kind-earthquake's rank-21 method is knowable from this repo.
  Also unknown: why `train3`'s v3 scored 731 (the committed `makeSubmit.py`
  imports `train1`'s feature builder, so the shipped inference path does not match
  `train3`'s `FEATURES`, but I have no run log to confirm that is what happened).
- **Repo A**: no CV scores are committed (`models/cv_scores_train3.csv` is
  gitignored), so I have no internal validation numbers at all.
- **Repo B**: the exact **225 / 387 / 449 column name lists** are stored in
  private run manifests outside Git (`final_ordinary/run.py:92`); I can give the
  block composition and the 112/224-column neighbour schema exactly, but not the
  225 base names.
- **Repo B**: the 50 "past/future/day" retrospective columns — only the consumer
  (`context_union.py`) is in the snapshot, not the builder.
- **Repo B**: the LightGBM/CatBoost/TabM hyperparameters of the V3 experts
  (`adapter.PARAMS['leaf63']` etc.) are in `deeper_lgb/adapter.py` which I did not
  read in full; the `leaf63` name implies `num_leaves=63`. [X]
- **Repo B**: which of V1/V2/V3 corresponds to which local recipe is stated by the
  team, not independently verifiable (no submission bytes in Git for V1).
- Neither repo's published code has been shown by me to reproduce its leaderboard
  score; both teams' best scores post-date their last public commit.

---

## Applicable to resilient-hedgehog

Ordered by cost to test, cheapest first. Effect sizes are **Repo B's local
seasonal numbers**, not predictions for our pipeline.

**Tier 1 — one afternoon each, no new data**

1. **Train a separate model on the 22,470 unmatched rows and route by NM-match
   presence.** We already have exactly this partition (2,062,577 / 22,470 — the
   identical counts). Repo B's log says this is worth **−122.8 s seasonal**, the
   single biggest item in their entire experiment table, and it collapses
   missing-cohort RMSE from ~3187 to ~1532. Train it on **all** 22,470 rows, not a
   sample (`train.py:213`). If we currently pool those rows into one model, this
   is the highest expected-value change available.
   *Source: `reports/experiments.csv`, `reports/selection_decision.md`,
   `src/taxiout/train.py:115-121,213`.*

2. **Switch the matched-row target to the residual `y − (MVT_TIME_UTC_mvt −
   AOBT_3_flt)` and add the proxy back at predict time.** Trivial code change
   (`residual.py:9-17`). Worth only −5 s on its own in Repo B's log, but it is the
   substrate for items 3, 6 and 7, and it is what merry-quicksand (257.55) and
   javidmardanov (268.56) also do. Keep `takeoff_minus_AOBT_3_flt` and
   `proxy_missing` as *features* as well — Repo B does both.

3. **Carve the third regime out of our matched rows: route `proxy < 0` to a
   raw-target model, and apply the residual correction to `proxy > 7200` rather
   than falling back.** Exact rule at `models/residual.py:4-6` and
   `predict.py:67-70`. The long-proxy half is worth −30.6 s locally but is
   tail-driven — test it with leave-one-day-out before trusting it. The negative
   half touches ~174 of 344,841 rows; do it for safety, not for score.

4. **Stop clipping the target and stop clamping predictions.** Repo B keeps every
   extreme label, fits raw squared loss, and serialises with `numpy.rint` nearest
   even and no clip (range −268 … 87,035 s). Repo A filters training to
   `30 ≤ y ≤ 7200` and floors predictions at 0 — and is the worse pipeline. If we
   filter or clamp, remove it and re-measure.

5. **Adopt the ranking-proportion seasonal metric for model selection**:
   `sqrt(192122/344841 · MSE_July + 152719/344841 · MSE_Nov)` (`metrics.py:23-28`).
   Costs nothing; changes which candidate we promote. Repo B's own table shows
   candidates that win on one month lose on the weighted score.

6. **Audit for label-derived features that go NaN on `ranking.parquet`.** Repo A's
   `PREV_TAXI_MEAN_*` / `QUEUE_REGIME` are computed from `TAXITIME_SEC_mvt`, which
   `makeSubmit.py:201-202` sets to NaN at inference — a silent 300-s-class defect
   that shipped. Grep our feature code for any use of the target in a rolling or
   group aggregate and confirm it is reproducible from `ranking.parquet` alone.
   Repo B hard-drops `BLOCK_TIME_UTC_mvt`/`TAXITIME_SEC_mvt` from the frame and
   *asserts* their absence (`availability.py:13,20-23`) — cheap insurance.

7. **Purge flight-ID overlap between training and validation folds**
   (`splits.py:26-31`). Two lines; makes our CV numbers honest.

**Tier 2 — a day or two**

8. **The "normalized" missing-proxy model.** On the unmatched cohort only, fit
   LightGBM on `Z = (y − 900) / sqrt(3600² + (schedule_proxy − 900)²)` with sample
   weights `scale² / mean(scale²)`, reconstruct `900 + scale·Z`, and blend it in
   at a *fixed* 25 %. The weighting makes the objective provably equal to raw MSE
   (`normalized_missing_tune.py:22-42,55`). Combined with an
   `ExtraTreesRegressor(n_estimators=300, max_features=0.7, min_samples_leaf=1,
   bootstrap=False)` at 18.75 %, this was worth **−7.04 s seasonal** for Repo B.
   Note: leaf=20 lost on all six panels — use leaf 1.
   Also note: "An analogous finite-clock formulation was negative" — do not apply
   it to the matched rows.

9. **Ordered-neighbour clock channels.** For each departure, take the last 4
   departures and last 4 completed arrivals at the same airport within 60 min
   (strictly earlier, same UTC month, excluding self and same flight), and emit
   per neighbour: its 5 clock offsets (`AOBT_3`, `EOBT_1`, `IOBT`, `LOBT`,
   `SCHED`), `aobt_second`, `movement_second`, `arrival_duration`, `age_sec`,
   `landing_age_sec`, `same_runway`, `same_stand`, `same_operator`,
   `padding_missing` → 112 columns. Repo B measured that the **24 peer
   departure-clock channels alone carry 98.7 %** of this block's gain, so a
   cheaper first cut is: the 4 preceding departures × {`takeoff_minus_AOBT_3`,
   `age_sec`, `same_runway`} = 12 columns.
   *Source: `sequence_context/cache.py:24,179`, `missing/sequence_flatten/build.py:21-23,98`.*

10. **Window-coverage features.** `observed_window_15m_sec`,
    `observed_window_60m_sec` = `min(seconds since month start, w·60)` and
    `month_edge` (`traffic.py:40-46`). Three columns that tell the model when a
    traffic count is truncated. Removing traffic entirely cost Repo B +3.5/+2.6 s,
    so the counts matter; these make them honest at boundaries.

11. **Category tokenisation.** `"s" + len(v) + ":" + v`, missing → `"m:"`, plus
    concatenated `airport_stand` and `airport_runway` tokens
    (`pipeline.py:11-13,41-43`). Cheap; prevents `"A1"+"0"` colliding with
    `"A"+"10"` in crossed keys. Relevant to us because we cross airport × stand ×
    runway.

12. **`MVT_ID_mvt` record-order peer features** (`final_missing/features.py:9-35`):
    within (airport, month), sort by ID, and for w ∈ {2,8,32} emit
    `median(neighbour_time) − own_time` and `p90−p10` of neighbour times. Six
    columns probing record provenance. Not isolated in their log, but it is the
    kind of feature that targets exactly the "off-block source disagreement"
    mechanism, and it costs ~20 lines.

**Tier 3 — larger commitments, with their measured caveats**

13. **More training rows before more model.** Their fixed-capacity learning curve
    found 816,060 rows beat a half-sample by **6.12 s** on the same cohort. If we
    subsample, stop.

14. **Blend families with fold-averaged weights** (LightGBM + CatBoost + a neural
    tabular model), weights averaged across folds and renormalised after zeroing
    (`compose.py:23-29`). Their best single expert had weight 0.44; the blend
    bought them roughly the gap between V2 (294.6) and V3 (278.0) together with
    the 225→387 expansion. Caveat: their *nonlinear* gating and shrinkage
    experiments both failed (−1.12 s and failed advancement gates), and XGBoost
    earned zero weight. Use fixed fold-averaged weights, not a learned gate.

15. **External data (METAR + OSM geometry).** Available and legal, and part of
    V3's 225-field base. But its contribution is never isolated in their log, and
    it is the most expensive item here. Do 1–12 first.

**Explicitly do not bother with**

- Huber loss with CatBoost default Newton leaves — Repo B got 19,609 s and 4,947 s
  RMSE (degenerate 1–3-tree models); with `leaf_estimation_method: Gradient` it
  was still worse than plain RMSE (667.9 / 425.0 vs 369.4 / 275.7).
- Adding "quality"/local-calendar flag features on their own: +2.04 s seasonal,
  below their own materiality bar.
- A 32-member neural ensemble (−0.39 s), relation attention (−1.02 s), nonlinear
  expert gates (−1.12 s), additive post-hoc calibration (+0.16 s and unstable).
- Copying anything from `col3name/air-data` on the strength of the 268.52 number.
  The code in that repository corresponds to submissions that scored **599.67 and
  731.15**.

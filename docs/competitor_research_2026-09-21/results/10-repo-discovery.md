# Slot 10 — repo-discovery: exhaustive sweep for public PRC 2026 work

Investigator: SubAgent (slot 10). Date of sweep: 2026-09-21.
Marking: **[S]** = read from a cited source. **[X]** = my own inference/extrapolation.

---

## 0. Already-known (covered by other slots — not re-investigated here)

| Repo | Team | LB rank / best RMSE (2026-09-21) |
|---|---|---|
| github.com/ChandanHegde07/OpenAir | — | (slot 06) |
| github.com/Phoenix-Ops-LTD/prc2026-taxiout | zestful-fountain | **29 / 273.5557** [S] |
| github.com/javidmardanov/PRC-Data-Challenge-2026 | elegant-alligator | **22 / 268.5583** [S] |
| github.com/sergiuv11/prc-data-challenge-2026 | jubilant-vase | **76 / 299.2862** [S] |
| github.com/ahmetabdullahgultekin/prc-taxiout-2026 | — | (slot 09) |

Ranks/scores above are from the public leaderboard mirror (see §5) [S].

---

## 1. NEW substantive repos (code + documented method)

### 1.1 saurabhdxt-labs/prc-challenge — team **merry-quicksand** ★ highest-value find
- URL: https://github.com/saurabhdxt-labs/prc-challenge [S]
- Who: GitHub org `saurabhdxt-labs`; team name `merry-quicksand` stated in README [S].
- Score: **rank 11 of 168, best RMSE 257.5532**, 22 submissions, last 2026-09-19 [S, leaderboard mirror].
- Licence GPL-3.0; 44 commits; dirs `configs/ plans/ reports/ scripts/ tests/ tools/` [S].

**Method [S]:** The central move is a *reparameterisation of the target*. Instead of regressing taxi-out
duration directly, they regress the disagreement between two clocks:
`y = proxy − delta`, where `proxy = MVT_TIME − AOBT_3` (known exactly for matched rows) and
`delta = BLOCK_TIME − AOBT_3` (the quantity actually predicted). On identical folds this moved
matched-row RMSE from **285.24 → 252.66** and is described as the single largest modelling decision in
their submission. The ~1.53 % of rows with no Network Manager match get a separate model conditioned
on airport identity and schedule offset. Models are gradient-boosted trees (LightGBM fold reports and
CatBoost native model files are both present in `reports/`) [S]. Validation holds out whole *months*
(Jan+Jul 2025 primary fold; Feb+Aug as a replication fold), target encodings are fitted inside folds
only, every RMSE claim carries a 95 % bootstrap interval with paired comparison, and they explicitly
state that no hyperparameter was chosen from leaderboard feedback. Naive benchmarks are documented
(constant 686.07; airport-mean 659.9). They log negative results too, including features that cost up
to 144 s. `reports/` holds 110+ dated artefacts including `FEATURE_IMPORTANCE_2026_09_10.md`,
`GAP_LOCATED_2026_09_10.md`, `PIPELINE_TO_245_2026_09_10.md`, `MATCHED_MODEL.md`,
`UNMATCHED_DIAGNOSTIC_2026_09_11.md`, `DATA_DICTIONARY.md`, `ADSB_GATE.md`, `RED_TEAM.md`, and
repeated references to a `PATH_245` target [S].

### 1.2 skylinkapi/prc-data-challenge-2026-kind-mango — team **kind-mango** ★ most detailed public writeup
- URL: https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango [S]
- Who: GitHub org `skylinkapi`; solo entry, team `kind-mango` [S].
- Score: **rank 53 of 168, best RMSE 287.3339** (v57), 55 submissions, last 2026-09-20 [S].
- Licence GPL-3.0. Files: `docs/ models/ src/ src_v2/ src_v3/ submission/ RECAP.md README.md REPRODUCE.md` [S].

**Method [S]:** Scoring stack = three-seed LightGBM with `linear_tree=True` (base "R_all_v26",
retrained through v57) + a per-airport "LIRF regime head" (5-member `R_norm_LIRF` mean with a
calibrated fallback-probability mixture) + a deterministic band-lookup for null-flight rows at LIRF
with `sd > 14400` + an `ITY340` offset rule for extreme schedule-delay rows, routed by per-airport
delegation. **97 base features**: 10 categorical (airport, destination, runway, stand, aircraft type,
wake, market segment, operator, flight rule, flight type), 14 temporal/schedule (hour, DoW, month,
schedule-delay variants, EOBT/IOBT differences, flight-nullity flags), 18 METAR (crosswind, headwind,
gust, visibility, ceiling, cloud codes, precipitation, temperature), 13 congestion (arrival counts,
same-runway traffic, taxi-in rolling means), 6 turnaround features linking departures to previous
arrivals, 4 disruption meters over 60-min windows, 18 EUROCONTROL ATFM daily metrics, 16 operator
historic encoders, physical (haversine distance, path length, turn count), 5 advanced physical proxies,
9 OPDI event counts, 9 OPDI live-taxi rolling statistics. **Validation:** train on 2025 months 2–6 and
8–10, early-stop on 11–12, hold out months 1 and 7; hold-out clean RMSE (30–7200 s) = 274.08 s.
**Ethics note:** they deliberately exclude `AOBT_3_flt` on ranking rows and `LOBT_flt`, saying the
organisers permit it but they consider it exploiting the ranking process — i.e. the opposite choice
from merry-quicksand [S]. **[X]** That self-imposed handicap is a plausible part of the ~30 s gap
between them and merry-quicksand.

**`RECAP.md` ablation ledger [S]** (unusually valuable — a full version-by-version log):

| Ver | RMSE | Δ | What |
|---|---|---|---|
| v10 | 560.91 | — | earlier best, broken metric |
| v21 | 330.12 | −230.79 | fixed 3 defects: TAXITIME filter hid the tail; predictions clipped at 7200; `sched_delay` clipped to [−1800, 3600] |
| v26 | 303.71 | −26.41 | **dropped** `ec_*` and `opdi_*` features |
| v33 | 301.87 | −1.84 | isolated `R_norm` gain |
| v40 | 299.97 | −1.90 | tempo features into base |
| v41 | 299.31 | −0.66 | tempo in `R_norm_LIRF`, cap removed |
| v47 | 296.57 | −2.74 | 12-month refit |
| v51 | 288.25 | −8.32 | trained on served rows only |
| v57 | **287.33** | — | team best |

Documented failures: v39 twelfth-audit rewrite (+352, one LIRF row = 62–73 % of the loss); v58 operator-
encoder leakage (+1.29); v59 arrival-side drift features (+0.37); v60 leave-one-month-out encoders broke
train/serve alignment (+2.25); v54 anchored-offset target `y − anchor` failed on bimodal EOBT_1 (+35.86);
v49 dropping 56 drifted columns from the LIRF head (+0.42). Also: classifiers overfit the 2025 hold-out
(+5 to +195 s live); runway high-cardinality features +7.6 s; quantile regression and SSL drift
correction gave no signal. Coverage/drift findings: `dewpt_spread` zeros at LEMD/LSZH, `eobt1_sched`
zeros at LTFM, LSZH January 2026 traffic 25 % below 2025. A v33 rebuild reproduced bit-identically
(0.0000 s max diff on 344,841 rows) [S].

### 1.3 col3name/air-data — team **kind-earthquake**
- URL: https://github.com/col3name/air-data [S]
- Score: **rank 21 of 168, best RMSE 268.5216**, 26 submissions, last 2026-09-17 [S].
  (Team identity inferred from `kind-earthquake_v2.parquet` / `_v3.parquet` in the repo root [S] — this
  is a direct filename match to the leaderboard naming convention, so I treat it as [S], not [X].)
- Method [S]: CatBoost with RMSE objective. Features: temporal; airport×airline×aircraft×destination
  interactions; airport×runway×stand interactions; 15/30/60-minute traffic counts; runway-load metrics;
  previous taxi-out statistics; traffic/queue/runway "regimes". Temporal validation over Sep–Dec.
  Files: `train0.py`…`train4.py` plus `train2.1.py`/`train2.2.py`, `submit.py`, `makeSubmit.py`,
  `data/`, `tests/`. README calls it "a strong first baseline" and names airport-normalised congestion
  regimes and a proper inference pipeline as next steps. No RMSE is printed in the README [S].

### 1.4 satam2/knowledgeable-helicopter — team **knowledgeable-helicopter**
- URL: https://github.com/satam2/knowledgeable-helicopter [S]
- Score: **rank 36 of 168, best RMSE 278.0146**, only 3 submissions, last 2026-09-16 [S].
- Method [S]: three-model split on a proxy = takeoff − observed `AOBT_3_flt`: a residual CatBoost for
  non-negative finite proxies (0–7200 s), a direct model for negative proxies, and a dedicated model for
  missing proxies. 30 categorical/calendar/clock/prior-traffic features; traffic windows use prior events
  within each UTC month. **No external datasets.** Selected model `residual_long_proxy_specialist`:
  dev RMSE 331.176 s (seasonal validation weighting Jul/Jan proportions), confirmation RMSE 266.406 s on
  December 2025. Five validation folds. GPL-3.0; `configs/ docs/ reports/ scripts/ src/taxiout/ tests/`.
- **[X]** Same proxy idea as merry-quicksand but modelled as a 3-way regime split rather than a delta
  reparameterisation; the 30-feature budget is ~1/3 of kind-mango's and it still sits 17 places higher,
  which suggests the target formulation dominates feature count here.

### 1.5 victoralcadi/prc-data-challenge-2026
- URL: https://github.com/victoralcadi/prc-data-challenge-2026 [S]
- Team name not stated; topics `aviation, eurocontrol, lightgbm, opensky-network, taxi-out-time` [S].
- Method [S]: LightGBM with month-holdout validation (Jan + Jul). Features are time-based, congestion/
  queue, and group-statistic, and are deliberately **anchored to takeoff time rather than push-back**
  because off-block timing is withheld on the test rows. Prints training RMSE in seconds alongside a
  naive airport-median baseline. Layout `src/prc2026/` (config, data, features, training, submission),
  `scripts/quicklook.py`, `tests/`, `docs/`, GitHub Actions CI, GPLv3, `pyproject.toml`.
- No own score reported; README cites the mid-September leaderboard (~246 s top, top-10 inside ~275 s) [S].

### 1.6 sqcm9896/prc-2026-likable-river — team **likable-river**
- URL: https://github.com/sqcm9896/prc-2026-likable-river [S]
- Score: **rank 91 of 168, best RMSE 315.8111**, 9 submissions, last 2026-09-09 [S].
- Method [S]: blend `perapt-catboost × 0.8 + lgbm-L2 × 0.2 + join-fail patch` over "v1 feats".
  Hold-out RMSE pooled 359.2 s (Jan 355.2, Jul 362.4). Governance rule: a feature only advances if
  **both** monthly hold-outs improve. Submission v3 validated 2026-09-07 with ETag verification.
  `src/`, `plan/`, `reports/`, `tests/`, `MODEL_CARD.md`, Python 3.12, GPLv3.

### 1.7 GLAPM/prc-2026-sincere-donkey — team **sincere-donkey**
- URL: https://github.com/GLAPM/prc-2026-sincere-donkey [S]
- Score: **rank 161 of 168, best RMSE 625.4325**, 7 submissions, last 2026-09-10 [S].
- Method [S]: heterogeneous ensemble of LightGBM + XGBoost + CatBoost. Stated "key innovation" is an
  early merge of categorical domains across all datasets before training so unseen categories at
  submission time are handled. Features: hour / DoW / month from scheduled UTC times, aircraft type,
  operator codes; outliers removed at <0 s or >2 h. 4 files only: `train_ensemble.py`, `eda_script.py`,
  README, LICENSE. No metrics in the README.
- **[X]** Low value as a method source, but the leaderboard score confirms this is a real entry.

### 1.8 kagervig/PRC-Challenge-2026
- URL: https://github.com/kagervig/PRC-Challenge-2026 [S]
- No team name, no RMSE in the repo [S]. 4 commits, created 2026-09-19 (very fresh).
- Contents [S]: `clean_flights.py`, `export_flights.py`, `count_distinct.py`, `cascade_analysis.py`,
  `airline_consistency.py`, `significance_analysis.py`, `per_airport.py`, `open_questions_analysis.py`,
  `model.py`, `tune.py`, `tune_congestion.py`, plus `plan.md`, `model_plan.md`, `learnings.md`,
  `todo.md`, `flights.csv`, `submitting.parquet`, `predictions/`, `updated-data/`. GPL-3.0.
- **[X]** The `cascade_analysis.py` / `airline_consistency.py` / `per_airport.py` trio plus a written
  `learnings.md` makes this worth a skim for feature ideas even though no score is attached.

### 1.9 fenil264/prc-2026-taxi-out — team **joyful-lemon**
- URL: https://github.com/fenil264/prc-2026-taxi-out [S]
- Score: **rank 78 of 168, best RMSE 300.2712**, 22 submissions, last 2026-09-16 [S].
- README documents only setup (uv, OpenSky credentials) and the submission naming
  `joyful-lemon_v{n}.parquet`; method/features/validation are **not** in the README — they would be in
  `notebooks/` (starting `01_data_access.ipynb`), `src/prc/`, `models/`, `reports/figures/` [S].

---

## 2. NEW repos that are empty or near-empty (registered but nothing to learn)

| Repo | Team | State |
|---|---|---|
| github.com/AbdullahKhan-x/PRC-2026 | logical-ladder (stated in README) | Only `download.py`, `peek.py`, `requirements.txt`, LICENSE. No model code. Not on the leaderboard as of 2026-09-21. [S] |
| github.com/Necklacesaura/PRC-Data-Challenge-2026 | — | Scaffolding only (`data/ notebooks/ src/ results/ docs/ PREPARATION.md`), "preparation phase". [S] |
| github.com/ruchir321/prc-data-challenge-2026 | — | Repository is empty. [S] |
| github.com/RavindraTarunokusumo/prc-data-challenge-2026 | — | Repository is empty. [S] |
| github.com/georgejhanlon/prc-taxiout-2026 | — | Repository is empty. [S] |
| github.com/victorvb22/taxi-out-prediction | — | Repository is empty. [S] |
| github.com/Enzoferrariz/PRC-Data-Challenge-2026 | — | README title only, no content. [S] |

**Deleted / vanished:** `github.com/LeeMarshall1113/prc-taxiout-2026` — described "Entry for the PRC Data
Challenge 2026: predicting taxi-out time at European airports (RMSE, closes 2026-10-11)" and is still in
search-engine indexes, but now returns **HTTP 404** and does not appear in a `user:LeeMarshall1113` repo
listing (22 repos, none PRC-related) [S]. **[X]** Taken private or deleted between indexing and today.

---

## 3. Official EUROCONTROL / euctrl-pru — NEGATIVE result on a starter kit

I listed all 72 repos in `org:euctrl-pru` [S]. For 2026 there is exactly **one** challenge repo:
- **github.com/euctrl-pru/prc_data_challenge_website_2026** — the Quarto source of the challenge site
  (`index.qmd`, `data.qmd`, `ranking.qmd`, `_site/`, plus a generated `_site/teams/<team>.html` page per
  registered team) [S]. Last updated 2026-09-20.

**There is no official 2026 starter kit, baseline notebook, or data-description code repo.** [S]
Prior-edition website repos exist (`prc_data_challenge`, `prc_data_challenge_website_2025`) but carry no
2026 baseline. Adjacent EUROCONTROL packages that may be useful as *data tooling* rather than baselines:
`euctrl-pru/OPDI` (OpenSky ETL pipelines), `euctrl-pru/HexAeroPy` / `HexAeroSpark` (detect runway,
taxiway and stand usage from trajectory coordinates), `euctrl-pru/opensky-airport-coverage`,
`euctrl-pru/paper-2022-OSN-SurfaceFuelBurn` (taxi-phase fuel/LTO work) [S].

Note: the official per-team pages (e.g. `.../dc2026/teams/outgoing-wizard.html`) contain only
registration metadata — name, country, members, affiliation, free-text blurb. **They do not list repo
URLs, scores, or submission history**, so they are not a repo-discovery vector [S].

---

## 4. Paper track / preprints — NEGATIVE result

- The official 2026 site says organisers "strongly encourage teams, particularly the top-ranked teams, to
  submit a paper to **JOAS** (Journal of Open Aviation Science) describing their approach and the inner
  workings of their models", and points at the 2024/2025 solution and data-description papers as
  precedent [S]. There is **no separate formal paper track or deadline** announced for 2026, and I found
  **no 2026-specific JOAS collection page** [S].
- **No arXiv preprint** about PRC 2026 taxi-out exists that I could find [S — repeated searches returned
  only GitHub repos and unrelated older taxi-out literature].
- **No Zenodo deposit** for PRC 2026 [S].
- **[X]** Expected: the competition closes 2026-10-11, so writeups would not be due yet.

Related but *not* PRC-2026 literature surfaced incidentally, useful to slot 19 rather than here:
"Interpretable Taxi-Out Time Prediction of Departure Flights Using Stacking Ensemble Learning and SHAP
Analysis" (Scientific Reports, 2026) and "Additional Taxi-Out Time Prediction for Flights at Busy Airports
by Fusing Flow Control Information" (Applied Sciences 14(21):9968) [S].

---

## 5. Leaderboard mirror (new infrastructure find)

- **github.com/arquicanedo/prc-challenge-2026** — a leaderboard viewer by Arquimedes Canedo, deployed at
  `https://prc-challenge-2026.vercel.app/`. It holds **no data**; it reads a CORS-bridging mirror at
  **`https://prc-leaderboard.fly.dev/data.json`**. MIT-licensed code; the data belongs to the organisers.
  The official `ranking.html` page links to this viewer as the leaderboard [S].
- I fetched `data.json` directly: as of **2026-09-21T02:49:19Z** it reports **2144 total submissions across
  168 teams**, top 5 = vigorous-whistle 239.1335, jovial-uniform 240.5497, jolly-lobster 242.0575,
  enthusiastic-daisy 243.6497, gentle-igloo 243.8347 [S]. (Slot 01 owns this; noted here because the
  endpoint is the machine-readable source and the viewer repo is how you find it.)
- Other 2026 facts from the official site: 1 Sep – 11 Oct 2026, €5,000 across the top 3, RMSE on Jan+Jul
  2026 movements, 344,841 scored departures, 5 submissions/day cap, 1 GB per team bucket,
  `<team>_v<n>.parquet` naming, discussion on the OpenSky Discord `#prc-data-competition` [S].

---

## 6. Platforms with NOTHING — explicit negative results

I searched each of these and found **zero** PRC-2026 taxi-out material:

- **GitLab** — no repos. Searched `gitlab.com "prc data challenge"`, `gitlab taxi-out 2026 eurocontrol`. [S]
- **Hugging Face** — no models, datasets or Spaces. [S]
- **Kaggle** — no notebooks or datasets (the competition is not hosted on Kaggle; data is credentialed
  S3 behind `credentials.json`, which **[X]** makes a Kaggle mirror both unlikely and licence-violating). [S]
- **Zenodo** — no deposits. [S]
- **arXiv** — no preprints (see §4). [S]
- **Medium / Substack** — no write-ups. Searches returned only unrelated "taxi app" and "NYC taxi
  duration" posts. [S]
- **LinkedIn / X (Twitter)** — no indexed posts describing a 2026 method. [S]
- **YouTube** — no videos. [S]
- **Discord** — the official invite is `https://discord.gg/RPh89jpVVz` and the channel is
  `#prc-data-competition` on the OpenSky Network server. **Per instruction I did not attempt to join.**
  Search engines index **no message content** from that channel — only the official site's pointer to it.
  So there is no publicly readable Discord material. [S]
- **CompeteHub** (`competehub.dev`) carries a listing page for the competition but it is a scraped
  mirror of the official description, no method content. [S]

### GitHub code-search coverage caveat
GitHub's code index for this topic is **partial**: searches on the distinctive column names
`TAXITIME_SEC_mvt`, `MVT_TIME_UTC_mvt`, `AOBT_3_flt`, `MVT_ID_mvt` return only 26–49 hits, all inside
`euctrl-pru/prc_data_challenge_website_2026`, `Phoenix-Ops-LTD/prc2026-taxiout`,
`ahmetabdullahgultekin/prc-taxiout-2026` and `col3name/air-data` [S]. Repos found by *repository* search
(kind-mango, merry-quicksand, likable-river, joyful-lemon, victoralcadi, satam2) do **not** appear in code
search at all. **[X]** Their file contents are not yet indexed, so code search alone would have missed
most of this list; repository search + web search were the productive vectors.

### Team-name cross-reference — NEGATIVE result for the top 10
I searched GitHub code for the exact quoted team names of the top-ranked teams
(vigorous-whistle, jovial-uniform, jolly-lobster, enthusiastic-daisy, gentle-igloo, gentle-tractor,
zesty-puzzle, bubbly-telephone, quick-boat, gentle-lemon, strong-turtle, strong-crane). Every hit was a
generated page inside `euctrl-pru/prc_data_challenge_website_2026/_site/teams/` — **no participant repo
anywhere on GitHub mentions any top-10 team name** [S]. **[X]** The entire top 10, including
youthful-giraffe (rank 7, 245.0207), is competing without a public repo. The best *public* entry on the
board is merry-quicksand at rank 11 / 257.5532 — ahead of javidmardanov (rank 22), which the brief called
"best public".

---

## 7. Full inventory of what I searched

GitHub MCP `search_repositories`: "prc data challenge 2026 taxi-out", "prc2026 OR prc-2026 OR PRC Data
Challenge", `org:euctrl-pru`, `topic:prc-data-challenge-2026`, "taxiout in:name,description,readme
created:>2026-08-01", "taxi-out in:description,readme created:>2026-08-20", "prc in:name
created:>2026-08-25 pushed:>2026-09-01", "taxi-out eurocontrol opensky created:>2026-08-15",
"344841 / submitting.parquet", twelve top-team names.
GitHub MCP `search_code`: `TAXITIME_SEC_mvt`, `MVT_TIME_UTC_mvt`, `AOBT_3_flt`, `MVT_ID_mvt`,
"prc-data-challenge 2026 taxi", and the quoted team names above.
WebSearch: prc-data-challenge / taxi-out / taxiout / TAXITIME_SEC / MVT_TIME_UTC / AOBT_3 / prc2026 /
euctrl / "PRC Data Challenge" crossed with github, gitlab, huggingface, kaggle, zenodo, arxiv, medium,
substack, linkedin, x.com, youtube, discord, JOAS; `site:ansperformance.eu dc2026 teams`.
WebFetch: official site root, `ranking.html`, a team page, the leaderboard mirror JSON, and every
candidate repo above.

---

## New leads worth a deep dive — ranked

1. **saurabhdxt-labs/prc-challenge (merry-quicksand, rank 11, 257.55)** — the best public entry on the
   board, ~11 s ahead of javidmardanov. The `y = proxy − delta` clock-disagreement reparameterisation is
   a testable, self-contained idea with a stated 285.24 → 252.66 effect on matched rows, plus a separate
   unmatched-row model for the 1.53 % NM-miss population. `reports/` (110+ files, incl.
   `PIPELINE_TO_245`, `FEATURE_IMPORTANCE`, `GAP_LOCATED`, `UNMATCHED_DIAGNOSTIC`, `DATA_DICTIONARY`,
   `RED_TEAM`) is the single richest public artefact set in the competition. **Read this first.**
2. **skylinkapi/prc-data-challenge-2026-kind-mango (rank 53, 287.33)** — worse score, but `RECAP.md` is a
   complete ablation ledger with signed deltas for ~15 accepted and rejected changes, a 97-feature
   catalogue with group counts, and hard-won failure modes (clipping/label-filter bugs worth 230 s;
   leave-one-month-out encoders breaking train/serve alignment; operator-encoder leakage; anchored-offset
   target failing on bimodal EOBT_1). Highest information-per-hour of anything I found. Also the only
   source naming LIRF as a special regime and flagging LSZH/LEMD/LTFM data-drift.
3. **col3name/air-data (kind-earthquake, rank 21, 268.52)** — a strong score from a plain CatBoost
   baseline with a compact, concrete feature recipe (15/30/60-min traffic counts, runway load,
   airport×runway×stand interactions, traffic/queue/runway regimes). Good calibration point for how far
   a clean baseline gets, and directly comparable to our own feature set.
4. **satam2/knowledgeable-helicopter (rank 36, 278.01, only 3 submissions)** — 30 features, no external
   data, three-way proxy regime split, and an explicit seasonal-vs-December validation contrast
   (331.18 dev vs 266.41 confirmation). The dev/confirmation gap is itself a lesson about fold choice.
5. **`https://prc-leaderboard.fly.dev/data.json` + arquicanedo/prc-challenge-2026** — machine-readable
   live leaderboard (168 teams, 2144 submissions, per-team `n` and `lastAt`). Hand to slot 01; it enables
   submission-curve tracking without scraping.
6. **victoralcadi/prc-data-challenge-2026** — no score, but the one repo that states the *leakage-safe*
   framing explicitly: anchor every feature to takeoff time, not push-back, because off-block is withheld
   on test rows. Worth a read purely as a leakage checklist.
7. **kagervig/PRC-Challenge-2026** — no score, very fresh, but has `cascade_analysis.py`,
   `airline_consistency.py`, `per_airport.py` and a written `learnings.md`. Cheap skim for feature ideas.
8. **fenil264/prc-2026-taxi-out (joyful-lemon, rank 78, 300.27)** — method lives only in the notebooks;
   requires actually opening `notebooks/` to learn anything. Low priority.
9. **sqcm9896/prc-2026-likable-river (rank 91, 315.81)** — weak score, but the "advance a feature only if
   *both* Jan and Jul hold-outs improve" gate is a cheap governance idea to copy into our own CV loop.

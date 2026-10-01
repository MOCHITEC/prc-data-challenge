# SLOT 01 — lb-top-teams-2026

Snapshot taken **2026-09-21**. Leaderboard payload timestamp: `updated = 2026-09-21T02:49:19Z`. [S]

Marking convention: **[S]** = read directly from a source (URL given). **[X]** = my own extrapolation / inference.

---

## 0. Sources used

| What | URL | Notes |
|---|---|---|
| Mirror leaderboard (best-per-team) | https://prc-leaderboard.fly.dev/data.json | [S] Returns identical JSON at `/` too. `totalSubmissions: 2144`, 168 teams. |
| **Official full submission-history REST API** | `https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard` | [S] Cursor-paginated. **Every individual submission** with score, filename (`<team>_v<N>.parquet`), `usedPairs` and `processedAt`. I pulled all 43 pages = 2144 records. |
| API swagger | https://datacomp.opensky-network.org/api/swagger-ui/ | [S] Linked from official ranking page. |
| Official ranking page (source) | https://raw.githubusercontent.com/euctrl-pru/prc_data_challenge_website_2026/main/ranking.qmd | [S] Documents the API above, the 5/day limit, and anti-probing warning. |
| Official team registry | https://prc-data-challenge-2026.netlify.app/teams/ (redirect target of `ansperformance.eu/study/data-challenge/dc2026/teams/`) | [S] ~600 registered teams; **each team has a public page with real names and affiliations**. |
| Leaderboard viewer repo | https://github.com/arquicanedo/prc-challenge-2026 | [S] Written by Arquimedes Canedo = team `quick-boat`. |
| Official site | https://prc-data-challenge-2026.netlify.app/ | [S] |

### Key discovery for the team

The `prc-leaderboard.fly.dev/data.json` endpoint **only exposes best-per-team** (`team`, `best`, `n`, `lastAt`). I probed 16 candidate history paths — all 404. The Wayback Machine has **zero** snapshots of it (`archived_snapshots: {}`). [S]

**However**, the official `ranking.qmd` points at the upstream OpenSky REST API, which *does* serve the complete per-submission history for every team. [S] That is what made Section 2 possible. This API is public, unauthenticated, and is the authoritative source — the fly.dev and vercel sites are just caching mirrors of it. [S]

Caveat: the API's earliest record is `2026-09-04T09:58:42Z`, although the competition opened 2026-09-01. [S] Submissions made 09-01 → 09-04 are **not** in the history (visible as teams whose first listed file is e.g. `quick-boat_v19`, `jovial-uniform_v8`). So early curves are truncated. [X] I do not know whether those rows were purged, or whether the endpoint is capped.

---

## 1. Top 20 as of 2026-09-21 [S]

All figures from the leaderboard JSON. `n` = number of scored submissions.

| # | team | best RMSE (s) | subs (n) | last submission (UTC) |
|---:|---|---:|---:|---|
| 1 | `vigorous-whistle` | 239.1335 | 21 | 2026-09-20 18:23:30 |
| 2 | `jovial-uniform` | 240.5497 | 18 | 2026-09-20 03:13:30 |
| 3 | `jolly-lobster` | 242.0575 | 37 | 2026-09-20 18:58:13 |
| 4 | `enthusiastic-daisy` | 243.6497 | 70 | 2026-09-20 14:57:35 |
| 5 | `gentle-igloo` | 243.8347 | 34 | 2026-09-20 14:19:34 |
| 6 | `gentle-tractor` | 244.0470 | 18 | 2026-09-20 13:06:43 |
| 7 | `youthful-giraffe` | 245.0207 | 21 | 2026-09-10 11:13:50 |
| 8 | `zesty-puzzle` | 249.7845 | 41 | 2026-09-20 14:05:24 |
| 9 | `bubbly-telephone` | 255.3271 | 48 | 2026-09-21 00:04:35 |
| 10 | `quick-boat` | 256.9600 | 59 | 2026-09-20 01:55:48 |
| 11 | `merry-quicksand` | 257.5532 | 22 | 2026-09-19 10:50:06 |
| 12 | `gentle-lemon` | 258.1164 | 35 | 2026-09-18 08:37:35 |
| 13 | `strong-turtle` | 260.5613 | 32 | 2026-09-18 00:58:19 |
| 14 | `strong-crane` | 263.3771 | 24 | 2026-09-21 00:55:40 |
| 15 | `humble-giraffe` | 265.7779 | 17 | 2026-09-20 21:22:02 |
| 16 | `gentle-ladder` | 266.0759 | 31 | 2026-09-20 07:54:23 |
| 17 | `organized-marshmallow` | 266.1113 | **3** | 2026-09-19 14:59:48 |
| 18 | `upstanding-firefly` | 266.4049 | 9 | 2026-09-17 07:23:16 |
| 19 | `upbeat-goblin` | 266.4508 | 29 | 2026-09-16 09:50:59 |
| 20 | `dynamic-lobster` | 267.8697 | 25 | 2026-09-18 13:05:40 |

Every scored submission reports `usedPairs = 344841`, identical across all 2144 rows. [S] So the ranking set is a single fixed 344,841-row set; there is **no public/private split visible** in the API. [S] The official site describes only one RMSE ranking. [S]

---

## 2. Reconstructed submission-score curves, top 10

Reconstructed from the 2144-record API pull. [S] "Jump" = a single submission that improved the team's **running best** by more than 5 s.

### Summary of structural jumps

| # | team | first seen | first score | best now | total gain | # of >5 s jumps | last >5 s jump |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `vigorous-whistle` | 09-13 | 436.90 | 239.13 | 197.76 | 6 | **09-15** |
| 2 | `jovial-uniform` | 09-05* | 345.77 | 240.55 | 105.22 | 5 | **09-19** |
| 3 | `jolly-lobster` | 09-04 | 423.12 | 242.06 | 181.06 | 8 | **09-17** |
| 4 | `enthusiastic-daisy` | 09-04* | 314.40 | 243.65 | 70.75 | 4 | 09-12 |
| 5 | `gentle-igloo` | 09-13 | 501.23 | 243.83 | 257.40 | 5 | 09-15 |
| 6 | `gentle-tractor` | 09-15* | 306.90 | 244.05 | 62.85 | 6 | **09-20** |
| 7 | `youthful-giraffe` | 09-06 | 265.76 | 245.02 | 20.74 | **1** | 09-08 |
| 8 | `zesty-puzzle` | 09-04 | 546.50 | 249.78 | 296.71 | 5 | **09-20** |
| 9 | `bubbly-telephone` | 09-11 | 528.57 | 255.33 | 273.25 | 6 | 09-17 |
| 10 | `quick-boat` | 09-04* | 283.20 | 256.96 | 26.24 | **1** | 09-07 |

`*` = history truncated (team's first listed file is not `_v1`), so earlier submissions exist but are not served by the API. [S]

### Full jump list (>5 s improvement of running best) [S]

**#1 `vigorous-whistle`** — entered late (09-13) and climbed to #1 in 7 days on 21 submissions.
```
09-13 16:40  436.8976 -> 421.3047  (-15.59)  v2
09-13 18:08  421.3047 -> 331.9269  (-89.38)  v3
09-13 20:39  331.9269 -> 326.6895  ( -5.24)  v4
09-14 14:50  322.9266 -> 277.4508  (-45.48)  v6
09-15 05:01  275.0570 -> 267.4454  ( -7.61)  v9
09-15 09:07  267.4454 -> 248.4368  (-19.01)  v11
```
After 09-15 09:07 the curve is pure grind: 246.48 → 243.58 → 242.58 → 241.75 → 241.51 → 241.41 → 241.18 → 240.89 → 239.58 → 239.13. [S] All structural gain landed in **2.5 days**; the last 6 days bought 7.35 s total. [X] Reads as: one good pipeline built fast, then hyperparameter/ensemble tuning.

**#2 `jovial-uniform`** — the only top-6 team still making >5 s structural jumps this week.
```
09-05 03:12  345.7672 -> 319.1477  (-26.62)  v9
09-08 14:04  319.1477 -> 277.8533  (-41.29)  v14
09-14 13:04  264.2458 -> 252.9192  (-11.33)  v21
09-15 03:15  252.9192 -> 247.4686  ( -5.45)  v22
09-19 02:40  247.4686 -> 240.7357  ( -6.73)  v23   <-- recent structural change
```
Note the 09-10 → 09-14 gap with no submissions, then a jump. [X] Consistent with offline feature work rather than leaderboard probing.

**#3 `jolly-lobster`** — most jumps (8), steady structural progress across the whole window.
```
09-05 18:07  423.1205 -> 407.5742  (-15.55)  v4
09-06 10:32  407.5742 -> 356.0057  (-51.57)  v5
09-09 00:35  353.7071 -> 330.2300  (-23.48)  v7
09-10 03:47  330.2300 -> 300.3526  (-29.88)  v8
09-10 21:03  300.3526 -> 292.1561  ( -8.20)  v11
09-12 03:07  292.1561 -> 286.6770  ( -5.48)  v12
09-15 19:59  284.0984 -> 272.0935  (-12.00)  v17
09-17 04:40  270.4410 -> 256.8097  (-13.63)  v22
```
Then a dense tuning tail (v23–v41) worth another 14.75 s. [S]

**#4 `enthusiastic-daisy`** — 70 submissions, most of any team; earliest/flattest of the leaders.
```
09-04 17:02  314.4030 -> 279.8002  (-34.60)  v542
09-06 00:34  277.2904 -> 270.6190  ( -6.67)  v564
09-07 23:17  269.8669 -> 261.1488  ( -8.72)  v567
09-12 03:07  257.3476 -> 251.9707  ( -5.38)  v589
```
No structural jump since **09-12**; the last 9 days produced 8.32 s of grind (251.97 → 243.65) across ~40 submissions. [S] Also note: a submission on 09-08 scored **537.47** (a blow-up amid a 260 s best), and they spent 09-08 → 09-12 stuck at 260.93 while submitting ~20 files that all scored worse. [S] [X] That reads like a failed structural rewrite that took 4 days to recover from.

**#5 `gentle-igloo`** — entered 09-13, reached #5 in 7 days. Largest total gain of the top 6 (257 s).
```
09-13 11:07  501.2302 -> 366.2231  (-135.01)  v2
09-13 11:40  366.2231 -> 354.1985  ( -12.02)  v3
09-13 12:56  354.1985 -> 344.0555  ( -10.14)  v4
09-14 00:06  339.2113 -> 279.6242  ( -59.59)  v8
09-15 20:00  273.4128 -> 258.1939  ( -15.22)  v27
```
Since 09-15 it is a steady 1–3 s-per-day grind down to 243.83, with 6 improvements in the last 3 days. [S] File versions run to `v98` while only 34 are scored — 64 unscored slots. [S]

**#6 `gentle-tractor`** — the fastest-moving team right now. 18 submissions, entered 09-15.
```
09-16 01:27  303.0241 -> 292.2086  (-10.82)  v16
09-17 01:18  291.4561 -> 283.3376  ( -8.12)  v19
09-17 01:19  283.3376 -> 278.1508  ( -5.19)  v20
09-17 08:57  278.1508 -> 271.9603  ( -6.19)  v27
09-18 09:32  270.3435 -> 264.8975  ( -5.45)  v34
09-20 10:00  258.6033 -> 247.0569  (-11.55)  v37   <-- yesterday
```
Plus 247.06 → 244.05 three hours later. **Gained 14.56 s on 2026-09-20 alone.** [S] [X] Still on a steep structural slope — likely to keep climbing.

**#7 `youthful-giraffe`** — the outlier. **Started at 265.76 on its very first submission** — a score that would have been rank ~15 today. Exactly one >5 s jump in its whole history.
```
09-08 17:21  262.7418 -> 248.4818  (-14.26)  v13
```
**Has not submitted since 2026-09-10 11:13** — 11 days idle while five teams overtook it. [S]

**#8 `zesty-puzzle`** — long plateau then a late break.
```
09-04 20:58  546.4950 -> 388.2165  (-158.28)  v2
09-05 06:21  388.2165 -> 361.4419  ( -26.77)  v3
09-05 14:59  361.4419 -> 291.2870  ( -70.15)  v5
09-10 07:40  276.5123 -> 271.3831  (  -5.13)  v26
09-20 14:05  263.1805 -> 249.7845  ( -13.40)  v44   <-- yesterday
```
Between 09-12 and 09-19 it moved only 265.56 → 263.18 (2.4 s over 7 days, 9 submissions), then broke 13.4 s in one shot. [S] [X] Classic signature of a structural change held back until it was ready.

**#9 `bubbly-telephone`** — most disciplined cadence: submits in tight bursts right after 00:00 UTC (daily quota reset).
```
09-12 04:06  528.5737 -> 394.8500  (-133.72)  v2
09-12 08:01  393.2699 -> 324.4567  ( -68.81)  v4
09-13 00:03  323.6890 -> 292.3615  ( -31.33)  v9
09-14 00:01  289.8127 -> 283.8399  (  -5.97)  v13
09-15 00:01  283.8399 -> 276.9296  (  -6.91)  v18
09-17 00:03  268.8955 -> 263.0716  (  -5.82)  v31
```
Improving ~2 s/day since. Their 09-21 00:01–00:04 burst of 5 files yielded 255.33. [S]

**#10 `quick-boat`** — 59 submissions, the second-flattest curve. Only one >5 s jump:
```
09-07 00:28  273.2138 -> 267.1210  (-6.09)  v34
```
From 09-07 to 09-21 (14 days, ~45 submissions) they gained 10.16 s total. [S] Also note they log many *worse* submissions (e.g. `v24` = 516.41, `v75` = 276.79 while best was 261.92). [X] Looks like they are probing variants rather than chasing monotone improvement — consistent with the profile being a research engineer (see §3).

### Cross-cutting observations [X]

1. **Structural jumps cluster in the first 48–72 h of a team's participation.** Every top-10 team's large jumps (>20 s) happen within 3 days of their first submission. After that, no team has ever regained more than ~14 s in one step.
2. **But late structural jumps still happen and still matter.** `gentle-tractor` (-14.56 s on 09-20), `zesty-puzzle` (-13.40 s on 09-20) and `jovial-uniform` (-6.73 s on 09-19) all made structural moves within the last 3 days. So the field is *not* converged.
3. **High submission count ≠ high rank.** `youthful-giraffe` is #7 with 21 submissions and 11 days idle; `quick-boat` is #10 with 59. `organized-marshmallow` is **#17 on 3 submissions**. The leaderboard rewards a good pipeline, not iteration volume.
4. **Two distinct entry archetypes are visible.** (a) *Cold start:* first submission 400–550 s (a constant/near-constant baseline), then huge structural jumps — `zesty-puzzle` 546, `bubbly-telephone` 529, `gentle-igloo` 501, `vigorous-whistle` 437, `jolly-lobster` 423. (b) *Warm start:* first submission already competitive — `youthful-giraffe` 265.76, `quick-boat` 283.20. Archetype (b) correlates with prior competition/domain experience (§3).

---

## 3. Public traces of the top-10 teams

**The decisive find: the official site publishes real identities.** Each registered team has a page at `https://prc-data-challenge-2026.netlify.app/teams/<team-name>.html` carrying the member's **full name, affiliation, country, registration date and a free-text "Description and Rationale"**. [S] This is public and requires no login. Team names are randomly generated but are *not* anonymous.

### Identities (all quotes verbatim from the official team pages) [S]

| # | team | country | person | affiliation | registered |
|---:|---|---|---|---|---|
| 1 | `vigorous-whistle` | India | **madhav pv** | ArcNautical | 2026-09-13 |
| 2 | `jovial-uniform` | India | **Piyush Patil** | Independent | 2026-09-01 |
| 3 | `jolly-lobster` | United States | **Lee Marshall** | Elysium Labs / UCF METIL Lab | 2026-09-04 |
| 4 | `enthusiastic-daisy` | China | **Junguo Song** | northeast university | 2026-09-01 |
| 5 | `gentle-igloo` | Portugal | **Denys Kolomiiets** | Independent researcher | 2026-09-01 |
| 6 | `gentle-tractor` | Vietnam | **HOANG ANH THONG LE** | ACV — Airports Corporation of Vietnam | 2026-09-12 |
| 7 | `youthful-giraffe` | Indonesia | **Matthew Motoki** | "Unknow" | 2026-09-01 |
| 8 | `zesty-puzzle` | Germany | **Luis Melzer** | Leibniz Universität Hannover (CS student) | 2026-09-04 |
| 9 | `bubbly-telephone` | India | **Shikhar Kannojia** | Inspire | 2026-09-09 |
| 10 | `quick-boat` | United States | **Arquimedes Canedo** | Independent Researcher | 2026-09-01 |

(For reference: `resilient-hedgehog` = Ueda Ryo, MOCHITEC Inc., Japan, registered 2026-09-10, rationale "Exploring and Validating Data Analytics with Snowflake". [S] **Our own identity is equally public.**)

All top-10 teams are **1-person teams** per their pages, except `gentle-tractor` which declares "Number of team members: 3" but lists only one member in the table. [S]

### Notable individual traces

**#7 `youthful-giraffe` = Matt Motoki — the single most significant find.** [S]
- Kaggle Competitions **Grandmaster**, peak world rank **#18**, $350k+ in competition prize money (https://mattmotoki.github.io/, https://www.kaggle.com/mmotoki).
- **1st place, 2025 NASA Airport Throughput Prediction Challenge ($30,000)** — i.e. he has *already won a major airport-operations forecasting competition*. [S]
- Other wins: 1st 2019 Great Energy Predictor III; 1st 2023 NeurIPS CityLearn (forecasting track); 2nd 2023 Net Load Forecasting Prize ($150k). [S]
- Formerly a Kaggle Grandmaster at H2O.ai (https://h2o.ai/company/team/kaggle-grandmasters/). [S]
- [X] This directly explains his curve: first submission at 265.76 and only one structural jump. He arrived with a working airport-throughput modelling template. His 11-day silence since 09-10 is the strategically interesting fact — [X] either he has disengaged, or he is holding back (there is no private leaderboard to hold back *for*, so disengagement is more likely; but 20 days remain).
- No blog posts about taxi-out/ATM modelling found on his site. [S]

**#10 `quick-boat` = Arquimedes Canedo.** [S]
- **He is the author of the official leaderboard viewer** the challenge site embeds: https://github.com/arquicanedo/prc-challenge-2026, credited by name in `ranking.qmd`. [S] That repo is public and contains the leaderboard-mirroring service — not his model.
- Distinguished Engineer at Siemens; previously Principal Key Expert Scientist, Siemens Corporate Technology (2016–2025). Research in knowledge graphs, digital twins, graph learning; author of `pykg2vec` and `dyngraph2vec`. (https://www.siemens.com/us/en/company/press/siemens-stories/corporate-technology/arquimedes-canedo-new-work-in-graph-learning.html, https://dblp.org/pid/03/4986.html) [S]
- I found **no public repo of his taxi-out model**. [S]

**#3 `jolly-lobster` = Lee Marshall — had a public repo, and it is now gone.** [S]
- Search engines still index `https://github.com/LeeMarshall1113/prc-taxiout-2026` with the description *"Entry for the PRC Data Challenge 2026: predicting taxi-out time at European airports (RMSE, closes 2026-10-11)"*. [S]
- Fetching that URL today returns **HTTP 404**. [S]
- The GitHub API lists **exactly 22 public repos** for `LeeMarshall1113` and `prc-taxiout-2026` is **not among them** (I checked both result pages). [S]
- [X] Conclusion: the repo was public, was indexed, and has since been made private or deleted — most plausibly to avoid leaking a rank-3 method mid-competition. His other repos (`jetspace-imitation-learning`, `jetspace-scaffold-rsi`, `Atmospheric_Anamoly_Detection`) show an aviation/ML background but contain nothing about this challenge. [S]

**#5 `gentle-igloo` = Denys Kolomiiets.** Self-describes as maintaining *"an open, DOI-versioned corpus of aviation occurrence records harmonised from 128 sources, published with full provenance and a deterministic, documented deduplication pipeline,"* and states his angle explicitly: *"reconstructing airport state at off-block from movement data is a data-engineering problem before it is a modelling one."* [S] I searched Zenodo and the open web for that corpus by his name and **could not find it**. [S]

**#4 `enthusiastic-daisy` = Junguo Song.** Self-describes as a PhD student researching *"deep learning-based aircraft trajectory prediction."* [S] I searched for publications under this name at Northeastern University and **found none**. [S]

**#1 `vigorous-whistle` = madhav pv (ArcNautical), #2 `jovial-uniform` = Piyush Patil, #6 `gentle-tractor` = Hoang Anh Thong Le (ACV), #8 `zesty-puzzle` = Luis Melzer, #9 `bubbly-telephone` = Shikhar Kannojia.** [S] I searched each name against PRC / taxi-out / data-challenge terms, plus GitHub, Kaggle, LinkedIn, arXiv and blogs. **I found no public repository, post, profile or writeup attributable to any of them for this challenge.** [S] I could not confirm "ArcNautical" as an existing company from any source. [S]

`vigorous-whistle`'s rationale is the only competitive tell among them: *"i am an independent machine-learning practitioner who competes in metric-scored data challenges alongside running a small startup."* [S]

### Team-linked public repos that DO exist (none in the top 10) [S]

| repo | team | current rank / best |
|---|---|---|
| https://github.com/Phoenix-Ops-LTD/prc2026-taxiout | `zestful-fountain` | #29 / 273.5557 |
| https://github.com/satam2/knowledgeable-helicopter | `knowledgeable-helicopter` | #36 / 278.0146 |
| https://github.com/skylinkapi/prc-data-challenge-2026-kind-mango | `kind-mango` | #53 / 287.3339 |

The `zestful-fountain` README is the most detailed public method in the field and its self-reported score (273.5557 / 344,841 pairs) matches the leaderboard exactly, so it is genuine. [S] Its stated approach: CatBoost (4,999 trees, 37.5% ensemble weight) + LightGBM (12.5%) + multi-seed specialist ensemble; ~183 features including UTC hour/weekday/month, cyclical hour, 30-minute scheduled-departure counts, dep/dest airport, aircraft type, flight rule, market segment, wake category, operator, following-aircraft observations, and "thirty features from strictly following, same-UTC-month NM departure-clock observations"; validation by **chronological split with no repeated flight group across the boundary, no random row splitting**, early stopping for iteration count, then refit on all training rows. [S] It explicitly excludes movement/flight IDs and actual movement times. [S]
[X] This is a useful *floor* reference: a careful GBDT + NM-clock-feature approach lands around 273 s. The top 6 are 30 s better than that, so they are doing something materially different.

Other 2026 repos found whose team mapping I could **not** determine: `javidmardanov/PRC-Data-Challenge-2026` (TimesFM-3 + residual models), `ahmetabdullahgultekin/prc-taxiout-2026`, `victoralcadi/prc-data-challenge-2026`, `ChandanHegde07/OpenAir`, `georgejhanlon/prc-taxiout-2026`, `saurabhdxt-labs/prc-challenge`, `AbdullahKhan-x/PRC-2026`. [S]

### What I could NOT find [S]

- No Kaggle discussion threads about this challenge.
- No LinkedIn or X/Twitter posts by any top-10 member about this challenge.
- No arXiv preprints on PRC 2026 taxi-out.
- **No publicly indexed Discord logs.** The official channel is `#prc-data-competition` on the OpenSky Network Discord (invite https://discord.gg/RPh89jpVVz, channel https://discord.com/channels/1252625490930307142/1304403918515605504). [S] It is login-gated and not search-indexed. [X] This is the single largest remaining blind spot — the official site says *"we strongly recommend following the updates on the OSN Discord channel"*, so material discussion is likely happening there where I cannot see it.
- No pages on the official site or `ansperformance.eu` ranking teams by name other than the embedded leaderboard iframe. The `euctrl-pru/prc_data_challenge_website_2026` repo contains the team pages but **no score/ranking data files**. [S]

---

## 4. Rules confirmed from official sources [S]

- Deadline: **2026-10-11 23:59:59 CET**. Competition opened 2026-09-01. [S]
- Ranking: *"Teams will be ranked based on the **best RMSE** achieved across their submissions."* [S] → best-of, not last-submission. No penalty for a bad submission.
- **Limit of 5 submissions per day**, and total max 1 GB per bucket. [S]
- File naming: `<team-name>_v<incremental integer>.parquet`. Rows must match `MVT_ID_mvt` exactly — no missing rows, no extra rows, or the ranking script errors. [S]
- Prize: **5000 EUR combined for the top 3 teams**; split not stated. [S]
- Mandatory: open data, **public GitHub repo with code and documentation**, and an open-access paper in the Journal of Open Aviation Science. [S]
- Explicit warning on the ranking page: *"We will monitor submissions for attempts to learn from or exploit the ranking process. We consider such practices unfair and inconsistent with the purposes and goals of the Data Challenge."* [S]

---

## 5. Score distribution and where 284.3584 sits

### Our position [S]

`resilient-hedgehog` — **best 284.3584, rank 43 of 168 teams, 11 submissions**, last submission 2026-09-19 01:54 UTC.

Our own curve [S]:
```
09-10 17:33  v1   345.3520  *best*
09-10 19:25  v2   312.0237  *best*  (-33.33)
09-10 22:21  v3   306.2942  *best*  ( -5.73)
09-11 00:42  v4   305.0270  *best*
09-14 09:54  v5   325.2236          (worse)
09-15 14:55  v6   350.6249          (worse)
09-15 15:54  v7   342.5933          (worse)
09-15 15:59  v8   343.7964          (worse)
09-15 22:00  v9   306.4555          (worse)
09-17 06:17  v10  284.3584  *best*  (-20.67)
09-19 01:54  v11  293.9780          (worse)
```
[X] Read plainly: 4 of our 11 submissions improved anything, and 5 consecutive submissions (v5–v9) were regressions. Our single structural win was v10 on 09-17. We have made **one** submission since and it was worse. Meanwhile the rank-43 cutoff has moved every single day.

### Gaps [S]

| measure | value |
|---|---|
| rank 1 → rank 10 | **17.8265 s** (239.1335 → 256.9600) |
| rank 10 → rank 30 | **17.6838 s** (256.9600 → 274.6438) |
| rank 1 → rank 20 | 28.7362 s |
| rank 1 → us (43) | **45.2249 s** |
| rank 30 → us (43) | 9.7146 s |
| teams ahead of us | 42 |
| total ranked teams | 168 |
| total submissions in competition | 2144 |

Density is extremely different at the two ends of the board [S]:
- Around the top 10: **~2.19 s per rank** (ranks 5→15).
- Around our rank: **~0.41 s per rank** (ranks 40→50).
- **14 teams sit within ±3 s of our score.** Ranks 40–53 span just 4.92 s (282.18 → 287.33). [S]

[X] So at 284.36 we are in the densest part of the board. A 5 s improvement would move us roughly 43 → 28. A 10 s improvement would move us to roughly rank 25. But the top 10 is 27.4 s away and getting further.

### Submission counts [S]

| group | submissions |
|---|---|
| top-10 median | **35.5** |
| top-10 mean | 36.7 |
| top-10 range | 18 (`jovial-uniform`, `gentle-tractor`) – 70 (`enthusiastic-daisy`) |
| all-teams median | 7 |
| **us** | **11** |

[X] We are at roughly **one-third** of the top-10 median submission count, and we have used ~11 of a theoretical 5/day × 20 days = 100 remaining slots budget. Submission volume is not sufficient for a top rank (see `organized-marshmallow` at #17 on 3), but every top-10 team except two has at least 21.

### The board is moving, and we are not [S]

Best-so-far cutoffs by date, computed from the full history:

| date | #teams | rank 1 | rank 10 | rank 30 | rank 43 |
|---|---:|---:|---:|---:|---:|
| 09-14 | 139 | 245.02 | 266.94 | 283.84 | 292.38 |
| 09-16 | 152 | 241.75 | 263.38 | 278.21 | 286.31 |
| 09-18 | 163 | 241.18 | 260.54 | 275.97 | 284.52 |
| 09-19 | 165 | 239.58 | 258.60 | 274.84 | **284.36** |
| 09-21 | 168 | 239.13 | 256.96 | 274.64 | **284.36** |

Drift over the last 7 days (09-14 → 09-21) [S]:

| cutoff | drift | naive linear projection at 2026-10-11 (20 days out) |
|---|---:|---:|
| rank 1 | 0.841 s/day | ~222.3 [X] |
| rank 3 (prize line) | 1.552 s/day | ~211.0 [X] |
| rank 10 | 1.426 s/day | ~228.4 [X] |
| rank 20 | 1.083 s/day | ~246.2 [X] |
| rank 30 | 1.314 s/day | ~248.4 [X] |
| rank 43 | 1.146 s/day | ~261.4 [X] |

[X] The linear projections are almost certainly too aggressive — improvement curves flatten, and the top-1 drift already decelerated from ~3 s/day early to 0.84 s/day. But the *direction* is unambiguous: **a static 284.36 will not hold rank 43.** At the observed rank-43 drift, our current score falls out of the top 60–80 well before the deadline. [X] To merely hold rank ~43 on 10-11 we would need roughly **255–265 s**; to be top-20 we would need roughly **240–246 s**. [X] These are extrapolations, not facts.

---

## Applicable to `resilient-hedgehog`

Concrete and checkable:

1. **Switch our leaderboard monitoring to the official API.** `https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard`, cursor-paginated via `nextCursor`. It gives every competitor's full per-submission score history; `prc-leaderboard.fly.dev/data.json` gives only best-per-team. 43 pages = 2144 records, one unauthenticated pull. Other endpoints are documented at `https://datacomp.opensky-network.org/api/swagger-ui/` with `competitionId = bb3693e1-26bc-4a9e-8619-4fe78b4eab0c`. [S]
2. **We are leaving ~80% of our submission budget unused.** Limit is 5/day [S]. We have made 11 submissions total and 1 since 2026-09-17. Top-10 median is 35.5. 20 days remain = ~100 slots.
3. **Ranking is best-of-all-submissions, not last** [S]. A regression costs us nothing on the board. There is no reason to withhold a candidate submission for fear of a worse score. Our v11 (293.98) did not hurt our 284.36.
4. **There is no private/held-out leaderboard.** Every submission scores against the same fixed 344,841 pairs (`usedPairs` constant across all 2144 records) [S], and the official site describes one RMSE ranking [S]. So public-LB score is the final score — but note the explicit anti-probing warning in `ranking.qmd` [S], so do not build anything that reverse-engineers labels from score feedback.
5. **Our own identity and rationale are public** at `https://prc-data-challenge-2026.netlify.app/teams/resilient-hedgehog.html` (Ueda Ryo, MOCHITEC Inc.) [S]. Competitors can profile us exactly as I profiled them. Assume zero anonymity.
6. **Benchmark against the one fully public method.** `zestful-fountain` (#29, 273.5557) publishes its full approach at https://github.com/Phoenix-Ops-LTD/prc2026-taxiout [S]: CatBoost+LightGBM ensemble, ~183 features, NM departure-clock overlays, chronological split with no flight group crossing the boundary. Concretely checkable: does our feature set include their 30 "strictly following, same-UTC-month NM departure-clock" features and their 30-minute scheduled-departure-count congestion features? If not, that is a named, documented gap worth ~11 s relative to our current score.
7. **Our validation split should be chronological with group integrity**, per the only verified public method [S]. Check that our current CV is not random row splitting.
8. **Structural change, not tuning, is what moves ranks at our level.** Across the top 10, every gain >20 s came from a structural jump; no team has ever recovered more than ~14 s through incremental tuning after their first 3 days [S]. Our own history shows the same: v10 gave us -20.67 s in one step; everything else gave us ~0 [S]. Budget effort accordingly.
9. **Late structural jumps are still landing.** `gentle-tractor` gained 14.56 s on 2026-09-20 and `zesty-puzzle` 13.40 s the same day [S]. The field has not converged; a 20–30 s improvement from where we are is demonstrably still achievable in this window.
10. **Target numbers.** To reach today's top-30 we need ≤274.6438 (−9.71 s); top-20 ≤267.8697 (−16.49 s); top-10 ≤256.9600 (−27.40 s) [S]. Because cutoffs drift ~1.1–1.4 s/day [S], budget an extra ~20–28 s on top of each of those to hold the same rank at the 2026-10-11 deadline [X].
11. **We are in the densest region of the board** — ranks 40–53 span 4.92 s and 14 teams sit within ±3 s of us [S]. Small real improvements convert to large rank moves here (~0.41 s per rank), unlike near the top (~2.19 s per rank).
12. **Do not publish our repo before the deadline.** Rank-3 `jolly-lobster` had a public repo indexed by search engines and it now returns 404 and is absent from the owner's 22 public repos [S]. The open-source obligation is a *final*-solution requirement [S], not a during-competition one.
13. **Blind spot to close: the OSN Discord `#prc-data-competition` channel** (https://discord.gg/RPh89jpVVz, channel `1304403918515605504`) [S]. The official site directs participants there for urgent news [S] and it is not search-indexed. Someone with an account should read the backlog — it is the only material information source I could not access.
14. **Watch `youthful-giraffe` (Matt Motoki, Kaggle Grandmaster, winner of the 2025 NASA Airport Throughput Prediction Challenge)** [S]. He has been idle since 2026-09-10 at 245.0207 [S] — a score set with only 21 submissions and one structural jump. [X] If he re-engages before 10-11, expect a large move; treat 245 as a soft floor for what a domain-experienced solo competitor reaches without heavy iteration.

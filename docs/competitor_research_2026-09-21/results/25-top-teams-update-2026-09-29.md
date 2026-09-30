# 25 — Top-teams update, 2026-09-29

This note covers what the leaders do that we do not, updated eight days after slots 01–24. It uses
web sources and public code only. Nothing was fitted. `src/`, `configs/`, `outputs/` and `data/` were
not modified. I did read `src/prc/features.py` with grep, to answer one question in §5.

Marking: **[S]** means the claim is read from the cited source. **[X]** means it is my inference. A
source that could not be read is named as unreachable; I did not fill the gap with a guess.

---

## 0. Sources and access

| source | status | how I read it |
|---|---|---|
| Official per-submission API `datacomp.opensky-network.org/api/competitions/bb3693e1-…/leaderboard` | **read** | 68 pages, **3,363 submissions, 199 teams**, last `processedAt` 2026-09-29T00:30:37Z. `usedPairs` is 344,841 on every row. [S] |
| Mirror `prc-leaderboard.fly.dev/data.json` | read | `updated` 2026-09-29T00:24:13Z, 165 teams. It reports `totalSubmissions` 3000 against 3,363 in the API. The mirror looks capped, so all figures below come from the API. [S] |
| Site source `euctrl-pru/prc_data_challenge_website_2026` | **read** (blobless `git clone`) | every commit since 2026-09-15 [S] |
| Team pages (`teams/<name>.qmd` in the site repo) | read for the top 30 | [S] |
| GitHub search (MCP repo and code search) | read | 17 repos match "prc data challenge 2026". Name searches ran for all top-30 team names. [S] |
| Public competitor repos | cloned blobless | kind-mango, merry-quicksand, satam2, RavindraTarunokusumo, radekacar, plus the known ones |
| GitHub REST API (`api.github.com`) and `github.com` web | **blocked for this session** (repo-scoped proxy, HTTP 403) | I used `git` over the proxy and `raw.githubusercontent.com` instead |
| OpenSky Discord `#prc-data-competition` | **unreachable** (login-gated) | Discord content below is **second-hand**, relayed from kind-mango's public `RECAP.md` and `docs/MODEL_ANALYSIS.md`. I did not re-verify it. |

---

## 1. Headline

1. **The board moved about 15 s at the top and about 20 s at rank 20 in one week.** Leader
   **strong-crane 220.6546**. Rank 3 is 224.50, rank 10 is 236.98, rank 20 is 247.54 and rank 30 is
   260.66. **We are rank 56 of 199 at 279.2590** (our v16, 2026-09-28 18:48, which is 0.26 s better
   than v14). **The gap to rank 1 is 58.6 s.** [S] API
2. **A wave ran from 2026-09-23 to 09-28.** Teams already below 300 s made **14 single uploads worth
   10 s or more in three days** (09-26 to 09-28). The usual rate is 0 to 3 a day. The leaders
   (vigorous-whistle, jolly-lobster, zesty-puzzle, gentle-igloo) did **not** jump in that wave. Their
   largest single step since 09-21 is 3.5 to 5.2 s. [S] API. §2.3
3. **No team in the top 23 has public code.** The best public code on the board belongs to
   merry-quicksand (#24, 250.73) and kind-mango (#31, 264.34). [S] §3
4. **kind-mango dropped its exclusion of `AOBT_3_flt` on 2026-09-26.** The switch took it from 281.87
   to 264.34 (-17.5 s). **Every public repo above rank 100 now reads `AOBT_3_flt`.** Not using it is
   no longer a live option anyone has shown to work. [S] §3.1
5. **Our own research says not to use one lever that has since been cleared.** In merry-quicksand's
   public code (09-11), external ADS-B data (adsb.lol) moved **161,036 matched rows at 9 airports**
   and scored **-7,722 board MSE** (281.52 to 267.45). Slot 21 §6.4 said "do not copy" because the
   organiser had not ruled on it. On **2026-09-18** the organiser ruled on Discord that open
   trajectory data and ADSB.lol are acceptable. The ruling is relayed second-hand by kind-mango (see
   §0). We use no ADS-B. [S] §4.2, §6

---

## 2. The board now, with trajectories

### 2.1 Top 20 (API, read 2026-09-29 00:30Z)

"Gain since 09-21" is measured from the team's best score before 2026-09-21 03:00Z.

| # | team | best | n | best 09-21 | best end 09-24 | gain since 09-21 | largest single step since 09-21 | share of gain in that step | improving uploads since 09-21 |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | strong-crane | 220.65 | 65 | 263.38 | 259.38 | **42.72** | 17.21 (09-26) | 40 % | 12 |
| 2 | vigorous-whistle | 224.46 | 38 | 239.13 | 234.14 | 14.67 | 3.51 (09-23) | 24 % | 14 |
| 3 | jolly-lobster | 224.50 | 67 | 242.06 | 234.51 | 17.56 | 5.11 (09-25) | 29 % | 26 |
| 4 | quick-boat | 226.81 | 73 | 256.96 | 233.18 | 30.15 | 16.07 (09-24) | 53 % | 10 |
| 5 | zesty-puzzle | 228.21 | 81 | 249.78 | 237.91 | 21.58 | 5.20 (09-21) | 24 % | 31 |
| 6 | upstanding-firefly | 234.99 | 17 | 266.40 | 243.93 | 31.42 | 21.53 (09-23) | 69 % | 6 |
| 7 | gentle-igloo | 235.17 | 72 | 243.83 | 240.14 | 8.67 | 3.60 (09-26) | 41 % | 17 |
| 8 | gentle-tractor | 235.92 | 30 | 244.05 | 236.31 | 8.13 | 3.63 (09-21) | 45 % | 10 |
| 9 | strong-turtle | 236.35 | 52 | 260.56 | 257.43 | 24.21 | 10.18 (09-28) | 42 % | 14 |
| 10 | jovial-uniform | 236.98 | 20 | 240.74 | 240.74 | 3.75 | 2.16 (09-25) | 57 % | 2 |
| 11 | humble-giraffe | 237.05 | 29 | 265.78 | 263.87 | 28.73 | 13.11 (09-27) | 46 % | 9 |
| 12 | unbelievable-snowflake | 240.55 | 36 | 306.27 | 304.15 | **65.72** | 18.33 (09-26) | 28 % | 19 |
| 13 | versatile-watermelon | 241.34 | 11 | 268.91 | 268.91 | 27.57 | 20.81 (09-28) | 75 % | 4 |
| 14 | enthusiastic-daisy | 242.27 | 107 | 243.65 | 243.29 | 1.38 | 0.41 | 30 % | 7 |
| 15 | youthful-giraffe | 242.75 | 31 | 245.02 | 245.02 | 2.27 | 1.32 (09-26) | 58 % | 3 |
| 16 | gentle-lemon | 243.43 | 46 | 258.12 | 254.74 | 14.69 | 5.21 (09-25) | 35 % | 10 |
| 17 | understanding-accordion | 244.22 | 25 | new 09-21 | 276.83 | — | 23.53 (09-26), from 274.65 | — | 23 |
| 18 | gracious-tornado | 244.92 | 20 | new 09-25 | — | — | 22.87 (09-26), from 289.20 | — | 16 |
| 19 | loyal-anchor | 246.82 | 26 | new 09-23 | 280.33 | — | 14.37 (09-28), from 267.20 | — | 23 |
| 20 | bubbly-telephone | 247.54 | 72 | 255.33 | 248.39 | 7.78 | 1.43 | 18 % | 16 |

**Our position** [S]: rank **56 of 199**, 279.2590, 16 uploads. Since 09-21 we uploaded v12 284.05,
v13 282.74, v14 279.52, v15 279.54 and v16 279.26. Thirteen teams sit within ±3 s of us.

**Cutoffs by date** [S], from the best per team at the end of each UTC day:

| day | teams | #1 | #3 | #10 | #20 | #30 | #50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 09-21 | 171 | 239.13 | 239.55 | 255.33 | 267.87 | 274.64 | 285.68 |
| 09-24 | 189 | 233.18 | 234.51 | 245.02 | 265.28 | 270.57 | 281.87 |
| 09-26 | 195 | 224.88 | 231.73 | 242.81 | 257.57 | 268.52 | 278.39 |
| 09-28 | 199 | 221.60 | 224.50 | 237.05 | 247.98 | 260.66 | 276.99 |

Rank 20 fell **17.3 s in four days** (09-24 to 09-28), while rank 10 fell 8.0 s over the same days.
[S] The brief's top five for 09-24 was jolly-lobster 234.51, vigorous-whistle 235.31, …. That
matches an intraday read. By the end of the UTC day quick-boat led at 233.18. [S]

### 2.2 Which teams won with one step and which with many

- **Many small steps, no structural jump since 09-21:** vigorous-whistle, jolly-lobster, zesty-puzzle,
  gentle-igloo, gentle-tractor, jovial-uniform, enthusiastic-daisy, youthful-giraffe and
  bubbly-telephone. zesty-puzzle is the extreme case: 31 improving uploads since 09-21, none over
  5.2 s, and it submits on a fixed 15-minute grid after 00:00 UTC. [S] [X] These teams were at
  234–244 before the wave and read as teams already holding their structural edge, now tuning.
- **One or two large steps, 09-23 to 09-28:** strong-crane, quick-boat, upstanding-firefly,
  strong-turtle, humble-giraffe, unbelievable-snowflake, versatile-watermelon,
  understanding-accordion, gracious-tornado, loyal-anchor, charming-eel (#22), noble-tractor (#23),
  outgoing-boat (#26), smart-jigsaw (#37) and kind-mango (#31). [S]

Full post-09-22 sequences for the jumpers (API) [S]:

```
strong-crane        09-25 00:00  v61 253.15 (-6.22)   09-26 00:01 v77 235.94 (-17.21)
                    09-26 07:35  v78 260.31           09-26 10:31 v79 233.71
                    09-26 11:24  v80 261.19, v81 260.73
                    09-27 v82 231.34, v84 227.85, v85 225.67
                    09-28 v88 243.53, v89 222.47, v90 263.75, v92 221.60   09-29 v95 233.04, v96 220.65
quick-boat          09-24 21:56  v102 238.45 (-16.07) then v103 234.72, v104 233.18 in 72 min
upstanding-firefly  09-23 07:39  v13 244.87 (-21.53; previous upload 09-17, 268.08)   09-26 v21 234.99 (-5.56)
humble-giraffe      09-27 09:46  v26 250.98 (-12.89)   09-27 14:29 v27 237.87 (-13.11)
unbelievable-snow.  09-26 01:18  v22 281.14 (-16.62)   09-26 06:18 v25 259.87 (-18.33)
versatile-waterm.   09-28 13:41  v8 248.10 (-20.81, first upload in 13 days)
strong-turtle       09-27 12:04  253.05 (-4.27), 18:36 250.59, 09-28 02:18 240.41 (-10.18)
```

**strong-crane alternates between two score levels** [S]. After reaching 235.94 on 09-26 it uploaded
files scoring 260.31, 261.19, 260.73 and 263.75, which is its pre-09-25 level, in between files that
kept improving. [X] One component appears to be switched on in some files and off in others, and it
is worth about 25–40 s when on. Two things fit this pattern equally well: parallel ablations of that
component, or an old pipeline run alongside a new one. The scores cannot tell them apart.

### 2.3 Did several teams jump at the same time? Yes

Count of single uploads that improved a team's running best by 10 s or more, where the previous best
was already under 300 s [S]:

```
09-13 2   09-14 3   09-15 3   09-16 3   09-17 2   09-18 0   09-19 1   09-20 2   09-21 0
09-22 2   09-23 2   09-24 3   09-25 1   09-26 6   09-27 5   09-28 3
```

- The cluster is **09-26 to 09-28: 14 jumps**. Before that the rate never exceeded 3 a day. [S]
- **In MSE terms** these jumps are 5,000 to 13,000 board MSE each. Examples: upstanding-firefly
  11,010; versatile-watermelon 10,758; understanding-accordion 12,370; strong-crane 8,417;
  humble-giraffe 6,638 + 6,411; unbelievable-snowflake 9,623 + 9,861. [S] (arithmetic on API scores)
- For scale, one row missed by 86,400 s is 21,648 MSE. The whole gap from kind-mango (264.34) to
  strong-crane (220.65) is 21,185 MSE. Our gap to the leader is 29,297 MSE. [S] (arithmetic)
- **I cannot tell from the scores whether a jump is a small set of extreme rows or a broad
  improvement.** kind-mango's jump on 09-26 (6,228 MSE) is a known broad change: adding
  `AOBT_3_flt` features (§3.1). It falls in the same size range as the others. [S] [X] I do not claim
  a row-level fingerprint.
- **Candidate triggers, dated** [S] where sourced, [X] for the link to the jumps:
  - **2026-09-17/18, Discord:** ADSB.lol and AIU daily files are accepted as open data if
    attributed. On off-block times from open trajectory data the organiser said: *"there are no such
    restrictions … we are ok"*. Relayed by kind-mango, `RECAP.md` "Discord confirmations".
  - **2026-09-17/18, Discord:** a participant reported corrupted training rows, and another said the
    ranking set "likely has the same corrupt instances". Same relay.
  - **2026-09-23, Discord:** a post suggested *"force the outliers of the submission to look like the
    training outliers"*. Relayed by kind-mango (`RECAP.md` tail-rules row;
    `docs/MODEL_ANALYSIS.md` "A Discord post (2026-09-23)…"). kind-mango priced two readings of it
    on its 2025 hold-out and rejected both: +1,639 MSE and -3 MSE.
  - **2026-09-26 to 09-28, public GitHub:** kind-mango published its switch to `AOBT_3_flt`. That
    push landed on 09-28 20:08Z, after most of the wave, so it cannot be the trigger. [S]
  - **The official site has published no rule change or announcement since 2026-09-18.** The only
    commit that touched a non-team page was `a666240` on 09-18: *"fix 10 airports not 11"* in
    `data.qmd` and `index.qmd`. Every other commit since 09-15 is "regenerate teams pages". [S]
- [X] Reading: the wave starts 5 days after the 09-18 ADS-B and open-trajectory ruling, and 0 to 3
  days after the 09-23 tail post. It lifted second-tier teams (253–300) to 236–250 while the leaders
  kept grinding, which is what you would see if the leaders already had something and the rest
  learned it. I cannot see Discord, so I cannot name what it was. **The Discord backlog for 09-18 to
  09-28 is the single most valuable unread source**, as slot 01 §3 already said, and more so now.

---

## 3. Public code from top-30 teams

**Coverage** [S]:
- **Ranks 1–23 have no public repo.** I searched GitHub for every top-30 team name, in five-name OR
  batches, found nothing relevant, and filtered the results for `prc|taxi|eurocontrol|opensky`. I
  also checked the known member accounts:
  - `Somnathab3` (strong-crane: Somnath Panigrahi, TU Dresden) has 9 public repos. Two are older PRC
    editions (`PRC_TOW_Predictor`, `PRC_2025_Likable-ant`). None is from 2026.
  - `arquicanedo` (quick-boat) has only the leaderboard viewer.
  - `LeeMarshall1113` (jolly-lobster) has 9 public repos, none of them PRC. `prc-taxiout-2026`
    still returns an authentication prompt to `git ls-remote`, so it is private or deleted.
- **The team pages hold no repo links.** The only method hints on them are self-descriptions
  (§3.3).

### 3.1 kind-mango (#31, 264.34): `skylinkapi/prc-data-challenge-2026-kind-mango`, **new since 09-21**

Head `2a4ec1b` (2026-09-28 22:08 +0200), *"Add NM clocks track B: v73 to v89, best 264.34 s"*.

**What they say** [S] (`README.md`, "Stance on `AOBT_3_flt` and `LOBT_flt` (changed 2026-09-26)"):
from v73 on, the base model reads the scored flight's own `AOBT_3_flt` and `LOBT_flt` (NM clocks).
They cite the data page, which blanks only `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`, and the
Discord rulings of 09-08 ("any provided value is fair to use") and 09-18 ("no such restrictions").
They add that the warning quoted in their older README *"came from the team's own brief … not from an
organiser page."*

**What the code and ledger show** [S] (`RECAP.md` rows v73–v89 and `models/*.meta.json`), live score
per step:

| ver | live | Δ | change |
|---|---:|---:|---|
| v67 | 281.87 | — | Best **without** NM clocks: 3 × LightGBM (`linear_tree`) + CatBoost 50/50, 118 columns |
| v73 | 270.58 | **-11.29** | Five clock columns in the plain-`y` LightGBM: `mvt_aobt3`, `aobt3_eobt1`, `aobt3_sched`, `mvt_lobt`, `lobt_aobt3`, clipped ±7,200 |
| v74 | 266.84 | -3.73 | Clocks added to the CatBoost member, plus a LightGBM member on `y − clip(MVT − AOBT_3, 0, 7200)` (the "anchored" member), blended 0.5 |
| v75 | 266.15 | -0.70 | Anchored CatBoost member |
| v76 | 265.27 | -0.88 | Optuna study on the clock-era columns: `linear_lambda` 15.1, 613 leaves, lr 0.015, `min_data_in_leaf` 515, 3,377 rounds |
| v85 | 264.55 | -0.72 | **Forward windows**: mean and count of other departures' `MVT − AOBT_3` taking off in `(t, t+15]` and `(t, t+30]`, per airport and per runway |
| v89 | 264.34 | -0.22 | One LightGBM per airport, mixed 50/50 with the global model |

Served base outside LIRF [S]:

```
0.5 × (0.5 plain LGBM + 0.5 anchored LGBM) + 0.5 × (0.5 plain CatBoost + 0.5 anchored CatBoost)
```

LIRF keeps its own regime head: a gate `p_fb` blending `sd` with `R_norm`, plus a "Step A" band table
on null-record rows with `sd > 14,400`.

Their measurements that bear on us [S]:
- On 2025 departures, **`MVT − AOBT_3` against the label has a median absolute error of 175 s; 21 %
  of rows lie within 60 s; the raw clock scores RMSE 376 s.** On the 584 rows labelled above
  7,200 s the clock's median is 1,082 s: *"it does not see long holds"*.
- At LIRF, `MVT − AOBT_3` has a median of 898 s on fallback rows and 900 s on genuine rows:
  *"NM serves a fixed taxi time there."*
- Hold-out error budget before the clocks, non-LIRF: clean rows 41,368 MSE (RMSE 220 s); one 24-h
  row 20,235; 35 tail rows 12,100; *"45 % of the error sits in 36 rows"*.
- They measured and closed: neighbour backward clocks (+733 MSE at LFPG from linear-leaf blow-ups),
  clock-bias maps, `ARVT_3`, XGBoost, CatBoost variants and blend refits.

**Unmatched rows:** a LIRF-specific head, as above. Outside LIRF the plain members are used and the
anchored members fall back where `AOBT_3` is null. [S]

**What this tells us** [X]: kind-mango's no-clock stack (281.87) scored about the same as our
clock-based stack (279.26). With the clocks bolted on in three days, they are **15 s ahead of us**.
Their public ledger locates the difference:
- they blend a plain-target model that reads the clock as a feature with the anchored model. We fit
  only the anchored target `g = AOBT_3 − BLOCK`.
- LightGBM with `linear_tree` and heavy `linear_lambda`.
- CatBoost as a 50/50 blend member, not as a replacement. It scored -2.87 s live for them.
- forward NM-clock windows.

Caveat: merry-quicksand's controlled head-to-head on one fold (slot 21 §3.4) found that the pure
y-target lost pooled (260.65 against 224.26 on the delta target) but won on rows where
`|delta| > 20 min`. The two sources disagree on the pooled blend, so this is a measured public lever,
not a certainty.

### 3.2 merry-quicksand (#24, 250.73): `saurabhdxt-labs/prc-challenge`, **stale since 2026-09-12**

- **The last commit, `fdeb704`, is dated 2026-09-12.** The team has scored 265.47 → 250.73 since
  then, and none of that is public. [S]
- **Their code reads `AOBT_3_flt`.** Target `delta = BLOCK − AOBT_3`; we copied this
  reparameterisation (slot 21). [S]
- **Their largest single live step came from ADS-B**, as shown in `reports/MSE_LEDGER.md` lines
  659–715 and commit `f51e303` (09-11) [S]:
  - An "ADN" stacker on 26 label-free features from **adsb.lol `globe_history`** (ODbL 1.0), fetched
    by `scripts/adsb_extract.py`.
  - It predicts a residual on the matched lane's delta, gated per airport, with a 2025
    reproduction guard.
  - It changed **161,036 matched rows at 9 airports**: EGLL 30,534, EHAM 27,044, LEMD 23,791,
    LEBL 22,358, EDDM 20,785, LSZH 16,960, EDDF 14,963, LFPG 4,596, LTFM 5. LIRF was excluded.
  - **Board: 281.5182 → 267.4518, -7,722 MSE, 1.36× their fold projection.**
- Their earlier 1-day census (`reports/ADSB_GATE.md`, 09-08) had closed ADS-B as NO-GO: 2 of 10
  airports passed, zero on-ground samples at LEMD, LFPG, LIRF and LTFM. The full build contradicted
  it. Rows moved at LEMD and LFPG, and the pooled live gain beat the projection. [S]
- **On whether ADS-B is allowed:** they emailed the organiser on 09-11 and got no ruling, only a
  redirect to Discord (ledger line 663). The upload was released the same day. [S]

### 3.3 Everything else public, with `AOBT_3_flt` use

| repo | team (rank, best, read 09-29) | `AOBT_3_flt` in code | change since 09-21 |
|---|---|---|---|
| skylinkapi/…-kind-mango | kind-mango (31, 264.34) | **yes since v73 (09-26)**; excluded before | the switch, above |
| saurabhdxt-labs/prc-challenge | merry-quicksand (24, 250.73) | yes (delta target) | none |
| col3name/air-data | kind-earthquake (34, 268.52) | only `AOBT_3 − EOBT_1`; the committed code is not the 268.52 code (slot 23) | none (last commit 09-05) |
| javidmardanov/PRC-Data-Challenge-2026 | elegant-alligator (35, 268.56) | yes, clock `MVT − AOBT_3` (slot 08) | none (09-15) |
| Phoenix-Ops-LTD/prc2026-taxiout | zestful-fountain (44, 273.56) | yes, CatBoost residual on `MVT − AOBT_3` | none (09-13) |
| satam2/knowledgeable-helicopter | knowledgeable-helicopter (49, 276.38) | yes, `Y − (MVT − AOBT_3)` | campaign closed 09-27 (`e3d2339`); explored EUMETSAT, lightning and IMERG sources, all stopped |
| ChandanHegde07/OpenAir | likable-eagle (50, 276.99) | yes, primary clock | none (09-15) |
| radekacar/joyous-rainbow | joyous-rainbow (100, 299.23) | not checked | **new** 09-25: README says *"2.9 % of departures produce 71 % of the squared error"* |
| RavindraTarunokusumo/prc-data-challenge-2026 | genuine-cabbage (not on the board) | yes (`src/prc/features.py`) | new day-1/day-2 "autonomous Claude researcher" branches; no upload |

Team-page self-descriptions for the new top teams [S] (said, not shown):
- **strong-crane:** Somnath Panigrahi, TU Dresden, "researcher working in aviation domain".
- **upstanding-firefly:** Jiawei Jing, Makuhari Development Corp., Japan. The page says they use
  *"OpenSky ADS-B data for an aircraft-noise research project"* and plan *"code on GitHub, write-up
  for JOAS"*.
- **humble-giraffe:** five high-school students, Indus International School Pune.
- **unbelievable-snowflake:** Daniel Lindner, Germany.
- **versatile-watermelon:** Tin Josip Curik, Maastricht University.
- **strong-turtle:** Nathaniel Ehimen, Nigeria.
- **understanding-accordion:** Alexander Beiser, TU Wien.
- **upbeat-goblin:** Henri Bano, m-click.aero; thesis on reconstructing stand arrival from ground
  surveillance.

**Correction to SUMMARY.md §B.** The `youthful-giraffe` page's member table lists **"Matthew Motoki",
affiliation "Unknow"**, country Indonesia (`teams/youthful-giraffe.qmd`, last changed 2026-09-12)
[S]. Slot 02 read the country and missed the member table. The name is on the official page.
Whether this is the Kaggle Grandmaster is still unproven [X].

**Answer to "do top teams use `AOBT_3_flt`?"** For ranks 1–23 it is **unknown**, because there is no
code. Every public repo at 250–277 s uses it, and kind-mango's measured cost of not using it was
17.5 s [S]. A top-5 score without it would take a signal no public source describes [X].
**Our reliance on `AOBT_3_flt` is not what separates us from the top.**

---

## 4. Public discussion and data since 2026-09-21

### 4.1 Official site

There is no announcement, FAQ, rule change or ranking change since 2026-09-18 (§2.3). The single
open issue (#3, 08-13) asks for a JOAS statement. [S]

### 4.2 Discord, second-hand via kind-mango (not re-verified)

| date | content |
|---|---|
| 09-09 | `BLOCK == SCHED` fallback rows **are** in the ranking set. Features from other movements (*"what is occupied, what is coming"*) are allowed. |
| 09-11 | No phase 2 planned. The organiser keeps the right to add one if reverse-engineering the ranking is too easy (*"we despise"*). Trino is not allowed. |
| 09-17/18 | **ADSB.lol and AIU daily files count as open data with attribution.** fl340 asked about `AOBT_3_flt` and ADS-B off-block extraction (*"~40 % of EDDM flights have a first observation near the gate"*). espinielli replied: *"the model is for post-ops … there are no such restrictions: if you can find open trajectory data with good ground coverage and extract off-block times, we are ok. Practically speaking it won't be possible."* |
| 09-17/18 | lbragado reported corrupted training rows. Another participant said the ranking set likely holds the same corruption. |
| 09-17/18 | Team GREKI reported that wake-turbulence sequencing on the same runway was worth only 0.25 s live for them. |
| 09-23 | *"Force the outliers of the submission to look like the training outliers."* kind-mango priced it and rejected it: +1,639 MSE blend, -3 MSE cap. |

Source: `RECAP.md` "Discord confirmations" and the tail-rules row; `docs/WINNING_PLAN.md` §7.2.

### 4.3 Data availability, checked today

`adsblol/globe_history_2026` publishes daily release tags for **every day of January 2026 and July
2026**, two or three parts per day. `globe_history_2025` exists for training. [S]
(`git ls-remote --tags`)

### 4.4 Web

There are no blog, LinkedIn, Kaggle, arXiv or Medium write-ups on PRC 2026 beyond the repos already
listed. CompeteHub carries a listing only. [S] (web searches, 2026-09-29)

---

## 5. How we differ from the public front-runners

| lever | kind-mango (264) | merry-quicksand (≤267, code at 09-12) | us (279) | source |
|---|---|---|---|---|
| reads `AOBT_3_flt` | yes (since 09-26) | yes | yes | [S] |
| target on the matched lane | **blend** of plain `y` (clock as feature) and anchored `y − clip(MVT−AOBT_3)` | anchored `delta` | anchored `g` only | [S] |
| second learner | CatBoost 50/50 (-2.87 s live) | CatBoost files present | CatBoost lost as configured, on hold | [S] |
| LightGBM | `linear_tree`, `linear_lambda` 15 | — | plain | [S] |
| neighbour clocks | backward closed; **forward** `(t, t+15/30]` kept | "congestion witness", airport-hour median `MVT − AOBT_3` | **backward only**, `[t−w−60 s, t−60 s)`, airport level, 30 and 60 min | [S] `src/prc/features.py:111-126, 198-208` (read, not edited) |
| external off-block measurement (ADS-B) | no | **yes, -7,722 MSE live** | **no** (no ADS-B in `src/` or `configs/`) | [S] |
| LIRF regime head / Step A | yes | local-day rule | own rules (P1 lane) | [S] |

---

## 6. The one thing I think we are missing

**An independent measurement of the scored flight's off-block time from open ADS-B surface tracks
(adsb.lol `globe_history`), used as a stacked residual on the matched lane.**

The evidence:
1. It is the **largest single live gain any public team has documented**: merry-quicksand v10 → v11,
   **-7,722 board MSE**, measured on the board, 1.36× its projection. It moved 161,036 matched rows
   at 9 of 10 airports (§3.2). [S]
2. Applied at our level, the same MSE would take 279.26 to about **265.1 s**, roughly 14 s and about
   24 ranks at today's density. [X] This is arithmetic that assumes the same absolute gain transfers,
   which is unmeasured.
3. **It is the one lever that brings information the organiser's files do not contain.** Every other
   public lever re-uses the same columns everyone has. A board that jumps 45 s is more likely to have
   gained information than modelling. [X]
4. **It was blocked in our own research only by a rules question, and that question has been
   answered.** Slot 21 §6.4 said "do not copy" because the organiser had not ruled. On 2026-09-18 the
   organiser said open trajectory data for off-block times is *"ok"* and ADSB.lol counts as open data
   with attribution. [S] That ruling reaches us second-hand via kind-mango (§0), so it must be
   confirmed on Discord before anything ships.
5. **Timing fits.** The ruling came on 09-18. The second-tier wave began 09-23 and peaked 09-26 to
   09-28, the time an ingest and stacker takes to build. upstanding-firefly jumped -21.5 s on 09-23
   six days after its previous upload, and its team page says it already works with OpenSky ADS-B data. [S] for the
   facts, [X] for the link.
6. Also: the organiser's reason for calling it impractical, poor coverage at LFPG, LIRF, LEMD and
   LTFM, did not stop merry-quicksand moving 23,791 rows at LEMD and 4,596 at LFPG. [S]

**What this does not explain.** ADS-B accounts for about 14 of the 58.6 s gap. kind-mango shows about
15 s more is available from modelling on the same columns (§3.1, §5). Roughly 25–30 s of the leader's
edge has no public explanation. That part most plausibly sits in the Discord backlog or in a
tail/fallback rule. Our own `visible-anomaly-detection.md` §7.2 bounds the latter at "3 to 80 s of
pure record corruption". [X]

**What it costs, from merry-quicksand's notes** [S]: about 190 GiB and about 20 h of 2026 ingest.
Their fetcher streams the release tarballs and parses without untarring, with a peak disk of about
3.5 GiB. The join key must be `CALLSIGN_flt`, not `FLIGHT_mvt`: at EHAM `FLIGHT_mvt` joins 2.2 % of
rows, `CALLSIGN_flt` 99.5 % (`reports/ADSB_GATE.md`). The ingest is disk- and network-heavy, and the
box is shared.

**Guardrail.** Do not approach the extreme-row lever by probing. The equal-sized paired jumps in §2.3
are exactly what the ranking page warns about. Any tail rule must come from 2025 evidence, as
`visible-anomaly-detection.md` does. [S] (`ranking.qmd`, organiser 09-11 "we despise")

---

## 7. What I could not establish

- **Discord.** I could not read it, and every Discord line above is relayed. The 09-18 ruling in
  particular should be confirmed first-hand before any ADS-B work ships.
- **The code of ranks 1–23.** None is public. No README exists to overclaim, and no code exists to
  check.
- **Whether the 09-26 to 09-28 jumps are ADS-B, AOBT_3 adoption, a tail rule or probing.** Scores
  alone cannot separate these (§2.3).
- **merry-quicksand's path from 265.47 to 250.73.** It is not in their repo.
- **strong-crane's switchable 25–40 s component.** It shows only as score levels (§2.2).

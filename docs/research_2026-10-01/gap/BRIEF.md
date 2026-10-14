# Brief: what separates us from the leaders (PRC Data Challenge 2026, taxi-out RMSE)

Task: predict TAXITIME_SEC_mvt (= MVT_TIME - BLOCK_TIME, s) for 344,841 departures (Jan+Jul 2026, 10 EU airports), RMSE.
Ranking DEP rows have BLOCK_TIME and TAXITIME blanked; all other columns and all ARR rows (with labels) remain.
Our team: resilient-hedgehog. Best LB v32 = 251.7333 (2026-10-02), rank ~30. Leader strong-crane 219.64; #2 vigorous-whistle
222.05; #3 jolly-lobster 222.91. Deadline 2026-10-11 21:59Z. Official leaderboard API (per-submission, paginated with nextCursor,
limit<=200, filter ?teamName=): https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard
Teams API: .../teams?page=&size= . Site source repo: github.com/euctrl-pru/prc_data_challenge_website_2026 (team pages teams/<name>.qmd).

Our model (v32): NM-matched rows (98.5%): y = (MVT - AOBT_3) + g(x), g from a 3-member LightGBM/CatBoost blend (+ per-airport
member, forward NM windows, sequence features, OSM stand distance, weather), then a learned residual from open ADS-B (adsb.lol)
pushback detection at 9 airports (none at LTFM: no ground coverage). NM-unmatched rows (1.5%): model B with LIRF schedule
re-parameterisation. Known error anatomy: RMSE is tail-dominated; the 1.5% unmatched rows held ~44-50% of squared error in CV;
LIRF schedule-fill and one-day rollover rows are labels "as reported"; genuine-row error is dominated by the airport off-block
being much earlier than NM AOBT_3 (pushback-then-wait), which only ADS-B can see.

Prior team research to read FIRST (do not redo): on branch origin/claude/model-b-and-arrival-features or origin/claude/beautiful-davinci-ehugk4,
read with `git -C /home/user/prc-data-challenge show <branch>:<path>`:
 docs/competitor_research_2026-09-21/results/01-lb-top-teams-2026.md, 25-top-teams-update-2026-09-29.md,
 public-code-deep-dive-2026-09-30.md (find with git ls-tree), 21-repo-merry-quicksand.md, 22-repo-kind-mango.md,
 docs/gm_research_2026-10-01/*.md, docs/research_2026-10-01/README.md (local working tree), docs/README.md section 8.2 (gap breakdown),
 docs/competitor_research_2026-09-21/results/adsb-learned-residual.md.

Rules: do not modify the repo; work under /tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/gap/<angle>/.
Python: /home/user/prc-data-challenge/.venv/bin/python (polars, requests...). Data: /home/user/prc-data-challenge/data/raw/.
Concurrent agents: keep memory < 2.5 GB, POLARS_MAX_THREADS=1. GitHub REST API may be blocked (403): use git clone --filter=blob:none
or raw.githubusercontent.com. Never probe the leaderboard with submissions. Mark facts [S]/measured vs inference [X].
Deliverable: Japanese, <= ~1000 words, findings ranked by how much of the ~32 s gap they could explain, each with evidence,
and concrete recommendations for the remaining 9 days.

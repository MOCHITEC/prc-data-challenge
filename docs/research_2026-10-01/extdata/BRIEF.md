# Brief: is there any external data we can REALLY use? (PRC Data Challenge 2026, taxi-out RMSE)

Task: predict TAXITIME_SEC_mvt = take-off (MVT) - off-block (BLOCK), seconds, for 344,841 departures in Jan + Jul 2026 at
EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM. Train: 2025 (all months). Ranking DEP rows have BLOCK and TAXITIME blanked.
Team best LB v34 = 246.75 (rank 32/230), leader 213.92. Deadline 2026-10-11 21:59Z (6 days left).

Prize conditions (docs/01_competition_facts.md section 4): external data must be openly accessible, usable and documented, under an
open licence; code published under GPLv3. Organiser statements (second-hand, via Discord relays): open trajectory data / adsb.lol allowed
(09-18), OPDI is open data (09-09), OpenSky Trino NOT allowed. Never use data that reveals the test labels (e.g. per-flight actual
off-block from a source that is the label itself is fine only if it is an independent observation like ADS-B; but published taxi-out
statistics FOR the test months are aggregated labels -> flag as compliance-sensitive, do not build on them).

What already exists (read first, do not redo): branch origin/claude/model-b-and-arrival-features (git -C /home/user/prc-data-challenge show <branch>:<path>):
 docs/external_data.md, docs/competitor_research_2026-09-21/results/16-external-data.md, external-data-survey-2026-09-29.md,
 adsb-learned-residual.md, docs/opdi_2026-09-30/, docs/weather_v2_2026-09-25/, docs/osm_stand_2026-09-28/,
 docs/research_2026-10-01/gap/README.md (local working tree), and recent commits on that branch (OPDI exit events ingest, positional ADS-B join).
In use already: METAR (IEM), OSM stands + OurAirports runways, adsb.lol ADS-B (Jan+Jul 2025 and 2026 only), OPDI flight list (turnaround).
Known gaps: LTFM has zero adsb.lol ground fixes (2025 and 2026); LFPG stand detection 3% despite 58% ground fixes; EDDF adsb.lol
coverage 2-8%; unmatched rows (1.5%) have no NM clock at all.

Rules: do not modify the repo; work under /tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/extdata/<angle>/.
Python: /home/user/prc-data-challenge/.venv/bin/python. Network via proxy is allowed; small samples only (< 2 GB total download).
Concurrent agents: keep memory < 3 GB. For each candidate report: what it contains, licence (quote it), period/coverage for our 10 airports
in 2025 and Jan/Jul 2026 (MEASURE on a small sample where possible), how it would enter the model (which rows, which error group),
a rough value estimate, cost (hours), and a verdict (USE / TRY / NO) with reasons. Mark [S] measured/read vs [X] inference.
Deliverable in Japanese, <= ~1000 words, ranked by expected value.

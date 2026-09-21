# Competitor research 2026-09-21 — 20 investigation slots

Commander: Claude (coordination only, no analysis of my own).
All findings produced by SubAgents and written verbatim to results/.

Rule for every slot: facts with sources only. Mark [S] = read from a source,
[X] = the agent's own extrapolation. Never present [X] as [S].

| # | slug | scope |
|---|------|-------|
| 01 | lb-top-teams-2026 | current PRC 2026 leaderboard top 10: identities, submission curves, any public traces |
| 02 | motoki-method | youthful-giraffe / Matt Motoki: his published methods across competitions |
| 03 | prc-2024-writeups | PRC 2024: every published top-10 writeup/repo, in detail |
| 04 | prc-2025-writeups | PRC 2025: every published top-10 writeup/repo, in detail |
| 05 | nasa-pushback | NASA/DrivenData Pushback Prediction: all prize-winner repos, code-level |
| 06 | repo-openair | ChandanHegde07/OpenAir: code-level read |
| 07 | repo-phoenix-ops | Phoenix-Ops-LTD/prc2026-taxiout: code-level read |
| 08 | repo-javidmardanov | javidmardanov (LB 268.56, best public): code-level read |
| 09 | repo-sergiuv11 | sergiuv11 + ahmetabdullahgultekin: code-level read |
| 10 | repo-discovery | exhaustive GitHub sweep for further 2026 public repos |
| 11 | ensembling-patterns | how aviation/tabular winners actually build blends |
| 12 | per-group-specialists | per-airport / per-segment specialist model patterns |
| 13 | target-transforms | target transformation & loss choice for heavy-tailed duration targets |
| 14 | validation-design | CV design in aviation comps with time structure |
| 15 | postprocess-overlays | post-processing, clipping, rule overlays |
| 16 | external-data | external data sources used by aviation comp winners |
| 17 | congestion-features | queue/congestion/demand feature catalogs |
| 18 | turnaround-features | arrival / turnaround / stand-linkage feature catalogs |
| 19 | atm-literature-sota | academic SOTA for taxi-out prediction |
| 20 | tabular-comp-craft | general tabular-competition craft not yet applied here |

## Slots added after slot 10's discovery sweep (2026-09-21)

| # | slug | scope | state |
|---|------|-------|-------|
| 21 | repo-merry-quicksand | saurabhdxt-labs/prc-challenge, rank 11, 257.55 — clock-disagreement reparameterisation | running |
| 22 | repo-kind-mango | skylinkapi/...-kind-mango — RECAP.md signed-delta ablation ledger | queued |
| 23 | repo-col3name-satam2 | col3name/air-data (rank 21, 268.52) + satam2 (rank 36, 278.01) | queued |
| 24 | repo-tail-sweep | fenil264, sqcm9896, GLAPM, victoralcadi, kagervig | queued |

Queued slots are waiting on the 20-concurrent-subagent limit.

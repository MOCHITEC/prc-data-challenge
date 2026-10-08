# ADS-B headroom and the gap to 10th place (2026-10-08, read-only analysis of committed records)

Abbreviations: LC = origin/claude/laughing-cray-vvwh0k; BD = origin/claude/beautiful-davinci-ehugk4; ALR = LC:docs/competitor_research_2026-09-21/results/adsb-learned-residual.md.

## Where we stand
- Team best: v47 = 234.48, rank 20/237 (LC cf0f289).
- 10th place is projected at about 219–222, so the gap is 12.5–15.5 s.

## ADS-B
- **2026 rank-row coverage** (ALR:175–203):
  - ADS-B joined 83.5%, ground fix 71%, stand exit seen 43%, pushback seen 21%.
  - LTFM has no ground fixes. LFPG and LIRF are thin, and LIRF has none in Jul 2026.
  - EDDF's 2026 ground density fell (median fixes 34 → 8–12).
- **2025 by coverage class** (ALR:483–486):

  | Class | Row share | Residual MSE reduction | Note |
  |---|---|---|---|
  | K0 no ADS-B | 17% | 0% | |
  | K1 air only | 26% | 0.4% | holds about 48% of SSE |
  | K2 mid-taxi | 24% | 16% | |
  | K3 stand exit | 33% | 25% | |

- **Delivered:** about 35 s of LB gain in total. Recent add-ons transfer only 10–35% of their CV gain to the LB.
- **Headroom:**
  - Rows where the pushback is seen are already at the measurement floor.
  - Making K2 as good as K3 is an oracle of about 2.2 s, and it is unreachable.
  - Realistic remaining ADS-B work is worth 0.2–1 s on the LB.

## Top teams (repo surveys; no top-30 code is public)
- **Legal ideas:**
  - full-year ADS-B in the residual stage;
  - stands inferred from ADS-B dwell plus a fallback match;
  - ADS-B censoring bounds;
  - within-day-demeaned neighbour features;
  - runway-state block;
  - 12-month cross-fit plus a context corrector.
- **Off-limits for us:** PRU or EUROCONTROL label aggregates, scale inference from LB scores, diagnostic submissions, timestamp precision, ID order.
- **The top-board spread looks substantially tail-driven:** a few rows missed by about a day are worth 10–40 s of LB.

## Verdict
- 10th place is not reachable by learnable improvements by 10-11.
- A realistic finish is about 231–233.
- Remaining levers, each under 3 s:
  - an ADS-B layer rebuild on v47 with 2026-like thinning, 0.5–1.5 s (integration session);
  - K2 stand inference and censoring bounds, 0.3–1.0 s (integration session);
  - within-day-demeaned neighbour features and a runway-state block, 0.5–1.5 s (this branch).
- Upload hygiene: v42 and v47 were overwritten by version collisions. List the bucket and use max + 1.

# PRC Data Challenge 2026 — Taxi-out time prediction on Snowflake

Solution repository for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/)
(EUROCONTROL Performance Review Commission with OpenSky Network): predict the taxi-out time (seconds,
RMSE) of departures at 10 major European airports for January and July 2026 from 2025 movement data.

Snowflake is used as the data and feature platform (raw load, typed staging, SQL feature generation,
fold tables, submission views); gradient-boosted models are trained outside the warehouse.

## Status

Planning phase. See `docs/`:

| Doc | Content |
|---|---|
| `docs/01_competition_facts.md` | Competition rules, data schema, submission mechanics, open questions |
| `docs/02_snowflake_env.md` | Snowflake environment, constraints, object design, security policy |
| `docs/03_prior_art.md` | Literature and past-edition survey |
| `docs/04_master_plan.md` | Phased plan, architecture, validation design, risks |

## Layout

```
docs/      planning and decision records
sql/       Snowflake DDL/DML, numbered in execution order (000 = manual ACCOUNTADMIN steps)
src/prc/   Python package: data access, features, training, submission validation
tests/     synthetic-data tests (feature parity, submission gate)
configs/   experiment configurations
refs/      public reference snapshots (leaderboard API)
```

## Reproduction

To be written with the pipeline (Phase 5 of the plan). Competition data are not redistributed here;
they are obtained from the organisers' buckets with your own credentials (`.env.example`).

## License

GNU GPLv3 (see `LICENSE`), as required by the competition prize conditions.

# Data audit, 2026-09-18

Why this exists. The 2026-09-16 investigation asked "why is the score stuck" and found one
data pathology: at LIRF, `BLOCK_TIME_UTC_mvt` is filled with `SCHED_TIME_UTC_mvt`, so on
those rows the label is not a taxi time at all. Acting on that single finding took the
leaderboard from 305.03 to 284.36 (v10, measured by a probe that changed only those rows).

This audit asks a different question: **what else is wrong with, or simply unknown about,
this data?** It is a census of the data itself, not of our model. Each agent takes one lens
and reports what it finds AND what it finds clean, so the clean areas are not re-audited.

Template for what counts as a finding: "a column contains something other than what its name
says, for an identifiable subset of rows, and the subset is identifiable from columns that
survive into the ranking set."

Definitions are `docs/investigation_2026-09-16/plan.md` §0.

# Protocol frozen before candidate outcomes

2026-09-18. Baseline: existing cached features, January–August training; August inner stopping,
refit on January–August; 1 seed, 1,200 round cap, feature_fraction=1, two threads;
LIRF unmatched schedule reparameterisation enabled to approximate the current best model B.
The cache is read-only. No historical OOF residuals are used.

Candidate generation spans clock revisions/precision, identity morphology, schedule phase,
arrival/stand continuity, neighbour reporting and weather/traffic changes. Exclude the ten
previously rejected columns. Freeze the candidate definitions before looking at October outcomes.

Broad inexpensive screen: fit airport × candidate-bin shrunken residual means on September,
evaluate October. Both months are predictions of the SAME January–August baseline. Numeric
cutpoints come only from September; categorical vocabularies are label-independent. 100 pseudo
observations at zero shrink cell corrections; unseen groups get zero. Score both matched and
unmatched with separate tables. Correct predictions then clip below zero as the baseline does.
Include airport-only correction, four deterministic hash controls and existing-feature controls.
Report full all-row RMSE differences (candidate minus baseline), paired calendar-day bootstrap
95% CIs (2,000 draws), matched/unmatched deltas, and a fixed baseline-worst-200 diagnostic.
No trimmed score decides adoption. Screen CIs are exploratory, not simultaneous claims.

Select at most five novel candidates by October full RMSE improvement with upper CI < 0 and
improvement beyond all noise controls; require matched improvement too, to prevent a handful
of unmatched rows alone determining selection. Freeze selection before opening November or
December errors. If fewer than five pass, use those that pass. Fit the selected correction on
September+October residuals of the SAME unchanged model, then confirm on November–December.
Use Bonferroni-adjusted day-bootstrap intervals for the selected candidates, separately report
both months and airports. This tests correction transfer for this fixed baseline, not correction
transfer to a model refitted through December.

Also test a shallow joint residual model containing all novel candidates (fixed complexity,
no evaluation early stopping), versus the same model on existing context and noise. Freeze
its hyperparameters before outcomes. Confirmation includes this predeclared joint test in the
multiplicity count. A positive correction is only a candidate for adoption: full feature refits
with feature_fraction=1 and a noise refit must verify shortlisted feature bundles. Full refits
use January–August and evaluation November–December, with September–October used only for
selection. Report their status separately; do not claim residual screens measure feature-refit
performance. Production geometry (12 months → January/July next year) remains untested locally.

Before outcomes: selection also must beat the airport-only correction, so mere airport
calibration cannot earn selection. Joint learner: separate matched/unmatched LightGBM models,
15 leaves, 120 rounds, learning rate .03, L2=100, minimum leaf 300/100, full row/column sampling.
Baseline context for its control is airport, runway, stand, operator, aircraft type, local hour,
three main clocks, temperature, departure count, arrival taxi mean, and baseline prediction.
The novel arm adds all 63 engineered columns; noise arm adds four hashes.

Feature-refit verification uses the baseline's fitted matched and unmatched round counts,
with no new early stopping. This holds capacity/seed/rows fixed across baseline, new features,
and hash-noise arms. Refit the selected scalar bundle and the predeclared all-candidate bundle,
plus one hash control, sequentially. String candidates enter as native categorical variables;
Boolean candidates explicitly enter as Int8. Confirmation comparisons against baseline and
noise are reported for both bundles with Bonferroni 98.75% intervals (four comparisons).
No choice between these bundles uses November–December outcomes.

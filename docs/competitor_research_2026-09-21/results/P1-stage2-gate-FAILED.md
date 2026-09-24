# P1 stage 2 — the fill-classifier gate: **FAILED**. Stages 4 and 5 are CLOSED.

Measured 2026-09-24. Script `docs/competitor_research_2026-09-21/scripts/p1_stage2_fill_auc.py`,
read-only, nothing entered the pipeline.

Label `f10 = |BLOCK_TIME - SCHED_TIME| <= 10 s`, observable on training rows only. Held out one
month at a time and scored on **LIRF unmatched rows only** (1,488 rows, 721 positive).

## Result against the gate set before the run

| recipe | held-out AUC | AUC on `ms >= 2h` | logloss | vs (b) |
|---|---:|---:|---:|---:|
| (a) unmatched rows only — our current recipe | 0.8418 | 0.7906 | 0.5450 | — |
| (b) + the 159,216 LIRF matched rows, unmatched x8 | 0.8751 | 0.8339 | 0.4603 | — |
| (c) + stake weight + flight prefix/identity | **0.9127** | 0.8930 | 0.4232 | +0.0376 |
| (c-) (c) without the stake weight | **0.9201** | 0.8988 | 0.3678 | +0.0450 |
| (c~) (c) with prefix/identity permuted — control | 0.8562 | 0.8187 | 0.5508 | −0.0189 |

**Gate: 0.93. Recipe (c) reaches 0.9127; the best variant of it reaches 0.9201. Both below.**
Stages 4 and 5 are recorded CLOSED, with the AUC as the reason, per the plan's own instruction.

The threshold was not arbitrary. Our classifier mixture previously measured 1,926.1 against the
reparameterisation's 1,632.1 — a 294 s gap — at an AUC of 0.84-0.92, and kind-mango's shipped
gate, the one worth −9.8 s live, sits at 0.9256. 0.92 does not close a 294 s gap.

## What the run established anyway

**The hypothesis was right; it just does not reach far enough.** Training on the LIRF matched rows,
where the fill is observable on 159,216 rows instead of 1,488, lifts held-out AUC 0.8418 → 0.8751.
Flight prefix and identity lift it further, to 0.9201. That is the ingredient merry-quicksand,
javidmardanov and kind-mango have and we did not, and it is worth about +0.078 of AUC. It is a real
mechanism; the gate says it is not a sufficient one.

**The negative control behaved correctly.** Permuting prefix and identity scores 0.8562, *below*
(b)'s 0.8751 — permuted features are worse than no features, which is what noise added to a model
should do. The stated failure condition (a control keeping more than 20% of the lift) does not
apply: it keeps −50%.

**The stake weight does not transfer.** `clip(1 + (max(ms,0)/3600)^2, 1, 25)`, which
javidmardanov and merry-quicksand both use, costs us 0.0074 of AUC (0.9201 → 0.9127) and 0.055 of
logloss. Do not import it.

**Calibration is monotone in every recipe** — (c-) runs 0.036 / 0.097 / 0.298 / 0.613 / 0.871 /
0.992 across sextiles. The score is usable as a probability; there is simply not enough separation
at the top.

## A methodology note, recorded because it nearly shipped

The first run of this script reported **AUC 1.0000 on all five recipes, including the permuted
control**. That is the signature of a leak, not of a classifier: a negative control cannot match
the real thing. The cause was in the script — it attached the `f10` label to the frame and then
called `numeric_feature_columns()` on that frame, which reads whatever numeric columns are present,
so the label became a feature. A single-feature check confirmed it (f10 scores 1.0000 against f10;
the best real feature is `mvt_minus_sched` at 0.6616).

The script now takes its feature list before the label is attached and asserts the label is absent.
A second anomaly in that output — `gust_ms_prev1h` at AUC 0.0000, a perfect inverse — turned out to
be 9 non-null rows out of 1,488, not a second leak.

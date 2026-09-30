"""Training / cross-validation for the two-clock model (docs/README sections 4-5).

Model A (rows with an NM off-block time): learn g(x) = TAXITIME - (MVT_TIME - AOBT_3) and predict proxy + g(x).
    A tree cannot represent the identity part of y = proxy - f(x), so the reparameterisation is the point.
Model B (rows without one): learn TAXITIME directly from schedule-based features.

Splitting the two regimes is measured, not assumed. One model over everything - same reparameterisation, target
LABEL - coalesce(proxy, 0), free to branch on nm_matched - is worse by 74.10 s overall and still worse by 6.04 s
once the 2,000 largest errors are removed, so it is not a tail artefact. It loses on both groups at once
(matched 218.5 -> 226.2, unmatched 1,875.3 -> 2,697.2): one model has to spend its capacity telling the regimes
apart before it can fit either. Do not merge them back.
Both use L2 on raw seconds and keep every labelled row (no outlier filtering) - docs/README.

Invariants enforced here (each one was a real bug caught in review):
* Every label-derived table (stand x runway reference, airport-runway-hour median, airport median) is fitted on
  the training side of the fold and only joined onto the validation/ranking side.
* Early stopping never sees the evaluation fold: the stopping point comes from an inner holdout carved out of
  the training months, and the final model is refitted on all training rows with a fixed number of rounds. The
  same code path therefore produces the same model whether or not the target frame carries labels.
* The invariant above is about the VALIDATION fold and says nothing about the training side, which is where the
  defect was. `fit_encodings` is fitted on every labelled training row and `_fit` then carves the inner
  early-stopping holdout out of that same encoded frame, so the rows that choose `best_iteration` read
  encodings their own labels helped build. `es_holdout_encodings` (off by default) repairs exactly that: the
  holdout block is re-encoded from a map fitted without its own calendar months. The rows the model is fitted
  and refitted on, and the target frame, are NOT cross-fitted and are byte-identical either way - the one
  channel from the flag to a prediction is the integer round count. The training rows' own self-inclusion is a
  separate, measured-as-harmful change and is deliberately not made here; see
  docs/competitor_research_2026-09-21/plans/P2-cross-fit-encodings.md §3.2.
* Predictions are returned keyed by MVT_ID_mvt, never by row position.
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .cv import Fold, metrics, month_block_folds, per_airport, year_month_key
from .features import CANDIDATE_COLS, CATEGORICALS, numeric_feature_columns

LABEL = "TAXITIME_SEC_mvt"
PROXY = "mvt_minus_aobt3"
# MVT_TIME - SCHED_TIME. Model B's optional reparameterisation offset; see TrainConfig.unmatched_sched_reparam.
SCHED_OFFSET = "mvt_minus_sched"
# MVT_TIME - LOBT. The anchor of the prediction-side bound; see TrainConfig.lobt_bound.
LOBT_OFFSET = "mvt_minus_lobt"
ENCODING_COLS = ["ref_p10", "ref_p20", "ref_med", "ref_n", "arh_med", "arh_n", "apt_med", "ref_p10_fb", "ref_missing"]
# Encodes what model A actually predicts - the offset between the two off-block clocks, AOBT_3 minus
# BLOCK_TIME - rather than the taxi time. Measured on one month: the median offset runs from -301 s to +56 s
# across operators and its spread from 237 s to 473 s, so the reporting habit is the operator's, not the
# airport's. Operator and airport are both categoricals already, but a tree cannot aggregate across rows, and
# learning 676 x 10 combinations by splitting alone is not the same as being handed the number.
OFFSET_COLS = ["op_offset_med", "op_offset_n"]
# Grouping keys model A may be split on; see TrainConfig.airport_models and
# docs/competitor_research_2026-09-21/plans/P11-per-airport-models.md.
#   adep     - the departure airport. The candidate.
#   shuffled - the same number of groups with the same size profile, membership drawn from a hash of
#              MVT_ID_mvt alone. The control: it carries the row-count cost and the blending gain of the
#              candidate and none of its airport information.
GROUP_KEYS = ("adep", "shuffled")
# The model family the fitter uses; see TrainConfig.model and
# docs/competitor_research_2026-09-21/plans/P10-catboost-rival.md. Everything outside `_fit` - folds, encodings,
# feature selection, routing, target parameterisation, the seed loop, the round-count scaling, assembly,
# metrics - is shared, so the two families differ in the fitter and in nothing else.
MODEL_FAMILIES = ("lightgbm", "catboost")
# CatBoost's settings when a recipe names none. Every key is explicit on purpose, because two of CatBoost's
# library defaults are not constants: `max_ctr_complexity` resolves to 1 at <= 150 iterations and to 4 at
# >= 200 (measured on catboost 1.2.10, P10 §2.3), so a short screen and a production-length run would silently
# run two different categorical algorithms; and the learning rate is chosen from the data size and iteration
# count when it is not given. `thread_count` matches LightGBM's `num_threads`, which is a determinism setting
# as well as a speed one. CatBoost has no `deterministic` flag to pass: on CPU a fixed `random_seed` and
# thread count give a bit-identical model (checked in tests/test_catboost_family.py and on real data).
CB_DEFAULTS: dict = {
    "loss_function": "RMSE",
    # 0.2, not the library's auto rule. That rule gives 0.073-0.076 for 2,000 iterations at 1.43-1.74M rows,
    # i.e. it plans to spend every round. Measured on the forward fold at a 20% training subsample (P10 results
    # §2): early stopping's best count was 1,973 of 2,000 at 0.1 and 1,984 of 2,000 at 0.2 - capped both times -
    # and 2,500 at 0.2 with a 5,000 cap. The higher rate is the one that reaches its own stopping point soonest.
    "learning_rate": 0.2,
    "depth": 6,
    "max_ctr_complexity": 1,
    "thread_count": 4,
    "verbose": 0,
    "allow_writing_files": False,
}
# Keys the fitter owns: the iteration count and the stopping rule come from num_rounds / early_stopping and the
# seed from `seed + k`, exactly as for LightGBM, so a recipe that also set them in cb_params would be setting
# them twice. `has_time` is refused because it needs the training block sorted by time and this fitter does not
# sort - silently ignoring the ordering is the failure mode it would have.
CB_OWNED_KEYS = (
    "iterations",
    "n_estimators",
    "num_boost_round",
    "num_trees",
    "random_seed",
    "random_state",
    "od_type",
    "od_wait",
    "early_stopping_rounds",
    "use_best_model",
    "has_time",
    "cat_features",
)


# TrainConfig fields added after the archive was written, with their defaults. Left out of the hash while they
# hold their default, so every archived experiment id still names the config that produced it; a recipe that
# sets one gets a new id.
_HASH_OPTIONAL: dict = {
    "model": "lightgbm",
    "cb_params": {},
    # The fold-time families merged from the article-FE / weather-v2 / OSM screens. Left out of the hash at their
    # off values so every experiment id recorded before the merge is still the id of the same recipe.
    "fe_freq": False,
    "fe_aggdiff": False,
    "fe_cyclic": False,
    "fe_meta_prior": False,
    "fe_noise_cols": 0,
    "fe_weather_v2": "",
    "fe_osm_dist": False,
    "fe_osm_null_airports": [],
    "sim_unseen_stands": 0.0,
    "fe_opdi": False,
    "opdi_target_null_day_share": 0.0,
}


@dataclass
class TrainConfig:
    name: str = "v1_two_clock"
    seed: int = 42
    params: dict = field(
        default_factory=lambda: {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "num_threads": 4,
            # deterministic alone does not bind: LightGBM still picks row-wise or column-wise histogram
            # construction by timing the first iterations, and on a loaded machine that choice flips, changing
            # the floating-point summation order and therefore the splits. Two runs whose only difference was
            # in model B's settings - which model A is fitted before and cannot see - gave model A 1273 and 959
            # rounds on the same fold. The documentation asks for one of force_row_wise / force_col_wise
            # alongside deterministic; row-wise is the one for many rows and few bins, which is this data
            # (2.08M rows, 72 features, 255 bins).
            "deterministic": True,
            "force_row_wise": True,
        }
    )
    num_rounds: int = 2000
    early_stopping: int = 100
    use_categoricals: bool = True
    use_linear_tree: bool = False  # docs/README: reported worth about -10 s CLEAN RMSE; compare explicitly
    linear_lambda: float = 1.0
    # The nine label-derived columns (stand x runway P10/P20/median, airport x runway x hour median, airport
    # median, and the coalesced fallback). Present since the first commit and never ablated; the flag exists to
    # settle whether they earn the 40 lines of fold-wise fitting they cost.
    use_encodings: bool = True
    reference_min_count: int = 30  # self-inclusion in a P10 over a small group is a leak of 1/n
    inner_holdout_months: int = 1  # months of the training side reserved for early stopping
    # P2 Fix 1 / Option A (docs/competitor_research_2026-09-21/plans/P2-cross-fit-encodings.md §3.1).
    # On, the inner early-stopping holdout rows carry encodings fitted on the fold's training rows MINUS
    # the calendar months those holdout rows occupy, instead of the whole-frame encoder that their own
    # labels helped build. The rows the stage-1 booster is fitted on, the stage-2 refit rows and the
    # target frame are untouched, so the only channel from this flag to a prediction is the integer
    # `rounds` chosen by early stopping. Off by default: with the flag off `_fit` builds exactly the
    # matrices it built before this field existed, and every existing recipe is unchanged.
    es_holdout_encodings: bool = False
    # The rows with no NM record are 1.2% of the data and roughly half the squared error. Model B sees ~22k
    # rows where model A sees 2.06M, so the obvious guess was that it needed *less* capacity, and measurement
    # says otherwise: 31 leaves is worse than 127, 255 is better than 127, 511 adds nothing. The effect is
    # small once the largest errors are excluded (-0.26 s at 2,000 dropped), so this is a weak result, not a
    # strong one - it is the current setting because nothing contradicts it, not because it was worth 34 s.
    unmatched_num_leaves: int = 255
    unmatched_min_data_in_leaf: int = 20
    unmatched_num_rounds: int = 3000
    # Model B's target at the airports where BLOCK_TIME is filled with SCHED_TIME: `LABEL - mvt_minus_sched`
    # instead of `LABEL`, with the same offset added back to the prediction. The identity
    # y - mvt_minus_sched == -(BLOCK - SCHED) holds on all 2.08M training rows, so at an airport that fills
    # BLOCK with SCHED the label *is* the offset on the filled rows (LIRF unmatched: 48.5% of rows, corr(y,
    # offset) 0.8964 against 0.0205 at LFPG). A tree cannot represent that identity - the same reason model A
    # is given its proxy rather than asked to reproduce it. Measured on the jan/jul/forward OOF
    # (docs/investigation_2026-09-16/results/mms-mixture-reparam.md §3.3): unmatched RMSE 2,326.5 -> 1,632.1,
    # better on each of the three folds, lower at nine of ten airports and higher at none, and every non-LIRF
    # prediction >= 4 h gone (10 -> 0). Off by default: the §6 decision rule has not been run on it, and the
    # rows it moves are the ones §6 calls unmeasurable in a full-data comparison.
    unmatched_sched_reparam: bool = False
    # Departure airports the reparameterisation applies to; every other row keeps the plain target. Restricting
    # it is the point, not a detail: the same shift on *all* unmatched rows measures 2,696.7 against the same
    # 2,326.5 baseline (EHAM 954.7 -> 3,408.9), because only LIRF has the fill mechanism. A row whose
    # mvt_minus_sched is null is never shifted either way (null rate is 0 on the ranking set).
    unmatched_sched_reparam_airports: tuple[str, ...] = ("LIRF",)
    # Averaging over seeds removes the luck of one random draw: the 2024 winner reported 1,612 -> 1,562 from
    # 20 seeds (docs/README). Cost is linear in n_seeds, so keep it small until the effect is measured.
    n_seeds: int = 1
    # Whether `params["deterministic"]` and `params["force_row_wise"]` are passed to LightGBM at all.
    # True - the default, and what every archived record was produced under - keeps them, so two runs of the
    # same config in different processes give a bit-identical model. False drops BOTH (never one: `deterministic`
    # without a forced histogram direction is the half-measure the params comment above already rejects) and
    # lets LightGBM order its floating-point sums by thread arrival, which is reproducible only in
    # distribution. Measured cost of keeping them on this box, at the production row count (1,737,573 matched
    # rows, 108 features, 4 threads): 0.310 s/round against 0.316 s/round with them off - i.e. nothing. See
    # docs/competitor_research_2026-09-21/results/determinism-cost.md. Do not flip the default: the byte
    # comparison in configs/determinism_check.yaml, configs/p11_base.yaml and the no-op detector used
    # throughout docs/ all assume it.
    lgb_deterministic: bool = True
    # Which folds to run. All three is the honest answer for a final number, but a screening run does not need
    # them: the matched rows of the forward fold alone are 324k, giving that statistic a resolution of about
    # 1.1 s - the same as the full three-fold set - at a third of the wall clock. forward is also the only fold
    # shaped like the real task, since jan and jul train on months that come after the month they predict.
    folds: tuple[str, ...] | None = None
    # Columns hidden from the model. Candidate features are always computed, so measuring one at a time is a
    # config change rather than a four-minute rebuild of the feature cache.
    exclude_features: tuple[str, ...] = ()
    # Candidate columns are hidden unless a config names them. They used to be opt-out, and configs/main.yaml
    # did not opt out, so v6 shipped with all nine of them - noise_control, a hashed id built to carry no
    # signal, included - and scored 350.62 against v4's 305.03. Screening columns must never reach production
    # by default.
    include_candidates: tuple[str, ...] = ()
    use_offset_encoding: bool = False  # under test; see OFFSET_COLS
    # Project the prediction onto [mvt_minus_lobt - slack, mvt_minus_lobt + slack]. Since y = MVT - BLOCK,
    # that window is exactly {p : |(BLOCK implied by p) - LOBT| <= slack}, and |BLOCK - LOBT| <= 3606 is an
    # identity on this data: 0 violations in 2,062,577 training rows, in all 12 months and at all 10
    # airports, with the maximum attained at exactly 3606 and nothing beyond. So the truth lies inside the
    # window, and clipping onto an interval that contains the truth is a metric projection - it cannot
    # increase the squared error on any row, whether or not it binds. That is a different operation from
    # capping the label, which five board measurements refute; see docs/competitor_research_2026-09-21/
    # plans/{P5-physics-lower-bound.md,P5-REVIEW.md} for the derivation, the 2026 transfer argument and the
    # enumerated cost. Off by default: the bound cannot be verified on 2026, where BLOCK is blanked.
    lobt_bound: bool = False
    # The system constant, not a tuned quantile. Re-deriving it on our data returns 3606 exactly; widening it
    # weakens the bound and narrowing it makes it false. Do not sweep this.
    lobt_bound_slack: float = 3606.0
    # ---- per-airport specialists on model A (P11). Off by default: with the flag off `predict_fold` calls
    # `_fit` exactly as it did before this field existed, and every existing recipe is unchanged.
    #
    # On, model A becomes one global fit PLUS one fit per group, combined as
    # `(1 - airport_blend_weight) * global + airport_blend_weight * group` on that group's rows. The global
    # fit is kept, not replaced: the published evidence is that a per-group model ties or slightly loses to a
    # global one at this group size (Filienko et al., KDD 2023 UC: global LightGBM 10.7232 vs per-airport
    # 10.7686 MAE over 10 airports), while the teams that shipped per-group models shipped them blended
    # (NASA pushback 1st place: unweighted mean of 3 per-airport CatBoosts and 1 global; jubilant-vase in
    # this competition: a 50/50 global/local blend at 3 of 10 airports, 290.488 -> 286.656 official).
    # Model B is deliberately NOT split. Its per-airport training sets on the production folds are
    # 670-3,791 rows with a 60-379 row inner holdout to early-stop on, which is a fit nobody can defend;
    # the counts are in P11 section 3.
    airport_models: bool = False
    airport_group_key: str = "adep"  # one of GROUP_KEYS
    # Fixed at 0.5 and not tuned. Every source that reports a per-group blend weight that transferred used a
    # declared constant (NASA 1st place 1/4 each, M5 1st place an arithmetic mean, jubilant-vase 0.5); the one
    # sourced measurement of per-airport TUNED weights is negative (Phoenix-Ops, 323.712 against 323.377,
    # rejected). 1.0 is pure replacement and 0.0 is the global model; both exist so the two ends of the
    # blend can be priced, not so the weight can be swept.
    airport_blend_weight: float = 0.5
    # Groups smaller than this keep the global prediction untouched. 20,000 is OpenAir's hard guard in this
    # competition (run_e20_ensemble.py:265). On model A every airport clears it on every fold by a factor of
    # five or more, so it is a floor, not a selector - it decides nothing about which airports are used.
    airport_min_group_rows: int = 20_000
    airport_group_seed: int = 20_260_924  # only read by the `shuffled` control
    # ---- fold-time candidate families from two general feature-engineering articles, pre-registered in
    # docs/article_fe_2026-09-24/plan.md. All off by default: with every flag off `predict_fold` runs exactly the
    # code it ran before these fields existed. They are built inside the fold, after the encodings, so the
    # feature cache - and therefore the base arm - is untouched.
    #   fe_freq       label-free frequency encodings (share within airport, counted over train + target rows)
    #   fe_aggdiff    group aggregate minus the row's own value, and the group's label spread
    #   fe_cyclic     sin/cos of the local minute of day
    #   fe_meta_prior an out-of-fold "taxi prior" from a no-NM model fitted on every labelled row, as a feature
    #   fe_noise_cols this many hashed-id columns, the column-count control for the families above
    fe_freq: bool = False
    fe_aggdiff: bool = False
    fe_cyclic: bool = False
    fe_meta_prior: bool = False
    fe_noise_cols: int = 0
    # External weather v2 (docs/weather_v2_2026-09-25/plan.md): "" off, "B" METAR present-weather columns,
    # "C" those plus Open-Meteo precipitation / snowfall / snow depth. Joined at the take-off hour inside the fold.
    fe_weather_v2: str = ""
    # OSM stand-to-runway distance (docs/osm_stand_2026-09-28/plan.md): great-circle km from the stand's OpenStreetMap
    # position to the threshold of the departure runway. The one stand descriptor defined for a stand the model
    # has never seen.
    fe_osm_dist: bool = False
    # Airports (ADEP) where `osm_stand_rwy_km` is set to null after it is computed, so the model sees it as
    # missing there. Empty (the default) changes nothing. Requires fe_osm_dist. Added for
    # configs/main_flseq_osm_nolirf.yaml, whose LIRF exclusion was chosen after the LIRF result was seen (post hoc).
    fe_osm_null_airports: list = field(default_factory=list)
    # Measurement only: drop every training row whose (airport, stand) falls in this share of stands, chosen by a
    # stable CRC32 of "ADEP|STAND", so those stands are unseen by the model exactly as a stand new in 2026 is.
    sim_unseen_stands: float = 0.0
    # OPDI airframe turnaround (docs/opdi_2026-09-30/plan.md). LICENCE FLAG, see src/prc/opdi.py.
    fe_opdi: bool = False
    # Measurement only: blank the OPDI columns on the TARGET side for whole days chosen by CRC32 of the date, the
    # way the January 2026 OPDI file has no flights on 10 of its 31 days.
    opdi_target_null_day_share: float = 0.0
    # ---- the model family (P10). "lightgbm" is the default and the only value any archived recipe used: with it
    # `_fit` is the LightGBM fitter it has always been and `cb_params` must stay empty. "catboost" swaps the fitter
    # for `_fit_catboost`, which keeps `_fit`'s signature and its six contract items (P10 §1.3) and is handed the
    # identical frames, feature list, categorical list, target and seeds. The LightGBM-only fields are refused
    # rather than ignored under CatBoost - see `__post_init__`.
    model: str = "lightgbm"
    # CatBoost settings, merged over CB_DEFAULTS. A separate namespace from `params`, because `params` is merged
    # over the LightGBM defaults and CatBoost rejects those keys outright (`feature_fraction`, `force_row_wise`,
    # `num_leaves` without Lossguide: P10 §1.2).
    cb_params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fe_osm_null_airports and not self.fe_osm_dist:
            raise ValueError("fe_osm_null_airports is set but fe_osm_dist is off; it would be silently ignored")
        if self.model not in MODEL_FAMILIES:
            raise ValueError(f"model must be one of {MODEL_FAMILIES}, got {self.model!r}")
        if self.model == "lightgbm":
            if self.cb_params:
                raise ValueError("cb_params is set but model is lightgbm; it would be silently ignored")
            return
        # CatBoost. Each refusal is a LightGBM setting that has no CatBoost meaning here and would otherwise be
        # dropped without a trace, which is how a "CatBoost" arm ends up measuring something nobody chose.
        if self.use_linear_tree:
            raise ValueError("use_linear_tree has no CatBoost equivalent; refusing rather than ignoring it")
        if self.params != TrainConfig.__dataclass_fields__["params"].default_factory():
            raise ValueError("params are LightGBM's; a CatBoost recipe sets cb_params instead")
        if (self.unmatched_num_leaves, self.unmatched_min_data_in_leaf) != (
            self.params["num_leaves"],
            self.params["min_data_in_leaf"],
        ):
            raise ValueError(
                "model B's capacity override (unmatched_num_leaves / unmatched_min_data_in_leaf) is a LightGBM "
                "setting CatBoost cannot honour; set them equal to params' num_leaves / min_data_in_leaf so "
                "model B is sized like model A, as it is in configs/main_v15.yaml"
            )
        owned = sorted(k for k in self.cb_params if k in CB_OWNED_KEYS)
        if owned:
            raise ValueError(f"cb_params may not set {owned}: the fitter owns them (see CB_OWNED_KEYS)")

    def hash(self) -> str:
        d = {k: v for k, v in self.__dict__.items() if not (k in _HASH_OPTIONAL and v == _HASH_OPTIONAL[k])}
        if self.model == "catboost":
            # The RESOLVED settings, so an edit to CB_DEFAULTS is a new experiment id rather than a silent one.
            d["cb_params"] = self.catboost_params()
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def catboost_params(self) -> dict:
        """CB_DEFAULTS with the recipe's cb_params on top. The fitter adds iterations, stopping and the seed."""
        return {**CB_DEFAULTS, **self.cb_params}

    def for_unmatched(self) -> TrainConfig:
        """A copy sized for model B's much smaller training set."""
        if self.model == "catboost":
            # `__post_init__` has already required model B's LightGBM capacity to equal model A's, so the only
            # override left to carry is the round cap - applied exactly as it is for LightGBM.
            return replace(self, num_rounds=self.unmatched_num_rounds)
        return replace(
            self,
            params={
                **self.params,
                "num_leaves": self.unmatched_num_leaves,
                "min_data_in_leaf": self.unmatched_min_data_in_leaf,
            },
            num_rounds=self.unmatched_num_rounds,
        )

    def lgb_params(self) -> dict:
        p = dict(self.params, seed=self.seed)
        if not self.lgb_deterministic:
            # Both or neither. Only the two keys this dataclass owns are dropped: a config that sets its own
            # `force_col_wise` keeps it, because that is a choice the config made and not the default this
            # flag exists to undo.
            p.pop("deterministic", None)
            p.pop("force_row_wise", None)
        if self.use_linear_tree:
            p["linear_tree"] = True
            p["linear_lambda"] = self.linear_lambda
        return p


# ---------- label-derived tables: fitted on the training side only ----------


def reference_table(train_dep: pl.DataFrame, min_count: int) -> pl.DataFrame:
    """PRU-style unimpeded reference per airport x stand x runway (P10, P20, median, n)."""
    lab = train_dep.filter(pl.col(LABEL).is_not_null() & (pl.col(LABEL) > 0) & (pl.col(LABEL) < 7200))
    return (
        lab.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
        .agg(
            [
                pl.col(LABEL).quantile(0.10).alias("ref_p10"),
                pl.col(LABEL).quantile(0.20).alias("ref_p20"),
                pl.col(LABEL).median().alias("ref_med"),
                pl.len().alias("ref_n"),
            ]
        )
        .filter(pl.col("ref_n") >= min_count)
    )


def airport_runway_hour_table(train_dep: pl.DataFrame, min_count: int = 10) -> pl.DataFrame:
    lab = train_dep.filter(pl.col(LABEL).is_not_null())
    return (
        lab.group_by(["ADEP_mvt", "RUNWAY_mvt", "hour_local"])
        .agg([pl.col(LABEL).median().alias("arh_med"), pl.len().alias("arh_n")])
        .filter(pl.col("arh_n") >= min_count)
    )


def operator_offset_table(train_dep: pl.DataFrame, min_count: int = 20) -> pl.DataFrame:
    """Median clock offset per operator x departure airport, fitted on the training side of the fold only."""
    lab = train_dep.filter(pl.col(LABEL).is_not_null() & pl.col(PROXY).is_not_null())
    return (
        lab.with_columns((pl.col(LABEL) - pl.col(PROXY)).alias("_off"))
        .group_by(["AIRCRAFT_OPERATOR_flt", "ADEP_mvt"])
        .agg([pl.col("_off").median().alias("op_offset_med"), pl.len().alias("op_offset_n")])
        .filter(pl.col("op_offset_n") >= min_count)
    )


def airport_median_table(train_dep: pl.DataFrame) -> pl.DataFrame:
    lab = train_dep.filter(pl.col(LABEL).is_not_null())
    return lab.group_by("ADEP_mvt").agg(pl.col(LABEL).median().alias("apt_med"))


def fit_encodings(train_dep: pl.DataFrame, cfg: TrainConfig) -> dict[str, pl.DataFrame]:
    return {
        "ref": reference_table(train_dep, cfg.reference_min_count),
        "arh": airport_runway_hour_table(train_dep),
        "apt": airport_median_table(train_dep),
        "op": operator_offset_table(train_dep),
    }


def attach_encodings(df: pl.DataFrame, enc: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Join the fitted tables onto any frame. Pure join: nothing is derived from df's own labels."""
    out = df.join(enc["ref"], on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left", maintain_order="left")
    out = out.join(enc["arh"], on=["ADEP_mvt", "RUNWAY_mvt", "hour_local"], how="left", maintain_order="left")
    out = out.join(enc["apt"], on="ADEP_mvt", how="left", maintain_order="left")
    out = out.join(enc["op"], on=["AIRCRAFT_OPERATOR_flt", "ADEP_mvt"], how="left", maintain_order="left")
    return out.with_columns(
        [
            pl.coalesce([pl.col("ref_p10"), pl.col("arh_med"), pl.col("apt_med")]).alias("ref_p10_fb"),
            pl.col("ref_n").is_null().cast(pl.Int8).alias("ref_missing"),
        ]
    )


# ---------- fold-time candidate families (docs/article_fe_2026-09-24/plan.md) ----------

# Frequency-encoding keys, each counted within the departure airport.
FREQ_KEYS: dict[str, tuple[str, ...]] = {
    "freq_stand": ("STAND_mvt",),
    "freq_stand_rwy": ("STAND_mvt", "RUNWAY_mvt"),
    "freq_operator": ("AIRCRAFT_OPERATOR_flt",),
    "freq_actype": ("AIRCRAFT_TYPE_mvt",),
    "freq_ades": ("ADES_mvt",),
    "freq_rwy_hour": ("RUNWAY_mvt", "hour_local"),
}
_CELL = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
# Built from the NM off-block proxy, so null on every row model B sees; kept out of model B's column set.
LANE_A_ONLY_PREFIXES = ("agg_proxy_", "meta_prior_minus_proxy")
# Model B's exclusions: every column derived from an NM clock, which is null on its rows.
_NM_CLOCK_PREFIXES = ("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt")
# The meta prior must not read a label-derived column: those tables are fitted on every training row, including
# the rows the prior is predicting out of fold, and would carry their labels back in.
_LABEL_DERIVED_PREFIXES = ("ref_", "arh_", "apt_", "op_offset", "agg_")
META_ROUNDS = 300
META_NOISE_SEED = 20_260_924


def unmatched_columns(num_cols: list[str]) -> list[str]:
    """The numeric columns model B is given: everything except the NM-clock and proxy-derived ones."""
    return [c for c in num_cols if not c.startswith(_NM_CLOCK_PREFIXES + LANE_A_ONLY_PREFIXES)]


def _check_same_rows(before: pl.DataFrame, after: pl.DataFrame, what: str) -> None:
    if after.height != before.height or not after["MVT_ID_mvt"].equals(before["MVT_ID_mvt"]):
        raise AssertionError(f"{what} changed the row count or order")


def frequency_features(tr: pl.DataFrame, tg: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Share of the airport's departures that carry this row's key, counted over the training AND target rows.

    No label is read, so counting the target rows is legitimate: at scoring time the ranking rows' stands,
    operators and runways are all known. A share rather than a count because the pool is 12 months in CV and
    12 + 2 in production. A row whose key is null gets null (a null key is not a category)."""
    keys = sorted({"ADEP_mvt", *(k for ks in FREQ_KEYS.values() for k in ks)})
    both = pl.concat([tr.select(keys), tg.select(keys)], how="vertical_relaxed")
    apt_n = both.drop_nulls("ADEP_mvt").group_by("ADEP_mvt").len().rename({"len": "_apt_n"})
    out = []
    for f in (tr, tg):
        g = f
        for name, ks in FREQ_KEYS.items():
            k = ["ADEP_mvt", *ks]
            tab = (
                both.drop_nulls(k)
                .group_by(k)
                .len()
                .join(apt_n, on="ADEP_mvt")
                .with_columns((pl.col("len") / pl.col("_apt_n")).cast(pl.Float32).alias(name))
                .select([*k, name])
            )
            g = g.join(tab, on=k, how="left", maintain_order="left")
        _check_same_rows(f, g, "frequency_features")
        out.append(g)
    return out[0], out[1]


def aggdiff_features(tr: pl.DataFrame, tg: pl.DataFrame, min_count: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A group statistic minus the row's own value, and the group's label spread.

    Needs the encodings attached first (ref_med, ref_p10, arh_med). A tree approximates `a - b` only with a
    staircase of axis-parallel splits, and model A is data-limited, so handing it the difference is the one
    thing a group encoding alone cannot do. `agg_ref_iqr` is fitted on the training side's labels, with the same
    label window and minimum count as `reference_table`; `agg_proxy_cell_dev` reads no label and pools both
    sides, like the frequency encodings."""
    lab = tr.filter(pl.col(LABEL).is_not_null() & (pl.col(LABEL) > 0) & (pl.col(LABEL) < 7200))
    iqr = (
        lab.group_by(_CELL)
        .agg(
            (pl.col(LABEL).quantile(0.75) - pl.col(LABEL).quantile(0.25)).alias("agg_ref_iqr"),
            pl.len().alias("_n"),
        )
        .filter(pl.col("_n") >= min_count)
        .drop("_n")
    )
    both = pl.concat([tr.select([*_CELL, PROXY]), tg.select([*_CELL, PROXY])], how="vertical_relaxed")
    cell_proxy = (
        both.drop_nulls(PROXY)
        .group_by(_CELL)
        .agg(pl.col(PROXY).median().alias("_proxy_cell_med"), pl.len().alias("_n"))
        .filter(pl.col("_n") >= min_count)
        .drop("_n")
    )
    out = []
    for f in (tr, tg):
        g = f.join(iqr, on=_CELL, how="left", maintain_order="left").join(
            cell_proxy, on=_CELL, how="left", maintain_order="left"
        )
        g = g.with_columns(
            (pl.col(PROXY) - pl.col("ref_med")).alias("agg_proxy_minus_ref_med"),
            (pl.col(PROXY) - pl.col("ref_p10")).alias("agg_proxy_minus_ref_p10"),
            (pl.col(PROXY) - pl.col("arh_med")).alias("agg_proxy_minus_arh_med"),
            (pl.col(PROXY) - pl.col("_proxy_cell_med")).alias("agg_proxy_cell_dev"),
            (pl.col(SCHED_OFFSET) - pl.col("ref_med")).alias("agg_sched_minus_ref_med"),
            pl.col("agg_ref_iqr").cast(pl.Float64),
        ).drop("_proxy_cell_med")
        _check_same_rows(f, g, "aggdiff_features")
        out.append(g)
    return out[0], out[1]


def cyclic_features(df: pl.DataFrame) -> pl.DataFrame:
    """sin/cos of the local minute of day, so 23:59 and 00:00 are neighbours. None of the ten airports has a
    half-hour offset, so the UTC minute is the local minute."""
    minute = pl.col("hour_local").cast(pl.Float64) * 60.0 + pl.col("MVT_TIME_UTC_mvt").dt.minute().cast(pl.Float64)
    angle = minute * (2.0 * np.pi / 1440.0)
    return df.with_columns(angle.sin().alias("cyc_sin"), angle.cos().alias("cyc_cos"))


def noise_features(df: pl.DataFrame, n: int) -> pl.DataFrame:
    """`n` hashed-id columns that cannot carry signal: the price of adding n columns to the model."""
    return df.with_columns(
        [
            (pl.col("MVT_ID_mvt").hash(seed=META_NOISE_SEED + k) % 1_000_000).cast(pl.Float64).alias(f"noise_{k}")
            for k in range(1, n + 1)
        ]
    )


def meta_prior_features(
    tr: pl.DataFrame, tg: pl.DataFrame, cfg: TrainConfig, num_cols: list[str], cat_cols: list[str], info: dict
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """An out-of-fold taxi prior from a model that never sees an NM clock, fitted on every labelled row.

    Model B has 22k training rows; this prior is fitted on all ~2M, on model B's own columns, so what it adds
    is the matched rows' evidence about taxi times that model B cannot see. On the training side the prior is
    cross-fitted by calendar month mod 3, so a row's prior never comes from a model that saw its label; the
    target side gets the model fitted on every training row. Label-derived columns are excluded from its input
    (see _LABEL_DERIVED_PREFIXES) - with them in, the training-side prior would not be out of fold."""
    cols = [c for c in unmatched_columns(num_cols) if not c.startswith(_LABEL_DERIVED_PREFIXES)]
    stacked = pl.concat([tr.select(cols + cat_cols), tg.select(cols + cat_cols)], how="vertical_relaxed")
    X, names = _matrix(stacked, cols, cat_cols)
    del stacked
    ntr = tr.height
    y = tr[LABEL].to_numpy().astype(float)
    grp = tr["MVT_TIME_UTC_mvt"].dt.month().to_numpy() % 3
    cat_idx = [names.index(c) for c in cat_cols]
    params = dict(
        cfg.params,
        num_leaves=63,
        learning_rate=0.1,
        seed=cfg.seed,
        bagging_seed=cfg.seed,
        feature_fraction_seed=cfg.seed,
    )
    groups = np.unique(grp)
    info["meta_prior_input_cols"] = len(cols) + len(cat_cols)
    if len(groups) < 2:
        # one calendar-month group cannot be cross-fitted; a prior filled on one side only would break parity
        info["meta_prior_warning"] = "fewer than two month groups; prior left null on both sides"
        prior_tr, prior_tg = np.full(ntr, np.nan), np.full(tg.height, np.nan)
    else:
        prior_tr = np.full(ntr, np.nan)
        Xtr = X[:ntr]
        for gk in groups:
            m = grp == gk
            d = lgb.Dataset(Xtr[~m], label=y[~m], categorical_feature=cat_idx, params=params)
            booster = lgb.train(params, d, num_boost_round=META_ROUNDS, callbacks=[lgb.log_evaluation(0)])
            prior_tr[m] = booster.predict(Xtr[m])
            del d, booster
        d = lgb.Dataset(Xtr, label=y, categorical_feature=cat_idx, params=params)
        booster = lgb.train(params, d, num_boost_round=META_ROUNDS, callbacks=[lgb.log_evaluation(0)])
        prior_tg = booster.predict(X[ntr:])
        del d, booster, Xtr
    del X
    out = []
    for f, p in ((tr, prior_tr), (tg, prior_tg)):
        g = f.with_columns(pl.Series("meta_prior", p, dtype=pl.Float64)).with_columns(
            (pl.col("meta_prior") - pl.col(PROXY)).alias("meta_prior_minus_proxy")
        )
        out.append(g)
    return out[0], out[1]


@functools.lru_cache(maxsize=2)
def _weather_v2_table(mode: str) -> pl.DataFrame:
    """The station-hour table for a weather-v2 mode, loaded once per process from the on-disk caches."""
    from .weather import fetch_metar, fetch_openmeteo, wx_code_table

    table = wx_code_table(fetch_metar())
    if mode == "C":
        om = fetch_openmeteo()
        table = table.join(om, on=["station", "hour_utc"], how="full", coalesce=True)
    elif mode != "B":
        raise ValueError(f"fe_weather_v2 must be '', 'B' or 'C', got {mode!r}")
    return table


def weather_v2_features(df: pl.DataFrame, mode: str) -> pl.DataFrame:
    """Join the weather-v2 columns at the take-off hour. No label is read; the same table serves every fold."""
    from .weather import OM_COLS, WX_V2_COLS, join_hourly

    cols = WX_V2_COLS + (OM_COLS if mode == "C" else [])
    return join_hourly(df, _weather_v2_table(mode), cols)


@functools.lru_cache(maxsize=1)
def _osm_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    from .osm import fetch_osm_aeroway, fetch_runway_ends, stand_table

    return stand_table(fetch_osm_aeroway()), fetch_runway_ends()


def null_osm_at_airports(df: pl.DataFrame, airports) -> pl.DataFrame:
    """Set `osm_stand_rwy_km` to null on rows whose ADEP is in `airports`. No airports: the frame is returned as is."""
    if not airports:
        return df
    at = pl.col("ADEP_mvt").cast(pl.Utf8).is_in(list(airports)).fill_null(False)
    return df.with_columns(
        pl.when(at)
        .then(None)
        .otherwise(pl.col("osm_stand_rwy_km"))
        .cast(df.schema["osm_stand_rwy_km"])
        .alias("osm_stand_rwy_km")
    )


def osm_distance_features(df: pl.DataFrame, null_airports=()) -> pl.DataFrame:
    """Attach `osm_stand_rwy_km`. Label-free; the same two tables serve every fold and the ranking set.
    `null_airports` (TrainConfig.fe_osm_null_airports) blanks the column at those departure airports."""
    from .osm import add_stand_runway_distance

    stands, ends = _osm_tables()
    return null_osm_at_airports(add_stand_runway_distance(df, stands, ends), null_airports)


@functools.lru_cache(maxsize=1)
def _opdi_table() -> pl.DataFrame:
    from .opdi import build_turnaround_cache

    path = data_dir_work() / "opdi_turnaround.parquet"
    if not path.exists():
        build_turnaround_cache(out=path)
    return pl.read_parquet(path)


def data_dir_work():
    from .config import data_dir

    return data_dir() / "work"


def day_null_mask(df: pl.DataFrame, share: float) -> np.ndarray:
    """True on rows whose take-off date (UTC) falls in the `share` of days picked by CRC32 of 'YYYY-MM-DD'."""
    import zlib

    days = df["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").fill_null("").to_list()
    return np.array([zlib.crc32(d.encode()) / 2**32 < share for d in days], dtype=bool)


def opdi_features(df: pl.DataFrame, null_day_share: float = 0.0) -> pl.DataFrame:
    """Attach the OPDI turnaround columns by MVT_ID_mvt (the cache covers every training and ranking departure)."""
    from .opdi import OPDI_COLS

    out = df.join(_opdi_table(), on="MVT_ID_mvt", how="left", maintain_order="left")
    if out.height != df.height:
        raise AssertionError("opdi_features changed the row count")
    if null_day_share > 0:
        blank = pl.Series("_blank", day_null_mask(out, null_day_share))
        out = out.with_columns([pl.when(blank).then(None).otherwise(pl.col(c)).alias(c) for c in OPDI_COLS])
    return out


def unseen_stand_mask(df: pl.DataFrame, share: float) -> np.ndarray:
    """True for rows whose (airport, stand) is in the held-out share, by a CRC32 that no library version changes."""
    import zlib

    keys = (df["ADEP_mvt"].cast(pl.Utf8).fill_null("") + "|" + df["STAND_mvt"].cast(pl.Utf8).fill_null("")).to_list()
    u = np.array([zlib.crc32(k.encode()) / 2**32 for k in keys])
    return (u < share) & df["STAND_mvt"].is_not_null().to_numpy()


def add_fold_features(
    tr: pl.DataFrame, tg: pl.DataFrame, cfg: TrainConfig, info: dict
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The label-free and encoding-derived families; the meta prior is added separately, after the column set
    is known. Records which columns it added, because a silent no-op here would read as a null result."""
    before = set(tr.columns)
    if cfg.fe_freq:
        tr, tg = frequency_features(tr, tg)
    if cfg.fe_aggdiff:
        tr, tg = aggdiff_features(tr, tg, cfg.reference_min_count)
    if cfg.fe_cyclic:
        tr, tg = cyclic_features(tr), cyclic_features(tg)
    if cfg.fe_noise_cols:
        tr, tg = noise_features(tr, cfg.fe_noise_cols), noise_features(tg, cfg.fe_noise_cols)
    if cfg.fe_weather_v2:
        tr, tg = weather_v2_features(tr, cfg.fe_weather_v2), weather_v2_features(tg, cfg.fe_weather_v2)
    if cfg.fe_osm_dist:
        na = cfg.fe_osm_null_airports
        tr, tg = osm_distance_features(tr, na), osm_distance_features(tg, na)
    if cfg.fe_opdi:
        tr, tg = opdi_features(tr), opdi_features(tg, cfg.opdi_target_null_day_share)
    info["fold_feature_cols"] = [c for c in tr.columns if c not in before]
    return tr, tg


def visible_numeric_columns(frame: pl.DataFrame, cfg: TrainConfig) -> list[str]:
    """The numeric columns the model is actually given, by the one rule everything must agree on.

    This used to be decided inside predict_fold while feature_parity_report reimplemented part of it, and the
    two drifted: the parity gate reported on columns the model never sees. The same duplication is what let v6
    ship nine screening columns."""
    cols = [c for c in numeric_feature_columns(frame) if c not in (LABEL, "month")]
    hidden = set(CANDIDATE_COLS) - set(cfg.include_candidates) | set(cfg.exclude_features)
    if not cfg.use_encodings:
        hidden |= set(ENCODING_COLS)
    if not cfg.use_offset_encoding:
        hidden |= set(OFFSET_COLS)
    return [c for c in cols if c not in hidden]


# ---------- model B's target parameterisation ----------


def sched_reparam_shift(frame: pl.DataFrame, cfg: TrainConfig) -> np.ndarray:
    """Per-row offset model B's target is measured against: `mvt_minus_sched` on the configured airports, 0
    elsewhere. Subtracted when fitting, added back when predicting, so one function defines both directions.

    Zero - not "no shift" - is what makes the two directions a single code path: with the flag off, or at an
    airport the config does not name, the arithmetic is `y - 0.0` and `pred + 0.0`, which is the unshifted
    model exactly. A row whose `mvt_minus_sched` is null is also given 0: shifting it would put a NaN into the
    label vector, and LightGBM trains on that without complaining."""
    if not cfg.unmatched_sched_reparam:
        return np.zeros(frame.height)
    airports = list(cfg.unmatched_sched_reparam_airports)
    if not airports or SCHED_OFFSET not in frame.columns or "ADEP_mvt" not in frame.columns:
        return np.zeros(frame.height)
    shift = (
        pl.when(pl.col("ADEP_mvt").is_in(airports).fill_null(False))
        .then(pl.col(SCHED_OFFSET))
        .otherwise(0.0)
        .fill_null(0.0)
        .cast(pl.Float64)
        .alias("_shift")
    )
    return frame.select(shift)["_shift"].to_numpy().astype(float)


def unmatched_target(frame: pl.DataFrame, cfg: TrainConfig) -> np.ndarray:
    """Model B's target in model space. Identical to the label whenever the reparameterisation is off."""
    return frame[LABEL].to_numpy().astype(float) - sched_reparam_shift(frame, cfg)


# ---------- fitting ----------


def _matrix(df: pl.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Design matrix as float32. LightGBM bins the input anyway, so float32 costs no accuracy and halves the
    matrix: at 2.1M rows x 60 features that is the difference between 1.7 GB and 0.85 GB."""
    cols = feature_cols + cat_cols
    X = df.select(cols)
    for c in cat_cols:
        # The integer code a category gets must be a function of the data, not of how it was chunked.
        # `cast(pl.Categorical)` was used here and is NOT: on the real 17-chunk frame, six casts of the same
        # column give six different encodings - a pure relabelling, so the data is intact, but the codes are
        # permuted (STAND_mvt: all 2,085,047 rows move; ADEP_mvt: 1.7M). LightGBM splits a categorical feature
        # on its code values, so a permutation is a different model: replaying one fit with only the codes
        # relabelled moved early stopping from 937 to 1101 rounds, which is the whole distance between two
        # production refits that were supposed to be the same build (1252 and 1175). An explicit Enum over the
        # sorted levels makes the code deterministic. Train and target share this encoding because `_fit`
        # stacks them and calls this once.
        s = X[c].cast(pl.Utf8).fill_null("__NA__")
        X = X.with_columns(s.cast(pl.Enum(sorted(s.unique().to_list()))).to_physical().cast(pl.Float32).alias(c))
    return X.cast(pl.Float32).to_numpy(), cols


def _inner_holdout_mask(train: pl.DataFrame, months: int) -> np.ndarray:
    """Boolean mask of the latest `months` calendar months of the training side, used for early stopping.

    A mask rather than two frames: the caller indexes one design matrix with it instead of materialising the
    training rows a second time."""
    empty = np.zeros(train.height, dtype=bool)
    if months <= 0 or "MVT_TIME_UTC_mvt" not in train.columns:
        return empty
    keys = train.select(
        (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym")
    )["ym"]
    uniq = sorted(keys.unique().to_list())
    if len(uniq) <= months:
        return empty
    return keys.is_in(uniq[-months:]).to_numpy()


def _es_encoder(train_f: pl.DataFrame, cfg: TrainConfig, stats: list[dict]):
    """Factory for P2 Fix 1: re-encode a lane's inner early-stopping holdout from a clean map.

    Returns `build(hold_rows) -> pl.DataFrame`, where `hold_rows` is the already-encoded holdout block
    of one lane and the result is the same rows, same order, same schema, with the nine encoding
    columns (and the two hidden offset columns) rebuilt from `fit_encodings` over `train_f` MINUS the
    calendar months those rows occupy.

    Why a whole-month exclusion rather than a leave-one-row-out: the folds are month-blocked for the
    reason `cv.month_block_folds` gives, and the inner holdout is itself a month, so excluding the
    month is the smallest exclusion that makes the block clean. Nothing else is excluded - a holdout
    row still reads the other nine or eleven training months, which is what keeps the column
    informative rather than merely leak-free.

    `train_f` is the FOLD's whole training frame, both lanes, exactly the frame `predict_fold` fits the
    production encoder on. Model A's rows are not the only rows in these tables and dropping model B's
    would be a second, unmeasured change.

    Cached by month set, so model A and model B - whose holdout months coincide in every production
    fold - pay for one encoder build, not two. `stats` receives one record per build; it is the
    built-in no-op control, because a repair that silently changed nothing would otherwise be
    indistinguishable from a repair that worked."""
    keys = year_month_key(train_f)
    cache: dict[frozenset, dict[str, pl.DataFrame]] = {}
    encoded = [*ENCODING_COLS, *OFFSET_COLS]

    def build(hold_rows: pl.DataFrame) -> pl.DataFrame:
        months = sorted(set(year_month_key(hold_rows).unique().to_list()))
        key = frozenset(months)
        if key not in cache:
            base = train_f.filter(~keys.is_in(months))
            if base.height == 0:
                raise ValueError(f"no training rows left after excluding the holdout months {months}")
            cache[key] = fit_encodings(base, cfg)
        enc = cache[key]
        out = attach_encodings(hold_rows.drop([c for c in encoded if c in hold_rows.columns]), enc)
        out = out.select(hold_rows.columns)
        if out.height != hold_rows.height:
            raise AssertionError("the early-stopping encoder changed the holdout row count")
        changed = {}
        for c in ENCODING_COLS:
            if c not in hold_rows.columns:
                continue
            a, b = hold_rows[c], out[c]
            changed[c] = int(((a != b) | (a.is_null() != b.is_null())).fill_null(True).sum())
        stats.append(
            {
                "rows": hold_rows.height,
                "months_excluded": months,
                "train_rows_used": int(train_f.height - keys.is_in(months).sum()),
                "table_groups": {t: int(enc[t].height) for t in ("ref", "arh", "apt")},
                "cols_changed": changed,
            }
        )
        return out

    return build


def _fit(
    train: pl.DataFrame,
    target_expr,
    cfg: TrainConfig,
    feature_cols: list[str],
    cat_cols: list[str],
    predict_frames: list[pl.DataFrame],
    es_rows=None,
) -> tuple[list[np.ndarray], int]:
    """Early-stop on an inner holdout of `train`, refit on all of `train`, then predict each frame.

    `target_expr(frame) -> np.ndarray` builds the model-space target, so the caller decides the parameterisation.
    Returns one prediction array per frame (in model space) and the number of rounds actually used.

    `es_rows` (P2 Fix 1) optionally re-encodes the inner-holdout block: `_es_encoder`'s `build`, taking the
    holdout rows and returning them with encodings fitted without their own months. The re-encoded block is
    appended to the SAME stacked frame rather than matrixed on its own, because `_matrix` builds each
    categorical's Enum over the stacked frame's own levels - a separately built matrix would carry a
    different integer coding and be a different model. The appended rows are a subset of `train`, so the
    level set and therefore every existing row's code are unchanged, and `Xall[a:b]` is byte-identical to
    what it was with `es_rows=None`. Stage 2 never sees the block.

    With `cfg.model == "catboost"` the whole call goes to `_fit_catboost`, which keeps this contract item for
    item; nothing above this function knows which family ran."""
    if cfg.model == "catboost":
        return _fit_catboost(train, target_expr, cfg, feature_cols, cat_cols, predict_frames, es_rows=es_rows)
    hold = _inner_holdout_mask(train, cfg.inner_holdout_months)
    n_hold = int(hold.sum())
    es_block = None
    if es_rows is not None and 0 < n_hold < train.height:
        es_block = es_rows(train.filter(pl.Series("_es_hold", hold)))
    frames = [train, *predict_frames] + ([es_block] if es_block is not None else [])
    n_pred = len(predict_frames)
    stacked = pl.concat([f.select(feature_cols + cat_cols) for f in frames], how="vertical_relaxed")
    Xall, cols = _matrix(stacked, feature_cols, cat_cols)
    del stacked
    offs, i = [], 0
    for f in frames:
        offs.append((i, i + f.height))
        i += f.height
    cat_idx = [cols.index(c) for c in cat_cols]
    params = cfg.lgb_params()

    # target_expr is elementwise, so evaluate it once over the training block and index the result rather than
    # filtering the frame twice (each filter copies every column of a multi-million-row frame).
    a, b = offs[0]
    y = target_expr(train)
    rounds = cfg.num_rounds
    if 0 < n_hold < train.height:
        Xtrain = Xall[a:b]
        # A view into the stack when Fix 1 is on, the fancy-indexed copy it always was when it is off.
        Xhold = Xall[offs[-1][0] : offs[-1][1]] if es_block is not None else Xtrain[hold]
        dtr = lgb.Dataset(Xtrain[~hold], label=y[~hold], categorical_feature=cat_idx, params=params)
        dva = lgb.Dataset(Xhold, label=y[hold], categorical_feature=cat_idx, reference=dtr)
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=cfg.num_rounds,
            valid_sets=[dva],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(cfg.early_stopping, verbose=False)],
        )
        best = int(booster.best_iteration or cfg.num_rounds)
        # scale the round count to the larger refit set, then cap
        rounds = int(min(cfg.num_rounds, max(10, round(best * train.height / max(1, train.height - n_hold)))))
        del dtr, dva, booster, Xtrain, Xhold  # the fancy-indexed copies are the peak of this function

    dfull = lgb.Dataset(Xall[a:b], label=y, categorical_feature=cat_idx, free_raw_data=False, params=params)
    # Refit under each seed and average. One seed is the default, and then this is the single fit it was before.
    acc = [np.zeros(e - s) for (s, e) in offs[1 : 1 + n_pred]]
    for k in range(max(1, cfg.n_seeds)):
        p_k = dict(params, seed=cfg.seed + k, bagging_seed=cfg.seed + k, feature_fraction_seed=cfg.seed + k)
        model = lgb.train(p_k, dfull, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
        for i, (s_, e_) in enumerate(offs[1 : 1 + n_pred]):
            acc[i] += model.predict(Xall[s_:e_])
    preds = [a_ / max(1, cfg.n_seeds) for a_ in acc]
    return preds, rounds


# ---------- the CatBoost fitter (P10) ----------

# What the most recent `_fit_catboost` call resolved, for `predict_fold` to copy into the record: the library's
# own resolved values of the settings P10 §8 says must be logged rather than assumed (`max_ctr_complexity` is
# the one whose library default moves with the iteration count), the early-stopping best count and the round
# count used. Overwritten by every call; read immediately after one.
_CB_LAST: dict = {}
_CB_LOGGED = (
    "max_ctr_complexity",
    "learning_rate",
    "depth",
    "one_hot_max_size",
    "has_time",
    "boosting_type",
    "grow_policy",
    "l2_leaf_reg",
    "border_count",
    "bootstrap_type",
    "subsample",
    "nan_mode",
)


def _cb_matrix(df: pl.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """CatBoost's two blocks: the numeric columns as C-ordered float32, the categoricals as an object array.

    `_matrix` cannot be reused: it codes the categoricals as float32, and CatBoost rejects a real-valued
    categorical outright (P10 §1.1 row 4). The numeric block is the same float32 matrix `_matrix` builds for
    the same columns, so the two families read identical numeric values, NaNs included.

    Each categorical is handed over as its level STRING - the same `Utf8 -> fill_null("__NA__")` values
    `_matrix` codes - so the level set, the null level and the level identities are the ones LightGBM sees.
    CatBoost hashes the value, so what reaches it is a function of the value alone and cannot depend on how
    the frame was chunked: the G11 failure (`cast(pl.Categorical)` coding the same column six ways) has no way
    in. The strings are shared objects from a per-column lookup table indexed by the same sorted-Enum code
    `_matrix` uses, so the block costs eight bytes a cell rather than a Python string each."""
    Xnum = np.ascontiguousarray(df.select(feature_cols).cast(pl.Float32).to_numpy(), dtype=np.float32)
    Xcat = np.empty((df.height, len(cat_cols)), dtype=object)
    for j, c in enumerate(cat_cols):
        s = df[c].cast(pl.Utf8).fill_null("__NA__")
        levels = sorted(s.unique().to_list())
        codes = s.cast(pl.Enum(levels)).to_physical().to_numpy()
        Xcat[:, j] = np.array([lv.encode() for lv in levels], dtype=object)[codes]
    return Xnum, Xcat


def _cb_pool(Xnum: np.ndarray, Xcat: np.ndarray, feature_cols: list[str], cat_cols: list[str], label=None):
    """A CatBoost Pool. `label=None` for every frame that is predicted: a predict pool never carries a label,
    so there is no route by which a target row's label can reach its own categorical statistics."""
    from catboost import FeaturesData, Pool

    fd = FeaturesData(
        num_feature_data=np.ascontiguousarray(Xnum, dtype=np.float32),
        cat_feature_data=np.ascontiguousarray(Xcat) if cat_cols else None,
        num_feature_names=list(feature_cols),
        cat_feature_names=list(cat_cols) if cat_cols else None,
    )
    return Pool(fd, label=None if label is None else np.asarray(label, dtype=float))


def _fit_catboost(
    train: pl.DataFrame,
    target_expr,
    cfg: TrainConfig,
    feature_cols: list[str],
    cat_cols: list[str],
    predict_frames: list[pl.DataFrame],
    es_rows=None,
) -> tuple[list[np.ndarray], int]:
    """`_fit` with CatBoost as the booster. Same signature, same return, and the same six contract items
    (P10 §1.3), each done the way `_fit` does it:

    1. early stopping reads the inner holdout of `train` only (`_inner_holdout_mask`, and the P2 Fix 1 block
       when `es_rows` is given), never the evaluation fold; the final model is refitted on all of `train` at a
       fixed round count;
    2. that round count is `_fit`'s formula verbatim - best count scaled by n_train / (n_train - n_hold),
       floored at 10, capped at num_rounds. CatBoost's `get_best_iteration()` is 0-based, so the best COUNT is
       it plus one, which is what LightGBM's `best_iteration` already is;
    3. train and every predict frame are stacked before the categoricals are coded, so all share one level set;
    4. `n_seeds` refits at `random_seed = seed + k`, predictions averaged;
    5. predictions are returned per predict frame, in model space;
    6. the refit Pool is built once and reused across seeds.

    Where the categorical statistics come from, since that is exactly how this project's two leaks happened:
    CatBoost's ordered target statistics are computed on the learn Pool only, which is the training side of
    the fold (or its non-holdout part during early stopping). The eval Pool's labels are used to pick the
    stopping round and for nothing else, and every predict Pool is built with no label at all. At apply time a
    row's statistics come from the fitted model's tables, not from the other rows it is predicted with -
    tests/test_catboost_family.py checks that a target frame's labels, and the rows it is batched with, cannot
    move a prediction."""
    from catboost import CatBoostRegressor

    hold = _inner_holdout_mask(train, cfg.inner_holdout_months)
    n_hold = int(hold.sum())
    es_block = None
    if es_rows is not None and 0 < n_hold < train.height:
        es_block = es_rows(train.filter(pl.Series("_es_hold", hold)))
    frames = [train, *predict_frames] + ([es_block] if es_block is not None else [])
    n_pred = len(predict_frames)
    stacked = pl.concat([f.select(feature_cols + cat_cols) for f in frames], how="vertical_relaxed")
    Xnum, Xcat = _cb_matrix(stacked, feature_cols, cat_cols)
    del stacked
    offs, i = [], 0
    for f in frames:
        offs.append((i, i + f.height))
        i += f.height
    params = cfg.catboost_params()

    a, b = offs[0]
    y = target_expr(train)
    rounds = cfg.num_rounds
    if 0 < n_hold < train.height:
        if es_block is not None:
            s_, e_ = offs[-1]
            dva = _cb_pool(Xnum[s_:e_], Xcat[s_:e_], feature_cols, cat_cols, y[hold])
        else:
            dva = _cb_pool(Xnum[a:b][hold], Xcat[a:b][hold], feature_cols, cat_cols, y[hold])
        dtr = _cb_pool(Xnum[a:b][~hold], Xcat[a:b][~hold], feature_cols, cat_cols, y[~hold])
        m = CatBoostRegressor(
            **params,
            iterations=cfg.num_rounds,
            random_seed=cfg.seed,
            od_type="Iter",
            od_wait=cfg.early_stopping,
            use_best_model=False,
        )
        m.fit(dtr, eval_set=dva)
        best_it = m.get_best_iteration()
        best = int(best_it + 1) if best_it is not None else cfg.num_rounds
        rounds = int(min(cfg.num_rounds, max(10, round(best * train.height / max(1, train.height - n_hold)))))
        del dtr, dva, m

    else:
        best = None
    dfull = _cb_pool(Xnum[a:b], Xcat[a:b], feature_cols, cat_cols, y)
    dpred = [_cb_pool(Xnum[s_:e_], Xcat[s_:e_], feature_cols, cat_cols) for (s_, e_) in offs[1 : 1 + n_pred]]
    # The pools hold their own copies; the stacked blocks are the largest thing left and nothing reads them now.
    del Xnum, Xcat
    acc = [np.zeros(p.num_row()) for p in dpred]
    for k in range(max(1, cfg.n_seeds)):
        model = CatBoostRegressor(**params, iterations=rounds, random_seed=cfg.seed + k)
        model.fit(dfull)
        for j, p in enumerate(dpred):
            acc[j] += model.predict(p)
        if k == 0:
            resolved = model.get_all_params()
            _CB_LAST.clear()
            _CB_LAST.update(
                {
                    # get_all_params() omits thread_count; it is read from what the estimator was given.
                    "resolved": {
                        **{key: resolved.get(key) for key in _CB_LOGGED},
                        "thread_count": model.get_params().get("thread_count"),
                    },
                    "es_best_count": best,
                    "rounds": rounds,
                    "tree_count": int(model.tree_count_),
                    "n_cat_features": len(cat_cols),
                    "n_num_features": len(feature_cols),
                }
            )
        del model
    preds = [a_ / max(1, cfg.n_seeds) for a_ in acc]
    return preds, rounds


# ---------- per-airport specialists: grouping and the grouped fit ----------


def _row_uniform(ids: np.ndarray, seed: int) -> np.ndarray:
    """A uniform [0, 1) draw per row that is a pure function of that row's id and `seed`.

    The murmur3 finaliser, written out in uint64 arithmetic rather than taken from `polars.hash` or
    `numpy.random`: a library hash is free to change between versions, and a control arm whose group
    membership moved between two runs would be indistinguishable from the effect it is controlling for.
    Frame-independent on purpose - a row's group must not depend on which other rows share its frame,
    or the training side and the target side could disagree about where a row belongs."""
    v = np.asarray(ids, dtype=np.float64)
    if v.size and not (np.isfinite(v).all() and np.array_equal(v, np.floor(v))):
        raise ValueError("row ids must be finite integers to derive a reproducible group")
    with np.errstate(over="ignore"):
        x = v.astype(np.int64).astype(np.uint64) + np.uint64(seed)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xFF51AFD7ED558CCD)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xC4CEB9FE1A85EC53)
        x ^= x >> np.uint64(33)
    return (x >> np.uint64(11)).astype(np.float64) / 9007199254740992.0


def _group_labels(
    train: pl.DataFrame, target: pl.DataFrame, cfg: TrainConfig
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Group label per row for both frames, plus the sorted list of groups that may be fitted.

    The group list comes from the TRAINING side and is sorted, so the fitting loop's order is a function of
    the data and nothing else. A target row whose group is not in that list (an airport with no training
    rows) keeps the global prediction."""
    if cfg.airport_group_key == "adep":

        def labels(frame: pl.DataFrame) -> np.ndarray:
            return frame["ADEP_mvt"].cast(pl.Utf8).fill_null("__NA__").to_numpy().astype(object)

        tr = labels(train)
        return tr, labels(target), sorted(set(tr.tolist()))
    if cfg.airport_group_key != "shuffled":
        raise ValueError(f"airport_group_key must be one of {GROUP_KEYS}, got {cfg.airport_group_key!r}")
    # The control copies the candidate's group COUNT and size profile from the training side's airport
    # partition, so the only thing it does not carry is which rows belong together.
    counts = (
        train.select(pl.col("ADEP_mvt").cast(pl.Utf8).fill_null("__NA__")).group_by("ADEP_mvt").len().sort("ADEP_mvt")
    )
    names = [f"G{i:02d}" for i in range(counts.height)]
    edges = np.cumsum(counts["len"].to_numpy().astype(float)) / max(1, train.height)
    if len(edges):
        edges[-1] = 1.0
    arr = np.array(names, dtype=object)

    def assign(frame: pl.DataFrame) -> np.ndarray:
        u = _row_uniform(frame["MVT_ID_mvt"].to_numpy(), cfg.airport_group_seed)
        return arr[np.clip(np.searchsorted(edges, u, side="right"), 0, max(0, len(names) - 1))]

    return assign(train), assign(target), list(names)


def _fit_grouped(
    train: pl.DataFrame,
    target: pl.DataFrame,
    target_expr,
    cfg: TrainConfig,
    feature_cols: list[str],
    cat_cols: list[str],
    info: dict,
    es_rows=None,
) -> np.ndarray:
    """One global fit plus one fit per group, combined as `(1-w)*global + w*group` on that group's rows.

    Returns model-space predictions aligned to `target`'s rows, so the caller's add-back, floor and bound
    are untouched: the blend happens BEFORE `_assemble`, which is what makes the pure-specialist arm
    recoverable as `2*blend - global` from the base arm's OOF without fitting it a second time.

    The global fit is first and the groups follow one at a time, each frame filtered inside the loop and
    dropped at the end of it: peak memory is the global fit's design matrix, exactly as before, and each
    group's is about a tenth of it. Two runs of this must agree byte for byte - see
    `configs/determinism_check_airport.yaml`."""
    w = float(cfg.airport_blend_weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"airport_blend_weight must be in [0, 1], got {w}")
    (g_glob,), rounds = _fit(train, target_expr, cfg, feature_cols, cat_cols, [target], es_rows=es_rows)
    info["rounds_matched"] = rounds
    tr_lab, tg_lab, groups = _group_labels(train, target, cfg)
    if len(groups) < 2:
        raise ValueError(f"airport_models needs at least two training groups, got {groups}")
    out = np.array(g_glob, dtype=float, copy=True)
    per: dict[str, dict] = {}
    blended = 0
    for name in groups:
        tr_m, tg_m = tr_lab == name, tg_lab == name
        n_tr, n_tg = int(tr_m.sum()), int(tg_m.sum())
        rec = {"train_rows": n_tr, "target_rows": n_tg, "fitted": False, "rounds": 0}
        if w > 0.0 and n_tg > 0 and n_tr >= cfg.airport_min_group_rows:
            sub_tr = train.filter(pl.Series("_g", tr_m))
            sub_tg = target.filter(pl.Series("_g", tg_m))
            (g_loc,), r_g = _fit(sub_tr, target_expr, cfg, feature_cols, cat_cols, [sub_tg], es_rows=es_rows)
            out[tg_m] = (1.0 - w) * out[tg_m] + w * g_loc
            rec.update(fitted=True, rounds=r_g)
            blended += n_tg
            del sub_tr, sub_tg, g_loc
        per[name] = rec
    # Everything a silent no-op would hide. This repo has twice shipped a change that turned out to be one,
    # and both times the only symptom was several recipes returning the same number to six figures.
    info["airport_group_key"] = cfg.airport_group_key
    info["airport_blend_weight"] = w
    info["airport_n_groups_fitted"] = sum(1 for r in per.values() if r["fitted"])
    info["airport_blended_rows"] = blended
    info["airport_ungrouped_target_rows"] = int(target.height - sum(r["target_rows"] for r in per.values()))
    info["airport_groups"] = per
    return out


def predict_fold(train_f: pl.DataFrame, target_f: pl.DataFrame, cfg: TrainConfig) -> tuple[pl.DataFrame, dict]:
    """Fit on train_f and predict target_f. Returns (DataFrame[MVT_ID_mvt, pred], info).

    target_f may be unlabelled (the ranking frame); the model is identical either way."""
    n_before = train_f.height
    train_f = train_f.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    info: dict = {"train_rows": train_f.height, "dropped_unlabelled": n_before - train_f.height}
    if cfg.model != "lightgbm":
        # Only when it is not the default, so a LightGBM record is byte-for-byte what it was before P10.
        info["model"] = cfg.model
    if cfg.sim_unseen_stands > 0:
        held = unseen_stand_mask(train_f, cfg.sim_unseen_stands)
        info["sim_unseen_stands_dropped_train_rows"] = int(held.sum())
        train_f = train_f.filter(pl.Series("_keep", ~held))
    enc = fit_encodings(train_f, cfg)
    tr, tg = attach_encodings(train_f, enc), attach_encodings(target_f, enc)
    if tg.height != target_f.height or not tg["MVT_ID_mvt"].equals(target_f["MVT_ID_mvt"]):
        raise AssertionError("attach_encodings changed row count or order")
    if (
        cfg.fe_freq
        or cfg.fe_aggdiff
        or cfg.fe_cyclic
        or cfg.fe_noise_cols
        or cfg.fe_weather_v2
        or cfg.fe_osm_dist
        or cfg.fe_opdi
    ):
        tr, tg = add_fold_features(tr, tg, cfg, info)

    # P2 Fix 1. One factory per lane so the two lanes' records stay separable in `info`; the month-keyed
    # cache lives inside each factory, and the encoder build is shared within a lane, not across them.
    info["es_holdout_encodings"] = bool(cfg.es_holdout_encodings)
    es_stats_a: list[dict] = []
    es_stats_b: list[dict] = []
    es_a = _es_encoder(train_f, cfg, es_stats_a) if cfg.es_holdout_encodings else None
    es_b = _es_encoder(train_f, cfg, es_stats_b) if cfg.es_holdout_encodings else None

    num_cols = visible_numeric_columns(tr, cfg)
    cat_cols = (
        [c for c in CATEGORICALS if c in tr.columns and c not in set(cfg.exclude_features)]
        if cfg.use_categoricals
        else []
    )
    if cfg.fe_meta_prior:
        tr, tg = meta_prior_features(tr, tg, cfg, num_cols, cat_cols, info)
        info.setdefault("fold_feature_cols", [])
        info["fold_feature_cols"] += ["meta_prior", "meta_prior_minus_proxy"]
        num_cols = visible_numeric_columns(tr, cfg)
    parts = []
    m_tr, m_tg = tr.filter(pl.col(PROXY).is_not_null()), tg.filter(pl.col(PROXY).is_not_null())
    if m_tg.height:
        if m_tr.height >= 200:
            cols_a = [c for c in num_cols if c != PROXY] + [PROXY]

            def _target_a(f: pl.DataFrame) -> np.ndarray:
                return f[LABEL].to_numpy().astype(float) - f[PROXY].to_numpy().astype(float)

            # With airport_models off this is the single `_fit` call it has always been; the grouped branch
            # is the only thing the flag adds, and it sets rounds_matched itself.
            if cfg.airport_models:
                g = _fit_grouped(m_tr, m_tg, _target_a, cfg, cols_a, cat_cols, info, es_rows=es_a)
            else:
                (g,), rounds = _fit(m_tr, _target_a, cfg, cols_a, cat_cols, [m_tg], es_rows=es_a)
                info["rounds_matched"] = rounds
            if cfg.model == "catboost":
                info["cb_matched"] = dict(_CB_LAST)
        else:
            info["rounds_matched"] = 0
            info["warning_matched"] = f"only {m_tr.height} labelled matched rows; predicting the proxy alone"
            g = np.zeros(m_tg.height)
        parts.append(pl.DataFrame({"MVT_ID_mvt": m_tg["MVT_ID_mvt"], "pred": m_tg[PROXY].to_numpy().astype(float) + g}))

    u_tr, u_tg = tr.filter(pl.col(PROXY).is_null()), tg.filter(pl.col(PROXY).is_null())
    if u_tg.height:
        cols_b = unmatched_columns(num_cols)
        # Model A's and model B's visible column counts have differed by one in past records without the
        # record saying which, so log both rather than inferring them from the config.
        info["n_features_unmatched"] = len(cols_b)
        if u_tr.height >= 200:
            cfg_b = cfg.for_unmatched()
            (g,), rounds = _fit(
                u_tr, lambda f: unmatched_target(f, cfg_b), cfg_b, cols_b, cat_cols, [u_tg], es_rows=es_b
            )
            # Back to seconds of taxi time. The fit above ran on `LABEL - shift`, so the model's output is in
            # that space and nothing downstream (_assemble's clip, the metrics, the submission) knows it.
            shift_tg = sched_reparam_shift(u_tg, cfg_b)
            g = g + shift_tg
            info["rounds_unmatched"] = rounds
            if cfg.model == "catboost":
                info["cb_unmatched"] = dict(_CB_LAST)
            if cfg_b.unmatched_sched_reparam:
                info["sched_reparam_train_rows"] = int((sched_reparam_shift(u_tr, cfg_b) != 0.0).sum())
                info["sched_reparam_target_rows"] = int((shift_tg != 0.0).sum())
        else:
            fallback = float(np.nanmedian(tr[LABEL].to_numpy().astype(float)))
            g = u_tg["ref_p10_fb"].fill_null(fallback).to_numpy().astype(float)
            info["rounds_unmatched"] = 0
        parts.append(pl.DataFrame({"MVT_ID_mvt": u_tg["MVT_ID_mvt"], "pred": g}))

    # Everything a silent no-op would hide. An empty record with the flag on means the holdout was never
    # re-encoded; a record whose `cols_changed` are all zero means it was re-encoded to the same values.
    if cfg.es_holdout_encodings:
        info["es_encoder_matched"] = es_stats_a
        info["es_encoder_unmatched"] = es_stats_b
    return _assemble(parts, tr, tg, target_f, info, cfg)


def _lobt_bound(out: pl.DataFrame, cfg: TrainConfig, info: dict) -> pl.DataFrame:
    """Project `pred` onto the window the LOBT clock admits, on the rows that have one.

    `y = MVT - BLOCK` and `|BLOCK - LOBT| <= slack` put the truth inside `mvt_minus_lobt +- slack`, so this
    is the metric projection onto a convex set containing the truth: `|clip(p) - y| <= |p - y|` row by row,
    with equality where it does not bind. The lower edge is raised to 0 so the existing zero floor stays in
    force - `max(m - slack, 0)` can sit above the truth only on rows whose label is negative, which the floor
    already overrode. Rows with no NM record have no window and are returned untouched."""
    slack = float(cfg.lobt_bound_slack)
    if not np.isfinite(slack) or slack <= 0.0:
        raise ValueError(f"lobt_bound_slack must be a positive finite number, got {slack!r}")
    m = pl.col(LOBT_OFFSET)
    lo, hi = pl.max_horizontal(m - slack, pl.lit(0.0)), m + slack
    # An inverted window means MVT precedes LOBT by more than the slack, which the zero floor and the bound
    # cannot both satisfy. It does not occur on either year's data; a silent clip with inverted bounds would
    # be a submission-destroying bug, so refuse rather than guess which edge wins.
    inverted = int(out.select((m.is_not_null() & (lo > hi)).sum()).item())
    if inverted:
        raise AssertionError(f"lobt bound is inverted (lo > hi) on {inverted} rows")
    before = out["pred"]
    out = out.with_columns(
        pl.when(m.is_null())
        .then(pl.col("pred"))
        .otherwise(pl.min_horizontal(pl.max_horizontal(pl.col("pred"), lo), hi))
        .alias("pred")
    )
    move = out["pred"] - before
    info["lobt_bound_slack"] = slack
    info["lobt_bound_rows"] = int((move.abs() > 1e-9).sum())
    info["lobt_bound_up"] = int((move > 1e-9).sum())
    info["lobt_bound_down"] = int((move < -1e-9).sum())
    info["lobt_bound_max_move"] = float(move.abs().max() or 0.0)
    return out


def _assemble(
    parts: list[pl.DataFrame],
    tr: pl.DataFrame,
    tg: pl.DataFrame,
    target_f: pl.DataFrame,
    info: dict,
    cfg: TrainConfig | None = None,
):
    pred = pl.concat(parts) if parts else pl.DataFrame({"MVT_ID_mvt": tg["MVT_ID_mvt"].head(0), "pred": []})
    fallback = float(np.nanmedian(tr[LABEL].to_numpy().astype(float))) if tr.height else 0.0
    bound = bool(cfg is not None and cfg.lobt_bound)
    if bound and LOBT_OFFSET not in target_f.columns:
        raise ValueError(f"lobt_bound is on but {LOBT_OFFSET} is not in the target frame")
    out = (
        target_f.select(["MVT_ID_mvt", LOBT_OFFSET] if bound else ["MVT_ID_mvt"])
        .join(pred, on="MVT_ID_mvt", how="left", maintain_order="left")
        .with_columns(pl.col("pred").fill_null(fallback).fill_nan(fallback).clip(lower_bound=0.0))
    )
    if bound:
        out = _lobt_bound(out, cfg, info)
    out = out.select(["MVT_ID_mvt", "pred"])
    if out.height != target_f.height:
        raise AssertionError("prediction join changed the row count")
    return out, info


def feature_parity_report(train_f: pl.DataFrame, target_f: pl.DataFrame, cfg: TrainConfig) -> pl.DataFrame:
    """Null-rate of every model feature on the training side vs the target side.

    A feature that is dense in training and empty at prediction time (or vice versa) sends every row down one
    branch of the tree; docs/README records a team losing a submission to exactly that."""
    enc = fit_encodings(train_f.filter(pl.col(LABEL).is_not_null()), cfg)
    tr, tg = attach_encodings(train_f, enc), attach_encodings(target_f, enc)
    cols = visible_numeric_columns(tr, cfg)
    rows = [
        {
            "feature": c,
            "null_rate_train": tr[c].null_count() / max(1, tr.height),
            "null_rate_target": tg[c].null_count() / max(1, tg.height),
        }
        for c in cols
    ]
    return (
        pl.DataFrame(rows)
        .with_columns((pl.col("null_rate_target") - pl.col("null_rate_train")).abs().alias("gap"))
        .sort("gap", descending=True)
    )


def _lane_metrics(joined: pl.DataFrame, y: np.ndarray, pv: np.ndarray) -> dict:
    """Split a fold's predictions by the routing key and report each lane on its own.

    The lanes are the two models: rows with an NM off-block time go to model A, rows without go to model B.
    Model B is about 1.2% of rows and roughly half the squared error, so a pooled RMSE moves by a few tenths
    when model B moves by hundreds, and a pooled comparison cannot say which one a change touched."""
    unmatched = joined[PROXY].is_null().to_numpy()
    apt = joined["ADEP_mvt"].to_numpy()
    out = {}
    for name, mask in (("matched", ~unmatched), ("unmatched", unmatched)):
        if not mask.any():
            out[name] = {"n": 0}
            continue
        out[name] = {**metrics(y[mask], pv[mask]), "by_airport": per_airport(y[mask], pv[mask], apt[mask])}
    return out


def run_cv(
    feats: pl.DataFrame,
    cfg: TrainConfig,
    out_dir: Path = Path("experiments"),
    log_name: str = "experiments.jsonl",
) -> dict:
    """Month-blocked CV on a labelled departure feature frame. Writes OOF predictions and a JSONL record.

    log_name exists so a synthetic run cannot land in the real-data experiment log: its RMSE level is an
    artefact of the generator, and mixing the two would corrupt the CV-to-LB correspondence we track.

    `overall` pools one prediction per row, so a fold whose rows an earlier fold already predicted contributes
    nothing to it. That is unchanged here; what is new is that the record says so: `overall.folds_pooled`,
    `overall.folds_shadowed` and each fold's `n_into_overall`. Read `overall` as the pooled OOF of
    `folds_pooled` alone - with the production fold list that is {jan, jul, forward} and not `jul_fwd`."""
    t0 = time.time()
    months = year_month_key(feats)
    folds: list[Fold] = month_block_folds(months.unique().to_list())
    if cfg.folds:
        wanted = set(cfg.folds)
        folds = [f for f in folds if f.name in wanted]
        if not folds:
            raise ValueError(f"cfg.folds={cfg.folds} matched none of the available folds")
    oof = pl.DataFrame({"MVT_ID_mvt": feats["MVT_ID_mvt"]}).with_columns(pl.lit(None, dtype=pl.Float64).alias("oof"))
    fold_metrics: dict[str, dict] = {}
    pooled: list[str] = []  # folds whose rows reach `overall`
    shadowed: list[str] = []  # folds an earlier fold had already predicted, so `overall` never saw them
    # Which fold predicted which row, long: one entry per (id, fold) actually predicted, so the rows `overall`
    # coalesced away are still recoverable. This goes to a SIDECAR parquet rather than a column in the OOF,
    # for two reasons. The wide file stays byte-identical, so every archived record and every `make compare`
    # against one keeps its meaning. And `compare.load_pair` joins two OOF frames, where polars silently
    # renames a second `fold` column to `fold_right` - a column would have been a quiet corruption of exactly
    # the tool that reads these files.
    fold_of: list[pl.DataFrame] = []
    for f in folds:
        tr_f = feats.filter(months.is_in(list(f.train_months)))
        va_f = feats.filter(months.is_in(list(f.valid_months)))
        p, info = predict_fold(tr_f, va_f, cfg)
        joined = va_f.select(["MVT_ID_mvt", LABEL, "ADEP_mvt", PROXY]).join(p, on="MVT_ID_mvt", how="inner")
        y, pv = joined[LABEL].to_numpy().astype(float), joined["pred"].to_numpy().astype(float)
        fold_metrics[f.name] = {
            **metrics(y, pv),
            **info,
            "by_airport": per_airport(y, pv, joined["ADEP_mvt"].to_numpy()),
            # The two lanes carry very different error and are routed to different models, so a pooled
            # number hides which one moved. Always report both, never one alone.
            **_lane_metrics(joined, y, pv),
        }
        # A four-fold run is ~45 min and used to print nothing until it finished, so a run that died partway
        # was indistinguishable from one still working. Print each fold as it lands, with both lanes.
        lm, lu = fold_metrics[f.name]["matched"], fold_metrics[f.name]["unmatched"]
        print(
            f"fold {f.name}: n={fold_metrics[f.name]['n']:,} rmse={fold_metrics[f.name]['rmse_full']:.2f}"
            f"  matched {lm.get('rmse_full', float('nan')):.2f} (n={lm['n']:,})"
            f"  unmatched {lu.get('rmse_full', float('nan')):.2f} (n={lu['n']:,})"
            f"  [{round(time.time() - t0)}s]",
            flush=True,
        )
        # Two folds can validate the same months - `jul` and `jul_fwd` both predict all of July, differing only
        # in how much history they trained on - and `pl.coalesce` keeps whichever came first in fold order
        # [jan, jul, jul_fwd, forward]. So `jul_fwd`'s 190,713 predictions were silently discarded from
        # `overall` while its rows still counted in the fold table: overall.n 672,428 against a fold sum of
        # 863,141, and overall.rmse_full equal to pooled{jan, jul, forward} in all six four-fold records
        # (docs/investigation_2026-09-16/results/fold-shape-vs-ranking.md §3.3). The pooling is kept as it was -
        # the logged numbers stay comparable - but which folds it used is now written down instead of inferred.
        merged = oof.join(p.rename({"pred": "_p"}), on="MVT_ID_mvt", how="left", maintain_order="left")
        n_new = int(merged.select((pl.col("oof").is_null() & pl.col("_p").is_not_null()).sum()).item())
        oof = merged.with_columns(pl.coalesce([pl.col("oof"), pl.col("_p")]).alias("oof")).drop("_p")
        fold_metrics[f.name]["n_into_overall"] = n_new
        (pooled if n_new else shadowed).append(f.name)
        # `pooled` marks the entry `overall` actually kept, so a reader can join on it and get one row per
        # id without knowing the coalesce order. Without it the file is a trap: `jul` and `jul_fwd` list the
        # same 190,713 ids, and a naive join double-counts every July row and reports 288.66 against the
        # true 281.84.
        fold_of.append(
            p.select("MVT_ID_mvt").with_columns(
                pl.lit(f.name).alias("fold"), pl.lit(bool(n_new), dtype=pl.Boolean).alias("pooled")
            )
        )
    scored = (
        feats.select(["MVT_ID_mvt", LABEL, "ADEP_mvt", PROXY])
        .join(oof, on="MVT_ID_mvt", how="inner")
        .drop_nulls(["oof", LABEL])
    )
    sy, sp = scored[LABEL].to_numpy().astype(float), scored["oof"].to_numpy().astype(float)
    overall = {
        **metrics(sy, sp),
        **_lane_metrics(scored, sy, sp),
        "folds_pooled": pooled,
        "folds_shadowed": shadowed,
    }
    rec = {
        "exp_id": f"{cfg.name}-{cfg.hash()}",
        "config": {**cfg.__dict__},
        "git_sha": _git_sha(),
        "folds": fold_metrics,
        "overall": overall,
        "train_seconds": round(time.time() - t0, 1),
        # What the model was given, not how wide the frame is. Logging the frame width is what let v6 ship
        # nine screening columns without the record showing it: 99 either way.
        "n_features": len(visible_numeric_columns(feats, cfg)),
        "n_frame_columns": len(numeric_feature_columns(feats)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / log_name, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    oof.write_parquet(out_dir / f"oof_{rec['exp_id']}.parquet")
    if fold_of:
        pl.concat(fold_of).write_parquet(out_dir / f"oof_{rec['exp_id']}_folds.parquet")
    return rec


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

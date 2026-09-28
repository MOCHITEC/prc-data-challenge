"""Item 20 `v4-config-not-reproducible`: resolve configs/submitted_v4.yaml through today's TrainConfig and
the v4-era config (configs/v4_linear_tree.yaml @ 9f31cef) through the v4-era TrainConfig, then diff.

No data is loaded and nothing is trained. The v4-era TrainConfig is extracted from
`git show 9f31cef:src/prc/train.py` with `ast` so that the old module's data imports never run.

Run:  .venv/bin/python docs/investigation_2026-09-16/scripts/v4-config-not-reproducible.py
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace  # noqa: F401  (needed by the exec'd old class)
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

V4_CODE_SHA = "9f31cef"  # last code commit before the v4 upload (2026-09-11T00:42:59Z); d7073bf is docs-only


def git_show(spec: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "show", spec], text=True)


def old_trainconfig_class():
    """Extract `class TrainConfig` from the v4-era train.py and exec it in isolation."""
    src = git_show(f"{V4_CODE_SHA}:src/prc/train.py")
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainConfig")
    mod = ast.Module(body=[cls], type_ignores=[])
    # dataclasses resolves string annotations via sys.modules[cls.__module__], so exec inside a real module.
    import types

    m = types.ModuleType("old_train_9f31cef")
    m.__dict__.update({"dataclass": dataclass, "field": field, "replace": replace,
                       "hashlib": __import__("hashlib"), "json": json})
    sys.modules[m.__name__] = m
    # `from __future__ import annotations` in the old file means annotations are strings; keep that behaviour.
    code = compile(ast.fix_missing_locations(mod), "old_train.py", "exec",
                   flags=__import__("__future__").annotations.compiler_flag, dont_inherit=True)
    exec(code, m.__dict__)
    return m.TrainConfig


def resolve_today(yaml_text: str):
    """Exactly what src/prc/pipeline.py::main does with a YAML today (lines 128-138)."""
    from prc.train import TrainConfig

    spec = yaml.safe_load(yaml_text)
    fields = {k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__}
    if "params" in fields:
        fields["params"] = {**TrainConfig().params, **fields["params"]}
    for k in ("folds", "exclude_features", "include_candidates"):
        if isinstance(fields.get(k), list):
            fields[k] = tuple(fields[k])
    cfg = TrainConfig(**fields)
    dropped = sorted(k for k in spec if k not in TrainConfig.__dataclass_fields__)
    return cfg, spec, dropped


def resolve_v4era(yaml_text: str, OldTC):
    """Exactly what src/prc/pipeline.py::main did at 9f31cef (line 123)."""
    spec = yaml.safe_load(yaml_text)
    cfg = OldTC(**{k: v for k, v in spec.items() if k in OldTC.__dataclass_fields__})
    dropped = sorted(k for k in spec if k not in OldTC.__dataclass_fields__)
    return cfg, spec, dropped


def model_b_params_today(cfg) -> dict:
    """Model B today: predict_fold -> _fit(..., cfg.for_unmatched(), ...) -> lgb_params(); the refit loop then
    sets seed/bagging_seed/feature_fraction_seed per seed k (train.py:301-302)."""
    b = cfg.for_unmatched()
    p = b.lgb_params()
    refit = [dict(p, seed=b.seed + k, bagging_seed=b.seed + k, feature_fraction_seed=b.seed + k)
             for k in range(max(1, b.n_seeds))]
    return {"early_stop_params": p, "num_rounds_cap": b.num_rounds, "refit_params_per_seed": refit}


def model_b_params_v4era(cfg) -> dict:
    """Model B at 9f31cef: predict_fold -> _fit(u_tr, ..., cfg, ...) with the *shared* cfg; the refit used
    `params` unchanged (one fit, seed=cfg.seed only)."""
    p = cfg.lgb_params()
    return {"early_stop_params": p, "num_rounds_cap": cfg.num_rounds, "refit_params_per_seed": [p]}


def main() -> None:
    today_yaml = (REPO / "configs/submitted_v4.yaml").read_text()
    v4era_yaml = git_show(f"{V4_CODE_SHA}:configs/v4_linear_tree.yaml")
    OldTC = old_trainconfig_class()

    cfg_now, spec_now, dropped_now = resolve_today(today_yaml)
    cfg_old, spec_old, dropped_old = resolve_v4era(v4era_yaml, OldTC)

    out = {
        "v4_code_sha": V4_CODE_SHA,
        "yaml_today_keys": sorted(spec_now),
        "yaml_v4era_keys": sorted(spec_old),
        "yaml_keys_dropped_by_loader_today": dropped_now,
        "yaml_keys_dropped_by_loader_v4era": dropped_old,
        "today": {
            "train_config": asdict(cfg_now),
            "model_A_lgb_params": cfg_now.lgb_params(),
            "model_A_num_rounds_cap": cfg_now.num_rounds,
            "model_A_refit_params_per_seed": [
                dict(cfg_now.lgb_params(), seed=cfg_now.seed + k, bagging_seed=cfg_now.seed + k,
                     feature_fraction_seed=cfg_now.seed + k) for k in range(max(1, cfg_now.n_seeds))],
            "model_B": model_b_params_today(cfg_now),
            "use_weather_from_spec": bool(spec_now.get("use_weather", False)),
        },
        "v4era": {
            "train_config": asdict(cfg_old),
            "model_A_lgb_params": cfg_old.lgb_params(),
            "model_A_num_rounds_cap": cfg_old.num_rounds,
            "model_A_refit_params_per_seed": [cfg_old.lgb_params()],
            "model_B": model_b_params_v4era(cfg_old),
            "use_weather_from_spec": bool(spec_old.get("use_weather", False)),
        },
    }
    print(json.dumps(out, indent=2, default=str))

    # Compact side-by-side of the hyper-parameters that decide the booster.
    def row(name, a, b):
        flag = "" if a == b else "  <-- DIFFERS"
        print(f"{name:42s} {str(a):28s} {str(b):28s}{flag}")

    A_old, A_new = out["v4era"]["model_A_lgb_params"], out["today"]["model_A_lgb_params"]
    B_old = out["v4era"]["model_B"]["early_stop_params"]
    B_new = out["today"]["model_B"]["early_stop_params"]
    print("\n== side by side (v4-era @9f31cef  |  submitted_v4.yaml on HEAD)")
    print(f"{'field':42s} {'v4-era':28s} {'today':28s}")
    for k in sorted(set(A_old) | set(A_new)):
        row(f"A.{k}", A_old.get(k, "<absent>"), A_new.get(k, "<absent>"))
    row("A.num_rounds(cap)", out["v4era"]["model_A_num_rounds_cap"], out["today"]["model_A_num_rounds_cap"])
    row("A.n_refits(seeds)", len(out["v4era"]["model_A_refit_params_per_seed"]),
        len(out["today"]["model_A_refit_params_per_seed"]))
    for k in sorted(set(B_old) | set(B_new)):
        row(f"B.{k}", B_old.get(k, "<absent>"), B_new.get(k, "<absent>"))
    row("B.num_rounds(cap)", out["v4era"]["model_B"]["num_rounds_cap"], out["today"]["model_B"]["num_rounds_cap"])
    row("B.n_refits(seeds)", len(out["v4era"]["model_B"]["refit_params_per_seed"]),
        len(out["today"]["model_B"]["refit_params_per_seed"]))
    row("B.refit extra seed keys",
        sorted(set(out["v4era"]["model_B"]["refit_params_per_seed"][0]) - set(B_old)),
        sorted(set(out["today"]["model_B"]["refit_params_per_seed"][0]) - set(B_new)))
    row("early_stopping", cfg_old.early_stopping, cfg_now.early_stopping)
    row("inner_holdout_months", cfg_old.inner_holdout_months, cfg_now.inner_holdout_months)
    row("reference_min_count", cfg_old.reference_min_count, cfg_now.reference_min_count)
    row("use_categoricals", cfg_old.use_categoricals, cfg_now.use_categoricals)
    row("use_weather(spec)", out["v4era"]["use_weather_from_spec"], out["today"]["use_weather_from_spec"])
    for k in ("use_encodings", "use_offset_encoding", "include_candidates", "exclude_features", "folds"):
        row(k, getattr(cfg_old, k, "<no such field>"), getattr(cfg_now, k, "<no such field>"))


if __name__ == "__main__":
    main()

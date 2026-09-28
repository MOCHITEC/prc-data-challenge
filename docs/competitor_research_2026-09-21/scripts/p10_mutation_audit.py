"""Mutation audit for P10 (the CatBoost family): break the new code on purpose, check that a test fails.

    POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/p10_mutation_audit.py

Four silent no-ops or leaks have shipped in this repository. Each mutant below is a defect a reviewer could
plausibly have written into the CatBoost path - a "CatBoost" arm that is really LightGBM, a leak through the
early-stopping split, a seed loop that does not loop, a categorical route that brings G11 back - and the audit
asks whether tests/test_catboost_family.py notices. Same mechanics as c1_c3_mutation_audit.py: a textual patch
to a COPY of the tree in a temporary directory; nothing under src/ is modified.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TESTS = ["tests/test_catboost_family.py"]
TR, PL = "src/prc/train.py", "src/prc/pipeline.py"

# (name, file, what it stands for, before, after)
MUTANTS = [
    (
        "dispatch-dropped",
        TR,
        "a 'catboost' recipe that silently runs LightGBM",
        "        return _fit_catboost(train, target_expr, cfg, feature_cols, cat_cols, predict_frames, es_rows=es_rows)",
        "        pass  # mutant",
    ),
    (
        "es-learns-on-the-holdout",
        TR,
        "early stopping learns on every training row, including the rows it stops on",
        "        dtr = _cb_pool(Xnum[a:b][~hold], Xcat[a:b][~hold], feature_cols, cat_cols, y[~hold])",
        "        dtr = _cb_pool(Xnum[a:b], Xcat[a:b], feature_cols, cat_cols, y)  # mutant",
    ),
    (
        "best-count-off-by-one",
        TR,
        "the 0-based best iteration read as a count",
        "        best = int(best_it + 1) if best_it is not None else cfg.num_rounds",
        "        best = int(best_it) if best_it is not None else cfg.num_rounds  # mutant",
    ),
    (
        "seed-loop-does-not-loop",
        TR,
        "every refit at the same seed, so n_seeds averages identical models",
        "        model = CatBoostRegressor(**params, iterations=rounds, random_seed=cfg.seed + k)",
        "        model = CatBoostRegressor(**params, iterations=rounds, random_seed=cfg.seed)  # mutant",
    ),
    (
        "no-round-scaling",
        TR,
        "the refit uses the early-stopping count unscaled, unlike LightGBM",
        "        rounds = int(min(cfg.num_rounds, max(10, round(best * train.height / max(1, train.height - n_hold)))))\n        del dtr, dva, m",
        "        rounds = int(min(cfg.num_rounds, max(10, best)))  # mutant\n        del dtr, dva, m",
    ),
    (
        "categoricals-not-declared",
        TR,
        "the categorical block built and never handed to CatBoost",
        "        cat_feature_data=np.ascontiguousarray(Xcat) if cat_cols else None,",
        "        cat_feature_data=None,  # mutant",
    ),
    (
        "G11-codes-from-cast-categorical",
        TR,
        "categoricals passed as codes from cast(pl.Categorical), a function of the chunking",
        "        Xcat[:, j] = np.array([lv.encode() for lv in levels], dtype=object)[codes]",
        "        Xcat[:, j] = np.array([str(v).encode() for v in df[c].cast(pl.Categorical).to_physical().to_list()], dtype=object)  # mutant",
    ),
    (
        "null-level-differs",
        TR,
        "a different null level from LightGBM's, so the two families see different level sets",
        '        s = df[c].cast(pl.Utf8).fill_null("__NA__")\n        levels = sorted(s.unique().to_list())\n        codes',
        '        s = df[c].cast(pl.Utf8).fill_null("NA")  # mutant\n        levels = sorted(s.unique().to_list())\n        codes',
    ),
    (
        "ctr-left-to-the-library",
        TR,
        "max_ctr_complexity left to the library, which moves it from 1 to 4 with the iteration count",
        '    "max_ctr_complexity": 1,\n',
        "",
    ),
    (
        "linear-tree-ignored",
        TR,
        "use_linear_tree silently ignored under CatBoost",
        "        if self.use_linear_tree:\n            raise",
        "        if False:  # mutant\n            raise",
    ),
    (
        "unmatched-cap-dropped",
        TR,
        "model B's round cap not carried under CatBoost",
        "            return replace(self, num_rounds=self.unmatched_num_rounds)",
        "            return replace(self)  # mutant",
    ),
    (
        "hash-changes-archived-ids",
        TR,
        "the new fields hashed at their defaults, so every archived experiment id stops naming its config",
        "if not (k in _HASH_OPTIONAL and v == _HASH_OPTIONAL[k])}",
        "if True}  # mutant",
    ),
    (
        "unknown-keys-dropped-again",
        PL,
        "the strict key guard removed: model_familly: catboost runs LightGBM silently",
        "    if unknown:\n        raise",
        "    if False:  # mutant\n        raise",
    ),
    (
        "predict-pool-order",
        TR,
        "predict pools built from the wrong offsets (the training block)",
        "for (s_, e_) in offs[1 : 1 + n_pred]]",
        "for (s_, e_) in offs[0:n_pred]]  # mutant",
    ),
]


def run(tree: Path) -> tuple[bool, str]:
    r = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *TESTS],
        cwd=tree,
        capture_output=True,
        check=False,
        text=True,
        env={"PATH": "/usr/bin:/bin", "POLARS_MAX_THREADS": "2", "PYTHONPATH": str(tree / "src")},
    )
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1]


def main() -> int:
    killed, survived = 0, []
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "tree"
        shutil.copytree(ROOT / "src", base / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "tests", base / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "configs", base / "configs")
        shutil.copy(ROOT / "pyproject.toml", base / "pyproject.toml")
        ok, line = run(base)
        print(f"{'unmutated tree':<34} {'PASS' if ok else 'FAIL'}  {line}", flush=True)
        if not ok:
            print("the unmutated tree does not pass; the audit would be meaningless")
            return 1
        for name, rel, why, before, after in MUTANTS:
            path = base / rel
            src = path.read_text()
            if src.count(before) != 1:
                print(f"{name:<34} SKIPPED - the patch target appears {src.count(before)} times", flush=True)
                survived.append(name)
                continue
            path.write_text(src.replace(before, after))
            ok, line = run(base)
            path.write_text(src)
            print(f"{name:<34} {'SURVIVED' if ok else 'KILLED  '}  {why}\n{'':<36}{line}", flush=True)
            if ok:
                survived.append(name)
            else:
                killed += 1
    print(f"\n{killed}/{len(MUTANTS)} mutants killed")
    if survived:
        print("SURVIVED (a defect the suite does not see): " + ", ".join(survived))
    return 0 if not survived else 2


if __name__ == "__main__":
    raise SystemExit(main())

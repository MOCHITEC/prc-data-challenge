"""Resume serial ten-month fits; checkpoint and attempt a commit after each complete arm."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.update(POLARS_MAX_THREADS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONPATH="src")
root = Path("docs/codex_fe_results")
out = Path("outputs/codex_fe")
state = {"stages": []}


def save():
    state["updated_utc"] = datetime.now(UTC).isoformat()
    temp = root / "continuation_state.tmp"
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(root / "continuation_state.json")


def run(script, *args):
    subprocess.run([sys.executable, "scripts/codex_fe/" + script, *args], check=True)


for arm in ["baseline", "gap", "selected"]:
    tag = arm + "_10m"
    state["current"] = tag
    save()
    if not (root / f"refit_{tag}.json").exists():
        print("starting", tag, flush=True)
        with (out / f"refit_{tag}.log").open("w") as log:
            subprocess.run(
                [sys.executable, "scripts/codex_fe/refit.py", arm, "--train-through", "10"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    info = json.loads((root / f"refit_{tag}.json").read_text())
    assert info["train_through"] == 10 and (out / f"refit_{tag}.parquet").exists()
    run("evaluate_continuation.py")
    run("checkpoint.py")
    added = subprocess.run(
        [
            "git",
            "add",
            "docs/codex_feature_report.md",
            "docs/codex_fe_results",
            "scripts/codex_fe",
            "src/prc/features.py",
            "tests/test_features.py",
            "configs/codex_fe_gap.yaml",
            "configs/codex_fe_selected.yaml",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = None
    if added.returncode == 0:
        commit = subprocess.run(
            ["git", "commit", "-m", f"Measure {arm} feature refit with ten training months"],
            check=False,
            capture_output=True,
            text=True,
        )
    state["stages"].append(
        {
            "arm": tag,
            "complete": True,
            "git_add_code": added.returncode,
            "git_message": added.stderr if added.returncode else commit.stdout + commit.stderr,
        }
    )
    save()
    print("completed", tag, flush=True)
state["current"] = "complete"
save()

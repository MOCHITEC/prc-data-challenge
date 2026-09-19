"""Record code, config, predictions and model fingerprints without changing old provenance."""

import hashlib
import json
import subprocess
from pathlib import Path

paths = [
    Path("src/prc/features.py"),
    Path("src/prc/train.py"),
    Path("configs/main.yaml"),
    Path("data/work/feat_meta.json"),
]
paths += sorted(Path("configs").glob("codex_fe*.yaml"))
paths += sorted(Path("scripts/codex_fe").glob("*.py"))
paths += sorted(Path("outputs/codex_fe").glob("refit_*.parquet"))
paths += sorted(Path("outputs/codex_fe").glob("model_*.txt"))
result = {
    "git_head": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
    "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    "tests": {"feature_tests": 8, "cache_and_config_tests": 11},
    "notes": "Working files could not be committed: git worktree index is read-only. Old provenance retained. Predictions are evaluation only, not submission files.",
}
Path("docs/codex_fe_results/continuation_provenance.json").write_text(json.dumps(result, indent=2))

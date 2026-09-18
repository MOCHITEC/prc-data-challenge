"""Serial experiment coordinator; baseline.py must already be running or completed."""

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.update(POLARS_MAX_THREADS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONPATH="src")
start = time.time()
while not Path("docs/codex_fe_results/baseline.json").exists():
    if time.time() - start > 3600:
        raise TimeoutError("baseline did not finish within one hour")
    time.sleep(10)
for script, arg in [
    ("screen.py", "screen"),
    ("screen.py", "confirm"),
    ("refit.py", "noise"),
    ("refit.py", "selected"),
    ("refit.py", "all"),
]:
    log = Path("outputs/codex_fe") / (script.removesuffix(".py") + "_" + arg + ".log")
    print("starting", script, arg, flush=True)
    with log.open("w") as f:
        subprocess.run(
            [sys.executable, "scripts/codex_fe/" + script, arg], stdout=f, stderr=subprocess.STDOUT, check=True
        )
    print("completed", script, arg, flush=True)

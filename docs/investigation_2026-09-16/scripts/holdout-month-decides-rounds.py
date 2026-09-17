"""Item 23 `holdout-month-decides-rounds`: the statistics are the Batch C set, so this delegates.

    POLARS_MAX_THREADS=2 .venv/bin/python docs/investigation_2026-09-16/scripts/holdout-month-decides-rounds.py \
        <baseline_exp_id> <rounds600_exp_id> <holdout2_exp_id>

Reported for the item beyond the shared table: `rounds_unmatched` per run and the matched / unmatched RMSE.

"""

import sys

from importlib.machinery import SourceFileLoader
from pathlib import Path

_m = SourceFileLoader("_batchc", str(Path(__file__).with_name("tree-identity-saturation.py"))).load_module()

if __name__ == "__main__":
    raise SystemExit(_m.main(sys.argv[1:]))

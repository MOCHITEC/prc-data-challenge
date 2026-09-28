"""Item 11 `linear-tree-model-b`, step (b): the statistics are the Batch C set, so this delegates.

    POLARS_MAX_THREADS=2 .venv/bin/python docs/investigation_2026-09-16/scripts/linear-tree-model-b.py \
        <baseline_exp_id> <linear_tree_exp_id>

Reported for the item beyond the shared table: `unm_pred_gt_43200` / `unm_pred_gt_14400` (forward-fold unmatched
OOF predictions above 12 h / 4 h) and the per-row LIRF table printed at the end.
"""

import sys

from importlib.machinery import SourceFileLoader
from pathlib import Path

_m = SourceFileLoader("_batchc", str(Path(__file__).with_name("tree-identity-saturation.py"))).load_module()

if __name__ == "__main__":
    raise SystemExit(_m.main(sys.argv[1:]))

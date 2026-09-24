"""Summarise OOF files from reporting_fingerprint_eval.py: pooled / NM-matched / NM-unmatched RMSE, sched AUC and
Brier, and day-block paired bootstrap against a reference file (report §4, §8)."""
import sys
import numpy as np
import polars as pl
from reporting_fingerprint_common import CACHE, load_all
from reporting_fingerprint_eval import paired_boot


def auc(score, lab):
    o = np.argsort(score)
    r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


meta = load_all().filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("split") == "train")).select(
    "MVT_ID_mvt", pl.col("FLIGHT_ID_mvt").is_null().alias("nm_null"),
    ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 6).alias("true_sched_copy"),
    ((pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds() > 3600).alias("late1h"))
ref = sys.argv[1]
rows = []
R = pl.read_parquet(CACHE / ref).join(meta, on="MVT_ID_mvt")
for f in sys.argv[1:]:
    P = pl.read_parquet(CACHE / f).join(meta, on="MVT_ID_mvt")
    e = (P["pred"] - P["y"]).to_numpy() ** 2
    nm = P["nm_null"].to_numpy()
    lab = P["true_sched_copy"].to_numpy().astype(int); ps = P["p_sched"].to_numpy(); late = P["late1h"].to_numpy()
    d_all = paired_boot(R, P); d_m = paired_boot(R.filter(~pl.col("nm_null")), P.filter(~pl.col("nm_null")))
    rows.append(dict(file=f.replace("oof_", "").replace(".parquet", ""), rmse=round(float(np.sqrt(e.mean())), 1),
                     rmse_nm=round(float(np.sqrt(e[~nm].mean())), 1), rmse_unm=round(float(np.sqrt(e[nm].mean())), 1),
                     auc_sched=round(auc(ps, lab), 4), auc_sched_late=round(auc(ps[late], lab[late]), 4),
                     brier_sched=round(float(np.mean((ps - lab) ** 2)), 5),
                     d_all=round(d_all["delta"], 2), p_all=d_all["p_better"], d_nm=round(d_m["delta"], 2),
                     ci_nm=[round(x, 2) for x in d_m["ci95"]], p_nm=d_m["p_better"]))
pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40)
print(pl.DataFrame(rows))

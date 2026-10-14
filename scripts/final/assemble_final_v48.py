"""Assemble the v48-based final file `handoff/2026-10-13/final_from_v48.parquet` from committed pieces.

Two committed versions of that path exist on `claude/laughing-cray-vvwh0k` (both also reachable from main):

    --june-layer own  ->  25f6246, md5 be8d89dcf79bd59a6f43d80193c95685 (default): June on the team's own ADS-B layer
                          on 29 of 30 days; 2026-06-11 (no adsb.lol data for the own layer) keeps the A12 layer
    --june-layer a12  ->  558c736, md5 c7ffa31edc2a4de5f1837682d90e14cb: all of June on the A12 layer

Neither file was built by a committed script (README.md section 3, item 6). This script is that combination step,
reconstructed from the committed post-layer pieces and checked row for row against both committed files.

Rows, in the order of final_submitting.parquet (month and day from the DEP row's MVT_TIME_UTC_mvt in
final_ranking.parquet, UTC):
    Jan, Jul : v48 unchanged (handoff/2026-10-08/C15_appear12_cov26eddf.parquet, leaderboard 234.4099; its 344,841
               ids are exactly the Jan+Jul ids of the final set)
    Feb      : own chain   = pre-ADS-B 5-member blend -> docs/final_2026-10-11/adsb_layer.py apply
               -> docs/final_2026-10-11/apply_overlays.py --laneB --lobt-clip --copy-feat unfixed
               (handoff/2026-10-13/inputs/orig_chain_blend_adsb2_ovl.parquet)
    Jun      : own chain, except on --a12-days (default 2026-06-11): A12 chain = the same blend
               -> docs/appear_tier_2026-10-05/final_adsb.py -> the same apply_overlays.py call
               (handoff/2026-10-13/inputs/blend_A12_ovl.parquet); with --june-layer a12 every June day uses it
Then a floor at 0 and Int32. The floor moves one row (Feb, LSZH, MVT_ID 204314027: own chain -10 -> 0); both committed
files carry 0 there, the validator (prc.validate_final) rejects negative values, and the later
docs/final_2026-10-11/assemble_step.py applies the same floor (`clip(0)`) as its last step.

Rebuilt here: the month/day split, the per-row choice of source, the floor at 0 and the file itself. Consumed as
committed inputs (not rebuilt): v48, and the two chain files, which already carry the members, the blend, the ADS-B
layer, the lane-B / LIRF overlays and the LOBT clip. The chain files were committed after 25f6246 (A12 chain in
e301db5, own chain in 68e7693 as the "old chain" of assemble_step.py). Their pre-ADS-B base is
handoff/2026-10-13/pre_adsb_base_blend5_final.parquet (column b of inputs/orig_chain_adsb_detail.parquet equals it on
every row); the own layer has no ADS-B component on 2026-06-11, where the A12 chain differs from the own chain on 78
rows. Rebuilding the overlays from that base is not possible from the repository alone: apply_overlays.py needs the
2025 training_*.parquet raw files (LIRF unmatched rule B) and feat_rank.parquet (LOBT clip), neither of which is
committed.

    python scripts/final/assemble_final_v48.py --out outputs/final/final_from_v48.parquet \\
        [--june-layer own|a12] [--reference <committed file>] [--expect-md5 <md5>]

All paths are relative to the repository root (run from there) unless given as absolute paths. polars 1.44.2
(requirements-lock.txt) writes the file byte-identical to the committed one.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import date
from pathlib import Path

# final_ranking.parquet carries an R extension type tag; read it as its storage type
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
N_ROWS = 670_790
INPUTS = "handoff/2026-10-13/inputs/"
# md5 of the committed inputs (handoff/2026-10-13/MD5SUMS, handoff/2026-10-13/README.md, RUNBOOK_final_submit.md)
INPUT_MD5 = {
    "template": "2c1f407296e217fd7d242c9d51ba27ad",
    "ranking": "1ce0c6ee9ea15b8c3d0cd282591d4e0a",
    "v48": "8ee6679f43ead60db1a1aec7876caf09",
    "own_chain": "3fe8fb33f6ef9724558015d90b756cd5",
    "a12_chain": "3221ce5bfb468f702575556e183acef7",
}
OUTPUT_MD5 = {"own": "be8d89dcf79bd59a6f43d80193c95685", "a12": "c7ffa31edc2a4de5f1837682d90e14cb"}


def md5(path: Path) -> str:
    h = hashlib.md5()  # integrity fingerprint, as in the hand-off MD5SUMS
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def pred(path: Path, name: str) -> pl.DataFrame:
    d = pl.read_parquet(path, columns=[ID, Y])
    assert d[ID].is_unique().all() and d[Y].null_count() == 0, path
    return d.select(ID, pl.col(Y).cast(pl.Int64).alias(name))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, default=Path("data/raw/final/final_submitting.parquet"))
    ap.add_argument("--ranking", type=Path, default=Path("data/raw/final/final_ranking.parquet"))
    ap.add_argument("--v48", type=Path, default=Path("handoff/2026-10-08/C15_appear12_cov26eddf.parquet"))
    ap.add_argument("--own-chain", type=Path, default=Path(INPUTS + "orig_chain_blend_adsb2_ovl.parquet"))
    ap.add_argument("--a12-chain", type=Path, default=Path(INPUTS + "blend_A12_ovl.parquet"))
    ap.add_argument("--june-layer", choices=["own", "a12"], default="own",
                    help="own: 25f6246 / be8d89dc (default); a12: 558c736 / c7ffa31e")
    ap.add_argument("--a12-days", default="2026-06-11",
                    help="comma list of June days kept on the A12 chain with --june-layer own "
                         "(days without own-layer ADS-B data)")
    ap.add_argument("--out", type=Path, default=Path("outputs/final/final_from_v48.parquet"))
    ap.add_argument("--reference", type=Path, default=None, help="committed file to compare with, row by row")
    ap.add_argument("--expect-md5", default=None, help="fail unless the output has this md5 (default: no check)")
    ap.add_argument("--no-input-md5", action="store_true", help="do not stop on an input whose md5 differs")
    a = ap.parse_args()

    paths = {"template": a.template, "ranking": a.ranking, "v48": a.v48,
             "own_chain": a.own_chain, "a12_chain": a.a12_chain}
    bad = []
    for k, p in paths.items():
        m = md5(p)
        ok = m == INPUT_MD5[k]
        print(f"input {k:9s} {p}  md5 {m}  {'ok' if ok else 'DIFFERS from ' + INPUT_MD5[k]}")
        bad += [] if ok else [k]
    if bad and not a.no_input_md5:
        print(f"stopping: input md5 differs for {bad} (pass --no-input-md5 to continue)", file=sys.stderr)
        return 1

    tpl = pl.read_parquet(a.template)
    assert tpl.columns == [ID, Y] and tpl.height == N_ROWS and tpl[ID].is_unique().all()
    rk = (pl.read_parquet(a.ranking, columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
          .select(ID, pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone("UTC").dt.date().alias("day")))
    d = tpl.select(ID).join(rk, on=ID, how="left", maintain_order="left")
    assert d["day"].null_count() == 0, "template id without a DEP row in final_ranking"
    d = d.with_columns(pl.col("day").dt.month().alias("m"))
    assert set(d["m"].unique().to_list()) == {1, 2, 6, 7} and all(dd.year == 2026 for dd in d["day"].unique().to_list())

    v48, own, a12 = pred(a.v48, "v48"), pred(a.own_chain, "own"), pred(a.a12_chain, "a12")
    jj = pl.col("m").is_in([1, 7])
    assert set(v48[ID].to_list()) == set(d.filter(jj)[ID].to_list()), "v48 ids are not exactly the Jan+Jul ids"
    for name, x in (("own", own), ("a12", a12)):
        assert x.height == N_ROWS and set(x[ID].to_list()) == set(d[ID].to_list()), f"{name} chain ids != template ids"
    d = (d.join(v48, on=ID, how="left", maintain_order="left").join(own, on=ID, how="left", maintain_order="left")
         .join(a12, on=ID, how="left", maintain_order="left"))

    a12_days = [date.fromisoformat(s) for s in a.a12_days.split(",") if s]
    jun_a12 = (pl.col("m") == 6) & (pl.lit(True) if a.june_layer == "a12" else pl.col("day").is_in(a12_days))
    src = pl.when(jj).then(pl.lit("v48")).when(jun_a12).then(pl.lit("a12")).otherwise(pl.lit("own"))
    d = d.with_columns(src.alias("src"))
    d = d.with_columns(pl.when(jj).then("v48").when(jun_a12).then("a12").otherwise("own").alias("raw"))
    assert d["raw"].null_count() == 0
    d = d.with_columns(pl.col("raw").clip(lower_bound=0).alias("p"))

    print(f"june layer: {a.june_layer}" + (f" (A12 kept on {a.a12_days})" if a.june_layer == "own" else ""))
    print(d.group_by("m", "src").agg(pl.len().alias("rows")).sort("m", "src"))
    fl = d.filter(pl.col("p") != pl.col("raw"))
    print(f"floor at 0: {fl.height} row(s) moved: {fl.select(ID, 'day', 'src', 'raw', 'p').to_dicts()}")

    out = d.select(ID, pl.col("p").cast(pl.Int32).alias(Y))
    assert out.schema == tpl.schema and out[ID].equals(tpl[ID]) and out.null_count().sum_horizontal().item() == 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)
    got = md5(a.out)
    known = OUTPUT_MD5["a12"] if a.june_layer == "a12" else OUTPUT_MD5["own"] if a.a12_days == "2026-06-11" else None
    vs = f" ({'matches' if got == known else 'differs from'} committed {known})" if known else ""
    print(f"wrote {a.out}: {out.height} rows, md5 {got}{vs}")

    rc = 0
    if a.reference is not None:
        ref = pl.read_parquet(a.reference)
        same_ids = ref[ID].equals(out[ID])
        print(f"reference {a.reference}: md5 {md5(a.reference)}, rows {ref.height}, schema {dict(ref.schema)}, "
              f"ids/order equal: {same_ids}")
        if same_ids:
            diff = (out[Y].cast(pl.Int64) - ref[Y].cast(pl.Int64)).abs()
            print(f"row-level: {int((diff != 0).sum())} rows differ, max |diff| {int(diff.max())}")
            rc |= int((diff != 0).sum() != 0)
        else:
            rc = 1
    if a.expect_md5 and got != a.expect_md5:
        print(f"md5 {got} != expected {a.expect_md5}", file=sys.stderr)
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())

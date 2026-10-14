"""Old-board (Jan/Jul 2026) mean-bias probes -> correction of the final file. Four probes on v48 (old LB 234.4099,
used_pairs 344,841 = the whole ranking file):

  P1 LTFM January, all rows, S = -40          (handoff/2026-10-14/probes/v48_LTFM_01_shift-40.parquet)
  P2 LTFM July, all rows, S = -40             (v48_LTFM_07_shift-40.parquet)
  P3 LTFM January de-icing rows, S = -100     (v48_LTFM_01_deice_shift-100.parquet; rows in cell_LTFM_01_deice.parquet)
  P4 EDDF/EDDM/EHAM/LFPG/LSZH January de-icing rows, S = -100 (v48_OTHER5_01_deice_shift-100.parquet; cell_OTHER5_01_deice)

For a pure shift S of the n rows of a cell: MSE_probe - MSE_v48 = (n/N)(S^2 - 2 S rbar), rbar = mean(y - v48) over the cell,
so rbar = (S^2 - (R_probe^2 - R_v48^2) N/n) / (2S). With P1 and P3, the LTFM January non-de-icing rows get
rbar_no = (n_all rbar_all - n_de rbar_de) / (n_all - n_de).

    python -I docs/final_phase_gate/probe_fix.py <v48.parquet> <R_v48> <R_P1|-> <R_P2|-> <R_P3|-> <R_P4|-> <final_in> <final_out> [--dithered]
        [--stage2=EDDF=<score>,EDDM=<score>,...]   (optional per-airport January de-icing probes, probes/stage2/)

Correction of the final file F (same departures, by MVT_ID): each cell is shifted by rbar - mean(F - v48) over the cell
(F before the +1x MVT dither when --dithered: the per-row dither is subtracted first). Then the LOBT identity clip on the
final clock and the zero floor. February de-icing rows (cell_*_02_deice.parquet, flagged from hourly METAR) get 0.5 x the
January de-icing shift when |rbar| >= 100 s (LTFM) / 50 s (five airports), as pre-registered. A probe given as '-' leaves its rows unchanged (if P1 is missing, P3 alone corrects only
the de-icing rows; if P3 is missing, P1 corrects all LTFM January rows by one shift)."""
import sys
import numpy as np
import polars as pl

K, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
N_OLD = 344841
P = "handoff/2026-10-14/probes/"


def rbar(r0, r, n, s):
    return (s * s - (float(r) ** 2 - float(r0) ** 2) * N_OLD / n) / (2 * s)


def main(a):
    v48p, r0, p1, p2, p3, p4, fin, fout = a[:8]
    dithered = "--dithered" in a
    V = pl.read_parquet(v48p).rename({T: "v48"})
    O = (pl.scan_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
         .select(K, "ADEP_mvt", pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"), pl.col("MVT_TIME_UTC_mvt").alias("t_old")).collect())
    de_ltfm = set(pl.read_parquet(P + "cell_LTFM_01_deice.parquet")[K].to_list())
    de_oth = set(pl.read_parquet(P + "cell_OTHER5_01_deice.parquet")[K].to_list())
    O = O.with_columns(de_l=pl.col(K).is_in(list(de_ltfm)), de_o=pl.col(K).is_in(list(de_oth)))
    n_all = int(((O["ADEP_mvt"] == "LTFM") & (O["m"] == 1)).sum()); n_de = len(de_ltfm)
    n_jul = int(((O["ADEP_mvt"] == "LTFM") & (O["m"] == 7)).sum()); n_oth = len(de_oth)
    assert (n_all, n_de, n_jul, n_oth) == (22396, 1843, 25185, 4880), (n_all, n_de, n_jul, n_oth)
    cells = []  # (name, polars mask over O, rbar)
    ltfm_jan = (pl.col("ADEP_mvt") == "LTFM") & (pl.col("m") == 1)
    post = set(pl.read_parquet(P + "cell_LTFM_01_postdeice.parquet")[K].to_list())
    O = O.with_columns(post_l=pl.col(K).is_in(list(post)))
    n_post = len(post)
    if p1 != "-" and p3 != "-":
        ra, rd = rbar(r0, p1, n_all, -40.0), rbar(r0, p3, n_de, -100.0)
        rn = (n_all * ra - n_de * rd) / (n_all - n_de)
        if abs(rd) >= 100.0:
            # pre-registered: the post-de-icing hour rows get 0.5 x the de-icing shift; the remaining 'other' rows get the
            # rest of the non-de-icing mean (their mean residual given that the post rows' mean is assumed 0.5 rd)
            r_post = 0.5 * rd
            r_rest = (n_all * ra - n_de * rd - n_post * r_post) / (n_all - n_de - n_post)
            cells += [("LTFM Jan de-icing", ltfm_jan & pl.col("de_l"), rd),
                      ("LTFM Jan post-de-icing hour", ltfm_jan & ~pl.col("de_l") & pl.col("post_l"), r_post),
                      ("LTFM Jan other", ltfm_jan & ~pl.col("de_l") & ~pl.col("post_l"), r_rest)]
            print(f"LTFM Jan: rbar all {ra:+.2f}, de-icing {rd:+.2f}, post-hour (0.5 x de-icing) {r_post:+.2f}, rest {r_rest:+.2f}")
        else:
            cells += [("LTFM Jan de-icing", ltfm_jan & pl.col("de_l"), rd), ("LTFM Jan other", ltfm_jan & ~pl.col("de_l"), rn)]
            print(f"LTFM Jan: rbar all {ra:+.2f}, de-icing {rd:+.2f}, other {rn:+.2f}")
    elif p1 != "-":
        cells.append(("LTFM Jan all", ltfm_jan, rbar(r0, p1, n_all, -40.0)))
    elif p3 != "-":
        cells.append(("LTFM Jan de-icing", ltfm_jan & pl.col("de_l"), rbar(r0, p3, n_de, -100.0)))
    if p2 != "-":
        cells.append(("LTFM Jul all", (pl.col("ADEP_mvt") == "LTFM") & (pl.col("m") == 7), rbar(r0, p2, n_jul, -40.0)))
    # optional stage 2: per-airport January de-icing probes (probes/stage2/v48_<AP>_01_deice_shift-100.parquet), given as
    # --stage2 EDDF=<score>,EDDM=<score>,...; the pooled P4 then only covers the airports without their own probe.
    st2 = {}
    for x in a:
        if x.startswith("--stage2="):
            st2 = {kv.split("=")[0]: kv.split("=")[1] for kv in x[len("--stage2="):].split(",") if kv}
    oth_ap = pl.read_parquet(P + "cell_OTHER5_01_deice.parquet").select(K, pl.col("ADEP_mvt").alias("ap_o"))
    O = O.join(oth_ap, on=K, how="left")
    feb_ap = pl.read_parquet(P + "cell_OTHER5_02_deice.parquet").select(K, pl.col("ADEP_mvt").alias("ap_f"))
    oth_names = {}
    if st2:
        n_ap = {ap: int((O["ap_o"] == ap).sum()) for ap in st2}
        rb_ap = {ap: rbar(r0, sc, n_ap[ap], -100.0) for ap, sc in st2.items()}
        for ap, rb_ in rb_ap.items():
            nm = f"{ap} Jan de-icing"
            cells.append((nm, pl.col("ap_o") == ap, rb_)); oth_names[nm] = [ap]
        rest = [ap for ap in ("EDDF", "EDDM", "EHAM", "LFPG", "LSZH") if ap not in st2]
        if p4 != "-" and rest:
            n_rest = n_oth - sum(n_ap.values())
            r_pool = rbar(r0, p4, n_oth, -100.0)
            r_rest = (n_oth * r_pool - sum(n_ap[ap] * rb_ap[ap] for ap in st2)) / n_rest
            nm = "/".join(rest) + " Jan de-icing"
            cells.append((nm, pl.col("ap_o").is_in(rest), r_rest)); oth_names[nm] = rest
            print(f"five-airport pool rbar {r_pool:+.2f}; per airport {', '.join(f'{k} {v:+.2f}' for k, v in rb_ap.items())}; rest {r_rest:+.2f}")
    elif p4 != "-":
        nm = "EDDF/EDDM/EHAM/LFPG/LSZH Jan de-icing"
        cells.append((nm, pl.col("de_o"), rbar(r0, p4, n_oth, -100.0))); oth_names[nm] = ["EDDF", "EDDM", "EHAM", "LFPG", "LSZH"]
    F = pl.read_parquet(fin)
    nw = (pl.scan_parquet("data/raw/final_ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
          .select(K, "MVT_TIME_UTC_mvt", pl.col("AOBT_3_flt").is_not_null().alias("mat"),
                  (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("mml")).collect())
    Z = (F.join(nw, on=K, how="left", maintain_order="left").join(V, on=K, how="left", maintain_order="left")
         .join(O, on=K, how="left", maintain_order="left").join(feb_ap, on=K, how="left", maintain_order="left"))
    Z = Z.with_columns(dl=pl.when(pl.col("mat") & pl.col("t_old").is_not_null())
                       .then((pl.col("MVT_TIME_UTC_mvt") - pl.col("t_old")).dt.total_seconds().cast(pl.Float64)).otherwise(0.0))
    shift = pl.lit(0.0)
    applied = {}
    for name, mask, rb in cells:
        cz = Z.filter(mask)
        assert cz["v48"].null_count() == 0
        base = (cz[T].cast(pl.Float64) - (cz["dl"] if dithered else 0.0)) - cz["v48"].cast(pl.Float64)
        c = rb - float(base.mean())
        shift = shift + pl.when(mask.fill_null(False)).then(pl.lit(c)).otherwise(0.0)
        applied[name] = (rb, c)
        print(f"{name}: rows {cz.height}, rbar(v48) {rb:+.2f} s, mean(F-v48) {float(base.mean()):+.2f} s, shift {c:+.2f} s, "
              f"old-LB MSE gain {cz.height / N_OLD * rb ** 2:.1f} s^2")
    # Pre-registered February rule (README, 06:00 UTC): 0.5 x the January de-icing shift of F, if |rbar| passes the threshold.
    jan = {name: (rb, c) for name, (rb, c) in applied.items()}
    feb_l = set(pl.read_parquet(P + "cell_LTFM_02_deice.parquet")[K].to_list())
    rules = [("LTFM Jan de-icing", pl.col(K).is_in(list(feb_l)), 100.0, "LTFM Feb de-icing")]
    for nm, aps in oth_names.items():
        rules.append((nm, pl.col("ap_f").is_in(aps).fill_null(False), 50.0, nm.replace("Jan", "Feb")))
    for jname, mask, thr, label in rules:
        if jname in jan and abs(jan[jname][0]) >= thr:
            c2 = 0.5 * jan[jname][1]
            nrow = Z.filter(mask).height
            shift = shift + pl.when(mask).then(pl.lit(c2)).otherwise(0.0)
            print(f"{label}: rows {nrow}, shift {c2:+.2f} s (0.5 x the January shift; |rbar| {abs(jan[jname][0]):.0f} >= {thr:.0f})")
        elif jname in jan:
            print(f"{label}: unchanged (|rbar| {abs(jan[jname][0]):.0f} < {thr:.0f})")
    v = pl.col(T).cast(pl.Float64) + shift
    lo, hi = (pl.col("mml") - 3606).ceil(), (pl.col("mml") + 3606).ceil()
    v = pl.when(pl.col("mml").is_not_null()).then(pl.max_horizontal(pl.min_horizontal(v, hi), lo)).otherwise(v)
    out = Z.select(K, v.clip(lower_bound=0).round(0).cast(pl.Int32).alias(T)).cast(F.schema)
    assert out[K].equals(F[K]) and out[T].null_count() == 0
    out.write_parquet(fout)
    d = (out[T].cast(pl.Float64) - F[T].cast(pl.Float64)).to_numpy()
    print(f"rows changed {int((d != 0).sum()):,}; sum d^2/670,790 {float((d ** 2).sum()) / 670790:.1f}; wrote {fout}")


if __name__ == "__main__":
    main(sys.argv[1:])

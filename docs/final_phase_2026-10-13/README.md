# Final phase: status of this thread (色々調査, branch claude/peaceful-babbage-md9l2x), 2026-10-13

Follows `origin/claude/model-b-and-arrival-features:docs/final_phase_2026-10-11/NO_FINAL_UPLOAD.md`.

## Uploads
- This thread has never uploaded anything to the team bucket: no `_vNN.parquet` and no `_final.parquet`.
- Submissions v40–v47 were made by the integration thread (データ分析). This thread only handed over member deltas and recipes.
- The `resilient-hedgehog_final.parquet` of 2026-10-09 was not made by this thread.
- This thread will not upload anything.

## Data access
- This container has no OSN credentials. The `AWS_*` variables present are rejected by s3.opensky-network.org with InvalidAccessKeyId.
- So `final_ranking.parquet` (Jan, Feb, Jun, Jul 2026) cannot be read here, and **no final candidate file was built in this thread.**

## Levers from this branch for the 4-month final

| Lever | In final v3 (laughing-cray 71a7d27)? | Recipe | Evidence | Expected |
|---|---|---|---|---|
| L6 gate retention | yes (members `*_l6`) | `fe_retention: true` | docs/research_2026-10-05_measure/l6_plan.md | done |
| LIRF cmix | yes (a2o `l2_opt_l6_cmix`) | `copy_mix_airports: [LIRF]` | copy_plan.md | done |
| NM-blind w = 0.25 | yes (lane B) | `nmblind_weight: 0.25` | docs/research_2026-10-06_nmblind | done |
| ucmix, LIRF unmatched | yes (lane B) | `ucmix_airports: [LIRF]` | docs/research_2026-10-07_ucmix | done |
| **cmix LTFM extension** | **no** | a2o with `copy_mix_airports: [LIRF, LTFM]` (configs/main_flseq_osm_nolirf_l2_opt_cmix_ltfm.yaml) on top of L6, 3 seeds | copy_plan.md, LTFM result: −0.25 s whole frame [−0.39, −0.12]; LTFM matched −2.26 [−3.25, −1.13]; all folds improve | about −0.03 to −0.1 s after the a2o weight |
| **Blend inner split, non-LIRF matched rows** | **no** | on non-LIRF matched rows, with c2: 0.4·(0.2·a2o + 0.8·pao) + 0.2·(p2o + c2 + m4). Simplest form: change only the inner a2o:pao split from 0.5:0.5 to 0.2:0.8 on non-LIRF matched rows; LIRF unchanged | docs/research_2026-10-08_blend: inner split alone −0.12 s (no c2), all folds; full non-LIRF reweight −0.22 [−0.30, −0.14] | −0.1 to −0.2 s. Re-measure with the c2 OOF first. |

- Both open levers act before the ADS-B layer.
- Per the team instruction, a candidate built on a changed base must rebuild the ADS-B residual and appear layers on that same base.

## Not for the final
- **Rejected after validation:** G-model, carrier echo, stamp (all and LIRF), cmix LEBL, ucday, day-demean, live-level floor.
- The restricted floor was never submitted.
- See the docs/research_2026-10-0*/ verdicts and docs/research_2026-10-08_audit.

## Ready-to-run tool for the blend lever
- `scripts/blend_nonlirf_reweight.py FRAME a2o pao p2o c2 m4 OUT [INNER_A2O=0.2]`
  - On matched rows outside LIRF it changes the inner a2o:pao split from 0.5:0.5 to 0.2:0.8. LIRF matched rows keep the production formula.
  - Output columns: MVT_ID_mvt, pred_prod, pred.
  - Use it before the ADS-B layer, and rebuild that layer on the new base.
- **Self-test on the 2025 OOFs:** a2o, pao, p2o and m4, with c2 stood in by p2o.
  - 664,050 matched rows; 614,592 moved; RMS change 3.75 s.
  - Matched RMSE 204.165 → 203.997, about −0.13 s whole frame. This matches the leave-one-fold-out estimate.

## 最終データ上の差分ファイル（このコンテナ内のみ。バケットには置いていない）
| ファイル | 内容 | 基準 | md5 |
|---|---|---|---|
| outputs/final_pb/delta_ltfm_a2o_w8.parquet | (a2o_ltfm − a2o) / 8。LTFM の照合あり行 87,420 行だけが動く。最終全体で RMS 2.6 秒 | 本番 a2o（main_flseq_osm_nolirf_l2_opt_l6_cmix、3 seeds）を最終データで予測したもの | bdf3a13acf3b9ec53733079f17ca02fe |
| outputs/final_pb/delta_inner_split.parquet | 0.3·(pao − a2o) / 4。LIRF 以外の照合あり行 601,583 行。月ごとの RMS は 3.4〜4.6 秒 | 同じ a2o と pao_l6 の最終データ予測（どちらも共通特徴量の union 修正なし版） | 90909b0dbdaf860cd12310f3b21625ab |

- **使い方**：ブレンド後、ADS-B 層を作る前の予測に足す。ベースが変わるので、ADS-B 層は作り直すこと。
- **注意**：統合スレッドの最終メンバーには union 修正（fl_/seq_）が入っているが、こちらの予測には入っていない。差分の大きさは目安として使い、正式に入れる場合は統合側のメンバーで作り直すのが望ましい。

# 引き継ぎ: このブランチで採用した、混合全体で 1 秒以下の施策（2026-10-06）

ブランチは `claude/peaceful-babbage-md9l2x`。どの施策も既定は OFF で、既存の実験 ID は変わらない。
本番パイプラインのブランチ（model-b-and-arrival-features）は、2026-10-05 にこのブランチへマージ済み。

| 施策 | 設定 | 対象 | 検証結果（全 fold） | 混合での見込み | 状態 |
|---|---|---|---|---|---|
| **L6 ゲートでの待ち** | `fe_retention: true` | model A（LGB の 3 member） | a2o −0.73、pao −0.76、p2o −0.83 秒。全 fold・jan も改善し、厳しい判定（STRICT）でも ADOPT | LGB の部分 −0.82 秒 → 混合全体で約 −0.4〜−0.6 秒 | **採用** |
| **LIRF の書き写しの混合** | `copy_mix_airports: [LIRF]` | model A、LIRF の照合できた行 | a2o で −0.49 秒（K=200 −0.70）。LIRF は jan −4.9、jul −4.1 秒 | L6 と合わせると LGB の部分 −1.18 秒（近似） | **採用**（a2o で検証済み） |
| 新スタンド対策 | `stand_dropout: 0.05` | 全行（スタンドのカテゴリ） | 学習に無いスタンドの行で −6.5 秒、知っているスタンドの行は ±0 | 約 −0.04 秒 | 採用候補（本番 member では未測定） |
| 航空会社の書き写し率 | `fe_carrier_echo: true` | model B のみ | 測定中 | −0.8〜−1.7 秒（簡易モデルでの見積り、確度は中） | 測定中 |

不採用にしたもの: LIRF の G モデルの混合（`gmodel_airports`）、月ごとの数え直し（`fe_month_scope`）、2 段目の補正モデル（4 か月の OOF）。

## 統合の方法
### 1. 差分を足す方法（再学習なし、すぐできる）
`scripts/build_member_delta.py BASE OUT old:new:w ...` を使う。各 member の変化に重みを掛けて、照合できた行だけに足す。
- 重みは v35〜v40 の混合 ((0.5 a2o + 0.5 pao) + p2o + c2 + m4) / 4 に従う。a2o 1/8、pao 1/8、p2o 1/4。
- 必要なランキング予測は、このブランチの `outputs/` にある（ローカルのみ。作成中）。
  - main_flseq_osm_nolirf_l2_opt.parquet / main_flseq_osm_nolirf_l2_opt_l6_cmix.parquet
  - main_flseq_osm_nolirf_l2_pa_opt.parquet / main_flseq_osm_nolirf_l2_pa_opt_l6.parquet
  - main_flseq_osm_nolirf_plainy_l2_opt.parquet / main_flseq_osm_nolirf_plainy_l2_opt_l6.parquet
- 近似の注意: 上に乗っている ADS-B の残差と overlay は学習し直さない。差分の大きさは数秒程度。

### 2. 正式に入れる方法
各 member の設定に上のフラグを足して、OOF と予測を作り直し、混合と ADS-B の残差を作り直す。
- CatBoost と MLP にも L6 を入れれば、上乗せの余地がある（未測定）。

## 根拠
- `docs/research_2026-10-05_measure/l6_plan.md`、`l6_*_verdict.txt`、`l6_blend_check.txt`
- `docs/research_2026-10-05_measure/copy_plan.md`、`cmix_verdict.txt`
- `docs/research_2026-10-05_newstand/results.md`

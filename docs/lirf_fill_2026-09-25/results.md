# LIRF マッチ行の流用分類器 — 結果（2026-09-25）

計画は `plan.md`（8cd1d7a、結果の前に push）。スクリプト `pretest.py`（読み取りのみ）。

## 事前テスト v1（forward fold）: 不合格 → この形は閉じる

```
LIRF matched rows fit 121,933 / stop 14,027 / validation 23,256
fill rate fit 17.46%, stop 17.37%, validation 19.96%
classifier: 88 rounds, validation AUC 0.8433, mean p 0.1774 vs rate 0.1996
top gain: flight_identity 21.2%, flight_prefix 18.6%, eobt_schedule_revision 12.1%, AIRCRAFT_OPERATOR_flt 10.9%,
          aobt3_minus_sched 7.8%, mvt_minus_sched 6.8%, STAND_mvt 5.8%, ADES_mvt 5.3%
```

| 予測 | fold 全体 | マッチ行 | LIRF マッチ行 |
|---|---:|---:|---:|
| base | 219.72 | 198.53 | 300.41 |
| 混合 `(1−p)·oof + p·(MVT−SCHED)` | 220.40 | 199.28 | 307.31 |
| シャッフル対照 | 244.60 | 225.99 | 503.14 |

| base → | K=0 | K=20 | K=200 | K=2000 | 判定 |
|---|---|---|---|---|---|
| 混合 | +0.67 [+0.29, +1.09] | +0.75 [+0.34, +1.17] | +0.86 [+0.60, +1.13] | +0.70 [+0.60, +0.80] | **REJECT（悪化）** |
| シャッフル | +24.87 [+20.55, +29.81] | +19.61 | +12.24 | +4.51 | REJECT |

- 条件 2（混合が ADOPT）を満たさない。**事前登録どおり、この形（v1）は閉じる。**
- シャッフル対照が +24.9 s 悪化するので、分類器が情報を持っていないわけではない。持っている情報が、base の model A が
  すでに取り込んでいる分（model A は `mvt_minus_sched` や `aobt3_minus_sched` を入力に持つ）を超えない。
- AUC 0.843 は、この競技で混合が効いたと記録のある水準（kind-mango の 0.9256、P1 のゲート 0.93）より明確に低い。
  未マッチ行の同種の分類器（P1 stage 2）の 0.9201 よりも低い。
- 注意: この混合は model A が流用行も学習している状態で重ねたので、流用を二重に数える側に偏る。ただし AUC 0.84 では、
  model A を流用行抜きで学習し直しても、流用行を当て損ねた分の損失が大きく、改善に転じる見込みは薄い。

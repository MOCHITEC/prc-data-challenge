# appear を 12 か月で学習し直した差分（統合用）

v39 で入れた appear の差分（2025 年の 4 か月で学習）を、12 か月で学習したものに置き換える差分。CV で **採用**
（`docs/appear_tier_2026-10-05/plan_months12.md`、結果は `results.md` §12）。

## 使い方

`appear12_delta.parquet`（`MVT_ID_mvt`、`delta` = 整数の秒、240,983 行）を、v39 の appear の差分が入ったままの提出ファイル
（v39・v40・v41・v42 など）に足す。

```python
d = pl.read_parquet("handoff/2026-10-06/appear12_delta.parquet")
new = (sub.join(d, on="MVT_ID_mvt", how="left")
          .with_columns((pl.col("TAXITIME_SEC_mvt") + pl.col("delta").fill_null(0)).clip(0).cast(pl.Int32))
          .select(sub.columns))
```

- 変えるのは appear の列がある行だけ（ADS-B の地上の fix ＋ callsign の結合がある行）。LTFM は 0。
- `v39_appear12_on_v39.parquet` は v39 に足したもの（validate OK、v39 ＋ 差分表と完全一致）。仕組みの確認用。

## 効果の見込み

- CV（2025 年 12 か月、208 万行）: 今の作り方（12 か月の plain 残差 ＋ 4 か月の appear の差分）に対して **−0.804 s [−0.924, −0.692]**。
  12 か月すべて、LTFM 以外の全空港で改善。ランキングと同じ月は 1 月 −0.61、7 月 −0.98。
- 差分の大きさ Σd²/N = 482。CV では MSE の改善が変更の大きさの 0.83 倍（1・7 月は 0.94 倍）なので、**LB は −0.85〜−0.95 s の見込み**
  （v39 は 0.88 倍だった）。1 秒未満なので、合意どおり他の施策と束ねて出す。

## 注意

- 差分は b（残差をかける前の base）に敏感。ファイルから b を戻すときに使う plain 残差を 4 か月版と 12 か月版で取り替えると、
  差分が行あたり RMS 15.7 s 変わる（差分そのものの RMS は 25.9 s）。ここでは v39 に合わせて 4 か月版で戻した。v39 と v40 は
  base が同じなので、v40 以降にもこの表をそのまま足せばよい。**v40 以降のファイルに `apply_delta12.py` をかけ直すのは不可**
  （それらのファイルの plain 残差は 12 か月版または 7 か月版で、4 か月版で b を戻すとずれる）。
- v42 のように base（マッチ行）自体を変えた後のファイルでも、b の変化が数秒〜十数秒なら差分への影響は小さい。
- EGLL は 2026 年の方が「自スタンドで最初に見えた」行が多い（2025 年の学習 0.13、2026 年 0.45）。v39 の 4 か月版でも同じずれ
  （0.18 対 0.45）があり、v39 の LB は CV より良かった。
- 作り方: `docs/appear_tier_2026-10-05/apply_delta12.py`（ログ `results/apply_delta12_on_v39.log`）。

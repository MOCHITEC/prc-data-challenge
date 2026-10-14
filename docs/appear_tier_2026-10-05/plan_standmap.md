# appear tier の自スタンドの座標を ADS-B から作る — 事前登録（2026-10-05 13:00 UTC、CV の前に書く）

## 0. なぜ

- 2026 年の ADS-B の地上の区間がある 248,346 行のうち 26,137 行は、`STAND_mvt` が OSM の parking_position と照合できず、appear の
  距離の列が空（EDDM 10,700 行 = 42 %、EDDF 8,827 行 = 27 %、LEBL 3,102 行）。そのうち 15,943 行は stage 1 の stand run がある。
- 下見（ラベル無し、2026 年の ADS-B だけ）: 10 分以上止まっていた stand run の位置の中央値をスタンドごとに取ると、
  5 便以上・ばらつき（中央値からの距離の中央値）40 m 以内のスタンドが 854 か所。OSM に無いもの 88 か所（EDDM 51、EDDF 13、LEBL 11）。
  OSM にもあるスタンドでの OSM との距離の中央値は 10.7（EDDM）〜43.0 m（LSZH）、50 m 以内は 52〜92 %。OSM の点（停止線など）と
  機体のトランスポンダの位置は数十 m ずれることが多く、`R_IN` = 50 m の判定を狂わせている可能性がある。

## 1. スタンドの地図（`docs/appear_tier_2026-10-05/stand_map.py`）

- **2026 年の ADS-B だけ**から作る（ランキングの出発便、`stand_dur` ≥ 600 s の stand run の最後の fix の位置）。2025 年のラベルの行とは
  重ならないので、2025 年の CV に情報は漏れない（自分の stand run が自分のスタンドの位置を決める循環も起きない）。
- (空港, 正規化した `STAND_mvt`) ごとに緯度・経度の中央値。5 便以上、ばらつき 40 m 以内のものだけ使う。

## 2. 群（`docs/appear_tier_2026-10-05/cv_standmap.py`、§6.2 と同じ 4 か月・a2・日単位の fold）

| 群 | 自スタンドの座標 |
|---|---|
| osm（今の appear4） | OSM のみ |
| fill | OSM、無ければ地図 |
| mapfirst | 地図、無ければ OSM |
| rand（対照） | mapfirst で地図を使うスタンドに、同じ空港の地図の別のスタンドの座標を入れる（行とスタンドの結びつきを壊す） |

すべて同じ fold・同じパラメータ・同じ乱数。appear の列の作り方（`R_IN`・`V_TAXI` など）は変えない。

## 3. 判定

fill と mapfirst のそれぞれについて、次をすべて満たせば合格。
1. 座標が osm と変わる行（「地図の行」: 自スタンドの座標の出どころが osm 群と違う行）で、osm に対する RMSE の差の日単位 95 % CI が
   改善側でゼロを除外する。
2. 同じ行で、rand に対しても改善側でゼロを除外する。
3. 全行で osm に対して悪化しない（95 % CI が悪化側でゼロを除外しない）。

両方合格なら、全行の osm に対する差が小さい（良い）ほうを採る。合格した群があれば、その地図で 2026 年の appear の列を作り直し、
`apply_delta.py --months` の A を差し替えて差分を作り直す（提出候補の更新。提出するかはユーザーと相談）。

## 4. 追記（2026-10-06 00:25 UTC、提出の前・CV の前に書く）: v38 の V2 残差の上でも効くか

v38（LB 242.0919）= v37 の 4 か月の残差を V2 の設定（`num_leaves` 255、`min_data_in_leaf` 300、`feature_fraction` 0.7、他は
`adsb_learn_cv.PARAMS`）に差し替えたもの（`claude/laughing-cray-vvwh0k` の `analysis/local_adsb_apply_swap.py`）。特徴量は同じだが、
木が大きくなった分 appear の効果の一部を先に拾っている可能性がある。そこで:

- **CV**（`docs/appear_tier_2026-10-05/cv_v2.py`）: 4 か月・a2・同じ fold で、すべて V2 の設定の 3 群: plain_v2、appear_fill_v2
  （fill の表）、shuffle_fill_v2（fill の表を §0 の並べ替えにかけた対照）。
- **判定**（plan.md §3 と同じ形）: appear_fill_v2 が plain_v2 と shuffle_fill_v2 の両方に対して、全行の RMSE の差の日単位 95 % CI が
  改善側でゼロを除外すること。
- **合格なら**: V2 の設定の P（plain）と A（appear fill）で差分を作り（`apply_delta.py v38 ... --months --standmap fill --params v2`）、
  v38 に載せて validate のうえ提出する（ユーザーの依頼）。**不合格なら**提出せず報告する。

# 外部データの再調査（2026-10-05）

観点は 2 つで、スクリプトは `offblock/` と `context/` にある。入出力パスはスクラッチ領域を指している。
[S] は実測または出典を読んだもの、[X] は推論。

## 結論

**新しく足す価値のある外部データはほぼ無い。** 伸びしろは、手持ちの adsb.lol の取り出し方（appear tier）と、OPDI の活用にある。

| 候補 | 判定 | 理由・見込み |
|---|---|---|
| **EUROCONTROL `All_Pre-Departure_Delay`** | **使用禁止** | AOBT − SOBT の日次集計そのもの [S]。学習データの日平均 max(0, BLOCK − SCHED) との相関は 0.998、比の中央値は全空港で 1.00。テスト月の集計ラベルにあたる。同じ系統の `Taxi-Out_Additional_Time` と `ATC_Pre-Departure_Delay` も禁止 |
| EUROCONTROL `ATFM_Slot_Adherence`（空港×日の規制対象出発の割合） | TRY | 離陸時刻ベースでラベルではない。genuine マッチ行で −0.35 秒（1 月⇔7 月の交差 CV）[S]。空港×日の特徴量の上限は −6 秒。ライセンスは EUROCONTROL の免責条件（出典の明記・非商用・改変禁止）。運営は open data 扱いと伝えられている（二次情報） |
| `Airport_Arrival_ATFM_Delay` | 低めの TRY | 月をまたぐと効かない [S]。2026 年 1 月の EHAM は学習範囲外の値 |
| X-Plane Gateway のスタンド座標 | TRY（小） | ライセンスは GPLv2 以降（原文は未確認）。座標の補完が効くのは EDDM（+13 ポイント）だけ。0.1〜0.3 秒 [X] |
| OPDI flight_events | 取り込み済み（別セッション） | v0.0.2 で 2026-07 まで。LTFM・LIRF・EGLL・EHAM は 0、LFPG はほぼ 0 [S]。opdi.aero にオープンライセンスの明示は無く、利用条件は ansperformance の免責（出典の明記・非商用・改変禁止）[S] |
| ADS-B Exchange（毎月 1 日の無料サンプル） | NO | LTFM の地上機は映っている [S] が、62 日中 2 日だけで、再配布禁止の規約 |
| adsb.fi / airplanes.live / theairtraffic | NO | 非商用・履歴無し、または既に adsb.lol に含まれる [S] |
| OpenSky REST・Trino | NO | 30 日制限、Trino は運営が不可 |
| AIP・NOTAM・ERA5・祝日／スト・機体 DB | NO | 非オープン、データに既にある、または weather_v2 で棄却済み |

**LTFM・LFPG・LIRF の地上の観測は、オープンなソースに無い。** これらの空港は、モデル側（スタンド、混雑の特徴量）でしか攻められない。

# 航空会社・機材・路線・便の同一性（operator_aircraft_route）

作成 2026-09-24。対象は 2025 年学習 DEP 2,085,047 行と ranking DEP 344,841 行（ARR 行は両方とも全列）。
再現スクリプトは `analysis/operator_aircraft_route_*.py`（00 → 02 → 04 → 06a → 06b の順でキャッシュを作り、他は独立）。
中間生成物は `data/interim/oar_*.parquet`（git 管理外）。外部データは OurAirports `airports.csv`（パブリックドメイン、
`data/external/ourairports_airports.csv`、`docs/external_data.md` に既載）。

用語: `r` = clip(y, 0, 3600) − 空港×スタンド（×時間帯）中央値（スタンドで説明できない残差）。`src` = `prc.clocks.with_clock_source` の出所ラベル（±60 秒）。
リーク判定: 本稿の特徴量はすべて ranking の DEP 行で観測可能な列（FLIGHT_mvt, CALLSIGN_flt, STAND_mvt, MVT/SCHED/AOBT_3 など）と
ARR 行の BLOCK 時刻、および学習側で fit したラベル統計の結合だけから作る。

__RESULTS__

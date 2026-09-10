# 外部データ台帳

賞金条件（docs/01 §4）は「使用した外部データが全てオープンにアクセス・利用可能で文書化されていること」「追加データセットはオープンソースライセンスで公開されているもの」。本ファイルがその文書。取得は全てコードから再現可能で、生データはリポジトリに含めない（`.gitignore`）。

| データ | 用途 | 出典 | ライセンス / 条件 | 取得コード |
|---|---|---|---|---|
| METAR（ASOS アーカイブ、10 空港、1 時間毎） | 気温・露点・風・降水・視程・現在天気 → 防除氷ゲート、低視程 | Iowa Environmental Mesonet, `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` | **パブリックドメイン**（「as-is」の履歴観測コレクション）。1 リクエスト/秒/IP のスロットル。米国外は降水量が欠ける場合あり | `src/prc/weather.py: fetch_metar()` |
| Open-Meteo 履歴 API（ERA5 / IFS） | METAR の欠損補完、降雪量 | `https://archive-api.open-meteo.com/v1/archive` | **CC BY 4.0**（出典表示が必要）。無償枠は非商用 10,000 コール/日 | 未使用（必要時に `weather.py` に追加） |
| OurAirports `airports.csv` / `runways.csv` | 滑走路端座標 → スタンドから滑走路までの距離特徴量 | `https://ourairports.com/data/` | **パブリックドメイン** | 未使用（Phase 3 で判断） |
| OpenStreetMap `aeroway=*` | 誘導路グラフ、`parking_position` の座標 | `https://www.openstreetmap.org/` | **ODbL**（表示 + 派生データベースは同ライセンス） | 未使用（Phase 3 で判断） |

**使わないと決めたもの**
- EUROCONTROL PRU の月次 Taxi-out Additional Time のうち **テスト期間（2026 年）の値**。オープンだが、テスト期間の目的変数の空港別月次集計に相当し趣旨に反する（docs/01 U12）。2025 年以前の値は空港特性の理解にのみ使用。
- ADS-B / OpenSky 履歴からのオフブロック時刻の再構成。趣旨違反かつライセンス不適合の恐れ。
- 2025 に密で 2026 に疎になる外部特徴量（他チームは OPDI で被害）。採用前に被覆率と敵対的検証で確認する。

## 検証記録

- **2026-09-10**: IEM から 10 空港・2025-01-01〜2026-08-01 の METAR を取得（138,446 観測時間、欠損はほぼゼロ）。2026 年 1 月の防除氷条件（気温 ≤ 3°C かつ 降水 or 凍結性現在天気）の時間割合は LSZH 18.0%、EHAM 12.8%、EDDM 11.2%、LTFM 9.5%、EDDF 8.3%、LFPG 1.5%、LEMD 0.4%、EGLL 0.1%、LEBL 0.0%、LIRF 0.0%。他チームの公開報告（docs/03 §8: LSZH 18.0%、EHAM 13.4%、EDDM 11.2%、LTFM 9.9%、EDDF 8.3%、LEBL/LIRF 0%）とほぼ一致し、両者を相互検証できた。

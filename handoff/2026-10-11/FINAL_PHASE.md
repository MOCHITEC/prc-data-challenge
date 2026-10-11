# 最終フェーズの連携メモ（beautiful-davinci / 外部知見、2026-10-11 15:00 UTC）

統合役（データ分析 / laughing-cray）と、ほかのスレッド向け。こちらからは別セッションへ直接メッセージを送れないので、ここに置く。
返事は各ブランチの docs か handoff/ に書いてもらえれば読む。

## 事実
- **締め切りは 2026-10-14 10:00 UTC に延長**（https://prc-data-challenge-2026.netlify.app/ のトップページ）。
- 主催者が 2026-10-08 11:04Z に `final_ranking.parquet` / `final_submitting.parquet` を公開（バケット prc-2026-datasets）。
  - 2026 年 1・2・6・7 月の 670,790 行。今の LB の 344,841 行（1・7 月）を含み、2・6 月の 325,949 行が新しい。
  - 列は ranking と同じ。
- 最終提出は `<team>_final.parquet` の 1 ファイル（ranking ページの "Final Submission"）。
- **バケットの `resilient-hedgehog_final.parquet`（10-09 16:02 にアップロード、採点は主催側のエラーで Failed）は弱い。**
  - 1・7 月の行で v47 = C15 と RMS 149 s 違う。ADS-B の無い LTFM でも RMS 65 s 違うので、土台そのものが C15 と別物。
  - ADS-B のある行の差は、ADS-B 補正の符号を反転したものと相関 0.69 → ADS-B の層が入っていない。
  - このまま最終になると大きく損をする。誰が作ったものか、書いておいてほしい。
- **v48 = C15 + 12 か月 appear + EDDF の間引き → LB 234.4099**（C15 から −0.073 s、チームの最良）。
  - ファイルは `handoff/2026-10-08/C15_appear12_cov26eddf.parquet`。
  - 差分だけなら `appear12_cov26eddf_delta_vs_c15.parquet`（base に依らない）。

## こちらで進めていること
1. adsb.lol の 2026 年 2・6 月（58 日）を取り込み中。stage 1 の出発便は、2・6 月だけ final_ranking から読む
   （`scripts/adsb_stage1_feasibility.our_departures`）。見込みは 3〜4 時間。
2. 取り込めたら、2・6 月の feat / appear(fill) を 1・7 月と同じ形で作る（`feat_2026fj.parquet`、`appear_fill_2026fj.parquet`）。
3. 2・6 月の行の ADS-B 補正を作る。土台の予測 b（ADS-B の層の前）を受け取れれば、b + c_A12(b)（EDDF は c_A12thin）を返す。

## 統合役にお願いしたいこと
- (a) 2・6 月の行（と、できれば 1・7 月の行も）の、ADS-B の層をかける前の土台 b を置いてほしい（MVT_ID_mvt, b）。
- (b) 最終ファイルを誰が組み立てるか決めてほしい。おすすめは次の組み合わせ:
  - 1・7 月の行: v48（または、統合役の v48/v49 系の最良にこちらの差分を足したもの）。
  - 2・6 月の行: 土台 b + ADS-B の層 + overlay 類。
- (c) apron など統合役の ADS-B の層を 2・6 月にもかけるなら、こちらの feat / joined ファイルを使ってよい（置き場所はここに追記する）。

## 追記（15:20 UTC）: バケットの final.parquet の土台は使えない
- v49 = 「final.parquet の 1・7 月の行 + こちらの ADS-B（`final_adsb.py`、12 か月 appear、EDDF は間引き版）」→ **LB 248.6458**。
  v48（234.4099）より 14 秒悪い。final.parquet の土台は C15 の土台よりかなり弱い。ADS-B 抜きのままなら、さらに悪いはず。
- **2・6 月の行も、C15 / v48 と同じ完全なパイプライン（5 メンバーの blend、model B、LIRF の overlay、ucmix、posjoin、LOBT の切り詰め）で
  予測する必要がある。** これは統合役の学習済みモデルでしかできない。
- ADS-B の層は、こちらの `docs/appear_tier_2026-10-05/final_adsb.py` で、どの土台にも後からかけられる。
  - 入力: ADS-B の層の前の予測（MVT_ID_mvt, TAXITIME_SEC_mvt）。
  - 出力: b + c_A12(b)（EDDF は c_A12thin）。LTFM と LIRF の unmatched の行、LOBT 境界の行は触らず、最後に LOBT で切り詰める。
  - 1・7 月の行で、以前の計算と一致することを確認済み。
  - 2・6 月の特徴量（feat_2026fj / appear_fill_2026fj）は、取り込みが終わりしだい作って、ここに置き場所を書く。

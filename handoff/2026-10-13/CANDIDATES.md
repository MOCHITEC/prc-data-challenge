# 最終 4 か月セット向けの部品（beautiful-davinci / 外部知見、2026-10-13 01:00 UTC）

`docs/final_phase_2026-10-11/NO_FINAL_UPLOAD.md`（model-b、1ebc0fd）に従う。
- **チームのバケットには何もアップロードしない。** 最終提出は RPC 実装スレッドだけが行う。
- このスレッドのバケットへのアップロードは、10-11 の v48（234.4099）と v49（248.6458）だけ。どちらも指示の前。
- 10-09 の `resilient-hedgehog_final.parquet` は、このスレッドが作ったものではない。

統合役の最終案 v3（laughing-cray 71a7d27、`docs/final_2026-10-11/README.md`）は、次の形になっている:
- 1・7 月 = v47
- 2・6 月 = blend → 統合役の ADS-B の層（2 月だけ）→ lane-B → LIRF の overlay → LOBT の切り詰め

これに対して、次の 3 つの部品を用意した。どれも v3 の作りにそのまま足せる。

## 部品 1: 6 月の ADS-B（統合役の `adsb_layer.py` 用）— 6 月の行に ADS-B の層を入れる

- **何か**: 統合役の `adsb_layer.py feat` の出力。6 月の 30 日分で、callsign の特徴量、位置での突き合わせ、その特徴量、長い停止。
  統合役の環境では、6 月の取り込みが 2026-06-01 で止まっている。
- **置き場所**: `handoff/2026-10-13/june_adsb_for_adsb_layer/{feat,pos}/`（19 MB、md5 は同じフォルダの `MD5SUMS`）。
- **使い方**:
  1. `feat/*` を統合役の `adsb_out/feat/` に、`pos/*` を `adsb_out/pos/` にコピーする。
  2. `adsb_layer.py apply --base <ADS-B の層の前の 670,790 行> --out ...` を回し直す。
  3. `done_*` が入っているので、`feat` は回し直さない。
- **作り方**:
  - stage 1 の入力は、こちらで取り込んだ adsb.lol の 2026 年 6 月（30 日、b53bdaa 以降の `adsb_ingest_2025.py run --days`）。
    出発便は final_ranking から読む。
  - 統合役のブランチ 71a7d27 の `docs/final_2026-10-11/adsb_layer.py` を、パスの 4 行（SCR、FINAL、HERE、RANK）だけ書き換えて
    `feat --days 2026-06-01..30` を回した。コードは統合役のもの。
- **検証**:
  - 同じ手順で作った 2026-02-01 の `feat` は、こちらの `adsb_learn_features.day_features` の出力と完全に一致した（4,387 行）。
  - 地上の fix がある行は 1 日 4,100〜4,600 行。
  - **2026-06-11 は adsb.lol 側でデータが欠けている**（prod も staging も地上の fix が約 1,900 件しかない）。その日の行は土台のまま。

## 部品 2: appear の層の増分（2・6 月の行）— 統合役が作り直した ADS-B の層に、appear の層が入っていない

- **分かったこと**: 1・7 月の行で次が成り立つ（b = v38 から戻した b、248,001 行）。
  - v48 − v47 = (c_A12 − c_P12)(b) + EDDF 分 − (c_A4 − c_P4)(b)
  - 相関 0.989、差の RMS 4 s（丸めと LOBT の切り詰めの分）。
  - つまり v47 の ADS-B には「12:v2 + apron などの層」に加えて、**4 か月 appear の項（RMS 44 s、v39 で LB −2.57 s）** が入っている。
  - 統合役が作り直した層（G12A / G7A / GL4 / AIR / POS / RULE）に appear は無い。統合役の由来表にも v39 の段がない。
  - 1・7 月で v47 と RMS 52 s 違う理由の大部分は、これだと考えられる [推測]。
- **足し方**: 統合役の環境で回す。

      git checkout origin/claude/beautiful-davinci-ehugk4 -- docs/appear_tier_2026-10-05/final_adsb.py handoff/2026-10-11/adsb
      SCRATCH=$PWD/handoff/2026-10-11/adsb POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
        .venv/bin/python docs/appear_tier_2026-10-05/final_adsb.py <b: ADS-B の層の前の blend> <out> --months 02,06 --increment

  - 出力 − b が増分。これを v3 の 2・6 月の行に足す。
  - 増分は c_A12 − c_P12。EDDF は c_A12thin − c_P12。LTFM、LIRF の unmatched、LOBT 境界の行は触らない。
  - 足したあとに LOBT の切り詰めをかけ直す。
  - **b は統合役が `adsb_layer.py apply --base` に渡したのと同じ土台にする**（ベースをそろえる規則）。
  - 必要なのは `scripts/adsb_learn_features.py`（どのブランチでも同じ）と lightgbm / polars だけ。
  - 4 か月分の特徴量とモデルは `handoff/2026-10-11/adsb/learn/`（md5 は `handoff/2026-10-11/adsb/MD5SUMS`）。
- **検証**:
  - 1・7 月で、増分 − 4 か月の項 = v48 − v47 を確かめた（上）。
  - v48 − v47 は LB で 234.4831 → 234.4099。appear の層そのものは v39 で −2.57 s。
  - [推測] 2・6 月にこれが無いと、その行は 1・7 月より appear の分だけ悪くなる。
- `final_adsb.py` の md5: `2d777569ee77317b640409d39478e53a`。appear の design は中に写したので、`adsb_appear.py` は要らない。

## 部品 3: 1・7 月を v47 から v48 にする差分

- **置き場所**: `handoff/2026-10-13/janjul_v48_minus_v47.parquet`（MVT_ID_mvt と delta、240,976 行、md5 `b7944c89037bc445e1e50a5792625433`）。
- **使い方**: 1・7 月の行が v47（md5 `efda7376eafa6efae178ceb92ff13007`）そのものの最終案に、MVT_ID_mvt で足す。
- **検証**: LB で v47 234.4831 → v48 234.4099。v48 のファイルは `handoff/2026-10-08/C15_appear12_cov26eddf.parquet`
  （md5 `8ee6679f43ead60db1a1aec7876caf09`）。

## 決めてもらうこと（適用していない）: 1・7 月の時刻のずらし
- final_ranking の 1・7 月は MVT が ±12 s 動く（標準偏差 4.9 s）。到着便では BLOCK が逆向きに動く（相関 −0.99）。
- 出発便の正解が「ずれ × 2」動いているなら、v47 / v48 をそのまま使うと約 +97 s² の誤差が増える（4 か月全体で +0.1 s 程度）。
- anchor に合わせて「+ ずれ × 1」を足すと、どの場合でも損は 24 s² 以下で済む（いちばん悪い場合で比べて最も損が小さい）。
  ただし LB では確かめられない。手で足すことが審査でどう見られるかも含めて、統合役と RPC で判断してほしい。

## 追記（2026-10-13 01:40 UTC）: 統合役の final_from_v48 に、2 月のスタンドの補正（appear の増分）を足した候補

**ベース**: laughing-cray 558c736 の `handoff/2026-10-13/final_from_v48.parquet`（md5 `c7ffa31edc2a4de5f1837682d90e14cb`）。
- 1・7 月 = v48。
- 2 月 = 統合役の自前の ADS-B の層だけ。appear の層が無い。
- 6 月 = こちらの A12 の層（appear 込み）。

**候補**: `handoff/2026-10-13/final_from_v48_plus_feb_appear.parquet`（md5 `6f2cd1af92d8e636c2159f4fd7e5413c`、670,790 行、validate OK）
- ベースの 2 月の行（103,311 行）に、appear の増分を足しただけ。1・6・7 月は変えていない。
- 変わった行の RMS 40 s、Σd²/N 246。

**増分の作り方**:

      SCRATCH=$PWD/handoff/2026-10-11/adsb .venv/bin/python docs/appear_tier_2026-10-05/final_adsb.py \
        pre_adsb_base_blend5_final.parquet incr_4m.parquet --months 01,02,06,07 --increment

  - 土台 b は統合役の `pre_adsb_base_blend5_final.parquet`（md5 `38edd5a6a6ab0328de841936cffa907a`）。統合役が ADS-B の層に渡したものと同じ。
  - 増分 = 出力 − round(b)。
  - ベースに足したあとで、LOBT の切り詰め（±3606 s）をかけ直した。
  - 2・6 月の増分の表: `handoff/2026-10-13/feb_jun_appear_increment_on_blend5.parquet`（md5 `fe3aedc0467fc0777ea2326c714f2a35`、MVT_ID_mvt / month / delta）。

**検証（1・7 月の行で、v48 との近さ）**: 統合役の chain（自前の層だけ）は v4b、0220c69 の `cand_v4b_refit4m_oursJanFebJul_A12Jun.parquet`。

| | v48 との RMS 差 | 差の中央値 |
|---|---|---|
| chain（自前の層だけ） | 53.2 s | 16 s |
| chain + 増分 | **43.4 s** | **13 s** |

- 増分を足すと v48 の作り方に近づく。残りは時刻のずらしと、層の作り直しの近似の分。
- v48 = 統合役の層 + appear の増分、という関係は 1・7 月で確かめた（相関 0.989）。

**6 月の扱い**:
- 今の 6 月は A12 の層（appear 込み）なので、増分は足していない。
- 統合役が 6 月を自前の層に替えたとき（v5）は、上の表の 6 月の delta を足すと v48 と同じ作り方になる（土台は同じ blend5）。

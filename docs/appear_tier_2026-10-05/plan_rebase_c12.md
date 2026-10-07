# 12 か月 appear を C12（v42）の base に合わせて作り直す — 事前登録（2026-10-07、CV の前に書く）

依頼: `claude/model-b-and-arrival-features` の `docs/handoff_2026-10-07/REQUEST_rebase_on_C12.md`。

## 0. 事実と仮説

- v46 = C12 + 12 か月 appear の差分表（v39 の b で評価）→ LB +0.026 s（見込み −0.85〜−0.95 s）。月を丸ごと外す CV
  （`plan_lmo12.md`）でも 12 か月版は効くので、2025 年の中の問題ではない。
- C12 は v40 の上で base 側（C10・PB のメンバー・NM-blind・MLP-L6）を差分として足したもので、ADS-B の層（plain の残差、apron、
  4 か月 appear）は古い b で評価したまま。差分表の行で b は RMS 22.9 s 動いた（依頼元の推定: b_C12 ≈ b_v39 + (v42 − v43)）。
- **仮説 S（古い b）**: appear の差分は b に敏感（b の戻し方を変えるだけで差分が RMS 15.7 s 動く）。base が動いた後で古い b の
  差分を足したので効果が消えた。C12 の b で評価し直せば効く。

## 1. CV での再現（`docs/appear_tier_2026-10-05/cv_stale.py`）

`cv_lmo12.py` と同じ行・モデル（1・7 月を丸ごと外す。P12 / A12 は他の 11 か月、P4 / A4 は他の 3 か月、V2、fill）。
検証の月の行で、モデルを 2 つの b で評価する: b（a2 の OOF、今まで通り）と b' = b + w·(b_alt − b)（base が動いた後）。
b_alt = `flwin_both-33ed7358b9` の OOF（fl_ の窓の修正 = C10 に近い base の変更）、w は appear の行で RMS(b' − b) = 23 s になる値。
未マッチ行は b' = b。参考に b_alt = `tc_drop-86a3693e18`（w = 1、RMS 26.7 s）でも同じ表を出す（判定には使わない）。

作る予測（すべて検証の月の行）:
- **C0**（C12 に当たる）= b' + c_P12(b) + (c_A4 − c_P4)(b)　… base だけ動き、ADS-B の層は古い b のまま
- **old**（v46 に当たる）= C0 − (c_A4 − c_P4)(b) + (c_A12 − c_P12)(b)
- **req**（依頼の式）= C0 + (c_A12 − c_P12)(b') − (c_A4 − c_P4)(b')
- **swap**（入っている 4 か月の項をそのまま抜く）= C0 − (c_A4 − c_P4)(b) + (c_A12 − c_P12)(b')
- **full**（ADS-B の層をすべて b' で作り直す）= b' + c_A12(b')
- 参考: **C0full** = b' + c_P12(b') + (c_A4 − c_P4)(b')

## 2. 判定（1・7 月をまとめた全行、日単位 bootstrap）

1. **S の確認**: old 対 C0 が −0.3 s より悪い（古い b の差分では効かない＝ v46 を再現）。そうでなければ S は棄却し、
   作り直しても LB の外れは直らない見込みとして報告する（提出しない）。
2. S を支持したとき、swap・req・full のうち C0 に対する改善が最大のものを選ぶ（95 % CI が改善側でゼロを除外することが条件）。
   ただし full は 2026 年では apron の層を作り直せない（apron のモデルはこちらに無い）ので、full が最大でも swap との差が
   0.1 s 未満なら swap を選ぶ。

## 3. 2026 年の候補（`docs/appear_tier_2026-10-05/apply_rebase_c12.py`）

- v38・v39・v42・v43 はチームのバケットのファイル。b_v39 は v39 から 4 か月の P4 と appear で戻す（前回と同じ）。
  ファイルに入っている 4 か月 appear の項 = v39 − v38（そのまま）。b_C12 = b_v39 + (v42 − v43)（appear の行）。
- swap: new = v42 − (v39 − v38) + (c_A12 − c_P12)(b_C12)。req: new = v42 + (c_A12 − c_P12 − c_A4 + c_P4)(b_C12)。
- 上書きの行（LIRF の copy band・time-zone class、LOBT の境界に乗った行）は依頼元と同じく触らず、最後に LOBT の切り詰めを
  かけ直す。validate のうえ `handoff/2026-10-07/` に置く。提出は空き番号と当日の枠を確かめてから（依頼の 3）。

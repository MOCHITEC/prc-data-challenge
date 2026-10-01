# 実験ログ

- `experiments.jsonl`: 実データの実験記録（1 行 1 実験）。`run_cv()` が追記する。git 管理。
- `pipeline.jsonl`: `python -m prc.pipeline` の 1 実行分の記録（設定、CV 要約、予測時の内訳、提出ファイル、所要秒）。git 管理。
- `synthetic_smoke.jsonl`: 合成データでパイプライン全体を通した記録。実データの成績とは無関係（RMSE の水準は生成器の作り物）。混同を避けるため別ファイルにしている。`--synthetic` を付けて実行すると CV 記録も実行記録もこのファイルに入る（付け忘れると実データのログが汚れる）。
- `submissions.jsonl`: 提出ログ（版番号、実験 ID、ローカル CV、LB、1 日 5 回カウンタの根拠）。git 管理。
- `oof_*.parquet`: fold 別の out-of-fold 予測。サイズが大きいため git 管理外（`.gitignore` の `*.parquet`）。

実験の再現に必要な情報は各行の `config` と `git_sha` に入る。

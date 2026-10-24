# モデル・特徴量・config を変更するときのチェックリスト（stg と prod に同じ挙動で反映する）

背景と仕組みは `env_parity.md`。手順を上から順に行い、各項目の確認をそろえてから prod に出す。

## 1. 変更の前

- [ ] 変更の種類を決める: 特徴量／モデル（バンドル）／config（較正・印・閾値）／挙動フラグ／フロント。
- [ ] 2026 年のデータを選択に使っていない（テストセット②）。市場オッズ・人気を特徴量にしていない。
- [ ] 回収率・較正に使う統計は as-of（日付が厳密に前）。

## 2. 特徴量・モデルの整合（推論に出す前）

- [ ] **列順の検証**: 学習側と推論側が同じビルダーを使い、バンドルの `manifest.json` の `feature_names`／`n_features` と、推論で組み立てた列が**名前・順序まで一致**（v3 は 301 列、v5 は 1,142 列。`v3_serving` の検証）。
- [ ] **推論一致検証**: 学習時の予測と、推論経路（serving）の予測が同じレースで一致（許容誤差内）。バックフィルしたレースで数十 R 比べる。
- [ ] バンドルを公開する場合は `publish_model`（`model_registry`）で sha256 つきの `manifest.json` を作る。ファイルを手で置換しない。
- [ ] ライブラリのバージョン（`libraries`）が推論環境（ホスト／Cloud Run／Jobs のイメージ）と互換（`check_compatibility`）。

## 3. 反映（同時に・同じ内容で）

- [ ] 変更を**コミット**する（作業ツリーの未コミット変更を混ぜない）。
- [ ] stg（ホスト）へ `stage_deploy create --ref <commit> ...`（JST 03:05 に自動反映）。config・コードは**ファイル単位**で、バックアップとロールバックつき。
- [ ] モデル／バンドルを切り替えるときは、環境変数（`KEIBA_V3_BUNDLE` など）ではなく `config/behavior_flags.json` か `latest.json` の切替で行う（二重管理にしない）。
- [ ] 窓の後（JST 04:10）の `env_parity_check` が一致（Slack に差が出ていない）。
- [ ] 対象日の**予測を再生成**（古いモデルの `race_predictions` を残さない）。画面の `model_version` が新しい版。
- [ ] stg の画面で確認: 勝率／複勝率／期待値／めぐAI印／推奨馬券が出る。

## 4. prod へ（UI）

- [ ] **同じコミット**で `bash scripts/ops/vps/promote.sh --commit <commit>`。前検査（stg のソースと昇格コミットの frontend が同じ、フラグのずれなし）が通る。
- [ ] 昇格後、`env_parity_check compare` が一致（`frontend.source_digest` が stg と同じ、`public_env` が同じ）。
- [ ] prod の画面（`https://app.megu-keiba.com`）で同じレースを開き、stg と同じ数値・印・推奨が出る。

## 5. ロールバック

- [ ] コード・config: ステージ時に作られたバックアップから戻す（`/tmp/rollback_*.sh` またはランナーの自動ロールバック）。
- [ ] モデル: `latest.json` を前の版に戻す（`set_latest_version`）。バンドルのパスを使う環境では前のパスに戻す。
- [ ] UI: `bash scripts/ops/vps/rollback.sh`。
- [ ] 戻したあとも `env_parity_check` で stg と prod が一致していること、予測を再生成したことを確認。

## 6. 記録

- [ ] 採用した版（バンドル名・`n_features`・config のハッシュ）を決定ログ（`training_pipeline_decisions`）に記す。マニフェスト（`python -m src.ops.release_manifest print`）を添えると再現できる。

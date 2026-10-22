# 学習済みモデルの配布方針（重みは公開リポジトリに置かない）

方針（2026-10-22 確定）: **学習済みの重み・符号化器は git で管理しない。** 公開リポジトリには、コード・設定・仕様だけを置く。
最終形（`decoupling_plan.md`）では、ホスト PC（研究専用）が成果物を GCS に置き、各環境が取りに行く一方向にする。

## 対象（git から外した 11 ファイル。実ファイルは各環境に残る）
| ファイル | 役割 | 読む場所 |
|---|---|---|
| `models/keiba_model.pkl`、`label_encoders.pkl`、`encoder.json` | 旧系統の予測モデルと符号化器 | `race_prediction_service`、`model_sync` ほか |
| `models/final_odds_bundle.json` | 想定オッズ（最終オッズ予測）のバンドル | `final_odds_predictor` |
| `models/pace_predictor/model_1f.txt`、`model_3f.txt` | ペース予測（LightGBM） | `train_pace_model` ほか |
| `config/growth_index/model.txt` | 成長度の木モデル | `growth_index` |
| `config/expected_megu/model.json` | 想定めぐ指数の補正量 | `expected_megu` |
| `config/pace_type_models.json` | 展開タイプの確率モデル | `pace_type` |
| `config/tracking_difficulty_v2/model.txt`、`model_demand.txt` | 追走難度 v2 | `tracking_difficulty_v2` |

各ディレクトリの `.gitignore`（`models/.gitignore`、`config/.gitignore`、`config/{growth_index,expected_megu,tracking_difficulty_v2}/.gitignore`）で、再度の追加を防いでいる。

## 配布の流れ
1. **作る（ホスト、研究専用）**: 学習・再学習で成果物ができる。
2. **公開（ホスト → GCS）**: `python -m src.ops.model_publish publish <component> ...` が `models/<component>/<version>/` と `latest.json`（sha256）を GCS に置く。同じ内容なら再アップロードしない。コンポーネント名は `config/gcp_job_map.json` の `models`。
3. **取得（各環境）**: `python -m src.ops.model_publish fetch <component>` が、`latest.json` → `manifest.json` → ファイルごとの sha256 を検証してから置く。検証に失敗したら何も置かない。
4. **確認**: `env-parity`（`docs/operations/env_parity.md`）のリリース・マニフェストで、stg と prod が同じモデル版か照合する。

## 環境ごとの扱い
- **ホスト PC（稼働中の stg）**: 現状の実ファイルをそのまま使う（git の管理から外れただけで、ファイルは消えていない）。`git pull` はしない運用なので、以後の差し替えは `stage_deploy` かホストでの学習結果の配置で行う。
- **開発 PC・CI・新規 clone**: 重みは無い。重みを使うテストは**ファイルが無ければ skip** する作りになっている（`pytest` は重み無しで通ることを、クリーンな worktree で確認済み）。重みを使う動作確認は、GCS から `fetch` するか、ホストから個別に取得する。
- **Cloud Run / Jobs / VPS**: `fetch` で取得する（`decoupling_plan.md` の D2）。イメージに焼かない。

## やってはいけないこと
- 重みや符号化器を `git add -f` で戻す。`secret_scan.sh` の検査対象ではないので、レビューで止める。
- 履歴に残った過去の重みは、履歴の書き換えで除去する（`history_purge_runbook.md`）。

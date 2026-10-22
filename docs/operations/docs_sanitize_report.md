# 文書の整理（削除とマスク）の記録

公開リポジトリに載せない方針（`docs/operations/security_hygiene.md`）に沿って、内部情報を含む文書を伏せ、不要な文書を削除した。履歴の書き換えはしていない（削除は `git rm`、マスクは置換のみ）。実値は書かない。

## 置き換えの規則（プレースホルダ）
| 内容 | 表記 |
|---|---|
| ホストのリポジトリ／ホーム | `<HOST_REPO>` / `<HOST_HOME>` |
| 旧トンネルのホスト名 | `<OLD_TUNNEL_HOST>` |
| 内部 API／確認用／モニター／stg のホスト名 | `<API_INTERNAL_HOST>` / `<PROD_TEST_HOST>` / `<MONITOR_HOST>` / `<STG_HOST>` |
| Cloudflare トンネル名 | `<TUNNEL_STG>` `<TUNNEL_PROD>` `<TUNNEL_VPS>` `<TUNNEL_DEV>` |
| VPS の ssh エイリアス | `<VPS_ALIAS>` |
| GCS バケット／GCP プロジェクト | `<BUCKET>` / `<GCP_PROJECT>` |
| サービスアカウントのメール | `<SA_API_EMAIL>` など |
| GCP の送信元 IP | `<GCP_EGRESS_IP>` |
| VPS の IP・SSH ポート・管理ユーザー | `<VPS_IP>` / `<SSH_PORT>` / `<VPS_ADMIN_USER>`（前回の対応） |

実値は、コミットされない `docs/local/`（`.gitignore` で除外）か各自の環境に置く。

## 削除した文書（7 件）
| ファイル | 理由 |
|---|---|
| `docs/todos/verify/results/summary-*.md`（5 件） | 検証の出力の残骸。実行環境の情報を含む。再実行は `docs/todos/verify/run_all.sh` |
| `docs/operations/stg_cron_instructions_20261013.md` | 一回限りの手順。現行の時間割は `stg_cron_schedule_20261014.md` |
| `docs/operations/feature_accumulation_cron_20261014.md` | 調査メモ。実装済みで、設計は `docs/html/operations/raceday_prediction_workflow.html` に移行済み |

参照していた `scraping_sla_audit_20261013.md`、`docs/todos/README.md`、`docs/html/index.html` のリンクは直した。

## マスクした文書（41 ファイル、置換 140 箇所）
| ファイル | 置換数 |
|---|---:|
| `docs/operations/vps_ui_deploy.md` | 42 |
| `docs/operations/decoupling_plan.md` | 13 |
| `docs/operations/service-endpoints.md` | 9 |
| `docs/decisions/MASTER.html` | 6 |
| `docs/operations/maintenance_deploy.md` | 5 |
| `docs/operations/raceday_weight_trigger.md` | 5 |
| `docs/decisions/AREA-06-data.html` | 4 |
| `docs/operations/go_live.md` | 4 |
| `docs/environment-stg.md` | 3 |
| `docs/operations/env_parity.md` | 3 |
| `docs/operations/gcp_jobs_migration.md` | 3 |
| `scripts/ops/vps/vars.env.example` | 3 |
| `docs/decisions/DEC-017-gcs-bucket-strict-validation.html` | 2 |
| `docs/html/data/DATASET_CATALOG.html` | 2 |
| `docs/html/modeling/chapters/index_variance.html` | 2 |
| `docs/html/modeling/chapters/jockey_trainer_roi.html` | 2 |
| `docs/html/modeling/chapters/lap_forecast.html` | 2 |
| `docs/html/modeling/chapters/pedigree_roi.html` | 2 |
| `docs/operations/ads_architecture.md` | 2 |
| `docs/operations/architecture_target.md` | 2 |
| `docs/operations/entries_timeline.md` | 2 |
| `docs/operations/gcp_smoke_results.md` | 2 |
| `docs/todos/verify/results/summary-20261010-163828.md` | 2 |
| `.env.prod.example` | 1 |
| `.env.stg.example` | 1 |
| `docs/decisions/DEC-025-gcs-postgresql-sync-and-monitoring.html` | 1 |
| `docs/git_management/todo/race-quality.md` | 1 |
| `docs/html/data/database_schemas.html` | 1 |
| `docs/html/design/FULLSTACK_ARCHITECTURE.html` | 1 |
| `docs/html/index.html` | 1 |
| `docs/html/modeling/chapters/drift_diagnosis.html` | 1 |
| `docs/html/modeling/chapters/megu_excusable.html` | 1 |
| `docs/html/modeling/chapters/yosou_procedure.html` | 1 |
| `docs/operations/cloud_run_api.md` | 1 |
| `docs/operations/frozen_tracking.md` | 1 |
| `docs/operations/maintenance_window.md` | 1 |
| `docs/operations/megu_v3a_rollout.md` | 1 |
| `docs/operations/stg_frontend_prod_build.md` | 1 |
| `docs/todos/verify/results/summary-20261002-163452.md` | 1 |
| `docs/todos/verify/results/summary-20261004-135912.md` | 1 |
| `scripts/ops/vps/ssh_config.example` | 1 |

## 残したもの（理由つき）・要確認
| 対象 | 状況 | 理由／次の一手 |
|---|---|---|
| `scripts/server/tunnel_tcpexposer*.sh`、`tunnel_stg_ui.sh`、`service_start.sh` ほか動くスクリプト | 旧トンネル名・ホストのパスを含む | `service_start.sh` などが読む。動作を変える改修が要るため未変更。旧トンネル（停止済み）の処理を外して整理する候補 |
| `docs/git_management/**`（35 件） | 古いブランチ運用の記録 | `update-requirements` スキル（`collect_state.py`）と `docs/html/index.html` が参照。スキル側を直してから削除する候補 |
| `docs/operations/vps-gcp-responsibilities.md`、`deployment-vps-vs-gcp.md`、`cost-estimate.md` | 2026-10-02 の旧設計 | `AREA-05`／`DEC-023`／`index.html` などから参照。現行は `architecture_target.md` と `decoupling_plan.md`。参照を直して削除する候補（旧設計の注記を先に付ける案もある） |
| `docs/html/modeling/chapters/training_pipeline_decisions.html`（約 4 箇所）、`v6_features.html` | 実値あり | v6 の作業が編集中のため未変更。作業完了後にマスク |
| `docs/html/book.html` | 実値が残っている（再生成前） | 上のマスク後に `python -m src.scripts.docs.build_docs_book` で再生成。v6 の未コミット変更と衝突するため、作業完了後に |
| `scripts/gcp/*.sh`、`scripts/cron/*.sh`、`deploy/gcp/**` | 実行系にホスト名・バケット名の既定値 | 動くコードのため未変更。環境変数化の候補 |
| `docs/requirements/data/schemas/observed/**`、`dev_fixture/real_day.tar.gz` | netkeiba 取得データの抜粋 | 公開は了承済み。**規約上の再確認が必要**（法的リスクの確認事項） |

## 機械検査
- `bash scripts/dev/secret_scan.sh --tracked` は、秘密らしい文字列なし。
- `tests/scripts/test_host_dependency_scan.py` は、別作業（モニターの prod 状況）が追加した 3 ファイルがホストのローカルデータを読むため、ベースラインの更新待ち（本作業とは無関係）。

## 履歴について
過去のコミットには、マスク前の値が残る。取り消すには履歴の書き換えと強制 push が必要で、clone した人には効かない。**実値を変えられるもの（IP、トンネル、Webhook、サービスアカウントの鍵）は、変更・再生成するのが確実。**

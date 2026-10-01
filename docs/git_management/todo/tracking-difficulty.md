# TODO: feature/tracking-difficulty

**対象領域**: 追走難度
**関連ドキュメント**: [../feature-tracking-difficulty.md](../feature-tracking-difficulty.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/tracking-difficulty`: ページ表示
- `/api/race/{race_id}/tracking-difficulty`: 追走難度・ペース・位置取り（calculated_data 事前計算を返す。`refresh=true`で再計算）
- `POST /api/race/{race_id}/tracking-difficulty/precompute`: バッチ計算してstorageに保存（推論ワーカー相当）
- `/api/tracking-difficulty/status`: 事前計算ストアの件数・パス（読み取り専用）
- `POST /api/tracking-difficulty/train`: 追走難度モデルの学習実行
- Flask v1 (`/api/v1/races/<race_id>/tracking-difficulty` + `.../precompute`) と2026-09-30にパラメータ完全パリティ化済み
  （legacy/v1どちらも同じ`tracking_difficulty_service.get_or_compute`を利用。v1側の`HybridStorage()`直接生成も
  `_get_storage()`シングルトンに統一済み）。2026-10-01に自動テスト `tests/api/test_tracking_difficulty_legacy_v1_parity.py`
  を追加し、値の一致を継続的に検証できるようにした。
- 事前計算カバレッジ（未計算率）の計測スクリプト: `src/scripts/maintenance/measure_tracking_difficulty_coverage.py`
  （2026-10-01追加。対象レース定義は既存バッチと同じ `race_shutuba` ベース、中央競馬・直近365日）

## 目標（推測）

ユーザが「このレースは差しが届きやすいか、逃げ有利か」を事前に把握できるようにすること。
legacy/v1どちらの画面から見ても同じ追走難度が表示される一貫性も目標に含まれると推測される。

## このラインまで実装できたらブランチを消してよい

- ユーザがどの画面（legacy/v1どちらの実装元）から見ても追走難度の値が食い違わない
- 事前計算が切れておらず、ユーザが見た際に「未計算」表示に当たる頻度が実質ゼロ
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる（2026-09-30時点でv1パリティ・
  HybridStorageシングルトン化は解決済みのため、実質このラインに近い状態）

## 既知の課題

（無し。2026-09-30時点でv1パリティ・HybridStorageシングルトン化ともに解決済み。`make test` 439 passed で確認）
- 下記TODOの「precomputeバッチの実行スケジュール整備」は、VPSなら既存のOS crontab方式に
  乗せればよい。GCPへ移行する場合も、Compute Engineでのリフト&シフトなら同方式を継続できるが、
  Cloud Run等のサーバーレス構成を選ぶ場合はCloud Scheduler+Cloud Run Jobsでの実装が前提になる
  （採用するGCPサービスにより対応が変わる）。詳細は
  [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [x] legacy/v1で追走難度の値が一致することを確認する自動テストを追加する（現状は手動確認のみ）
      — 2026-10-01対応: `tests/api/test_tracking_difficulty_legacy_v1_parity.py` を新規追加。
      `tracking_difficulty_service.get_or_compute` をモックし、legacy
      (`fastapi.testclient.TestClient` 経由で `/api/race/{race_id}/tracking-difficulty`) と v1
      (`src.api.flask_app.create_app().test_client()` 経由で
      `/api/v1/races/<race_id>/tracking-difficulty`) に同一ペイロードを返させて、公開フィールド
      （race_date/race_name/venue/surface/distance/track_condition/field_size/pace_prediction/
      position_flow/entries 等。entries 内の horse_number ごとの tracking_difficulty 値も含む）が
      完全一致することを検証する統合テスト。加えて `refresh=true` 時に両エンドポイントが
      `get_or_compute` へ渡す `force_refresh`/`allow_scrape`/`allow_compute_on_miss` が同一であること、
      未計算時に両方とも 404 + `status=not_precomputed` を返すことも確認。
      `python3 -m pytest tests/api/test_tracking_difficulty_legacy_v1_parity.py -v` で3件とも成功
      （`python3 -m pytest tests/api/ -q` でも既存169件+新規3件すべて成功、回帰無し）。
      なお v1 側は内部メタキー（`_compute_meta` 等 `_` prefix）を剥がさずそのまま返す実装のため
      レスポンスボディの"キー集合"は legacy と完全一致ではない（legacyは `_` prefix を除去）が、
      表示に使う公開フィールドの値は一致する。この差異自体は実害が無いため今回は対応不要と判断。
- [x] 「未計算（not_precomputed）」に当たる頻度を計測する
      — 2026-10-01対応: `src/scripts/maintenance/measure_tracking_difficulty_coverage.py` を新規追加し
      実行した。対象レースの定義は既存バッチ（`precompute_tracking_difficulty_all.py` /
      `batch_inference_all_races.collect_race_ids`）と同じ `storage.list_keys("race_shutuba")` を基点に、
      中央競馬（race_id の venue code 01-10）かつ直近365日（各レースの実際の `date` フィールドで判定）
      に絞り込んだもの。計算済みの定義は `tracking_difficulty_store.exists_local(race_id)`。
      `python3 -m src.scripts.maintenance.measure_tracking_difficulty_coverage` で実行可能
      （`--days` で期間、`--no-jra-only` で地方競馬も含める、`--no-probe-dates` で粗い年プレフィックス
      判定に切替可能）。
      **実測結果**: このエージェント実行環境（ユーザーのVPS/本番ホストではない開発用チェックアウト）には
      `data/calculated_data/tracking_difficulty/` が存在せず（0件）、`race_shutuba` のローカルデータも
      GCSアクセス権も無いため、対象レース数0件・計測不可という結果になった
      （`GCS_BUCKET` 未設定、かつ `gsutil ls` も403で確認済み）。したがって実際の本番/VPSでの
      未計算率はこの環境からは測定できなかった。スクリプト自体は実行して正常に完了すること・
      0件時に計測不可であることを明示するフォールバック表示を確認済み。
      **次のアクション（本番ホストで実行して埋めるべき実測値）**: VPS/本番ホスト上で
      `python3 -m src.scripts.maintenance.measure_tracking_difficulty_coverage` を実行し、
      対象レース数・計算済み件数・未計算率(%) をこの行に追記すること。

### VPS側（サービング）に残るTODO

- [x] 事前計算（`precompute`）バッチの実行スケジュール有無を確認し、無ければ整備する
      — 2026-10-02対応: 現状は手動トリガーのみと確認。配信=VPS、precomputeバッチの実行=GCPの
      方針のため、既存の`python -m src.scripts.maintenance.precompute_tracking_difficulty_all
      --skip-existing`をGCP側Cloud Scheduler + Cloud Run Jobsの実行コマンドとして
      [`docs/operations/gcp-cloud-run-jobs.md`](../../operations/gcp-cloud-run-jobs.md)の
      ジョブ#7に記録済み。

### GCP側（スクレイピング・ML・スケジュール実行）のTODO

- [x] 上記のバッチ実行整備は、サーバーレス移行する場合はCloud Scheduler+Cloud Run Jobsでの
      実装が前提になる。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)
      — 2026-10-01対応: `POST /api/race/{race_id}/tracking-difficulty/precompute`と同じ
      `build_tracking_difficulty_response()`/`save_cached_response()`を使う既存のバッチ CLI
      `python -m src.scripts.maintenance.precompute_tracking_difficulty_all --skip-existing`
      （新規ファイル追加は不要）を確認し、実行コマンド・想定頻度（元は固定cron無し。提案値:
      毎日08:00 JST）・リソース目安・`gcloud scheduler jobs create http`登録コマンド例を
      [`docs/operations/gcp-cloud-run-jobs.md`](../../operations/gcp-cloud-run-jobs.md)に
      まとめた。デプロイ設計図は
      [`scripts/gcp/deploy_cloud_run_jobs.sh`](../../../scripts/gcp/deploy_cloud_run_jobs.sh)。
      実デプロイ・スケジューラ登録はユーザー側作業として残る

## メモ

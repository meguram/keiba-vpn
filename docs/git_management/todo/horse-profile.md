# TODO: feature/horse-profile

**対象領域**: 馬プロフィール・馬名検索・関係者統計
**関連ドキュメント**: [../feature-horse-profile.md](../feature-horse-profile.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-10-02時点）

- `/api/horse/{horse_id}/detail`: 馬詳細情報（netkeiba + SmartRC + 統計）を集約して返す。
  2026-10-02: `smartrc_race`ロード・`_build_pedigree`・`_calc_horse_stats`を個別にtry/exceptで保護し、
  想定外の例外時も空データで継続 or `JSONResponse({"error": ...}, 500)`でJSON応答を返すよう修正
  （非JSONの素の500による呼び出し元の表示崩れを防止）
- `/api/horse/{horse_id}/recent_races`: horse_resultのrace_historyから直近N走（新しい順）
- `/api/horse/{horse_id}/race_performance_history`: 戦績のうちrace_performance生成済みのものだけを返す
- `/api/person/{ptype}/{person_id}/stats`: 騎手・調教師の成績情報
- `/api/horse-names/index-meta`: 馬名インデックスの参照用メタ（パス・頭数・生成時刻）
- `/api/horse-names/search`: 馬名検索・候補返却。NFKC正規化・ひらがな/カタカナ変換・
  （かな入力時の）pykakasi読み照合に加え、2026-10-02に小書き文字（拗音・促音）とヴ行の
  表記ゆれ畳み込み（`_fold_kana_variant`）を追加し、「ドゥラメンテ」⇔「ドウラメンテ」等の
  カタカナ表記違いも検索候補にヒットするようにした
- ページURLは無し（`/api/*`のみで構成される機能領域）

## 目標（推測）

ユーザが馬名・騎手名・調教師名を検索すれば、その馬/人物の基本情報と成績がすぐ確認できること。
他の分析系ブランチ（race-detail・bloodline-pedigree等）が参照する基盤データとしての役割も
兼ねると推測される。

## このラインまで実装できたらブランチを消してよい

- 馬名検索・騎手/調教師統計で「見つからない」「情報が古い」といった事象がユーザ体験上
  問題にならないレベルまで解消されている
- 他ブランチ（race-detail等）から参照される際にレスポンスが安定している
  （タイムアウト・欠損データによる連鎖的な表示崩れが無い）
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

（無し）
- 騎手・調教師統計（`/api/person/{ptype}/{person_id}/stats`が参照するデータ）の生成は
  `scripts/cron/update_jockey_trainer_stats.sh`によるOS crontab定期実行が前提。GCPへ移行する場合、
  Compute Engineでのリフト&シフトなら同方式を継続できるが、Cloud Run等のサーバーレス構成を
  選ぶ場合はCloud Scheduler+Cloud Run Jobsでの再設計が必要になる（採用するGCPサービスにより
  対応が変わる）。詳細は
  [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [x] 馬名検索（`/api/horse-names/search`）の表記ゆれ対応（旧馬名・カタカナ表記違い等）を検証する
      — 2026-10-02対応: 本番の `data/knowledge/horse_name_index.json`（37,393頭、読み取り専用）を
      直接ロードし、`_normalize_search_text`/`_is_kana_only_query` と同一ロジックを `/tmp` 上の
      スタンドアロンスクリプトで再現して検証（本番ファイルへの書き込みは一切なし）。
      既存実装はNFKC正規化・ひらがな/カタカナ変換・（かな入力時の）pykakasi読み照合まで対応済みと
      確認。一方で「ドゥラメンテ」⇔「ドウラメンテ」、「ヴァイオレット」⇔「バイオレット」のような
      小書き文字（拗音・促音）・ヴ行の表記ゆれは未吸収でヒットしないことを確認したため、
      `src/api/app.py` の `_normalize_search_text` に小書き文字→大書き・ヴ→ブの畳み込み変換
      （`_fold_kana_variant` / `_VU_DIGRAPH_FOLD` / `_SMALL_KANA_FOLD_TABLE`）を追加し、
      検索候補に重畳させる形で対応。新規依存ライブラリは追加していない。
      なお「ディープインパクト」「キタサンブラック」等の引退済み大物馬がインデックスに
      含まれていない事象を確認したが、これは表記ゆれではなく、インデックスが
      `horse_result` キャッシュ（走査対象: `data/cache/horse_result` 等）から構築される
      データ網羅性の制約（既知の限界）であり、本TODOの対応範囲外と判断した。
      旧馬名（改名履歴）はそもそもインデックスのスキーマに保持されておらず、同様に
      データ構造上の制約として別途の設計検討が必要（本TODOでは対応せず）。
      既存テスト（`tests/api/test_endpoints.py` のhorse_names系、`tests/utils/test_horse_name_index*.py`）は
      全て通過を確認。
- [x] 他ブランチ（race-detail等）からの参照時に、本APIのタイムアウト・エラーが連鎖的に
      表示崩れを起こしていないか確認する
      — 2026-10-02対応: `feature/race-detail` ブランチの `templates/race/race_detail.html` を
      `git show` で確認した結果、`/api/horse-names/search` や `/api/horse/{horse_id}/detail` を
      直接fetchしておらず、`/api/race/{race_id}` レスポンスに同梱された `horses` から
      馬詳細カードを構築していることを確認（連鎖呼び出しは発生しない）。
      一方 `templates/analysis/growth_curve.html`・`templates/analysis/tracking_difficulty.html`・
      `templates/admin/data_viewer.html` は両APIをfetchしているが、いずれも `resp.ok` を
      チェックしてから `try/catch` で個別のモーダル・パネル内にエラー表示する実装済みで、
      ページ全体への連鎖崩れは起きない構造であることをコードから確認した。
      サーバ側では `HybridStorage.load` がGCSタイムアウト・リトライを内包し失敗時は
      `None`/stale ローカルキャッシュへフォールバックする設計のため、通常は例外を上げない。
      ただし `/api/horse/{horse_id}/detail` のみ、集計処理（`_build_pedigree`/`_calc_horse_stats`）や
      `smartrc_race` ロードが未保護で、万一の例外発生時にJSON化できない500（素のテキスト応答）に
      なり得る実装だったため、`/api/race/{race_id}` 等の他エンドポイントと同様の防御的
      try/except（各ステップを個別に保護し、失敗時は空データで継続。最終的に例外が残っても
      `JSONResponse({"error": ...}, status_code=500)` を返す）を追加した。
      既存テスト（`tests/api/test_endpoints.py` のhorse_detail系）は全て通過を確認。

### VPS側（サービング）に残るTODO

- [x] 騎手・調教師統計（`/api/person/{ptype}/{person_id}/stats`）の集計対象期間・更新頻度を明記する
      — 2026-10-02対応: 現状は`scripts/cron/update_jockey_trainer_stats.sh`経由のOS crontabで
      `python -m src.pipeline.build_jockey_trainer_stats`を定期実行（更新頻度は同スクリプトの
      cron設定に準拠）。配信=VPS、統計生成の実行=GCPの方針のため、同コマンドをGCP側Cloud
      Scheduler + Cloud Run Jobsの実行コマンドとして
      [`docs/operations/gcp-cloud-run-jobs.md`](../../operations/gcp-cloud-run-jobs.md)の
      ジョブ#5に記録済み（既存コマンドのまま、新規実装は不要だった）。

### GCP側（スクレイピング・ML・スケジュール実行）のTODO

- [x] 上記の統計更新頻度は、サーバーレス移行する場合はCloud Scheduler+Cloud Run Jobsでの
      実行頻度に置き換わる。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)
      — 2026-10-01対応: `scripts/cron/update_jockey_trainer_stats.sh`が呼んでいる既存CLI
      `python -m src.pipeline.build_jockey_trainer_stats`（新規ファイル追加は不要。AGENTS.md
      記載の正式エントリ）を確認し、実行コマンド・想定頻度（現行OS crontab`30 20 * * *`UTC=
      毎日05:30 JSTと同一）・リソース目安・`gcloud scheduler jobs create http`登録コマンド例を
      [`docs/operations/gcp-cloud-run-jobs.md`](../../operations/gcp-cloud-run-jobs.md)に
      まとめた。デプロイ設計図は
      [`scripts/gcp/deploy_cloud_run_jobs.sh`](../../../scripts/gcp/deploy_cloud_run_jobs.sh)。
      実デプロイ・スケジューラ登録はユーザー側作業として残る

## メモ

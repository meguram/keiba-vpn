# TODO: feature/monitor-quality

**対象領域**: 監視・データ品質チェック
**関連ドキュメント**: [../feature-monitor-quality.md](../feature-monitor-quality.md)
**最終更新**: 2026-10-01

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-10-01時点）

- `/monitor`: スクレイピング状態のリアルタイムモニタリングボード（レガシー版。開発者専用の独立監視ポータル :9090 とは別物）
- `/data-viewer`: 生JSONデータビューア
- カバレッジマトリクス: `/api/date-race-matrix`（レース×カテゴリ）・`/api/date-raw-matrix`（Raw: GCS+PG/要件）・`/api/date-calculated-matrix`（Calculated: megu_index/flat parquet）・`/api/coverage-calendar`（年別カテゴリカレンダー）
- データ品質チェック: `/api/quality-check/enqueue`（投入）・`jobs`（一覧）・`health`（健全性）・`remediate`（修復）・`calendar`（カレンダー表示）
  - `enqueue` で3種チェック（presence/raw_content/calculated）が全て揃うと、`src/api/quality_health.run_check()`
    が内部で `src/api/quality_auto_remediation.maybe_remediate_after_checks()` を呼び、warn/fail なら
    `remediate_date()` を自動実行する（`KEIBA_QUALITY_AUTO_REMEDIATE` 未設定時は `KEIBA_ENV=stg` でのみ有効）。
    ただし `enqueue` 自体を定期実行する cron は無く、起点は `/monitor` からの手動投入、または手動バッチ
    （`src/scripts/data/run_quality_health_batch.py` / `run_quality_auto_remediation.py`）のみ。
- `/api/row-data-coverage`: 行固有派生カテゴリ（race_shutuba_meta等）のGCSカバレッジ
- `/api/monitor/missing-dates-summary`: 開催日別のJRAレース未取得件数集計
- `/api/monitor/opening-date-info`: 開催日種別（非開催ラベル用）
- `/api/monitor/context`: UI向け環境・GCS/DB接続・集計モード情報
- ユーザ向け `/race/{race_id}`（`/api/race/{race_id}`）: 開催日の品質ヘルス（`get_health_view()`）が
  warn/fail の場合に `quality_health_warning` をレスポンスに付与し、レースヘッダーに警告バッジを表示
  （ブロックはしない。チェック未実行/ok/na の日は何も表示しない）

## 目標（推測）

ユーザに表示するデータの品質（欠損・不整合）を、ユーザが気づく前に運用者が検知・修復できる
状態にすること。ユーザ視点では「表示されているレース情報・指数が信頼できる」ことが最終目的で、
本ブランチ自体はそれを裏で保証する運用者向け機能と推測される。

## このラインまで実装できたらブランチを消してよい

- スキーマ検証・カバレッジチェックが自動修復トリガーと連動し、運用者が毎日手動確認しなくても
  品質劣化に気づける状態になっている
- ユーザに表示される情報が品質チェック未通過のまま出ることが無いと確認できている
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

- `/monitor`・開発者専用監視ポータル(:9090、`src/monitor/app.py`)はいずれもVPS上の
  常駐プロセス前提（`start_monitor.sh`でnohupデーモン化、`.monitor.pid`管理）。GCPへ移行する
  場合、Compute Engineでのリフト&シフトならそのまま常駐させられるが、Cloud Run等の
  スケールtoゼロ環境を選ぶ場合は常時起動が必要なポータルの維持方法を別途検討する必要がある
  （採用するGCPサービスにより対応が変わる）。
  詳細は [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [x] `/api/quality-check/remediate`（修復）が自動トリガーと連動しているか、現状手動実行のみかを確認する
      — 2026-10-01対応: コードを確認した結果、**部分的に自動化済み、ただし全体を起動する入口は手動のみ**。
      `src/api/quality_health.py` の `run_check()` は、3チェック種別（presence/raw_content/calculated）が
      全て揃った状態で呼ばれると、内部で `quality_auto_remediation.maybe_remediate_after_checks()` を
      呼び、`overall_status` が warn/fail なら `remediate_date()` を自動実行する（`KEIBA_QUALITY_AUTO_REMEDIATE`
      で有効/無効を切替可能。未設定時は `KEIBA_ENV=stg` の場合のみ有効）。つまり「チェック実行 → 修復」の
      連動自体は実装済みで、`/api/quality-check/remediate` を人間が手動で叩く必要は無い。
      一方で、その起点となる `run_check()`／`quality_check_queue.enqueue()` を自動的に呼ぶ cron やバッチは
      存在しない（`scripts/cron/setup_all_cron.sh` を確認したが quality-check 関連のエントリは
      `page_quality_check`（毎日23時、代表サンプルページのスキーマ照合のみで remediate 連動なし）だけ）。
      バッチスクリプト `src/scripts/data/run_quality_health_batch.py`（チェック実行）・
      `src/scripts/data/run_quality_auto_remediation.py`（修復のみ再実行）は存在するが、
      いずれも cron 未登録で現状は手動実行専用。
      **結論**: 修復トリガー自体はチェック結果に連動して自動発火するが、チェックを定期的に走らせる
      仕組みが無いため、実運用では「運用者が `/monitor` からチェックを手動投入する」まではまだ人手が必要。
      日次チェックを自動化したい場合は `run_quality_health_batch.py` を cron に追加する対応が今後の候補
      （本TODOの範囲外のため実装はしていない）。
- [x] 品質チェック未通過のデータがユーザ表示側で実際にブロック・警告されているか確認する
      （チェック機能はあるが表示側との連動が無ければ意味が薄い）
      — 2026-10-01対応: 調査の結果、**連動していなかった**。`/api/quality-check/*` を参照しているのは
      `templates/admin/monitor.html`（運用者向け `/monitor` ダッシュボード）のみで、ユーザ向けの
      `/race/{race_id}`（`templates/race/race_detail.html`）およびそのデータ取得元 `/api/race/{race_id}`
      （`src/api/app.py` の `get_race_detail`）は品質チェック結果を一切参照していなかった
      （`race_quality.html` は名前が似ているが「レース質分析」という別機能で、データ品質とは無関係）。
      ブロックは行われていなかったため、未通過データでもそのままユーザに表示される状態だった。
      ユーザ体験上の問題と判断し、**小規模な改善を実装した**（大規模な設計変更ではなく警告表示のみ）:
      - `src/api/app.py` の `get_race_detail()` に、レース結果の開催日について
        `src.api.quality_health.get_health_view()` を参照し、`overall_display_status` が
        `warn`/`fail` の場合のみレスポンスに `quality_health_warning: {status, checked_at}` を
        追加するよう変更（未チェック/`ok`/`na` の場合は何も追加せず、過剰な警告を避ける）。
      - `templates/race/race_detail.html` に `#qualityWarnBadge` を追加し、`quality_health_warning` が
        warn/fail のときのみレースヘッダーに警告バッジ（例: 「⚠ データ品質: 要確認」）を表示するよう
        JS を追加。表示をブロックすることはせず、警告バッジのみ（既存方針「ブロックではなく警告」に合わせた）。
      - `tests/api/test_endpoints.py` / `tests/api/test_quality_health.py`（計130件）で動作確認済み（全件 pass）。

### 常時稼働ホスト（VPS / GCP Compute Engine）の場合のTODO

- [ ] `/monitor`（レガシー版）と開発者専用監視ポータル(:9090)の役割重複を整理し、
      どちらかに統合できないか検討する（:9090ポータルは常駐プロセス前提のため、
      常時稼働ホスト方式でのみ現行の形のまま統合検討ができる）

### GCPサーバーレス（Cloud Run等）移行時のTODO

- [ ] 上記の統合検討は、サーバーレス移行する場合は先に「常時起動が必要なポータルの維持方法」
      （[`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)参照）
      を決めてからでないと着手できない

## メモ

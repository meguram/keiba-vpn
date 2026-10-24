# TODO: feature/cushion

**対象領域**: クッション値（トラックコンディション）
**関連ドキュメント**: [../feature-cushion.md](../feature-cushion.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-10-01時点）

- `/api/cushion/data`・`/api/cushion/stats`: クッション値・含水率データの取得・統計
- `POST /api/cushion/scrape` + `/scrape/status`: スクレイピング実行・状態確認
- `POST /api/cushion/live` + `/live/status` + `/live/check`（軽量更新チェック）: JRA公式馬場情報ページからのライブ取得
- `/api/cushion/schedule`: 直近のポーリングスケジュール
- `POST /api/cushion/admin/sync-gcs`: ローカルcushion_values.jsonを年別にGCSへ同期（開発者ログイン必須）
- `POST /api/cushion/admin/sync-preprocessed`: GCS preprocessed/cushion_dataの日次JSONをjra_cushion年別JSONにマージ（開発者のみ）
- ページURLは無し（`/api/*`のみで構成される機能領域。2026-09-30にユーザ判断で本ブランチは保持継続）
- **構造変更検知・Slack通知（2026-10-01追加）**: `src/scraper/jra_baba_live.py`の
  `JRABabaLiveScraper.scrape()`が、`.unit`要素はあるのに`.time`/`.cushion`が
  1件も取れない、または`.contents_header h2`はあるのに開催ヘッダーの想定
  フォーマットに1件も一致しない、といったJRA公式ページのHTML構造変更の疑いを
  検知すると`src/utils/notify.py`の`notify_slack()`でまとめて1通通知する
  （`SLACK_WEBHOOK_URL`未設定時は無害にスキップ）。本番cronの実エントリ
  `run_cron_job()`では、更新ハッシュ変化後のフルスクレイプが0件だった場合にも
  別途通知する。`/api/cushion/live`の例外発生時も通知する。

## 目標（推測）

ユーザ（あるいはそのデータを使う予測モデル）が、当日の馬場状態（クッション値・含水率）を
リアルタイムに近い形で把握できること。track-speed・race-quality等の予測精度向上を裏で支える
データ基盤としての位置づけと推測される。

## このラインまで実装できたらブランチを消してよい

- レース当日、JRA公式発表からライブ取得までの遅延がユーザ／モデル利用に支障が無いレベルに
  抑えられている
- GCS同期・過去データマージが定期的に行われ、履歴データに欠損が無い
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

（無し。以前 `/api/cushion/scrape`(+status) が `scraping-queue` に誤分類されていたが移動済み。
v1側との重複実装や`HybridStorage()`直接生成も無いことを確認済み）
- 下記TODOの「ポーリングスケジュールの稼働確認」「sync-gcs等の実行頻度確認」は、VPSなら
  既存のOS crontab/daemon thread方式を前提に調査すればよい。GCPへ移行する場合も、
  Compute Engineでのリフト&シフトなら同方式を前提に調査できるが、Cloud Run等の
  サーバーレス構成を選ぶ場合はスケジュール自体がCloud Scheduler+Cloud Run Jobsに
  置き換わる前提になる（採用するGCPサービスにより対応が変わる）。詳細は
  [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [x] JRA公式ページの構造変更でライブ取得が失敗した場合の検知・アラートを追加する
      （検知ロジック自体はホスト方式に依存しない）
      — 2026-10-01対応: `src/scraper/jra_baba_live.py`に構造変更検知を追加。
      (1) `_fetch_cushion_data`: `.unit`は存在するが`.time`/`.cushion`が1件も
      取得できない場合、(2) `_fetch_venue_info`: `.contents_header h2`は取得
      できるが想定フォーマット（第N回○○競馬第M日...）に1件も一致しない場合、
      をそれぞれ「構造変更の疑い」として記録し、`scrape()`完了時に
      `src/utils/notify.py`の`notify_slack()`で1通にまとめて通知（開催なし週
      との誤検知を避けるため、該当セレクタの外枠自体が存在する場合のみ発火）。
      さらに本番cronの実エントリである`run_cron_job()`（`has_new_data()`で
      ハッシュ変化を検知した後のフルスクレイプが0件だった場合）にも通知を追加。
      `/api/cushion/live`（`src/api/app.py`の`_run_baba_live_scrape`）は
      `scrape()`経由で自動的に同じ検知を継承するほか、例外発生時のSlack通知も
      追加（既存の`cron失敗`系通知パターンに合わせた）。新規テスト
      `tests/scraper/test_jra_baba_live_structure_alert.py`（10件）で検知の
      発火/非発火を確認。`python3 -m pytest tests/ --ignore=tests/scraper/manual
      --ignore=tests/research/manual` で460 passed / 0 failed。JRA公式サイトへの
      実アクセスは行わず、モックHTMLのみで検証。

### VPS側（サービング）に残るTODO

（本ファイルでは該当なし。スクレイピング・ライブ取得・GCS同期はいずれもGCP側へ移動。
VPS側は`/api/cushion/data`・`/api/cushion/stats`の配信のみ）

### GCP側（スクレイピング・ML・スケジュール実行）のTODO

- [x] `POST /api/cushion/live` のポーリングスケジュールの実行コマンドを整理する
      — 2026-10-02対応: `src/scraper/jra_baba_live.py`（`JRABabaLiveScraper.scrape()`、構造変更
      検知・Slack通知込み、2026-10-01実装）が`python -m src.scraper.jra_baba_live`で直接実行可能。
      Cloud Scheduler + Cloud Run Jobsの実行コマンドとして
      [`docs/operations/gcp-cloud-run-jobs.md`](../../operations/gcp-cloud-run-jobs.md)の
      ジョブ#14に追記した。
- [ ] `admin/sync-gcs`・`admin/sync-preprocessed`（`POST /api/cushion/admin/*`）はGCP側Cloud Run Jobs
      化の実行コマンドをまだ整理していない。実処理関数の特定・CLIラッパー追加が必要（未対応）。

## メモ

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

## 現状の実装（2026-09-30時点）

- `/api/cushion/data`・`/api/cushion/stats`: クッション値・含水率データの取得・統計
- `POST /api/cushion/scrape` + `/scrape/status`: スクレイピング実行・状態確認
- `POST /api/cushion/live` + `/live/status` + `/live/check`（軽量更新チェック）: JRA公式馬場情報ページからのライブ取得
- `/api/cushion/schedule`: 直近のポーリングスケジュール
- `POST /api/cushion/admin/sync-gcs`: ローカルcushion_values.jsonを年別にGCSへ同期（開発者ログイン必須）
- `POST /api/cushion/admin/sync-preprocessed`: GCS preprocessed/cushion_dataの日次JSONをjra_cushion年別JSONにマージ（開発者のみ）
- ページURLは無し（`/api/*`のみで構成される機能領域。2026-09-30にユーザ判断で本ブランチは保持継続）

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

- [ ] JRA公式ページの構造変更でライブ取得が失敗した場合の検知・アラートを追加する
      （検知ロジック自体はホスト方式に依存しない）

### 常時稼働ホスト（VPS / GCP Compute Engine）の場合のTODO

- [ ] `POST /api/cushion/live` のポーリングスケジュール（`/schedule`で確認できる想定）が
      実際に稼働しているか、現行のOS crontab/daemon thread方式を前提に確認する
- [ ] `admin/sync-gcs`・`admin/sync-preprocessed` の実行頻度を確認し、履歴データの欠損期間が
      無いか点検する（現行の定期実行方式を前提にした点検）

### GCPサーバーレス（Cloud Run等）移行時のTODO

- [ ] 上記2件のスケジュール確認は、サーバーレス移行する場合はCloud Scheduler+Cloud Run Jobsの
      実行ログを前提にした確認に置き換わる。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)

## メモ

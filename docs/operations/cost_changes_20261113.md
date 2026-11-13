# 費用のための GCP の設定変更（2026-11-13）: GCS の監査ログ（24 時間だけ）と Artifact Registry の保持

ユーザーの承認: 2026-11-13 14:1x JST（統括役のセッション経由）。実施は費用の監査役。背景は費用の台帳 `docs/operations/cost_ledger.md` の #1〜#3。
Scheduler の削除は別の記録 `docs/operations/deleted_scheduler_keiba_g_20261113.md` に残す。

## 1. GCS のデータアクセスの監査ログ（DATA_READ）を 24 時間だけ有効にする

### 目的
GCS の外への転送（課金の `Download APAC`、1 日 4〜10GiB）と一覧（ListObjects、1 日 約 1.2 万回）のうち、呼び手が分かっていない分を、
監査ログの呼び手（principal・呼び元 IP・メソッド）で数える。

### 変える前の設定（2026-11-13 14:2x JST に控えた）
- プロジェクト horse-racing-466415 の IAM ポリシーに `auditConfigs` は**無い**（どのサービスもデータアクセスの監査ログは無効）。`version` 1、`etag` `BwZemVZZEkQ=`、バインディング 23 件。
- 控えの取り方: `gcloud projects get-iam-policy horse-racing-466415 --format=json`。

### 入れる設定
```
auditConfigs:
- service: storage.googleapis.com
  auditLogConfigs:
  - logType: DATA_READ
```
DATA_WRITE・ADMIN_READ は入れない（ADMIN の書き込みは元から常に記録されている）。

### ログの量の見積もり（Cloud Logging の無料分 月 50GiB に収まるか）
- GCS の読み出し系の要求（Cloud Monitoring の `storage.googleapis.com/api/request_count`、直近 24 時間）: ReadObject 約 20 万・GetObjectMetadata 約 1.2 万・ListObjects 約 1.2 万・GetBucketMetadata 約 2,000 ＝ 約 23 万件/日。
  最悪は 11/07（Jobs の並走がフル）の 約 116 万件/日。
- 監査ログ 1 件 約 1.5〜2KB として、24 時間で **約 0.35〜0.45GiB**（最悪 約 2.3GiB）。
- 今の取り込みは 1 日 約 0.01GiB（`logging.googleapis.com/billing/bytes_ingested`）。合わせても月 50GiB の無料分に十分収まる。保存は既定の 30 日（無料）。

### 戻すとき（24 時間後）
`auditConfigs` を消した IAM ポリシーに戻す（変える前の状態）。
```
gcloud projects get-iam-policy horse-racing-466415 --format=json > /tmp/p.json
# /tmp/p.json から "auditConfigs" を消す
gcloud projects set-iam-policy horse-racing-466415 /tmp/p.json
```

### 実施の記録
（下に追記する: 有効にした時刻・戻した時刻）

## 2. Artifact Registry の自動掃除の保持を 2 日 → 1 日にする

### 変える前の設定（リポジトリ asia-northeast1 `keiba-vpn`、2026-11-13 に控えた）
```
[{"name":"keep-live","action":{"type":"Keep"},"condition":{"tagState":"TAGGED","tagPrefixes":["live-"]}},
 {"name":"keep-recent","action":{"type":"Keep"},"mostRecentVersions":{"keepCount":3}},
 {"name":"delete-old","action":{"type":"Delete"},"condition":{"olderThan":"172800s"}}]
```
容量は 16GiB（API の版 13・Jobs の版 6。11/11〜12 の配備が 2 日の保持で残っていた）。費用は月 約 240 円。
（もう 1 つのリポジトリ `keiba` は 0.04GiB で、掃除の設定は無い。変えない。）

### 変えた後
`delete-old` の `olderThan` だけを `86400s`（1 日）にする。`keep-live`（`live-` タグ付き）と `keep-recent`（パッケージごとに新しい 3 版）は変えない（CLAUDE.md の再発防止の決まり 4）。
同じ設定を書いているスクリプトもそろえる（そろえないと、次の API の配備で 2 日に戻る）:
`scripts/gcp/deploy_cloud_run_api.sh`、`scripts/gcp/setup_cloud_run_api.sh`、`scripts/gcp/setup_project.sh`。CLAUDE.md の記述も 1 日に直す。

見込み: 1 日の配備回数 × 1 日分 ＋ 新しい 3 版 ＋ live- 付きが残る。月 約 −80 円（台帳の #3）。

### 戻すとき
`olderThan` を `172800s` に戻して、上の 3 つのスクリプトと同じコマンドを流す。

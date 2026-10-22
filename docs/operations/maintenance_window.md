# 毎日のメンテナンス時間帯（JST 03:00–04:00）

毎日 **日本時間 03:00–04:00** をメンテナンス時間帯とし、この間はユーザーに『メンテナンス中』の表示を出す。
サーバの再起動やフロントの入れ替えは、この窓の中で行う（ステージした変更の自動反映: [maintenance_deploy.md](maintenance_deploy.md)）。

## 利用者から見える動き

| 時刻（JST） | 表示 |
|---|---|
| 02:30〜03:00 | 画面の上端に黄色い帯『まもなくメンテナンスを行います（03:00〜04:00 日本時間）。あと約N分で始まります』（操作は続けられる） |
| 03:00〜04:00 | 全画面の『メンテナンス中です』。終了予定時刻・残り時間のカウントダウン・お詫びを表示。**URL は変えない**（ブックマークを壊さない）。HTTP は 503 + `Retry-After` + `noindex` |
| 04:00 | 自動で『終了しました』→ 元のページに戻る（開いたままのタブも同様） |

- ブラウザの時計は使わない。API（`/api/maintenance/status`）が返す『残り秒』を受信時刻から数える（端末の時計がずれていても正しい）。
- メンテナンス中にどの画面の API 呼び出しが 503 を受けても、各画面が個別にエラーを出さず、共通のメンテナンス画面に切り替わる（`X-Keiba-Maintenance: 1` を検知）。API の 502/504（再起動中）は、状態を取り直して判断する。

## 仕組み（3 層）

1. **Next.js の `middleware.ts`**（ページ）: メンテナンス中は全ページを `/maintenance` に**書き換え**（リダイレクトではない）、503 で返す。判定は API の状態を 15 秒だけメモリに持つ。管理者の Cookie（`keiba_dev_session`）があれば通す（本当の検証は API 側）。`/api/` `/_next/` `/data/image/` `/login` は素通り。
2. **FastAPI のミドルウェア / Flask の `before_request`**（API）: メンテナンス中は一般ユーザー向けの経路を `503` + `Retry-After`（終了までの秒）+ `{"maintenance": true, "message", "ends_at", ...}` + `Cache-Control: no-store` で返す。実装は `src/api/maintenance_mw.py`、判定は `src/utils/maintenance.py`（純粋関数・時計を注入できる・JST を zoneinfo で明示するのでホストの UTC に依存しない）。
3. **画面（`MaintenanceShell`）**: 事前告知バナー、全画面のメンテナンス表示、フェッチの共通ハンドラ、自動復帰。

### 止めないもの（除外）

`config/maintenance.json` の `exempt_exact` / `exempt_prefixes`:

- ヘルスチェック: `/api/health`, `/api/v1/health`、状態: `/api/maintenance/status`, `/api/v1/maintenance/status`
- ログイン: `/login`, `/logout`, `/api/auth/`（管理者が窓の中でもログインして確認できる）
- 運用 API: `/api/scrape-jobs`, `/api/scrape-queue`, `/api/auto-scrape`, `/api/admin`, `/api/monitor`, `/api/quality-check`, `/api/backfill`, `/api/race-lists-backfill`, `/api/train`
- 静的資産: `/static/`, `/data/image/`, `/_next/`

**止めるのはユーザー向けの表示だけ。** 取得・予測生成・cron・ランナーは止めない。監視ポータル（別の Flask）は対象外。

### バイパス（窓の中でも通す）

1. 管理者としてログイン済み（Cookie の署名を API が検証）。dev 環境の自動ログインも同様（`KEIBA_MAINTENANCE_DEV_BYPASS=0` で無効にして確認できる）。
2. ヘッダー `X-Keiba-Maintenance-Bypass: <トークン>`（環境変数 `KEIBA_MAINTENANCE_BYPASS_TOKEN` と一致したとき。未設定なら使えない）。
3. 同一ホストの内部呼び出し（ループバック、かつ `X-Forwarded-For` 等のプロキシ転送ヘッダーが無い。cron・ランナー・監視が API を直接叩く場合）。ブラウザは Next.js のプロキシ経由で転送ヘッダーが付くため対象外。

バイパスの利用は 1 分に 1 回だけ `maintenance bypass: reason=... path=...` として監査ログに残る。開始・終了・事前告知の遷移も `maintenance start|end|notice` で 1 行ずつ残る。

## 有効化・設定

- **有効化**: `.env` に `KEIBA_MAINTENANCE_WINDOW=1`（**既定はオフ**）。dev 環境は更に `KEIBA_MAINTENANCE_DEV=1` が要る（dev は既定で無効）。環境変数の追加後は API の再起動が要る（Flask も同様）。
- **設定の正**: `config/maintenance.json`（時間帯 `daily.start/end`、`notice_minutes`、文言、除外、バイパス、`notify.slack_start_end`）。**30 秒キャッシュで読み直す**ので、書き換えるだけで再起動なしに反映される。壊れていれば既定値で動く（メンテナンスを誤って有効にしない側）。
- **止めたいとき（再起動なし）**: `config/maintenance.json` の `"enabled": false`（約 30 秒で反映）。完全に外すなら `.env` から `KEIBA_MAINTENANCE_WINDOW` を消して再起動。

## 臨時メンテナンス

`config/maintenance.json` の `override` を書き換える（再起動不要、30 秒で反映）:

```json
"override": {"start": "2026-10-25 13:00", "end": "2026-10-25 14:30", "message": "システム更新のため臨時メンテナンスを行います。"}
```

時刻は JST（`YYYY-MM-DD HH:MM` か、オフセット付きの ISO 8601）。開始の 30 分前から事前告知バナーが出る。終えたら `override` を `null` に戻す。毎日の窓と重なるときは**終了が遅い方**が使われる。
自動反映ランナーが窓を延長するときは、git 管理の設定を書き換えず `data/local/meta/maintenance_override.json` に書く（同じ形式の `{"override": {...}}`。環境変数 `KEIBA_MAINTENANCE_OVERRIDE` で場所を変更可）。不要になったら削除する。

## 開発（dev）での見た目の確認

- `?maintenance=notice&remaining=900`（事前告知バナー）、`?maintenance=active&remaining=1500`（全画面）、`?maintenance=ended`（終了）を任意のページに付ける（`next dev` のみ。`next start` の本番ビルドでは無効）。
- API 側を試す: `KEIBA_ENV=dev KEIBA_MAINTENANCE_WINDOW=1 KEIBA_MAINTENANCE_DEV=1 KEIBA_MAINTENANCE_DEV_BYPASS=0` に、時計の上書き `KEIBA_MAINTENANCE_NOW="2026-10-22 03:10"`（dev のときだけ有効。`/api/maintenance/status?at=...` も同様）か、`KEIBA_MAINTENANCE_CONFIG` で別の設定ファイル（`override` に今の時刻の前後を入れる）を指す。`KEIBA_MAINTENANCE_CACHE_TTL=0` で設定の再読み込みを即時にできる。

## フロント自体が止まっている間

Next.js を入れ替える数秒間は接続できない。窓の中なら、開いているタブは直前の状態（メンテナンス画面）のまま待つ（状態の取り直しに失敗しても表示は保つ）。新規の接続だけが一時的に失敗する。長く止める場合は次を使う（手動）:

- `static/maintenance.html`: 外部の読み込みが無い単体の静的ページ。リバースプロキシの `error_page 502 503 504` に指定できる（`/static/maintenance.html` として API 経由でも取れる）。
- `scripts/ops/maintenance_holding_server.py --port 23200 --seconds 600`: 同じポートで同じ静的ページを 503 で返す小さなサーバ（標準ライブラリのみ）。**自動反映ランナーは使わない**（切替スクリプトと同じポートを取り合うため）。

現在の公開経路は tcpexposer のトンネルが Next.js（stg は :23200）へ直結しており、リバースプロキシは無い。入れる場合の例（nginx）:

```nginx
error_page 502 503 504 /maintenance_static;
location = /maintenance_static { internal; root <HOST_REPO>/static; rewrite ^ /maintenance.html break; add_header Retry-After 60 always; add_header Cache-Control "no-store" always; }
```

## 通知（Slack）

- **誤警報の抑制**: `scrape_status_notify`（15 分おきの異常判定・日次サマリー・週次カバレッジ）は、メンテナンス時間帯と**終了後 10 分**を『予定された停止』として扱い、何も送らず状態も更新しない（API の 503・再起動・ジョブの停止を異常にしない）。機能が無効（`KEIBA_MAINTENANCE_WINDOW` 未設定）のときは抑制しない。
- **開始/終了の通知（任意・既定オフ）**: `config/maintenance.json` の `notify.slack_start_end` を `true` にし、`install_maintenance_deploy_cron.sh --with-notify` で cron を入れる。各イベント 1 日 1 回。

## 3:00–4:00 JST（= 18:00–19:00 UTC）に走るジョブとの関係（現在の crontab、提案のみ・変更なし）

ホストの cron は UTC。窓は 18:00–19:00 UTC。

| 時刻（UTC / JST） | ジョブ | 窓との関係・提案 |
|---|---|---|
| 17:00 / 02:00 | backfill-2024-fast（毎日）、backfill-2021-fast（月・木） | 窓の直前に始まり、窓にかかることがある。ユーザー向けではない取得。API の再起動の影響を受けない（別プロセス）。**そのままでよい** |
| 18:00 / 03:00 | backfill-2023-fast（毎日）、horse-remarks-daily（毎日）、backfill-2020-fast（火・金） | 窓の開始と同時。取得系で API を止めない。**そのままでよい**。ただし 2〜3 本が同時に始まる（DEC-023 の待ちは `yield_to_rescrape` が処理） |
| 18:05 / 03:05 | **maintenance-deploy**（新規。ステージがあるときだけ動く） | 窓の中で前提条件（キュー空・運用ジョブなし・load<14・ディスク）を待ってから適用 |
| 19:00 / 04:00 | backfill-2022-fast（毎日） | 窓の終了と同時。そのままでよい |
| 19:30 / 04:30 | rotate_logs | 窓の後。そのままでよい |
| 毎 2/3/5 分 | start_remote_access、server_watchdog、ensure_monitor、ensure_postgres | 止めない。watchdog は切替時に `.frontend-build-*.lock` を共有するため、再ビルド中に二重起動しない。API の再起動の数秒の間に watchdog が重ねて起動しないかは、運用実績（手動の `restart_server.sh` と同じ）で問題なし |
| 15:00 / 00:00、22:00 / 07:00 ほか | sync_pg_from_gcs、export_tables_daily、raceday-* | 窓の外 |

- **窓の中に重い計算（再学習・全期間の評価）を入れない**。load が上がると反映ランナーが待つ（`max_load` 既定 14）。heavy_run の長い評価は窓の外に。
- **計画的な再起動は、窓の中だけ**にする（ステージ → ランナー）。それ以外の時間帯の手動再起動は、スクレイピングキューが空であることを確認する従来の運用のまま。
- 窓の直前（02:00〜03:00 JST）に始まる長い取得は、窓の中で反映ランナーが待つ前提条件（`BLOCK_PATTERNS`: raceday-*、weight_triggered_predict、serving-state-precompute、lap_forecast_daily、tracking_v2_precompute、調教の取り直し）に当たらない限り、反映の邪魔をしない。

## 検証

`tests/utils/test_maintenance.py`（時刻判定の境界 02:59:59 / 03:00:00 / 03:59:59 / 04:00:00、UTC 跨ぎ、日跨ぎ設定、override、バイパス、通知抑制）、`tests/api/test_maintenance_mw.py`（FastAPI の 503・Retry-After・除外・バイパス・無効時の素通り・Flask）、`frontend/lib/maintenance.test.mjs`（カウントダウン、境界の先取り、プレビュー、オープンリダイレクト防止）。dev ではヘッドレスブラウザで『通常 → 事前告知 → 開始（ページを開いたまま自動で全画面へ）→ 終了（自動復帰）』と、API の 503 を受けたときの共通切替を確認した。

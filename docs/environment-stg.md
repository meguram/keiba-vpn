# ステージング（stg）環境での動作確認

このリポジトリでは **`KEIBA_ENV=stg`** を設定すると、本番と同じ GCS バケット・prefix を参照したまま、**画面上で stg であることが分かる**構成にします（データ分離用の別バケットは必須ではありません）。

## 設定

`.env` に次を追加（または `.env.example` 参照）:

```env
KEIBA_ENV=stg
KEIBA_DEPLOYMENT_LABEL=stg
# ログと libc のローカル時刻を JST に（未設定でホストが UTC のときのずれ防止）
TZ=Asia/Tokyo
```

- `TZ` … オプションだが **推奨**。アプリ側はログを JST 固定で出すが、他プロセスや `date` と揃えると運用しやすいです。
- `KEIBA_DEPLOYMENT_LABEL` … ナビ左のバッジ文言（省略時は `STG`）。
- `APP_ENV=stg` でも同義（`KEIBA_ENV` が優先）。

## 一括起動（stg）

本番相当のスタックを起動し、`meguai-stg.tcpexposer.com` へトンネルします。

```bash
cd keiba-vpn
./service_start --env stg
# またはサービスのみ（トンネル手動）:
./scripts/server/tunnel_tcpexposer.sh stg background
./scripts/server/tunnel_tcpexposer.sh stg check
```

| 項目 | stg（本 PC） |
|------|----------------|
| Next.js | `:3001`（`npm run dev`、実 API） |
| Flask | `:5000` |
| FastAPI | `:8000`（`--prod`） |
| 公開 URL | `https://meguai-stg.tcpexposer.com/` |
| 環境 | `KEIBA_ENV=stg`（STG バッジ・`X-Keiba-Env: stg`） |

**dev** との違い: dev は `./service_start`（モック UI `:3000` のみ、`meguai-dev`）。stg は GCS・PostgreSQL・Redis を使う本番相当構成です。

## 一括検証（推奨）

API が起動済み（例: `http://127.0.0.1:8000`）であることを確認してから（全サービスの URL・ポートは [operations/service-endpoints.md](operations/service-endpoints.md)）:

```bash
cd keiba-vpn
python3 -m src.scripts.maintenance.verify_endpoint_data
python3 -m src.scripts.maintenance.verify_stg_smoke
# 別ポート: KEIBA_STG_BASE=http://127.0.0.1:9000 python3 -m src.scripts.maintenance.verify_stg_smoke
```

- `verify_endpoint_data.py` … 主要 JSON API（認証必須の管理系は 401 を許容）
- `verify_stg_smoke.py` … 公開 HTML 全般 + `.env` に `DEV_PASSWORD` があれば `/cron-jobs` 等の開発者ページ

## 動作確認の目安

| 確認項目 | 期待 |
|----------|------|
| レスポンスヘッダ | 全レスポンスに `X-Keiba-Env: stg` |
| `GET /api/health` | JSON に `keiba_env`, `is_staging`, `gcs_bucket`, `gcs_prefix` |
| `GET /api/auth/status` | 上記に加え `is_developer` |
| ブラウザ | ナビにオレンジの **STG** バッジ |
| OpenAPI タイトル | `ML-AutoPilot Keiba [STG]` |

```bash
curl -sI http://127.0.0.1:8000/ | grep -i X-Keiba-Env
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool
```

## cron・バッチ

stg でも **GCS パスは .env のまま**です。負荷を抑えたい場合は `KEIBA_PROFILE=vps` や **`data/queue/queue_load_settings.json`**（`parallel` / `stagger_sec`）・環境変数 `SCRAPE_QUEUE_PARALLEL`（未設定時は **4**、運用で多い場合は **2〜3** 推奨）・`SCRAPE_QUEUE_STAGGER_SEC`（既定 **1.0** 秒）・`SCRAPE_QUEUE_THROTTLE_SEC`（ジョブ間ウェイト、既定 **1.2** 秒）を併用してください。

- **キュー smart_skip ポリシー**: カタログ上のタスクは **可変ページ＝既定で上書き再取得**、**`horse_pedigree_5gen` のみ不変＝既定スキップ**。ジョブ JSON に明示した `smart_skip` は従来どおり尊重します。旧 pending に `smart_skip: true` が残っている場合は `python3 -m src.scripts.scraping.bump_queue_pending_mutable_refresh` で除去できます（`--dry-run` あり）。
- **ワーカーログ**: 管理画面のライブログは `data/queue/.worker_log_ring.jsonl` と API `/api/scrape-queue/worker-logs` を参照します。表示されないときは **API プロセス再起動**（`scripts/server/restart_server.sh`）で `queue_worker_log` のハンドラが載り直ることを確認してください。

- **時刻**: ホストが UTC の場合、`crontab` の「何時に動くか」は **`CRON_TZ=Asia/Tokyo`** がないと UTC 解釈になります。`scripts/cron/setup_*.sh` で投入するブロックには `CRON_TZ=Asia/Tokyo` を含めています。既存の crontab は `install` を再実行するか、手動で各ブロック先頭に追記してください。
- **外部 cron とキュー**: `auto_scrape --task …` の **netkeiba 取得は既定で** `data/queue/scrape_queue.json` に投入され、キューワーカーが実行します（`KEIBA_AUTO_SCRAPE_USE_QUEUE=0` で従来の直取得に戻せます）。JRA 馬場ライブ・馬名インデックス・成長曲線はキュー対象外です。`python3 -m src.scripts.scraping.run_external_cron_month_coverage` は **金曜 `weekly-update` と同一の** `run_weekly_update_for_dates`（レース `race_result` / 指数は上書き再取得、`horse_profile` は成績・プロフィール上書き・血統ページは別ジョブ、`horse_pedigree_5gen` は未保持のみキュー投入・既存はスキップ）を窓内の全開催日に対して実行したうえで、`race_shutuba` / `smartrc` をキューへ載せます。「デイリ出馬表」APIトリガも利用できます。
- 定期実行: `scripts/cron/setup_raceday_eve_cron.sh`
- 状態ファイル: `data/local/meta/auto_scrape_status.json`（管理画面 `/cron-jobs` と同期）。`last_run` は **+09:00 付きの日本時間**で記録されます。

## 本番へ戻すとき

`.env` から `KEIBA_ENV` / `KEIBA_DEPLOYMENT_LABEL` を削除するか、`KEIBA_ENV=prod` に変更してプロセスを再起動します。

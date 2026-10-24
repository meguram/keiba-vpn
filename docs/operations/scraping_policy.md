# 取得（スクレイピング）の方針：誠実な形式

取得元（netkeiba など）の運営に支障を与えず、隠さず、制限を受けたら止まる。実装は `src/scraper/`（client.py・identity.py・access_guard.py）と `src/ops/`（fetch_agent.py・access_hold.py）。

## 1. 名乗る（User-Agent）
- 固定で識別できる UA だけを使う: `Mozilla/5.0 (compatible; keiba-research-bot/1.0; +<連絡先>)`（`src/scraper/identity.py`）。
- 連絡先は環境変数 `KEIBA_SCRAPER_CONTACT`（メールまたは URL）。リポジトリには書かない。未設定でも動くが、起動時に警告を出す。
- ブラウザを装わない: 複数ブラウザの UA の切り替え、`Sec-Ch-Ua`・`Sec-Fetch-*` などのブラウザ固有ヘッダーは使わない。
- 対象: NetkeibaClient（netkeiba・SmartRC）、JRA 系の取得（jra_baba_live・jra_calendar・jra_cushion・jra_facts）、診断・GCP 到達試験（deploy/gcp/netkeiba_smoke/smoke.py）。
- **注意**: 未知の UA を 400 で弾くサーバーがありうる。反映後の最初の取得で 400 が出たら、**その時点で止めて**原因を確認する（UA を偽装に戻さない。取得元に問い合わせる）。

## 2. 間隔と量
- 1 リクエストごとに 2.2〜4.0 秒（±15% のゆらぎ）。連続 14 回ごとに 6〜12 秒、60 回ごとに 22〜40 秒の休止（`NETKEIBA_THROTTLE_*`・`NETKEIBA_BURST_*`・`NETKEIBA_SESSION_COOLDOWN_*`）。
- 同時接続: プロセスあたり 1（`NETKEIBA_MAX_CONCURRENT_REQUESTS`、既定 1）。全プロセス合算で既定 2（`NETKEIBA_GLOBAL_MAX_SLOTS`。120 秒待っても空かないときだけ best-effort で続行）。
- 取得量の記録: `data/local/meta/scrape_timing/` と各ジョブのログ。問い合わせ・報告の際は、直近の 1 日あたりの件数を集計して示す。

## 3. 制限を受けたら止まる（回避しない）
- 429／503／403 は、**同じ UA・同じ IP**で、待機を延ばして上限回数（3 回）だけ再試行する。UA の切り替え・別の IP・別の経路（ホスト／VPS／GCP）への切り替えはしない。
- AccessGuard（400 以上のエラー（存在しないページを除く）・接続エラー・ブロックページ）が検知したら全停止し、キュー全体の一時停止フラグ（`scrape_access_pause`）を立てる。
- VPS の取得の実行役は、制限の疑いを受けたら**保留**（`data/local/meta/fetch_access_hold.json`）に入り、人が解除するまで新しいタスクを取らない。再起動しても解除されない。Slack に注意（重要度: 緊急）を出す。
  - 状態: `python -m src.ops.access_hold status`
  - 解除（原因を確認したあと）: `python -m src.ops.access_hold clear`
- 影モード（`config/fetch_shadow.json`）は、本番と同じ IP で動く。どちらかが制限を疑ったら両方止まる。対象は必要最小限（entries-sync・race-odds-t60/t15・race-weight-check・daily-race-lists）、1 日の上限 800 件、制限は 1 日 1 回で打ち切り。

## 4. 取得元への連絡
- 制限を受けた／UA が拒否された場合は、回避せず、取得元の窓口に問い合わせる。
- 公開・商用に使う前に、利用規約の「承認」を書面で得る（`docs/operations/security_hygiene.md`・メモリの netkeiba 規約のまとめを参照）。

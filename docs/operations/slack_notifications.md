# Slack 通知（カタログ・実装・有効化）

結果が揃った時点の**予測成績の通知**と、スクレイピングの**完了ステータス・異常の通知**。送信基盤は `src/utils/slack_notify.py`
（Block Kit の組み立て・送信・プレビュー）。既存の `src/utils/notify.py` の `notify_slack(text)`（cron 失敗・GCP コストなどのテキスト 1 件）は変更していない。
時間割は `docs/operations/stg_cron_schedule_20261014.md` と、監視ポータル（`docs/monitoring-portal.md`）の cron 一覧を参照。

## 1. 現状（調査結果。値は書かない）

- コードの設定項目は **`SLACK_WEBHOOK_URL`**（Incoming Webhook、`.env.example` に空で記載）のみ。Bot Token 方式は使っていない。
- ホストの `.env` には `SLACK_WEBHOOK_URL` の行がある（`.env.stg` には無い）。**調査時点で値は空**（= 通知は送られない）。`SLACK_BOT_TOKEN` / `SLACK_CHANNEL` は無い。
- 既存の呼び出し元（すべて `notify_slack`、送れなくても本処理は止めない）: FastAPI の cron 失敗（structure_check / disk_cache_cleanup / queue_maintain /
  logs_retention / daily_shutuba）、スクレイピングキューの長期失敗の滞留（`job_queue.check_stale_failed_jobs_and_notify`）、JRA 馬場のライブ取得（`jra_baba_live`）、
  構造監視（`structure_monitor`）、GCP の日次コスト（`gcp_daily_cost_report`）。
- つまり、**Webhook の値を設定すれば既存の通知はそのまま届き、今回の通知も同じ Webhook で届く**。

## 2. 環境変数（名前のみ。値は `.env` にだけ置く。コード・文書・ログに出さない）

| 変数 | 役割 | 未設定のとき |
|---|---|---|
| `SLACK_WEBHOOK_URL` | 既定の宛先（1 チャンネルで足りるならこれだけ） | 何も送らない |
| `SLACK_WEBHOOK_URL_RESULTS` | 日次の結果通知の宛先 | `SLACK_WEBHOOK_URL` を使う |
| `SLACK_WEBHOOK_URL_REPORT` | スクレイピングの日次サマリー・週次カバレッジの宛先 | `SLACK_WEBHOOK_URL` を使う |
| `SLACK_WEBHOOK_URL_ALERT` | 異常（注意/緊急）の宛先 | `SLACK_WEBHOOK_URL` を使う |
| `KEIBA_SLACK_NOTIFY` | `1` のときだけ今回の通知（結果・スクレイピング状況）を送る。**既定はオフ** | 送らない（dry-run は動く） |
| `KEIBA_SLACK_NOTIFY_ON_FINAL` | `1` なら、日次の予測成績が「確定」になった瞬間（`refresh_for_dates`）にも結果通知を送る（1 日 1 回に重複防止） | cron の再試行だけ |
| `KEIBA_PUBLIC_BASE_URL` | 結果通知の『詳細を開く』リンクの基点 URL（任意） | リンクなし |
| `KEIBA_NOTIFY_API_BASE` | スクレイピング状況が `/api/scrape-jobs` を読む API の URL（既定 `http://127.0.0.1:23220`） | 既定値 |

チャンネルの分け方の案: ①まず 1 チャンネル（`SLACK_WEBHOOK_URL` のみ）。②通知が増えたら `#…-results`（結果）と `#…-alerts`（異常。モバイルの通知を強めに）に分ける。

## 3. 通知カタログ

重要度: **通常**（青/緑）・**注意**（黄）・**緊急**（赤）。同じ内容の繰り返しは「抑制」の列のとおり。

| ID | 通知 | トリガー | 頻度 | 重要度 | 宛先 | 抑制 | 状態 |
|---|---|---|---|---|---|---|---|
| R1 | **日次の結果通知**（予測成績の集計・推奨馬券の回収率と的中率・印別・消印・累計） | その日の全レースの確定着順と払戻が揃った時（`prediction_results` が `final`） | 開催日に 1 回 | 通常 | results | 日付ごとの送信済みフラグ。確定しない日は 20:30 JST 以降に暫定で 1 回 | **実装済み** |
| S1 | **スクレイピングの日次サマリー**（信号 緑/黄/赤・タスク別・取得/失敗・スキーマ違反・欠損・次回予定） | 開催日は 21:30 JST、非開催日は 07:30 JST | 毎日 1 回 | 通常（赤なら注意の色） | report | 開催日/非開催日で片方だけ | **実装済み** |
| S2 | **スクレイピングの異常の即時通知** | 予定時刻から 20 分以上たっても始まらない／連続 2 回失敗（3 回で緊急）／実行中が 6 時間超／スキーマ違反の急増／403・429・DNS 系の失敗の連続（取得停止の兆候）／調教被覆の低下 | 15 分おきに判定 | 注意 / 緊急 | alert | 同じ異常は復旧まで 60 分おきに最大 3 回。重要度が上がったらすぐ。復旧時に『復旧しました』を 1 回 | **実装済み** |
| S3 | **週次のカバレッジ要約**（データヘルス、2020 年以降の充足、調教の被覆、スキーマ違反 7 日） | 月曜 08:00 JST | 週 1 回 | 通常 / 調教被覆が下限割れなら緊急の色 | report | — | **実装済み** |
| W1 | ドリフト監視のスコアカード（警告以上の段階） | 週次の `drift_monitor` 完了後 | 週 1 回（警告以上のみ） | 注意 / 緊急 | alert | 同じ段階は 1 回 | 案（`notify_event` で `latest.json` の段階を送る） |
| W2 | 凍結ルールの前向き追跡（tp-59/tp-62）の週次 | `frozen_tracking` の週次完了後 | 週 1 回 | 通常 | report | — | 案（`frozen-tracking` の成果物ができたら接続） |
| W3 | v4/v5 shadow の比較 | 週次 | 週 1 回 | 通常 | report | — | 案 |
| O1 | API / FastAPI / ランナーの停止 | ハートビート（`cron_runs` の更新）が止まる | 15 分おきに判定 | 緊急 | alert | S2 と同じ | S2 に含む（cron ジョブの未開始/停止） |
| O2 | ホストの負荷・メモリ・ディスクの逼迫 | load>18 が続く、空きメモリ<10GB、ディスク使用率>90% | 15 分おき | 注意 / 緊急 | alert | S2 と同じ | 案（`notify_event` で接続） |
| O3 | 予測のないレース、前日夜の予測生成の完了 | 前日 23:00 JST 以降 | 開催前日に 1 回 | 通常 / 注意 | report | — | 案 |
| E1 | 既存: cron 失敗・長期失敗の滞留・馬場取得・構造監視・GCP コスト | 各処理の失敗時 | 都度 | 注意 | default | 各処理の既存の仕様 | 既存（変更なし） |

案の項目は `slack_notify.notify_event(title, lines, severity=, channel=, link=)` で 1 行で送れる（送信基盤・リトライ・秘密のマスクは共通）。

## 4. 見た目（見本）

`docs/operations/slack_samples/*.json` が Slack に送るペイロード（`attachments[0].blocks`、左の帯の色が重要度/結果）。
Slack に近い見た目の HTML は、`--dry-run` で `data/local/meta/slack_preview/` に出る（コミットしない）。以下は同じ内容をテキストにしたもの。

- 本文 1 行目に『回収率・損益・的中』が見えるので、モバイルの通知とスレッドの冒頭だけで要点が分かる。詳細（券種別・印別・累計・注目）はその下。
- 結果通知の色: 回収率 100% 以上は緑、それ以外は青（赤にはしない。1 日の回収率は大きくぶれるため）。スクレイピング状況の色は信号（緑/黄/赤）。

### 日次の結果通知（回収率 100% 超の日の見本。合成データ）

```
┃ 結果 2026/10/18（日）［確定］
┃ 推奨馬券 回収率 130.0%　損益 +¥3,000　的中 5/12R
┃ 購入 → 払戻: ¥10,000 → ¥13,000          的中レース: 5/12 (41.7%)
┃ ◎ 勝率: 4/12 (33.3%)                     ◎ 3着内: 9/12 (75.0%)
┃ 勝率1位の的中: 3/12 (25.0%)              対象レース: 12R（予測なし 0R / 結果なし 0R）
┃ 券種別
┃ 券種    購入    払戻  回収率  的中/点
┃ 単勝   2,000   3,000    150%      2/6
┃ 3連複  8,000  10,000    125%     3/30
┃ 印別の着順
┃ 印  頭数  1着  連対  3着内
┃ ◎     12  33%   58%    75%
┃ 消      9   0%    0%     0%
┃ 消印　9頭のうち3着内 0頭（0.0%）（目標 5%以下）
┃ 累計（週・月・年の回収率と損益、先週比の矢印）／ 注目（穴の的中・大きな払戻・大外れ）／ モデル・LogLoss・詳細リンク
```

`results_dev_mock_20261017.json` は dev のモックデータの日（回収率 0%、累計つき）の実出力。

### スクレイピングの日次サマリー・異常（合成データ。赤の見本）

```
┃ スクレイピング状況 10/18（日）開催日　［要対応］
┃ ジョブ: 7 件（成功 5 / 失敗・未開始 3）        キュー: 実行 0 / 待機 2 / 失敗理由 8 件
┃ スキーマ違反（24h）: 34 件                      調教の被覆: 20%（2026-10）
┃ タスク別（グループ・件・成功・失敗/未開始）／ 問題のあるジョブ（状態・最終開始・所要）／ 取得結果（最終実行）／ 次回予定
┃ 検知中の異常
┃ • [緊急] 調教の取り直し が連続 3 回失敗
┃ • [緊急] 取得の停止の兆候: 403/429/DNS 系の失敗が 6 件
┃ • [注意] 夜間の一括取得 が予定時刻から 95 分たっても始まっていない
┃ • [緊急] 調教データの被覆が低下（2026-10: 20%）
```

## 5. 実装と使い方

| ファイル | 役割 |
|---|---|
| `src/utils/slack_notify.py` | Block Kit の部品（section/fields/context/表）、`build_message`（色の帯・ブロック数上限対応）、`send_payload`（429 の Retry-After・5xx の指数バックオフ・4xx は再試行しない・例外を投げない・Webhook URL をログ/エラーに出さない）、`send_test`、`notify_event`、プレビュー（HTML/テキスト） |
| `src/scripts/data/daily_results_notify.py` | R1。結果が揃ったかの判定、送信済みフラグ（`data/local/meta/slack_sent/results_<日付>.json`）、dry-run |
| `src/scripts/data/scrape_status_notify.py` | S1〜S3。`cron_status.build_status`（crontab と `cron_run.sh` の記録）、`/api/scrape-jobs`、`schema_violations`、`auto_scrape_status.json`、`data_health`、`drift_monitor` の読み取りのみ |
| `scripts/cron/daily_results_notify.sh` | R1 の cron。20 分おき。結果が揃うまで何もしない（終了コード 10 は正常な待機）。20:30 JST 以降は暫定で 1 回 |
| `scripts/cron/scrape_status_notify.sh` | S1〜S3 の cron（`alerts` / `summary-eve` / `summary-morning` / `coverage`） |
| `scripts/cron/install_slack_notify_cron.sh` | 上の cron をまとめて登録（タグ `KEIBA-SLACK-NOTIFY`、冪等、登録前に crontab を退避）。**ユーザーの承認後に root で実行** |
| `src/pipeline/inference/prediction_results.py` | `refresh_for_dates` の末尾に、`KEIBA_SLACK_NOTIFY=1` かつ `KEIBA_SLACK_NOTIFY_ON_FINAL=1` のときだけ確定日の結果通知を呼ぶフック（失敗しても無視） |
| `tests/scripts/test_slack_notify.py` | 整形・送信（429/5xx/4xx/例外/秘密のマスク）・結果判定・重複送信防止・異常の抑制と復旧・dry-run。送信はすべてモック |

### dry-run（送らずに見た目を確認する）

```
python -m src.scripts.data.daily_results_notify --date 20261018 --allow-provisional    # → data/local/meta/slack_preview/results_20261018.{json,html}
python -m src.scripts.data.scrape_status_notify --mode summary                          # → scrape_summary_*.{json,html}
python -m src.scripts.data.scrape_status_notify --mode alerts                           # 状態ファイルは更新しない
```

### 有効化の手順（ユーザーが判断して行う。ここまでは何も送っていない）

1. ホストの `.env` に Webhook の値を設定（`SLACK_WEBHOOK_URL`。分けるなら `_RESULTS` / `_REPORT` / `_ALERT`）と `KEIBA_SLACK_NOTIFY=1`。（任意で `KEIBA_SLACK_NOTIFY_ON_FINAL=1`、`KEIBA_PUBLIC_BASE_URL`）
2. 接続テスト（最小メッセージ 1 通）: `cd <repo> && python -m src.scripts.data.daily_results_notify --send-test`
3. 反映ファイルをホストに配置（`git pull` はしない。ファイル単位）。上の表の `src/` `scripts/cron/` `tests/` のうち `tests` 以外。`prediction_results.py` は既存ファイルなので差分適用（フック 1 関数と呼び出し 1 行）。
4. 直近の開催日で dry-run を実行して内容を確認 → `bash scripts/cron/install_slack_notify_cron.sh <repo>`（root）。
5. 止めるとき: `.env` の `KEIBA_SLACK_NOTIFY` を 0 にするか外す（cron は何もせず終了）。cron 行は `crontab -l | grep -v KEIBA-SLACK-NOTIFY` で外す。

### 終了コード（`daily_results_notify`）

0=完了 / 10=結果が未確定（待機）/ 11=送信済み / 12=開催なし / 13=無効 / 14=送信失敗（フラグは付けず次回再試行）/ 1=その他。

## 6. 注意

- 公開リポジトリのため、チャンネル名・URL・トークン・メンバー名はコード・文書・コミットに書かない（上の見本は合成/dev モックのデータ）。
- スクレイピング状況の判定は**読み取りのみ**で、netkeiba・GCS・DB へアクセスしない。DEC-023 のキュー待ち（`yield_to_rescrape` するジョブ）は未開始でも異常にしない。
- `crontab` の読み取りには root が要る（ホストの KEIBA 行は root の crontab）。cron は root で動かす。読めない場合、ジョブ一覧が空になり S2 は何も検知しない（S1 のジョブ欄は 0 件）ので、その旨をサマリーに出す改善余地あり。
- 開催日/非開催日の判定は `race_lists` の保存（無ければ土日）。会期カレンダー（`race_day_schedule`）に切り替える余地あり。
- 結果通知の回収率は「推奨馬券を上限 2,000 円/R で買った仮想購入」の集計（`prediction_results.md` と同じ）。実際の購入額ではない。

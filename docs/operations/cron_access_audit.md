# cron による外部サイトへのアクセスの監査（2026-10-22）

目的: netkeiba（ほか外部サイト）への不要なアクセスを減らし、負荷と規約上のリスク（会員向けページの取得を含む）を下げる。
根拠: ホストの実際の crontab（root）、`config/gcp_job_map.json`、各タスクのコード、`docs/html/modeling/chapters/feature_gap_audit.html`（使用中／採用ゼロ／未使用）、直近のバックフィルのログ。

## 結論
| 判定 | 件数（cron の行） | 内容 |
|---|---:|---|
| 維持 | 23 | 日々の予測・表示・特徴量に必要。進行中のバックフィル（2020・2021 の fast、全馬、調教）を含む |
| 頻度を下げる | 2 | 2026 年の fast・full のバックフィルを、毎日→週 1 回 |
| 停止（不要） | 7 | 完了済みのバックフィル 6 本（2022〜2025 の fast、2024・2025 の full）と、採用ゼロの `horse-remarks-daily` |
| 停止（研究専用・任意） | 1 | `sporthorse-births-nightly`（`--with-research` を付けたときだけ） |
| 対象外（外部アクセスなし） | 約 20 | ローカル計算・通知・死活監視・GCS 同期など |

## 見つかった無駄の正体
1. **完了済みのバックフィルが、毎日 8 本、ログインと会員ページの確認だけをして終わっていた。** 2022〜2026 年の fast と 2024〜2026 年の full は、ログ上「未処理 0 件」。それでも `ScraperRunner` を作って最後に `client.close()` を呼ぶと、遅延ログインの仕組みが働いて **ログイン → 会員限定のスピード指数ページの取得（ログイン有効性の確認）** が走っていた。
   - 修正: `NetkeibaClient.close()` が、通信していないときはログインしない（`src/scraper/client.py`、テスト `tests/scraper/test_close_no_login.py`）。これで、**作業の無い定期ジョブは、どれもログインしなくなる**（完了済みのバックフィルに限らない）。
   - 同じアカウントで何度もログインすると、動いている別のジョブのセッションを無効にする衝突もあった（コード内のコメントにある既知の問題）。
2. **`horse-remarks-daily`**（馬の戦績ページの備考・指数）は、現行モデルで採用ゼロ。直近のログで **1 日 476〜929 件**の馬ページを取得しており、不要なアクセスとして最も大きい。止める（必要なら `scripts/cron/horse_remarks_daily.sh` を手動実行）。週 1 回に減らしても、取得する馬の総数はほぼ変わらないため、減らすのではなく止める判断。
3. 2026 年の fast・full は、未処理が 0 件。新しい開催日の補填は `weekly-update`・`recent-retry` が行うため、週 1 回の安全網にとどめる。

## 判定の表（外部サイトに触れるタスク）
| タスク（cron_job_id） | 取るもの | 消費者 | 判定 | 根拠・影響 |
|---|---|---|---|---|
| daily-race-lists（朝・夕） | レース一覧 | 開催カレンダー、全取得の起点 | 維持 | 軽い（日付ごとに 1 ページ） |
| entries-sync（10 分おき確認） | 出馬表（公示・枠順確定の時点のみ） | 予測表示 | 維持 | 確認は 10 分おきでも、実取得は checkpoint のみ |
| jra-baba-morning | JRA のクッション値・含水率 | めぐ指数 | 維持 | 取得済みなら止まる想定。JRA は別サイト（規約は別途） |
| raceday-runner | 各レースのバンドル（T-60／T-15） | 予測、期待値、想定オッズ | 維持 | 開催日のみ。バンドルに指数・パドック・偏差値を含む（下の「依存」を参照） |
| raceday-result-runner | 速報結果（T+15） | 予測成績 | 維持 | 開催日のみ |
| raceday-evening | 結果・確定オッズ・SmartRC | 予測成績、評価 | 維持 | |
| raceday-eve | 翌日の出馬表・馬柱・追い切り・SmartRC | 予測の入力 | 維持 | 追い切りは特徴量に使用中 |
| raceday-weight-runner | 馬体重の検知 | 予測のタイミング | 維持 | |
| weekly-update（金） | 直近 10 日の確定結果・指数・偏差値・馬 | 学習母表、想定オッズ | 維持 | スピード指数と偏差値を含む（下の「依存」） |
| raceday-comment（朝・夜） | 厩舎コメント（会員向け） | 特徴量（tp-05g、使用中） | 維持 | ログインが要る唯一の採用中カテゴリ |
| new-horse-pedigree | 新馬の 5 世代血統 | 血統特徴量 | 維持 | |
| recent-retry（朝・夕） | 直近の未取得分 | データの完備 | 維持 | 未取得が無ければ通信なし（ログインもしなくなった） |
| barometer-retry（毎日・金） | 偏差値の取り直し | 想定オッズ（16 列） | 維持 | 想定オッズを再学習して外せば不要 |
| horse-training-backfill（月〜木の夜） | 調教の過去分（約 2,200 頭／晩） | 調教特徴量（採用） | 維持 | 進行中。503 で自動停止する |
| backfill-2021-fast／2020-fast | 過去分の補填 | 学習の母数 | 維持 | 未処理 73／108 日、53／108 日（進行中） |
| backfill-all-horse | 馬情報の一括 | 履歴特徴量 | 維持 | 未処理 14,980／39,707 頭（1 回に約 500 頭） |
| backfill-2026-fast／full | 新しい開催日の補填 | | **頻度減（毎日→週 1）** | 未処理 0 件。安全網のみ |
| backfill-2022〜2025-fast、2024・2025-full | 過去分の補填 | | **停止** | 未処理 0 件で完了済み（ログで確認）。毎回ログインと会員ページの確認だけをしていた |
| horse-remarks-daily | 馬の備考・スタート指数等 | なし（採用ゼロ） | **停止** | 1 日 476〜929 件。再開は手動実行で可 |
| sporthorse-births-nightly | 外部サイトの誕生日 | 研究のみ（活性値は不採用の結論） | **停止（任意）** | 外部サイトのセッション依存。必要なときは手動 |

外部アクセスの無い cron（ローカル計算・通知・同期など）: page_quality_check、update_jockey_trainer_stats、export_tables_daily、feature-accumulate、serving-state-*、horse-name-index、lap-forecast、tracking-v2、drift／frozen／odds-drift、slack-*、sync_pg_from_gcs、maintenance-deploy、server-watchdog ほか。

## 会員向けページと、ログインを不要にできるか
- 会員向けの取得: スピード指数（`race_index`）、パドック（`race_paddock`）、厩舎コメント（`race_trainer_comment`）。コード上 `need_login=True`。
- **現状、ログインは不要にできない。** 厩舎コメントが特徴量に使われている（tp-05g、追い切りとの合算で勝率 −0.0009）ため。
- **依存の注意**: 現行の想定オッズモデル（`models/final_odds_bundle.json`、ホスト版）の特徴量に、スピード指数由来 19 列、パドック 5 列、偏差値 16 列が入っている。これらの取得を止めるには、想定オッズモデルをそれらなしで再学習する必要がある（想定単勝の精度向上の作業と一緒に行うのが効率的）。再学習後に、スピード指数・パドック・偏差値の取得（バンドル・週次・barometer-retry）を外し、さらに厩舎コメントを外す判断をすれば、**ログイン自体が不要**になる。
- 推論の入力（`race_prediction_service`）が読むカテゴリは、取得対象に残っていることを `tests/scraper/test_serving_inputs_covered.py` で確認している。

## 見込み削減
- 完了済みバックフィル 8 本のログイン＋会員ページ確認: 1 日あたり約 24 リクエスト、ログイン 8 回、会員ページ 8 回 → 0。作業の無い他の定期ジョブも、ログインしなくなる。
- `horse-remarks-daily`: 1 日 476〜929 件（直近のログ） → 0。
- 取得対象のカテゴリは減らしていない（推論の入力は毎日そろう）。

## ホストへの適用（ユーザーが root で実行）
```bash
sudo bash ~/work/keiba-vpn/scripts/ops/cron_prune_apply.sh --dry-run            # 変更前後の行を表示
sudo bash ~/work/keiba-vpn/scripts/ops/cron_prune_apply.sh                      # バックアップを作って適用(/var/backups/keiba-cron/)
sudo bash ~/work/keiba-vpn/scripts/ops/cron_prune_apply.sh --with-research      # 研究専用の誕生日取得も止める場合
sudo bash ~/work/keiba-vpn/scripts/ops/cron_prune_restore.sh                    # 元に戻す（最新のバックアップ）
```
- 行は消さず、`# PRUNED 2026-10-22: 理由 | 元の行` の形でコメントにする。頻度を変えた行は `# RESCHEDULED …` の注記を残す。何度実行しても同じ結果。
- スクリプトは `stage_deploy`（`scripts/` 配下）でホストに置く。`client.py` の修正は、ステージ（`src/scraper/client.py`）で反映する。

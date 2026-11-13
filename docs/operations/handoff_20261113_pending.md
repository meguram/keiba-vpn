# 2026-11-13 の持ち越し（夜の作業の結果・残りの作業）

11/13 の夜に、バックグラウンドのセッションが引き継いで進めた（統括役の判断。ユーザーは不在で、推奨案で進めてよいと承認済み）。
残りは **④ root の crontab の変更（ユーザーの作業）** と、下の「次にやること」。

## 済んだこと（commit と CI）
| 内容 | commit | CI |
|---|---|---|
| ① 11/13 の週次の結果: rc=0（18:47）。めぐ指数の 10/31・11/01・11/07・11/08 は 4 日とも埋まった（race_result_flat に対する割合 1.0） | — | — |
| ② 毎朝の書き出しの GCS の全件一覧を 0 にした（本体 6 カテゴリ＋付随 7 カテゴリ、parse_drift_docs） | 3fcb2a2b | 緑 |
| ③ cron の登録スクリプト（共通ログインの時刻・sporthorse の外し方・serving-state のタグ）、job_map・一覧の HTML | 7c4a8adb | 緑 |
| ⑤ 毎晩の PG 同期の GCS の全件一覧をやめた | c03b86f7 | 緑 |
| 消した Cloud Scheduler 13 件を、master への昇格で作り直さない（job_map のグループに retired） | 7e1cc81f | 緑 |
| verify_past_values の全件一覧をやめた（対象はローカルの flat、基準は保存、読む件数の上限） | 280a6ab0 | 緑 |
| 開催日の後処理（馬場速度）の race_result の全件一覧をやめた | 4d424c7a | 見届け中 |
| 週次の odds_drift の全件一覧と全文書の読み出しをやめた | 8fe68f44 | 見届け中 |
| cron の登録スクリプトの安全な入れ替えを共通の関数に（scripts/cron/lib_crontab.sh。今日の 3 本） | f7edd68d | 見届け中 |
| setup_all_cron.sh の remove_all の修正、見出しの重なりを消す・verify の cron を入れ直すスクリプト、job_map の細かい点、docs の状態 | この docs の commit と同じ | — |

## ④ root の crontab の変更（未実施・ユーザーの作業）
2026-11-13 夜に `sudo bash scripts/cron/install_shared_login_cron.sh` を、Claude Code の権限の確認（auto mode。Destructive or Unwanted Actions）が止めた。回り込まずに止めた。
ユーザーが次を順に実行する。どのスクリプトも、変える前の crontab を `data/local/meta/cron_backups/` に退避し、次の場合は中止する（中止したら何も変わらない）。
- root でない。
- crontab -l が失敗した。
- 有効な行が少なすぎる。
- 行数が合わない。

```bash
cd /home/jovyan/work/keiba-vpn
sudo bash scripts/cron/install_shared_login_cron.sh                 # 共通ログインを 5 15 * * *（JST 00:05）に（今は 09:05）
sudo bash scripts/cron/install_sporthorse_births_cron.sh --remove    # root の sporthorse の行を外す（研究専用・毎日 SKIP）
KEIBA_CRON_BACKUP_DIR=$HOME/cron_backups bash scripts/cron/install_sporthorse_births_cron.sh --remove   # jovyan の行も外す（jovyan のまま）
sudo bash scripts/cron/install_serving_state_cron.sh                # serving-state の 2 行をタグ付きで入れ直す（時刻は同じ）
sudo bash scripts/cron/install_verify_past_values_cron.sh           # 過去分の検査を 2 時間おき → 1 日 1 回（JST 04:25）・cron_run.sh で包む
sudo bash scripts/cron/dedupe_setup_all_headers.sh --dry-run        # 見出しのブロックの重なり（6 回分）を確かめる
sudo bash scripts/cron/dedupe_setup_all_headers.sh                  # 重なりを消す（ジョブの行は 1 行も変わらない。PATH= の重なりだけ有効な行が減る）
sudo crontab -l | grep -nE 'KEIBA-SHARED-LOGIN|KEIBA-SPORTHORSE|KEIBA-SERVING-STATE|KEIBA-PAST-VALUE-CHECK'   # 確認
crontab -l | grep -c KEIBA-SPORTHORSE                               # 0
```
- 確かめること:
  - 共通ログインは `5 15`。
  - sporthorse の行は無い。
  - serving-state は 2 行。
  - 過去分の検査は `25 19` で cron_run.sh 付き。
- **`setup_all_cron.sh install` は使わない**。生成する行が、その後に手で整理した crontab より古く、廃止したジョブ（backfill-2024-full など）を戻すため。見出しの重なりは上の dedupe で消す。
- 入れ直したら、monitor-cloudrun-handoff（config/scheduled_jobs.json の持ち主）に知らせる。scheduled_jobs.json の audit_ok の「重複した行は無い」は、今は誤り（6 回分重なっている）。
- 入れ直すまでの影響:
  - 共通ログインは毎日 09:05 のまま（次の月替わりは 12/01 の 09:05）。
  - sporthorse は毎日 SKIP で動くだけ。serving-state は、タグが無いだけで動きは同じ。
  - 過去分の検査は 2 時間おきのまま。ただし GCS の一覧は取らず、未検査 0 件なので読み出しも 0。

## 次にやること（まだ手を付けていない）
- **古い登録スクリプト（install_* 約 15 本）を lib_crontab.sh にそろえる**。今日の 4 本（host_postprocess・vps_alive・vps_mirror_pull・shared_login）と verify は済み。sporthorse・serving_state・cron_catchup・jma_weather は独自の守りがある。
- **Cloud Run の API 側の GCS の全件一覧**（費用の監査役が回数と月額を見積もる。本番の API なので、直すのはユーザーが戻ってから計画を立てて）。どれも監視・運用の画面の**要求ごと**に呼ばれ、HybridStorage の一覧のキャッシュ（当年 60 秒・過去年 1 時間）が効く。
  | 場所 | 関数 | 呼ばれ方 |
  |---|---|---|
  | src/api/app.py:2004 | `_build_row_data_coverage`（GET /api/scrape-status） | 要求ごと（race_shutuba の当年の一覧） |
  | src/api/app.py:2141 | `get_date_race_matrix`（GET /api/date-race-matrix） | 要求ごと（カテゴリごとの一覧） |
  | src/api/app.py:2448 | `scrape_missing`（POST /api/scrape-missing） | 要求ごと（手で押す取り直し） |
  | src/api/app.py:3242 | `_build_scrape_summary_all`（GET /api/scrape-summary-all） | 要求ごと（全カテゴリ × 年の一覧） |
  | src/api/monitor_coverage.py:114 | `_build_gcs_matrix`（監視の網羅の表。app.py:2200〜2255、quality_health.py:237 から） | 要求ごと |
  | src/data_health/coverage.py:197 | `one(cat, year)`（データの健全性の網羅） | 呼び出し元の頻度による（要確認） |
- `src/pipeline/betting/scan_odds_phase.py:49` も race_odds の一覧を取るが、手で流す研究用（cron・job_map に無い）なので直していない。定期にするときは、odds_drift と同じく名前で読む。
- 監視（`src/scripts/data/scrape_status_notify.py` の collect_prod）は、job_map のグループの retired を見ていない。消したグループのジョブを「動いていない」と数え続ける可能性がある（費用には関係しない）。

## 11/14 朝 7:00 の書き出しで確かめること（②が入った最初の回）
- `logs/export_tables_daily.log` の最後の回:
  - 「一覧なしの追記」の行が 13 カテゴリ分あり、`LOCAL_EXPORT_FAIL`・`RUN_AXES_FAIL` が無い。
  - `update_run_axes` の出力がある。
  - parse_drift_docs の結果がある。基準の年は初回だけ読み、`data/local/meta/parse_drift_docs_baseline/` に保存される。
- `data/page_reference/tables/2026/_export_local_report.json` の errors が 0。
- race_result に 11/14 以降のレースが入っていない（当日は翌日から）。
- 1〜2 日後に、課金エクスポートで GCS の転送・Class A が減ったことを確かめる。

## 置き場（参考）
- 11/13 に用意したパッチ（適用済み）: `data/local/meta/pending_patches/20261113/`
- 走法の軸の 10/26 以降の穴埋め（`backfill_horse_lap --years 2026`）は、ユーザーが 11/13 に起動した。調教の過去分の取得（keiba-vpn-b0）が終わるまで待機中。終わった翌朝の update_run_axes が run_axes を作り直す。

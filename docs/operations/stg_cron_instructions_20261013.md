# stg PC 向け指示書: スクレイピング cron の点検と修正（2026-10-13）

背景は `docs/operations/scraping_sla_audit_20261013.md`。開発 PC の cron は壊れていて（cron の PATH の python3 に bs4 が無い・`auto_scrape.sh` の実行ビット欠落・月曜祝日開催の取りこぼし）、stg PC でも同じ問題が無いかを確認し、あれば直す。
**stg PC の現状は未確認。** 下の手順は「確認 → 該当するものだけ修正 → 検証」の順で、確認結果の報告を最後に求める。

## 0. 前提: コードの取り込み

この指示書とバロメーター取り直し（`barometer-retry`）は、開発 PC 側でコミット・プッシュされた後に `git pull` で入る。
**pull 前に手順 3 の `install` を実行しない**（`barometer-retry` が無いタスク名で cron が失敗する）。

```bash
cd <keiba-vpn のパス>
git pull
python3 -m src.scraper.auto_scrape --help | grep barometer-retry   # 出れば取り込み済み
```

## 1. 確認（読むだけ。変更しない）

```bash
cd <keiba-vpn のパス>
date; cat /etc/timezone 2>/dev/null; echo "KEIBA_ENV=$(grep ^KEIBA_ENV .env)"      # A. TZ と環境
crontab -l | grep -E "KEIBA|^PATH|^CRON_TZ"                                         # B. 登録内容
ls -l scripts/auto_scrape.sh scripts/cron/*.sh                                      # C. 実行ビット（x が付いているか）
for py in /opt/conda/bin/python3 "$(command -v python3)"; do echo "$py"; "$py" -c "import bs4,sys;print('bs4 ok')" 2>&1 | tail -1; done   # D. cron が使う python に bs4 があるか
env -i PATH=/usr/bin:/bin python3 -c "import bs4" 2>&1 | tail -1                    # E. cron 相当の PATH の python3
tail -n 5 logs/auto_scrape*.log logs/raceday_*.log logs/daily_race_lists_*.log logs/weekly_update.log 2>/dev/null | cut -c1-200   # F. 直近の実行結果
python3 -m src.scraper.auto_scrape --status | head -40                              # G. 各タスクの最終実行
```

### 判定表

| 確認結果 | 意味 | 対応 |
|---|---|---|
| A: TZ が UTC | `setup_all_cron.sh` の想定どおり | 手順 3A |
| A: TZ が JST | `setup_all_cron.sh install` は禁止（全ジョブが 9 時間ずれる） | 手順 3B |
| B: 空、または `KEIBA-VPN-ALL` タグが無い | cron 未登録 | 手順 3A/3B |
| B: `KEIBA-VPN-AUTO-SCRAPE` タグの行がある | 開発 PC と同じ手書き登録。`setup_all_cron.sh` では消えず二重実行になる | 手順 2 で先に除去 |
| C: `auto_scrape.sh` に x が無い | `Permission denied` で失敗する | `chmod +x scripts/auto_scrape.sh scripts/cron/*.sh` |
| D/E: bs4 が無い | cron から python3 を直接呼ぶ行は全滅する | 直接呼びをやめ、ラッパー（`run_auto_scrape_logged.sh` / `auto_scrape.sh`）経由にする。`setup_all_cron.sh` は前者を使う |
| F: `ModuleNotFoundError` / `Permission denied` | 上記のどれか | 該当行へ対応 |
| G: weekly-update / raceday-eve などが「未実行」 | 定期実行されていない | 手順 3 |

## 2. 旧手書きエントリの除去（B で `KEIBA-VPN-AUTO-SCRAPE` があった場合のみ）

```bash
crontab -l > /tmp/crontab.before.$(date +%Y%m%d%H%M)     # 必ずバックアップ
crontab -l | grep -v "KEIBA-VPN-AUTO-SCRAPE" | crontab -
crontab -l | grep -c "KEIBA-VPN-AUTO-SCRAPE"              # 0 になること
```

他プロジェクトの行（daily_news など）は消さない。

## 3A. UTC マシン: `setup_all_cron.sh` で登録

```bash
bash scripts/cron/setup_all_cron.sh show      # 内容のプレビュー（変更なし）。時刻が UTC 換算になっていること
bash scripts/cron/setup_all_cron.sh install   # 既存の keiba-vpn タグ行を消して再登録。他プロジェクトの行は残る
bash scripts/cron/setup_all_cron.sh status
```

このスクリプトの設計（JST）:

- 毎日 07:30 `raceday-runner` / `raceday-result-runner` … 非開催日は数秒でスキップ。**月曜祝日開催もこれで拾う**
- 毎日 17:30 `raceday-evening`、毎日 18:00 `raceday-eve`（翌日が開催日のときだけ動く）
- 毎日 07:00 / 17:00 `daily-race-lists`、05:00–09:00 の 10 分毎 `jra-baba-morning`
- 金 18:00 `weekly-update`、金 18:30 `horse-name-index`
- 毎日 08:30 と金 19:30 `barometer-retry`（今回追加。公開済み・未取得のバロメーターだけ取り直す）
- 深夜バックフィル、騎手・調教師統計、PG 同期、watchdog

## 3B. JST マシン: 雛形から登録

`setup_all_cron.sh` は使わない。`scripts/cron/crontab.jst.example` の先頭コメントの手順で登録する。**雛形にはバックフィル・PG 同期・watchdog は含まれない**ので、これらも必要なら別途判断（stg でこれらを回す前提なら、UTC マシンにするか、`setup_all_cron.sh show` の該当行を JST の時刻に直して追加する）。

## 4. 検証（登録直後）

```bash
crontab -l | grep -E "KEIBA|^PATH"
bash scripts/cron/setup_all_cron.sh test            # ドライラン検証（UTC マシン）
python3 -m src.scraper.auto_scrape --task barometer-retry   # 実リクエストは「公開済み・未取得」の分だけ。数秒で終わる
tail -n 20 logs/barometer_retry.log
```

翌日以降に次を確認する（失敗はログの `exit=` と `Traceback` で分かる）。

| いつ | 何を見る |
|---|---|
| 翌朝 07:00 / 07:30 | `logs/daily_race_lists_am.log`、`logs/raceday_runner.log`（非開催日は `skipped non-race-day` で正常） |
| 次の開催日の前日 18:00 | `logs/raceday_eve.log`（出馬表・馬柱・追い切りの取得） |
| 次の開催日 07:30〜 | `logs/raceday_runner.log`（各 R の T-60 / T-15）、`logs/raceday_result_runner.log` |
| 次の開催日 17:30 | `logs/raceday_evening.log`（速報結果・確定オッズ） |
| 次の金曜 18:00 以降 | `logs/weekly_update.log`、`logs/barometer_retry.log` |

stg の受け入れ判定は既存どおり `bash docs/todos/verify/stg_data_complete.sh`（`KEIBA_ENV=stg python -m src.data_health --require-complete`）。

## 5. 注意

- `.env` に netkeiba のログイン情報（`netkeiba_id` / `netkeiba_pw`）が必要（結果ラップ・バロメーター・馬柱で使用）。
- netkeiba へのアクセスは同時 1 本（DEC-023）。cron を足すときに既存ジョブと同時刻に重ねない（特に 07:30 前後）。
- `KEIBA_ENV=stg` が `.env` にあること。dev のままだと GCS ではなく `data/dev_mock` に保存される。
- `crontab -r`（全削除）は使わない。他プロジェクトの行が消える。

## 6. 報告してほしいこと

1. 手順 1 の A〜G の結果（TZ、`crontab -l` の keiba-vpn 行、`bs4` の有無、ログのエラー有無）
2. どの手順（2 / 3A / 3B）を実施したか
3. 手順 4 のドライラン結果と、翌日以降の確認結果（特に `raceday_runner` と `weekly_update` が成功したか）
4. **2026-10-16（金）18 時以降**: 月曜祝日（10/12）開催分のバロメーターが同じ金曜に公開されたか（`logs/barometer_retry.log` の `saved` / `pending`）

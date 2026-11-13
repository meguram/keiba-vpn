# 気象庁の時別値（風向・風速など）の取得

> **2026-11-13: 定期取得を廃止した**（ユーザーの判断「今使うところが無いなら削除して良い」）。
> - 理由: 本番の予測・推論の特徴量には使っていない。使うのは研究とコラムだけ（めぐコラム「競馬場ごとの風の影響」wind-venue は作成済みの静的ファイル、ptrain_wind・folklore_build・wind_adjust_2016、再学習の候補 14 は既定 no）。
>   再開した 11-13 13:40 の回で、毎時 `fetch --from 20160101` が月ファイル 1,310 件（約 345MiB）を読み直していた（月 約 4,400 円相当）ことも分かった。
> - 外したもの: ホストの root の crontab の `KEIBA-JMA-WEATHER` の行（`install_jma_weather_cron.sh --remove`。前の crontab は data/local/meta/cron_backups/ に退避）。GCP の jma-weather ジョブは `enabled=false`・`ready=false`（削除はしていない）。
> - 残したもの: コード（`src/scraper/jma_weather.py`・`scripts/cron/jma_weather_run.sh`。1 日 1 回・前回からの範囲だけを読む作りに直してある）と、GCS の月ファイル・`jma_weather_hourly_flat.parquet`（2026-11-12 までの分）。
> - 再開の手順（候補 14 を採用するときなど）:
>   1. 空白を手で 1 回補う: `sudo env KEIBA_ENV=stg PYTHONPATH=. /opt/conda/bin/python3 -m src.scraper.jma_weather fetch --from <最後に取れた日>`（範囲の月ファイルだけを読む。2016 年からにすると約 345MiB ≒ 6 円）。
>   2. 必要なら `export` で flat を作り直す。
>   3. `sudo bash scripts/cron/install_jma_weather_cron.sh` で毎時の行を戻す（取得は 1 日 1 回だけ。月 約 1.5〜3 円）。費用の見積もりを示してから行う。


目的: 競馬場ごとの風向・風速（レース当日の追い風・横風など）を、トラックバイアス・位置取り・ペースの分析と予測に使う。出典: 気象庁ホームページ（過去の気象データ検索）。

| 項目 | 内容 |
|------|------|
| コード | `src/scraper/jma_weather.py`（`stations` / `fetch` / `daily` / `report`） |
| 観測所 | `config/jma_stations.json`（競馬場ごとに、風を観測している現行の観測所のうち最寄り。距離つき）。競馬場の座標は住所からの概数 |
| 保存 | `HybridStorage` のカテゴリ `jma_weather_hourly`、キー `<競馬場コード>_<年>`（stg/prod は **GCS** の `others/jma_weather_hourly/`、dev は `data/dev_mock`）。中身は `days` に 日付 → 24 時間分 |
| 範囲 | 2016-01-01 以降。新しい日から取り、取得済みの日は取り直さない |
| 間隔 | 約 1.5 秒。エラー（200 以外・表が読めない）は 1 件で即中断し `data/local/meta/jma_weather_halt.json` を書く。確認して削除するまで再開しない。エラーには競馬場コード・日付・URL が付く。ただし昨日以降の日の「表が無い」は、まだ載っていないだけのことがあるので止めずに飛ばす（2026-10-26 0:40 に 1 回これで止まり、11-13 まで 18 日間止まっていた）。止めの印があるあいだ `jma_weather_run.sh` は終了コード 5 で終わる（cron の記録で失敗に見え、連続失敗の通知に乗る） |
| cron | 毎時 40 分に `scripts/cron/jma_weather_run.sh` が起動するが、取得は **1 日 1 回だけ**（6 時以降の最初の回）。範囲は `fetch --from <前回済んだ日の 2 日前>`〜昨日（済んだ記録が無ければ直近 3 日。31 日より前には戻らないので、それより長く止まったときは手で補う）。済んだ日は `data/local/meta/jma_weather_daily_done` に記録し、以降の回は GCS も読まずに終わる。失敗した日は `jma_weather_daily_fail` に数え、3 回を超えたら終了コード 6 でその日はもう回さない。費用は 1 回 約 2.7〜5.4MB（競馬場 10 × 月ファイル 1〜2 件 × 約 270KB）→ 月 約 1.5〜3 円。登録は `install_jma_weather_cron.sh`。GCP の jma-weather ジョブ（daily だけ・毎時）はこの記録を持てないので、ready=false にしてある |
| 過去分 | `fetch --from 20160101` は、取得済みの判定に月ファイル 1,310 件（約 345MiB ≒ 6 円）を毎回読むので、**定期では回さない**。空白を補うときだけ手で 1 回回す。2026-11-13 に、毎時これを回していて月 約 4,400 円相当の転送になっていたのを止めた（GCP の jma-weather ジョブの定義からも外した） |
| 列 | `hour`（1〜24。その時刻の観測＝直前 10 分の平均風）、`wind_dir`（16 方位＋静穏）、`wind_dir_deg`（北=0 から時計回り。静穏は null）、`wind_speed_ms`、`temp_c`、`precip_mm`、`humidity_pct`、`pressure_hpa`（気象台等のみ）、`sunshine_h`、`flags`（`)`=準正常値、`]`=資料不足値） |

風向は「風が吹いてくる方位」（北風＝北から南へ吹く）。コースの向きとの突き合わせでは、進行方向との関係を取り違えないこと。

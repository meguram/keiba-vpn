# 気象庁の時別値（風向・風速など）の取得

目的: 競馬場ごとの風向・風速（レース当日の追い風・横風など）を、トラックバイアス・位置取り・ペースの分析と予測に使う。出典: 気象庁ホームページ（過去の気象データ検索）。

| 項目 | 内容 |
|------|------|
| コード | `src/scraper/jma_weather.py`（`stations` / `fetch` / `daily` / `report`） |
| 観測所 | `config/jma_stations.json`（競馬場ごとに、風を観測している現行の観測所のうち最寄り。距離つき）。競馬場の座標は住所からの概数 |
| 保存 | `HybridStorage` のカテゴリ `jma_weather_hourly`、キー `<競馬場コード>_<年>`（stg/prod は **GCS** の `others/jma_weather_hourly/`、dev は `data/dev_mock`）。中身は `days` に 日付 → 24 時間分 |
| 範囲 | 2016-01-01 以降。新しい日から取り、取得済みの日は取り直さない |
| 間隔 | 約 1.5 秒。エラー（200 以外・表が読めない）は 1 件で即中断し `data/local/meta/jma_weather_halt.json` を書く。確認して削除するまで再開しない |
| cron | 毎時 40 分に `scripts/cron/jma_weather_run.sh`（flock で重複なし。直近 3 日の補い → 2016 年以降）。登録は `install_jma_weather_cron.sh` |
| 列 | `hour`（1〜24。その時刻の観測＝直前 10 分の平均風）、`wind_dir`（16 方位＋静穏）、`wind_dir_deg`（北=0 から時計回り。静穏は null）、`wind_speed_ms`、`temp_c`、`precip_mm`、`humidity_pct`、`pressure_hpa`（気象台等のみ）、`sunshine_h`、`flags`（`)`=準正常値、`]`=資料不足値） |

風向は「風が吹いてくる方位」（北風＝北から南へ吹く）。コースの向きとの突き合わせでは、進行方向との関係を取り違えないこと。

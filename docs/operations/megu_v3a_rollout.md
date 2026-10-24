# めぐ指数 v3a の本番反映 手順書（2026-10-19）

状態: 工程4-1（page_reference の parquet 差し替え）は完了。4-2〜4-5 は下表の順に実行する。

| 段 | 内容 | 前提 | 実行（ホスト） | 確認 | ロールバック |
|---|---|---|---|---|---|
| 4-2 | DB に model_version='v3a' を追加（係数19行＋megu_index 約31万行。v2 は不変） | 退避済み: data/local/backup/megu_db_before_v3a_20261019/ | `sudo -n env -u DATABASE_URL KEIBA_ENV=stg PYTHONPATH=. nice -n 19 python3 -m src.scripts.data.reflect_megu_v3a_to_db --execute` | 年ごとの件数ログ（DB の v3a 行 ≥ upsert 件数）。v2 の行数が row_counts.json と同じ | 同コマンド末尾を `--rollback --execute`（v3a の行だけ削除） |
| 4-3 | GCS 正本を退避してから v3a 版に更新（7 年分、約42MB、退避は約7MB） | 4-2 の検証後。**次の日次計算（金曜 weekly-update 18:00 JST など）より前** | `KEIBA_ENV=stg PYTHONPATH=. python3 -m src.scripts.data.reflect_megu_v3a_to_gcs`（dry-run）→ `--execute` | 各年 md5 一致のログ。再実行で `done: []` | `--rollback --execute`（退避 chuou/backup/megu_index_before_v3a_20261019/ から戻す） |
| 4-4 | 日次 cron を v3a に切替 | 4-3 の後 | ホストの `.env` に追記（下記）→ API 再起動（cron は次回起動から有効） | 翌日の raceday-evening / weekly-update 後に下記コマンド | `.env` の2行を削除（DB の v3a 行は残るだけで無害） |
| 4-5 | 他セッション(tp-39)へ通知 | 4-4 後 | SendMessage | — | — |

## 4-4 の設定

```
KEIBA_MEGU_V3A_DIR=<HOST_REPO>/config/megu_v3a
KEIBA_MEGU_V3A_DB=1
```

- `.env` は `src.scraper.storage` の import 時に読み込まれる（cron の auto_scrape・API・recent-retry すべてで有効）。cron のラッパーには入れない（一箇所に集める）。
- `KEIBA_MEGU_V3A_DIR` が無いと v3a の列は付かず、日次で追加された行の v3a 列が NaN になる（既存行は v3a 済みのまま）。`KEIBA_MEGU_V3A_DB=1` が無いと DB の v3a 行は日次では更新されない（parquet と GCS が正本）。

## 注意（順序）

日次計算（`compute_for_date`）は、毎回ローカルの parquet を GCS の正本へ上げる。**4-3 の前に日次計算が走ると、退避なしで GCS が上書きされる**ので、金曜の weekly-update 前に 4-3 と 4-4 を終える。

## 4-4 の確認（翌日の計算の後）

```
grep "めぐ指数" logs/raceday_evening.log | tail -3        # megu_v3a_db が出る（compute_for_date の戻り値は auto_scrape が記録）
python3 - <<'PY'
import pandas as pd
d = pd.read_parquet("data/page_reference/tables/2026/megu_index_flat.parquet")
last = d.sort_values("race_id").tail(300)
print(last[["megu_index_v3a_abs_robust", "megu_v3a_status", "megu_v3a_version"]].isna().mean())   # 新しい行にも v3a 列が付いている
PY
```

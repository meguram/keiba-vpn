# stg PC 向け指示書: ログイン状態に依存するデータの取り直し（2026-10-14）

背景: 学習データの整合性検証（`docs/html/modeling/chapters/training_pipeline_decisions.html` の tp-01・tp-02）で、`race_index`（タイム指数）が出走馬の約半数分しか入っていないこと、結果ページの `last_3f`・通過順・馬体重が一部期間で壊れていることが分かった。`parsers.py` 冒頭は、これらのページを「要ログイン（プレミアム）」と明記している。**取得時にログインが実質的に効いていなかった期間のデータが欠けている疑いが強い。**

この指示書は「確認 → 判定 → 取り直し → 検証」の順。**取り直し（手順 3）は、手順 2 の判定が通ってから始める。**

## 1. 現状（実測。2020〜2025 年、出馬表 20,733 レース・286,060 頭に対する取得率）

| カテゴリ | 要ログイン（`parsers.py`） | 頭数ベースの被覆 | 9 割以上そろったレース | 0 頭のレース | 判定 |
|---|---|---|---|---|---|
| `race_index` | 要ログイン | 47.4% | 34.8% | 8.7% | **取り直し対象**（13,514 レース） |
| `race_paddock` | 要ログイン | 11.8% | 0.1% | 63.4% | 要確認（全頭分が本来取れるカテゴリか未確認） |
| `race_oikiri` | 一部 | 57.7% | 5.4% | 0% | 要確認（1 頭複数行。同上） |
| `race_trainer_comment` | 要ログイン | 16.4% | 11.2% | 64.3% | 要確認（同上） |
| `race_result` の通過・上り | 要ログイン（プレミアム） | 値が壊れたレースが約 2,810 | – | – | 取り直しまたは `race_result_on_time` から補う |
| `race_odds` / `race_shutuba_past` / `race_result_lap`（flat） | 不要／一部 | 99.9〜100% | 99.9〜100% | 0% | 問題なし（ただし lap は上記 2,810 レースの値が壊れている） |
| `race_barometer` / `horse_training` | 要ログイン | 未測定 | – | – | 手順 2 のあとに測る |

- 取得日との関係（`race_index`、標本 40 + 40 レース）: 2026-03-24〜04-11 に取得したレースは全頭、2026-04-15 に取得したレースは 3 頭だけ。**同じコードで取得日により結果が違う**。
- 直近の実行（2026-10-13 21:26 JST 以降の学習 PC ランナー）でも、`race_index` は 40 レース中 9 割以上そろったのが 12（30%）、平均 6.4 頭（出馬表は平均 13.9 頭）。ログには `ログイン成功` が 55 回、失敗は 0 回。
- **ログイン成功の判定が弱い**: `src/scraper/client.py:837` は、ログインの POST 後に `nkauth` クッキーがあれば成功とする。会員状態が各ページで効いているか（全頭の指数が出るか）は確認していない。

## 2. 先にやること: ログインが実質的に効いているかの判定（少数のリクエストのみ）

netkeiba への負荷を避け、ランナー（`scrape_from_health_plan`）が止まっている時間帯に行う。

```bash
cd <keiba-vpn のパス>
python3 - <<'PY'
from bs4 import BeautifulSoup
from src.scraper.client import NetkeibaClient

c = NetkeibaClient()                       # auto_login=True
for rid in ("202509040410", "202305050704"):          # 出走頭数 11 と 16
    html = c.fetch(f"https://race.netkeiba.com/race/speed.html?race_id={rid}", use_cache=False)
    rows = BeautifulSoup(html, "html.parser").select("table.SpeedIndex_Table tr, table[class*='SpeedIndex'] tr")
    print(rid, "指数表の行数(目安):", len(rows), "| HTML 長:", len(html))
PY
```

| 結果 | 意味 | 次 |
|---|---|---|
| 行数が出走頭数に近い（9 割以上） | ログインは効いている。過去の取得時だけ効いていなかった | 手順 3 へ |
| 行数が 3 前後 | ログインしても全頭が出ない（会員状態・アカウント・セッションの問題） | 手順 3 は**始めない**。`.env` の `netkeiba_id` / `netkeiba_pw`、プレミアム会員の有効状態、同時ログイン（他 PC のセッションによる切断）を確認 |

あわせて、同じ手順で `race_paddock`・`race_trainer_comment`・`race_barometer` の 1 レースを取り、全頭分が返るかを見る（返らなければ、そのカテゴリは本来全頭分は取れない）。

## 3. 取り直し（手順 2 が通ってから）

対象一覧（開発側で生成済み。ホスト PC の `data/local/meta/rescrape/`）:

- `race_index_incomplete_2020_2025.txt`: `race_index` の取得頭数が出馬表の 90% 未満のレース（13,514 件、うち 0 頭 1,808 件。年別: 2020 年 2,250／2021 年 2,373／2022 年 2,298／2023 年 2,264／2024 年 2,124／2025 年 2,205）。
- 2026 年は不完全のため学習に使わない。2026 年分は別途、通常のスクレイピング計画に従う。

投入（1 レース 1 ジョブ。既存の単一レース投入 CLI を使う）:

```bash
# まず 1 件でドライラン
python3 -m src.scripts.scraping.enqueue_race_scrape_job --race-id 202509040410 --tasks race_index --overwrite --dry-run

# 年ごとに分けて投入（優先度は低め。通常の定期取得を妨げない）
while read rid; do
  python3 -m src.scripts.scraping.enqueue_race_scrape_job --race-id "$rid" --tasks race_index --overwrite --priority 100000
done < <(grep '^2025' data/local/meta/rescrape/race_index_incomplete_2020_2025.txt)
```

- 実行中の `scrape_from_health_plan` が終わってから、年単位（2025 → 2024 → … の順）で進める。1 年分（約 2,200 レース）ごとに手順 4 で検証してから次へ。
- netkeiba へのアクセスは既存の制限（同時 1 リクエスト、バースト制限、HTTP 400 のサーキットブレーカー）に従う。
- 結果ページの通過・上り（約 2,810 レース）は、`race_result` を取り直す代わりに、GCS の `race_result_on_time` から補えることを確認済み（上り 94〜99.6% 正常）。取り直す場合は `--tasks race_result` を同じ要領で。

## 4. 検証

各年の投入完了後、同じ基準で被覆を再計算し、`race_index_incomplete_summary.json` の件数が減ることを確認する。

```bash
python3 - <<'PY'
import polars as pl
T = "data/page_reference/tables"; Y = "2025"
sb = pl.read_parquet(f"{T}/{Y}/race_shutuba_flat.parquet", columns=["race_id","horse_number"]).select("race_id", pl.col("horse_number").cast(pl.Int64)).unique()
ix = pl.read_parquet(f"{T}/{Y}/race_index_flat.parquet", columns=["race_id","horse_number"]).select("race_id", pl.col("horse_number").cast(pl.Int64)).drop_nulls().unique()
n = sb.group_by("race_id").len().rename({"len":"n_sb"})
per = ix.join(sb, on=["race_id","horse_number"]).group_by("race_id").len().join(n, on="race_id", how="right").with_columns(pl.col("len").fill_null(0))
print("9割以上そろったレース:", (per["len"] >= 0.9*per["n_sb"]).mean())
PY
```

目標は 9 割以上そろったレースが 95% 以上。flat テーブルは取り直し後に再 export が必要（`python -m src.scraper.export_tables`。引数は `--help` で確認）。

## 5. 再発防止（コード修正の提案。この指示書では実施しない）

1. **ログイン有効性の検証**: ジョブ開始時（または一定間隔）に、既知の過去レースの指数ページを 1 件取り、行数が出走頭数の 9 割以上でなければ、そのセッションでの取得を止めて再ログインする。`client.py` の判定を、クッキーの有無だけにしない。
2. **取得頭数の検証**: `race_index`・`race_paddock`・`race_trainer_comment` などの保存時に、出馬表の頭数と比べ、不足なら `failure_reason=partial` として待機に戻す（`schema_validation` と同様の扱い）。
3. **データチェックの拡張**: `src/data_health` の判定に「出馬表頭数に対する被覆率」を追加し、本書 1 の表を自動で出す。
4. **結果パーサーの列検出**（`parsers.py:193-243`）: 14 列目が通過順に見えるかで表の種類を決めており、通過順が空欄・`**` の期間は既定の列位置に落ちて値がずれる。ヘッダー名で列を決める。

## 6. 報告してほしいこと

- 手順 2 の行数（2 レース分）と、パドック・厩舎コメント・バロメーターの 1 レースずつの返り頭数。
- ログイン失敗や同時ログインの疑いがあれば、その時刻とログ。
- 取り直しの進捗（年ごとの被覆率の前後）。

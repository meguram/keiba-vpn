# スクレイプ保存 JSON のスキーマ

## 正本（Python）

**定義の正本は `src/scraper/schema_defs.json`**（git 管理。全環境で同一）。`src/scraper/schemas.py` はそれを読み込み、`validate(category, data)` で検証します。`HybridStorage.save` 前後で診断用に呼ばれる想定で、型・必須キー・エントリ行の欠損を集計します（厳密なビジネスルールまでは含みません）。

- バージョン: `schemas.SCHEMA_VERSION`
- ユニットテスト: `tests/scraper/test_schema_examples_validate.py`（`tests/scraper/fixtures/schema_examples/*.json` を検証）

## JSON Schema（外部ツール用）

`json/` 以下に **JSON Schema（Draft 2020-12）** の例を置きます（現状は代表カテゴリのみ。正本との差分は PR で揃える運用を推奨）。

- `json/race_shutuba.schema.json` … 出馬表 `race_shutuba` の最小制約例
- 検証例: `tests/scraper/test_schema_examples_validate.py` 内の `test_race_shutuba_matches_jsonschema`（`jsonschema` 必須）

## 手動スモークとの関係

`tests/scraper/manual/requirements_sample_scrape_test.py` は、ネット取得の成否に加え **`schemas.validate`** の結果を `detail` に追記します。取得できてもスキーマ不一致なら `WARN` になります。


## 実データからの再構成（2026-10-06〜）

定義は手書きではなく、**実際に取得したデータの観測結果**から作る・見直す運用です。

```bash
# 1) 実データを走査して観測プロファイルを作る（加算マージ。何度でも、複数環境の結果も積み上げられる）
python -m src.scraper.schema_infer collect --source samples            # この PC: scrape_process_samples の実サンプル（1件ずつ）
KEIBA_ENV=stg python -m src.scraper.schema_infer collect --source storage --per-category 200   # 学習PC: GCS の実データ
python -m src.scraper.schema_infer collect --source dir:/path/to/mirror                         # GCS ミラー配置のディレクトリ

# 2) 現行スキーマとの差を見る（適合率・未定義キー・必須の出現率・型の食い違い）
python -m src.scraper.schema_infer report

# 3) schema_defs.json に反映（既定は保守的: 新カテゴリ(advisory)・未定義キーを任意で追加・provenance 更新のみ）
python -m src.scraper.schema_infer apply [--promote] [--demote] [--dry-run]
```

- 観測プロファイル（`observed/<category>.json`）も git 管理します。どの環境で何件見たか（`runs`）が残ります。
- `dev` のモック（`_meta.dev_mock`）は架空データなので、推論には**使いません**。
- 必須の判定は「**ファイル数が 30 以上**かつ出現率 99.5% 以上」。1 ファイル内の複数要素は相関するため、要素数ではなくファイル数で根拠を数えます。
- `advisory: true` のカテゴリは、不合格でも保存を止めず `_meta.schema_validation` に記録するだけです。実データで適合率を確認してから外して厳格化します。
- 型の食い違い・一度も観測されない定義済みキーは、自動では直さず report に出します（人が判断）。
- スキーマを持たないカテゴリは `schema_defs.json` の `no_schema_reason`（pending / unused / not_json）に理由を書きます。`tests/scraper/test_schema_infer.py` が「全カテゴリにスキーマか理由がある」ことを CI で確認します。
- `python -m src.data_health` が、環境ごとに **schema fingerprint**（定義が同一か）と**直近データの適合率**を確認し、`index.html` で環境間の不一致を警告します。


## 保存時のチェックと違反の記録

- スクレイピングしたデータは `HybridStorage.save` で**保存前に必ずスキーマ検証**します（`KEIBA_SCHEMA_STRICT` 既定 1 = 不合格は保存しない）。
- 不合格のとき、**どの項目がどの値で引っかかったか**（`field` / `rule` / `expected` / `actual` / `index` / `where`）を
  `data/local/meta/schema_violations/<category>.jsonl` に記録し、拒否したデータ本体を `data/local/quarantine/<category>/<key>.json` に隔離します
  （同じ key が後で合格して保存されたら削除）。確認: `python -m src.scraper.schema_violations summary|show|tail`。
- 隔離データは `schema_infer collect` の入力にもなり、スキーマの見直し（実データに合わせた再構成）に使えます。
- 2020 年以降の全データが**保存済みの状態でスキーマに適合しているか**は `python -m src.data_health --require-complete`
  （stg: `bash docs/todos/verify/stg_data_complete.sh`）で判定します。

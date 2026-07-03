# スクレイプ保存 JSON のスキーマ

## 正本（Python）

`src/scraper/schemas.py` の `SCHEMAS` と `validate(category, data)` が**正本**です。`HybridStorage.save` 前後で診断用に呼ばれる想定で、型・必須キー・エントリ行の欠損を集計します（厳密なビジネスルールまでは含みません）。

- バージョン: `schemas.SCHEMA_VERSION`
- ユニットテスト: `tests/scraper/test_schema_examples_validate.py`（`tests/scraper/fixtures/schema_examples/*.json` を検証）

## JSON Schema（外部ツール用）

`json/` 以下に **JSON Schema（Draft 2020-12）** の例を置きます（現状は代表カテゴリのみ。正本との差分は PR で揃える運用を推奨）。

- `json/race_shutuba.schema.json` … 出馬表 `race_shutuba` の最小制約例
- 検証例: `tests/scraper/test_schema_examples_validate.py` 内の `test_race_shutuba_matches_jsonschema`（`jsonschema` 必須）

## 手動スモークとの関係

`tests/scraper/manual/requirements_sample_scrape_test.py` は、ネット取得の成否に加え **`schemas.validate`** の結果を `detail` に追記します。取得できてもスキーマ不一致なら `WARN` になります。

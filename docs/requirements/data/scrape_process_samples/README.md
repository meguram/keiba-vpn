# scrape_process 用・実スクレイプ JSON サンプル（ローカル専用）

**このディレクトリの `*.json` は git 管理しません**（netkeiba の取得データのため、公開リポジトリに置かない。netkeiba 利用規約 第14条）。
手元で書き出して確認する用途のみ。コミットしないこと（`.gitignore` と `scripts/dev/secret_scan.sh` が検査します）。

以下は書き出し手順の記録です。

`docs/requirements/data/scrape_process.md` の「保存 JSON のサンプル（折りたたみ）」と同一内容の正本です。**ダミーではなく**、要件書 config のサンプル ID で取得した（またはローカル L2 に残っている）実データです。

## 更新手順

1. **フル取得して書き出し（推奨・netkeiba ログイン要）**

   ```bash
   python3 tests/scraper/manual/requirements_sample_scrape_test.py --export-samples
   ```

   `--quick` を付けると `skip_existing=True` になり、既にキャッシュや GCS 相当があればネット負荷を抑えます。

2. **既存の `data/cache/` と `page_reference/race_lists` だけ取り込み**

   ```bash
   python3 -m src.scripts.docs.gen_scrape_process_samples --from-cache
   ```

   キャッシュに無いカテゴリの `.json` は作られません（要件書の折りたたみからも欠落します）。

3. **Markdown のみ再整形**（`scrape_process_samples/*.json` を手編集したあと）

   ```bash
   python3 -m src.scripts.docs.gen_scrape_process_samples
   ```

`--from-cache` / `--export-samples` 実行時は、本ディレクトリの既存 `*.json` をいったん削除してから書き直します（古いダミー残骸防止）。

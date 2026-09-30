# data/page_reference/ ポータブルバンドル

> ⚠️ **このファイルはプレースホルダーです。**
> `AGENTS.md` / `docs/operations/PROJECT_LAYOUT.md` / `.gitignore`（`!data/page_reference/BUNDLE.md` の
> 除外例外）から本ファイルへの参照があったが、リポジトリ内のどこにも実体が存在しなかった
> （`evaluate-keiba-architecture` スキルによる 2026-09-30 のアーキテクチャ・ヘルスチェックで検出）。
> 元ファイルが見つからなかったため、他ドキュメント（下記「出典」）から確認できた情報のみで
> 再構成した。**「未確認」と書かれた項目は、実際にバンドルを作成・コピーした経験がある人が
> 確認・修正すること。**

## これは何か

`data/page_reference/` は、GCSクレデンシャル無しでもUI表示・一部の特徴量生成が動くように、
必要最小限のデータを事前計算・ローカル保存した**ポータブルバンドル**（AGENTS.md 記載）。
`.gitignore` でディレクトリ全体が除外対象だが、本ファイル（`BUNDLE.md`）だけは例外的に
追跡される（手順書のため）。

## 既知のサブパス（出典: DEC-025 / AGENTS.md）

| パス | 内容 | 出典 |
|---|---|---|
| `data/page_reference/race_lists/{YYYYMMDD}.json` | 開催日・race_id 一覧（ETLの日付軸） | `docs/decisions/DEC-025-gcs-postgresql-sync-and-monitoring.html` |
| `data/page_reference/race_day_schedule/{YYYYMMDD}.json` | 発走時刻スナップショット | 同上、`AGENTS.md`（要件表↔ストレージ整合の節） |
| `data/page_reference/tables/{YYYY}/race_result_flat.parquet` | flat特徴量（めぐ指数・モニターCalculated用） | 同上 |
| `data/page_reference/`（血統アーティファクト、`meta/person` 等） | 血統・人物メタ | `AGENTS.md` レイアウト表（内容の詳細は未確認） |

**未確認**: 上記以外に本ディレクトリ配下に存在するサブパスの全量（血統アーティファクトの具体的な
ファイル名・`meta/person` の構造など）。実際に populated な環境（stg機など）の
`find data/page_reference -type f | head -50` 等で確認して追記すること。

## 別PCへのコピー手順（未確認・一般的なrsync手順として記載）

```bash
# 元PC（データが揃っている環境）で圧縮
tar czf page_reference_bundle.tar.gz -C /path/to/keiba-vpn data/page_reference

# コピー先PCで展開（リポジトリルートで実行）
tar xzf page_reference_bundle.tar.gz
```

または直接同期する場合:

```bash
rsync -avz --progress <元PC>:/path/to/keiba-vpn/data/page_reference/ ./data/page_reference/
```

**未確認**: 増分同期の推奨方法（毎回全量tarで良いか、rsyncの`--delete`が必要か等）、
コピー後に必要な追加手順（インデックス再構築など）があるか。

## このファイルの更新について

内容を確認・修正した場合は、本冒頭の「プレースホルダー」注記を削除すること。

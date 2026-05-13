# keiba-vpn — エージェント向けメモ

競馬データ基盤（スクレイピング・特徴量・ML・Web）。README の詳細は `README.md`。

## レイアウト

| 領域 | 役割 |
|------|------|
| `api/` | FastAPI（`app.py`） |
| `templates/` `static/` | Jinja2 / CSS |
| `scraper/` | netkeiba / SmartRC / JRA 取得・パース・キュー |
| `pipeline/` | 学習・特徴量（LightGBM / XGBoost / CatBoost 等） |
| `research/` | 血統・コース・評価スクリプト |
| `data/` | 生データ・メタ・特徴量（大容量。不要な一括編集は避ける） |
| `data/meta/structure/` | 各 JSON の構造メタ |
| `data/features/` | 特徴量ストア。ブロック例: `base_tbl`（4キーのみの ``shutuba``）・`race_tbl`・`race_horse_tbl`・`horse_tbl`（血統・馬プロファイル等）・`race_jockey_tbl` / `race_trainer_tbl`・`jockey_tbl` / `trainer_tbl`・`jockey_trainer_stats/`（メタ）。**ラベル**: `target/rank_tbl/<年>/rank.parquet`（`race_id`,`horse_id`,`rank`）は `python -m pipeline.build_rank_target`。**馬単位エンティティ**（`horse/ped_tbl|result_tbl|training_tbl/{馬ID先頭4桁}/{horse_id}.parquet`、年ではなく馬シャード）は `python -m pipeline.build_horse_entity_store`。`docs/html/data_features_reference.html` の `#join-architecture`。出馬表 raw は `python -m pipeline.register_raw_table_features` → `_raw_table_feature_selection.json`。騎手・調教師は `python -m pipeline.build_jockey_trainer_stats`（定期: `scripts/update_jockey_trainer_stats.sh` 等）。 |
| `docs/modeling/` | レーティング・データセット仕様（設計の正） |
| `config/` `scripts/` `utils/` | 設定・cron・ユーティリティ |
| `main.py` | サーバエントリ（`python main.py --port 8000`） |

**ped_tbl 増分**: `python -m pipeline.sync_ped_tbl_for_horses --horse-ids …`（ローカル `horse_pedigree_5gen` 参照）。出馬表 `race_shutuba` 保存直後の自動生成は `.env` で `KEIBA_SYNC_PED_TBL_ON_SHUTUBA=1` のときのみ。接木は `KEIBA_PED_TBL_MERGE_GEN5`（未設定時 1）。

## 作業の指針（短く）

- 既存の命名・モジュール分割に合わせ、**依頼範囲だけ**変更する。
- ファイル探索は `Read` / リポジトリ `Grep` / `Glob` を優先（巨大 `data/` の丸読みは避ける）。
- モデリングや指標の意味を変える変更は、**`docs/modeling/` の該当仕様**と整合を取る。
- スクレイパや保存形式を触る場合は、検証系スクリプト（`scraper/validate_storage.py` 等）の有無を確認してから進める。

## 環境

- Python: `requirements.txt`、`.env.example` → `.env`（認証はユーザー環境）。
- テスト一式: リポジトリルートで `python3 -m unittest discover -s tests -t . -p 'test_*.py' -v`
- 騎手・調教師統計のマージキー検証: `python3 -m unittest tests.pipeline.test_jockey_trainer_stats -v`
- netkeiba 実 HTML を叩く手動スモーク（unittest 対象外）: `tests/scraper/manual/netkeiba_horse_page_smoke.py`, `tests/scraper/manual/netkeiba_speed_index_smoke.py`

## ユーザー向け応答

- ユーザーとの説明・コミットメッセージは **日本語**（プロジェクトルールに従う）。

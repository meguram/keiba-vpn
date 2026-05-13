"""
特徴量ストア `data/features/` のブロック別レイアウト（結合キー単位）。

- base_tbl: 予測ベースのスパイン（``shutuba.parquet``）= **race_id + horse_id + jockey_id + trainer_id のみ**（他列は載せない）
- race_tbl: レース単位 = race_id + 列
- race_horse_tbl: race_id + horse_id + 列（当該レース・当該馬の出馬表列・斤量・ID 有効フラグ・指数等）
- horse_tbl: horse_id + 列（血統・馬名・性齢など馬単位で正規化する出馬表由来・将来の通算統計）
- race_jockey_tbl: race_id + jockey_id + 列（レース内の騎手単位に正規化できる情報）
- race_trainer_tbl: race_id + trainer_id + 列（レース内の調教師単位）
- jockey_tbl / trainer_tbl: 騎手・調教師 ID 軸の参照表

ローカル flat テーブル（race_*_flat.parquet）は当面 race_id + horse_number を含む。
列ストアへの登録時に horse_id へ正規化し、馬番はストアに残さない。
"""

from __future__ import annotations

from pathlib import Path

FEATURES_DIR = Path("data/features")

BLOCK_BASE_TBL = "base_tbl"
BLOCK_RACE_TBL = "race_tbl"
BLOCK_RACE_HORSE_TBL = "race_horse_tbl"
BLOCK_RACE_JOCKEY_TBL = "race_jockey_tbl"
BLOCK_RACE_TRAINER_TBL = "race_trainer_tbl"
BLOCK_JOCKEY_TBL = "jockey_tbl"
BLOCK_TRAINER_TBL = "trainer_tbl"
BLOCK_HORSE_TBL = "horse_tbl"

JT_STATS_META_SUBDIR = "jockey_trainer_stats"

# 目的変数・ラベル（``_registry.json`` 外。``pipeline/build_rank_target.py`` で生成）
FEATURE_TARGET_SUBDIR = Path("target")
RANK_TBL_SUBDIR = FEATURE_TARGET_SUBDIR / "rank_tbl"

MERGE_KEYS_BASE_TBL: tuple[str, ...] = ("race_id", "horse_id", "jockey_id", "trainer_id")
MERGE_KEYS_RACE_TBL: tuple[str, ...] = ("race_id",)
MERGE_KEYS_RACE_HORSE_TBL: tuple[str, ...] = ("race_id", "horse_id")
MERGE_KEYS_HORSE_TBL: tuple[str, ...] = ("horse_id",)
MERGE_KEYS_RACE_JOCKEY_TBL: tuple[str, ...] = ("race_id", "jockey_id")
MERGE_KEYS_RACE_TRAINER_TBL: tuple[str, ...] = ("race_id", "trainer_id")
MERGE_KEYS_JOCKEY_TBL: tuple[str, ...] = ("jockey_id",)
MERGE_KEYS_TRAINER_TBL: tuple[str, ...] = ("trainer_id",)

# flat ソース読み込み用（馬番はここでのみ使用し、ストア書き出し前に除去）
SOURCE_ROW_KEYS: tuple[str, ...] = ("race_id", "horse_number")


def features_subdir(block: str) -> Path:
    return FEATURES_DIR / block


def jt_race_features_path(base: Path | str = ".") -> Path:
    """代表パス（年別分割時は最小年のファイル）。実データは ``race_horse_tbl/<YYYY>/jt_race_features.parquet``。"""
    b = Path(base)
    rhd = b / FEATURES_DIR / BLOCK_RACE_HORSE_TBL
    if rhd.is_dir():
        years = sorted(
            p.name for p in rhd.iterdir() if p.is_dir() and len(p.name) == 4 and p.name.isdigit()
        )
        for y in years:
            p = rhd / y / "jt_race_features.parquet"
            if p.is_file():
                return p
    legacy = rhd / "jt_race_features.parquet"
    return legacy


def jt_race_features_path_legacy(base: Path | str = ".") -> Path:
    b = Path(base)
    return b / FEATURES_DIR / JT_STATS_META_SUBDIR / "jt_race_features.parquet"


def jt_meta_dir(base: Path | str = ".") -> Path:
    """マニフェスト・マージ仕様 JSON の保存先（Parquet 本体はブロック別）。"""
    return Path(base) / FEATURES_DIR / JT_STATS_META_SUBDIR


def rank_tbl_dir(base: Path | str = ".") -> Path:
    """着順ラベル Parquet のルート: ``data/features/target/rank_tbl/``。"""
    return Path(base) / FEATURES_DIR / RANK_TBL_SUBDIR


def rank_parquet_path(year: str | int, base: Path | str = ".") -> Path:
    """1 年分: ``data/features/target/rank_tbl/<YYYY>/rank.parquet``。"""
    return rank_tbl_dir(base) / str(year) / "rank.parquet"

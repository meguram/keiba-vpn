"""
`data/local/tables/*_flat.parquet` から特徴量ストア用に選別した生列の定義。

方針:
  - `docs/html/modeling/pre_race_feature_whitelist.html` の LayerA 採用列を主に従う。
  - `race_shutuba_flat` / `race_index_flat` の odds・popularity は登録しない。
  - `race_result_flat` 等の結果確定後テーブルはここでは登録しない（別系統）。
  - `race_oikiri_flat` は race_id+horse_number が一意でないため、このモジュールの自動登録対象外。

**訓練データ結合（設計の正）**

- **予測ベース（スパイン）**: `base_tbl` の ``shutuba`` は **`race_id` + `horse_id` + `jockey_id` + `trainer_id` のみ**。
- **レース特徴量**: `race_id` をキーに `race_tbl` へ join。
- **馬特徴量（当該レースの馬行）**: `race_id` + `horse_id` で `race_horse_tbl`（斤量・当レース体重・ID 有効フラグ等）。
- **馬特徴量（馬単位・血統・プロファイル）**: `horse_id` で `horse_tbl`（馬名・性齢・父・母父など。将来の通算統計もここを想定）。
- **騎手・調教師特徴量**: `jockey_id` / `trainer_id` をキーに join。
- **舞台（コース条件）特徴量**: `place_id` + 芝ダ障 + 距離 + 列 W。実データでは `place_id` に相当するものとして主に `shutuba_venue_code`、芝ダは `shutuba_surface`、距離は `shutuba_distance` を用いる。

列ストアは ``pipeline/feature_layout.py`` のブロック配下に保存する（``base_tbl`` / ``race_tbl`` /
``race_horse_tbl`` 等）。結合キーは **馬番を使わず** ``race_id`` + 各種 string ``*_id`` を正とする。

**目標ディレクトリ**（ブロック名）:
``base_tbl``（4キーのみ）、``race_tbl``（``race_id``）、``race_horse_tbl``（``race_id``+``horse_id``）、
``horse_tbl``（``horse_id``・血統・馬プロファイル）、``jockey_tbl`` / ``trainer_tbl`` 等。

**ID の欠損表現**（`pipeline/id_value_policy.py` と登録スクリプトで実施）:
  文字列 ID は ``"0"`` や空を **null** にし、別列 ``shutuba_*_present`` で有効可否を **0/1** 表現する（0 を「ID値」として使わない）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawTableColumnSpec:
    """特徴量ストアに保存する1列の仕様。"""

    store_name: str
    """特徴量 stem（``data/features/<block>/{store_name}.parquet``）。"""
    source: str
    """`data/local/tables/{year}/{source}_flat.parquet` の {source}。"""
    column: str
    """ソース Parquet 内の列名。"""
    note: str = ""


# race_shutuba_flat: LayerA 採用列（odds / popularity 除外、キー列はストア側で常に含むため省略）
_SHUTUBA_RACE_LEVEL_COLS: tuple[str, ...] = (
    "date",
    "venue",
    "surface",
    "distance",
    "direction",
    "grade",
    "race_class",
    "weather",
    "track_condition",
    "start_time",
    "field_size",
    "race_name",
    "venue_code",
    "round",
    "weight_rule",
    "course_type",
)
_SHUTUBA_ENTRY_LEVEL_COLS: tuple[str, ...] = (
    "bracket_number",
    "horse_name",
    "horse_id",
    "sex_age",
    "jockey_weight",
    "jockey_name",
    "jockey_id",
    "trainer_name",
    "trainer_id",
    "weight",
    "weight_change",
    "sire",
    "dam_sire",
)

# race_horse_tbl の ``shutuba_race_horse.parquet``（race_id + horse_id）へ載せる当該レースの馬行
_SHUTUBA_RACE_HORSE_SOURCE_COLS: tuple[str, ...] = (
    "bracket_number",
    "horse_id",
    "weight",
    "weight_change",
    "jockey_weight",
)

# horse_tbl の ``shutuba_horse.parquet``（horse_id）へ載せる馬単位（血統・プロファイル）
_SHUTUBA_HORSE_PROFILE_SOURCE_COLS: tuple[str, ...] = (
    "horse_id",
    "horse_name",
    "sex_age",
    "sire",
    "dam_sire",
)

SHUTUBA_SPECS: tuple[RawTableColumnSpec, ...] = tuple(
    RawTableColumnSpec(f"shutuba_{col}", "race_shutuba", col, "出馬表・条件・斤量・馬名等（LayerA）")
    for col in (_SHUTUBA_RACE_LEVEL_COLS + _SHUTUBA_ENTRY_LEVEL_COLS)
)

SHUTUBA_RACE_LEVEL_STORE_NAMES: frozenset[str] = frozenset(f"shutuba_{c}" for c in _SHUTUBA_RACE_LEVEL_COLS)
SHUTUBA_ENTRY_LEVEL_STORE_NAMES: frozenset[str] = frozenset(f"shutuba_{c}" for c in _SHUTUBA_ENTRY_LEVEL_COLS)

# race_index_flat: 指数列のみ（odds / popularity 除外）
# time_index_m は HTML 上の推定抽出で欠損が 0 と区別しづらいため特徴量ストアには載せない。
# speed_recent はリストのため idx_speed_recent_1..3 に展開して登録（register 側で派生）。
INDEX_SPECS: tuple[RawTableColumnSpec, ...] = (
    RawTableColumnSpec("idx_speed_max", "race_index", "speed_max", "スピード指数・最大"),
    RawTableColumnSpec("idx_speed_avg", "race_index", "speed_avg", "スピード指数・平均"),
    RawTableColumnSpec("idx_speed_distance", "race_index", "speed_distance", "距離別スピード指数"),
    RawTableColumnSpec("idx_speed_course", "race_index", "speed_course", "コース別スピード指数"),
)

INDEX_SPEED_RECENT_EXPANDED_SPECS: tuple[RawTableColumnSpec, ...] = (
    RawTableColumnSpec("idx_speed_recent_1", "race_index", "speed_recent", "近3走・1走前相当(sk__index1)"),
    RawTableColumnSpec("idx_speed_recent_2", "race_index", "speed_recent", "近3走・2走前相当(sk__index2)"),
    RawTableColumnSpec("idx_speed_recent_3", "race_index", "speed_recent", "近3走・3走前相当(sk__index3)"),
)

# race_shutuba_past_flat: 生の JSON/テキストブロック（後段でパース・集約）
SHUTUBA_PAST_SPECS: tuple[RawTableColumnSpec, ...] = (
    RawTableColumnSpec("past_past_races", "race_shutuba_past", "past_races", "過去走ブロック（文字列）"),
    RawTableColumnSpec("past_training", "race_shutuba_past", "training", "調教ブロック（文字列）"),
)

# race_paddock_flat: 重複キーあり → 登録スクリプト側で last 一意化
PADDOCK_SPECS: tuple[RawTableColumnSpec, ...] = (
    RawTableColumnSpec("paddock_rank", "race_paddock", "paddock_rank", "パドック順位（要一意化）"),
    RawTableColumnSpec("paddock_comment", "race_paddock", "paddock_comment", "パドックコメント"),
)

# race_trainer_comment_flat: horse_number が float・欠損あり → キー正規化が必要
TRAINER_COMMENT_SPECS: tuple[RawTableColumnSpec, ...] = (
    RawTableColumnSpec("tcomment_comment", "race_trainer_comment", "comment", "厩舎コメント本文"),
    RawTableColumnSpec("tcomment_evaluation", "race_trainer_comment", "evaluation", "厩舎コメント評価"),
    RawTableColumnSpec("tcomment_trainer_name", "race_trainer_comment", "trainer_name", "コメント欄の調教師名"),
    RawTableColumnSpec("tcomment_questioner", "race_trainer_comment", "questioner", "取材者"),
)

ALL_RAW_SPECS: tuple[RawTableColumnSpec, ...] = (
    SHUTUBA_SPECS
    + INDEX_SPECS
    + INDEX_SPEED_RECENT_EXPANDED_SPECS
    + SHUTUBA_PAST_SPECS
    + PADDOCK_SPECS
    + TRAINER_COMMENT_SPECS
)

# 出馬表由来の string ID に付随する派生列（register_raw_table_features で登録）
SHUTUBA_ID_PRESENCE_STORE_NAMES: tuple[str, ...] = (
    "shutuba_horse_id_present",
    "shutuba_jockey_id_present",
    "shutuba_trainer_id_present",
)


def all_raw_feature_store_stems() -> tuple[str, ...]:
    """特徴量ストアに載る raw 系 stem の一覧。

    出馬表由来は ``shutuba``（base・4キーのみ）/ ``shutuba_race_horse`` / ``shutuba_horse`` /
    ``shutuba_race_jockey`` / ``shutuba_race_trainer`` に集約。
    レース条件列は従来どおり ``shutuba_<race_col>``（race_tbl）。
    """
    race_level = tuple(f"shutuba_{c}" for c in _SHUTUBA_RACE_LEVEL_COLS)
    head = ("shutuba", "shutuba_race_horse", "shutuba_horse", "shutuba_race_jockey", "shutuba_race_trainer")
    tail = tuple(
        s.store_name
        for s in (INDEX_SPECS + INDEX_SPEED_RECENT_EXPANDED_SPECS + SHUTUBA_PAST_SPECS + PADDOCK_SPECS + TRAINER_COMMENT_SPECS)
    )
    return head + race_level + tail

"""
``data/features/horse/`` 配下のエンティティストア（馬単位・レジストリ外）。

レイアウト::

  data/features/horse/
  ├── _manifest.json
  ├── ped_tbl/{shard4}/{horse_id}.parquet    # 血統: 牡スロットのみロング表（path_fm 等で位置）
  ├── result_tbl/{shard4}/{horse_id}.parquet # 成績: race_result_flat 由来（payoff/lap_times/pace は除く）
  └── training_tbl/{shard4}/{horse_id}.parquet  # 調教: ローカル JSON がある場合のみ（任意）

``shard4`` は ``horse_id`` の**先頭4文字**（``scraper/horse_pedigree_5gen_bulk`` のミラーと同じ）。
短い ID は左ゼロ埋めしてから先頭4文字に揃える。
年ディレクトリは持たず、**出走が増えたら当該馬のファイルだけ**を上書き更新する想定。
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline.features.feature_layout import FEATURES_DIR

HORSE_ENTITY_ROOT = FEATURES_DIR / "horse"
PED_TBL = "ped_tbl"
RESULT_TBL = "result_tbl"
TRAINING_TBL = "training_tbl"


def horse_entity_root(base: Path | str = ".") -> Path:
    return Path(base) / HORSE_ENTITY_ROOT


def horse_shard4(horse_id: str) -> str:
    """
    ローカル ``horse_pedigree_5gen/{shard}/{horse_id}.json`` および ``ped_tbl`` 等のシャード。

    数字だけ抜き出すと ``000a000e46`` が ``0000`` になりミラー実体（``000a/``）と不一致になるため、
    **生の horse_id 先頭4文字**を使う（``horse_pedigree_5gen_bulk.cmd_mirror_local`` と一致）。
    """
    s = str(horse_id).strip()
    if len(s) >= 4:
        return s[:4]
    if not s:
        return "0000"
    return s.zfill(4)[:4]


def ped_parquet_path(horse_id: str, base: Path | str = ".") -> Path:
    sh = horse_shard4(horse_id)
    return horse_entity_root(base) / PED_TBL / sh / f"{horse_id}.parquet"


def result_parquet_path(horse_id: str, base: Path | str = ".") -> Path:
    sh = horse_shard4(horse_id)
    return horse_entity_root(base) / RESULT_TBL / sh / f"{horse_id}.parquet"


def training_parquet_path(horse_id: str, base: Path | str = ".") -> Path:
    sh = horse_shard4(horse_id)
    return horse_entity_root(base) / TRAINING_TBL / sh / f"{horse_id}.parquet"


def is_male_pedigree_slot_upto(gen: int, position: int, *, max_generation: int = 10) -> bool:
    """血統表の牡スロット（偶数 position）。世代上限は将来の10世代まで拡張可能。"""
    if gen < 1 or gen > max_generation:
        return False
    try:
        pos = int(position)
    except (TypeError, ValueError):
        return False
    return pos % 2 == 0

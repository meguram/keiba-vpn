"""
ID 系カラムの欠損表現（0 埋めを避ける）。

問題:
  欠損を数値の ``0`` や文字列 ``"0"`` で表すと、**実在する ID が 0 である**ケースと区別できない。

方針（特徴量ストア ``data/features/<block_tbl>/`` 向け）:
  - **文字列 ID**（馬・騎手・調教師など）: 空・空白のみ・``"0"`` を **pandas の欠損（``pd.NA``）** にし、Parquet では null として保存する（``StringDtype``）。
  - **補助列** ``shutuba_{horse_id|jockey_id|trainer_id}_present`` : ``1`` = 上記ルールで有効な ID、``0`` = 欠損またはプレースホルダ。ここでの ``0`` は「ID 値そのもの」ではなく **真偽の否定** なので、ID の 0 とは混同しない。

主キー ``horse_number`` はレース上 1 始まりが前提のため、0 以下・欠損の行は結合キーとして不正として落とす（0 を「不明の馬番」と保存しない）。
"""

from __future__ import annotations

import pandas as pd

# 出馬表 flat で string として持つ netkeiba 系 ID（このモジュールの正規化対象）
NETKEIBA_STRING_ID_COLUMNS: frozenset[str] = frozenset({"horse_id", "jockey_id", "trainer_id"})


def sanitize_netkeiba_string_id(series: pd.Series) -> pd.Series:
    """空文字・空白のみ・数値 0 とみなす表記（``0`` / ``0.0`` / ``00`` 等）を欠損にし、有効値はトリムして返す。"""
    out = series.astype("string")
    t = out.str.strip()
    # 誤って数値化された 0 や文字列 "0" / "0.0" をプレースホルダとして欠損扱い（実 ID と混同しない）
    zero_like = t.str.fullmatch(r"0+\.?0*", case=False)
    missing = t.isna() | (t == "") | zero_like.fillna(False)
    return t.mask(missing, pd.NA)


def netkeiba_string_id_present(series: pd.Series) -> pd.Series:
    """1 = 有効な ID（null でない）、0 = 欠損またはプレースホルダ。dtype int8（欠損なし）。"""
    return sanitize_netkeiba_string_id(series).notna().astype("int8")

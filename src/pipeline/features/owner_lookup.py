"""推論時の馬主（owner）の引き当て。馬主系特徴量（factory_conn の ent_o / win_o / streak_o）の入力になる。

horse_result.owner（無ければ horse_profile.owner）を引く。分からない馬は owner=NaN と owner_missing=1。
取得経路は ``scraper.horse_profile_eve``（前日夕方の raceday-eve で horse_result が無い馬だけ取得する）。
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def owner_lookup(storage: Any, horse_ids: list[str], workers: int = 8) -> pd.DataFrame:
    """戻り値: horse_id, owner, owner_missing（0/1）。"""
    from src.scraper.horse_profile_eve import owner_table

    return owner_table(storage, horse_ids, workers=workers)

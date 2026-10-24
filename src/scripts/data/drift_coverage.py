"""入力データの供給（被覆）の点検: モデルの入力になる『馬ごとの取得データ』が、評価・予測の行に対してどれだけ揃っているか。

2026 年のバックフィル予測では、調教ページ(horse_training)が 4 月以降ほぼ揃っておらず、調教系の特徴量(tl_・wc_)が欠損だらけで予測された（drift_diagnosis 章）。
学習側(2020-2025)は全件取り直しで被覆がほぼ 100% なので、被覆の低下は『モデルの劣化』ではなく『入力データの不足』として現れる。週次監視で先に気づけるようにする。

  行(race_id, horse_id, date)ごとに、その馬の horse_training の保存記録に『そのレース日の調教の行』があるか(has_block)、記録そのものがあるか(has_rec)。
読み取りのみ（storage.load）。GCS/DB へは書かない。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd


def default_loader(workers: int = 8) -> Callable[[list[str]], dict[str, Any]]:
    def load_many(horse_ids: list[str]) -> dict[str, Any]:
        from src.utils.project_env import load_project_dotenv

        load_project_dotenv()
        from src.scraper.storage import HybridStorage

        st = HybridStorage()

        def one(h: str):
            try:
                return h, st.load("horse_training", h)
            except Exception:
                return h, None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return dict(ex.map(one, horse_ids))

    return load_many


def block_dates(records: dict[str, Any]) -> dict[str, set[str]]:
    """馬ごとの『調教の行がある race_date』の集合。"""
    from src.pipeline.features import training_serving as ts

    out: dict[str, set[str]] = {}
    for h, rec in records.items():
        if not rec:
            out[h] = set()
            continue
        laps = ts.records_to_laps({h: rec})
        out[h] = set(laps["race_date"].unique()) if not laps.empty else set()
    return out


def census(df: pd.DataFrame, load_many: Callable[[list[str]], dict[str, Any]] | None = None, blocks: dict[str, set[str]] | None = None,
           records_present: dict[str, bool] | None = None) -> pd.DataFrame:
    """df: race_id, horse_id, date の行。戻り値: 月ごとの行数・調教の行がある率(block)・記録がある率(rec)。"""
    d = df[["race_id", "horse_id", "date"]].copy()
    d["horse_id"] = d["horse_id"].astype(str)
    d["dstr"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    if blocks is None:
        recs = (load_many or default_loader())(sorted(d["horse_id"].unique().tolist()))
        blocks = block_dates(recs)
        records_present = {h: bool(r) for h, r in recs.items()}
    d["has_block"] = [dd in blocks.get(h, set()) for h, dd in zip(d["horse_id"], d["dstr"])]
    d["has_rec"] = [bool((records_present or {}).get(h, bool(blocks.get(h)))) for h in d["horse_id"]]
    d["ym"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m")
    g = d.groupby("ym").agg(rows=("has_block", "size"), block=("has_block", "mean"), rec=("has_rec", "mean")).reset_index()
    return g


def coverage_status(table: pd.DataFrame, floor: float = 0.90) -> dict:
    """直近の月の調教の被覆が floor を下回るか（学習側はほぼ 100%）。"""
    if len(table) == 0:
        return {"status": "no_data"}
    last = table.iloc[-1]
    return {"status": "alert" if float(last["block"]) < floor else "ok", "latest_month": str(last["ym"]), "block": float(last["block"]),
            "rec": float(last["rec"]), "floor": floor, "rows": int(last["rows"]),
            "table": table.round(4).to_dict("records")}

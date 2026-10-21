"""予測の入力の供給（調教）の保険: そのレースの出走馬のうち、そのレース日の調教の行が保存されている馬の割合を応答に付ける。

V5 の調教系の列（tl_*・wc_*、約 87 列）は、horse_training にそのレース日の行が無い馬では欠損のまま予測される（尖りが下がる）。
被覆が低いレースには注記を付け、利用者・監視が気づけるようにする。KEIBA_TRAINING_COVERAGE=1 のときだけ（既定は何もしない）。
読み取りのみ。失敗しても応答は返す。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

LEVELS = ((0.9, "ok"), (0.5, "low"), (0.0, "poor"))
NOTES = {
    "ok": None,
    "low": "調教データが揃っていない馬がいます。調教系の評価は参考程度に見てください。",
    "poor": "調教データがほとんど揃っていません（調教系の評価が欠けた予測です）。順位付けの確度は下がります。",
}


def enabled() -> bool:
    return os.environ.get("KEIBA_TRAINING_COVERAGE", "0").strip().lower() in ("1", "true", "yes")


def level_of(rate: float) -> str:
    for floor, name in LEVELS:
        if rate >= floor:
            return name
    return "poor"


def block_flags(records: Mapping[str, Mapping | None], race_date: str) -> dict[str, tuple[bool, bool]]:
    """馬ごとに (保存記録がある, そのレース日の行がある)。race_date は YYYY-MM-DD。"""
    from src.pipeline.features import training_serving as ts

    out: dict[str, tuple[bool, bool]] = {}
    for h, rec in records.items():
        if not rec:
            out[h] = (False, False)
            continue
        laps = ts.records_to_laps({h: dict(rec)})
        has = (not laps.empty) and bool((laps["race_date"] == race_date).any())
        out[h] = (True, has)
    return out


def summarize(flags: Mapping[str, tuple[bool, bool]]) -> dict[str, Any]:
    n = len(flags)
    if n == 0:
        return {"n": 0, "rate": None, "level": "unknown"}
    with_rec = sum(1 for r, _ in flags.values() if r)
    with_block = sum(1 for _, b in flags.values() if b)
    rate = with_block / n
    lvl = level_of(rate)
    return {"n": n, "n_with_record": with_rec, "n_with_block": with_block, "rate": round(rate, 3), "level": lvl, "note": NOTES[lvl]}


def attach_training_coverage(resp: dict, race_date: str, storage: Any = None) -> dict:
    """応答に training_coverage を付ける。無効・失敗のときは resp をそのまま返す。race_date は YYYYMMDD または YYYY-MM-DD。"""
    if not enabled():
        return resp
    try:
        preds = resp.get("predictions") or []
        digits = "".join(ch for ch in str(race_date) if ch.isdigit())
        if not preds or len(digits) != 8:
            return resp
        rdate = f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
        hids = [str(p.get("horse_id") or "") for p in preds if isinstance(p, dict) and p.get("horse_id")]
        if storage is None:
            from src.scraper.storage import HybridStorage

            storage = HybridStorage()

        def one(h: str):
            try:
                return h, storage.load("horse_training", h)
            except Exception:
                return h, None

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(hids)))) as ex:
            recs = dict(ex.map(one, hids))
        return {**resp, "training_coverage": summarize(block_flags(recs, rdate))}
    except Exception:  # noqa: BLE001
        return resp

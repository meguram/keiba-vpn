"""出走馬の馬主（owner）が未取得の馬だけ、馬の基本情報（horse_result）を取得して保存する（前日夕方 raceday-eve 用）。

馬主は出馬表になく、馬のプロフィールページ（horse_result.owner / horse_profile.owner）にだけある。新馬・初出走の馬は
horse_result がまだ無いので、T-60 の予測までに馬主が揃わない。ここでは horse_result が無い馬だけを取得する。
既存の horse_result は上書きしない（owner が空の既存オブジェクトは件数だけ数えて残す）。馬主は JRA の馬ではほぼ不変。
"""

from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger("scraper.horse_profile_eve")


def enabled() -> bool:
    return os.environ.get("KEIBA_EVE_HORSE_PROFILE", "0").strip().lower() in ("1", "true", "yes")


def _owner_of(storage: Any, horse_id: str, bypass: bool = False) -> tuple[bool, str]:
    """(horse_result が存在するか, owner)。horse_result に owner が無ければ horse_profile を見る。bypass=True は取得直後の確認用（キャッシュを使わない）。"""
    rec = storage.load("horse_result", horse_id, bypass_cache=bypass)
    exists = bool(rec)
    owner = str((rec or {}).get("owner") or "").strip()
    if not owner:
        prof = storage.load("horse_profile", horse_id, bypass_cache=bypass)
        owner = str((prof or {}).get("owner") or "").strip()
    return exists, owner


def classify(storage: Any, horse_ids: list[str], workers: int = 8, bypass: bool = False) -> dict[str, list[str]]:
    """出走馬を分類する。missing=horse_result が無い（取得対象）、no_owner=あるが owner が空（取得しない）、ok=owner あり。"""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda h: (h, *_owner_of(storage, h, bypass)), horse_ids))
    out: dict[str, list[str]] = {"missing": [], "no_owner": [], "ok": []}
    for h, exists, owner in res:
        out["ok" if owner else ("no_owner" if exists else "missing")].append(h)
    return out


def fetch_missing(runner: Any, horse_ids: list[str]) -> dict[str, Any]:
    """horse_result が無い馬を 1 頭ずつ取得して保存する（プロフィール+戦績。血統ページは取らない）。
    アクセス制限の疑い（AccessRestrictionDetected）が出たら中断する。"""
    from src.scraper.access_guard import AccessRestrictionDetected

    lo = float(os.environ.get("HORSE_PROFILE_EVE_MIN_SEC", "0.8"))
    hi = float(os.environ.get("HORSE_PROFILE_EVE_MAX_SEC", "2.0"))
    stats: dict[str, Any] = {"total": len(horse_ids), "saved": 0, "no_data": 0, "error": 0, "aborted": False}
    for i, hid in enumerate(horse_ids, 1):
        try:
            data = runner.scrape_horse(hid, skip_existing=True, with_history=True, skip_pedigree=True)
            stats["saved" if data else "no_data"] += 1
        except AccessRestrictionDetected as e:
            logger.error("アクセス制限の疑いで馬情報の取得を中断: %s", e)
            stats["aborted"] = True
            break
        except Exception as e:  # noqa: BLE001
            stats["error"] += 1
            logger.error("馬情報の取得に失敗 %s: %s", hid, e)
        if i % 25 == 0:
            logger.info("  馬情報 %d/%d: %s", i, len(horse_ids), {k: v for k, v in stats.items() if k != "total"})
        time.sleep(random.uniform(lo, hi))
    return stats


def fill_for_races(storage: Any, race_ids: list[str], runner: Any | None = None, *, dry_run: bool = False, max_horses: int = 400) -> dict[str, Any]:
    """レースの出走馬のうち horse_result が無い馬を取得する。戻り値に前後の分類件数を入れる。"""
    from src.scraper.horse_training_recent import horse_ids_for_races

    hids = horse_ids_for_races(storage, race_ids)
    before = classify(storage, hids)
    res: dict[str, Any] = {"horses": len(hids), "before": {k: len(v) for k, v in before.items()}}
    todo = before["missing"][:max_horses]
    if len(before["missing"]) > max_horses:
        res["capped"] = len(before["missing"]) - max_horses
    if dry_run or not todo:
        return res
    if runner is None:
        from src.scraper.run import ScraperRunner

        runner = ScraperRunner()
    res["fetch"] = fetch_missing(runner, todo)
    res["after"] = {k: len(v) for k, v in classify(storage, hids, bypass=True).items()}
    return res


def owner_table(storage: Any, horse_ids: list[str], workers: int = 8):
    """推論用: horse_id ごとの owner と、未取得フラグ（owner が分からない馬は owner=NaN, owner_missing=1）。"""
    import numpy as np
    import pandas as pd

    ids = [str(h) for h in horse_ids]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda h: _owner_of(storage, h)[1], ids))
    df = pd.DataFrame({"horse_id": ids, "owner": [o if o else np.nan for o in res]})
    df["owner_missing"] = df["owner"].isna().astype(int)
    return df

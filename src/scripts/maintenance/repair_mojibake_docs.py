#!/usr/bin/env python3
"""文字コードの誤判定で文字化けして保存された race.netkeiba.com 由来のデータを、修正済みクライアントで取り直して置き換える。

対象カテゴリ: race_result_on_time（速報結果）, race_shutuba_past（馬柱）, race_oikiri（追い切り）, race_paddock（パドック）,
race_index（タイム指数）, race_trainer_comment（厩舎コメント）。出馬表は repair_mojibake_shutuba.py（馬体重の引き継ぎあり）。

  python -m src.scripts.maintenance.repair_mojibake_docs --category race_oikiri --year 2026            # 対象の件数（既定はドライラン）
  python -m src.scripts.maintenance.repair_mojibake_docs --category race_oikiri --year 2026 --apply    # 取り直して置き換え

- 対象は、指定年の文書のうち「更新が --since〜--until（既定 2026-05-01〜2026-10-13 11:45 UTC＝文字コード修正の直前）」で、
  文字列に U+FFFD を含むもの。直った分は次回の対象外なので、中断しても再開できる。
- 置き換える前の JSON は data/local/backup/mojibake_docs/<category>/ に退避する。取り直した結果が文字化けのまま／項目が減った場合は、旧版へ戻す。
- ログインが要るカテゴリ（race_paddock, race_index, race_trainer_comment）は自動ログインする。同じアカウントの別セッションを無効にしうるので、
  無効になって取れなかった分は「読み取れなかった」として数え、再実行で拾う。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("maintenance.repair_mojibake_docs")
BAD = "�"
LOGIN_CATEGORIES = {"race_paddock", "race_index", "race_trainer_comment"}
METHODS = {
    "race_result_on_time": "scrape_race_result_on_time",
    "race_shutuba_past": "scrape_shutuba_past",
    "race_oikiri": "scrape_oikiri",
    "race_paddock": "scrape_paddock",
    "race_index": "scrape_speed_index",
    "race_trainer_comment": "scrape_trainer_comment",
}


def has_mojibake(obj: Any) -> bool:
    """文書のどこかの文字列に U+FFFD があるか（_meta は見ない）。"""
    if isinstance(obj, str):
        return BAD in obj
    if isinstance(obj, dict):
        return any(has_mojibake(v) for k, v in obj.items() if k != "_meta")
    if isinstance(obj, list):
        return any(has_mojibake(v) for v in obj)
    return False


def info_score(obj: Any) -> int:
    """文書の中の「空でない値」の数（_meta は除く）。取り直しで項目が消えていないかを見る目安。"""
    if isinstance(obj, dict):
        return sum(info_score(v) for k, v in obj.items() if k != "_meta")
    if isinstance(obj, list):
        return sum(info_score(v) for v in obj)
    return 0 if obj is None or obj == "" or (isinstance(obj, float) and obj != obj) else 1


def acceptable(new: dict | None, old: dict) -> bool:
    """取り直した結果が置き換えてよいか: 文字化けが無く、馬の行が減らず、空でない値の数が旧版の 9 割以上。
    レース後のページは表示が減るものがある（追い切りの調教日・ラップなど）ため、行数だけでなく値の数でも見る。"""
    if not new or has_mojibake(new):
        return False
    if len(new.get("entries") or []) < len(old.get("entries") or []):
        return False
    return info_score(new) >= 0.9 * info_score(old)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=sorted(METHODS))
    ap.add_argument("--year", default="2026")
    ap.add_argument("--since", default="2026-05-01T00:00:00+00:00")
    ap.add_argument("--until", default="2026-10-13T11:45:00+00:00")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.scraper.run import ScraperRunner

    cat = args.category
    runner = ScraperRunner(auto_login=cat in LOGIN_CATEGORIES)
    st = runner.storage
    if st._dev_store is not None or not st.gcs_enabled:
        raise SystemExit("GCS に接続できない環境（dev または未設定）です。KEIBA_ENV=stg で実行すること")
    lo = datetime.fromisoformat(args.since).timestamp()
    hi = datetime.fromisoformat(args.until).timestamp()
    stamps = st.batch_list_blobs(cat, args.year)
    window = sorted(k for k, v in stamps.items() if lo <= v < hi)
    targets = []
    for k in window:
        doc = st.load(cat, k, bypass_cache=True)
        if doc and has_mojibake(doc):
            targets.append(k)
    if args.limit:
        targets = targets[: args.limit]
    logger.info("[%s] 対象 %d / 期間内 %d / 全体 %d", cat, len(targets), len(window), len(stamps))
    if not args.apply:
        return 0

    bak = Path("data/local/backup/mojibake_docs") / cat
    bak.mkdir(parents=True, exist_ok=True)
    method = getattr(runner, METHODS[cat])
    stats = {"repaired": 0, "unreadable": 0, "failed": 0}
    for i, rid in enumerate(targets):
        try:
            old = st.load(cat, rid, bypass_cache=True)
            (bak / f"{rid}.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
            new = method(rid, skip_existing=False)
            if acceptable(new, old):
                stats["repaired"] += 1
            else:
                stats["unreadable"] += 1
                logger.warning("取り直した結果が使えないため旧版へ戻す: %s", rid)
                st.save(cat, rid, old)
        except Exception as exc:
            stats["failed"] += 1
            logger.error("失敗 %s: %s", rid, exc)
        if (i + 1) % 25 == 0:
            logger.info("進捗 %d/%d %s", i + 1, len(targets), stats)
        time.sleep(random.uniform(1.0, 2.0))
    logger.info("完了 %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

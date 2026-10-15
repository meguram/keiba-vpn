#!/usr/bin/env python3
"""文字コードの誤判定で文字化けして保存された race_shutuba を、修正済みクライアントで取り直して置き換える。

背景: クライアントの文字コード判定の修正（2026-10-13 20:45 JST）より前に取得した 2026 年の出馬表は、レース名・馬名・騎手名が文字化けし、
ヘッダ（クラス・日付・面・頭数など）が空で保存されていた（約 1,600 件）。
  python -m src.scripts.maintenance.repair_mojibake_shutuba --ids-file ids.txt            # 対象の確認（既定はドライラン）
  python -m src.scripts.maintenance.repair_mojibake_shutuba --ids-file ids.txt --apply    # 取り直して置き換え

- 対象は ids-file の race_id のうち、保存済みの JSON に文字化け（U+FFFD）があるもの。直った分は次回の対象外なので、中断しても再開できる。
- 置き換える前の JSON は data/local/backup/shutuba_mojibake/ に退避する。新しいページで読み取れなかった（文字化け・ヘッダが空）レースは触らない。
- 新しいページに馬体重が無い馬は、旧版の馬体重・増減を引き継ぐ。ログイン不要のページ。1 件ごとに 2〜4 秒空ける。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("maintenance.repair_mojibake_shutuba")
BAD = "�"


def is_mojibake(doc: dict | None) -> bool:
    if not doc:
        return False
    if BAD in str(doc.get("race_name", "")):
        return True
    return any(BAD in str(e.get("horse_name", "")) for e in (doc.get("entries") or []) if isinstance(e, dict))


def header_ok(doc: dict) -> bool:
    return bool(doc.get("date")) and bool(doc.get("surface")) and (doc.get("distance") or 0) > 0 and not is_mojibake(doc) and bool(doc.get("entries"))


def merge_weights(new: dict, old: dict) -> dict:
    """新しい出馬表に馬体重・増減が無い馬だけ、旧版（取得時点の値）を引き継ぐ。"""
    olds = {e.get("horse_number"): e for e in (old.get("entries") or []) if isinstance(e, dict)}
    out = dict(new)
    entries = []
    for e in new.get("entries") or []:
        e = dict(e)
        o = olds.get(e.get("horse_number"))
        if o and not e.get("weight") and o.get("weight") and o.get("horse_id") == e.get("horse_id"):
            e["weight"], e["weight_change"] = o.get("weight"), o.get("weight_change", 0)
        entries.append(e)
    out["entries"] = entries
    meta = dict(out.get("_meta") or {})
    meta["mojibake_repaired_at"] = datetime.now().isoformat(timespec="seconds")
    out["_meta"] = meta
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.scraper.run import ScraperRunner

    runner = ScraperRunner(auto_login=False)   # 出馬表はログイン不要
    st = runner.storage
    if st._dev_store is not None or not st.gcs_enabled:
        raise SystemExit("GCS に接続できない環境（dev または未設定）です。KEIBA_ENV=stg で実行すること")
    ids = [x.strip() for x in args.ids_file.read_text().split() if x.strip()]
    targets = [k for k in ids if is_mojibake(st.load("race_shutuba", k, bypass_cache=True))]
    if args.limit:
        targets = targets[: args.limit]
    logger.info("対象 %d / %d レース", len(targets), len(ids))
    if not args.apply:
        print("\n".join(targets[:20]))
        return 0

    bak = Path("data/local/backup/shutuba_mojibake")
    bak.mkdir(parents=True, exist_ok=True)
    stats = {"repaired": 0, "unreadable": 0, "failed": 0}
    for i, rid in enumerate(targets):
        try:
            old = st.load("race_shutuba", rid, bypass_cache=True)
            html = runner.client.fetch(runner.RACE_CARD_URL.format(race_id=rid), use_cache=False)
            new = runner.card_parser.parse(html, race_id=rid)
            if not header_ok(new):
                stats["unreadable"] += 1
                logger.warning("読み取れませんでした: %s", rid)
            else:
                (bak / f"{rid}.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
                if st.save("race_shutuba", rid, merge_weights(new, old)):
                    stats["repaired"] += 1
                else:
                    stats["failed"] += 1
                    logger.warning("保存できませんでした: %s", rid)
        except Exception as exc:
            stats["failed"] += 1
            logger.error("失敗 %s: %s", rid, exc)
        if (i + 1) % 25 == 0:
            logger.info("進捗 %d/%d %s", i + 1, len(targets), stats)
        time.sleep(random.uniform(2.0, 4.0))
    logger.info("完了 %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

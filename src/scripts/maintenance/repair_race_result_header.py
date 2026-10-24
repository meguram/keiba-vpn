#!/usr/bin/env python3
"""距離・面が空で保存された race_result を、修正済みパーサで取り直して「面・方向・距離」だけ書き換える。

背景: 2026 年にページの表記が「芝右 外1200m」と内外回りを含む形に変わり、旧パーサが読めず距離 0・面が空で保存されていた
（2026 年の 260 レース）。他の項目（着順・払戻・ラップなど）は変えない。

  python -m src.scripts.maintenance.repair_race_result_header --year 2026            # 対象の一覧（既定はドライラン）
  python -m src.scripts.maintenance.repair_race_result_header --year 2026 --apply    # 取り直して書き換え

- 対象は、GCS の race_result のうち surface が空かつ distance が 0（以下）のもの。直った分は次回の対象外なので、中断しても再開できる。
- 書き換える前の JSON は data/local/backup/race_result_header/ に退避する。読み取れなかったレースは触らない。
- 元ページへのアクセスは 1 件ごとに 2〜4 秒空ける（ログイン不要のページ）。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("maintenance.repair_race_result_header")


def needs_repair(doc: dict | None) -> bool:
    return bool(doc) and not doc.get("surface") and (doc.get("distance") or 0) <= 0


def patch_header(old: dict, new: dict) -> dict | None:
    """old の面・方向・距離だけを new の値で置き換えた dict を返す。new が読めていなければ None。"""
    if not new.get("surface") or (new.get("distance") or 0) <= 0:
        return None
    out = dict(old)
    for k in ("surface", "direction", "distance"):
        out[k] = new.get(k, "" if k != "distance" else 0)
    meta = dict(out.get("_meta") or {})
    meta["header_repaired_at"] = datetime.now().isoformat(timespec="seconds")
    out["_meta"] = meta
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="先頭から N 件だけ（試し用）")
    ap.add_argument("--ids-file", type=Path, help="対象 race_id の一覧（1 行 1 件）。省略時は GCS 全件を調べる（遅い）")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.scraper.run import ScraperRunner

    runner = ScraperRunner(auto_login=False)   # 結果ページはログイン不要。ログインすると他のジョブのセッションを無効にしうる
    st = runner.storage
    if st._dev_store is not None or not st.gcs_enabled:
        raise SystemExit("GCS に接続できない環境（dev または未設定）です。KEIBA_ENV=stg で実行すること")
    if args.ids_file:
        keys = [x.strip() for x in args.ids_file.read_text().split() if x.strip()]
        targets = [k for k in keys if needs_repair(st.load("race_result", k, bypass_cache=True))]
    else:
        keys = sorted(k for k in st.batch_list_blobs("race_result", args.year))
        targets = [k for k in keys if needs_repair(st.load("race_result", k, bypass_cache=True))]
    if args.limit:
        targets = targets[: args.limit]
    logger.info("対象 %d / %d レース", len(targets), len(keys))
    if not args.apply:
        print("\n".join(targets[:20]), "..." if len(targets) > 20 else "")
        return 0

    bak = Path("data/local/backup/race_result_header")
    bak.mkdir(parents=True, exist_ok=True)
    stats = {"repaired": 0, "unreadable": 0, "failed": 0}
    for i, rid in enumerate(targets):
        try:
            old = st.load("race_result", rid, bypass_cache=True)
            html = runner.client.fetch(runner.RACE_RESULT_URL.format(race_id=rid), use_cache=False)
            new = runner.result_parser.parse(html, race_id=rid)
            patched = patch_header(old, new)
            if patched is None:
                stats["unreadable"] += 1
                logger.warning("読み取れませんでした: %s", rid)
            else:
                (bak / f"{rid}.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
                if st.save("race_result", rid, patched):
                    stats["repaired"] += 1
                else:
                    stats["failed"] += 1
                    logger.warning("保存できませんでした: %s", rid)
        except Exception as exc:
            stats["failed"] += 1
            logger.error("失敗 %s: %s", rid, exc)
        if (i + 1) % 20 == 0:
            logger.info("進捗 %d/%d %s", i + 1, len(targets), stats)
        time.sleep(random.uniform(2.0, 4.0))
    logger.info("完了 %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""dev ストア（data/dev_mock）に入ってしまった実データのうち、GCS に無いものだけを GCS へ追加する（上書きしない）。

背景: 2026-10-09〜10 に stg ホストの cron が KEIBA_ENV=dev で動き、取得データの一部が GCS ではなく dev ストアに保存された。
  python -m src.scripts.maintenance.restore_dev_store_to_gcs --since 2026-10-09            # 何を足すかの一覧（既定はドライラン）
  python -m src.scripts.maintenance.restore_dev_store_to_gcs --since 2026-10-09 --apply    # 足す

- 対象は、--since 以降に更新された、モックでない（_meta.dev_mock が無い）JSON。local_only のカテゴリは除く。
- 既に GCS にあるキーは触らない。書き込みは ``if_generation_match=0``（無いときだけ作成）なので、競合しても上書きしない。
- スキーマに適合しないものは足さない。元の _meta（取得時刻など）は保ち、復元した旨を _meta.restored_from に記録する。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("maintenance.restore_dev_store_to_gcs")


def candidates(root: Path, since_ts: float) -> list[tuple[str, str, Path]]:
    """(カテゴリ, キー, パス)。dev ストアの JSON のうち since 以降に更新されたモックでないもの。"""
    out = []
    for p in sorted(root.rglob("*.json")):
        try:
            if p.stat().st_mtime < since_ts:
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or (d.get("_meta") or {}).get("dev_mock"):
            continue
        out.append((p.relative_to(root).parts[0], p.stem, p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="YYYY-MM-DD。この日以降に更新された dev ストアのファイルが対象")
    ap.add_argument("--apply", action="store_true", help="GCS に書く（省略時はドライラン）")
    ap.add_argument("--root", type=Path, default=Path("data/dev_mock"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.scraper import schemas
    from src.scraper.storage import HybridStorage

    st = HybridStorage()
    if st._dev_store is not None or not st.gcs_enabled:
        raise SystemExit("GCS に接続できない環境（dev または未設定）です。KEIBA_ENV=stg で実行すること")
    since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    bucket = st._get_bucket()
    stats = {"対象": 0, "既にGCSにある": 0, "追加": 0, "スキーマ不適合": 0, "local_only": 0, "失敗": 0}
    by_cat: dict[str, int] = {}
    for cat, key, path in candidates(args.root, since):
        stats["対象"] += 1
        if st.CATEGORY_MAP.get(cat) in (None, "local_only"):
            stats["local_only"] += 1
            continue
        blob = bucket.blob(st._gcs_blob_path(cat, key))
        if blob.exists():
            stats["既にGCSにある"] += 1
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        vr = schemas.validate(cat, data)
        if not vr.get("passed") and "skipped" not in vr:
            stats["スキーマ不適合"] += 1
            logger.warning("スキーマ不適合のため足さない: %s/%s", cat, key)
            continue
        by_cat[cat] = by_cat.get(cat, 0) + 1
        if not args.apply:
            continue
        data.setdefault("_meta", {})["restored_from"] = f"dev_mock {datetime.now().date()}"
        try:
            blob.upload_from_string(json.dumps(data, ensure_ascii=False, indent=1), content_type="application/json",
                                    if_generation_match=0, timeout=30)
            stats["追加"] += 1
        except Exception as exc:  # 競合（他が先に作った）も含め、上書きはしない
            stats["失敗"] += 1
            logger.warning("追加できなかった %s/%s: %s", cat, key, exc)
        time.sleep(0.02)
    logger.info("%s: %s | カテゴリ別の追加対象 %s", "適用" if args.apply else "ドライラン", stats, by_cat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

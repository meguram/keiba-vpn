"""
クッション値 JSON を GCS（HybridStorage カテゴリ jra_cushion）へ年別保存する。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scraper.storage import HybridStorage


def upload_cushion_values_to_gcs(
    *,
    json_path: Path | None = None,
    also_full: bool = False,
    dry_run: bool = False,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """
    cushion_values.json を年別で GCS に保存する。
    blob: chuou/data/others/jra_cushion/{YYYY}.json
    """
    base = base_dir or Path.cwd()
    src = json_path or (base / "data" / "jra_baba" / "cushion_values.json")
    if not src.is_file():
        return {"ok": False, "error": f"ファイルがありません: {src}"}

    raw = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return {"ok": False, "error": "トップレベルは配列である必要があります"}

    by_year: dict[int, list] = defaultdict(list)
    for row in raw:
        if not isinstance(row, dict):
            continue
        y = row.get("year")
        if y is None:
            continue
        try:
            yi = int(y)
        except (TypeError, ValueError):
            continue
        by_year[yi].append(row)

    counts = {str(y): len(by_year[y]) for y in sorted(by_year)}
    gcs_prefix = f"{HybridStorage.GCS_OTHERS}/jra_cushion/"
    if dry_run:
        st = HybridStorage(base_dir=str(base))
        return {
            "ok": True,
            "dry_run": True,
            "total_rows": len(raw),
            "years": sorted(by_year.keys()),
            "counts": counts,
            "also_full": also_full,
            "gcs_prefix": gcs_prefix,
            "gcs_enabled": st.gcs_enabled,
        }

    storage = HybridStorage(base_dir=str(base))
    if not storage.gcs_enabled:
        return {"ok": False, "error": "GCS が無効です（GCS_BUCKET 等）"}

    for y in sorted(by_year):
        payload = {
            "records": by_year[y],
            "_meta": {"source": str(src), "year": y, "kind": "cushion_year"},
        }
        storage.save("jra_cushion", str(y), payload)

    if also_full:
        storage.save(
            "jra_cushion",
            "full",
            {"records": raw, "_meta": {"source": str(src), "kind": "cushion_all"}},
        )

    storage.invalidate_blob_cache("jra_cushion", "")
    return {
        "ok": True,
        "years": sorted(by_year.keys()),
        "counts": counts,
        "full_saved": also_full,
        "total_rows": len(raw),
        "gcs_prefix": gcs_prefix,
    }

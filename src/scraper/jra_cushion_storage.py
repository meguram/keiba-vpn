"""
クッション値 JSON を GCS（HybridStorage カテゴリ jra_cushion）へ年別保存する。
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.scraper.storage import HybridStorage


def _key(row: dict) -> tuple:
    return (row.get("date"), row.get("venue_code"))


def _by_key(rows: list) -> dict:
    return {_key(r): r for r in rows if isinstance(r, dict)}


def merge_year_records(existing: list, incoming: list) -> list:
    """既存（GCS）と新しい行を (date, venue_code) で統合する。既存のキーは落とさず、同じキーは新しい方を採る。"""
    merged = _by_key(existing)
    merged.update(_by_key(incoming))
    return [merged[k] for k in sorted(merged, key=lambda k: (str(k[0]), str(k[1])))]


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
    if json_path is not None:
        src = json_path
    else:
        # 取得処理（jra-baba-morning）の出力先が正。古い data/local/jra_baba は旧配置
        cands = [base / "data" / "page_reference" / "cushion" / "cushion_values.json", base / "data" / "local" / "jra_baba" / "cushion_values.json"]
        src = next((c for c in cands if c.is_file()), cands[0])
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

    written, skipped = [], []
    for y in sorted(by_year):
        old = storage.load("jra_cushion", str(y), bypass_cache=True) or {}
        merged = merge_year_records(old.get("records") or [], by_year[y])
        if old and _by_key(old.get("records") or []) == _by_key(merged):
            skipped.append(y)          # GCS と同じ内容なら書かない
            continue
        if old:                        # 上書きする前に旧版をローカルへ退避
            bak = base / "data" / "local" / "backup" / "jra_cushion"
            bak.mkdir(parents=True, exist_ok=True)
            (bak / f"{y}_{int(time.time())}.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
        payload = {
            "records": merged,
            "_meta": {"source": str(src), "year": y, "kind": "cushion_year"},
        }
        storage.save("jra_cushion", str(y), payload)
        written.append(y)

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
        "written_years": written,
        "skipped_unchanged_years": skipped,
        "counts": counts,
        "full_saved": also_full,
        "total_rows": len(raw),
        "gcs_prefix": gcs_prefix,
    }

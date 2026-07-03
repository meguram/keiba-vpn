"""
GCS 保存成功後に、指定カテゴリだけ data/local/mirror/ に JSON コピーするための設定。

data/meta/local_mirror_config.json および環境変数 GCS_LOCAL_MIRROR_CATEGORIES（カンマ区切り）で上書き可能。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_CONFIG_CACHE: dict[str, Any] = {"path": None, "mtime": 0.0, "data": None}
_LOCK: dict[str, Any] = {}


def _config_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "data" / "local" / "meta" / "local_mirror_config.json"


def get_local_mirror_config(base_dir: str | Path) -> dict[str, Any]:
    """
    { "enabled": bool, "categories": [ "race_oikiri", ... ] }
    未設定時は enabled false / categories 空。env GCS_LOCAL_MIRROR_CATEGORIES をマージ。
    """
    base = Path(base_dir)
    p = _config_path(base)
    mtime = 0.0
    try:
        mtime = p.stat().st_mtime
    except OSError:
        pass
    if _CONFIG_CACHE.get("data") is not None and str(_CONFIG_CACHE.get("path")) == str(
        p
    ) and float(_CONFIG_CACHE.get("mtime") or 0) == mtime:
        return _merge_env(dict(_CONFIG_CACHE["data"]))

    data: dict[str, Any] = {"enabled": False, "categories": []}
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {"enabled": False, "categories": []}
    if not isinstance(data.get("categories"), list):
        data["categories"] = []
    _CONFIG_CACHE["path"] = str(p)
    _CONFIG_CACHE["mtime"] = mtime
    _CONFIG_CACHE["data"] = data
    return _merge_env(data)


def _merge_env(data: dict[str, Any]) -> dict[str, Any]:
    out = {**data}
    raw = (os.environ.get("GCS_LOCAL_MIRROR_CATEGORIES") or "").strip()
    if raw:
        extra = [x.strip() for x in raw.split(",") if x.strip()]
        seen: set[str] = set()
        cats: list[str] = []
        for c in (out.get("categories") or []) + extra:
            c = str(c).strip()
            if c and c not in seen:
                seen.add(c)
                cats.append(c)
        out["categories"] = cats
        if extra:
            out["enabled"] = bool(out.get("enabled", True))
    return out


def save_local_mirror_config(
    base_dir: str | Path, *, enabled: bool, categories: list[str]
) -> dict[str, Any]:
    p = _config_path(base_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    cl: list[str] = []
    for c in categories:
        s = str(c).strip()
        if s and s not in seen:
            seen.add(s)
            cl.append(s)
    payload = {"enabled": bool(enabled), "categories": cl, "updated_at": time.time()}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(p)
    _CONFIG_CACHE["data"] = None
    return payload


def mirrorable_storage_categories() -> list[str]:
    """HybridStorage.CATEGORY_MAP から local_only 以外（ミラー可能）。"""
    from src.scraper.storage import HybridStorage

    out: list[str] = []
    for cat, v in HybridStorage.CATEGORY_MAP.items():
        if v == "local_only":
            continue
        out.append(cat)
    return sorted(out)

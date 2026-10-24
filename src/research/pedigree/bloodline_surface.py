"""血統研究: 芝・ダート・障害のサーフェス分離ユーティリティ。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SURFACE_KEYS: tuple[str, ...] = ("turf", "dirt", "jump")
SURFACE_LABELS: dict[str, str] = {
    "turf": "芝",
    "dirt": "ダート",
    "jump": "障害",
}
# netkeiba race_result_slim の surface 表記
SURFACE_RAW: dict[str, tuple[str, ...]] = {
    "turf": ("芝",),
    "dirt": ("ダート", "ダ"),
    "jump": ("障", "障害"),
}


def normalize_surface_key(surface: str | None) -> str:
    """API/UI の surface 引数を turf|dirt|jump に正規化。"""
    if not surface:
        return "turf"
    s = str(surface).strip().lower()
    aliases = {
        "芝": "turf", "turf": "turf", "grass": "turf",
        "ダート": "dirt", "ダ": "dirt", "dirt": "dirt",
        "障害": "jump", "障": "jump", "jump": "jump", "steeple": "jump",
    }
    if s in aliases:
        return aliases[s]
    if s in SURFACE_KEYS:
        return s
    return "turf"


def classify_surface_value(raw: str) -> str | None:
    """出走行の surface 文字列 → turf|dirt|jump。不明は None。"""
    v = (raw or "").strip()
    for key, tokens in SURFACE_RAW.items():
        if v in tokens:
            return key
    return None


def filter_df_by_surface(df: pd.DataFrame, surface_key: str) -> pd.DataFrame:
    if df.empty or "surface" not in df.columns:
        return df.iloc[0:0].copy()
    sk = normalize_surface_key(surface_key)
    tokens = SURFACE_RAW[sk]
    mask = df["surface"].astype(str).str.strip().isin(tokens)
    return df.loc[mask].copy()


def compute_surface_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = {k: 0 for k in SURFACE_KEYS}
    if df.empty or "surface" not in df.columns:
        return counts
    for raw in df["surface"].astype(str):
        sk = classify_surface_value(raw)
        if sk:
            counts[sk] += 1
    return counts


def surface_output_dir(base: Path, surface_key: str) -> Path:
    return base / "by_surface" / normalize_surface_key(surface_key)


def write_surface_meta(
    base: Path,
    counts: dict[str, int],
    *,
    years: list[int] | None = None,
    scope_race_horses: int | None = None,
) -> Path:
    """by_surface/_meta.json を書き出す。"""
    meta_dir = base / "by_surface"
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "surfaces": {
            k: {"label": SURFACE_LABELS[k], "run_count": int(counts.get(k, 0))}
            for k in SURFACE_KEYS
        },
        "total_run_count": int(sum(counts.get(k, 0) for k in SURFACE_KEYS)),
    }
    if years is not None:
        payload["years"] = years
    if scope_race_horses is not None:
        payload["scope_race_horses"] = scope_race_horses
    path = meta_dir / "_meta.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_surface_meta(base: Path) -> dict[str, Any]:
    path = base / "by_surface" / "_meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_count_for_surface(meta: dict[str, Any], surface_key: str) -> int:
    sk = normalize_surface_key(surface_key)
    surfaces = meta.get("surfaces") or {}
    entry = surfaces.get(sk) or {}
    return int(entry.get("run_count") or 0)

"""保存データの出どころ（source）の扱い。公開面に出してよいかの判断材料。

source: own（自前計算）／jra（JRA 公式）／netkeiba／smartrc／unknown。
既存の取得データは取得元が netkeiba（または SmartRC）なので、カテゴリごとの既定を持つ。新しい取得元のデータには ``_meta.source`` を付ける。
JRA 公式などから取る実装は、利用規約の確認が済むまで作らない（docs/operations/public_data_inventory.md）。
"""

from __future__ import annotations

from typing import Any

SOURCES = ("own", "jra", "netkeiba", "smartrc", "unknown")

# カテゴリ → 既定の出どころ。ここに無いカテゴリは unknown（安全側＝一般には出さない）
CATEGORY_DEFAULT_SOURCE: dict[str, str] = {
    "race_shutuba": "netkeiba", "race_result": "netkeiba", "race_result_lap": "netkeiba", "race_result_payoff": "netkeiba",
    "race_odds": "netkeiba", "race_pair_odds": "netkeiba", "race_trifecta_odds": "netkeiba",
    "race_index": "netkeiba", "race_paddock": "netkeiba", "race_barometer": "netkeiba", "race_trainer_comment": "netkeiba",
    "race_shutuba_past": "netkeiba", "race_oikiri": "netkeiba", "horse_result": "netkeiba", "horse_profile": "netkeiba",
    "horse_race_history": "netkeiba",
    "smartrc_race": "smartrc",
    "jra_cushion": "jra", "race_day_schedule": "jra",
    "race_predictions": "own",
}


def source_of(doc: Any, category: str | None = None) -> str:
    """保存データの出どころ。_meta.source があればそれ、無ければカテゴリの既定。"""
    if isinstance(doc, dict):
        meta = doc.get("_meta")
        if isinstance(meta, dict) and meta.get("source") in SOURCES:
            return str(meta["source"])
    return CATEGORY_DEFAULT_SOURCE.get(category or "", "unknown")


def stamp(doc: dict, source: str) -> dict:
    """出どころの印を付ける（元の dict は変えない）。"""
    if source not in SOURCES:
        raise ValueError(f"source は {SOURCES} のどれか: {source}")
    out = dict(doc)
    meta = dict(out.get("_meta") or {})
    meta["source"] = source
    out["_meta"] = meta
    return out


def public_ok(source: str, *, jra_ok: bool = False) -> bool:
    """一般公開に出してよい出どころか。own は常に可。jra は取得元の規約確認（jra_ok）が済んでから。"""
    if source == "own":
        return True
    return source == "jra" and jra_ok

"""一般公開の許可リスト（config/public_fields.json）の読み込みと判定。

mode=private（現状）: 従来の匿名／会員の出し分けのまま。mode=public: 開発者（所有者・私的利用）以外には A 分類（自前計算）だけを出し、
B（事実）・C（取得元固有）・O（市場オッズ由来）は release の許可が true になるまで出さない。
環境変数 KEIBA_PUBLIC_MODE=public|private で上書きできる。仕様: docs/operations/public_data_inventory.md
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(os.environ.get("KEIBA_PUBLIC_FIELDS") or Path(__file__).resolve().parents[2] / "config" / "public_fields.json")
_TTL = 30.0
_lock = threading.Lock()
_cache: tuple[float, dict] | None = None


def config() -> dict[str, Any]:
    global _cache
    now = time.time()
    with _lock:
        if _cache and now - _cache[0] < _TTL:
            return _cache[1]
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {"mode": "public", "release": {}, "predictions": {}, "race_detail": {}, "paths_public_mode": {}}  # 読めないときは安全側（公開モード・何も許可しない）
    with _lock:
        _cache = (now, data)
    return data


def reload() -> None:
    global _cache
    with _lock:
        _cache = None


def is_public_mode() -> bool:
    env = (os.environ.get("KEIBA_PUBLIC_MODE") or "").strip().lower()
    if env in ("public", "1", "true"):
        return True
    if env in ("private", "0", "false"):
        return False
    return str(config().get("mode", "public")).lower() == "public"


def release(name: str) -> bool:
    return bool((config().get("release") or {}).get(name, False))


def section(name: str) -> dict[str, Any]:
    return config().get(name) or {}


def b_allowed() -> bool:
    return release("b_facts_public")


def identifiers_allowed() -> bool:
    return release("minimal_identifiers_public") or release("b_facts_public")


def odds_allowed() -> bool:
    return release("odds_derived_public")


def entry_allowed_fields(role: str) -> set[str]:
    """公開モードで、役割（anonymous|member）が見られる予測の行のフィールド。"""
    p = section("predictions")
    out = set(p.get("entry_A_anonymous", []))
    if role == "member":
        out |= set(p.get("entry_A_member", []))
        if odds_allowed():
            out |= set(p.get("entry_O", []))
    if b_allowed():
        out |= set(p.get("entry_B", []))
    elif identifiers_allowed():
        out |= set(p.get("entry_identifiers", []))
    return out


def top_allowed_fields(role: str) -> set[str]:
    p = section("predictions")
    out = set(p.get("top_always", []))
    if identifiers_allowed():
        out |= set(p.get("top_identifiers", []))
    if role == "member":
        out |= set(p.get("top_A_member", []))
        if odds_allowed():
            out |= set(p.get("top_O", []))
    return out


PAID_ZONE_PATH = re.compile(r"^/api/v1/betting/(optimize|race-values)$")


def _compile(names: list[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in names]


def owner_only_path(path: str) -> bool:
    """公開モードで、開発者（所有者）だけが入れるパスか。"""
    if not is_public_mode():
        return False
    pp = section("paths_public_mode")
    # 有料ゾーン: 推奨馬券と一覧は、会員には『推奨度の段階だけ』の出力（オッズ・期待値の数値なし）で出すので、開発者専用にしない
    if PAID_ZONE_PATH.search(path):
        from src.utils import paid_zone

        if paid_zone.enabled():
            return False
    pats = list(pp.get("owner_only", []))
    if not odds_allowed():
        pats += pp.get("owner_only_unless_odds_derived", [])
    if not b_allowed():
        pats += pp.get("owner_only_unless_b_facts", [])
    return any(r.search(path) for r in _compile(pats))

"""netkeiba の血統ページ（db.netkeiba.com/horse/ped/<id>/）に載っている『○○系』を、種牡馬の馬 ID ごとに貯める小さな台帳。

血統ページを JSON にする処理（parse_blood_table_5gen）は系統名を捨てるので、ここで別に拾う。
ページの父のセル（その馬の父）に『○○系』が付くので、1 ページで 1 頭ぶんの種牡馬の系統が分かる。
置き場: data/local/pedigree_lines/sire_line_map.json（環境変数 KEIBA_SIRE_LINE_MAP で変更可）。
形式: {"updated": "...", "lines": {"<種牡馬の馬ID>": {"name": 種牡馬名, "line": "○○系", "from": 取得に使った馬ID, "at": 取得日}}}
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
JST = timezone(timedelta(hours=9))


def default_path() -> Path:
    return Path(os.environ.get("KEIBA_SIRE_LINE_MAP") or REPO / "data/local/pedigree_lines/sire_line_map.json")


def _doc(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """種牡馬の馬 ID -> {name, line, from, at}。無ければ空。"""
    return _doc(path or default_path()).get("lines") or {}


def is_complete(path: Path | None = None) -> bool:
    """対象の種牡馬を全部取り終えたか（fetch_sire_lines が最後に印を付ける）。印があるまで、画面の『系統』は出さない。"""
    return bool(_doc(path or default_path()).get("complete_at"))


def set_complete(path: Path | None = None) -> None:
    path = path or default_path()
    _save(load(path), path, complete_at=datetime.now(JST).isoformat(timespec="seconds"))


def lines_from_html(html: str) -> dict[str, tuple[str, str]]:
    """血統ページの HTML から『○○系』が付いたセルを {馬ID: (名前, 系統)} で返す。"""
    from src.scraper.sire_line_extractor import extract_sires_from_ped_html

    out: dict[str, tuple[str, str]] = {}
    for a in extract_sires_from_ped_html(html):
        if a.get("horse_id") and a.get("sire_line"):
            out[str(a["horse_id"])] = (str(a["name"]).replace("\xa0", " ").strip(), str(a["sire_line"]).strip())
    return out


def _save(cur: dict[str, dict[str, Any]], path: Path, complete_at: str | None = None) -> None:
    complete_at = complete_at or _doc(path).get("complete_at")       # 完了の印は、台帳を書き直しても残す
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"updated": datetime.now(JST).isoformat(timespec="seconds"), "complete_at": complete_at, "lines": cur}, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def record_html(html: str, source_horse_id: str, path: Path | None = None) -> int:
    """HTML から拾った系統を台帳に足す（既存の種牡馬は新しい取得で上書き）。足した件数を返す。"""
    found = lines_from_html(html)
    if not found:
        return 0
    path = path or default_path()
    cur = load(path)
    today = datetime.now(JST).strftime("%Y-%m-%d")
    for hid, (name, line) in found.items():
        cur[hid] = {"name": name, "line": line, "from": source_horse_id, "at": today}
    _save(cur, path)
    return len(found)


def mark_missing(sire_id: str, source_horse_id: str, path: Path | None = None) -> None:
    """ページに『○○系』の表記が無かった種牡馬を、line=None で残す（何度も取りに行かないため）。"""
    path = path or default_path()
    cur = load(path)
    cur.setdefault(sire_id, {"name": None, "line": None, "from": source_horse_id, "at": datetime.now(JST).strftime("%Y-%m-%d")})
    _save(cur, path)


def labels(path: Path | None = None) -> dict[str, str]:
    """種牡馬の馬 ID -> 『○○系』（表記があったものだけ）。"""
    return {k: v["line"] for k, v in load(path).items() if v.get("line")}

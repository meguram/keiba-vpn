"""netkeiba の血統ページ（db.netkeiba.com/horse/ped/<id>/）に載っている『○○系』を、種牡馬の馬 ID ごとに貯める小さな台帳。

血統ページを JSON にする処理（parse_blood_table_5gen）は系統名を捨てるので、ここで別に拾う。
ページの父のセル（その馬の父）に『○○系』が付くので、1 ページで 1 頭ぶんの種牡馬の系統が分かる。
置き場: data/local/pedigree_lines/sire_line_map.json（環境変数 KEIBA_SIRE_LINE_MAP で変更可）。
形式: {"updated": "...", "lines": {"<種牡馬の馬ID>": {"name": 種牡馬名, "line": "○○系", "from": 取得に使った馬ID, "at": 取得日}}}

並走（ホストと VPS/GCP が同時に取る）の前提: 書き込みは『ロックして読み直し→足す→置き換え』なので、同じ台帳に複数のプロセスが書いても取りこぼさない。
別の台帳（別の機械で取ったもの）は merge() で足し合わせられる（種牡馬 ID が鍵なので順序によらず同じ結果）。差は diff() で見る。
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import random
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
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
    with _locked(path):
        _save(load(path), path, complete_at=datetime.now(JST).isoformat(timespec="seconds"))


def lines_from_html(html: str) -> dict[str, tuple[str, str]]:
    """血統ページの HTML から『○○系』が付いたセルを {馬ID: (名前, 系統)} で返す。"""
    from src.scraper.sire_line_extractor import extract_sires_from_ped_html

    out: dict[str, tuple[str, str]] = {}
    for a in extract_sires_from_ped_html(html):
        if a.get("horse_id") and a.get("sire_line"):
            out[str(a["horse_id"])] = (str(a["name"]).replace("\xa0", " ").strip(), str(a["sire_line"]).strip())
    return out


@contextmanager
def _locked(path: Path):
    """台帳の書き換えを 1 プロセスずつにする（ロックが取れない場所なら、ロックなしで進める）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = open(str(path) + ".lock", "a")
    except OSError:
        yield
        return
    try:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield
    finally:
        fh.close()


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
    today = datetime.now(JST).strftime("%Y-%m-%d")
    with _locked(path):
        cur = load(path)
        for hid, (name, line) in found.items():
            cur[hid] = {"name": name, "line": line, "from": source_horse_id, "at": today}
        _save(cur, path)
    return len(found)


def mark_missing(sire_id: str, source_horse_id: str, path: Path | None = None, reason: str = "no_line", name: str | None = None) -> None:
    """ページを取れたが『○○系』が無かった種牡馬を、line=None と理由つきで残す（何度も取りに行かないため）。

    reason: no_line=父のセルはあるが系統の表記が無い（『系統なし』） / no_father=子のページに父が載っていない / father_mismatch=父が別の馬だった。
    通信の失敗はここに残さない（残すと二度と取り直さなくなる）。
    """
    path = path or default_path()
    with _locked(path):
        cur = load(path)
        cur.setdefault(sire_id, {"name": name, "line": None, "reason": reason, "from": source_horse_id, "at": datetime.now(JST).strftime("%Y-%m-%d")})
        _save(cur, path)


def record_page(html: str, sire_id: str, child_id: str, path: Path | None = None) -> str:
    """子（child_id）の血統ページから sire_id の系統を台帳に残し、結果を返す。

    line=系統が取れた / no_line・no_father・father_mismatch=ページは正常でデータ側に無い（理由つきで台帳に残す） /
    no_table=血統表が無い（メンテ画面など。台帳には残さず、呼び出し側が取り直す）。
    """
    from src.scraper.sire_line_extractor import extract_direct_sire, extract_sires_from_ped_html

    if not extract_sires_from_ped_html(html):
        return "no_table"
    record_html(html, child_id, path)
    if sire_id in labels(path):
        return "line"
    father = extract_direct_sire(html)
    if not father:
        status = "no_father"
    elif str(father.get("horse_id") or "") != sire_id:
        status = "father_mismatch"
    else:
        status = "no_line"
    mark_missing(sire_id, child_id, path, reason=status, name=(father or {}).get("name"))
    return status


def fetch_and_record(fetch, url: str, sire_id: str, child_id: str, path: Path | None = None,
                     tries: int = 3, waits: tuple[float, ...] = (3.0, 10.0), sleep=time.sleep) -> str:
    """血統ページを取って record_page する。通信エラーや血統表の無い応答は、軽く間を置いて最大 tries 回まで取り直す。

    アクセス制限の疑い（HTTP 400/403 の連続など）は取り直さずそのまま送出する（回数を増やすと悪化するため）。
    tries 回とも駄目なら 'failed'（台帳には残さないので、後の取得で再挑戦される）。
    """
    from src.scraper.scrape_access_pause import is_block_suspect_http_400

    for i in range(tries):
        try:
            status = record_page(fetch(url), sire_id, child_id, path)
            if status != "no_table":
                return status
            logger.warning("血統表が無い応答 %s (%d/%d)", child_id, i + 1, tries)
        except Exception as exc:  # noqa: BLE001
            if is_block_suspect_http_400(exc):
                raise
            logger.warning("取得エラー %s: %s (%d/%d)", child_id, exc, i + 1, tries)
        if i + 1 < tries:
            sleep(waits[min(i, len(waits) - 1)] * random.uniform(0.8, 1.2))
    return "failed"


def _newer(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """同じ種牡馬の 2 つの記録のうち採る方。表記ありは『表記なし』に勝つ。どちらも表記ありなら取得日の新しい方（同日は a）。"""
    if bool(a.get("line")) != bool(b.get("line")):
        return a if a.get("line") else b
    return b if str(b.get("at") or "") > str(a.get("at") or "") else a


def merge(other: dict[str, dict[str, Any]], path: Path | None = None) -> int:
    """別の台帳（別の機械で取ったもの）を足し合わせる。台帳が変わった種牡馬の数を返す。完了の印は触らない。"""
    path = path or default_path()
    with _locked(path):
        cur = load(path)
        changed = 0
        for hid, rec in other.items():
            new = _newer(cur[hid], rec) if hid in cur else rec
            if cur.get(hid) != new:
                cur[hid], changed = new, changed + 1
        if changed:
            _save(cur, path)
    return changed


def diff(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """2 つの台帳の差。only_a / only_b は片方にだけある種牡馬、line は両方にあるが系統が違うもの（表記なし同士は同じ）。"""
    return {
        "only_a": sorted(set(a) - set(b)),
        "only_b": sorted(set(b) - set(a)),
        "line": sorted(k for k in set(a) & set(b) if (a[k].get("line") or None) != (b[k].get("line") or None)),
    }


def ancestor_sire_children(rec: dict[str, Any], max_gen: int = 5) -> dict[str, str]:
    """5 世代血統レコードに載る牡の祖先（max_gen 世代まで）-> その子の馬 ID。

    祖先の『○○系』は、その子の血統ページ（父のセル）にしか付かない。子は 1 世代目の祖先ならレコードの馬自身、
    それ以外は (世代-1, 位置>>1) の祖先。性別は、無ければ位置の偶奇（偶数=牡）で決める。
    """
    by_pos: dict[tuple[int, int], str] = {}
    for a in rec.get("ancestors") or []:
        if a.get("horse_id") and a.get("generation") is not None and a.get("position") is not None:
            by_pos[(int(a["generation"]), int(a["position"]))] = str(a["horse_id"])
    out: dict[str, str] = {}
    for a in rec.get("ancestors") or []:
        hid, g, p = a.get("horse_id"), a.get("generation"), a.get("position")
        if not hid or g is None or p is None or int(g) > max_gen:
            continue
        g, p = int(g), int(p)
        if a.get("sex") == "牝" or (a.get("sex") != "牡" and p % 2):      # 性別が空の祖先が大半。位置が偶数なら牡（奇数は牝）
            continue
        child = str(rec.get("horse_id") or "") if g == 1 else by_pos.get((g - 1, p >> 1), "")
        if child:
            out.setdefault(str(hid), child)
    return out


def labels(path: Path | None = None) -> dict[str, str]:
    """種牡馬の馬 ID -> 『○○系』（表記があったものだけ）。"""
    return {k: v["line"] for k, v in load(path).items() if v.get("line")}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="別の機械の台帳との差を見る／足し合わせる")
    ap.add_argument("other", type=Path, help="比べる（足す）台帳の JSON")
    ap.add_argument("--merge", action="store_true", help="差を見るだけでなく、こちらの台帳に足す")
    a = ap.parse_args()
    theirs = load(a.other)
    d = diff(load(), theirs)
    print({k: len(v) for k, v in d.items()}, "手元:", len(load()), "相手:", len(theirs))
    if a.merge:
        print("足した:", merge(theirs))

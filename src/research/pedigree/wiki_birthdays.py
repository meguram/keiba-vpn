"""netkeiba で年しか取れなかった馬の月日を Wikipedia(英) の infobox から 1 頭ずつ探す（再開可能）。

採用の条件: ページ見出しが馬名と一致し、infobox の生年が netkeiba の年と同じ。年が違えば year_mismatch として記録のみ。

    python -m src.research.pedigree.wiki_birthdays --in nk_birth_results.jsonl --out wiki_month_results.jsonl [--follow]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request

UA = "keiba-vpn-research/1.0 (personal research; https://github.com/meguram/keiba-vpn)"
MONTHS = {m: i for i, m in enumerate("January February March April May June July August September October November December".split(), 1)}
API = "https://en.wikipedia.org/w/api.php"

_TEMPLATE = re.compile(r"\{\{\s*(?:birth[ -]date(?: and age)?|bda)\s*\|(?:\s*[a-z]+\s*=\s*[a-z]+\s*\|)*\s*(\d{4})(?:\s*\|\s*(\d{1,2}))?(?:\s*\|\s*(\d{1,2}))?", re.I)
_DMY = re.compile(r"(\d{1,2})\s+([A-Z][a-z]+)\s+(\d{4})")
_MDY = re.compile(r"([A-Z][a-z]+)\s+(\d{1,2}),?\s+(\d{4})")
_MY = re.compile(r"([A-Z][a-z]+)\s+(\d{4})")
_Y = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\(.*?\)", "", s).lower())


def _strip_noise(t: str) -> str:
    """出典・注記・コメント内の日付を拾わないよう取り除く。"""
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*?/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    for _ in range(3):
        t = re.sub(r"\{\{\s*(?:cite|efn|sfn|harv|refn|notelist)[^{}]*\}\}", "", t, flags=re.I)
    return re.sub(r"<[^>]+>|\[\[|\]\]|'''?", "", t)


def parse_date(text: str):
    """文中で最初に現れる日付を (year, month, day) で返す。取れない部分は None。"""
    t = _strip_noise(text)
    cands = []
    for rx, kind in ((_TEMPLATE, "tpl"), (_DMY, "dmy"), (_MDY, "mdy"), (_MY, "my"), (_Y, "y")):
        for m in rx.finditer(t):
            if kind == "tpl":
                v = (int(m[1]), int(m[2]) if m[2] else None, int(m[3]) if m[3] else None)
            elif kind == "dmy":
                if m[2] not in MONTHS:
                    continue
                v = (int(m[3]), MONTHS[m[2]], int(m[1]))
            elif kind == "mdy":
                if m[1] not in MONTHS:
                    continue
                v = (int(m[3]), MONTHS[m[1]], int(m[2]))
            elif kind == "my":
                if m[1] not in MONTHS:
                    continue
                v = (int(m[2]), MONTHS[m[1]], None)
            else:
                v = (int(m[1]), None, None)
            cands.append((m.start(), -len(m[0]), v))
            break
    if not cands:
        return None
    return min(cands)[2]


def foaled_field(wikitext: str):
    m = re.search(r"\|\s*(?:foaled|birth_date|born)\s*=\s*((?:.|\n)*?)(?=\n\s*\|\s*\w+\s*=|\n\}\})", wikitext)
    return parse_date(m.group(1)) if m else None


def api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("wikipedia api failed")


def lookup(name: str, year: int) -> dict:
    d = api({"action": "query", "generator": "search", "gsrsearch": f"{name} racehorse foaled {year}", "gsrlimit": 6,
             "prop": "revisions", "rvprop": "content", "rvslots": "main"})
    pages = sorted(d.get("query", {}).get("pages", []), key=lambda p: p.get("index", 99))
    best = None
    for p in pages:
        title = p["title"]
        if norm(title) != norm(name):
            continue
        text = (p.get("revisions") or [{}])[0].get("slots", {}).get("main", {}).get("content", "")
        if not re.search(r"infobox\s+(?:horse|racehorse)|\|\s*foaled\s*=", text, re.I):
            continue
        fd = foaled_field(text)
        if not fd:
            continue
        y, mo, dd = fd
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        rec = {"title": title, "url": url, "wiki_year": y, "month": mo, "day": dd}
        if y == year:
            return {**rec, "status": "day" if dd else ("month" if mo else "year_only")}
        best = best or {**rec, "status": "year_mismatch"}
    return best or {"status": "not_found"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.2)
    a = ap.parse_args(argv)
    done: set[str] = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["horse_id"] for l in open(a.out, encoding="utf-8") if l.strip()}
    idle = 0
    with open(a.out, "a", encoding="utf-8") as fo:
        while True:
            todo = []
            for l in open(a.inp, encoding="utf-8"):
                if not l.strip():
                    continue
                r = json.loads(l)
                m = re.fullmatch(r"(\d{4})年", r.get("birthday_raw", ""))
                if r.get("status") == "ok" and m and r["horse_id"] not in done:
                    todo.append((r["horse_id"], r.get("name") or r.get("name_list", ""), int(m[1])))
            if not todo:
                idle += 1
                if not a.follow or idle > 40:
                    break
                time.sleep(60)
                continue
            idle = 0
            for hid, name, year in todo:
                if not re.search(r"[A-Za-z]", name):
                    res = {"status": "non_latin_name"}
                else:
                    try:
                        res = lookup(re.sub(r"\s*\(.*?\)\s*$", "", name).strip(), year)
                    except Exception as e:
                        res = {"status": "error", "error": repr(e)[:120]}
                fo.write(json.dumps({"horse_id": hid, "name": name, "nk_year": year, **res}, ensure_ascii=False) + "\n")
                fo.flush()
                done.add(hid)
                time.sleep(a.sleep)
    print("DONE", len(done), flush=True)


if __name__ == "__main__":
    main()

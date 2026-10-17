"""祖先種牡馬の誕生月日を Wikidata と Wikipedia(英/日)の冒頭文から探すサブプラン（直列・低頻度）。

優先順位は要検討ゾーン（zone_review.csv）に出る種牡馬を近い世代から。採用基準:
  * Wikidata の誕生日が日精度で、参照が付いている、または Wikipedia 冒頭文と一致する
  * Wikipedia(英/日)の冒頭文に日付がある（日本語版は競走馬・種牡馬のページと分かる場合だけ）
  * 1 月 1 日と年精度の値は仮置きとして不採用
判定結果は birth_search_log.csv に、採用分は birth_manual.csv にも追記する（--apply のとき）。

    python -m src.research.pedigree.birth_subplans --zone data/meta/dormello/zone_review.csv \
        --ancestors /tmp/ancestor_birthdays.csv --dir data/meta/dormello --limit 20 --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

from src.research.pedigree import wiki_birthdays as wb

UA = "keiba-vpn-research/1.0 (personal research; https://github.com/meguram/keiba-vpn)"
LOG_COLUMNS = ["horse_id", "name", "status", "birth_year", "birth_month", "birth_day", "source_urls", "searched_at", "note"]
MANUAL_COLUMNS = ["horse_id", "name", "birth_year", "birth_month", "birth_day", "source_url", "note"]
_JA_DATE = re.compile(r"(\d{4})年(?:(\d{1,2})月(?:(\d{1,2})日)?)?")
_HORSE_WORDS = re.compile(r"horse|stallion|racehorse|thoroughbred|sire|競走馬|種牡馬")


def split_names(raw: str) -> tuple[str, str]:
    """'ノーザンテーストNorthern Taste(加)' -> ('Northern Taste', 'ノーザンテースト')。"""
    s = re.sub(r"\s*[\(（][^)）]*[\)）]\s*$", "", raw.replace("\xa0", " ")).strip()
    ja = "".join(re.findall(r"[ァ-ヶー・]+", s))
    latin = re.sub(r"[ァ-ヶー・]+", "", s).strip()
    return latin, ja


def decide(wd: dict | None, en_lead, ja_lead, year: int):
    """(status, (y, m, d), evidence) を返す。status は adopted / candidate / none。"""
    leads = [(src, v) for src, v in (("en", en_lead), ("ja", ja_lead)) if v and v[0] == year and v[1]]
    wd_day = None
    if wd and wd["precision"] >= 11 and wd["year"] == year and not (wd["month"] == 1 and wd["day"] == 1):
        wd_day = (wd["year"], wd["month"], wd["day"])
    if wd_day and any(v[2] and v != wd_day for _, v in leads):
        return "candidate", wd_day, "Wikidata と Wikipedia 冒頭文の日付が不一致（要確認）"
    if wd_day and (wd["has_ref"] or any(v == wd_day for _, v in leads)):
        basis = "Wikidata(参照付き)" if wd["has_ref"] else "Wikidata + Wikipedia 冒頭文が一致"
        return "adopted", wd_day, basis
    if leads:
        src, v = leads[0]
        return "adopted", v, f"{'英語版' if src == 'en' else '日本語版'} Wikipedia 冒頭文"
    if wd_day:
        return "candidate", wd_day, "Wikidata のみ（参照なし・冒頭文の裏付けなし）"
    return "none", None, ""


def priority_list(zone_rows: list[dict], ancestors: dict[str, dict], done: set[str], max_gen: int = 3, min_year: int = 1900):
    """zone_review の近い世代の種牡馬のうち、年のみ・min_year 以降生まれ・未検索を影響の大きい順に返す。"""
    weight: dict[str, int] = defaultdict(int)
    for r in zone_rows:
        if r["min_gen"] and int(r["min_gen"]) <= max_gen:
            weight[r["stallion_id"]] += int(r["n_horses"])
    out = []
    for sid, n in weight.items():
        a = ancestors.get(sid)
        if a and a["precision"] == "year" and int(a["birthday"][:4]) >= min_year and sid not in done:
            out.append((sid, a["name"], int(a["birthday"][:4]), n))
    return sorted(out, key=lambda x: -x[3])


def _get(base: str, params: dict) -> dict:
    url = base + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
        return json.load(r)


def wikidata_birth(name: str, year: int) -> dict | None:
    d = _get("https://www.wikidata.org/w/api.php", {"action": "wbsearchentities", "search": name, "language": "en", "type": "item", "limit": 6})
    ids = [c["id"] for c in d.get("search", [])]
    if not ids:
        return None
    ents = _get("https://www.wikidata.org/w/api.php", {"action": "wbgetentities", "ids": "|".join(ids), "props": "claims|descriptions", "languages": "en"})["entities"]
    for i in ids:
        e = ents[i]
        cl = e.get("claims", {}).get("P569")
        desc = (e.get("descriptions", {}).get("en", {}) or {}).get("value", "")
        if not cl or not _HORSE_WORDS.search(desc):
            continue
        v = cl[0]["mainsnak"]["datavalue"]["value"]
        y, mo, dd = int(v["time"][1:5]), int(v["time"][6:8]), int(v["time"][9:11])
        if y == year:
            return {"q": i, "year": y, "month": mo, "day": dd, "precision": v["precision"], "has_ref": bool(cl[0].get("references"))}
    return None


def _lead(lang: str, title: str) -> str:
    d = _get(f"https://{lang}.wikipedia.org/w/api.php", {"action": "query", "titles": title, "prop": "extracts", "exintro": 1, "explaintext": 1, "exchars": 500, "redirects": 1})
    p = d["query"]["pages"][0]
    return "" if p.get("missing") else p.get("extract", "")


def en_lead_date(name: str):
    for title in (f"{name} (horse)", name):
        t = _lead("en", title)
        if t and re.search(r"horse|stallion|sire|thoroughbred|racehorse", t[:300], re.I):
            m = re.search(r"\(([^)]*?)\)", t)
            if m:
                return wb.parse_date(re.split(r"\s[–—-]\s|–|—", m.group(1))[0]), title
    return None, None


def ja_lead_date(ja_name: str):
    for title in (f"{ja_name} (競走馬)", ja_name):
        t = _lead("ja", title)
        if t and re.search(r"競走馬|種牡馬", t[:200]):
            m = re.search(r"[（(]([^）)]*)[）)]", t)
            if m:
                dd = _JA_DATE.search(re.split(r"\s*[-–—―]\s*", m.group(1))[0])
                if dd:
                    return (int(dd[1]), int(dd[2]) if dd[2] else None, int(dd[3]) if dd[3] else None), title
    return None, None


def run(a) -> None:
    d = Path(a.dir)
    log_p, man_p = d / "birth_search_log.csv", d / "birth_manual.csv"
    done = {r["horse_id"] for r in csv.DictReader(open(log_p, encoding="utf-8"))} if log_p.exists() else set()
    done |= {r["horse_id"] for r in csv.DictReader(open(man_p, encoding="utf-8"))} if man_p.exists() else set()
    anc = {r["horse_id"]: r for r in csv.DictReader(open(a.ancestors, encoding="utf-8"))}
    todo = priority_list(list(csv.DictReader(open(a.zone, encoding="utf-8"))), anc, done, a.max_gen)[: a.limit]
    today = date.today().isoformat()
    log_rows, man_rows = [], []
    for sid, raw, year, n in todo:
        latin, ja = split_names(raw)
        wd = en = ja_l = None
        urls, title_en, title_ja = [], None, None
        try:
            if latin:
                wd = wikidata_birth(latin, year)
                time.sleep(a.sleep)
                en, title_en = en_lead_date(latin)
                time.sleep(a.sleep)
            if ja:
                ja_l, title_ja = ja_lead_date(ja)
                time.sleep(a.sleep)
        except Exception as e:
            print(sid, raw, "error", repr(e)[:80])
            continue
        status, v, basis = decide(wd, en, ja_l, year)
        if wd:
            urls.append(f"https://www.wikidata.org/wiki/{wd['q']}")
        if en and title_en:
            urls.append("https://en.wikipedia.org/wiki/" + urllib.parse.quote(title_en.replace(" ", "_")))
        if ja_l and title_ja:
            urls.append("https://ja.wikipedia.org/wiki/" + urllib.parse.quote(title_ja.replace(" ", "_")))
        st = {"adopted": "adopted", "candidate": "candidate", "none": "no_data"}[status]
        y, mo, dd = v if v else (year, "", "")
        log_rows.append([sid, raw, st, y, mo or "", dd or "", " ; ".join(urls), today, basis])
        if status == "adopted":
            man_rows.append([sid, raw, y, mo or "", dd or "", " ; ".join(urls), basis])
        print(f"{sid} {raw} {year} (影響 {n}) -> {st} {v or ''} {basis}", flush=True)
    if a.apply:
        for p, cols, rows in ((log_p, LOG_COLUMNS, log_rows), (man_p, MANUAL_COLUMNS, man_rows)):
            new = not p.exists()
            with open(p, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(cols)
                w.writerows(rows)
        print("追記:", len(log_rows), "ログ /", len(man_rows), "採用")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--ancestors", required=True, help="ancestor_birthdays.csv（precision 列を持つ統合テーブル）")
    ap.add_argument("--dir", required=True, help="birth_search_log.csv / birth_manual.csv のあるディレクトリ")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-gen", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--apply", action="store_true")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()

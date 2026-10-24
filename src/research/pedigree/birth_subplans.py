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
    leads = [(src, v) for src, v in (("en", en_lead), ("ja", ja_lead)) if v and v[0] == year and v[1] and not (v[1] == 1 and v[2] == 1)]
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


def priority_list(zone_rows: list[dict], ancestors: dict[str, dict], done: set[str], max_gen: int = 3, min_year: int = 1900, min_impact: int = 1):
    """zone_review の近い世代の種牡馬のうち、年のみ・min_year 以降生まれ・未検索を影響の大きい順に返す。"""
    weight: dict[str, int] = defaultdict(int)
    for r in zone_rows:
        if r["min_gen"] and int(r["min_gen"]) <= max_gen:
            weight[r["stallion_id"]] += int(r["n_horses"])
    out = []
    for sid, n in weight.items():
        a = ancestors.get(sid)
        if a and a["precision"] == "year" and int(a["birthday"][:4]) >= min_year and sid not in done and n >= min_impact:
            out.append((sid, a["name"], int(a["birthday"][:4]), n))
    return sorted(out, key=lambda x: -x[3])


def all_year_only_list(ancestors: dict[str, dict], done: set[str], min_year: int = 1900):
    """ゾーンに関係なく、年のみ・min_year 以降生まれ・未検索の祖先を表の順（出現の多い順）で返す。"""
    return [(sid, a["name"], int(a["birthday"][:4]), 0) for sid, a in ancestors.items()
            if a["precision"] == "year" and int(a["birthday"][:4]) >= min_year and sid not in done]


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


_FOALED = re.compile(r"\b(?:foaled|born)\s+(?:on\s+)?((?:\d{1,2}\s+[A-Z][a-z]+|[A-Z][a-z]+\s+\d{1,2},?)\s+\d{4})")


def body_foaled_date(text: str):
    """冒頭文の本文にある 'foaled April 20, 1904' / 'born on 20 April 1904' 形式の日付（日まであるものだけ）。"""
    m = _FOALED.search(text[:700])
    if not m:
        return None
    v = wb.parse_date(m.group(1))
    return v if v and v[1] and v[2] else None


def en_lead_date(name: str):
    for title in (f"{name} (horse)", name):
        t = _lead("en", title)
        if t and re.search(r"horse|stallion|sire|thoroughbred|racehorse", t[:300], re.I):
            m = re.search(r"\(([^)]*?)\)", t)
            v = wb.parse_date(re.split(r"\s[–—-]\s|–|—", m.group(1))[0]) if m else None
            if not (v and v[1]):  # 括弧内に月が無ければ、本文の foaled/born を探す
                v = body_foaled_date(t) or v
            return v, title
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


_SPARQL = """SELECT ?h ?en ?ja ?d ?p (COUNT(?ref) AS ?nref) WHERE {
  ?h wdt:P31 wd:Q726 . ?h p:P569 ?st . ?st psv:P569 ?v . ?v wikibase:timePrecision ?p ; wikibase:timeValue ?d . FILTER(?p >= 11)
  OPTIONAL { ?st prov:wasDerivedFrom ?ref }
  OPTIONAL { ?h rdfs:label ?en FILTER(lang(?en)="en") }
  OPTIONAL { ?h rdfs:label ?ja FILTER(lang(?ja)="ja") }
} GROUP BY ?h ?en ?ja ?d ?p"""


def wikidata_bulk_rows() -> list[dict]:
    """Wikidata の「馬」で誕生日が日精度のものを一括で取る（1 回の照会）。"""
    req = urllib.request.Request("https://query.wikidata.org/sparql", data=urllib.parse.urlencode({"query": _SPARQL}).encode(),
                                 headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    return json.load(urllib.request.urlopen(req, timeout=180))["results"]["bindings"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9ぁ-んァ-ヶー一-龥]", "", re.sub(r"\(.*?\)", "", s).lower())


def match_bulk(rows: list[dict], ancestors: dict[str, dict], skip: set[str], min_year: int = 1800):
    """年のみの祖先を (正規化した馬名, 生年) で Wikidata の日付つき馬に一意に突き合わせる。1 月 1 日は仮置きとして除く。"""
    idx: dict[tuple[str, int], list[tuple[str, str, int]]] = defaultdict(list)
    for r in rows:
        d = r["d"]["value"][:10]
        for k in ("en", "ja"):
            if k in r:
                idx[(_norm(r[k]["value"]), int(d[:4]))].append((r["h"]["value"].rsplit("/", 1)[-1], d, int(r["nref"]["value"])))
    out = []
    for hid, a in ancestors.items():
        if a["precision"] != "year" or hid in skip or int(a["birthday"][:4]) < min_year:
            continue
        y = int(a["birthday"][:4])
        latin, ja = split_names(a["name"])
        cands = []
        for nm in (latin, ja):
            if nm:
                cands += idx.get((_norm(nm), y), [])
        cands = list(dict.fromkeys(cands))
        if len(cands) == 1 and not cands[0][1].endswith("-01-01"):
            q, d, nref = cands[0]
            out.append({"horse_id": hid, "name": a["name"], "year": y, "q": q, "date": d, "nref": nref})
    return out


def upsert_log(path: Path, rows: list[list]) -> None:
    """検索ログを horse_id ごとに最新の行へ置き換える（新規は追記）。"""
    existing = list(csv.reader(open(path, encoding="utf-8"))) if path.exists() else [LOG_COLUMNS]
    new = {r[0]: r for r in rows}
    out, seen = [], set()
    for r in existing:
        if r and r[0] in new:
            out.append(new[r[0]])
            seen.add(r[0])
        else:
            out.append(r)
    out += [r for h, r in new.items() if h not in seen]
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(out)


def run_bulk(a) -> None:
    d = Path(a.dir)
    log_p, man_p = d / "birth_search_log.csv", d / "birth_manual.csv"
    adopted = {r["horse_id"] for r in csv.DictReader(open(man_p, encoding="utf-8"))} if man_p.exists() else set()
    anc = {r["horse_id"]: r for r in csv.DictReader(open(a.ancestors, encoding="utf-8"))}
    matches = match_bulk(wikidata_bulk_rows(), anc, adopted, a.min_year)
    today = date.today().isoformat()
    log_rows, man_rows = [], []
    for m in matches:
        y, mo, dd = map(int, m["date"].split("-"))
        url = f"https://www.wikidata.org/wiki/{m['q']}"
        wd = {"q": m["q"], "year": y, "month": mo, "day": dd, "precision": 11, "has_ref": m["nref"] > 0}
        en = ja_l = None
        if not wd["has_ref"]:  # 参照なしは Wikipedia 冒頭文で裏付けを取る
            latin, ja = split_names(m["name"])
            try:
                en, _ = en_lead_date(latin) if latin else (None, None)
                time.sleep(a.sleep)
                ja_l, _ = ja_lead_date(ja) if ja else (None, None)
                time.sleep(a.sleep)
            except Exception:
                pass
        status, v, basis = decide(wd, en, ja_l, m["year"])
        st = {"adopted": "adopted", "candidate": "candidate", "none": "no_data"}[status]
        yy, mm, d2 = v if v else (m["year"], "", "")
        log_rows.append([m["horse_id"], m["name"], st, yy, mm or "", d2 or "", url, today, f"{basis}（Wikidata 一括照会）" if basis else "一括照会で突き合わせたが採用基準を満たさず"])
        if status == "adopted":
            man_rows.append([m["horse_id"], m["name"], yy, mm or "", d2 or "", url, f"{basis}（Wikidata 一括照会）"])
        print(f"{m['horse_id']} {m['name']} {m['year']} -> {st} {v or ''} {basis}", flush=True)
    if a.apply:
        upsert_log(log_p, log_rows)
        new = not man_p.exists()
        with open(man_p, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(MANUAL_COLUMNS)
            w.writerows(man_rows)
        print("反映:", len(log_rows), "ログ /", len(man_rows), "採用")


def run(a) -> None:
    if a.wikidata_bulk:
        return run_bulk(a)
    d = Path(a.dir)
    log_p, man_p = d / "birth_search_log.csv", d / "birth_manual.csv"
    done = {r["horse_id"] for r in csv.DictReader(open(log_p, encoding="utf-8"))} if log_p.exists() else set()
    done |= {r["horse_id"] for r in csv.DictReader(open(man_p, encoding="utf-8"))} if man_p.exists() else set()
    anc = {r["horse_id"]: r for r in csv.DictReader(open(a.ancestors, encoding="utf-8"))}
    if a.all_year_only:
        todo = all_year_only_list(anc, done, a.min_year)[: a.limit]
    else:
        todo = priority_list(list(csv.DictReader(open(a.zone, encoding="utf-8"))), anc, done, a.max_gen, min_impact=a.min_impact)[: a.limit]
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
    ap.add_argument("--zone", help="zone_review.csv（--all-year-only のときは不要）")
    ap.add_argument("--wikidata-bulk", action="store_true", help="Wikidata の日付つきの馬を一括で取り、年のみの祖先と馬名・生年で突き合わせる")
    ap.add_argument("--all-year-only", action="store_true", help="ゾーンに関係なく、年のみの全祖先を表の順に調べる")
    ap.add_argument("--min-year", type=int, default=1900, help="--all-year-only で調べる最小の生年")
    ap.add_argument("--ancestors", required=True, help="ancestor_birthdays.csv（precision 列を持つ統合テーブル）")
    ap.add_argument("--dir", required=True, help="birth_search_log.csv / birth_manual.csv のあるディレクトリ")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max-gen", type=int, default=3)
    ap.add_argument("--min-impact", type=int, default=3, help="ゾーンに現れる馬数の下限。これ未満の種牡馬は調べない（採用率が低く費用対効果が悪いため）")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    if not a.all_year_only and not a.zone and not a.wikidata_bulk:
        ap.error("--zone / --all-year-only / --wikidata-bulk のいずれかが必要です")
    run(a)


if __name__ == "__main__":
    main()

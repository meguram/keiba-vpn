"""JBIS（www.jbis.jp）の馬ページから誕生日を取る。robots.txt の Crawl-delay: 600 に従い 1 リクエスト 10 分、標準ライブラリのみ。

    python jbis_births.py fetch --queue jbis_queue.json --out jbis_results.jsonl [--delay 600] [--limit N]
    python jbis_births.py apply --results jbis_results.jsonl --dir data/meta/dormello   # 採用分を birth_manual.csv / birth_search_log.csv に反映

キューは [[horse_id, 名前, 生年, JBIS ID, Wikidata Q, 影響], ...]。403/429/5xx を受けたら即停止する（回避しない）。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

UA = "keiba-vpn-research/1.0 (personal research; https://github.com/meguram/keiba-vpn)"
MONTHS = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}
_DOB = re.compile(r"Date of Birth \(Age\)\s*([A-Z][a-z]{2})\s+(\d{1,2}),\s*(\d{4})")
_DOB_YEAR = re.compile(r"Date of Birth \(Age\)\s*(\d{4})\b")


def parse_page(html: str) -> dict:
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    title = (re.search(r"<title>(.*?)</title>", html, re.S) or [None, ""])[1].strip()
    raw = (re.search(r"Date of Birth \(Age\).{0,50}", text) or [""])[0]  # 読み取りの確認用に原文を残す
    m = _DOB.search(text)
    if m and m[1] in MONTHS:
        return {"title": title, "year": int(m[3]), "month": MONTHS[m[1]], "day": int(m[2]), "dob_text": raw}
    y = _DOB_YEAR.search(text)
    return {"title": title, "year": int(y[1]) if y else None, "month": None, "day": None, "dob_text": raw}


_ANCHOR = re.compile(r'<a[^>]*href="/horse/(\d{10})/"[^>]*>(.*?)</a>', re.S)


def _norm_name(s: str) -> str:
    s = re.sub(r"[\(（][^)）]*[\)）]\s*$", "", s.replace("\xa0", " ")).strip().lower()
    return re.sub(r"[^a-z0-9ぁ-んァ-ヶー一-龥]", "", s)


def parse_search(html: str) -> list[dict]:
    """検索結果ページの /horse/<ID>/ へのリンクを、表示名と直後の文字列（生年が書かれていれば拾う）つきで返す。"""
    out, seen = [], set()
    ms = list(_ANCHOR.finditer(html))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else m.end() + 300
        hid = m[1]
        label = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m[2])).strip()
        tail = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html[m.end():min(end, m.end() + 300)]))
        if hid in seen or not label:
            continue
        seen.add(hid)
        y = re.search(r"\b(1[89]\d{2}|20\d{2})\b", tail[:120])
        out.append({"id": hid, "label": label, "year_near": int(y[1]) if y else None})
    return out


def select_candidate(cands: list[dict], name: str, year: int) -> tuple[str | None, str]:
    """馬名が一致し、生年が分かるなら一致するものを 1 件だけ選ぶ。(JBIS ID or None, 理由)"""
    want = _norm_name(name)
    same = [c for c in cands if _norm_name(c["label"]) == want]
    if not same:
        return None, "search_none"
    with_year = [c for c in same if c["year_near"] == year]
    if len(with_year) == 1:
        return with_year[0]["id"], "ok"
    if len(with_year) > 1:
        return None, "search_ambiguous"
    unknown = [c for c in same if c["year_near"] is None]
    if len(same) == 1 and unknown:
        return same[0]["id"], "ok_name_only"  # 生年は馬ページで照合する
    return None, "search_ambiguous" if len(same) > 1 else "search_year_mismatch"


def _get(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch(queue_path: Path, out_path: Path, delay: float, limit: int | None, initial_wait: float = 0.0) -> int:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    done = set()
    if out_path.exists():
        done = {json.loads(l)["horse_id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [q for q in queue if q[0] not in done]
    if limit:
        todo = todo[:limit]
    print(f"queue {len(queue)} done {len(done)} todo {len(todo)} delay {delay}s", flush=True)
    if initial_wait and todo:
        print(f"直前のアクセスから間隔を空けるため {initial_wait:.0f} 秒待機", flush=True)
        time.sleep(initial_wait)
    last = [0.0]

    def wait() -> None:
        gap = delay - (time.time() - last[0])
        if gap > 0:
            time.sleep(gap)

    def request(url: str) -> str | int:
        wait()
        try:
            return _get(url)
        except urllib.error.HTTPError as e:
            last[0] = time.time()
            if e.code in (403, 429) or e.code >= 500:
                print(f"{time.strftime('%H:%M:%S')} HTTP {e.code} のため停止（回避しない）: {url}", flush=True)
                return -1
            return e.code
        except Exception as e:  # noqa: BLE001
            print(f"{time.strftime('%H:%M:%S')} 通信エラーのため停止: {e!r}", flush=True)
            return -2
        finally:
            last[0] = time.time()

    for hid, name, year, jbis, q, impact in todo:
        rec = {"horse_id": hid, "name": name, "nk_year": year, "jbis": jbis, "q": q, "impact": impact}
        if not jbis:  # ID 不明: まず検索で探す
            query = urllib.parse.quote(re.sub(r"[\(（][^)）]*[\)）]\s*$", "", name.replace("\xa0", " ")).strip())
            res = request(f"https://www.jbis.jp/navi_search/?keyword={query}")
            if isinstance(res, int):
                if res < 0:
                    return 3 if res == -1 else 4
                rec.update(status=f"search_http_{res}")
                _append(out_path, rec)
                continue
            cands = parse_search(res)
            jbis, why = select_candidate(cands, name, year)
            rec.update(search_candidates=cands[:8], search_result=why, jbis=jbis)
            if not jbis:
                rec["status"] = why
                _append(out_path, rec)
                print(f"{time.strftime('%H:%M:%S')} {hid} {name} {year} -> 検索: {why}（候補 {len(cands)}）", flush=True)
                continue
        url = f"https://www.jbis.jp/horse/{jbis}/"
        res = request(url)
        if isinstance(res, int):
            if res < 0:
                return 3 if res == -1 else 4
            rec.update(url=url, status=f"http_{res}")
        else:
            rec.update(url=url, status="ok", **parse_page(res))
        _append(out_path, rec)
        print(f"{time.strftime('%H:%M:%S')} {hid} {name} {year} -> {rec.get('year')}-{rec.get('month')}-{rec.get('day')} ({rec.get('title', '')[:30]})", flush=True)
    print("DONE", flush=True)
    return 0


def _append(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def decide(rec: dict) -> tuple[str, str]:
    """(status, 理由)。日まで取れて生年が netkeiba と一致し、1 月 1 日でなければ採用。"""
    if rec.get("status") != "ok":
        return "no_data", f"JBIS 取得失敗 {rec.get('status')}"
    if not rec.get("year"):
        return "no_data", "JBIS ページに生年なし"
    if rec["year"] != rec["nk_year"]:
        return "candidate", f"JBIS の生年 {rec['year']} が netkeiba {rec['nk_year']} と不一致（要確認）"
    if not (rec.get("month") and rec.get("day")):
        return "no_data", "JBIS も年のみ"
    if rec["month"] == 1 and rec["day"] == 1:
        return "no_data", "JBIS が 1 月 1 日（仮置きの疑い）"
    return "adopted", "JBIS の Date of Birth"


def apply(results_path: Path, d: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.research.pedigree.birth_subplans import upsert_log

    today = date.today().isoformat()
    recs = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    man_p = d / "birth_manual.csv"
    have = {r["horse_id"] for r in csv.DictReader(open(man_p, encoding="utf-8"))} if man_p.exists() else set()
    log_rows, man_rows = [], []
    for r in recs:
        st, why = decide(r)
        y, mo, dd = (r["year"], r["month"], r["day"]) if st == "adopted" else (r["nk_year"], "", "")
        log_rows.append([r["horse_id"], r["name"], st, y, mo or "", dd or "", r["url"], today, why])
        if st == "adopted" and r["horse_id"] not in have:
            man_rows.append([r["horse_id"], r["name"], y, mo, dd, r["url"], f"JBIS（{r.get('title', '')[:40]}）の Date of Birth"])
    upsert_log(d / "birth_search_log.csv", log_rows)
    with open(man_p, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(man_rows)
    print(f"反映: ログ {len(log_rows)} 件 / 採用 {len(man_rows)} 件")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--queue", required=True, type=Path)
    f.add_argument("--out", required=True, type=Path)
    f.add_argument("--delay", type=float, default=600.0)
    f.add_argument("--limit", type=int)
    f.add_argument("--initial-wait", type=float, default=0.0)
    p = sub.add_parser("apply")
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--dir", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        return fetch(a.queue, a.out, a.delay, a.limit, a.initial_wait)
    apply(a.results, a.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

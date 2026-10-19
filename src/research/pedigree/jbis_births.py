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
    for i, (hid, name, year, jbis, q, impact) in enumerate(todo):
        if i:
            time.sleep(delay)
        url = f"https://www.jbis.jp/horse/{jbis}/"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                html = r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) or e.code >= 500:
                print(f"{time.strftime('%H:%M:%S')} HTTP {e.code} のため停止（回避しない）: {url}", flush=True)
                return 3
            rec = {"horse_id": hid, "name": name, "nk_year": year, "jbis": jbis, "q": q, "impact": impact, "url": url, "status": f"http_{e.code}"}
        except Exception as e:  # noqa: BLE001
            print(f"{time.strftime('%H:%M:%S')} 通信エラーのため停止: {e!r}", flush=True)
            return 4
        else:
            rec = {"horse_id": hid, "name": name, "nk_year": year, "jbis": jbis, "q": q, "impact": impact, "url": url, "status": "ok", **parse_page(html)}
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{time.strftime('%H:%M:%S')} {hid} {name} {year} -> {rec.get('year')}-{rec.get('month')}-{rec.get('day')} ({rec.get('title', '')[:30]})", flush=True)
    print("DONE", flush=True)
    return 0


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

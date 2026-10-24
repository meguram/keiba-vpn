"""SporthorseData の pedigree ページから誕生日を取得する（疑似セッション必須。403/チャレンジ時は即停止）。

    python -m src.research.pedigree.sporthorse_births build-queue \\
        --ancestors /tmp/ancestor_birthdays_for_subplan.csv --zone data/meta/dormello/zone_review.csv \\
        --nk /tmp/nk_birth_results_local.jsonl --dir data/meta/dormello --out /tmp/sporthorse_queue.json

    python -m src.research.pedigree.sporthorse_session probe

    python -m src.research.pedigree.sporthorse_births fetch \\
        --queue /tmp/sporthorse_queue.json --out data/meta/dormello/sporthorse_results.jsonl \\
        --delay 45 --limit 20

    python -m src.research.pedigree.sporthorse_births apply \\
        --results data/meta/dormello/sporthorse_results.jsonl --dir data/meta/dormello

キュー 1 行: [horse_id, name, nk_year, preferred_url, impact]。URL が空なら slug 候補を順に試す。
``sporthorse_urls.csv`` の上書きは ``--dir`` から自動読み込み。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

from src.research.pedigree import manual_checklist as mc
from src.research.pedigree.sporthorse_session import (
    DEFAULT_SESSION_PATH,
    SessionState,
    fetch_html,
    is_challenge,
    load_session,
    session_path,
)
MON = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_DOB_FULL = re.compile(r"(?:Date of Birth|born)[:\s]*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})", re.I)
_DOB_YEAR = re.compile(r"(?:Date of Birth|born)[:\s]*([0-9]{4})\b", re.I)


def parse_page(html: str) -> dict:
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    title = (re.search(r"<title>(.*?)</title>", html, re.S) or [None, ""])[1].strip()
    raw = ""
    m = _DOB_FULL.search(text)
    if m:
        parts = m[1].split("-")
        if len(parts) == 3 and parts[1].lower() in MON:
            y, mo, d = int(parts[2]), MON[parts[1].lower()], int(parts[0])
            raw = m.group(0)
            return {"title": title, "year": y, "month": mo, "day": d, "dob_text": raw}
    y = _DOB_YEAR.search(text)
    raw = y.group(0) if y else ""
    return {"title": title, "year": int(y[1]) if y else None, "month": None, "day": None, "dob_text": raw}


def _norm_title_name(title: str) -> str:
    t = re.sub(r"\s*\|\s*SporthorseData.*$", "", title, flags=re.I)
    t = re.sub(r"\s*-\s*Thoroughbred.*$", "", t, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", t.lower())


def title_matches(name: str, title: str) -> bool:
    want = re.sub(r"[^a-z0-9]", "", mc.latin_name({"name": name, "name_en": name}, name).lower())
    if not want:
        want = re.sub(r"[^a-z0-9]", "", name.lower())
    got = _norm_title_name(title)
    return bool(want) and (want in got or got in want)


def urls_for_row(
    horse_id: str,
    name: str,
    nk_year: int,
    preferred: str,
    nk_rec: dict,
    base_url: str,
    overrides: dict[str, str],
) -> list[str]:
    if overrides.get(horse_id):
        u = overrides[horse_id]
        return [u] + [c for c in _slug_urls(name, nk_rec, base_url) if c != u]
    if preferred and preferred.startswith("https://sporthorse-data.com/"):
        return [preferred] + [c for c in _slug_urls(name, nk_rec, base_url) if c != preferred]
    return _slug_urls(name, nk_rec, base_url)


def _slug_urls(name: str, nk_rec: dict, base_url: str) -> list[str]:
    en = mc.latin_name(nk_rec, name)
    key = en if mc.slug(en) else name
    return [f"{base_url.rstrip('/')}/{c}" for c in mc.slug_candidates(key)]


def load_url_overrides(d: Path) -> dict[str, str]:
    p = d / "sporthorse_urls.csv"
    if not p.exists():
        return {}
    rows = list(csv.reader(open(p, encoding="utf-8")))
    return {r[0]: r[1] for r in rows[1:] if len(r) > 1 and r[1]}


def load_nk_map(nk: Path | None) -> dict[str, dict]:
    nk_map: dict[str, dict] = {}
    if nk and nk.exists():
        for line in open(nk, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                nk_map[r["horse_id"]] = r
    return nk_map


def make_queue_list(
    ancestors: Path,
    zone: Path,
    nk: Path | None,
    d: Path,
    limit: int,
    base_url: str,
    sires: Path | None,
) -> list[list]:
    anc = {r["horse_id"]: r for r in csv.DictReader(open(ancestors, encoding="utf-8"))}
    zone_rows = list(csv.DictReader(open(zone, encoding="utf-8")))
    nk_map = load_nk_map(nk)
    manual = {r["horse_id"] for r in csv.DictReader(open(d / "birth_manual.csv", encoding="utf-8"))} if (d / "birth_manual.csv").exists() else set()
    sire_map = json.loads(sires.read_text(encoding="utf-8")) if sires and sires.exists() else {}
    rows = mc.build_rows(anc, zone_rows, nk_map, manual, limit, base_url, sire_map)
    overrides = load_url_overrides(d)
    return [[r["horse_id"], r["name"], r["year"], overrides.get(r["horse_id"], r["url"]), r["impact"]] for r in rows]


def results_done_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    return {json.loads(l)["horse_id"] for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()}


def pending_summary(queue: list[list], results_path: Path) -> dict:
    done = results_done_ids(results_path)
    pending = [q for q in queue if q[0] not in done]
    return {"queue_total": len(queue), "fetched": len(done), "pending": len(pending)}


def status_report(
    ancestors: Path,
    zone: Path,
    nk: Path | None,
    d: Path,
    queue_path: Path | None,
    results_path: Path,
    limit: int,
    base_url: str,
    sires: Path | None,
    session_file: Path,
) -> dict:
    queue = json.loads(queue_path.read_text(encoding="utf-8")) if queue_path and queue_path.exists() else make_queue_list(
        ancestors, zone, nk, d, limit, base_url, sires
    )
    summ = pending_summary(queue, results_path)
    sess = {"present": session_file.exists()}
    if sess["present"]:
        try:
            st = load_session(session_file)
            sess.update(valid=True, cookies=len(st.cookies), saved_at=st.saved_at)
        except Exception as e:  # noqa: BLE001
            sess.update(valid=False, error=str(e)[:120])
    disabled = (d / "sporthorse_cron_disabled").exists()
    return {"disabled": disabled, "session": sess, **summ}


def print_status(rep: dict) -> None:
    print(
        f"pending={rep['pending']} queue_total={rep['queue_total']} fetched={rep['fetched']} "
        f"session={'ok' if rep['session'].get('valid') else 'missing' if not rep['session'].get('present') else 'bad'} "
        f"cron_disabled={rep['disabled']}",
        flush=True,
    )


def build_queue(
    ancestors: Path,
    zone: Path,
    nk: Path | None,
    d: Path,
    out: Path,
    limit: int,
    base_url: str,
    sires: Path | None,
) -> int:
    queue = make_queue_list(ancestors, zone, nk, d, limit, base_url, sires)
    out.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"queue {len(queue)} -> {out}", flush=True)
    return 0


def run_nightly(
    ancestors: Path,
    zone: Path,
    nk: Path | None,
    d: Path,
    queue_path: Path,
    results_path: Path,
    sess_path: Path,
    delay: float,
    max_seconds: float,
    limit: int,
    base_url: str,
    sires: Path | None,
) -> int:
    """セッション検証 → キュー再生成 → 期限まで fetch / apply を繰り返す。"""
    if (d / "sporthorse_cron_disabled").exists():
        print("SKIP: sporthorse_cron_disabled があるため実行しません", flush=True)
        return 0
    if not sess_path.exists():
        print(f"SKIP: セッション未登録 ({sess_path})。ブラウザでクッキーを保存してください", flush=True)
        return 0
    from src.research.pedigree.sporthorse_session import probe

    st = load_session(sess_path)
    if probe(st, sess_path) != 0:
        return 3
    build_queue(ancestors, zone, nk, d, queue_path, limit, base_url, sires)
    rep = status_report(ancestors, zone, nk, d, queue_path, results_path, limit, base_url, sires, sess_path)
    print_status(rep)
    if rep["pending"] == 0:
        print("DONE: 未処理なし", flush=True)
        return 0
    deadline = time.time() + max_seconds
    batches = 0
    while time.time() < deadline:
        rep = status_report(ancestors, zone, nk, d, queue_path, results_path, limit, base_url, sires, sess_path)
        if rep["pending"] == 0:
            break
        remain_s = deadline - time.time()
        batch_limit = min(rep["pending"], max(1, int(remain_s / max(delay, 1.0)) - 1))
        rc = fetch(queue_path, results_path, delay, batch_limit, st, d, base_url, nk)
        apply(results_path, d)
        batches += 1
        print_status(rep)
        if rc != 0:
            print(f"fetch が {rc} で停止（この実行分は apply 済み）", flush=True)
            return rc
    print(f"DONE: batches={batches}", flush=True)
    return 0


def _append(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def fetch(
    queue_path: Path,
    out_path: Path,
    delay: float,
    limit: int | None,
    sess: SessionState,
    d: Path,
    base_url: str,
    nk: Path | None,
    initial_wait: float = 0.0,
) -> int:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    done = results_done_ids(out_path)
    todo = [q for q in queue if q[0] not in done]
    if limit:
        todo = todo[:limit]
    overrides = load_url_overrides(d)
    nk_map = load_nk_map(nk)
    print(f"queue {len(queue)} done {len(done)} todo {len(todo)} delay {delay}s", flush=True)
    if initial_wait and todo:
        print(f"直前のアクセスから間隔を空けるため {initial_wait:.0f} 秒待機", flush=True)
        time.sleep(initial_wait)
    last = [0.0]

    def wait() -> None:
        gap = delay - (time.time() - last[0])
        if gap > 0:
            time.sleep(gap)

    for hid, name, year, preferred, impact in todo:
        rec = {"horse_id": hid, "name": name, "nk_year": year, "impact": impact}
        urls = urls_for_row(hid, name, year, preferred or "", nk_map.get(hid, {}), base_url, overrides)
        if not urls:
            rec.update(status="no_url")
            _append(out_path, rec)
            print(f"{time.strftime('%H:%M:%S')} {hid} {name} -> no_url", flush=True)
            continue
        parsed = None
        used_url = None
        for url in urls[:14]:
            wait()
            code, html, hdrs = fetch_html(url, sess)
            last[0] = time.time()
            if code in (403, 429) or (code >= 500 and code != 404):
                print(f"{time.strftime('%H:%M:%S')} HTTP {code} のため停止（回避しない）: {url}", flush=True)
                return 3
            if code == 404:
                continue
            if is_challenge(html, hdrs):
                print(f"{time.strftime('%H:%M:%S')} Cloudflare チャレンジのため停止: {url}", flush=True)
                return 3
            if code != 200:
                continue
            p = parse_page(html)
            if not title_matches(name, p.get("title", "")):
                continue
            parsed, used_url = p, url
            break
        if not parsed:
            rec.update(status="not_found", tried=urls[:14])
            _append(out_path, rec)
            print(f"{time.strftime('%H:%M:%S')} {hid} {name} {year} -> not_found", flush=True)
            continue
        rec.update(status="ok", url=used_url, **parsed)
        _append(out_path, rec)
        print(
            f"{time.strftime('%H:%M:%S')} {hid} {name} {year} -> {rec.get('year')}-{rec.get('month')}-{rec.get('day')} "
            f"({rec.get('title', '')[:36]})",
            flush=True,
        )
    print("DONE", flush=True)
    return 0


def decide(rec: dict) -> tuple[str, str]:
    if rec.get("status") != "ok":
        return "no_data", f"Sporthorse 取得失敗 {rec.get('status')}"
    if not rec.get("year"):
        return "no_data", "ページに生年なし"
    if rec["year"] != rec["nk_year"]:
        return "candidate", f"Sporthorse の生年 {rec['year']} が netkeiba {rec['nk_year']} と不一致（要確認）"
    if not (rec.get("month") and rec.get("day")):
        return "no_data", "Sporthorse も年のみ"
    if rec["month"] == 1 and rec["day"] == 1:
        return "no_data", "1 月 1 日（仮置きの疑い）"
    return "adopted", "SporthorseData の Date of Birth"


def apply(results_path: Path, d: Path) -> None:
    from src.research.pedigree.birth_subplans import upsert_log

    today = date.today().isoformat()
    recs = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    man_p = d / "birth_manual.csv"
    have = {r["horse_id"] for r in csv.DictReader(open(man_p, encoding="utf-8"))} if man_p.exists() else set()
    log_rows, man_rows = [], []
    urls_p = d / "sporthorse_urls.csv"
    url_rows = list(csv.reader(open(urls_p, encoding="utf-8"))) if urls_p.exists() else [["horse_id", "url"]]
    url_map = {r[0]: r[1] for r in url_rows[1:] if len(r) > 1}
    for r in recs:
        st, why = decide(r)
        y, mo, dd = (r["year"], r["month"], r["day"]) if st == "adopted" else (r["nk_year"], "", "")
        src = r.get("url") or ""
        log_rows.append([r["horse_id"], r["name"], st, y, mo or "", dd or "", src, today, why])
        if st == "adopted" and r["horse_id"] not in have:
            man_rows.append([r["horse_id"], r["name"], y, mo, dd, src, f"Sporthorse（{r.get('title', '')[:40]}）の Date of Birth"])
        if r.get("status") == "ok" and src:
            url_map[r["horse_id"]] = src
    upsert_log(d / "birth_search_log.csv", log_rows)
    with open(man_p, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(man_rows)
    if url_map:
        urls_p.parent.mkdir(parents=True, exist_ok=True)
        with open(urls_p, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows([["horse_id", "url"], *[[h, u] for h, u in sorted(url_map.items())]])
    print(f"反映: ログ {len(log_rows)} 件 / 採用 {len(man_rows)} 件", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, help=f"セッション JSON（既定: {DEFAULT_SESSION_PATH}）")
    ap.add_argument("--base-url", default="https://sporthorse-data.com/pedigree")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-queue")
    b.add_argument("--ancestors", required=True, type=Path)
    b.add_argument("--zone", required=True, type=Path)
    b.add_argument("--nk", type=Path)
    b.add_argument("--dir", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--limit", type=int, default=3000)
    b.add_argument("--sires", type=Path)
    f = sub.add_parser("fetch")
    f.add_argument("--queue", required=True, type=Path)
    f.add_argument("--out", required=True, type=Path)
    f.add_argument("--dir", required=True, type=Path, help="sporthorse_urls.csv を読むディレクトリ")
    f.add_argument("--nk", type=Path)
    f.add_argument("--delay", type=float, default=45.0, help="リクエスト間隔（秒）。既定 45")
    f.add_argument("--limit", type=int)
    f.add_argument("--initial-wait", type=float, default=0.0)
    p = sub.add_parser("apply")
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--dir", required=True, type=Path)
    st = sub.add_parser("status", help="未処理件数とセッション状態（実アクセスなし）")
    st.add_argument("--ancestors", required=True, type=Path)
    st.add_argument("--zone", required=True, type=Path)
    st.add_argument("--nk", type=Path)
    st.add_argument("--dir", required=True, type=Path)
    st.add_argument("--queue", type=Path, help="既存キューがあればそれを使う（無ければ都度生成）")
    st.add_argument("--results", type=Path, help="既定: <dir>/sporthorse_results.jsonl")
    st.add_argument("--limit", type=int, default=3000)
    st.add_argument("--sires", type=Path)
    st.add_argument("--json", action="store_true")
    n = sub.add_parser("run-nightly", help="夜間バッチ（probe → build-queue → fetch/apply ループ）")
    n.add_argument("--ancestors", required=True, type=Path)
    n.add_argument("--zone", required=True, type=Path)
    n.add_argument("--nk", type=Path)
    n.add_argument("--dir", required=True, type=Path)
    n.add_argument("--queue", type=Path, help="既定: <dir>/sporthorse_queue.json")
    n.add_argument("--results", type=Path, help="既定: <dir>/sporthorse_results.jsonl")
    n.add_argument("--delay", type=float, default=45.0)
    n.add_argument("--max-seconds", type=float, default=7 * 3600.0, help="1 回の cron 実行の上限秒（既定 7 時間）")
    n.add_argument("--limit", type=int, default=3000, help="キューに載せる最大頭数")
    n.add_argument("--sires", type=Path)
    a = ap.parse_args(argv)
    sess_p = session_path(a.session)
    if a.cmd == "build-queue":
        return build_queue(a.ancestors, a.zone, a.nk, a.dir, a.out, a.limit, a.base_url, a.sires)
    if a.cmd == "apply":
        apply(a.results, a.dir)
        return 0
    if a.cmd == "status":
        results_p = a.results or (a.dir / "sporthorse_results.jsonl")
        rep = status_report(a.ancestors, a.zone, a.nk, a.dir, a.queue, results_p, a.limit, a.base_url, a.sires, sess_p)
        if a.json:
            print(json.dumps(rep, ensure_ascii=False))
        else:
            print_status(rep)
        return 0
    if a.cmd == "run-nightly":
        queue_p = a.queue or (a.dir / "sporthorse_queue.json")
        results_p = a.results or (a.dir / "sporthorse_results.jsonl")
        return run_nightly(
            a.ancestors, a.zone, a.nk, a.dir, queue_p, results_p, sess_p,
            a.delay, a.max_seconds, a.limit, a.base_url, a.sires,
        )
    sess = load_session(sess_p)
    return fetch(a.queue, a.out, a.delay, a.limit, sess, a.dir, a.base_url, a.nk, a.initial_wait)


if __name__ == "__main__":
    sys.exit(main())

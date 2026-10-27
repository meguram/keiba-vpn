"""全 GET エンドポイントのレイテンシを、ランダム（モンキー）なパラメータで測る。

測るもの（エンドポイントごと）:
  - プロセス初回 : 新しいプロセスでの、そのエンドポイントの最初の 1 回（モジュール読み込み・キャッシュが空）
  - 新規キー初回 : 同じプロセスで、まだ誰も見ていないレース/馬/開催日を最初に開いたとき
  - 2 回目以降   : 同じキーをもう一度開いたとき（誰かが先に開いた後・事前生成後に当たる）

モード:
  pools   サンプル（開催日・レース・馬・騎手・調教師）を作って JSON に保存する（測定プロセスとは別プロセスで作り、測定を温めない）
  inproc  アプリを同じプロセスで呼ぶ（FastAPI TestClient / Flask test_client。ライフサイクルのワーカーは起動しない）。
          netkeiba 等への通信は遮断して回数を記録する（取得が走るエンドポイントの検出）
  http    動いているサーバーに HTTP で当てる（FastAPI / Flask / Next.js）。inproc で取得が走った印のあるものは --skip-file で外す

例（stg ホストで）:
  python -m src.scripts.ops.latency_monkey pools --out /tmp/lm/pools.json
  python -m src.scripts.ops.latency_monkey inproc --pools /tmp/lm/pools.json --out /tmp/lm/inproc
  python -m src.scripts.ops.latency_monkey http --pools /tmp/lm/pools.json --out /tmp/lm/http --skip-file /tmp/lm/inproc/net_risk.json

副作用の恐れがある GET（enqueue・logout・管理系の再構築など）は EXCLUDE で対象外にし、レポートに理由つきで載せる。
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import socket
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Any, Callable

# 副作用（取得・キュー投入・ログアウト・再構築・外部への確認通信）の恐れがある GET は測らない
EXCLUDE = re.compile(
    r"enqueue|logout|^/docs|^/redoc|^/openapi\.json|rebuild|cushion/live/check|^/api/admin/|^/api/v1/admin/|"
    r"^/api/fetch-future-calendar|^/static/|^/monitor|^/server-logs|^/cron-jobs|^/data-viewer|^/scrape"
)
NET_BLOCK = ("netkeiba", "smartrc", "jra.go.jp", "jra.jp", "keibalab", "yahoo")
SLOW_FIRST_MS = 20_000          # 初回がこれより遅いエンドポイントは、繰り返しを 1 回に減らす
LOAD_PER_CORE = 0.5
STR_PARAMS = {"q": "ディープ", "sort": "", "side": "sire"}


@dataclass
class Endpoint:
    kind: str                   # F=FastAPI / K=Flask / N=Next.js ページ
    path: str                   # テンプレート（{x} / <x>）
    name: str = ""
    required: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)
    excluded: str = ""          # 空でなければ対象外の理由

    @property
    def eid(self) -> str:
        return f"{self.kind} {self.path}"


# ───────── エンドポイントの列挙 ─────────

def _norm(path: str) -> str:
    return re.sub(r"<(?:[a-z]+:)?([A-Za-z_]+)>", r"{\1}", path)


def _flask_query_names(fn: Callable[..., Any]) -> list[str]:
    import inspect

    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return []
    return sorted(set(re.findall(r"args\.get\(\s*[\"']([A-Za-z_]+)[\"']", src)))


def build_catalog(include_excluded: bool = True) -> list[Endpoint]:
    from src.api.app import app

    eps: list[Endpoint] = []
    for r in app.routes:
        ms = (getattr(r, "methods", None) or set()) - {"HEAD", "OPTIONS"}
        if "GET" not in ms:
            continue
        d = getattr(r, "dependant", None)
        req = [q.name for q in d.query_params if q.field_info.is_required()] if d else []
        opt = [q.name for q in d.query_params if not q.field_info.is_required()] if d else []
        fn = getattr(r, "endpoint", None)
        eps.append(Endpoint("F", r.path, getattr(fn, "__name__", ""), req, opt))
    from src.api.flask_app import create_app

    fa = create_app()
    for rule in fa.url_map.iter_rules():
        if "GET" not in rule.methods:
            continue
        fn = fa.view_functions[rule.endpoint]
        eps.append(Endpoint("K", _norm(str(rule.rule)), fn.__name__, [], _flask_query_names(fn)))
    for e in eps:
        if EXCLUDE.search(e.path):
            e.excluded = "副作用または対象外（取得・キュー投入・管理・文書・静的）"
    eps.sort(key=lambda e: (e.kind, e.path))
    return eps if include_excluded else [e for e in eps if not e.excluded]


def next_pages() -> list[Endpoint]:
    """frontend/app 配下のページ（動的セグメントは [id] を {id} にする）。"""
    root = Path(__file__).resolve().parents[3] / "frontend" / "app"
    out = []
    for p in sorted(root.rglob("page.tsx")):
        rel = p.parent.relative_to(root).as_posix()
        if rel.startswith(("admin", "(")) or "/_" in rel:
            continue
        path = "/" + re.sub(r"\[([^\]]+)\]", r"{\1}", "" if rel == "." else rel)
        out.append(Endpoint("N", path, "page"))
    return out


# ───────── サンプル（開催日・レース・馬） ─────────

def build_pools(client_get: Callable[[str, str], tuple[int, Any, float]], seed: int, hot_days: int = 14) -> dict[str, Any]:
    rng = random.Random(seed)
    status, doc, _ = client_get("F", "/api/scrape-dates")
    raw = (doc or {}).get("dates", []) if isinstance(doc, dict) else (doc or [])
    dates = sorted({x if isinstance(x, str) else str(x.get("date")) for x in raw if x})
    today = _date.today()
    lo, hi = (today - timedelta(days=hot_days)).strftime("%Y%m%d"), (today + timedelta(days=7)).strftime("%Y%m%d")
    hot = [d for d in dates if lo <= d <= hi]
    cold = [d for d in dates if d < lo]
    picks = [(d, True) for d in rng.sample(hot, min(3, len(hot)))] + [(d, False) for d in rng.sample(cold, min(8, len(cold)))]
    items = []
    for d, is_hot in picks:
        _, rl, _ = client_get("F", f"/api/race-list/{d}")
        races = [r["race_id"] for r in ((rl or {}).get("races") or []) if isinstance(r, dict) and r.get("race_id")]
        for rid in rng.sample(races, min(3, len(races))):
            _, det, _ = client_get("F", f"/api/race/{rid}")
            ents = ((det or {}).get("race_shutuba") or {}).get("entries") or [] if isinstance(det, dict) else []
            ents = [e for e in ents if isinstance(e, dict)]
            items.append({
                "date": d, "hot": is_hot, "race_id": rid,
                "horse_ids": [str(e["horse_id"]) for e in ents if e.get("horse_id")][:6],
                "jockey_ids": [str(e["jockey_id"]) for e in ents if e.get("jockey_id")][:3],
                "trainer_ids": [str(e["trainer_id"]) for e in ents if e.get("trainer_id")][:3],
            })
    return {"made_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed, "items": items}


def param_sets(ep: Endpoint, pools: dict[str, Any], rng: random.Random, n_sets: int) -> list[tuple[str, dict[str, str]]]:
    """(具体的な URL パス, クエリ) の組を n_sets 個。パラメータの無いエンドポイントは 1 組だけ。"""
    need_path = re.findall(r"\{([A-Za-z_]+)\}", ep.path)
    names = set(need_path) | set(ep.required) | {n for n in ep.optional if n in ("date", "race_id", "horse_id")}
    if not names:
        return [(ep.path, {})]
    items = pools.get("items") or []
    if not items:
        return [(ep.path, {})]
    chosen = rng.sample(items, min(n_sets, len(items)))
    out = []
    for it in chosen:
        vals = {
            "race_id": it["race_id"], "rid": it["race_id"], "date": it["date"], "race_date": it["date"], "target_date": it["date"],
            "horse_id": rng.choice(it["horse_ids"] or [""]), "stallion_id": rng.choice(it["horse_ids"] or [""]),
            "sire_id": rng.choice(it["horse_ids"] or [""]), "person_id": rng.choice(it["jockey_ids"] or [""]),
            "ptype": "jockey", "category": "race_shutuba", "key": it["race_id"], "analysis_type": "sire",
        }
        vals.update(STR_PARAMS)
        path = ep.path
        vals["id"] = rng.choice(it["horse_ids"] or [""]) if ep.path.startswith("/horse") else it["race_id"]
        for n in need_path:
            path = path.replace("{" + n + "}", vals.get(n, "x"))
        q = {n: vals[n] for n in names - set(need_path) if vals.get(n)}
        out.append((path, q))
    return out


# ───────── 通信の遮断（inproc） ─────────

class NetGuard:
    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._orig = socket.getaddrinfo

    def __enter__(self) -> "NetGuard":
        guard = self

        def blocked(host, *a, **k):
            if isinstance(host, str) and any(b in host for b in NET_BLOCK):
                guard.attempts.append(host)
                raise OSError(f"latency_monkey: 外部への通信を遮断しました ({host})")
            return guard._orig(host, *a, **k)

        socket.getaddrinfo = blocked
        return self

    def __exit__(self, *exc: object) -> None:
        socket.getaddrinfo = self._orig


# ───────── クライアント ─────────

def _token() -> str:
    from src.api.auth import ROLE_DEV, _make_token

    return _make_token(role=ROLE_DEV)


def make_inproc_clients() -> Callable[[str, str, dict[str, str] | None, float], tuple[int, bytes, float]]:
    from fastapi.testclient import TestClient

    from src.api.app import app
    from src.api.auth import COOKIE_NAME
    from src.api.flask_app import create_app

    tok = _token()
    fc = TestClient(app, raise_server_exceptions=False)
    fc.cookies.set(COOKIE_NAME, tok)
    kc = create_app().test_client()
    try:
        kc.set_cookie(COOKIE_NAME, tok)
    except TypeError:
        kc.set_cookie("localhost", COOKIE_NAME, tok)

    def get(kind: str, path: str, query: dict[str, str] | None = None, timeout: float = 90.0) -> tuple[int, bytes, float]:
        box: dict[str, Any] = {}

        def run() -> None:
            t0 = time.perf_counter()
            try:
                if kind == "K":
                    r = kc.get(path, query_string=query or {})
                    box["r"] = (r.status_code, r.get_data())
                else:
                    r = fc.get(path, params=query or {})
                    box["r"] = (r.status_code, r.content)
            except Exception as exc:  # noqa: BLE001
                box["r"] = (599, type(exc).__name__.encode())
            box["ms"] = (time.perf_counter() - t0) * 1000

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            return 598, b"timeout", timeout * 1000
        return box["r"][0], box["r"][1], box["ms"]

    return get


def make_http_client(bases: dict[str, str]) -> Callable[[str, str, dict[str, str] | None, float], tuple[int, bytes, float]]:
    from urllib.parse import urlencode

    from src.api.auth import COOKIE_NAME

    cookie = f"{COOKIE_NAME}={_token()}"

    def get(kind: str, path: str, query: dict[str, str] | None = None, timeout: float = 90.0) -> tuple[int, bytes, float]:
        url = bases[kind].rstrip("/") + path + (("?" + urlencode(query)) if query else "")
        req = urllib.request.Request(url, headers={"Cookie": cookie, "User-Agent": "keiba-latency-monkey/1", "Accept": "*/*"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                return int(r.status), body, (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            return int(e.code), e.read(), (time.perf_counter() - t0) * 1000
        except Exception as exc:  # noqa: BLE001
            return 599, type(exc).__name__.encode(), (time.perf_counter() - t0) * 1000

    return get


# ───────── 測定 ─────────

def wait_for_load() -> float:
    cores = os.cpu_count() or 1
    for _ in range(60):
        la = os.getloadavg()[0]
        if la <= cores * LOAD_PER_CORE:
            return la
        time.sleep(1)
    return os.getloadavg()[0]


def measure(eps: list[Endpoint], pools: dict[str, Any], get: Callable[..., tuple[int, bytes, float]], *, seed: int, repeats: int, n_sets: int,
            budget_s: float, guard: NetGuard | None, skip: set[str], log: Callable[[str], None]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    order = [e for e in eps if not e.excluded]
    rng.shuffle(order)
    t_start = time.time()
    rows: list[dict[str, Any]] = []
    for i, ep in enumerate(order, 1):
        if ep.eid in skip:
            continue
        if time.time() - t_start > budget_s:
            log(f"時間切れ: {i - 1}/{len(order)} 本で打ち切り")
            break
        sets = param_sets(ep, pools, rng, n_sets)
        slow = False
        for si, (path, q) in enumerate(sets):
            reps = 1 if slow else repeats
            for ci in range(reps):
                load = wait_for_load()
                before = len(guard.attempts) if guard else 0
                status, body, ms = get(ep.kind, path, q)
                net = (len(guard.attempts) - before) if guard else 0
                rows.append({"eid": ep.eid, "kind": ep.kind, "path": path, "query": q, "set": si, "call": ci, "status": status,
                             "ms": round(ms, 1), "bytes": len(body), "load1": round(load, 2), "net": net})
                if ci == 0 and ms > SLOW_FIRST_MS:
                    slow = True
        r0 = [r for r in rows if r["eid"] == ep.eid]
        log(f"[{i}/{len(order)}] {ep.eid}  初回 {r0[0]['ms']:.0f}ms → 2回目 {(r0[1]['ms'] if len(r0) > 1 else float('nan')):.0f}ms  HTTP {r0[0]['status']}")
    return rows


def _med(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 1) if xs else None


def summarize(rows: list[dict[str, Any]], eps: list[Endpoint]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["eid"], []).append(r)
    out = []
    meta = {e.eid: e for e in eps}
    for eid, rs in by.items():
        ok = [r for r in rs if 200 <= r["status"] < 300]
        first0 = next((r for r in rs if r["set"] == 0 and r["call"] == 0), None)
        new_key = [r["ms"] for r in ok if r["call"] == 0 and r["set"] >= 1]
        warm = [r["ms"] for r in ok if r["call"] >= 1]
        codes = sorted({r["status"] for r in rs})
        out.append({
            "eid": eid, "name": meta[eid].name if eid in meta else "", "n": len(rs), "statuses": codes,
            "first_process_cold_ms": first0["ms"] if first0 and 200 <= first0["status"] < 300 else None,
            "first_new_key_ms": _med(new_key), "warm_ms": _med(warm), "max_ms": max((r["ms"] for r in rs), default=None),
            "bytes": _med([float(r["bytes"]) for r in ok]), "net_attempts": sum(r["net"] for r in rs),
            "example": (rs[0]["path"] + ("?" + "&".join(f"{k}={v}" for k, v in rs[0]["query"].items()) if rs[0]["query"] else "")),
        })
    out.sort(key=lambda s: -(s["first_process_cold_ms"] or s["first_new_key_ms"] or 0))
    return out


BUCKETS = [(100, "<0.1秒"), (300, "<0.3秒"), (1000, "<1秒"), (3000, "<3秒"), (10_000, "<10秒"), (float("inf"), "10秒以上")]


def bucket_counts(vals: list[float]) -> dict[str, int]:
    c = {name: 0 for _, name in BUCKETS}
    for v in vals:
        for lim, name in BUCKETS:
            if v < lim:
                c[name] += 1
                break
    return c


def render_html(summary: list[dict[str, Any]], excluded: list[Endpoint], title: str) -> str:
    def cell(v: Any, unit: str = "") -> str:
        return "" if v is None else f"{v:,.0f}{unit}" if isinstance(v, (int, float)) else html.escape(str(v))

    def cls(v: float | None) -> str:
        return "" if v is None else "bad" if v >= 3000 else "warn" if v >= 1000 else "ok" if v < 300 else ""

    rows = "".join(
        f"<tr><td>{html.escape(s['eid'])}</td><td>{html.escape(s['example'])}</td>"
        f"<td class='{cls(s['first_process_cold_ms'])}' data-sort='{s['first_process_cold_ms'] or -1}'>{cell(s['first_process_cold_ms'])}</td>"
        f"<td class='{cls(s['first_new_key_ms'])}' data-sort='{s['first_new_key_ms'] or -1}'>{cell(s['first_new_key_ms'])}</td>"
        f"<td class='{cls(s['warm_ms'])}' data-sort='{s['warm_ms'] or -1}'>{cell(s['warm_ms'])}</td>"
        f"<td data-sort='{s['max_ms'] or -1}'>{cell(s['max_ms'])}</td><td data-sort='{s['bytes'] or -1}'>{cell(s['bytes'])}</td>"
        f"<td>{','.join(map(str, s['statuses']))}</td><td data-sort='{s['net_attempts']}'>{s['net_attempts'] or ''}</td></tr>"
        for s in summary
    )
    ex = "".join(f"<li>{html.escape(e.eid)}</li>" for e in excluded)
    return f"""<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:13px sans-serif;margin:16px}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:3px 8px;text-align:right}}
td:first-child,td:nth-child(2),th:first-child{{text-align:left}}th{{cursor:pointer;background:#eee}}.ok{{background:#e6f7e6}}.warn{{background:#fff3cd}}.bad{{background:#f8d7da}}</style>
<h2>{html.escape(title)}</h2><p>ミリ秒。ヘッダを押すと並べ替え。緑&lt;0.3秒・黄≥1秒・赤≥3秒。</p>
<table id=t><thead><tr><th>エンドポイント</th><th>例</th><th>プロセス初回</th><th>新規キー初回(中央値)</th><th>2回目以降(中央値)</th><th>最大</th><th>応答バイト</th><th>HTTP</th><th>外部通信を試みた回数</th></tr></thead><tbody>{rows}</tbody></table>
<h3>対象外 ({len(excluded)})</h3><ul>{ex}</ul>
<script>document.querySelectorAll('#t th').forEach((h,i)=>h.onclick=()=>{{const b=document.querySelector('#t tbody'),r=[...b.rows],d=h.dataset.d=h.dataset.d==='1'?'-1':'1';
const v=x=>{{const c=x.cells[i];return c.dataset.sort!==undefined?+c.dataset.sort:c.textContent}};r.sort((a,b)=>(v(a)>v(b)?1:-1)*d).forEach(x=>b.appendChild(x))}})</script>"""


def write_outputs(out: Path, rows: list[dict[str, Any]], eps: list[Endpoint], title: str) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    summary = summarize(rows, eps)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "net_risk.json").write_text(json.dumps(sorted(s["eid"] for s in summary if s["net_attempts"]), ensure_ascii=False), encoding="utf-8")
    (out / "report.html").write_text(render_html(summary, [e for e in eps if e.excluded], title), encoding="utf-8")
    cold = [s["first_process_cold_ms"] for s in summary if s["first_process_cold_ms"] is not None]
    key = [s["first_new_key_ms"] for s in summary if s["first_new_key_ms"] is not None]
    warm = [s["warm_ms"] for s in summary if s["warm_ms"] is not None]
    stats = {"endpoints": len(summary), "process_cold": bucket_counts(cold), "new_key": bucket_counts(key), "warm": bucket_counts(warm)}
    (out / "buckets.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


# ───────── CLI ─────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["pools", "inproc", "http", "list"])
    ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "keiba-latency-monkey"))
    ap.add_argument("--pools")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3, help="同じキーを開く回数（1 回目=初回、2 回目以降=温まった後）")
    ap.add_argument("--sets", type=int, default=3, help="エンドポイントごとに試すランダムなキーの組数")
    ap.add_argument("--budget-min", type=float, default=40)
    ap.add_argument("--only", help="エンドポイント名の部分一致（絞り込み）")
    ap.add_argument("--skip-file", help="外すエンドポイントの JSON 配列（inproc で外部通信を試みたもの）")
    ap.add_argument("--fastapi", default="http://127.0.0.1:23220")
    ap.add_argument("--flask", default="http://127.0.0.1:23210")
    ap.add_argument("--next", default="http://127.0.0.1:23200")
    ap.add_argument("--pages", action="store_true", help="http モードで Next.js のページも測る")
    ap.add_argument("--allow-net", action="store_true", help="inproc でも外部通信を遮断しない（既定は遮断）")
    a = ap.parse_args(argv)
    out = Path(a.out)

    def log(msg: str) -> None:
        print(msg, flush=True)

    if a.mode == "list":
        for e in build_catalog():
            print(("対象外 " if e.excluded else "       ") + e.eid, e.required, e.optional[:4])
        return 0

    if a.mode == "pools":
        os.environ.setdefault("KEIBA_PRERENDER_DIR", tempfile.mkdtemp(prefix="lm-pre-"))
        raw = make_inproc_clients()

        def cg(kind: str, path: str) -> tuple[int, Any, float]:
            s, b, ms = raw(kind, path)
            try:
                return s, json.loads(b), ms
            except ValueError:
                return s, None, ms

        pools = build_pools(cg, a.seed)
        Path(a.pools or out / "pools.json").parent.mkdir(parents=True, exist_ok=True)
        Path(a.pools or out / "pools.json").write_text(json.dumps(pools, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"サンプル {len(pools['items'])} 件（開催日 {len({i['date'] for i in pools['items']})} 日）")
        return 0

    pools = json.loads(Path(a.pools).read_text(encoding="utf-8")) if a.pools else {"items": []}
    skip = set(json.loads(Path(a.skip_file).read_text(encoding="utf-8"))) if a.skip_file else set()
    eps = build_catalog()
    if a.pages and a.mode == "http":
        eps += next_pages()
    if a.only:
        eps = [e for e in eps if a.only in e.path or a.only in e.name]
    guard: NetGuard | None = None
    if a.mode == "inproc":
        os.environ["KEIBA_PRERENDER_DIR"] = tempfile.mkdtemp(prefix="lm-pre-")
        get = make_inproc_clients()
        guard = None if a.allow_net else NetGuard()
        if guard:
            guard.__enter__()
    else:
        get = make_http_client({"F": a.fastapi, "K": a.flask, "N": a.next})
    try:
        rows = measure(eps, pools, get, seed=a.seed, repeats=a.repeats, n_sets=a.sets, budget_s=a.budget_min * 60, guard=guard, skip=skip, log=log)
    finally:
        if guard:
            guard.__exit__()
    summary = write_outputs(out, rows, eps, f"レイテンシ（{a.mode}・seed {a.seed}）")
    log(f"完了: {len(summary)} エンドポイント → {out}/report.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())

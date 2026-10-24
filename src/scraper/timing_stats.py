"""スクレイピング実測時間の記録と集計。

記録（常時）:
  - ScraperRunner.scrape_* の 1 呼び出しごと（kind="step"）と、requests.Session.get 1 回ごと（kind="http"）を
    ``data/local/meta/scrape_timing/events/YYYYMMDD.jsonl`` に追記する（日付ごとにファイルを分ける）。
  - step には http 回数・http 合計秒・待機秒（= 全体 - http。ScraperRunner の間隔制御など意図的な sleep を含む）を持たせる。
  - 無効化: ``KEIBA_SCRAPE_TIMING=0``。http を記録しない: ``KEIBA_SCRAPE_TIMING=steps``。
    保存先の変更: ``KEIBA_SCRAPE_TIMING_DIR``。記録の失敗はスクレイピングを止めない。

集計（T-X など「いつ回すか」の判断材料）:
  python -m src.scraper.timing_stats report                      # 直近 7 日、データ点（メソッド/phase）ごと
  python -m src.scraper.timing_stats report --days 30 --by hour  # 時間帯別
  python -m src.scraper.timing_stats report --method scrape_odds --json
  既定ではキャッシュヒット（http 0 回）と、他の scrape_* から呼ばれた入れ子の呼び出しを除く
  （--include-cache / --include-nested で含める）。
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import statistics
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

ODDS_TYPE_LABEL = {"1": "単勝", "2": "複勝", "4": "馬連", "5": "ワイド", "6": "馬単", "7": "3連複"}

_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCK = threading.Lock()
_TLS = threading.local()


def _mode() -> str:
    return (os.environ.get("KEIBA_SCRAPE_TIMING") or "all").strip().lower()


def events_dir() -> Path:
    override = (os.environ.get("KEIBA_SCRAPE_TIMING_DIR") or "").strip()
    return Path(override) if override else _ROOT / "data" / "local" / "meta" / "scrape_timing" / "events"


def http_label(url: str) -> str:
    m = re.search(r"api_get_jra_odds\.html.*[?&]type=(\d+)", url)
    if m:
        return f"オッズAPI/{ODDS_TYPE_LABEL.get(m.group(1), 'type' + m.group(1))}"
    m = re.search(r"//([^/]+)/(?:[^?]*/)?([A-Za-z0-9_]+)\.html", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"//([^/]+)", url)
    return m.group(1) if m else url[:60]


def record(event: dict[str, Any]) -> None:
    """1 イベントを JSONL に追記する。失敗は握りつぶす。"""
    try:
        if _mode() in ("0", "off", "false", "no"):
            return
        now = datetime.now()
        event = {"ts": now.isoformat(timespec="milliseconds"), "env": os.environ.get("KEIBA_ENV", ""), **event}
        d = events_dir()
        with _LOCK:
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{now:%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _acc() -> dict[str, float]:
    a = getattr(_TLS, "acc", None)
    if a is None:
        a = _TLS.acc = {"n": 0, "sec": 0.0}
    return a


def wrap_session_get(orig: Callable[..., Any]) -> Callable[..., Any]:
    """requests.Session.get を計測付きにする（スレッドごとの累積と http イベント記録）。"""

    @functools.wraps(orig)
    def timed_get(url, *a, **kw):
        t0 = time.perf_counter()
        status, size = None, 0
        try:
            resp = orig(url, *a, **kw)
            status, size = resp.status_code, len(getattr(resp, "content", b"") or b"")
            return resp
        finally:
            sec = time.perf_counter() - t0
            acc = _acc()
            acc["n"] += 1
            acc["sec"] += sec
            if _mode() == "all":
                record({"kind": "http", "label": http_label(str(url)), "sec": round(sec, 4), "status": status, "bytes": size})

    return timed_get


def timed_scrape(fn: Callable[..., Any]) -> Callable[..., Any]:
    """ScraperRunner.scrape_* 用デコレータ。"""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        depth = getattr(_TLS, "depth", 0)
        _TLS.depth = depth + 1
        acc = _acc()
        n0, s0 = acc["n"], acc["sec"]
        t0 = time.perf_counter()
        ok = False
        try:
            result = fn(self, *args, **kwargs)
            ok = bool(result)
            return result
        finally:
            _TLS.depth = depth
            wall = time.perf_counter() - t0
            http_n = int(acc["n"] - n0)
            http_sec = acc["sec"] - s0
            key = args[0] if args and isinstance(args[0], (str, int)) else ""
            record({
                "kind": "step", "method": fn.__name__, "key": str(key), "phase": kwargs.get("phase"),
                "sec": round(wall, 4), "http_n": http_n, "http_sec": round(http_sec, 4),
                "wait_sec": round(max(0.0, wall - http_sec), 4), "ok": ok, "nested": depth > 0,
            })

    return wrapper


# ── 集計 ────────────────────────────────────────────────────────────────────


def load_events(since: datetime, until: datetime | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    d = events_dir()
    if not d.is_dir():
        return out
    day = since.date()
    last = (until or datetime.now()).date()
    while day <= last:
        p = d / f"{day:%Y%m%d}.jsonl"
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("ts", "") >= since.isoformat():
                    out.append(e)
        day += timedelta(days=1)
    return out


def _stats(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    return {"n": len(xs), "min": xs[0], "med": statistics.median(xs),
            "p95": xs[min(len(xs) - 1, math.ceil(len(xs) * 0.95) - 1)], "max": xs[-1]}


def _group_key(e: dict[str, Any], by: str) -> str:
    if by == "hour":
        return e["ts"][11:13] + "時台"
    if by == "weekday":
        return "月火水木金土日"[datetime.fromisoformat(e["ts"]).weekday()] + "曜"
    if by == "env":
        return e.get("env") or "-"
    return ""


def summarize(events: list[dict[str, Any]], *, method: str | None = None, include_cache: bool = False,
              include_nested: bool = False, by: str = "") -> dict[str, Any]:
    steps: dict[str, list[dict]] = {}
    https: dict[str, list[dict]] = {}
    for e in events:
        if e.get("kind") == "step":
            if method and e.get("method") != method:
                continue
            if e.get("nested") and not include_nested:
                continue
            if e.get("http_n", 0) == 0 and not include_cache:
                continue
            name = e["method"] + (f"[{e['phase']}]" if e.get("phase") else "")
            g = _group_key(e, by)
            steps.setdefault(f"{name} {g}".strip(), []).append(e)
        elif e.get("kind") == "http" and not method:
            https.setdefault((_group_key(e, by) + " " if by else "") + e["label"], []).append(e)
    return {
        "steps": {k: {**_stats([x["sec"] for x in v]),
                      "ok_rate": sum(1 for x in v if x.get("ok")) / len(v),
                      "http_med": statistics.median(x["http_sec"] for x in v),
                      "wait_med": statistics.median(x["wait_sec"] for x in v),
                      "req_med": statistics.median(x["http_n"] for x in v)} for k, v in sorted(steps.items())},
        "http": {k: {**_stats([x["sec"] for x in v]), "bytes_max": max(x.get("bytes", 0) for x in v),
                     "err_rate": sum(1 for x in v if not x.get("status") or x["status"] >= 400) / len(v)}
                 for k, v in sorted(https.items())},
    }


def print_report(s: dict[str, Any], days: int) -> None:
    print(f"=== スクレイピング実測時間（直近 {days} 日, 秒）===")
    print(f"{'データ点(メソッド[phase])':<40}{'n':>6}{'成功率':>7}{'min':>8}{'med':>8}{'p95':>8}{'max':>8}{'HTTP med':>9}{'待機 med':>9}{'req':>5}")
    for k, v in s["steps"].items():
        print(f"{k:<40}{v['n']:>6}{v['ok_rate'] * 100:>6.0f}%{v['min']:>8.2f}{v['med']:>8.2f}{v['p95']:>8.2f}{v['max']:>8.2f}"
              f"{v['http_med']:>9.2f}{v['wait_med']:>9.2f}{v['req_med']:>5.0f}")
    if s["http"]:
        print(f"\n{'HTTP 1 回ごと':<40}{'n':>6}{'err':>7}{'min':>8}{'med':>8}{'p95':>8}{'max':>8}  最大bytes")
        for k, v in s["http"].items():
            print(f"{k:<40}{v['n']:>6}{v['err_rate'] * 100:>6.0f}%{v['min']:>8.2f}{v['med']:>8.2f}{v['p95']:>8.2f}{v['max']:>8.2f}  {v['bytes_max']:>9}")
    if not s["steps"] and not s["http"]:
        print("（記録がありません。スクレイピングを実行すると自動で蓄積されます）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.scraper.timing_stats")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="蓄積した実測時間を集計表示")
    r.add_argument("--days", type=int, default=7)
    r.add_argument("--method", help="例: scrape_odds（指定時は HTTP 表を出さない）")
    r.add_argument("--by", choices=["", "hour", "weekday", "env"], default="")
    r.add_argument("--include-cache", action="store_true")
    r.add_argument("--include-nested", action="store_true")
    r.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    ev = load_events(datetime.now() - timedelta(days=args.days))
    s = summarize(ev, method=args.method, include_cache=args.include_cache, include_nested=args.include_nested, by=args.by)
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        print_report(s, args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""VPS の API キャッシュ(nginx proxy_cache)を、参照用データの更新に合わせて自動で取り直す（事前ロード）。

- 更新の知らせ: GCS の ``ops/cache_dirty/<YYYYMMDD UTC>/<キー>``（キー = race_id 12桁 / 開催日8桁 / ``_ALL``）。書き手は
  ``src/utils/cache_marker.py``（取り込み・ミラー・配備）。ここでは 1 周に **一覧 1 回**（昨日以降の接頭辞）だけ読み、個別の読み出しはしない。
- 取り直し: ローカルの nginx(既定 127.0.0.1:8080, Host=app.megu-keiba.com = 本番の通り道と同じキー)へ ``X-Keiba-Refresh`` 付きで GET。
  nginx がキャッシュ読みを飛ばして API から取り直し、保存(上書き)する。結果の保持期間は API の Cache-Control(確定レースは 30 日)で決まる。
- GCP の読みは「一覧 1 回/周」だけ。キャッシュ本体は VPS のディスク(/var/cache/nginx/keiba_api)。
- 状態: /var/lib/keiba-ui/cache-warm/state.json（処理済みの目印の世代と、未処理の取り直し待ち）。
- 標準ライブラリ + google-cloud-storage(pull 用 venv)だけで動く。systemd タイマー(keiba-cache-warm.timer)が 60 秒おきに起動。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

PREFIX = "ops/cache_dirty"
ALL_KEY = "_ALL"
RACE_RE = re.compile(r"^\d{12}$")
DATE_RE = re.compile(r"^\d{8}$")
JST = timezone(timedelta(hours=9))

RACE_PATHS = (
    "/api/race/{id}",
    "/api/race/{id}/predictions",
    "/api/race/{id}/result",
    "/api/race/{id}/tracking-difficulty",
    "/api/v1/races/{id}/megu-index-predicted",
    "/api/v1/races/{id}/analysis",
)
GLOBAL_PATHS = (
    "/api/scrape-dates?picker_past_days=14&picker_future_days=14",
    "/api/scrape-dates?picker_past_days=120",
    "/api/scrape-dates?filter=meeting",
    "/api/megu-index-dates",
)
MAX_QUEUE = 5000
MAX_ATTEMPTS = 3


def env(name: str, default: str) -> str:
    return (os.environ.get(name) or "").strip() or default


def jst_today(now: datetime | None = None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(JST)


def parse_marker(name: str) -> tuple[str, str] | None:
    """``ops/cache_dirty/<日>/<キー>`` → (日, キー)。形が違うものは None。"""
    parts = name.split("/")
    if len(parts) != 4 or "/".join(parts[:2]) != PREFIX:
        return None
    day, key = parts[2], parts[3]
    if not DATE_RE.match(day):
        return None
    if key == ALL_KEY or RACE_RE.match(key) or DATE_RE.match(key):
        return day, key
    return None


def date_paths(date: str) -> list[str]:
    return [f"/api/race-list/{date}", f"/api/race-calendar?month={date[:6]}"]


def window_dates(now: datetime | None = None) -> list[str]:
    t = jst_today(now)
    return [(t + timedelta(days=d)).strftime("%Y%m%d") for d in (-1, 0, 1, 2)]


def paths_for_key(key: str, now: datetime | None = None) -> list[tuple[str, bool]]:
    """目印 1 つぶんの取り直し先 (パス, 展開するか)。race_id → そのレースの画面用 API。開催日 → 一覧・カレンダー・日付ピッカー。
    _ALL → 全体 + 前後数日（一覧は、載っているレースの詳細・予測まで展開する）。"""
    if RACE_RE.match(key):
        return [(p.format(id=key), False) for p in RACE_PATHS]
    if DATE_RE.match(key):
        return [(p, False) for p in date_paths(key) + list(GLOBAL_PATHS)]
    if key == ALL_KEY:
        out = [(p, False) for p in GLOBAL_PATHS]
        for d in window_dates(now):
            out += [(f"/api/race-list/{d}", True), (f"/api/race-calendar?month={d[:6]}", False)]
        return out
    return []


def normalize_date(value: object) -> str | None:
    digits = re.sub(r"\D", "", str(value or ""))[:8]
    return digits if len(digits) == 8 else None


def race_ids_in_list(doc: object) -> list[str]:
    races = doc.get("races") if isinstance(doc, dict) else None
    return [str(r["race_id"]) for r in races or [] if isinstance(r, dict) and RACE_RE.match(str(r.get("race_id") or ""))]


def load_state(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(d, dict):
            d.setdefault("seen", {})
            d.setdefault("queue", [])
            return d
    except (OSError, ValueError):
        pass
    return {"seen": {}, "queue": []}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def discover(blobs: list[tuple[str, int]], state: dict, now: datetime | None = None) -> list[str]:
    """一覧(名前, 世代)から、新しい/世代が変わった目印を拾い、取り直し先を queue に積む。積んだパスの数を返す。
    古い日の目印は seen から捨てる。_ALL は各日の最新だけで足りるが、世代が違えば毎回取り直す。"""
    seen: dict = state["seen"]
    queued = {q["p"] for q in state["queue"]}
    added: list[str] = []
    fresh: dict[str, int] = {}
    for name, gen in sorted(blobs):
        pk = parse_marker(name)
        if pk is None:
            continue
        fresh[name] = gen
        if seen.get(name) == gen:
            continue
        seen[name] = gen
        for p, expand in paths_for_key(pk[1], now):
            if p not in queued and len(state["queue"]) < MAX_QUEUE:
                state["queue"].append({"p": p, "n": 0, "x": 1} if expand else {"p": p, "n": 0})
                queued.add(p)
                added.append(p)
    for name in list(seen):
        if name not in fresh:
            del seen[name]
    return added


class Warmer:
    def __init__(self, base: str, host: str, token: str, timeout: float = 30.0):
        self.base = base.rstrip("/")
        self.host = host
        self.token = token
        self.timeout = timeout

    def fetch(self, path: str) -> tuple[int, bytes]:
        req = urllib.request.Request(self.base + path, method="GET", headers={
            "Host": self.host, "X-Keiba-Refresh": self.token, "User-Agent": "keiba-cache-warm", "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, b""
        except (urllib.error.URLError, OSError, TimeoutError):
            return 0, b""


def drain(state: dict, warmer: Warmer, *, max_requests: int, max_sec: float, pause: float, log=print) -> dict:
    """queue を先頭から処理する。レース詳細の応答の ``date`` から、その開催日の一覧も取り直す。"""
    q = deque(state["queue"])
    done = failed = 0
    started = time.monotonic()
    dates_done: set[str] = set()
    requeue: list[dict] = []
    while q and done + failed < max_requests and time.monotonic() - started < max_sec:
        item = q.popleft()
        path = item["p"]
        status, body = warmer.fetch(path)
        if status == 200:
            done += 1
            m = re.fullmatch(r"/api/race/(\d{12})", path)
            if m:
                try:
                    d = normalize_date(json.loads(body).get("date"))
                except (ValueError, AttributeError):
                    d = None
                if d and d not in dates_done:
                    dates_done.add(d)
                    for p in date_paths(d):
                        if not any(x["p"] == p for x in q):
                            q.appendleft({"p": p, "n": 0})
            elif item.get("x"):      # 全体の取り直し: 一覧に載っているレースの詳細・予測も取り直す
                try:
                    ids = race_ids_in_list(json.loads(body))
                except ValueError:
                    ids = []
                have = {x["p"] for x in q}
                for rid in ids:
                    for p in (f"/api/race/{rid}", f"/api/race/{rid}/predictions"):
                        if p not in have and len(q) < MAX_QUEUE:
                            q.append({"p": p, "n": 0})
        elif status in (404, 400, 401, 403, 422):      # データ無し・対象外は取り直しても同じ（まだ無いレースなど）
            done += 1
        else:
            failed += 1
            item["n"] = int(item.get("n", 0)) + 1
            if item["n"] < MAX_ATTEMPTS:
                requeue.append(item)
            log(f"取り直し失敗 {path} -> {status}（{item['n']}/{MAX_ATTEMPTS}）")
        if pause:
            time.sleep(pause)
    state["queue"] = list(q) + requeue
    return {"ok": done, "ng": failed, "left": len(state["queue"])}


def list_markers(bucket_name: str, now: datetime | None = None) -> list[tuple[str, int]]:
    """GCS を 1 回 list する（昨日(UTC)以降の接頭辞だけ）。"""
    from google.cloud import storage  # pull 用 venv にある

    yday = ((now or datetime.now(timezone.utc)) - timedelta(days=1)).strftime("%Y%m%d")
    client = storage.Client()
    it = client.list_blobs(bucket_name, prefix=PREFIX + "/", start_offset=f"{PREFIX}/{yday}/")
    return [(b.name, int(b.generation or 0)) for b in it]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="目印の一覧と取り直し先を表示するだけ（取り直しも状態の更新もしない）")
    ap.add_argument("--path", action="append", default=[], help="目印を介さず、このパスを取り直す（複数可。動作確認用）")
    args = ap.parse_args(argv)

    state_dir = Path(env("WARM_STATE_DIR", "/var/lib/keiba-ui/cache-warm"))
    state_file = state_dir / "state.json"
    base = env("WARM_BASE", f"http://127.0.0.1:{env('WARM_PORT', '8080')}")
    host = env("WARM_HOST", "app.megu-keiba.com")
    token_file = Path(env("WARM_TOKEN_FILE", "/etc/keiba-ui/refresh.token"))
    max_requests = int(env("WARM_MAX_PER_TICK", "200"))
    max_sec = float(env("WARM_MAX_SEC", "150"))
    pause = float(env("WARM_PAUSE_SEC", "0.05"))

    state = load_state(state_file)
    if args.path:
        for p in args.path:
            state["queue"].append({"p": p, "n": 0})
    else:
        bucket = env("GCS_BUCKET", "")
        if not bucket:
            print("GCS_BUCKET が未設定です", file=sys.stderr)
            return 2
        added = discover(list_markers(bucket), state)
        if added:
            print(f"目印から {len(added)} 件を取り直し待ちに追加（待ち合計 {len(state['queue'])}）")
    if args.dry_run:
        for item in state["queue"][:50]:
            print("  ", item["p"])
        print(f"(dry-run) 待ち {len(state['queue'])} 件。取り直しも状態の更新もしていません")
        return 0
    try:
        token = token_file.read_text(encoding="utf-8").strip()
    except OSError as e:
        print(f"リフレッシュ用トークンを読めません: {e}", file=sys.stderr)
        return 2
    if not token:
        print("リフレッシュ用トークンが空です", file=sys.stderr)
        return 2
    if not state["queue"]:
        save_state(state_file, state)
        return 0
    res = drain(state, Warmer(base, host, token), max_requests=max_requests, max_sec=max_sec, pause=pause)
    save_state(state_file, state)
    print(f"取り直し ok={res['ok']} ng={res['ng']} 残り={res['left']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

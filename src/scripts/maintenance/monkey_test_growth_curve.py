#!/usr/bin/env python3
"""成長曲線 API のランダムモンキーテスト（数十件・表示可否を検証）。

Usage:
  python3 -m src.scripts.maintenance.monkey_test_growth_curve
  python3 -m src.scripts.maintenance.monkey_test_growth_curve --n 40 --base-url http://127.0.0.1:8000
  python3 -m src.scripts.maintenance.monkey_test_growth_curve --use-testclient
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

INDEX_PATH = ROOT / "data" / "calculated_data" / "knowledge" / "horse_name_index.json"
CACHE_HR = ROOT / "data" / "cache" / "horse_result"


def collect_horse_ids() -> list[str]:
    ids: set[str] = set()
    if INDEX_PATH.is_file():
        raw = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        horses = raw.get("horses") or raw
        if isinstance(horses, dict):
            ids.update(horses.keys())
        elif isinstance(horses, list):
            for h in horses:
                if isinstance(h, dict) and h.get("horse_id"):
                    ids.add(str(h["horse_id"]))
    if CACHE_HR.is_dir():
        for p in CACHE_HR.rglob("*.json"):
            ids.add(p.stem)
    return sorted(ids)


def validate_display_payload(body: dict) -> list[str]:
    """フロント growth_curve.html の render 要件に沿って検証。"""
    issues: list[str] = []
    if body.get("error"):
        issues.append(str(body["error"]))
        return issues
    if not body.get("horse_name"):
        issues.append("missing horse_name")
    races = body.get("races")
    if not isinstance(races, list) or len(races) == 0:
        issues.append("races empty or invalid")
    for key in ("total_races", "avg_weight", "weight_range"):
        if key not in body:
            issues.append(f"missing {key}")
    wr = body.get("weight_range")
    if not isinstance(wr, list) or len(wr) != 2:
        issues.append("weight_range invalid")
    else:
        try:
            float(wr[0])
            float(wr[1])
        except (TypeError, ValueError):
            issues.append("weight_range not numeric")
    try:
        float(body.get("avg_weight", 0))
    except (TypeError, ValueError):
        issues.append("avg_weight not numeric")
    if races:
        for i, r in enumerate(races[:5]):
            if not r.get("date"):
                issues.append(f"race[{i}] missing date")
    return issues


def fetch_http(base_url: str, horse_id: str, *, fetch_speed_index: bool) -> tuple[int, dict]:
    q = "fetch_speed_index=false" if not fetch_speed_index else "fetch_speed_index=true"
    url = f"{base_url.rstrip('/')}/api/growth-curve/{horse_id}?{q}"
    try:
        with urllib.request.urlopen(url, timeout=90) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else "{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"error": raw[:200]}
        return e.code, body


def fetch_testclient(horse_id: str, *, fetch_speed_index: bool) -> tuple[int, dict]:
    os.environ.setdefault("REQUIRE_AUTH", "false")
    os.environ.setdefault("HORSE_NAME_INDEX_DISABLE_BOOTSTRAP", "1")
    from fastapi.testclient import TestClient

    from src.api.app import app

    client = TestClient(app, raise_server_exceptions=False)
    q = f"fetch_speed_index={'true' if fetch_speed_index else 'false'}"
    r = client.get(f"/api/growth-curve/{horse_id}?{q}")
    try:
        body = r.json()
    except Exception:
        body = {"error": r.text[:200]}
    return r.status_code, body


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=50, help="テスト件数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-url", default=os.environ.get("MONKEY_BASE_URL", "http://127.0.0.1:8000"))
    p.add_argument("--use-testclient", action="store_true")
    p.add_argument("--fetch-speed-index", action="store_true", help="タイム指数補完あり")
    args = p.parse_args()

    all_ids = collect_horse_ids()
    if not all_ids:
        print("馬ID候補がありません")
        return 1

    n = min(args.n, len(all_ids))
    random.seed(args.seed)
    sample = random.sample(all_ids, n)

    mode = "TestClient" if args.use_testclient else args.base_url
    print(f"成長曲線モンキーテスト: {n}件 (seed={args.seed}) via {mode}")
    print(f"{'horse_id':<14} {'HTTP':<5} {'races':<6} {'result':<6} 備考")
    print("-" * 72)

    ok_count = 0
    failures: list[tuple[str, int, str]] = []
    t0 = time.time()

    for hid in sample:
        if args.use_testclient:
            st, body = fetch_testclient(hid, fetch_speed_index=args.fetch_speed_index)
        else:
            st, body = fetch_http(args.base_url, hid, fetch_speed_index=args.fetch_speed_index)

        issues = validate_display_payload(body) if st == 200 else [body.get("error", f"HTTP {st}")]
        disp_ok = st == 200 and not issues
        n_races = len(body.get("races") or []) if isinstance(body, dict) else 0
        mark = "OK" if disp_ok else "NG"
        reason = issues[0] if issues else "ok"
        if disp_ok:
            ok_count += 1
        else:
            failures.append((hid, st, reason))
        print(f"{hid:<14} {st:<5} {n_races:<6} {mark:<6} {reason}")

    print()
    print(f"表示可能: {ok_count}/{n} ({time.time() - t0:.1f}s)")
    if failures:
        print("\n失敗一覧:")
        for hid, st, msg in failures[:20]:
            print(f"  {hid} ({st}): {msg}")
        if len(failures) > 20:
            print(f"  ... 他 {len(failures) - 20} 件")
    return 0 if ok_count == n else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
stg / 本番同等のローカル起動に対し、API と主要 HTML ページが閲覧可能か一括確認する。

  KEIBA_STG_BASE=http://127.0.0.1:8000 python3 scripts/verify_stg_smoke.py

- /api/health に keiba_env 等が含まれること（stg 想定時は keiba_env=stg）
- 公開 HTML が 200 かつ HTML らしき本文
- DEV_PASSWORD があれば /login 後に開発者ページ数件
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import requests  # noqa: E402

BASE = os.environ.get("KEIBA_STG_BASE", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 45


def _html_ok(text: str) -> bool:
    t = text[:8000].lower()
    return "<!doctype html" in t or "<html" in t


def main() -> int:
    fails = 0
    s = requests.Session()
    s.headers.setdefault("User-Agent", "verify_stg_smoke/1.0")

    print(f"BASE={BASE}\n")

    # --- health (stg) ---
    try:
        r = s.get(f"{BASE}/api/health", timeout=TIMEOUT)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as e:
        print(f"NG  health  exception: {e}")
        return 2

    if r.status_code != 200:
        print(f"NG  health  HTTP {r.status_code}")
        return 2

    env = data.get("keiba_env", "")
    x_hdr = r.headers.get("X-Keiba-Env", "")
    print(f"OK  health  keiba_env={env!r}  X-Keiba-Env={x_hdr!r}")
    if os.environ.get("KEIBA_ENV", "").strip().lower() in ("stg", "staging"):
        if env != "stg":
            print(f"WARN: .env は stg だが health.keiba_env={env!r}")
        if x_hdr != "stg":
            print(f"WARN: X-Keiba-Env が stg でない: {x_hdr!r}")

    # --- sample race for /race/{id} ---
    race_id = "202006010501"
    try:
        r0 = s.get(f"{BASE}/api/scrape-dates", timeout=TIMEOUT)
        if r0.status_code == 200:
            dj = r0.json()
            dates = dj.get("dates") or []
            if dates:
                d0 = dates[-1]
                r1 = s.get(f"{BASE}/api/race-list/{d0}", timeout=TIMEOUT)
                if r1.status_code == 200:
                    races = r1.json().get("races") or []
                    if races and races[0].get("race_id"):
                        race_id = str(races[0]["race_id"])
    except Exception:
        pass
    print(f"sample race_id={race_id}")

    public_paths = [
        "/",
        "/login",
        "/tracking-difficulty",
        "/bloodline",
        "/bloodline-vector",
        "/pedigree-map",
        "/bloodline-cluster",
        "/course-bloodline",
        "/pedigree-race-stats",
        "/myostatin",
        "/note-aptitude-race",
        "/track-speed",
        "/growth-curve",
        f"/race/{race_id}",
    ]

    print("\n--- 公開 HTML ---")
    for path in public_paths:
        url = f"{BASE}{path}"
        try:
            r = s.get(url, timeout=TIMEOUT, allow_redirects=True)
        except Exception as e:
            print(f"NG  {path:<32} exception {e}")
            fails += 1
            continue
        ok = r.status_code == 200 and _html_ok(r.text)
        if ok:
            print(f"OK  {path:<32} {len(r.text)} bytes")
        else:
            print(f"NG  {path:<32} HTTP {r.status_code} html_like={_html_ok(r.text)}")
            fails += 1

    # --- dev pages (optional) ---
    pw = os.environ.get("DEV_PASSWORD", "").strip()
    if pw:
        print("\n--- 開発者ページ（ログイン後）---")
        try:
            r = s.post(
                f"{BASE}/login",
                data={"password": pw, "next": "/"},
                timeout=TIMEOUT,
                allow_redirects=True,
            )
        except Exception as e:
            print(f"NG  login  {e}")
            fails += 1
            pw = ""

    if pw:
        dev_paths = [
            "/cron-jobs",
            "/queue-status",
            "/monitor",
            "/data-viewer",
        ]
        for path in dev_paths:
            url = f"{BASE}{path}"
            try:
                r = s.get(url, timeout=TIMEOUT, allow_redirects=True)
            except Exception as e:
                print(f"NG  {path:<32} {e}")
                fails += 1
                continue
            ok = r.status_code == 200 and _html_ok(r.text)
            if ok:
                print(f"OK  {path:<32} {len(r.text)} bytes")
            else:
                # ログイン失敗時は login ページへ
                loginish = "password" in r.text.lower() and r.status_code in (200, 401, 403)
                print(f"NG  {path:<32} HTTP {r.status_code} html={_html_ok(r.text)} loginish={loginish}")
                fails += 1
    else:
        print("\n--- 開発者ページ: DEV_PASSWORD なしのためスキップ ---")

    print(f"\nHTML チェック結果: {'全て OK' if fails == 0 else f'NG {fails} 件'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

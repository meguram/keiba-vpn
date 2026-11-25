"""匿名（または無料会員）で API を取得し、出してはいけない欄が 1 つも無いことを確かめる。

範囲（2026-11-25 ユーザーの確定）: 追い切り・厩舎コメント・馬柱の指数と備考・確定前のオッズ・人気・払戻・予測の数値（勝率・複勝率・順位）は出さない。
確定の結果（race_result）の確定オッズ・人気・払戻だけは出してよい。判定は src/api/tier.py の find_scope_leaks。

使い方:
  python -m src.scripts.ops.check_anon_scope --base https://app.megu-keiba.com --date 20261123
  KEIBA_SCOPE_COOKIE=<無料会員のセッション> python -m src.scripts.ops.check_anon_scope --base ... --date ...   # 無料会員で確かめる
費用: 1 回で、レース数 × 3 件 ＋ 5 件程度の要求（既定 12 レースで約 40 件）。周期では流さない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from src.api.auth import COOKIE_NAME
from src.api.tier import find_scope_leaks


def _get(base: str, path: str, cookie: str) -> tuple[int, Any]:
    headers = {"User-Agent": "keiba-scope-check", "Accept": "application/json"}
    if cookie:
        headers["Cookie"] = f"{COOKIE_NAME}={cookie}"
    req = urllib.request.Request(base.rstrip("/") + path, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, None


def check(base: str, dates: list[str], max_races: int, cookie: str = "") -> list[str]:
    """漏れの一覧（"パス: 位置"）。空なら合格。"""
    problems: list[str] = []

    def scan(path: str, *, confirmed: bool | None = None, must: bool = True) -> Any:
        code, body = _get(base, path, cookie)
        if code != 200 or body is None:
            if must:
                problems.append(f"{path}: HTTP {code}")
            return None
        c = bool(((body.get("access") or {}).get("result_confirmed")) if isinstance(body, dict) else False) if confirmed is None else confirmed
        problems.extend(f"{path}: {x}" for x in find_scope_leaks(body, confirmed=c))
        return body

    for p in ("/api/prediction-results", "/api/prediction-results/news", "/api/prediction-results/mark-card"):
        scan(p, confirmed=False)
    rids: list[str] = []
    for d in dates:
        lst = scan(f"/api/race-list/{d}", confirmed=False)
        for r in (lst.get("races") if isinstance(lst, dict) else lst if isinstance(lst, list) else None) or []:
            if isinstance(r, dict) and r.get("race_id"):
                rids.append(str(r["race_id"]))
    for rid in rids[:max_races]:
        detail = scan(f"/api/race/{rid}")
        scan(f"/api/race/{rid}/predictions", confirmed=False)
        entries = ((detail or {}).get("race_shutuba") or {}).get("entries") or []
        hid = next((e.get("horse_id") for e in entries if isinstance(e, dict) and e.get("horse_id")), None)
        if hid:
            scan(f"/api/race/{rid}/entry/{hid}/card", confirmed=False, must=False)
            if rid == rids[0]:
                for p in (f"/api/horse/{hid}/detail", f"/api/data/race_oikiri/{rid}", f"/api/data/race_odds/{rid}",
                          f"/api/betting/pair-odds/{rid}", f"/api/odds/history/{rid}"):
                    code, _ = _get(base, p, cookie)
                    if code not in (401, 403):                 # 会員限定（匿名・無料は 401。anti_scraping の 401・403 も可）
                        problems.append(f"{p}: HTTP {code}（401 のはず）")
    if not rids:
        problems.append("レースが 1 件も見つからない（--date を確かめる）")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--date", action="append", required=True, help="YYYYMMDD（複数可）")
    ap.add_argument("--max-races", type=int, default=12)
    a = ap.parse_args(argv)
    probs = check(a.base, a.date, a.max_races, os.environ.get("KEIBA_SCOPE_COOKIE", ""))
    for p in probs:
        print("LEAK", p)
    print("OK" if not probs else f"NG {len(probs)} 件")
    return 0 if not probs else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""GCP(Cloud Run Jobs) の送信元 IP から netkeiba の公開ページに届くかの試験。

公開ページのみ・ログインなし・低頻度。VPS の試験(/tmp/netkeiba_trial.py)と同じ 30 件
（db.netkeiba.com のレース結果と race.netkeiba.com の結果ページを交互）。
1 リクエスト 1 行の JSON を標準出力へ。403/429 が 1 回でも出たら即中止。最後に summary 行（verdict=PASS/FAIL）。
終了コード: PASS=0、FAIL=1（Cloud Run Jobs の実行が失敗扱いになる）。
標準ライブラリだけで動く。
"""
from __future__ import annotations

import gzip
import json
import os
import random
import socket
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASS_RATE = 0.93  # 28/30
BLOCK_WORDS = ["アクセスが集中", "Access Denied", "captcha", "Forbidden", "しばらく時間をおいて"]
STOP_CODES = (403, 429)
CONSECUTIVE_FAIL_LIMIT = 5

# 本番クライアント(src/scraper/client.py + identity.py)と同じ方針: ブラウザを装わない固定の識別可能な UA。
# 連絡先は環境変数 KEIBA_SCRAPER_CONTACT（メールまたは URL。イメージには焼かない。ジョブの環境変数で渡す）。
# urllib は br を展開できないので Accept-Encoding は gzip, deflate。
_TOKEN = "keiba-research-bot/1.0"
_CONTACT = (os.environ.get("KEIBA_SCRAPER_CONTACT") or "").strip()
HEADERS = {
    "User-Agent": f"Mozilla/5.0 (compatible; {_TOKEN}; +{_CONTACT})" if _CONTACT else f"Mozilla/5.0 (compatible; {_TOKEN})",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


def plan(ids: list[str]) -> list[dict]:
    out = []
    for i, rid in enumerate(ids):
        if i % 2 == 0:
            out.append({"i": i, "host": "db.netkeiba.com", "url": f"https://db.netkeiba.com/race/{rid}/", "marker": "race_table_01", "encoding": "euc-jp", "referer": "https://db.netkeiba.com/"})
        else:
            out.append({"i": i, "host": "race.netkeiba.com", "url": f"https://race.netkeiba.com/race/result.html?race_id={rid}", "marker": "RaceTable01", "encoding": "utf-8", "referer": "https://race.netkeiba.com/"})
    return out


def fetch(item: dict, timeout: float = 30.0) -> dict:
    """1 URL を取得して結果を返す。例外は握りつぶして err に入れる。"""
    t = time.time()
    res = {"i": item["i"], "host": item["host"], "code": None, "bytes": 0, "marker_ok": False, "block_word": False, "err": None}
    try:
        req = urllib.request.Request(item["url"], headers={**HEADERS, "Referer": item["referer"]})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res["code"] = r.status
            body = r.read()
            enc = (r.headers.get("Content-Encoding") or "").lower()
            if enc == "gzip":
                body = gzip.decompress(body)
            elif enc == "deflate":
                body = zlib.decompress(body)
            res["bytes"] = len(body)
            txt = body.decode(item["encoding"], errors="replace")
            res["marker_ok"] = item["marker"] in txt
            res["block_word"] = any(w in txt for w in BLOCK_WORDS)
    except urllib.error.HTTPError as e:
        res["code"] = e.code
        res["err"] = "HTTP"
        try:  # 拒否の理由を調べるための診断（ヘッダの一部と本文の先頭だけ。個人情報・秘密は含まない）
            h = e.headers
            res["diag"] = {k: (h.get(k) or "")[:80] for k in ("Server", "Via", "CF-RAY", "X-Cache", "Content-Type", "Retry-After")}
            res["diag"]["body_head"] = e.read(300).decode("utf-8", errors="replace").replace("\n", " ")[:300]
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        res["err"] = type(e).__name__
    res["sec"] = round(time.time() - t, 2)
    return res


def egress_ip(timeout: float = 10.0) -> str | None:
    try:
        with urllib.request.urlopen("https://api.ipify.org", timeout=timeout) as r:
            return r.read().decode().strip()
    except Exception:  # noqa: BLE001
        return None


def judge(results: list[dict], planned: int, stop_reason: str | None) -> dict:
    """合否判定。合格 = 中止なし・403/429 が 0 件・遮断語 0 件・成功率が PASS_RATE 以上（成功 = 200 かつ目印あり）。"""
    ok = sum(1 for r in results if r["code"] == 200 and r["marker_ok"])
    n403_429 = sum(1 for r in results if r["code"] in STOP_CODES)
    blocks = sum(1 for r in results if r["block_word"])
    reasons = []
    if stop_reason:
        reasons.append(f"中止: {stop_reason}")
    if n403_429:
        reasons.append(f"403/429 が {n403_429} 件")
    if blocks:
        reasons.append(f"遮断を示す語が {blocks} 件")
    rate = ok / planned if planned else 0.0
    if rate < PASS_RATE:
        reasons.append(f"成功率 {ok}/{planned} = {rate:.0%} が基準 {PASS_RATE:.0%} 未満")
    return {"verdict": "FAIL" if reasons else "PASS", "ok": ok, "planned": planned, "done": len(results), "n403_429": n403_429, "block_words": blocks, "success_rate": round(rate, 3), "reasons": reasons}


def run(ids: list[str], *, fetch_fn=fetch, sleep_fn=time.sleep, min_sec=4.5, max_sec=7.0, emit=None) -> dict:
    emit = emit or (lambda rec: print(json.dumps(rec, ensure_ascii=False), flush=True))
    items = plan(ids)
    results: list[dict] = []
    consec = 0
    stop_reason = None
    for item in items:
        r = fetch_fn(item)
        results.append(r)
        emit({"event": "request", **r})
        if r["code"] in STOP_CODES:
            stop_reason = f"HTTP {r['code']}（{item['host']}）"
            break
        consec = 0 if (r["code"] == 200 and r["marker_ok"]) else consec + 1
        if consec >= CONSECUTIVE_FAIL_LIMIT:
            stop_reason = f"{CONSECUTIVE_FAIL_LIMIT} 回連続で失敗"
            break
        if item is not items[-1]:
            sleep_fn(random.uniform(min_sec, max_sec))
    j = judge(results, len(items), stop_reason)
    return j


def main() -> int:
    ids = [l.strip() for l in (HERE / "race_ids.txt").read_text().splitlines() if l.strip()]
    n = int(os.environ.get("SMOKE_N", "30"))
    slot = os.environ.get("SMOKE_SLOT", "")
    ids = ids[:n]
    t0 = time.time()
    ip = egress_ip()
    print(json.dumps({"event": "start", "host": socket.gethostname(), "egress_ip": ip, "slot": slot, "planned": len(ids)}, ensure_ascii=False), flush=True)
    j = run(ids, min_sec=float(os.environ.get("SMOKE_MIN_SEC", "4.5")), max_sec=float(os.environ.get("SMOKE_MAX_SEC", "7.0")))
    print(json.dumps({"event": "summary", "egress_ip": ip, "slot": slot, "elapsed_s": round(time.time() - t0), **j}, ensure_ascii=False), flush=True)
    return 0 if j["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

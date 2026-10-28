#!/usr/bin/env python3
"""VPS のアクセス集計の送信（systemd タイマーが 15 分おき）。標準ライブラリだけ（VPS にはリポジトリが無い）。root で動かす（nginx のログを読むため。読み取りだけ）。

nginx のアクセスログを ``vps_traffic_probe.py`` で日別に集計し、長期の履歴を ``/var/lib/keiba-ui/traffic/days/`` に積んで（ログは約 14 日で消える）、
直近 90 日分を監視ポータルへ送る。生の IP・UA は VPS の外に出ない（訪問者は日ごとの ``sha256`` の個数に畳んだ値だけ）。
送り先は監視ポータル（keiba-monitor）。**keiba-api には送らない**（インスタンス課金で、15 分以内の ping は月 約 7,800 円になる。docs/operations/cost_recalc_20261028.md）。
設定: /etc/keiba-ui/heartbeat.env の TRAFFIC_URL（無ければ HEARTBEAT_URL の /ops/heartbeat を /ops/traffic に読み替える）と KEIBA_HEARTBEAT_SECRET（ハートビートと同じ）。
    python3 traffic_client.py --print    # 集計だけして要約を表示（送らない）
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import heartbeat_client as hb  # noqa: E402  read_env / sign / encode を使い回す
import vps_traffic_probe as probe  # noqa: E402

STORE = os.environ.get("KEIBA_TRAFFIC_STORE", "/var/lib/keiba-ui/traffic")
KEEP_DAYS = 130
SEND_DAYS = 90
FULL_DAYS = 14            # 直近はページ・流入元まで送り、それより古い日は合計だけ（送信サイズを抑える）
MAX_BODY = 240 * 1024
SLIM_KEYS = ("requests", "page_views", "rsc", "api", "assets", "visitors", "internal", "status", "bot_kind")
H_TS, H_SIG = "X-Keiba-Heartbeat-Ts", "X-Keiba-Heartbeat-Sig"


def merge(days: dict, store: str = STORE) -> int:
    """日別の集計を保存する。ログの端で欠けて件数が減る上書きはしない。書いた日数を返す。"""
    n = 0
    os.makedirs(os.path.join(store, "days"), exist_ok=True)
    for day, d in days.items():
        path = os.path.join(store, "days", f"{day}.json")
        try:
            with open(path, encoding="utf-8") as f:
                if (json.load(f).get("requests") or 0) > (d.get("requests") or 0):
                    continue
        except (OSError, ValueError):
            pass
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        os.replace(tmp, path)
        n += 1
    return n


def load_store(store: str = STORE, today: str | None = None) -> dict:
    """保存済みの日別を読み、古い日（KEEP_DAYS 超）は消す。"""
    out: dict = {}
    d = os.path.join(store, "days")
    if not os.path.isdir(d):
        return out
    cutoff = time.strftime("%Y%m%d", time.localtime(time.time() - KEEP_DAYS * 86400))
    for name in sorted(os.listdir(d)):
        day = name[:-5]
        if not name.endswith(".json") or len(day) != 8:
            continue
        if day < cutoff:
            try:
                os.remove(os.path.join(d, name))
            except OSError:
                pass
            continue
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                out[day] = json.load(f)
        except (OSError, ValueError):
            continue
    return out


def build_bundle(res: dict, stored: dict, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    days = sorted(stored)[-SEND_DAYS:]
    full = set(days[-FULL_DAYS:])
    out_days = {}
    for day in days:
        d = stored[day]
        out_days[day] = d if day in full else {k: d[k] for k in SLIM_KEYS if k in d}
    return {"v": 1, "sent_at": int(now), "readable": bool(res.get("readable")), "reason": res.get("reason"), "files": len(res.get("files") or []),
            "lines": res.get("lines") or 0, "windows": res.get("windows") or {}, "days": out_days}


def endpoint(env: dict) -> str:
    url = env.get("TRAFFIC_URL", "")
    if not url and env.get("HEARTBEAT_URL"):
        url = env["HEARTBEAT_URL"].replace("/ops/heartbeat", "/ops/traffic")
    return url


def send(url: str, secret: str, body: bytes, timeout: float = 30.0) -> int:
    import urllib.request

    ts = int(time.time())
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", H_TS: str(ts), H_SIG: hb.sign(secret, ts, body), "User-Agent": "keiba-vps-traffic/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.status)


def main(argv: list[str]) -> int:
    res = probe.run()
    if res.get("readable"):
        merge(res["days"])
    bundle = build_bundle(res, load_store())
    body = hb.encode(bundle)
    if "--print" in argv:
        print(json.dumps({"readable": bundle["readable"], "reason": bundle["reason"], "days": len(bundle["days"]), "bytes": len(body),
                          "windows": bundle["windows"]}, ensure_ascii=False))
        return 0
    if len(body) > MAX_BODY:  # 想定外に大きいときは古い日の詳細を落として送る
        for day in sorted(bundle["days"])[:-7]:
            bundle["days"][day] = {k: bundle["days"][day][k] for k in SLIM_KEYS if k in bundle["days"][day]}
        body = hb.encode(bundle)
    env = hb.read_env(hb.ENV_FILE)
    url, secret = endpoint(env), env.get("KEIBA_HEARTBEAT_SECRET", "")
    if not url or not secret:
        print(f"{hb.ENV_FILE} に TRAFFIC_URL（または HEARTBEAT_URL）と KEIBA_HEARTBEAT_SECRET が必要です（値は表示しません）", file=sys.stderr)
        return 2
    for attempt in (1, 2):
        try:
            code = send(url, secret, body)
            if 200 <= code < 300:
                return 0
            print(f"HTTP {code}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001  秘密は例外文に入らない
            print(f"送信失敗({attempt}): {type(exc).__name__}", file=sys.stderr)
        time.sleep(3)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

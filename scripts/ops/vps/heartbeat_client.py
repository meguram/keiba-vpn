#!/usr/bin/env python3
"""VPS のハートビート送信（systemd タイマーが 5 分おき）。標準ライブラリだけ（VPS にはリポジトリが無い）。

採取: MemAvailable・swap・ディスク・load・uptime・systemd ユニットの状態/再起動回数/メモリ・UI と上流 API の HTTP 確認。
形式と閾値の評価は GCP 側（src/ops/heartbeat_schema.py）。署名は HMAC-SHA256(secret, f"{ts}.{body}")。
設定: /etc/keiba-ui/heartbeat.env（権限 600）の HEARTBEAT_URL と KEIBA_HEARTBEAT_SECRET。
    python3 heartbeat_client.py --print    # 採取だけして JSON を表示（送らない）
VPS は Slack の秘密を持たない: 通知は GCP 側が行う。送信に失敗しても exit 1 になるだけで UI には影響しない。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ENV_FILE = os.environ.get("HEARTBEAT_ENV_FILE", "/etc/keiba-ui/heartbeat.env")
VARS_FILE = os.environ.get("KEIBA_VARS_FILE", "/etc/keiba-ui/vars.env")
UNITS = ("nginx", "cloudflared-prod", "keiba-ui", "keiba-fetch-agent")
SCHEMA_VERSION = 1
STG_ROOT, STG_PROXY_PORT = "/srv/keiba-ui-stg", "8081"


def read_env(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def meminfo() -> dict[str, float]:
    d: dict[str, float] = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            k, _, rest = line.partition(":")
            d[k] = float(rest.split()[0]) / 1024.0  # kB → MB
    return d


def unit_state(name: str) -> dict:
    try:
        out = subprocess.run(["systemctl", "show", name, "-p", "ActiveState", "-p", "NRestarts", "-p", "MemoryCurrent"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {"active": "unknown", "restarts": 0, "mem_mb": None}
    kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    mem = kv.get("MemoryCurrent", "")
    return {"active": kv.get("ActiveState", "unknown"), "restarts": int(kv["NRestarts"]) if kv.get("NRestarts", "").isdigit() else 0,
            "mem_mb": round(int(mem) / 1048576, 1) if mem.isdigit() else None}


def http_code(url: str, timeout: float = 8.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return int(r.status)
    except urllib.error.HTTPError as e:
        return int(e.code)
    except Exception:  # noqa: BLE001
        return 0


def collect(vars_env: dict[str, str]) -> dict:
    mi = meminfo()
    total, avail = mi.get("MemTotal", 0.0), mi.get("MemAvailable", 0.0)
    st = os.statvfs("/")
    disk_pct = round(100.0 * (1 - st.f_bavail / st.f_blocks), 1) if st.f_blocks else None
    load = os.getloadavg()
    app_port, proxy_port = vars_env.get("APP_PORT", "3000"), vars_env.get("PROXY_PORT", "8080")
    app_root = vars_env.get("APP_ROOT", "/srv/keiba-ui")
    try:
        release = os.path.basename(os.path.realpath(os.path.join(app_root, "current")))
    except OSError:
        release = ""
    with open("/proc/uptime", encoding="utf-8") as f:
        uptime = float(f.read().split()[0])
    # nginx 経由でも画面の静的ファイルまでしか見ない。/api/ は Cloud Run(keiba-api)に転送されるので、5 分おきに叩かない
    # （API は『CPU を常に割り当て』のインスタンス課金で、15 分以内の定期リクエストは月 約 7,800 円になる。docs/operations/cost_recalc_20261028.md）
    checks = {"nextjs_http": http_code(f"http://127.0.0.1:{app_port}/version.json"),
              "nginx_http": http_code(f"http://127.0.0.1:{proxy_port}/version.json")}
    units = list(UNITS)
    if os.path.isdir(STG_ROOT):  # stg の画面（2 つ目のインスタンス）を入れているときだけ見る
        units.append("keiba-ui-stg")
        checks["stg_ui_http"] = http_code(f"http://127.0.0.1:{STG_PROXY_PORT}/version.json")
    return {
        "v": SCHEMA_VERSION, "host": "vps", "sent_at": int(time.time()), "release": release[:80],
        "mem": {"total_mb": round(total, 1), "available_mb": round(avail, 1), "used_mb": round(total - avail, 1)},
        "swap": {"total_mb": round(mi.get("SwapTotal", 0.0), 1), "used_mb": round(mi.get("SwapTotal", 0.0) - mi.get("SwapFree", 0.0), 1)},
        "disk": {"root_used_pct": disk_pct},
        "load": {"l1": round(load[0], 2), "l5": round(load[1], 2)},
        "uptime_s": round(uptime),
        "units": {n: unit_state(n) for n in units},
        "checks": checks,
    }


def encode(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign(secret: str, ts: int, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()


def send(url: str, secret: str, body: bytes, timeout: float = 15.0) -> int:
    ts = int(time.time())
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", "X-Keiba-Heartbeat-Ts": str(ts), "X-Keiba-Heartbeat-Sig": sign(secret, ts, body),
        "User-Agent": "keiba-vps-heartbeat/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.status)


def main(argv: list[str]) -> int:
    payload = collect(read_env(VARS_FILE))
    if "--print" in argv:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0
    env = read_env(ENV_FILE)
    url, secret = env.get("HEARTBEAT_URL", ""), env.get("KEIBA_HEARTBEAT_SECRET", "")
    if not url or not secret:
        print(f"{ENV_FILE} に HEARTBEAT_URL と KEIBA_HEARTBEAT_SECRET が必要です（値は表示しません）", file=sys.stderr)
        return 2
    body = encode(payload)
    for attempt in (1, 2):
        try:
            code = send(url, secret, body)
            if 200 <= code < 300:
                return 0
            print(f"HTTP {code}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001  秘密は例外文に入らない（ヘッダ値は文に出ない）
            print(f"送信失敗({attempt}): {type(exc).__name__}", file=sys.stderr)
        time.sleep(3)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

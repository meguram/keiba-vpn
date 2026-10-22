#!/usr/bin/env python3
"""VPS の状況を 1 回採取して JSON を標準出力へ出す（標準ライブラリのみ・読み取りだけ・何も変更しない）。

ホスト機の収集スクリプト（collect_prod_status.py）が ``ssh keiba-vps python3 - < このファイル`` で流す。
VPS に何も導入しない（新しい権限・ファイルは要らない）。deploy ユーザーで読める範囲だけを見て、見えないものは null にする。
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

APP_PORT = os.environ.get("APP_PORT", "3000")
PROXY_PORT = os.environ.get("PROXY_PORT", "8080")
UNITS = ["keiba-ui", "nginx", "cloudflared-prod", "keiba-fetch-agent", "keiba-ui-health.timer", "keiba-heartbeat.timer",
         "keiba-release-pull-ui.timer", "keiba-release-pull-fetch.timer"]


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:  # noqa: BLE001
        return ""


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def meminfo() -> dict:
    kv: dict[str, float] = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        parts = line.replace(":", "").split()
        if len(parts) >= 2:
            kv[parts[0]] = float(parts[1]) / 1024.0
    total, avail = kv.get("MemTotal"), kv.get("MemAvailable")
    return {
        "total_mb": total, "available_mb": avail, "used_mb": (total - avail) if total is not None and avail is not None else None,
        "swap_total_mb": kv.get("SwapTotal"), "swap_used_mb": (kv["SwapTotal"] - kv["SwapFree"]) if "SwapTotal" in kv and "SwapFree" in kv else None,
    }


def unit(name: str) -> dict:
    out = _run(["systemctl", "show", name, "-p", "LoadState", "-p", "ActiveState", "-p", "SubState", "-p", "NRestarts",
                "-p", "MemoryCurrent", "-p", "ActiveEnterTimestamp"])
    kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    if kv.get("LoadState") in (None, "not-found"):
        return {"installed": False}

    def num(v: str | None):
        try:
            return float(v) if v not in (None, "", "[not set]") and int(float(v)) < 2**62 else None
        except ValueError:
            return None

    mem = num(kv.get("MemoryCurrent"))
    return {"installed": True, "active": kv.get("ActiveState"), "sub": kv.get("SubState"), "restarts": int(num(kv.get("NRestarts")) or 0),
            "memory_mb": round(mem / 1048576, 1) if mem is not None else None, "since": kv.get("ActiveEnterTimestamp") or None}


def http(url: str, timeout: float = 8.0, want_body: bool = False) -> dict:
    t = time.time()
    try:
        # healthcheck.sh(curl) と同じ見え方にする（python-urllib の UA は bot として弾かれて 403 になる）
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.81.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(64 * 1024).decode("utf-8", "replace")
            res = {"code": r.status, "ms": round((time.time() - t) * 1000)}
    except urllib.error.HTTPError as e:
        return {"code": e.code, "ms": round((time.time() - t) * 1000)}
    except Exception as e:  # noqa: BLE001
        return {"code": None, "error": type(e).__name__, "ms": round((time.time() - t) * 1000)}
    if want_body:
        try:
            res["json"] = json.loads(body)
        except ValueError:
            res["json"] = None
    return res


def tunnel_connections() -> dict:
    out = _run(["journalctl", "-u", "cloudflared-prod", "-n", "300", "--no-pager", "-o", "cat"], 10.0)
    if not out.strip():
        return {"readable": False}
    lines = out.splitlines()
    reg = [i for i, ln in enumerate(lines) if "Registered tunnel connection" in ln]
    lost = [i for i, ln in enumerate(lines) if "Unregistered tunnel connection" in ln or "Lost connection" in ln]
    return {"readable": True, "registered": len(reg), "recently_lost": bool(lost) and (not reg or max(lost) > max(reg))}


def main() -> None:
    disk = _run(["df", "--output=pcent,avail", "-B1M", "/"]).splitlines()
    disk_pct = disk_avail = None
    if len(disk) >= 2:
        parts = disk[1].split()
        try:
            disk_pct, disk_avail = int(parts[0].strip("%")), int(parts[1])
        except (ValueError, IndexError):
            pass
    up = (_read("/proc/uptime") or "0").split()[0]
    load = (_read("/proc/loadavg") or "").split()[:3]
    root = "/srv/keiba-ui"
    health = None
    try:
        health = json.loads(_read("/var/lib/keiba-ui/health.json") or "null")
    except ValueError:
        health = None
    out = {
        "probe_version": 1, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": os.uname().nodename,
        "mem": meminfo(), "load": [float(x) for x in load] if load else None, "uptime_s": float(up),
        "disk": {"used_pct": disk_pct, "avail_mb": disk_avail},
        "units": {u: unit(u) for u in UNITS},
        "ui": http(f"http://127.0.0.1:{APP_PORT}/version.json", want_body=True),
        "upstream_api": http(f"http://127.0.0.1:{PROXY_PORT}/api/maintenance/status", timeout=12.0),
        "current_release": os.path.basename(os.path.realpath(f"{root}/current")) if os.path.exists(f"{root}/current") else None,
        "releases": sorted(os.listdir(f"{root}/releases"))[-5:] if os.path.isdir(f"{root}/releases") else None,
        "health_timer_state": health,
        "tunnel": tunnel_connections(),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

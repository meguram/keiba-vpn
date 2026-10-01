"""各環境（学習PC / VPS）の調査スクリプトが共有する、レポートの形式と計測ヘルパ。

レポートは JSON 1 ファイル。**秘密値は含めない**（環境変数は「設定の有無」だけ記録する）。
``decide.py`` が複数環境のレポートを読み、設計上の判断を出す。
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STATUSES = ("ok", "warn", "ng", "skip", "error")


def new_report(role: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "metrics": {},
        "checks": [],
    }


def add_check(report: dict, check_id: str, status: str, detail: str = "", **values: Any) -> None:
    """調査項目を1件記録する。``values`` は ``decide.py`` が読む数値・事実。"""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}: {status}")
    report["checks"].append({"id": check_id, "status": status, "detail": detail, "values": values})


def run_check(report: dict, check_id: str, fn) -> None:
    """``fn(report) -> (status, detail, values)`` を実行し、例外は error として記録する（他の調査は続行）。"""
    try:
        status, detail, values = fn(report)
    except Exception as e:  # noqa: BLE001
        add_check(report, check_id, "error", f"{type(e).__name__}: {e}")
        return
    add_check(report, check_id, status, detail, **values)


def check_values(report: dict, check_id: str) -> dict:
    for c in report.get("checks", []):
        if c["id"] == check_id and c["status"] in ("ok", "warn", "ng"):
            return c.get("values") or {}
    return {}


def save_report(report: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


def load_report(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"レポート形式が違います: {data.get('schema_version')} (期待 {SCHEMA_VERSION})")
    return data


def print_summary(report: dict) -> None:
    mark = {"ok": "OK  ", "warn": "WARN", "ng": "NG  ", "skip": "SKIP", "error": "ERR "}
    print(f"=== 調査結果 role={report['role']} host={report['host']} ===")
    for c in report["checks"]:
        print(f"[{mark[c['status']]}] {c['id']}: {c['detail']}")


# ── 計測ヘルパ ────────────────────────────────────────


def memory_info() -> dict:
    """搭載メモリ・空きメモリ・スワップ（MB）。取得できない項目は None。"""
    try:
        import psutil

        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        return {
            "mem_total_mb": round(vm.total / 2**20),
            "mem_available_mb": round(vm.available / 2**20),
            "swap_total_mb": round(sw.total / 2**20),
        }
    except ImportError:
        pass
    info: dict[str, int | None] = {"mem_total_mb": None, "mem_available_mb": None, "swap_total_mb": None}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            kb = int(rest.split()[0])
            if key == "MemTotal":
                info["mem_total_mb"] = kb // 1024
            elif key == "MemAvailable":
                info["mem_available_mb"] = kb // 1024
            elif key == "SwapTotal":
                info["swap_total_mb"] = kb // 1024
    except OSError:
        pass
    return info


def current_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t


def env_presence(names: list[str]) -> dict[str, bool]:
    """環境変数が設定されているか（値は記録しない）。"""
    return {n: bool((os.environ.get(n) or "").strip()) for n in names}


def library_versions() -> dict[str, str]:
    from src.pipeline.models.model_registry import library_versions as _lv

    return _lv()


def run_inference_probe(model_dir: str | Path, rows: int = 18, timeout: int = 600) -> dict:
    """``inference_probe`` を別プロセスで実行して結果（dict）を返す。"""
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "src.scripts.diagnose.inference_probe", "--model-dir", str(model_dir), "--rows", str(rows)],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip()[-500:])
    return json.loads(proc.stdout.strip().splitlines()[-1])

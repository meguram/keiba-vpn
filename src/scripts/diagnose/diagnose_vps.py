"""【VPS で実行】メモリ余裕・外部接続の遅延・スクレイピング可否・推論メモリを調査してレポート(JSON)を出す。

目的: 「実機でしか分からないこと」を集め、``decide.py``（開発PC）が判断できるようにする。
  - メモリ（MemAvailable/スワップ）・CPU・ディスク、常駐プロセスのメモリ上位、Redis のメモリ
  - GCS / Cloud SQL への遅延（ページ表示・集約オブジェクト設計の判断材料）
  - 推論のメモリ・時間（別プロセスで実測。モデルは --model-dir か、疑似アンサンブルを自動生成）
  - netkeiba にVPSのIPからアクセスできるか（**--netkeiba-trial を付けたときだけ、1リクエスト**）

使い方（VPS上のリポジトリルート、サービス稼働中の状態で。読み取りのみ）:
  KEIBA_ENV=prod python -m src.scripts.diagnose.diagnose_vps --out reports/vps.json
  KEIBA_ENV=prod python -m src.scripts.diagnose.diagnose_vps --model-dir /path/to/model --netkeiba-trial
サービス稼働中に測るのが重要（空きメモリは「稼働中の余裕」を見るため）。開催日の最繁忙時間帯にも1回測るとよい。
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.scripts.diagnose.common import (
    add_check,
    env_presence,
    library_versions,
    memory_info,
    new_report,
    print_summary,
    run_check,
    run_inference_probe,
    save_report,
)


def _ms(samples: list[float]) -> dict:
    s = sorted(samples)
    return {
        "median_ms": round(statistics.median(s) * 1000),
        "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))] * 1000),
        "n": len(s),
    }


def check_system(report):
    mem = memory_info()
    disk = shutil.disk_usage(".")
    load1 = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    libs = library_versions()
    status = "ok" if (mem["mem_available_mb"] or 0) >= 400 else "warn"
    return status, (
        f"メモリ 空き{mem['mem_available_mb']}/{mem['mem_total_mb']}MB, スワップ{mem['swap_total_mb']}MB, "
        f"CPU{os.cpu_count()}, load1={load1}, ディスク空き{disk.free / 2**30:.0f}GB"
    ), {
        **mem, "cpu_count": os.cpu_count(), "load1": load1, "disk_free_gb": round(disk.free / 2**30, 1),
        "libraries": libs,
        "env_configured": env_presence(["GCS_BUCKET", "DATABASE_URL", "REDIS_URL", "KEIBA_MODEL_STORE", "KEIBA_ENV"]),
    }


def check_processes(report):
    try:
        import psutil
    except ImportError:
        return "skip", "psutil 未インストール", {}
    procs = []
    for p in psutil.process_iter(["name", "memory_info", "cmdline"]):
        try:
            procs.append((p.info["memory_info"].rss / 2**20, p.info["name"], " ".join(p.info["cmdline"] or [])[:60]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    procs.sort(reverse=True)
    top = [{"rss_mb": round(r), "name": n, "cmd": c} for r, n, c in procs[:8]]
    return "ok", "RSS上位: " + ", ".join(f"{t['name']}={t['rss_mb']}MB" for t in top[:4]), {"top": top}


def check_redis(report):
    try:
        import redis
    except ImportError:
        return "skip", "redis パッケージ未インストール", {}

    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    r = redis.Redis.from_url(url, socket_timeout=3)
    try:
        info = r.info("memory")
    except redis.exceptions.RedisError as e:
        return "warn", f"Redisに接続できません: {type(e).__name__}", {"connected": False}
    used = round(info["used_memory"] / 2**20, 1)
    t = time.perf_counter()
    r.ping()
    return "ok", f"Redis使用{used}MB maxmemory={info.get('maxmemory_human')}", {
        "used_mb": used, "maxmemory_mb": round(info.get("maxmemory", 0) / 2**20), "ping_ms": round((time.perf_counter() - t) * 1000, 1),
    }


def check_gcs_latency(report, args):
    from src.scraper.storage import HybridStorage

    storage = HybridStorage(".")
    if not storage.gcs_enabled:
        return "skip", "GCS未接続（GCS_* 認証情報と GCS_BUCKET を確認）", {"connected": False}
    bucket = storage._get_bucket()
    t = time.perf_counter()
    blobs = [b for b in bucket.list_blobs(max_results=200) if 0 < (b.size or 0) < 300_000][: args.gcs_objects]
    list_s = time.perf_counter() - t
    if not blobs:
        return "warn", "小さなオブジェクトが見つかりません", {"connected": True, "list_ms": round(list_s * 1000)}
    samples, sizes = [], []
    for b in blobs:
        t = time.perf_counter()
        data = b.download_as_bytes()
        samples.append(time.perf_counter() - t)
        sizes.append(len(data))
    st = _ms(samples)
    return ("ok" if st["median_ms"] < 150 else "warn"), (
        f"GCS 1オブジェクト取得 中央{st['median_ms']}ms / p95 {st['p95_ms']}ms（中央{statistics.median(sizes) / 1024:.0f}KB）"
    ), {"connected": True, "list_ms": round(list_s * 1000), "median_object_kb": round(statistics.median(sizes) / 1024), **st}


def check_cloud_sql_latency(report, args):
    from sqlalchemy import text

    from src.db.session import get_session_factory, init_engine

    from sqlalchemy.exc import OperationalError

    init_engine()
    samples = []
    try:
        with get_session_factory()() as s:
            for _ in range(args.db_queries):
                t = time.perf_counter()
                s.execute(text("SELECT 1"))
                samples.append(time.perf_counter() - t)
    except OperationalError as e:
        return "warn", f"DBに接続できません: {str(e.orig)[:80] if e.orig else type(e).__name__}", {"connected": False}
    st = _ms(samples)
    return ("ok" if st["median_ms"] < 50 else "warn"), f"DB SELECT 1 中央{st['median_ms']}ms / p95 {st['p95_ms']}ms", {"connected": True, **st}


def check_netkeiba(report, args):
    """VPSのIPから netkeiba に**1回だけ**アクセスして、ブロックされていないかを見る。"""
    if not args.netkeiba_trial:
        return "skip", "--netkeiba-trial 未指定（スクレイピング可否は未確認）", {"tried": False}
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        args.netkeiba_url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"},
    )
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            code, body = resp.status, resp.read(200_000)
    except urllib.error.HTTPError as e:
        code, body = e.code, b""
    except Exception as e:  # noqa: BLE001
        return "ng", f"接続失敗: {type(e).__name__}: {e}", {"tried": True, "status_code": None, "blocked": True}
    sec = time.perf_counter() - t
    blocked = code in (403, 429, 503) or b"captcha" in body.lower()
    return ("ng" if blocked else "ok"), f"HTTP {code} / {sec:.1f}s / {'ブロックの疑い' if blocked else '取得可'}", {
        "tried": True, "status_code": code, "blocked": blocked, "sec": round(sec, 2), "body_bytes": len(body),
    }


def check_inference(report, args):
    model_dir = Path(args.model_dir) if args.model_dir else None
    tmp = None
    if model_dir is None or not model_dir.is_dir():
        tmp = tempfile.TemporaryDirectory()
        model_dir = Path(tmp.name)
        code = (
            "from src.scripts.maintenance.make_pseudo_ensemble import train_pseudo_ensemble;"
            f"train_pseudo_ensemble(r'{model_dir}', {args.pseudo_features})"
        )
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=900)
        label = f"疑似アンサンブル({args.pseudo_features}特徴量)"
    else:
        label = f"モデル {model_dir}"
    try:
        probe = run_inference_probe(model_dir, rows=args.probe_rows)
    finally:
        if tmp:
            tmp.cleanup()
    mem = memory_info()
    return "ok", (
        f"{label}: 推論ピークRSS {probe['peak_rss_mb']}MB / ロード{probe['load_sec']}s / 予測{probe['predict_sec_median']}s "
        f"(現在の空き{mem['mem_available_mb']}MB)"
    ), {**probe, "model_label": label, "mem_available_at_probe_mb": mem["mem_available_mb"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/vps.json")
    ap.add_argument("--model-dir", default="", help="学習済みモデルのディレクトリ（無ければ疑似アンサンブルを生成して測る）")
    ap.add_argument("--pseudo-features", type=int, default=1000)
    ap.add_argument("--probe-rows", type=int, default=18)
    ap.add_argument("--gcs-objects", type=int, default=10)
    ap.add_argument("--db-queries", type=int, default=20)
    ap.add_argument("--netkeiba-trial", action="store_true", help="netkeiba へ1回だけアクセスしてブロック有無を確認する")
    ap.add_argument("--netkeiba-url", default="https://race.netkeiba.com/top/")
    args = ap.parse_args(argv)

    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    report = new_report("vps")
    run_check(report, "system", check_system)
    run_check(report, "processes", check_processes)
    run_check(report, "redis", check_redis)
    run_check(report, "gcs_latency", lambda r: check_gcs_latency(r, args))
    run_check(report, "cloud_sql_latency", lambda r: check_cloud_sql_latency(r, args))
    run_check(report, "netkeiba", lambda r: check_netkeiba(r, args))
    run_check(report, "inference", lambda r: check_inference(r, args))

    print_summary(report)
    print(f"\nレポート: {save_report(report, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

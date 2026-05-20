"""
既存 JSON の sex フィールド補完スクリプト
==========================================
data/local/horse_pedigree_5gen に保存済みの JSON のうち、
sex フィールドが空（未設定）のファイルに対してのみ血統ページを再取得し、
性別情報（牡 / 牝）を書き込む。

【設計】
  - scrape_ancestors_upward.py と同じ「永続ワーカー方式」を採用
  - NetkeibaClient は 1ワーカー1インスタンス（再ログイン不要）
  - キューからファイルを受け取り、ループで処理し続ける

Usage:
  python scripts/scraping/patch_sex.py
  python scripts/scraping/patch_sex.py --workers 4
  python scripts/scraping/patch_sex.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from typing import NamedTuple
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.research.pedigree.pedigree_similarity import parse_horse_sex_from_ped_html  # noqa: E402
from src.scraper.client import NetkeibaClient                                # noqa: E402

logger = logging.getLogger("patch_sex")

PED_DIR     = _ROOT / "data" / "local" / "horse_pedigree_5gen"
STATUS_PATH = _ROOT / "data" / "local" / "meta" / "patch_sex_status.json"
PED_URL     = "https://db.netkeiba.com/horse/ped/{horse_id}/"

# HTTP エラーによる停止判定（ancestors_upward と共通ロジック）
_HTTP_404_STREAK_LIMIT = 5


def _is_fatal_http_error(err: str) -> tuple[bool, str]:
    if "400 Client Error" in err or ("400" in err and "netkeiba" in err.lower()):
        return True, f"HTTP 400 (IPブロック疑い): {err}"
    for token in ("403", "429", "503", "Forbidden", "Too Many",
                  "Service Unavailable", "ConnectionError", "Network is unreachable"):
        if token in err:
            return True, f"{token}: {err}"
    return False, ""


# ── ステータス書き込み ────────────────────────────────────────────
def _write_status(done: int, total: int, updated: int, failed: int,
                  t0: float, *, running: bool) -> None:
    elapsed = time.time() - t0
    rate = done / elapsed * 3600 if elapsed > 0 else 0
    remaining = max(total - done, 0)
    eta_sec = remaining / (rate / 3600) if rate > 0 else -1
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps({
            "running": running,
            "done": done,
            "total": total,
            "updated": updated,
            "failed": failed,
            "elapsed_sec": round(elapsed, 1),
            "rate_per_hour": round(rate, 1),
            "eta_sec": round(eta_sec, 1),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.debug("status write error: %s", e)


# ── ワーカースレッド ──────────────────────────────────────────────
def _worker(
    worker_id: int,
    task_queue: queue.Queue,
    counters: dict,
    lock: threading.Lock,
    stop_event: threading.Event,
    t0: float,
    total: int,
    *,
    dry_run: bool,
) -> None:
    """1ワーカー = 1 NetkeibaClient。キューが空になるまでループ処理。"""
    client = NetkeibaClient()
    logger.info("[worker-%d] 起動・ログイン完了", worker_id)

    while not stop_event.is_set():
        try:
            path: Path = task_queue.get(timeout=2)
        except queue.Empty:
            break

        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[worker-%d] read error %s: %s", worker_id, path, e)
            task_queue.task_done()
            with lock:
                counters["done"] += 1
                counters["failed"] += 1
            continue

        horse_id = str(rec.get("horse_id") or "").strip()
        if not horse_id:
            task_queue.task_done()
            with lock:
                counters["done"] += 1
            continue

        # sex が既に設定済みならスキップ
        if str(rec.get("sex") or "").strip():
            task_queue.task_done()
            with lock:
                counters["done"] += 1
            continue

        url = PED_URL.format(horse_id=horse_id)
        try:
            html = client.fetch(url)
        except Exception as e:
            err = str(e)

            # HTTP 404 → failed に追加し連続カウント、上限超えで全停止
            if "404 Client Error" in err or ("404" in err and "netkeiba" in err.lower()):
                logger.warning("[worker-%d] HTTP 404 %s: %s", worker_id, horse_id, err)
                with lock:
                    counters["done"] += 1
                    counters["failed"] += 1
                    counters["http404_streak"] = counters.get("http404_streak", 0) + 1
                    streak = counters["http404_streak"]
                task_queue.task_done()
                if streak >= _HTTP_404_STREAK_LIMIT:
                    reason = f"HTTP 404 が {_HTTP_404_STREAK_LIMIT} 件連続 → IPブロック疑い"
                    logger.error("[worker-%d] ★ %s → 全停止", worker_id, reason)
                    stop_event.set()
                    break
                continue

            # HTTP 400 / 403 / 429 / 503 etc. → 即停止
            fatal, reason = _is_fatal_http_error(err)
            if fatal:
                logger.error("[worker-%d] ★ アクセスエラー → 全停止: %s", worker_id, reason)
                stop_event.set()
                task_queue.task_done()
                break

            logger.warning("[worker-%d] fetch error %s: %s", worker_id, horse_id, err)
            task_queue.task_done()
            with lock:
                counters["done"] += 1
                counters["failed"] += 1
            continue

        # 正常取得 → HTTP 404 連続カウントをリセット
        with lock:
            counters["http404_streak"] = 0

        sex = parse_horse_sex_from_ped_html(html, horse_id)

        if not dry_run:
            rec["sex"] = sex
            try:
                path.write_text(
                    json.dumps(rec, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.warning("[worker-%d] write error %s: %s", worker_id, path, e)

        task_queue.task_done()
        with lock:
            counters["done"] += 1
            counters["updated"] += 1
            done = counters["done"]

        # 100件ごとにステータス更新・ログ出力
        if done % 100 == 0:
            with lock:
                d = counters["done"]
                u = counters["updated"]
                f = counters["failed"]
            elapsed = time.time() - t0
            rate = d / elapsed * 3600 if elapsed > 0 else 0
            eta = max(total - d, 0) / (rate / 3600) if rate > 0 else -1
            logger.info(
                "[worker-%d] %d / %d  更新:%d  失敗:%d  速度:%.0f件/h  ETA:%.0f分",
                worker_id, d, total, u, f, rate, eta / 60 if eta > 0 else -1,
            )
            _write_status(d, total, u, f, t0, running=True)

    logger.info("[worker-%d] 終了", worker_id)


# ── メイン ───────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="sex フィールド補完")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="sex が空のファイルだけでなく全ファイルを対象にする")
    args = ap.parse_args()

    # ルートロガーは WARNING 以上のみ（NetkeibaClient の GET/クールダウン行を抑制）
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # patch_sex 自身の進捗ログは INFO まで表示
    logging.getLogger("patch_sex").setLevel(logging.INFO)

    # 対象ファイル収集
    logger.info("対象ファイルを収集中...")
    task_queue: queue.Queue = queue.Queue()
    skipped = 0

    for f in sorted(PED_DIR.rglob("*.json")):
        if args.all:
            task_queue.put(f)
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            if not str(rec.get("sex") or "").strip():
                task_queue.put(f)
            else:
                skipped += 1
        except Exception:
            task_queue.put(f)

    total = task_queue.qsize()
    logger.info("対象: %d ファイル  (sex 既存でスキップ: %d)", total, skipped)

    if total == 0:
        logger.info("補完が必要なファイルはありません。")
        _write_status(0, 0, 0, 0, time.time(), running=False)
        return

    t0 = time.time()
    counters = {"done": 0, "updated": 0, "failed": 0}
    lock = threading.Lock()
    stop_event = threading.Event()

    _write_status(0, total, 0, 0, t0, running=True)

    threads = []
    for i in range(args.workers):
        t = threading.Thread(
            target=_worker,
            args=(i, task_queue, counters, lock, stop_event, t0, total),
            kwargs={"dry_run": args.dry_run},
            daemon=True,
            name=f"patch-worker-{i}",
        )
        t.start()
        threads.append(t)

    # 定期ステータス書き込み（メインスレッドから）
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(5)
            with lock:
                d = counters["done"]
                u = counters["updated"]
                f = counters["failed"]
            _write_status(d, total, u, f, t0, running=True)
    except KeyboardInterrupt:
        logger.info("中断シグナル受信 → 停止中...")
        stop_event.set()

    for t in threads:
        t.join(timeout=10)

    with lock:
        d = counters["done"]
        u = counters["updated"]
        f = counters["failed"]

    _write_status(d, total, u, f, t0, running=False)
    elapsed = time.time() - t0
    logger.info(
        "完了: %d / %d  更新:%d  失敗:%d  経過:%.1f秒",
        d, total, u, f, elapsed,
    )


if __name__ == "__main__":
    main()

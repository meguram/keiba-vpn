"""race_result に出た馬の horse_result を GCS からローカルキャッシュへ同期。"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from src.scraper.storage import HybridStorage
from src.scripts.data.ml_warehouse.sqlite_builder import horse_ids_from_race_results
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

DEFAULT_PROGRESS_PATH = Path("logs/sync_horse_result_progress.json")


def _disk_cache_usable(path: Path) -> bool:
    """既存キャッシュ JSON が有効なら GCS の read を省略する。"""
    try:
        if not path.is_file() or path.stat().st_size < 16:
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        return bool(data.get("horse_id"))
    except Exception:
        return False


def _write_local_cache_verified(storage: HybridStorage, hid: str, data: dict) -> bool:
    """horse_result を L2 に書き、実ファイルを検証する。"""
    if not storage._is_locally_cached("horse_result"):
        return False
    storage._write_local_cache("horse_result", hid, data)
    cpath = storage._local_cache_path("horse_result", hid)
    if _disk_cache_usable(cpath):
        return True
    # 1 回だけ再試行（一時 I/O 失敗など）
    storage._write_local_cache("horse_result", hid, data)
    return _disk_cache_usable(cpath)


def _write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sync_horse_results(
    base_dir: str | Path,
    years: list[str],
    *,
    limit: int = 0,
    interval: float = 0.05,
    progress_file: str | Path = DEFAULT_PROGRESS_PATH,
) -> dict[str, int]:
    base = Path(base_dir).expanduser().resolve()
    storage = HybridStorage(str(base))
    if not storage.gcs_enabled:
        raise SystemExit("GCS 未接続のため horse_result 同期不可")

    horse_ids = sorted(horse_ids_from_race_results(base, years))
    if limit > 0:
        horse_ids = horse_ids[:limit]
    stats = {
        "total": len(horse_ids),
        "ok": 0,
        "empty": 0,
        "fail": 0,
        "skipped_disk": 0,
        "cache_write_fail": 0,
    }
    progress_path = Path(progress_file)
    started = time.time()
    _write_progress(progress_path, {**stats, "done": 0, "phase": "sync", "started_at": started})

    for i, hid in enumerate(horse_ids, 1):
        cpath = storage._local_cache_path("horse_result", hid)
        if _disk_cache_usable(cpath):
            stats["ok"] += 1
            stats["skipped_disk"] += 1
        else:
            try:
                data = storage.load("horse_result", hid)
                if data and data.get("horse_id"):
                    if _write_local_cache_verified(storage, hid, data):
                        stats["ok"] += 1
                    else:
                        stats["cache_write_fail"] += 1
                        logger.warning("horse_result キャッシュ書込検証失敗: %s", hid)
                else:
                    stats["empty"] += 1
            except Exception as e:
                logger.debug("load fail %s: %s", hid, e)
                stats["fail"] += 1
        if i % 100 == 0 or i == len(horse_ids):
            logger.info(
                "[%d/%d] ok=%d empty=%d fail=%d skipped_disk=%d cache_write_fail=%d",
                i,
                len(horse_ids),
                stats["ok"],
                stats["empty"],
                stats["fail"],
                stats["skipped_disk"],
                stats["cache_write_fail"],
            )
            print(
                f"[{i}/{len(horse_ids)}] ok={stats['ok']} "
                f"empty={stats['empty']} fail={stats['fail']} "
                f"skipped_disk={stats['skipped_disk']} "
                f"cache_write_fail={stats['cache_write_fail']}",
                flush=True,
            )
            _write_progress(
                progress_path,
                {
                    **stats,
                    "done": i,
                    "phase": "sync",
                    "started_at": started,
                    "last_horse_id": hid,
                },
            )
        if interval > 0 and i < len(horse_ids):
            time.sleep(interval)
    _write_progress(progress_path, {**stats, "done": len(horse_ids), "phase": "done", "finished": True})
    return stats


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser(description="horse_result GCS→ローカルキャッシュ")
    ap.add_argument("--years", default="2020,2021,2022,2023,2024,2025")
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--limit", type=int, default=0, help="テスト用件数上限")
    ap.add_argument("--interval", type=float, default=0.05)
    args = ap.parse_args()
    years = [y.strip() for y in args.years.split(",") if y.strip()]
    st = sync_horse_results(args.base_dir, years, limit=args.limit, interval=args.interval)
    print("完了:", st)
    if st.get("cache_write_fail", 0):
        logger.error("horse_result キャッシュ書込検証失敗: %d 件", st["cache_write_fail"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
race_shutuba_flat 馬の血統補完スクリプト（10世代対応）
=======================================================
data/local/tables/*/race_shutuba_flat.parquet に登録されているすべての
horse_id を対象に、5世代血統 JSON が未取得のものをスクレイプする。

Phase ①: shutuba 馬の 5世代血統スクレイプ
  - race_shutuba_flat.parquet 全年の horse_id を収集
  - data/local/horse_pedigree_5gen/ に JSON が存在しない馬を取得
  - scrape_ancestors_upward.py と同じ「ローカル直書き」方式

Phase ②: 祖先の上向きスクレイプ（10世代化）
  - Phase ① 完了後に scrape_ancestors_upward.py を呼び出す
  - 新規 JSON に含まれる祖先 horse_id を再帰的にスクレイプ

【アクセス制御】
  - NetkeibaClient（interval=2.0 秒）
  - HTTP 400/403/429/503 → 全停止
  - HTTP 404 が 5 件連続 → 全停止

Usage:
  python scripts/scraping/scrape_shutuba_pedigree.py
  python scripts/scraping/scrape_shutuba_pedigree.py --workers 2
  python scripts/scraping/scrape_shutuba_pedigree.py --phase 1   # Phase ① のみ
  python scripts/scraping/scrape_shutuba_pedigree.py --phase 2   # Phase ② のみ
  python scripts/scraping/scrape_shutuba_pedigree.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from src.research.pedigree.pedigree_similarity import (  # noqa: E402
    parse_blood_table_5gen,
    parse_horse_sex_from_ped_html,
)
from scripts.scrape_pedigree_5gen import build_pedigree_record  # noqa: E402
from src.scraper.client import NetkeibaClient  # noqa: E402
from src.utils.keiba_logging import script_basic_config  # noqa: E402

logger = logging.getLogger("scrape_shutuba_pedigree")

TABLES_DIR     = _ROOT / "data" / "local" / "tables"
LOCAL_PED_DIR  = _ROOT / "data" / "local" / "horse_pedigree_5gen"
STATUS_PATH    = _ROOT / "data" / "local" / "meta" / "shutuba_pedigree_status.json"
MISSING_PATH   = _ROOT / "data" / "local" / "meta" / "shutuba_missing_pedigree.json"
PED_URL        = "https://db.netkeiba.com/horse/ped/{horse_id}/"

_HTTP_404_STREAK_LIMIT = 5


def local_path(horse_id: str) -> Path:
    sub = horse_id[:4] if len(horse_id) >= 4 else "_"
    return LOCAL_PED_DIR / sub / f"{horse_id}.json"


# ── shutuba horse_id 収集 ──────────────────────────────────────
def collect_shutuba_horse_ids() -> set[str]:
    """全年の race_shutuba_flat.parquet から horse_id を収集。"""
    ids: set[str] = set()
    found_files: list[Path] = []
    for p in sorted(TABLES_DIR.glob("*/race_shutuba_flat.parquet")):
        found_files.append(p)
        try:
            df = pd.read_parquet(p, columns=["horse_id"])
            batch = set(df["horse_id"].dropna().astype(str).unique())
            ids |= batch
        except Exception as e:
            logger.warning("スキップ（読込失敗）: %s — %s", p, e)
    logger.info("shutuba parquet: %d ファイル → %d ユニーク horse_id", len(found_files), len(ids))
    return ids


def collect_existing_ids() -> set[str]:
    """data/local/horse_pedigree_5gen 以下の JSON 一覧。"""
    ids = {f.stem for f in LOCAL_PED_DIR.rglob("*.json")}
    logger.info("既存 JSON: %d ファイル", len(ids))
    return ids


# ── HTTP エラー判定 ───────────────────────────────────────────
def _is_fatal_http_error(err: str) -> tuple[bool, str]:
    if "400 Client Error" in err or ("400" in err and "netkeiba" in err.lower()):
        return True, f"HTTP 400 (IPブロック疑い): {err}"
    for token in ("403", "429", "503", "Forbidden", "Too Many",
                  "Service Unavailable", "ConnectionError", "Network is unreachable"):
        if token in err:
            return True, f"{token}: {err}"
    return False, ""


# ── 共有状態 ─────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock           = threading.Lock()
        self.scraped        = 0
        self.not_found      = 0
        self.error_count    = 0
        self.stop_flag      = False
        self.stop_reason    = ""
        self.http404_streak = 0

    def stop(self, reason: str) -> None:
        with self.lock:
            self.stop_flag   = True
            self.stop_reason = reason

    def record_http404(self) -> bool:
        with self.lock:
            self.http404_streak += 1
            return self.http404_streak >= _HTTP_404_STREAK_LIMIT

    def reset_404_streak(self) -> None:
        with self.lock:
            self.http404_streak = 0


# ── ステータス書き込み ─────────────────────────────────────────
def _write_status(state: SharedState, done: int, total: int, t0: float,
                  *, running: bool) -> None:
    elapsed = time.time() - t0
    rate_per_hour = state.scraped / elapsed * 3600 if elapsed > 0 else 0
    remaining = max(total - done, 0)
    eta_sec = remaining / (rate_per_hour / 3600) if rate_per_hour > 0 else -1
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps({
            "phase":          "1_shutuba_5gen",
            "done":           done,
            "total":          total,
            "scraped":        state.scraped,
            "not_found":      state.not_found,
            "error_count":    state.error_count,
            "rate_per_hour":  round(rate_per_hour, 1),
            "eta_sec":        round(eta_sec, 0) if eta_sec > 0 else -1,
            "elapsed_min":    round(elapsed / 60, 1),
            "running":        running,
            "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── ワーカー ──────────────────────────────────────────────────
def worker_fn(
    task_queue: "queue.Queue[str | None]",
    state: SharedState,
    *,
    dry_run: bool,
    worker_id: int,
    done_counter: list[int],
    total: int,
    t0: float,
) -> None:
    client = NetkeibaClient(interval=2.0, auto_login=True)
    logger.info("[worker-%d] 開始", worker_id)
    try:
        while True:
            if state.stop_flag:
                break
            try:
                horse_id = task_queue.get(timeout=5)
            except queue.Empty:
                continue
            if horse_id is None:
                task_queue.put(None)
                break

            # ローカルに既存なら skip（並走ワーカーが先に書いたケース）
            if local_path(horse_id).exists():
                with state.lock:
                    done_counter[0] += 1
                task_queue.task_done()
                continue

            url = PED_URL.format(horse_id=horse_id)
            try:
                html = client.fetch(url)
            except Exception as e:
                err = str(e)
                if "404 Client Error" in err or ("404" in err and "netkeiba" in err.lower()):
                    logger.warning("[worker-%d] HTTP 404 %s: %s", worker_id, horse_id, err[:80])
                    with state.lock:
                        state.not_found += 1
                        done_counter[0] += 1
                    if state.record_http404():
                        state.stop(f"HTTP 404 が {_HTTP_404_STREAK_LIMIT} 件連続")
                        task_queue.task_done()
                        break
                    task_queue.task_done()
                    continue

                fatal, reason = _is_fatal_http_error(err)
                if fatal:
                    logger.error("[worker-%d] ★ アクセスエラー → 全停止: %s", worker_id, reason)
                    state.stop(f"worker-{worker_id}: {reason}")
                    task_queue.task_done()
                    break

                logger.warning("[worker-%d] 取得失敗 %s: %s", worker_id, horse_id, err[:80])
                with state.lock:
                    state.error_count += 1
                    done_counter[0] += 1
                task_queue.task_done()
                continue

            if not html or "血統情報がありません" in html or "<title>エラー" in html:
                logger.warning("[worker-%d] データなし: %s", worker_id, horse_id)
                with state.lock:
                    state.not_found += 1
                    done_counter[0] += 1
                task_queue.task_done()
                continue

            ancestors = parse_blood_table_5gen(html)
            state.reset_404_streak()

            if not dry_run:
                sex = parse_horse_sex_from_ped_html(html, horse_id)
                rec  = build_pedigree_record(horse_id, ancestors,
                                             source="scrape_shutuba_pedigree", sex=sex)
                dest = local_path(horse_id)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            else:
                logger.info("[worker-%d][DRY-RUN] %s (%d祖先)", worker_id, horse_id, len(ancestors))

            with state.lock:
                state.scraped += 1
                done_counter[0] += 1
                done = done_counter[0]

            if done % 20 == 0:
                _write_status(state, done, total, t0, running=True)
                elapsed = time.time() - t0
                rate = state.scraped / elapsed * 3600 if elapsed > 0 else 0
                logger.info(
                    "[Phase①] 進捗: %d/%d  取得=%d  404=%d  Error=%d  [%.0f件/時, %.0f分経過]",
                    done, total, state.scraped, state.not_found, state.error_count,
                    rate, elapsed / 60,
                )

            task_queue.task_done()

    except Exception as e:
        logger.error("[worker-%d] 予期しないエラー: %s", worker_id, e, exc_info=True)
        state.stop(str(e))
    finally:
        client.close()
        logger.info("[worker-%d] 終了", worker_id)


# ── Phase ①: shutuba 5gen スクレイプ ─────────────────────────
def phase1(*, workers: int = 2, dry_run: bool = False) -> int:
    """shutuba horse_id の 5gen 血統を取得（未取得分のみ）。"""
    import concurrent.futures

    logger.info("=== Phase ①: shutuba 馬の 5世代血統スクレイプ ===")
    horse_ids = collect_shutuba_horse_ids()
    existing  = collect_existing_ids()
    todo      = sorted(horse_ids - existing)

    logger.info("shutuba 馬: %d 頭  既存: %d 頭  未取得: %d 頭",
                len(horse_ids), len(horse_ids & existing), len(todo))

    # 欠損リストを保存（再開・確認用）
    MISSING_PATH.parent.mkdir(parents=True, exist_ok=True)
    MISSING_PATH.write_text(json.dumps(todo, ensure_ascii=False), encoding="utf-8")
    logger.info("未取得リスト → %s", MISSING_PATH)

    if not todo:
        logger.info("Phase ①: 全 shutuba 馬の血統取得済み。スキップ。")
        return 0

    t0 = time.time()
    state = SharedState()
    done_counter: list[int] = [0]
    total = len(todo)

    task_q: queue.Queue[str | None] = queue.Queue()
    for hid in todo:
        task_q.put(hid)
    task_q.put(None)  # 終了シグナル

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futs = [
            executor.submit(
                worker_fn, task_q, state,
                dry_run=dry_run, worker_id=i,
                done_counter=done_counter, total=total, t0=t0,
            )
            for i in range(workers)
        ]
        concurrent.futures.wait(futs)

    _write_status(state, done_counter[0], total, t0, running=False)
    elapsed = time.time() - t0
    logger.info(
        "Phase ① 完了: 取得=%d  404=%d  エラー=%d  計=%d/%d  %.0f分",
        state.scraped, state.not_found, state.error_count,
        done_counter[0], total, elapsed / 60,
    )
    if state.stop_flag:
        logger.error("★ 強制停止: %s", state.stop_reason)
    return state.scraped


# ── ancestor_index.json 再構築 ────────────────────────────────
ANCESTOR_INDEX_PATH = _ROOT / "data" / "local" / "research" / "ancestor_index.json"


def rebuild_ancestor_index() -> None:
    """Phase ① 後に ancestor_index.json を再構築する。
    scrape_ancestors_upward.py の高速パスが正しく機能するよう更新する。"""
    logger.info("ancestor_index.json を再構築中...")
    idx: dict[str, str] = {}
    scanned = 0
    for f in LOCAL_PED_DIR.rglob("*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        scanned += 1
        # 馬本体も含める（horse_id と name）
        hid  = str(rec.get("horse_id") or "").strip()
        name = str(rec.get("sire") or "").strip()  # sire は参考のみ
        if hid and hid not in idx:
            idx[hid] = name
        # 祖先も追加
        for a in rec.get("ancestors") or []:
            aid  = str(a.get("horse_id") or "").strip()
            aname = str(a.get("name") or "").strip()
            if aid and aid not in idx:
                idx[aid] = aname
        if scanned % 5000 == 0:
            logger.info("  再構築: %d ファイルスキャン済み...", scanned)

    ANCESTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANCESTOR_INDEX_PATH.write_text(
        json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info("ancestor_index.json 再構築完了: %d エントリ → %s", len(idx), ANCESTOR_INDEX_PATH)


# ── Phase ②: 祖先上向きスクレイプ（外部スクリプト委任） ─────
def phase2(*, workers: int = 2, dry_run: bool = False) -> None:
    """scrape_ancestors_upward.py を呼び出して祖先を再帰的にスクレイプ（10世代化）。"""
    logger.info("=== Phase ②: 祖先上向きスクレイプ（10世代化） ===")

    # ancestor_index.json が存在しない場合のみ再構築（Phase ① 後に新規馬が追加された場合対応）
    if not ANCESTOR_INDEX_PATH.exists():
        logger.info("ancestor_index.json が未生成 → Phase ② 開始前に構築します")
        rebuild_ancestor_index()
    else:
        logger.info("ancestor_index.json が存在 → そのまま使用（高速起動モード）")

    script = _ROOT / "scripts" / "scrape_ancestors_upward.py"
    cmd = [sys.executable, str(script), "--workers", str(workers)]
    if dry_run:
        cmd.append("--dry-run")

    logger.info("実行: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        logger.error("Phase ② 異常終了: return code=%d", result.returncode)
    else:
        logger.info("Phase ② 完了")


# ── メイン ───────────────────────────────────────────────────
def main() -> None:
    script_basic_config()

    ap = argparse.ArgumentParser(description="shutuba 馬の血統補完（10世代対応）")
    ap.add_argument("--phase", choices=["1", "2", "all"], default="all",
                    help="1=5gen scrape のみ / 2=祖先上向きのみ / all=両方（既定）")
    ap.add_argument("--workers", type=int, default=2,
                    help="並列ワーカー数（既定 2）")
    ap.add_argument("--dry-run", action="store_true",
                    help="実際には保存しない（確認用）")
    ap.add_argument("--check", action="store_true",
                    help="欠損件数を確認して終了（スクレイプしない）")
    args = ap.parse_args()

    if args.check:
        script_basic_config()
        horse_ids = collect_shutuba_horse_ids()
        existing  = collect_existing_ids()
        todo      = sorted(horse_ids - existing)
        print(f"shutuba 馬: {len(horse_ids)} 頭")
        print(f"  既存 JSON: {len(horse_ids & existing)} 頭")
        print(f"  未取得:    {len(todo)} 頭")
        pct = 100.0 * len(horse_ids & existing) / len(horse_ids) if horse_ids else 0
        print(f"  取得率:    {pct:.1f}%")
        if todo:
            print(f"  未取得例:  {todo[:5]}")
        return

    if args.phase in ("1", "all"):
        scraped = phase1(workers=args.workers, dry_run=args.dry_run)
        logger.info("Phase ① 新規取得: %d 頭", scraped)

    if args.phase in ("2", "all"):
        phase2(workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""
祖先血統の上向き再帰スクレイパー（並列対応版）
=====================================================
現在 data/local/horse_pedigree_5gen に存在しない全祖先 horse_id を対象に
db.netkeiba の血統ページを取得し、同ディレクトリにローカル直書きする。

【アルゴリズム】
  1. full_sire_tree_nodes.parquet と既存 JSON から「JSONなし馬 ID」を収集
  2. ThreadPoolExecutor で --workers 並列にスクレイプ
  3. 取得した新祖先 ID を動的にキューへ追加（再帰的に上流を辿る）
  4. 三大始祖（Darley Arabian 等）で自然収束

【レート制限】
  - 各ワーカーが独立した NetkeibaClient を持つ
  - NETKEIBA_MAX_CONCURRENT_REQUESTS で in-flight 同時数を制御
  - NETKEIBA_THROTTLE_MIN/MAX (秒) でリクエスト間隔を制御

【再開・停止】
  - data/meta/ancestors_not_found.json に 404 馬を記録してスキップ
  - data/meta/ancestors_upward_status.json にリアルタイム進捗を書き込み
  - アクセスエラー(403/429/503)検知で全ワーカー即停止

Usage:
  python scripts/scraping/scrape_ancestors_upward.py
  python scripts/scraping/scrape_ancestors_upward.py --workers 4 --max-requests 5000
  python scripts/scraping/scrape_ancestors_upward.py --dry-run
  python scripts/scraping/scrape_ancestors_upward.py --clear-not-found  # 404キャッシュリセット
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.research.pedigree.pedigree_similarity import parse_blood_table_5gen, parse_horse_sex_from_ped_html  # noqa: E402
from scripts.scrape_pedigree_5gen import build_pedigree_record     # noqa: E402
from src.scraper.client import NetkeibaClient                          # noqa: E402

logger = logging.getLogger("scrape_ancestors_upward")

PED_URL        = "https://db.netkeiba.com/horse/ped/{horse_id}/"
LOCAL_PED_DIR  = _ROOT / "data" / "local" / "horse_pedigree_5gen"
NOT_FOUND_PATH = _ROOT / "data" / "local" / "meta" / "ancestors_not_found.json"
PROGRESS_PATH  = _ROOT / "data" / "local" / "meta" / "ancestors_upward_progress.json"
STATUS_PATH    = _ROOT / "data" / "local" / "meta" / "ancestors_upward_status.json"
NODES_PARQUET  = _ROOT / "data" / "local" / "research" / "pedigree_race_index" / "full_sire_tree_nodes.parquet"

# ── パス計算 ─────────────────────────────────────────────────
def local_path(horse_id: str) -> Path:
    sub = horse_id[:4] if len(horse_id) >= 4 else "_"
    return LOCAL_PED_DIR / sub / f"{horse_id}.json"


# ── ステータスファイル書き込み ────────────────────────────────
def _write_status(
    scraped: int,
    not_found: int,
    total_existing: int,
    total_target: int,
    t0: float,
    *,
    running: bool,
) -> None:
    elapsed = time.time() - t0
    rate_per_hour = scraped / elapsed * 3600 if elapsed > 0 else 0
    remaining = max(total_target - total_existing, 0)
    eta_sec = remaining / (rate_per_hour / 3600) if rate_per_hour > 0 else -1
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps({
            "scraped_this_run":   scraped,
            "not_found_this_run": not_found,
            "total_existing":     total_existing,
            "total_target":       total_target,
            "remaining":          remaining,
            "rate_per_hour":      round(rate_per_hour, 1),
            "eta_sec":            round(eta_sec, 0) if eta_sec > 0 else -1,
            "elapsed_min":        round(elapsed / 60, 1),
            "running":            running,
            "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ancestor_index.json（scrape_pedigree_5gen._build_ancestor_index が生成）
ANCESTOR_INDEX_PATH = _ROOT / "data" / "local" / "research" / "ancestor_index.json"


# ── 既存 JSON 収集 ────────────────────────────────────────────
def collect_existing_ids() -> set[str]:
    ids = {f.stem for f in LOCAL_PED_DIR.rglob("*.json")}
    logger.info("既存 JSON: %d ファイル", len(ids))
    return ids


def collect_ancestor_ids_from_local() -> set[str]:
    """全 JSON の ancestors から種牡馬（偶数 position）の horse_id のみを収集。

    高速パス: ancestor_index.json が存在すれば全ファイル読み込みを省略する。
    同ファイルは scrape_pedigree_5gen._build_ancestor_index() が生成する。
    存在しない場合のみ全 JSON を直接スキャンする（フォールバック）。
    """
    # ── 高速パス: キャッシュが使える場合 ──
    if ANCESTOR_INDEX_PATH.exists():
        try:
            idx: dict = json.loads(ANCESTOR_INDEX_PATH.read_text(encoding="utf-8"))
            ids = set(idx.keys())
            logger.info(
                "ancestor_index.json から祖先 ID を読み込み: %d 件 (%s)",
                len(ids), ANCESTOR_INDEX_PATH,
            )
            return ids
        except Exception as e:
            logger.warning("ancestor_index.json 読み込み失敗 → フォールバック: %s", e)

    # ── フォールバック: 全 JSON をスキャン（遅い） ──
    logger.info("ancestor_index.json が無いため全 JSON をスキャンします（初回は時間がかかります）")
    ids: set[str] = set()
    scanned = 0
    for f in LOCAL_PED_DIR.rglob("*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        scanned += 1
        if scanned % 5000 == 0:
            logger.info("  スキャン中... %d ファイル完了", scanned)
        for a in rec.get("ancestors") or []:
            # 偶数 position のみ（種牡馬スロット）
            if int(a.get("position", 1)) % 2 != 0:
                continue
            aid = str(a.get("horse_id") or "").strip()
            if aid:
                ids.add(aid)
    logger.info("全 JSON スキャン完了: %d ファイル → %d ユニーク祖先 ID", scanned, len(ids))
    return ids


# HTTP エラーコードによる停止判定
# ・400 → IP ブロック確定 → 即停止
# ・404 → _HTTP_404_STREAK_LIMIT 件連続で発生したら停止（単発は not_found へ追加してスキップ）
_HTTP_404_STREAK_LIMIT = 5


def _is_fatal_http_error(err: str) -> tuple[bool, str]:
    """
    fetch 例外メッセージを受け取り (即停止すべきか, 理由) を返す。
    Returns (True, reason) → 全ワーカー停止
    Returns (False, "")   → スキップして継続
    """
    # HTTP 400 = IP ブロック or ネットワーク障害
    if "400 Client Error" in err or ("400" in err and "netkeiba" in err.lower()):
        return True, f"HTTP 400 (IPブロック疑い): {err}"
    # 既存の致命的エラー
    for token in ("403", "429", "503", "Forbidden", "Too Many",
                  "Service Unavailable", "ConnectionError", "Network is unreachable"):
        if token in err:
            return True, f"{token}: {err}"
    return False, ""


# ── ワーカー関数 ──────────────────────────────────────────────
class SharedState:
    """スレッド間で共有する状態。"""
    def __init__(self, existing: set[str], not_found: set[str]):
        self.lock            = threading.Lock()
        self.existing        = existing          # スクレイプ済み horse_id
        self.not_found       = not_found         # 404 horse_id
        self.enqueued        = set(existing) | set(not_found)  # キューに追加済み
        self.scraped         = 0
        self.nf_count        = 0
        self.error_count     = 0
        self.stop_flag       = False
        self.stop_reason     = ""
        self.req_count       = 0
        self.http404_streak  = 0   # HTTP 404 連続カウント（全ワーカー合算）

    def stop(self, reason: str) -> None:
        with self.lock:
            self.stop_flag   = True
            self.stop_reason = reason

    def record_http404(self) -> bool:
        """HTTP 404 を記録し、連続 _HTTP_404_STREAK_LIMIT 件に達したら True を返す。"""
        with self.lock:
            self.http404_streak += 1
            self.nf_count       += 1
            over_limit = self.http404_streak >= _HTTP_404_STREAK_LIMIT
        return over_limit

    def reset_http404_streak(self) -> None:
        with self.lock:
            self.http404_streak = 0


def worker_fn(
    task_queue: "queue.Queue[str | None]",
    state: SharedState,
    *,
    dry_run: bool,
    worker_id: int,
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
            if horse_id is None:  # 終了シグナル
                task_queue.put(None)
                break

            # 既存チェック（ロック不要：ローカルファイルは不変）
            if local_path(horse_id).exists():
                with state.lock:
                    state.existing.add(horse_id)
                task_queue.task_done()
                continue

            url = PED_URL.format(horse_id=horse_id)
            try:
                html = client.fetch(url)
            except Exception as e:
                err = str(e)
                # HTTP 404 → not_found に追加し連続カウント、上限超えで全停止
                if "404 Client Error" in err or (
                    "404" in err and "netkeiba" in err.lower()
                ):
                    logger.warning("[worker-%d] HTTP 404 %s: %s", worker_id, horse_id, err)
                    with state.lock:
                        state.not_found.add(horse_id)
                    if state.record_http404():
                        reason = (
                            f"HTTP 404 が {_HTTP_404_STREAK_LIMIT} 件連続 → IPブロック疑い"
                        )
                        logger.error("[worker-%d] ★ %s → 全停止", worker_id, reason)
                        state.stop(f"worker-{worker_id}: {reason}")
                        task_queue.task_done()
                        break
                    task_queue.task_done()
                    continue

                # HTTP 400 / 403 / 429 / 503 etc. → 即停止
                fatal, reason = _is_fatal_http_error(err)
                if fatal:
                    logger.error("[worker-%d] ★ アクセスエラー → 全停止: %s", worker_id, reason)
                    state.stop(f"worker-{worker_id}: {reason}")
                    task_queue.task_done()
                    break

                logger.warning("[worker-%d] 取得失敗 %s: %s", worker_id, horse_id, err)
                with state.lock:
                    state.error_count += 1
                task_queue.task_done()
                continue

            # データなし判定（HTTP 200 + コンテンツレベルの「ページなし」）
            if not html or "血統情報がありません" in html or "<title>エラー" in html:
                logger.warning("[worker-%d] データなし: %s", worker_id, horse_id)
                with state.lock:
                    state.not_found.add(horse_id)
                    state.nf_count += 1
                task_queue.task_done()
                continue

            ancestors = parse_blood_table_5gen(html)
            # 祖先が少なくてもスキップしない。
            # 三大始祖など血統の起点となる馬や、血統ページに空白スロットがある馬も
            # 欠損値（血統不明）としてそのまま保存する。
            if len(ancestors) == 0:
                logger.info("[worker-%d] 祖先0件（ルート祖先か空白ページ）保存: %s", worker_id, horse_id)

            # 偶数 position = 種牡馬スロットのみ次のキューへ追加
            new_ids = [
                str(a.get("horse_id", ""))
                for a in ancestors
                if a.get("horse_id") and int(a.get("position", 1)) % 2 == 0
            ]

            if not dry_run:
                sex = parse_horse_sex_from_ped_html(html, horse_id)
                rec  = build_pedigree_record(horse_id, ancestors, source="scrape_ancestors_upward", sex=sex)
                dest = local_path(horse_id)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            else:
                logger.info("[worker-%d][DRY-RUN] %s (%d祖先)", worker_id, horse_id, len(ancestors))

            # 正常取得 → HTTP 404 連続カウントをリセット
            state.reset_http404_streak()

            with state.lock:
                state.scraped += 1
                state.req_count += 1
                state.existing.add(horse_id)
                # 新規祖先をキューへ
                for nid in new_ids:
                    if nid and nid not in state.enqueued:
                        state.enqueued.add(nid)
                        task_queue.put(nid)

            task_queue.task_done()

    except Exception as e:
        logger.error("[worker-%d] 予期しないエラー: %s", worker_id, e, exc_info=True)
        state.stop(str(e))
    finally:
        client.close()
        logger.info("[worker-%d] 終了", worker_id)


# ── メインループ ──────────────────────────────────────────────
class Stats(NamedTuple):
    scraped:   int
    not_found: int
    error:     int


def run(
    *,
    workers: int = 4,
    max_requests: int = 0,
    dry_run: bool = False,
    force: bool = False,
) -> Stats:
    import concurrent.futures

    t0 = time.time()

    # NETKEIBA_MAX_CONCURRENT_REQUESTS を workers に合わせて設定
    # (既にプロセス内で設定済みでなければ)
    if not os.environ.get("NETKEIBA_MAX_CONCURRENT_REQUESTS"):
        os.environ["NETKEIBA_MAX_CONCURRENT_REQUESTS"] = str(workers)
        # セマフォキャッシュをリセット
        try:
            from src.scraper.client import reset_netkeiba_inflight_semaphore_cache
            reset_netkeiba_inflight_semaphore_cache()
        except Exception:
            pass

    # not_found キャッシュ読み込み
    NOT_FOUND_PATH.parent.mkdir(parents=True, exist_ok=True)
    not_found: set[str] = set()
    if NOT_FOUND_PATH.exists():
        try:
            d = json.loads(NOT_FOUND_PATH.read_text(encoding="utf-8"))
            not_found = set(d.get("ids", []))
            logger.info("not_found キャッシュ: %d 件", len(not_found))
        except Exception:
            pass

    # 既存 JSON 収集
    logger.info("既存 JSON 収集中...")
    existing = collect_existing_ids()
    logger.info("  既存 JSON: %d ファイル", len(existing))

    logger.info("祖先 ID 収集中...")
    ancestor_ids = collect_ancestor_ids_from_local()
    logger.info("  祖先 ID: %d", len(ancestor_ids))

    extra_ids: set[str] = set()
    if NODES_PARQUET.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(NODES_PARQUET, columns=["horse_id"])
            extra_ids = set(df["horse_id"].astype(str).tolist())
            logger.info("  tree nodes: %d", len(extra_ids))
        except Exception as e:
            logger.warning("tree nodes 読み込みスキップ: %s", e)

    candidates = (ancestor_ids | extra_ids)
    if force:
        todo_ids = sorted(candidates - not_found)
    else:
        todo_ids = sorted(candidates - existing - not_found)

    total_target = len(candidates)
    logger.info("スクレイプ候補: %d 頭 (全候補: %d)", len(todo_ids), total_target)

    if not todo_ids:
        logger.info("スクレイプ対象なし。終了。")
        return Stats(0, 0, 0)

    # ── キューとワーカー起動 ──────────────────────────────────
    task_queue: "queue.Queue[str | None]" = queue.Queue()
    for hid in todo_ids:
        task_queue.put(hid)

    state = SharedState(existing=set(existing), not_found=not_found)
    state.enqueued = set(todo_ids) | existing | not_found

    _write_status(0, 0, len(existing), total_target, t0, running=True)

    futures = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers,
                                                      thread_name_prefix="ped-worker")
    for wid in range(workers):
        f = executor.submit(worker_fn, task_queue, state, dry_run=dry_run, worker_id=wid)
        futures.append(f)

    # ── 進捗監視スレッド ──────────────────────────────────────
    def progress_watcher():
        while not state.stop_flag:
            with state.lock:
                sc = state.scraped
                nf = state.nf_count
                ex = len(state.existing)
                ec = state.error_count
            remaining_q = task_queue.qsize()
            elapsed = time.time() - t0
            rate = sc / elapsed * 3600 if elapsed > 0 else 0
            logger.info(
                "進捗: 取得=%d  404=%d  エラー=%d  Qサイズ≈%d  速度≈%.0f件/時  経過%.1f分",
                sc, nf, ec, remaining_q, rate, elapsed / 60,
            )
            _write_status(sc, nf, ex, total_target, t0, running=True)

            # not_found IDs を定期的に中間保存（解析用）
            with state.lock:
                current_nf_ids = sorted(state.not_found)
            try:
                NOT_FOUND_PATH.write_text(
                    json.dumps({"ids": current_nf_ids, "count": len(current_nf_ids)},
                               ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

            # max_requests チェック
            if max_requests > 0 and sc >= max_requests:
                logger.info("--max-requests %d 到達 → 停止", max_requests)
                state.stop(f"max-requests {max_requests}")
                # キューを空にして終了シグナルを送る
                try:
                    while True:
                        task_queue.get_nowait()
                        task_queue.task_done()
                except Exception:
                    pass
                task_queue.put(None)
                break

            time.sleep(20)

    watcher = threading.Thread(target=progress_watcher, daemon=True, name="watcher")
    watcher.start()

    # ── 終了待機 ──────────────────────────────────────────────
    try:
        # キューが空になるまで待つ（task_done ベース）
        while True:
            if state.stop_flag:
                # 残タスクをドレイン
                drained = 0
                while True:
                    try:
                        task_queue.get_nowait()
                        task_queue.task_done()
                        drained += 1
                    except Exception:
                        break
                if drained:
                    logger.info("停止フラグ: キューから %d 件をドレイン", drained)
                break
            # キューが空 かつ ワーカーが全部待機状態
            if task_queue.empty():
                # 全タスク完了を確認
                task_queue.join()
                break
            time.sleep(1)

        # 終了シグナルを送って全ワーカーを止める
        task_queue.put(None)
        executor.shutdown(wait=True, cancel_futures=False)

    except KeyboardInterrupt:
        logger.warning("Ctrl+C で中断")
        state.stop("KeyboardInterrupt")
        task_queue.put(None)
        executor.shutdown(wait=False, cancel_futures=True)

    # ── 最終集計 ──────────────────────────────────────────────
    elapsed = time.time() - t0
    sc, nf, ec = state.scraped, state.nf_count, state.error_count

    _write_status(sc, nf, len(state.existing), total_target, t0, running=False)

    # not_found キャッシュ保存
    if not dry_run:
        all_nf = sorted(state.not_found)
        NOT_FOUND_PATH.write_text(
            json.dumps({"ids": all_nf, "count": len(all_nf)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        PROGRESS_PATH.write_text(json.dumps({
            "last_run_scraped":   sc,
            "last_run_not_found": nf,
            "total_local_json":   len(list(LOCAL_PED_DIR.rglob("*.json"))),
            "not_found_total":    len(state.not_found),
            "workers":            workers,
            "stop_reason":        state.stop_reason,
            "elapsed_min":        round(elapsed / 60, 1),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "=== 完了%s ===\n"
        "  取得成功: %d 頭\n"
        "  404/データなし: %d 頭\n"
        "  エラー: %d 件\n"
        "  経過時間: %.1f 分",
        f" ({state.stop_reason})" if state.stop_reason else "",
        sc, nf, ec, elapsed / 60,
    )
    return Stats(sc, nf, ec)


def _rebuild_ancestor_index_local() -> None:
    """ローカル JSON から ancestor_index.json を再構築する（HybridStorage 不要版）。"""
    idx: dict[str, str] = {}
    scanned = 0
    for f in LOCAL_PED_DIR.rglob("*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        scanned += 1
        for a in rec.get("ancestors") or []:
            aid  = str(a.get("horse_id") or "").strip()
            aname = str(a.get("name") or "").strip()
            if aid and aid not in idx:
                idx[aid] = aname
    ANCESTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANCESTOR_INDEX_PATH.write_text(
        json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info("ancestor_index.json 再構築完了: %d エントリ (スキャン %d ファイル)",
                len(idx), scanned)


def main() -> None:
    # ルートロガーは WARNING 以上のみ（NetkeibaClient の GET 行を抑制）
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
    )
    # このスクリプト自身の進捗ログは INFO まで表示
    logging.getLogger("scrape_ancestors_upward").setLevel(logging.INFO)

    ap = argparse.ArgumentParser(
        description="祖先血統の上向き再帰スクレイパー（並列対応）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--workers",      type=int,   default=4,
                    help="並列ワーカー数 (default: 4)")
    ap.add_argument("--max-requests", type=int,   default=0,
                    help="最大取得件数 (0=無制限)")
    ap.add_argument("--dry-run",      action="store_true",
                    help="保存せず動作確認のみ")
    ap.add_argument("--force",        action="store_true",
                    help="既存ローカル JSON も再取得")
    ap.add_argument("--clear-not-found", action="store_true",
                    help="not_found キャッシュをクリアして再試行")
    args = ap.parse_args()

    if args.clear_not_found and NOT_FOUND_PATH.exists():
        NOT_FOUND_PATH.unlink()
        logger.info("not_found キャッシュをクリアしました")

    stats = run(
        workers=args.workers,
        max_requests=args.max_requests,
        dry_run=args.dry_run,
        force=args.force,
    )
    logger.info("最終: scraped=%d not_found=%d error=%d",
                stats.scraped, stats.not_found, stats.error)

    # 新規取得があった場合 ancestor_index.json を再構築（次回起動を高速化）
    if stats.scraped > 0 and not args.dry_run:
        logger.info("ancestor_index.json を更新中（次回高速起動のため）...")
        try:
            _rebuild_ancestor_index_local()
        except Exception as e:
            logger.warning("ancestor_index.json 更新失敗（次回フォールバックで再構築）: %s", e)


if __name__ == "__main__":
    main()

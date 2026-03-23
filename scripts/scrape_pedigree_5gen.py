"""
5世代血統データ取得スクリプト (Phase 1 + Phase 2)

Phase 1: horse_result に存在する全馬の5世代血統を取得・保存
Phase 2: Phase 1 で発見された全種牡馬の5世代血統を取得・保存
SmartRC: 取得済みの系統コード (ll/sl/cl) を統合

保存先: GCS horse_pedigree_5gen/{horse_id}.json

Usage:
  python scripts/scrape_pedigree_5gen.py --phase 1
  python scripts/scrape_pedigree_5gen.py --phase 2
  python scripts/scrape_pedigree_5gen.py --phase all
  python scripts/scrape_pedigree_5gen.py --phase 1 --workers 3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.pedigree_similarity import parse_blood_table_5gen
from scraper.client import NetkeibaClient
from scraper.storage import HybridStorage

logger = logging.getLogger("scrape_pedigree_5gen")

PED_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"

SMARTRC_PREFIX_MAP = {
    (1, 0): "f",      (1, 1): "m",
    (2, 0): "ff",     (2, 1): "fm",
    (2, 2): "mf",     (2, 3): "mm",
    (3, 0): "fff",    (3, 1): "ffm",
    (3, 2): "fmf",    (3, 3): "fmm",
    (3, 4): "mff",    (3, 5): "mfm",
    (3, 6): "mmf",    (3, 7): "mmm",
}

_progress_lock = threading.Lock()


def enrich_with_smartrc(
    ancestors: list[dict], smartrc_horse: dict | None,
) -> list[dict]:
    """SmartRC のデータから系統コードを祖先データに統合。"""
    if not smartrc_horse:
        return ancestors

    for anc in ancestors:
        key = (anc["generation"], anc["position"])
        prefix = SMARTRC_PREFIX_MAP.get(key)
        if not prefix:
            continue

        anc["smartrc_code"] = smartrc_horse.get(f"{prefix}_code", "")
        anc["ll_code"] = smartrc_horse.get(f"{prefix}_llcode", "")
        anc["sl_code"] = smartrc_horse.get(f"{prefix}_slcode", "")
        anc["cl_code"] = smartrc_horse.get(f"{prefix}_clcode", "")
        anc["country"] = smartrc_horse.get(f"{prefix}_country", "")

    return ancestors


def build_pedigree_record(
    horse_id: str,
    ancestors: list[dict],
    source: str = "netkeiba",
) -> dict:
    """保存用の血統レコードを構築。"""
    sire_name = ""
    dam_name = ""
    dam_sire_name = ""
    for a in ancestors:
        if a["generation"] == 1 and a["position"] == 0:
            sire_name = a["name"]
        elif a["generation"] == 1 and a["position"] == 1:
            dam_name = a["name"]
        elif a["generation"] == 2 and a["position"] == 2:
            dam_sire_name = a["name"]

    return {
        "horse_id": horse_id,
        "sire": sire_name,
        "dam": dam_name,
        "dam_sire": dam_sire_name,
        "ancestors": ancestors,
        "ancestor_count": len(ancestors),
        "source": source,
    }


class AccessError(Exception):
    """403/429/503 等のアクセスエラー。検知時に即停止すべき。"""
    pass


class StopFlag:
    """スレッド間で共有する停止フラグ。"""
    def __init__(self):
        self._stop = False
        self._lock = threading.Lock()
        self.reason = ""

    def stop(self, reason: str = ""):
        with self._lock:
            self._stop = True
            self.reason = reason

    @property
    def is_set(self) -> bool:
        return self._stop


def _scrape_one(
    horse_id: str,
    client: NetkeibaClient,
    storage: HybridStorage,
    source: str,
    stop_flag: StopFlag | None = None,
) -> tuple[str, bool, str]:
    """1馬分の血統を取得・保存。(horse_id, success, error_msg) を返す。
    アクセスエラー (403/429/503) は AccessError として raise する。"""
    if stop_flag and stop_flag.is_set:
        return (horse_id, False, "停止済み")
    try:
        url = PED_URL.format(horse_id=horse_id)
        html = client.fetch(url)
        ancestors = parse_blood_table_5gen(html)

        if len(ancestors) < 5:
            return (horse_id, False, f"祖先不足({len(ancestors)})")

        record = build_pedigree_record(horse_id, ancestors, source=source)
        storage.save("horse_pedigree_5gen", horse_id, record)
        return (horse_id, True, "")
    except Exception as e:
        err_str = str(e)
        if any(code in err_str for code in ("403", "429", "503", "Forbidden",
                                              "Too Many", "Service Unavailable",
                                              "ConnectionError", "Connection aborted",
                                              "Network is unreachable")):
            raise AccessError(f"アクセスエラー ({horse_id}): {err_str}") from e
        return (horse_id, False, err_str)


def _run_sequential(
    todo: list[str],
    storage: HybridStorage,
    source: str,
    label: str,
) -> int:
    """単一クライアントで順次スクレイピング。アクセスエラー時は即停止。

    スレッドセーフなクライアントを1つ使用し、netkeiba のレート制限を
    確実に遵守する。
    """
    total = len(todo)
    logger.info("%s: %d 頭を順次取得開始", label, total)

    stop_flag = StopFlag()
    success = 0
    fail = 0
    t0 = time.time()

    client = NetkeibaClient(interval=3.0, auto_login=True)
    try:
        for i, horse_id in enumerate(todo):
            if stop_flag.is_set:
                logger.warning("%s: 停止フラグ検知、中断 (%s)", label, stop_flag.reason)
                break

            try:
                hid, ok, err = _scrape_one(horse_id, client, storage, source,
                                           stop_flag=stop_flag)
            except AccessError as e:
                logger.error("★ アクセスエラー検知 — 即座に停止: %s", e)
                stop_flag.stop(str(e))
                break

            if ok:
                success += 1
            else:
                fail += 1
                if err:
                    logger.warning("失敗 %s: %s", hid, err)

            done = i + 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed * 3600 if elapsed > 0 else 0
                logger.info(
                    "%s 進捗: %d/%d (成功=%d, 失敗=%d) [%.0f件/時, 経過 %.0f分]",
                    label, done, total, success, fail, rate, elapsed / 60,
                )
    except KeyboardInterrupt:
        logger.warning("%s: Ctrl+C で中断", label)
    finally:
        client.close()

    elapsed = time.time() - t0
    logger.info(
        "%s 完了: 成功=%d, 失敗=%d (計 %d/%d, %.0f分)",
        label, success, fail, success + fail, total, elapsed / 60,
    )
    return success


def phase1(storage: HybridStorage, **_kwargs):
    """Phase 1: horse_result に存在する全馬の5世代血統を取得。"""
    logger.info("=== Phase 1: 全馬の5世代血統取得 ===")

    hr_keys = storage.list_keys("horse_result")
    existing = set(storage.list_keys("horse_pedigree_5gen"))
    todo = [k for k in hr_keys if k not in existing]

    logger.info("対象: %d 頭 (既存: %d, 新規: %d)", len(hr_keys), len(existing), len(todo))

    if not todo:
        logger.info("Phase 1: 全馬取得済み")
        return 0

    return _run_sequential(todo, storage, source="phase1", label="Phase 1")


_ANCESTOR_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "research" / "ancestor_index.json"


def _build_ancestor_index(storage: HybridStorage) -> dict[str, str]:
    """全 horse_pedigree_5gen から祖先IDを抽出しインデックスファイルに保存。"""
    ped_keys = storage.list_keys("horse_pedigree_5gen")
    logger.info("祖先インデックス構築: %d レコードをスキャン...", len(ped_keys))

    stallion_ids: dict[str, str] = {}
    batch_size = 50
    for batch_start in range(0, len(ped_keys), batch_size):
        batch = ped_keys[batch_start:batch_start + batch_size]
        for key in batch:
            data = storage.load("horse_pedigree_5gen", key)
            if not data:
                continue
            for anc in data.get("ancestors", []):
                hid = anc.get("horse_id", "")
                name = anc.get("name", "")
                if hid and name and hid not in stallion_ids:
                    stallion_ids[hid] = name
        if (batch_start + batch_size) % 500 == 0:
            logger.info("  スキャン進捗: %d/%d (ユニーク祖先: %d)",
                        min(batch_start + batch_size, len(ped_keys)),
                        len(ped_keys), len(stallion_ids))

    _ANCESTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ANCESTOR_INDEX_PATH.write_text(
        json.dumps(stallion_ids, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    logger.info("祖先インデックス保存: %d 頭 → %s", len(stallion_ids), _ANCESTOR_INDEX_PATH)
    return stallion_ids


def phase2(storage: HybridStorage, **_kwargs):
    """Phase 2: Phase 1 で発見された全種牡馬の5世代血統を取得。"""
    logger.info("=== Phase 2: 種牡馬の5世代血統取得 ===")

    if _ANCESTOR_INDEX_PATH.exists():
        logger.info("祖先インデックスを読み込み: %s", _ANCESTOR_INDEX_PATH)
        stallion_ids = json.loads(_ANCESTOR_INDEX_PATH.read_text(encoding="utf-8"))
    else:
        stallion_ids = _build_ancestor_index(storage)

    existing = set(storage.list_keys("horse_pedigree_5gen"))
    todo_ids = [hid for hid in stallion_ids if hid not in existing]

    logger.info(
        "種牡馬/祖先: %d 頭発見, 未取得: %d 頭",
        len(stallion_ids), len(todo_ids),
    )

    if not todo_ids:
        logger.info("Phase 2: 全祖先取得済み")
        return 0

    return _run_sequential(todo_ids, storage, source="phase2_stallion", label="Phase 2")


def enrich_smartrc(storage: HybridStorage):
    """保存済みの horse_pedigree_5gen に SmartRC の系統コードを統合。"""
    logger.info("=== SmartRC 系統コード統合 ===")

    smartrc_keys = storage.list_keys("smartrc_race")
    logger.info("smartrc_race: %d レース", len(smartrc_keys))

    horse_smartrc: dict[str, dict] = {}
    for key in smartrc_keys:
        data = storage.load("smartrc_race", key)
        if not data:
            continue
        for hid, hdata in data.get("horses", {}).items():
            if hid and hid not in horse_smartrc:
                horse_smartrc[hid] = hdata

    logger.info("SmartRC 馬データ: %d 頭", len(horse_smartrc))

    ped_keys = storage.list_keys("horse_pedigree_5gen")
    enriched = 0

    for key in ped_keys:
        smartrc = horse_smartrc.get(key)
        if not smartrc:
            continue

        ped = storage.load("horse_pedigree_5gen", key)
        if not ped:
            continue

        ancestors = ped.get("ancestors", [])
        had_smartrc = any(a.get("ll_code") for a in ancestors)
        if had_smartrc:
            continue

        ancestors = enrich_with_smartrc(ancestors, smartrc)
        ped["ancestors"] = ancestors
        ped["smartrc_enriched"] = True
        storage.save("horse_pedigree_5gen", key, ped)
        enriched += 1

    logger.info("SmartRC 統合完了: %d 馬を更新", enriched)


def summary(storage: HybridStorage):
    """保存済みデータの概要を表示。"""
    keys = storage.list_keys("horse_pedigree_5gen")
    logger.info("=== horse_pedigree_5gen サマリー ===")
    logger.info("  総レコード: %d", len(keys))

    if not keys:
        return

    all_ancestor_ids: set[str] = set()
    all_sires: set[str] = set()
    smartrc_count = 0
    total_ancestors = 0

    import random
    sample = random.sample(keys, min(200, len(keys)))

    for key in sample:
        data = storage.load("horse_pedigree_5gen", key)
        if not data:
            continue
        ancs = data.get("ancestors", [])
        total_ancestors += len(ancs)
        if data.get("smartrc_enriched"):
            smartrc_count += 1
        for a in ancs:
            if a.get("horse_id"):
                all_ancestor_ids.add(a["horse_id"])
            if a.get("generation") == 1 and a.get("position") == 0:
                all_sires.add(a.get("name", ""))

    logger.info("  サンプル %d 馬 平均祖先数: %.1f", len(sample), total_ancestors / max(len(sample), 1))
    logger.info("  ユニーク祖先 ID: %d", len(all_ancestor_ids))
    logger.info("  ユニーク種牡馬: %d", len(all_sires))
    logger.info("  SmartRC 統合済み: %d / %d", smartrc_count, len(sample))


def main():
    parser = argparse.ArgumentParser(description="5世代血統データ取得")
    parser.add_argument("--phase", choices=["1", "2", "all", "smartrc", "summary"], default="all")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    storage = HybridStorage()

    if args.phase == "summary":
        summary(storage)
        return

    if args.phase == "smartrc":
        enrich_smartrc(storage)
        return

    if args.phase in ("1", "all"):
        phase1(storage)
        _build_ancestor_index(storage)

    if args.phase in ("2", "all"):
        phase2(storage)

    if args.phase == "all":
        enrich_smartrc(storage)
        summary(storage)


if __name__ == "__main__":
    main()

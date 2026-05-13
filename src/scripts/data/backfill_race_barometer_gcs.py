#!/usr/bin/env python3
"""
GCS のみ: race_result に存在するレースIDについて偏差値 (race_barometer) を
skip なしで再取得し、GCS へ上書き保存する（ローカル常設 mirror / data/local 本体は作らない）。

前提:
- .env に GCS 設定
- .env の netkeiba_id / netkeiba_pw（プレミアム等の有料会員。credit=1 API 用）
- 保存は scraper.storage.HybridStorage: local_only 以外は GCS のみ（persist_context で mirror 抑止）

使用例::

  # 接続・ログインのスモーク（5 レース）
  python scripts/data/backfill_race_barometer_gcs.py --from-year 2023 --to-year 2025 --limit 5

  # 本番（全件・時間がかかる）
  python scripts/data/backfill_race_barometer_gcs.py --from-year 2023 --to-year 2026
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# リポジトリ root
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("backfill_barometer")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except Exception:
        pass


def _collect_race_ids(storage, from_y: int, to_y: int) -> list[str]:
    out: set[str] = set()
    for y in range(from_y, to_y + 1):
        ys = str(y)
        keys = storage.list_keys("race_result", year=ys)
        for k in keys:
            if not k or len(k) < 4:
                continue
            try:
                yk = int(k[:4])
            except ValueError:
                continue
            if yk < from_y or yk > to_y:
                continue
            out.add(k)
    return sorted(out)


def main() -> int:
    _load_env()
    ap = argparse.ArgumentParser(description="race_barometer GCS 上書き再取得")
    ap.add_argument(
        "--from-year",
        type=int,
        default=2023,
        help="race_result の年キー下限（含む、既定 2023）",
    )
    ap.add_argument(
        "--to-year",
        type=int,
        default=2026,
        help="race_result の年キー上限（含む、既定 2026=当年）",
    )
    ap.add_argument("--limit", type=int, default=0, help="デバッグ: 先頭 N 件だけ（0=無制限）")
    ap.add_argument(
        "--interval-sec",
        type=float,
        default=0.4,
        help="レースあたりの待機（クライアント節度に加算される NetkeibaClient 間隔）",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="race_id 列挙と件数だけ（通信・GCS 書き込みなし）",
    )
    args = ap.parse_args()

    from src.utils.keiba_logging import script_basic_config

    script_basic_config()

    from src.scraper import persist_context
    from src.scraper.run import ScraperRunner

    if args.from_year > args.to_year:
        logger.error("from-year > to-year")
        return 2

    runner = ScraperRunner(interval=args.interval_sec, cache=False, auto_login=True)
    st = runner.storage
    if not st.gcs_enabled:
        logger.error("GCS 未接続: 本スクリプトは GCS 有効が前提です。")
        return 1
    st.check_gcs_connectivity(quick=True)
    if not st._gcs_healthy:  # noqa: SLF001
        logger.error("GCS バックオフ中: 落ち着いてから再実行してください。")
        return 1

    rids = _collect_race_ids(st, args.from_year, args.to_year)
    if not rids:
        logger.error("race_id が0件: GCS の race_result 一覧を取得できていません。")
        return 1
    ntot = len(rids)
    if args.limit and args.limit > 0:
        rids = rids[: int(args.limit)]

    logger.info("対象レース: %d / %d 件 (from=%d to=%d dry_run=%s)", len(rids), ntot, args.from_year, args.to_year, args.dry_run)
    if args.dry_run:
        for i, r in enumerate(rids[:20]):
            logger.info("  [%d] %s", i + 1, r)
        if len(rids) > 20:
            logger.info("  ...")
        return 0

    if not os.environ.get("netkeiba_id") or not os.environ.get("netkeiba_pw"):
        logger.error("netkeiba_id / netkeiba_pw が .env にありません。")
        return 1

    tok = persist_context.set_skip_local_mirror(True)
    n_ok = n_no_data = n_err = 0
    t0 = time.time()
    try:
        for i, rid in enumerate(rids):
            try:
                d = runner.scrape_barometer(rid, skip_existing=False)
            except Exception as e:
                n_err += 1
                logger.error("[%d/%d] %s: %s", i + 1, len(rids), rid, e)
                continue
            if d and d.get("entries"):
                n_ok += 1
            else:
                n_no_data += 1
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                logger.info("進捗: %d/%d ok=%d nodata=%d err=%d (%.1fs)", i + 1, len(rids), n_ok, n_no_data, n_err, el)
    finally:
        persist_context.reset_skip_local_mirror(tok)
        try:
            st.flush_weekly_access()
        except Exception:
            pass

    el = time.time() - t0
    logger.info(
        "完了: total=%d ok(件あり)=%d データなし(保存スキップ)=%d 例外=%d 所要=%.1fs",
        len(rids),
        n_ok,
        n_no_data,
        n_err,
        el,
    )
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

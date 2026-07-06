"""
2026年データ完全化バッチ (2026/1/1〜7/2)

欠損カテゴリを検出してスクレイプし、派生カテゴリを migrate で補完する。
収束するまでサイクルを繰り返す。

実行:
    python -m src.scripts.scraping.complete_2026_data
    python -m src.scripts.scraping.complete_2026_data --dry-run
    python -m src.scripts.scraping.complete_2026_data --max-cycles 10
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

logger = logging.getLogger("complete_2026_data")
BASE = Path(__file__).parent.parent.parent.parent
COVERAGE_DIR = BASE / "data" / "local" / "meta" / "date_coverage" / "2026"
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATE_START = "20260101"
DATE_END   = "20260702"

# 補完対象カテゴリと対応するスクレイパメソッド
SCRAPE_CATEGORIES = {
    "race_result":     "scrape_race_result",
    "race_shutuba":    "scrape_race_card",
    "race_result_lap": "scrape_race_result_lap",
}

# 派生カテゴリ（migrate_row_data で生成）のソースカテゴリ
MIGRATE_SOURCES = ["race_result", "race_result_on_time", "race_shutuba"]


def load_coverage() -> dict[str, dict]:
    """日付別カバレッジ情報を読み込む。"""
    coverage: dict[str, dict] = {}
    for f in sorted(COVERAGE_DIR.glob("*.json")):
        if DATE_START <= f.stem <= DATE_END:
            try:
                coverage[f.stem] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                pass
    return coverage


def find_missing(storage, categories: list[str]) -> dict[str, list[str]]:
    """
    カテゴリ別の欠損 race_id リストを返す。
    GCS の exists() は遅いため、coverage カウントで先絞りしてから確認する。
    """
    coverage = load_coverage()
    missing: dict[str, list[str]] = {c: [] for c in categories}

    for date_str, d in coverage.items():
        total = d.get("total_races", 0)
        cats  = d.get("categories", {})
        race_ids = d.get("race_ids", [])

        for cat in categories:
            n = cats.get(cat, 0)
            if isinstance(n, int) and n < total:
                for rid in race_ids:
                    try:
                        if not storage.exists(cat, rid):
                            missing[cat].append(rid)
                    except Exception:
                        pass

    return missing


def scrape_batch(runner, method_name: str, race_ids: list[str], interval: float = 2.0) -> int:
    """指定メソッドで race_ids を順次スクレイプ。成功数を返す。"""
    method = getattr(runner, method_name, None)
    if method is None:
        logger.warning("メソッド不明: %s", method_name)
        return 0

    ok_count = 0
    for i, rid in enumerate(race_ids, 1):
        logger.info("  [%d/%d] %s %s", i, len(race_ids), method_name, rid)
        try:
            result = method(rid, skip_existing=False)
            if result:
                ok_count += 1
                logger.info("    -> 保存成功")
            else:
                logger.warning("    -> None 返却（ページなし or スキップ）")
        except Exception as exc:
            logger.error("    -> エラー: %s", exc)
        time.sleep(interval)

    return ok_count


def run_migrate(dry_run: bool = False) -> None:
    """migrate_row_data_to_unique_paths で派生カテゴリを補完する。"""
    cmd = [
        sys.executable, "-m",
        "src.scripts.scraping.migrate_row_data_to_unique_paths",
        "--year", "2026",
        "--include-horses",
    ]
    if dry_run:
        cmd.append("--dry-run")

    logger.info("migrate 実行: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE))
        if proc.returncode == 0:
            logger.info("migrate 完了")
        else:
            logger.error("migrate エラー:\n%s", proc.stderr[-2000:])
        # ログに出力
        for line in proc.stdout.splitlines()[-30:]:
            logger.info("  [migrate] %s", line)
    except Exception as exc:
        logger.error("migrate 実行失敗: %s", exc)


def run_backfill_full(dry_run: bool = False) -> None:
    """backfill full フェーズを実行して残り3日分を補完。"""
    cmd = [
        sys.executable, "-m", "src.scraper.backfill",
        "--year", "2026",
        "--phase", "full",
        "--max-dates", "20",
    ]
    if dry_run:
        cmd.append("--dry-run")

    logger.info("backfill full 実行: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE))
        for line in (proc.stdout + proc.stderr).splitlines()[-20:]:
            logger.info("  [backfill] %s", line)
    except Exception as exc:
        logger.error("backfill 実行失敗: %s", exc)


def coverage_summary() -> dict:
    """カバレッジの現在の集計を返す。"""
    coverage = load_coverage()
    total_races = 0
    cat_got: dict[str, int] = {}
    cat_total: dict[str, int] = {}

    for d in coverage.values():
        t = d.get("total_races", 0)
        total_races += t
        for cat, n in d.get("categories", {}).items():
            if isinstance(n, int):
                cat_got[cat] = cat_got.get(cat, 0) + n
                cat_total[cat] = cat_total.get(cat, 0) + t

    incomplete = {
        cat: {
            "got": cat_got[cat],
            "total": cat_total[cat],
            "pct": round(cat_got[cat] / cat_total[cat] * 100, 1),
        }
        for cat in cat_total
        if cat_got.get(cat, 0) < cat_total[cat]
    }
    return {
        "dates": len(coverage),
        "total_races": total_races,
        "incomplete": incomplete,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "complete_2026_data.log", encoding="utf-8"),
        ],
    )

    parser = argparse.ArgumentParser(description="2026年データ完全化バッチ")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=5, help="最大サイクル数")
    parser.add_argument("--interval", type=float, default=2.0, help="リクエスト間隔（秒）")
    parser.add_argument("--categories", nargs="*",
                        default=list(SCRAPE_CATEGORIES.keys()),
                        help="対象カテゴリ（省略=全）")
    args = parser.parse_args()

    from src.utils.project_env import load_project_dotenv
    load_project_dotenv()

    from src.scraper.run import ScraperRunner
    from src.scraper.storage import HybridStorage

    logger.info("=== 2026年データ完全化バッチ 開始 ===")
    logger.info("対象期間: %s 〜 %s", DATE_START, DATE_END)
    logger.info("最大サイクル数: %d", args.max_cycles)

    # 初期サマリー
    summary = coverage_summary()
    logger.info("初期状態: %d日, %d レース", summary["dates"], summary["total_races"])
    for cat, v in sorted(summary["incomplete"].items(), key=lambda x: x[1]["pct"]):
        logger.info("  未完全: %s  %d/%d (%.1f%%)", cat, v["got"], v["total"], v["pct"])

    if args.dry_run:
        logger.info("[DRY-RUN] 実際のスクレイプは行いません")

    storage = HybridStorage(auto_login=False)

    # ── サイクルループ ──────────────────────────────
    for cycle in range(1, args.max_cycles + 1):
        logger.info("")
        logger.info("━━━ サイクル %d / %d ━━━", cycle, args.max_cycles)

        # 欠損 race_id を検出
        logger.info("欠損 race_id を検出中...")
        scrape_cats = [c for c in args.categories if c in SCRAPE_CATEGORIES]
        missing = find_missing(storage, scrape_cats)

        total_missing = sum(len(v) for v in missing.values())
        logger.info("欠損合計: %d件", total_missing)
        for cat, ids in missing.items():
            logger.info("  %s: %d件", cat, len(ids))

        if total_missing == 0:
            logger.info("✅ 全カテゴリ完全 — サイクル終了")
            break

        if not args.dry_run:
            runner = ScraperRunner(interval=args.interval, auto_login=True)

            for cat, ids in missing.items():
                if not ids:
                    continue
                method_name = SCRAPE_CATEGORIES[cat]
                logger.info("━ %s (%d件) をスクレイプ中...", cat, len(ids))
                ok = scrape_batch(runner, method_name, ids, interval=args.interval)
                logger.info("  完了: %d / %d 成功", ok, len(ids))

            # backfill full（残り3日分等）
            logger.info("backfill full フェーズを実行...")
            run_backfill_full(dry_run=args.dry_run)

            # 派生カテゴリ補完
            logger.info("migrate で派生カテゴリを補完...")
            run_migrate(dry_run=args.dry_run)

        # サイクル後のサマリー
        summary = coverage_summary()
        logger.info("")
        logger.info("サイクル %d 後の状態:", cycle)
        remaining_incomplete = {
            cat: v for cat, v in summary["incomplete"].items()
            if v["pct"] < 100
        }
        if remaining_incomplete:
            for cat, v in sorted(remaining_incomplete.items(), key=lambda x: x[1]["pct"]):
                logger.info("  未完全: %s  %d/%d (%.1f%%)", cat, v["got"], v["total"], v["pct"])
        else:
            logger.info("  ✅ 全カテゴリ 100%% 完全")
            break

    # 最終サマリー
    logger.info("")
    logger.info("=== 最終サマリー ===")
    summary = coverage_summary()
    for cat, v in sorted(summary["incomplete"].items(), key=lambda x: x[1]["pct"]):
        logger.info("  %s: %d/%d (%.1f%%)", cat, v["got"], v["total"], v["pct"])
    if not summary["incomplete"]:
        logger.info("  ✅ 全カテゴリ完全")
    logger.info("=== 2026年データ完全化バッチ 終了 ===")


if __name__ == "__main__":
    main()

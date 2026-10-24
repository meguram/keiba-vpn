"""
date_coverage.py — レース開催日 × カテゴリ別カバレッジ ローカルインデックス管理

設計:
  - data/local/meta/date_coverage/{YYYY}/{YYYYMMDD}.json に
    「その開催日の各カテゴリに何レース分のデータが GCS に存在するか」を記録。
  - スクレイピングタスク完了時に呼び出し → GCS スキャン不要でモニターページを更新。
  - /api/coverage-calendar はローカルインデックスを読むだけで応答 (GCS 課金ゼロ)。

ファイル形式:
  {
    "date": "20260613",
    "total_races": 36,
    "race_ids": ["202602010101", ...],
    "categories": {
      "race_shutuba": 36,
      "race_result": 36,
      ...
    },
    "updated_at": "2026-06-15T21:45:00+09:00",
    "schema_version": 1
  }

更新フック:
  - auto_scrape.py: raceday-evening / raceday-eve / weekly-update 完了時
  - backfill.py: 各日処理後
  - 手動/初期: python -m src.scraper.date_coverage build [--year YYYY]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.scraper.storage import HybridStorage

logger = logging.getLogger("scraper.date_coverage")

# プロジェクトルート: src/scraper/date_coverage.py → src/scraper/ → src/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

COVERAGE_DIR = _PROJECT_ROOT / "data/local/meta/date_coverage"
RACE_LIST_DIR = _PROJECT_ROOT / "data/page_reference/race_lists"
SCHEMA_VERSION = 1

# モニタリング対象カテゴリ（レース単位）
# scrape_process.md の行定義に完全準拠。smartrc は中止のため除外。
TRACK_CATEGORIES: list[str] = [
    # ── 出走前データ ──────────────────────────────────
    "race_shutuba",        # 出馬表HTML            (SLA 1,3)
    "race_shutuba_meta",   # レース情報HTML         (SLA 1,3 / 派生)
    "race_index",          # レースタイム指数HTML   (SLA 1)
    "race_paddock",        # レースパドックHTML     (SLA 3)
    "race_odds",           # レースオッズHTML       (SLA 3)
    # ── レース結果（速報）─────────────────────────────
    "race_result_on_time",         # レース結果HTML     (SLA 4,5)
    "race_result_on_time_payoff",  # レース払戻HTML     (SLA 4,5 / 派生)
    "race_result_on_time_lap",     # レースラップHTML   (SLA 4,5 / 派生)
    "race_result_on_time_corner",  # レース通過順位HTML (SLA 4,5 / 派生)
    # ── レース結果（確定）─────────────────────────────
    "race_result",          # レース結果DB          (SLA 6)
    "race_result_meta",     # レース情報DB           (SLA 6 / 派生)
    "race_result_payoff",   # レース払戻DB           (SLA 6 / 派生)
    "race_result_track",    # レース馬場情報DB       (SLA 6 / 派生)
    "race_result_corner",   # レース通過順位DB       (SLA 6 / 派生)
    "race_result_lap_times",# レースラップDB         (SLA 6 / 派生)
    "race_result_lap",      # レース個別ラップDB     (SLA 6)
    # 走行データ（タイム指数）は翌週金曜18:00公開 → SLA 6 相当
    "race_barometer",       # 走行データ/タイム指数  (SLA 6 / db.netkeiba.com 翌週金曜公開)
]

# 独立 blob を持たず親カテゴリに内包される派生カテゴリ
DERIVED_CATEGORY_PARENT: dict[str, str] = {
    "race_shutuba_meta": "race_shutuba",
    "race_result_on_time_payoff": "race_result_on_time",
    "race_result_on_time_lap": "race_result_on_time",
    "race_result_on_time_corner": "race_result_on_time",
    "race_result_meta": "race_result",
    "race_result_payoff": "race_result",
    "race_result_track": "race_result",
    "race_result_corner": "race_result",
    "race_result_lap_times": "race_result",
}


def apply_derived_category_na(row: dict[str, bool | None]) -> dict[str, bool | None]:
    """派生カテゴリ欠損を親が充足なら N/A (None) に降格。"""
    out = dict(row)
    for derived, parent in DERIVED_CATEGORY_PARENT.items():
        if out.get(derived) is False and out.get(parent) is True:
            out[derived] = None
    return out


NOT_AVAILABLE_DIR = _PROJECT_ROOT / "data/local/meta/not_available"


def _now_jst_iso() -> str:
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).isoformat(timespec="seconds")


def _coverage_path(date: str) -> Path:
    """data/local/meta/date_coverage/{YYYY}/{YYYYMMDD}.json"""
    return COVERAGE_DIR / date[:4] / f"{date}.json"


def _not_available_path(category: str, year: str) -> Path:
    """data/local/meta/not_available/{category}/{year}.json"""
    return NOT_AVAILABLE_DIR / category / f"{year}.json"


def load_not_available(category: str, year: str) -> set[str]:
    """指定カテゴリ・年の「取得試行したが存在しない」race_id セットを返す。"""
    path = _not_available_path(category, year)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("race_ids", []))
    except Exception as e:
        logger.warning("not_available load 失敗 [%s/%s]: %s", category, year, e)
        return set()


def record_not_available(
    category: str, race_id: str, reason: str = "empty_response"
) -> None:
    """指定カテゴリ・レースIDの「データ取得試行済みだが存在しない」マーカーをローカルに保存する。

    data/local/meta/not_available/{category}/{year}.json へ追記（重複なし）。
    """
    year = race_id[:4] if race_id and len(race_id) >= 4 else ""
    if not year:
        return
    path = _not_available_path(category, year)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: set[str] = set()
    meta: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            existing = set(data.get("race_ids", []))
            meta = data.get("_meta", {})
        except Exception:
            pass

    if race_id in existing:
        return  # 既に記録済み

    existing.add(race_id)
    payload = {
        "category": category,
        "year": year,
        "race_ids": sorted(existing),
        "count": len(existing),
        "updated_at": _now_jst_iso(),
        "_meta": meta,
    }
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        logger.info("N/A マーカー記録: %s/%s (%s)", category, race_id, reason)
    except Exception as e:
        logger.warning("N/A マーカー保存失敗 [%s/%s]: %s", category, race_id, e)


def load_date_coverage(date: str) -> dict | None:
    """ローカルの coverage ファイルを読む。存在しない場合は None。"""
    path = _coverage_path(date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("coverage load 失敗 [%s]: %s", date, e)
        return None


def load_year_coverage(year: int) -> dict[str, dict]:
    """年別の全日カバレッジを {YYYYMMDD: coverage_dict} 形式で返す。"""
    year_dir = COVERAGE_DIR / str(year)
    if not year_dir.exists():
        return {}
    result: dict[str, dict] = {}
    for path in sorted(year_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            result[path.stem] = data
        except Exception:
            pass
    return result


def _load_race_ids_for_date(date: str) -> list[str]:
    """race_lists/{date}.json からレースIDリストを返す。"""
    path = RACE_LIST_DIR / f"{date}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            r["race_id"]
            for r in data.get("races", [])
            if isinstance(r, dict) and "race_id" in r
        ]
    except Exception as e:
        logger.warning("race_list 読み込み失敗 [%s]: %s", date, e)
        return []


def update_date_coverage(date: str, storage: "HybridStorage") -> dict:
    """
    指定日のカバレッジを GCS 問い合わせで計算してローカルに保存する。

    内部では batch_list_blobs(cat, year) を使うため、
    同一 year の複数日をまとめて更新する場合は
    update_year_coverage() の方が GCS 呼び出し回数を抑えられる。
    """
    race_ids = _load_race_ids_for_date(date)
    if not race_ids:
        logger.debug("race_list なし、coverage スキップ: %s", date)
        return {}

    year = date[:4]
    race_id_set = set(race_ids)
    categories_count: dict[str, int] = {}

    for cat in TRACK_CATEGORIES:
        try:
            all_keys = set(storage.batch_list_blobs(cat, year).keys())
            categories_count[cat] = len(race_id_set & all_keys)
        except Exception as e:
            logger.warning("  %s カバレッジ取得エラー: %s", cat, e)
            categories_count[cat] = -1

    coverage = {
        "date": date,
        "total_races": len(race_ids),
        "race_ids": race_ids,
        "categories": categories_count,
        "updated_at": _now_jst_iso(),
        "schema_version": SCHEMA_VERSION,
    }

    _write_coverage(date, coverage)
    return coverage


def _build_date_race_id_map_from_gcs(
    year_str: str, storage: "HybridStorage"
) -> dict[str, list[str]]:
    """
    GCS の race_shutuba キーから {date: [race_id, ...]} を構築する。
    ローカル race_lists が空の年（2022-2025 等）に使用する。
    """
    try:
        shutuba_keys = list(storage.batch_list_blobs("race_shutuba", year_str).keys())
    except Exception as e:
        logger.warning("race_shutuba キー取得失敗 [%s]: %s", year_str, e)
        return {}

    if not shutuba_keys:
        return {}

    logger.info("  GCS race_shutuba から日付マップ構築中: %d 件 / %s", len(shutuba_keys), year_str)

    date_map: dict[str, list[str]] = {}
    # 全件ロードは高コストなので、存在するキーを race_shutuba データの `date` フィールドで分類
    # バッチサイズで並列ロードして集計する
    import concurrent.futures

    def _load_date(rid: str) -> tuple[str, str] | None:
        try:
            d = storage.load("race_shutuba", rid)
            if d:
                raw_date = d.get("date") or ""
                dt = raw_date.replace("-", "")[:8]
                if len(dt) == 8:
                    return rid, dt
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exe:
        for res in exe.map(_load_date, shutuba_keys):
            if res:
                rid, dt = res
                date_map.setdefault(dt, []).append(rid)

    return date_map


def update_year_coverage(
    year: int,
    storage: "HybridStorage",
    dates: list[str] | None = None,
) -> dict[str, dict]:
    """
    年全体（または指定日リスト）のカバレッジを効率的に一括更新する。

    batch_list_blobs を各カテゴリ 1 回呼ぶだけで、
    その年の全 race_id の存在有無を把握できる。

    ローカル race_lists が空の場合は GCS race_shutuba から日付マップを構築する。

    Args:
        year:    対象年
        storage: HybridStorage インスタンス
        dates:   指定する場合はその日だけ更新（None = 年全体）

    Returns:
        {date: coverage_dict}
    """
    year_str = str(year)

    # 対象日の race_ids を収集
    if dates:
        target_dates = [d for d in dates if d[:4] == year_str]
    else:
        target_dates = sorted(
            p.stem for p in RACE_LIST_DIR.glob(f"{year_str}*.json")
        )

    if not target_dates:
        return {}

    date_race_ids: dict[str, list[str]] = {
        d: _load_race_ids_for_date(d) for d in target_dates
    }
    date_race_ids = {d: ids for d, ids in date_race_ids.items() if ids}

    # ローカル race_lists が空の場合、GCS race_shutuba から日付マップを補完
    if not date_race_ids and storage is not None:
        logger.info(
            "ローカル race_list が空のため GCS shutuba から日付マップを構築: %s", year_str
        )
        date_race_ids = _build_date_race_id_map_from_gcs(year_str, storage)
        if dates:
            date_race_ids = {d: v for d, v in date_race_ids.items() if d in set(dates)}

    if not date_race_ids:
        return {}

    logger.info(
        "coverage 一括更新: %d 日 / %d 年 (カテゴリ %d)",
        len(date_race_ids), year, len(TRACK_CATEGORIES),
    )

    # カテゴリごとに年全体のキーを 1 回だけ取得
    cat_keys: dict[str, set[str]] = {}
    for cat in TRACK_CATEGORIES:
        try:
            cat_keys[cat] = set(storage.batch_list_blobs(cat, year_str).keys())
        except Exception as e:
            logger.warning("  %s キー取得エラー: %s", cat, e)
            cat_keys[cat] = set()

    results: dict[str, dict] = {}
    for date, race_ids in date_race_ids.items():
        race_id_set = set(race_ids)
        categories_count = {
            cat: len(race_id_set & cat_keys[cat]) for cat in TRACK_CATEGORIES
        }
        coverage = {
            "date": date,
            "total_races": len(race_ids),
            "race_ids": race_ids,
            "categories": categories_count,
            "updated_at": _now_jst_iso(),
            "schema_version": SCHEMA_VERSION,
        }
        _write_coverage(date, coverage)
        results[date] = coverage

    logger.info("coverage 一括更新完了: %d 日", len(results))
    return results


def _build_year_from_gcs(year: int, storage: "HybridStorage") -> dict[str, dict]:
    """
    GCS race_shutuba から日付マップを構築して年全体を再構築する。
    ローカル race_lists が空/不足している履歴年向け。

    処理順:
    1. batch_list_blobs('race_shutuba', year) で全 race_id を取得
    2. 各 race_id の shutuba データをロードして date フィールドから日付を集計
    3. update_year_coverage で per-category カバレッジを計算・保存
    """
    year_str = str(year)
    logger.info("GCS full rebuild: %s", year_str)

    gcs_date_map = _build_date_race_id_map_from_gcs(year_str, storage)
    if not gcs_date_map:
        logger.warning("GCS date map が空: %s", year_str)
        return {}

    logger.info("  検出された開催日: %d 日", len(gcs_date_map))

    # ローカル race_lists を GCS データで補完（既存 races があれば優先しない）
    local_map: dict[str, list[str]] = {}
    for dt, race_ids in gcs_date_map.items():
        local_ids = _load_race_ids_for_date(dt)
        local_map[dt] = local_ids if local_ids else race_ids

    # cat_keys を取得（年単位 1 回）
    cat_keys: dict[str, set[str]] = {}
    for cat in TRACK_CATEGORIES:
        try:
            cat_keys[cat] = set(storage.batch_list_blobs(cat, year_str).keys())
        except Exception as e:
            logger.warning("  %s キー取得エラー: %s", cat, e)
            cat_keys[cat] = set()

    results: dict[str, dict] = {}
    for date, race_ids in sorted(local_map.items()):
        race_id_set = set(race_ids)
        coverage = {
            "date": date,
            "total_races": len(race_ids),
            "race_ids": race_ids,
            "categories": {cat: len(race_id_set & cat_keys[cat]) for cat in TRACK_CATEGORIES},
            "updated_at": _now_jst_iso(),
            "schema_version": SCHEMA_VERSION,
        }
        _write_coverage(date, coverage)
        results[date] = coverage

    logger.info("GCS full rebuild 完了: %s → %d 日", year_str, len(results))
    return results


def _write_coverage(date: str, coverage: dict) -> None:
    out_path = _coverage_path(date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.write_text(
            json.dumps(coverage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "coverage 保存: %s (%d R, %d カテゴリ)",
            date,
            coverage.get("total_races", 0),
            len(coverage.get("categories", {})),
        )
    except Exception as e:
        logger.error("coverage 保存失敗 [%s]: %s", date, e)


# ---------------------------------------------------------------------------
# CLI: python -m src.scraper.date_coverage build [--year YYYY] [--date YYYYMMDD]
# ---------------------------------------------------------------------------
def _cli_build(args: list[str]) -> None:
    """全履歴 or 指定年の coverage index を一括構築する CLI。"""
    import argparse
    import os

    from src.utils.keiba_logging import script_basic_config

    script_basic_config()

    parser = argparse.ArgumentParser(
        prog="python -m src.scraper.date_coverage",
        description="date_coverage index を一括構築する",
    )
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--year", type=int, help="対象年 (省略 = 全年)")
    parser.add_argument("--date", help="特定日 YYYYMMDD (単日更新)")
    parser.add_argument(
        "--years-from", type=int, default=2020, help="開始年 (default: 2020)"
    )
    parser.add_argument(
        "--use-gcs-fallback",
        action="store_true",
        default=False,
        help="ローカル race_lists が不足している履歴年を GCS shutuba から全件再構築する",
    )
    opts = parser.parse_args(args)

    with open(".env") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    from src.scraper.storage import HybridStorage

    storage = HybridStorage()

    if opts.date:
        y = int(opts.date[:4])
        logger.info("単日更新: %s", opts.date)
        update_year_coverage(y, storage, dates=[opts.date])
        return

    if opts.year:
        years = [opts.year]
    else:
        now_year = datetime.now().year
        years = list(range(opts.years_from, now_year + 1))

    logger.info("一括構築開始: %s (gcs_fallback=%s)", years, opts.use_gcs_fallback)
    for y in years:
        logger.info("=== %d 年 ===", y)
        if opts.use_gcs_fallback:
            _build_year_from_gcs(y, storage)
        else:
            update_year_coverage(y, storage)

    logger.info("一括構築完了")


if __name__ == "__main__":
    import sys

    _cli_build(sys.argv[1:])

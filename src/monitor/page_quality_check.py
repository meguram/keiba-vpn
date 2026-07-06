"""
ページ品質チェック — 毎日 08:00 実行

各スクレイピングカテゴリの代表URLに対してスクレイパを実行し、
期待スキーマと照合して safe / caution / out を判定する。

タグの定義:
  safe    : 期待フィールドが全て存在し、未知の新フィールドもない
  caution : 期待フィールドは全て存在するが、新しい未知フィールドが追加されていた
            （後方互換性があるが、スキーマ定義の更新を検討すること）
  out     : 期待フィールドが欠損、型エラー、パースエラー、または HTTP エラー

結果は data/local/page_quality/latest.json に保存し、
過去30件は page_quality/history/ に YYYY-MM-DD.json として保持する。

CLI:
    python -m src.monitor.page_quality_check            # 全カテゴリ実行
    python -m src.monitor.page_quality_check --dry-run  # 実行せず定義を表示
    python -m src.monitor.page_quality_check --category race_shutuba
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
import traceback
from datetime import datetime, date
from pathlib import Path
from typing import Any

logger = logging.getLogger("monitor.page_quality_check")


# ── ストレージパス ──────────────────────────────────────────────────────────

_BASE = Path(__file__).parent.parent.parent
RESULT_DIR = _BASE / "data" / "local" / "page_quality"
LATEST_PATH = RESULT_DIR / "latest.json"
HISTORY_DIR = RESULT_DIR / "history"

# ── テスト定義 ─────────────────────────────────────────────────────────────
#
# 各エントリで使用する固定サンプル ID:
#   RACE_ID   : 202309030811  (2023年宝塚記念 — 確定済みの過去レース)
#   HORSE_ID  : 2019105219    (イクイノックス — horse/result/ped/training が揃っている)
#   DATE      : 20250518      (2025年5月18日開催 — race_lists 確認用)
#
# _fetch(cfg) の戻り値は (data: dict | None, extra_keys: set[str], error: str | None)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RACE_ID = "202309030811"
SAMPLE_HORSE_ID = "2019105219"
SAMPLE_DATE = "20250518"

# known_top_keys: スキーマ定義（top_required + top_optional）に含まれる全キー。
# 実データにこれ以外のキーが存在すると caution を発火する。
# ここでは代表的なキーを列挙（schemas.py の SCHEMAS から自動取得も可能）。

_CHECK_TARGETS: list[dict[str, Any]] = [
    # ──────────────── レース系 ────────────────
    {
        "category": "race_shutuba",
        "label": "出馬表",
        "url": f"https://race.netkeiba.com/race/shutuba.html?race_id={SAMPLE_RACE_ID}",
        "fetch_mode": "html",
        "parser": "RaceCardParser",
        "parse_kwargs": {"race_id": SAMPLE_RACE_ID},
        "requires_login": False,
    },
    {
        "category": "race_result",
        "label": "確定結果（DB）",
        "url": f"https://db.netkeiba.com/race/{SAMPLE_RACE_ID}/",
        "fetch_mode": "html",
        "parser": "RaceResultParser",
        "parse_kwargs": {"race_id": SAMPLE_RACE_ID},
        "requires_login": False,
    },
    {
        "category": "race_result_on_time",
        "label": "速報結果（race.netkeiba）",
        "url": f"https://race.netkeiba.com/race/result.html?race_id={SAMPLE_RACE_ID}",
        "fetch_mode": "html",
        "parser": "RaceResultOnTimeParser",
        "parse_kwargs": {"race_id": SAMPLE_RACE_ID},
        "requires_login": False,
    },
    {
        "category": "race_result_lap",
        "label": "ラップ（DB）",
        "url": f"https://db.netkeiba.com/race/{SAMPLE_RACE_ID}/",
        "fetch_mode": "html",
        "parser": "RaceResultParserForLap",
        "parse_kwargs": {"race_id": SAMPLE_RACE_ID},
        "requires_login": False,
    },
    {
        "category": "race_lists",
        "label": "レース一覧",
        "url": f"https://race.netkeiba.com/top/race_list.html?kaisai_date={SAMPLE_DATE}",
        "fetch_mode": "html",
        "parser": "RaceListParser",
        "parse_kwargs": {"date": SAMPLE_DATE},
        "requires_login": False,
    },
    # ──────────────── 馬系 ────────────────
    {
        "category": "horse_result",
        "label": "馬情報（horse_result）",
        "url": f"https://db.netkeiba.com/horse/{SAMPLE_HORSE_ID}/",
        "fetch_mode": "html",
        "parser": "HorseParser",
        "parse_kwargs": {"horse_id": SAMPLE_HORSE_ID},
        "requires_login": False,
    },
    {
        "category": "horse_pedigree_5gen",
        "label": "5世代血統",
        "url": f"https://db.netkeiba.com/horse/ped/{SAMPLE_HORSE_ID}/",
        "fetch_mode": "pedigree_5gen",
        "parser": "pedigree_5gen",
        "parse_kwargs": {"horse_id": SAMPLE_HORSE_ID},
        "requires_login": False,
    },
    {
        "category": "horse_training",
        "label": "調教",
        "url": f"https://db.netkeiba.com/horse/training.html?id={SAMPLE_HORSE_ID}",
        "fetch_mode": "html",
        "parser": "TrainingParser",
        "parse_kwargs": {"horse_id": SAMPLE_HORSE_ID},
        "requires_login": True,
    },
    # ──────────────── 馬場バロメーター（API） ────────────────
    {
        "category": "race_barometer",
        "label": "バロメーター（API）",
        "url": f"https://race.netkeiba.com/api/api_get_jra_odds.html?type=b4&race_id={SAMPLE_RACE_ID}",
        "fetch_mode": "api_barometer",
        "parser": "BarometerParser",
        "parse_kwargs": {"race_id": SAMPLE_RACE_ID},
        "requires_login": True,
    },
    # ──────────────── 追い切り ────────────────
    {
        "category": "race_oikiri",
        "label": "追い切り（レース）",
        "url": f"https://race.netkeiba.com/race/oikiri.html?race_id={SAMPLE_RACE_ID}",
        "fetch_mode": "html",
        "parser": "OikiriParser",
        "parse_kwargs": {"race_id": SAMPLE_RACE_ID},
        "requires_login": True,
    },
]


# ── フェッチ + パース ──────────────────────────────────────────────────────

def _get_client():
    """遅延インポート（テスト時に余計なログを抑制）。"""
    from src.scraper.client import NetkeibaClient
    return NetkeibaClient


def _fetch_and_parse(cfg: dict[str, Any], client) -> tuple[dict | None, str | None]:
    """
    指定 cfg に従ってページを取得・パースする。

    Returns:
        (data, error_msg)  ― error_msg は None なら成功
    """
    try:
        mode = cfg["fetch_mode"]

        if mode == "api_barometer":
            from src.scraper.parsers import BarometerParser
            data = BarometerParser().parse_from_api(client._session, **cfg["parse_kwargs"])

        elif mode == "pedigree_5gen":
            # 独自関数を使う血統パーサー
            from src.research.pedigree.pedigree_similarity import parse_blood_table_5gen
            from src.scripts.scraping.scrape_pedigree_5gen import build_pedigree_record
            html = client.fetch(cfg["url"])
            if not html:
                return None, "empty response"
            horse_id = cfg["parse_kwargs"]["horse_id"]
            ancestors = parse_blood_table_5gen(html)
            if not ancestors:
                return None, "ancestors empty"
            data = build_pedigree_record(horse_id, ancestors, source="page_quality_check")

        elif mode == "html":
            parser_name = cfg["parser"]
            # 遅延インポートで parser クラスを取得
            if parser_name == "RaceCardParser":
                from src.scraper.parsers import RaceCardParser as PC
            elif parser_name in ("RaceResultParser", "RaceResultParserForLap"):
                from src.scraper.parsers import RaceResultParser as PC
            elif parser_name == "RaceResultOnTimeParser":
                from src.scraper.parsers import RaceResultOnTimeParser as PC
            elif parser_name == "RaceListParser":
                from src.scraper.parsers import RaceListParser as PC
            elif parser_name == "HorseParser":
                from src.scraper.parsers import HorseParser as PC
            elif parser_name == "OikiriParser":
                from src.scraper.parsers import OikiriParser as PC
            elif parser_name == "TrainingParser":
                from src.scraper.parsers import TrainingParser as PC
            else:
                return None, f"unknown parser: {parser_name}"

            html = client.fetch(cfg["url"])
            if not html:
                return None, "empty response"
            data = PC().parse(html, **cfg["parse_kwargs"])
        else:
            return None, f"unknown fetch_mode: {mode}"

        return data, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ── スキーマ照合 ───────────────────────────────────────────────────────────

def _known_keys_for(category: str) -> set[str]:
    """schemas.py SCHEMAS から category の既知キー一覧を返す。"""
    try:
        from src.scraper.schemas import SCHEMAS
        schema = SCHEMAS.get(category, {})
        known: set[str] = set()
        known.update(schema.get("top_required", {}).keys())
        known.update(schema.get("top_optional", {}).keys())
        # 内部メタは常に許容
        known.update({"_meta", "_raw", "scraped_at"})
        return known
    except Exception:
        return set()


def _detect_extra_keys(data: dict, category: str) -> list[str]:
    """既知スキーマにないトップレベルキーを返す（caution 判定用）。"""
    known = _known_keys_for(category)
    if not known:
        return []
    return [k for k in data if k not in known]


def _classify(
    data: dict | None,
    error: str | None,
    category: str,
) -> dict[str, Any]:
    """
    fetch+parse 結果からタグ（safe/caution/out）と詳細を返す。

    Returns:
        {
            "tag":        "safe" | "caution" | "out",
            "passed":     bool,
            "error":      str | None,
            "extra_keys": list[str],
            "validation": dict,   # schemas.validate() の生結果
        }
    """
    if error or data is None:
        return {
            "tag": "out",
            "passed": False,
            "error": error or "data is None",
            "extra_keys": [],
            "validation": {},
        }

    try:
        from src.scraper.schemas import validate
        report = validate(category, data)
    except Exception as exc:
        report = {"passed": False, "error": str(exc)}

    passed = report.get("passed", False)
    extra_keys = _detect_extra_keys(data, category)

    if not passed:
        tag = "out"
    elif extra_keys:
        tag = "caution"
    else:
        tag = "safe"

    return {
        "tag": tag,
        "passed": passed,
        "error": None,
        "extra_keys": extra_keys,
        "validation": report,
    }


# ── 実行エンジン ──────────────────────────────────────────────────────────

def run_checks(
    categories: list[str] | None = None,
    interval_sec: float = 2.0,
) -> dict[str, Any]:
    """
    全（または指定）カテゴリの品質チェックを実行し、結果 dict を返す。

    Args:
        categories: None = 全カテゴリ, list[str] = 絞り込み
        interval_sec: リクエスト間隔（秒）
    """
    from src.utils.project_env import load_project_dotenv
    load_project_dotenv()

    NetkeibaClient = _get_client()
    client = NetkeibaClient(auto_login=True, interval=interval_sec)

    targets = _CHECK_TARGETS
    if categories:
        targets = [t for t in targets if t["category"] in categories]

    results: list[dict[str, Any]] = []
    for cfg in targets:
        cat = cfg["category"]
        logger.info("チェック開始: %s (%s)", cat, cfg["url"])
        t0 = time.monotonic()

        data, error = _fetch_and_parse(cfg, client)
        elapsed = round(time.monotonic() - t0, 2)

        classified = _classify(data, error, cat)

        entry = {
            "category": cat,
            "label": cfg["label"],
            "url": cfg["url"],
            "tag": classified["tag"],
            "passed": classified["passed"],
            "error": classified["error"],
            "extra_keys": classified["extra_keys"],
            "elapsed_sec": elapsed,
            "requires_login": cfg.get("requires_login", False),
            "validation_summary": {
                "top_missing": classified["validation"].get("top_missing", []),
                "top_type_errors": [e.get("field") for e in classified["validation"].get("top_type_errors", [])],
                "top_constraint_errors": [e.get("field") for e in classified["validation"].get("top_constraint_errors", [])],
                "entry_count": classified["validation"].get("entry_count", 0),
            },
        }
        results.append(entry)
        logger.info(
            "  -> %s  elapsed=%.1fs  extra=%s  missing=%s",
            classified["tag"].upper(),
            elapsed,
            classified["extra_keys"],
            classified["validation"].get("top_missing", []),
        )
        time.sleep(interval_sec)

    summary_tag = "safe"
    for r in results:
        if r["tag"] == "out":
            summary_tag = "out"
            break
        if r["tag"] == "caution":
            summary_tag = "caution"

    return {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "summary_tag": summary_tag,
        "total": len(results),
        "safe_count": sum(1 for r in results if r["tag"] == "safe"),
        "caution_count": sum(1 for r in results if r["tag"] == "caution"),
        "out_count": sum(1 for r in results if r["tag"] == "out"),
        "results": results,
    }


# ── 結果保存 ──────────────────────────────────────────────────────────────

def save_result(result: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    today_str = date.today().isoformat()
    history_path = HISTORY_DIR / f"{today_str}.json"

    LATEST_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 30日超の古いファイルを削除
    all_hist = sorted(HISTORY_DIR.glob("*.json"))
    for old in all_hist[:-30]:
        old.unlink(missing_ok=True)

    logger.info("結果保存: %s / %s", LATEST_PATH, history_path)


# ── エントリポイント ──────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ページ品質チェック")
    parser.add_argument("--dry-run", action="store_true", help="定義を表示して終了")
    parser.add_argument("--category", nargs="*", help="実行するカテゴリ（未指定=全）")
    parser.add_argument("--interval", type=float, default=2.0, help="リクエスト間隔（秒）")
    args = parser.parse_args()

    if args.dry_run:
        print(f"{'category':<30} {'label':<25} {'url'}")
        for t in _CHECK_TARGETS:
            print(f"  {t['category']:<28} {t['label']:<23} {t['url']}")
        return

    try:
        result = run_checks(categories=args.category, interval_sec=args.interval)
        save_result(result)

        # 終了ステータスの要約をログに出力
        tag = result["summary_tag"]
        logger.info(
            "=== page_quality_check 完了: %s  safe=%d caution=%d out=%d ===",
            tag.upper(),
            result["safe_count"],
            result["caution_count"],
            result["out_count"],
        )
        if tag == "out":
            out_cats = [r["category"] for r in result["results"] if r["tag"] == "out"]
            logger.warning("OUT カテゴリ: %s", ", ".join(out_cats))
        if tag == "caution":
            caut_cats = [r["category"] for r in result["results"] if r["tag"] == "caution"]
            logger.warning("CAUTION カテゴリ: %s", ", ".join(caut_cats))

    except Exception:
        logger.error("page_quality_check 実行中に例外:\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

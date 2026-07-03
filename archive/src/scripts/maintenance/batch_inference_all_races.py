#!/usr/bin/env python3
"""モデル情報が揃っているタスクについて、過去全レースの推論をバッチ実行する。

例:
  python3 -m src.scripts.maintenance.batch_inference_all_races --dry-run
  python3 -m src.scripts.maintenance.batch_inference_all_races --models all
  python3 -m src.scripts.maintenance.batch_inference_all_races --models race_predictions,final_odds,tracking_difficulty
  python3 -m src.scripts.maintenance.batch_inference_all_races --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from src.pipeline.inference.final_odds_service import (  # noqa: E402
    build_final_odds_response,
    load_cached_response as load_final_odds_cached,
    save_cached_response as save_final_odds_cached,
)
from src.pipeline.inference.race_prediction_service import (  # noqa: E402
    build_race_prediction_response,
    load_cached as load_race_pred_cached,
    save_cached as save_race_pred_cached,
)
from src.pipeline.inference.tracking_difficulty_service import (  # noqa: E402
    build_tracking_difficulty_response,
    save_cached_response as save_tracking_cached,
)
from src.pipeline.inference.tracking_difficulty_store import (  # noqa: E402
    exists_local as tracking_exists_local,
    update_index_meta as tracking_update_index_meta,
)
from src.pipeline.mlflow.catalog import MODEL_CATALOG, ModelLifecycle  # noqa: E402
from src.scraper.storage import HybridStorage  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("BatchInferenceAll")


def available_models() -> dict[str, bool]:
    root = _ROOT
    out: dict[str, bool] = {}
    out["race_predictions"] = (root / "models" / "keiba_model.pkl").is_file()
    out["final_odds"] = (root / "models" / "final_odds_bundle.json").is_file()
    # 追走難度は Booster 未配置でもヒューリスティックで推論可能
    spec_td = MODEL_CATALOG.get("tracking_difficulty")
    out["tracking_difficulty"] = bool(
        spec_td
        and spec_td.lifecycle == ModelLifecycle.ACTIVE
        and (
            (root / "models" / "tracking_difficulty.lgb").is_file()
            or (root / "models" / "tracking_difficulty.txt").is_file()
        )
    )
    if spec_td and spec_td.lifecycle == ModelLifecycle.ACTIVE:
        out["tracking_difficulty"] = True
    out["pace_predictor"] = (root / "models" / "pace_predictor").is_dir()
    return out


def collect_race_ids(storage: HybridStorage, *, from_lists: bool) -> list[str]:
    if from_lists:
        lists_dir = _ROOT / "data" / "calculated_data" / "race_lists"
        if not lists_dir.is_dir():
            lists_dir = _ROOT / "data" / "page_reference" / "race_lists"
        ids: list[str] = []
        for f in sorted(lists_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                for r in data.get("races", []):
                    rid = r.get("race_id") or r.get("id")
                    if rid:
                        ids.append(str(rid))
            except Exception:
                continue
        return sorted(set(ids))
    keys = storage.list_keys("race_shutuba") or []
    return sorted(k.replace(".json", "") for k in keys if k)


def _has_shutuba(storage: HybridStorage, race_id: str) -> bool:
    return bool(storage.load("race_shutuba", race_id))


def run_one(
    storage: HybridStorage,
    race_id: str,
    model: str,
    *,
    skip_existing: bool,
    allow_scrape: bool,
) -> str:
    """Returns: ok | skip | fail"""
    if model == "race_predictions":
        if skip_existing and load_race_pred_cached(storage, race_id):
            return "skip"
        payload = build_race_prediction_response(
            race_id, storage, allow_scrape=allow_scrape
        )
        if payload.get("status") == "success" and payload.get("predictions"):
            save_race_pred_cached(storage, race_id, payload, source="batch_inference")
            return "ok"
        return "fail"

    if model == "final_odds":
        if skip_existing and load_final_odds_cached(storage, race_id):
            return "skip"
        payload = build_final_odds_response(
            race_id, storage, allow_scrape=allow_scrape
        )
        if payload.get("entries"):
            save_final_odds_cached(storage, race_id, payload, source="batch_inference")
            return "ok"
        return "fail"

    if model == "tracking_difficulty":
        if skip_existing and tracking_exists_local(race_id):
            return "skip"
        payload = build_tracking_difficulty_response(
            race_id,
            storage,
            allow_scrape=allow_scrape,
            pre_race_only=True,
        )
        if payload.get("entries"):
            save_tracking_cached(storage, race_id, payload, source="batch_inference")
            return "ok"
        return "fail"

    raise ValueError(f"未知のモデル: {model}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="race_predictions,final_odds,tracking_difficulty",
        help="カンマ区切り、または all",
    )
    parser.add_argument("--from-race-lists", action="store_true", help="race_lists から ID 収集")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--scrape", action="store_true", help="データ欠損時にスクレイプ")
    parser.add_argument("--limit", type=int, default=0, help="テスト用件数上限")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--race-id", action="append", default=[], help="単体テスト用")
    args = parser.parse_args(argv)

    avail = available_models()
    logger.info("利用可能モデル: %s", {k: v for k, v in avail.items() if v})

    if args.models.strip().lower() == "all":
        models = [m for m, ok in avail.items() if ok and m != "pace_predictor"]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    models = [m for m in models if avail.get(m)]
    if not models:
        logger.error("実行可能なモデルがありません")
        return 1

    storage = HybridStorage()
    if args.race_id:
        race_ids = args.race_id
    else:
        race_ids = collect_race_ids(storage, from_lists=args.from_race_lists)
    if args.limit > 0:
        race_ids = race_ids[: args.limit]

    logger.info(
        "対象: models=%s races=%d skip_existing=%s scrape=%s",
        models,
        len(race_ids),
        args.skip_existing,
        args.scrape,
    )

    if args.dry_run:
        return 0

    stats = {m: {"ok": 0, "skip": 0, "fail": 0} for m in models}
    t_all = time.perf_counter()

    for i, rid in enumerate(race_ids, 1):
        if not _has_shutuba(storage, rid):
            for m in models:
                stats[m]["fail"] += 1
            continue
        for m in models:
            try:
                st = run_one(
                    storage,
                    rid,
                    m,
                    skip_existing=args.skip_existing,
                    allow_scrape=args.scrape,
                )
                stats[m][st] += 1
            except Exception as e:
                logger.warning("FAIL %s [%s]: %s", rid, m, e)
                stats[m]["fail"] += 1
        if i % 50 == 0 or i == len(race_ids):
            logger.info(
                "[%d/%d] %s",
                i,
                len(race_ids),
                " | ".join(
                    f"{m}: ok={stats[m]['ok']} skip={stats[m]['skip']} fail={stats[m]['fail']}"
                    for m in models
                ),
            )

    elapsed = round(time.perf_counter() - t_all, 1)
    if "tracking_difficulty" in models:
        tracking_update_index_meta(batch_source="batch_inference_all_races")
    logger.info("完了 %.1fs — %s", elapsed, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

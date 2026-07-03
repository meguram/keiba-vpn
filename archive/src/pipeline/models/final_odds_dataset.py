"""最終オッズ予測の学習データセット構築（年次 split）。"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

import pandas as pd

from src.pipeline.features.feature_builder import build_race_features
from src.pipeline.models.final_odds_features import enrich_final_odds_features
from src.pipeline.models.final_odds_progress import FinalOddsProgress

logger = logging.getLogger(__name__)

TRAIN_YEARS_DEFAULT = ("2020", "2021", "2022", "2023", "2024")
EVAL_YEARS_DEFAULT = ("2025",)

_LOAD_CATEGORIES = [
    ("race_shutuba", "race_card"),
    ("race_index", "speed_index"),
    ("race_shutuba_past", "shutuba_past"),
    # race_paddock は当日観察データ（前日18時時点で存在しない）ため除外
    # race_barometer はレース後ラップ計算値（ajax_race_result_horse_laptime）ため除外
    ("race_oikiri", "oikiri"),
]


_DATE_FORMATS = ("%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d")


def _parse_date(s: str) -> datetime | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _filter_horse_history_by_date(hr: dict, before_date_str: str) -> dict:
    """race_history を target 日付より前のみに絞る（未来レースのリーク防止）。"""
    target = _parse_date(before_date_str)
    if target is None:
        return hr
    filtered = [r for r in hr.get("race_history", []) if (d := _parse_date(r.get("date", ""))) is not None and d < target]
    return {**hr, "race_history": filtered}


def race_id_year(race_id: str) -> str:
    rid = str(race_id).strip()
    return rid[:4] if len(rid) >= 4 else ""


def _card_from_result(result: dict) -> dict:
    return {
        "race_id": result.get("race_id", ""),
        "race_name": result.get("race_name", ""),
        "venue": result.get("venue", ""),
        "surface": result.get("surface", ""),
        "distance": result.get("distance", 0),
        "direction": result.get("direction", ""),
        "weather": result.get("weather", ""),
        "track_condition": result.get("track_condition", ""),
        "field_size": result.get("field_size", 0),
        "date": result.get("date", ""),
        "entries": result.get("entries", []),
    }


def _assemble_race_data(storage: Any, race_id: str, result: dict) -> dict | None:
    card = storage.load("race_shutuba", race_id)
    if not card or not card.get("entries"):
        card = _card_from_result(result)
    if not card.get("entries"):
        return None

    race_data: dict[str, Any] = {
        "race_id": race_id,
        "race_card": card,
        "race_result": result,
    }
    for storage_cat, key in _LOAD_CATEGORIES:
        blob = storage.load(storage_cat, race_id)
        if blob:
            race_data[key] = blob

    # 対象レース日付（馬の履歴フィルタ用）
    race_date = result.get("date") or card.get("date", "")

    horses: dict[str, Any] = {}
    for e in card.get("entries", []):
        hid = e.get("horse_id", "")
        if not hid:
            continue
        hr = storage.load("horse_result", hid)
        if hr:
            # 対象レース以降の出走結果を除外してリークを防ぐ
            hr = _filter_horse_history_by_date(hr, race_date)
            horses[hid] = hr
    race_data["horses"] = horses
    return race_data


def _odds_labels_for_race(
    storage: Any,
    race_id: str,
    result: dict,
) -> dict[int, dict[str, float]]:
    """馬番 → {win_odds, place_min, place_max}（確定・発売終了オッズ）。"""
    labels: dict[int, dict[str, float]] = {}

    for e in result.get("entries", []):
        hn = int(e.get("horse_number") or 0)
        wo = float(e.get("odds") or 0)
        if hn > 0 and wo > 0:
            labels[hn] = {"win_odds": wo, "place_min": 0.0, "place_max": 0.0}

    odds = storage.load("race_odds", race_id)
    if odds:
        for e in odds.get("entries", []):
            hn = int(e.get("horse_number") or 0)
            if hn <= 0:
                continue
            pm = float(e.get("place_odds_min") or 0)
            px = float(e.get("place_odds_max") or 0)
            wo = float(e.get("win_odds") or 0)
            if hn not in labels and wo > 0:
                labels[hn] = {"win_odds": wo, "place_min": pm, "place_max": px}
            elif hn in labels:
                if pm > 0:
                    labels[hn]["place_min"] = pm
                if px > 0:
                    labels[hn]["place_max"] = px
                if wo > 0 and labels[hn]["win_odds"] <= 0:
                    labels[hn]["win_odds"] = wo

    for hn, lab in labels.items():
        wo = lab["win_odds"]
        if lab["place_min"] <= 0:
            lab["place_min"] = max(wo * 0.22, 1.0)
        if lab["place_max"] <= 0:
            lab["place_max"] = max(wo * 0.52, lab["place_min"] + 0.1)

    return labels


def build_final_odds_dataset(
    storage: Any,
    *,
    train_years: tuple[str, ...] = TRAIN_YEARS_DEFAULT,
    eval_years: tuple[str, ...] = EVAL_YEARS_DEFAULT,
    max_races: int | None = None,
    min_field_size: int = 5,
    progress: FinalOddsProgress | None = None,
) -> pd.DataFrame:
    """
    2020-2024 → split=train、2025 → split=eval の行を返す。

    各行は1頭。ラベルは確定単勝・複勝オッズ（log 変換列付き）。
    """
    train_set = set(train_years)
    eval_set = set(eval_years)
    target_years = train_set | eval_set

    progress = progress or FinalOddsProgress(enabled=False)
    progress.set_phase("scan", message="race_result キー一覧")

    keys = storage.list_keys("race_result")
    target_keys = [
        k.replace(".json", "")
        for k in sorted(keys)
        if race_id_year(k.replace(".json", "")) in target_years
    ]
    if max_races is not None:
        target_keys = target_keys[:max_races]

    progress.set_phase(
        "dataset",
        total=len(target_keys),
        message=f"特徴量+ラベル ({len(train_years)}年学習/{len(eval_years)}年評価)",
    )
    progress.update(meta={"target_races": len(target_keys), "all_keys": len(keys)})

    rows: list[dict] = []
    processed = 0
    skipped = 0

    for race_id in target_keys:
        year = race_id_year(race_id)
        result = storage.load("race_result", race_id)
        if not result or not result.get("entries"):
            skipped += 1
            continue

        fs = int(result.get("field_size") or len(result["entries"]))
        if fs < min_field_size:
            skipped += 1
            continue

        race_data = _assemble_race_data(storage, race_id, result)
        if not race_data:
            skipped += 1
            continue

        try:
            feat_df = build_race_features(race_data)
        except Exception as exc:
            logger.debug("feature skip %s: %s", race_id, exc)
            skipped += 1
            continue

        if feat_df.empty:
            skipped += 1
            continue

        feat_df = enrich_final_odds_features(feat_df)
        if "race_name" not in feat_df.columns:
            feat_df["race_name"] = result.get("race_name", "")

        labels = _odds_labels_for_race(storage, race_id, result)
        split = "train" if year in train_set else "eval" if year in eval_set else ""

        for _, row in feat_df.iterrows():
            hn = int(row.get("horse_number") or 0)
            lab = labels.get(hn)
            if not lab or lab["win_odds"] <= 0:
                continue
            d = row.to_dict()
            d["split"] = split
            d["data_year"] = year
            d["target_win_odds"] = lab["win_odds"]
            d["target_place_min"] = lab["place_min"]
            d["target_place_max"] = lab["place_max"]
            d["target_log_win_odds"] = math.log(lab["win_odds"])
            d["target_log_place_min"] = math.log(max(lab["place_min"], 1.0))
            d["target_log_place_max"] = math.log(max(lab["place_max"], 1.0))
            rows.append(d)

        processed += 1
        if processed % 10 == 0 or processed == len(target_keys):
            progress.update(
                current=processed,
                message=f"rows={len(rows):,} skip={skipped}",
                meta={"n_rows": len(rows), "n_skipped": skipped},
            )
        if processed % 500 == 0:
            logger.info("final_odds dataset: %d races, %d rows", processed, len(rows))

    progress.update(
        current=len(target_keys),
        message=f"dataset 完了 rows={len(rows):,}",
        meta={"n_rows": len(rows), "n_skipped": skipped, "n_processed": processed},
    )
    logger.info(
        "final_odds dataset done: races=%d skipped=%d rows=%d",
        processed,
        skipped,
        len(rows),
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

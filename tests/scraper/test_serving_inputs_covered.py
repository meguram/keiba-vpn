"""cron の整理で取得を減らしても、推論の入力になるカテゴリは、取得対象に残っていること。"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _serving_categories() -> set[str]:
    src = (ROOT / "src/pipeline/inference/race_prediction_service.py").read_text(encoding="utf-8")
    cats = set(re.findall(r'\("[a-z_]+",\s*"(race_[a-z_]+)"\)', src))
    cats |= set(re.findall(r'storage\.load\("(race_[a-z_]+)"', src))
    return cats


def _bundle_categories() -> set[str]:
    src = (ROOT / "src/scraper/run.py").read_text(encoding="utf-8")
    start = src.index("def scrape_race_all")
    body = src[start:start + 6000]
    return set(re.findall(r'\("(race_[a-z_]+)",\s+"(?:scrape_|_scrape_)', body))


def test_serving_inputs_are_still_fetched_by_the_race_bundle():
    serving, bundle = _serving_categories(), _bundle_categories()
    assert {"race_shutuba", "race_shutuba_past", "race_oikiri"} <= serving
    # 馬柱・追い切り・出馬表は前日取得(raceday-eve)でも取る。指数・パドック・偏差値は当日バンドル/週次で取る
    assert serving - {"race_shutuba"} <= bundle | {"race_shutuba_past", "race_oikiri"}, serving - bundle

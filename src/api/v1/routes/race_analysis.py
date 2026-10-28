"""レースの分析結果 API（めぐ分析）。出馬表 + 分析モジュールの結果を 1 本で返す。DB は使わない。"""

from __future__ import annotations

import logging
import re

from flask import Blueprint, jsonify

bp = Blueprint("v1_race_analysis", __name__)
logger = logging.getLogger(__name__)

_RACE_ID = re.compile(r"^\d{12}$")


@bp.get("/races/<race_id>/analysis")
def race_analysis(race_id: str):
    if not _RACE_ID.match(race_id):
        return jsonify({"error": "race_id が不正です"}), 400
    from src.research.race_analysis.service import analyze_race
    from src.scraper.storage import HybridStorage

    try:
        return jsonify(analyze_race(HybridStorage(), race_id))
    except Exception as exc:
        logger.exception("race-analysis failed: %s", race_id)
        return jsonify({"error": str(exc)}), 500

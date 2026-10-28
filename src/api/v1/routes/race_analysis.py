"""レースの分析結果 API（めぐ分析）。事前に計算して保存した結果を読んで返すだけ（表示のたびに計算しない）。DB は使わない。"""

from __future__ import annotations

import re

from flask import Blueprint, jsonify

bp = Blueprint("v1_race_analysis", __name__)

_RACE_ID = re.compile(r"^\d{12}$")
PENDING_MESSAGE = "このレースの分析はまだ用意できていません。出馬表の取得後に自動で集計され、表示できるようになります。"


@bp.get("/races/<race_id>/analysis")
def race_analysis(race_id: str):
    if not _RACE_ID.match(race_id):
        return jsonify({"error": "race_id が不正です"}), 400
    from src.research.race_analysis import store

    payload = store.read(race_id)
    if payload is None:
        return jsonify({"status": "pending", "message": PENDING_MESSAGE, "race": {"race_id": race_id}, "entries": [], "provisional_entries": False, "modules": []})
    payload.pop("fingerprint", None)
    return jsonify(payload)

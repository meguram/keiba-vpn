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
    from src.utils import public_policy as P

    if P.is_public_mode() and not P.actual_odds_allowed():
        from src.api.flask_auth import is_logged_in as is_developer_session

        if not is_developer_session():   # 実オッズ・人気・確定払戻（データで予想の該当馬）は開発者だけ
            from src.research.race_analysis.graded_picks import redact_actual_odds

            payload = redact_actual_odds(payload)
    from src.api.flask_auth import is_logged_in as _is_dev

    if not _is_dev():                    # 選んだ期間の回収率は、開発者以外には出さない（src/api/tier.py の strip_research_roi）
        from src.api.tier import strip_research_roi

        payload = strip_research_roi(payload)
    return jsonify(payload)

"""レースの分析結果 API（めぐ分析）。事前に計算して保存した結果を読んで返すだけ（表示のたびに計算しない）。DB は使わない。"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify

bp = Blueprint("v1_race_analysis", __name__)

_JST = timezone(timedelta(hours=9))
_RACE_ID = re.compile(r"^\d{12}$")
PENDING_MESSAGE = "このレースの分析はまだ用意できていません。出馬表の取得後に自動で集計され、表示できるようになります。"


PERSONA_KEYS = ("id", "view", "status", "label", "description", "figure", "notes")


def _race_start(payload: dict, doc: dict) -> datetime | None:
    """発走の時刻（JST）。めぐ分析の race を先に、欠ければ欄のファイルの start_time で補う。分からなければ None（出さない）。"""
    race = payload.get("race") or {}
    try:
        y, m, d = (int(x) for x in str(race.get("date") or "").split("-"))
    except ValueError:
        return None
    for t in (race.get("start_time"), doc.get("start_time")):
        try:
            hh, mm = (int(x) for x in str(t or "").split(":")[:2])
            return datetime(y, m, d, hh, mm, tzinfo=_JST)
        except ValueError:
            continue
    return None


def with_pedigree_persona(race_id: str, payload: dict, now: datetime | None = None, developer: bool | None = None) -> dict:
    """血統予想家の欄（別のファイル race_analysis/pedigree_persona/<race_id>.json）を合わせる。表示のときに計算しない。

    - 発走の時刻を過ぎたら出さない（結果の画面・過去のレースで、一覧と着順を並べない。会議の 1 号の条件 1）。
    - ホストが visible を付けた欄だけを出す（影の開催では付けない）。開発者には影の欄も出す。
    - 内部の項目（入力の版・規則の sha など）は落とし、欄の表示に要る項目だけを返す。"""
    if payload.get("status") != "ok":
        return payload
    from src.research.pedigree_persona import store as pstore

    doc = pstore.read(race_id)
    if not doc or doc.get("id") != "pedigree_persona":
        return payload
    start = _race_start(payload, doc)
    now = now or datetime.now(_JST)
    if start is None or now >= start:
        return payload
    if not doc.get("visible"):
        if developer is None:
            from src.api.flask_auth import is_logged_in

            developer = is_logged_in()
        if not developer:
            return payload
    mod = {k: doc[k] for k in PERSONA_KEYS if k in doc}
    return payload | {"modules": [*(payload.get("modules") or []), mod]}


@bp.get("/races/<race_id>/analysis")
def race_analysis(race_id: str):
    if not _RACE_ID.match(race_id):
        return jsonify({"error": "race_id が不正です"}), 400
    from src.research.race_analysis import store

    payload = store.read(race_id)
    if payload is None:
        return jsonify({"status": "pending", "message": PENDING_MESSAGE, "race": {"race_id": race_id}, "entries": [], "provisional_entries": False, "modules": []})
    payload.pop("fingerprint", None)
    payload = with_pedigree_persona(race_id, payload)
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

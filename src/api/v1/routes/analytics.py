"""追加 REST ルート（分析・馬券・血統）。"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from src.api.flask_auth import is_logged_in, require_internal, require_login
from src.api.v1.delegates import (
    get_final_odds,
    get_horse_aptitude,
    get_race_note_3d_v2,
    optimize_betting,
    query_pedigree_race_stats,
)
from src.db.session import get_session, init_engine
from src.db.models import SavedAnalysis
from sqlalchemy import select

bp = Blueprint("v1_analytics", __name__)


@bp.get("/races/<race_id>/final-odds")
def final_odds(race_id: str):
    refresh = request.args.get("refresh", "").lower() in ("1", "true", "yes")
    return jsonify(get_final_odds(race_id, refresh=refresh))


@bp.get("/pedigree-race-stats/query")
def pedigree_race_stats_query():
    payload, code = query_pedigree_race_stats(dict(request.args))
    return jsonify(payload), code


@bp.get("/bloodline-cluster/horse-aptitude")
def bloodline_horse_aptitude():
    payload, code = get_horse_aptitude(
        request.args.get("horse_id"),
        request.args.get("horse_name"),
    )
    return jsonify(payload), code


@bp.get("/pedigree/race-note-3d-v2")
def pedigree_race_note_v2():
    race_id = request.args.get("race_id", "")
    if not race_id:
        return jsonify({"error": "race_id required"}), 400
    payload, code = get_race_note_3d_v2(race_id)
    return jsonify(payload), code


@bp.post("/betting/optimize")
@require_login
def betting_optimize():
    body = request.get_json(silent=True) or {}
    payload, code = optimize_betting(body)
    return jsonify(payload), code


@bp.get("/auth/status")
def auth_status():
    return jsonify({"logged_in": is_logged_in()})


@bp.post("/auth/login")
def auth_login():
    body = request.get_json(silent=True) or {}
    from src.api.flask_auth import login_response

    return login_response(body.get("password", ""), body.get("remember", True))


@bp.post("/auth/logout")
def auth_logout():
    from src.api.flask_auth import logout_response

    return logout_response()


@bp.get("/saved-analyses")
@require_login
def list_saved_analyses():
    init_engine()
    with get_session() as session:
        rows = session.scalars(select(SavedAnalysis).limit(100)).all()
    return jsonify({
        "items": [
            {
                "id": str(r.id),
                "name": r.name,
                "analysis_type": r.analysis_type,
                "filter_conditions": r.filter_conditions,
            }
            for r in rows
        ]
    })


@bp.post("/saved-analyses")
@require_login
def create_saved_analysis():
    body = request.get_json(silent=True) or {}
    init_engine()
    with get_session() as session:
        row = SavedAnalysis(
            name=body.get("name", "無題"),
            analysis_type=body.get("analysis_type", "sire"),
            filter_conditions=body.get("filter_conditions") or {},
        )
        session.add(row)
        session.flush()
        aid = str(row.id)
    return jsonify({"id": aid}), 201


@bp.get("/admin/health")
@require_internal
def admin_health():
    return jsonify({"status": "ok", "scope": "internal"})

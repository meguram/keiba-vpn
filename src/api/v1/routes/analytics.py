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
    logged = is_logged_in()
    return jsonify({"logged_in": logged, "is_developer": logged, "is_admin": logged, "is_member": logged})


@bp.post("/admin/git-pull")
@require_login
def admin_git_pull():
    from src.api.git_pull_runner import run_git_pull

    result = run_git_pull()
    code = 200 if result.get("status") in ("ok", "updated", "up_to_date", "skipped") else 500
    return jsonify(result), code


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


# ---------------------------------------------------------------------------
# 2-A: フィルター統計（F-XX、認証必須）
# ---------------------------------------------------------------------------

@bp.post("/filter/stats")
@require_login
def filter_stats():
    """絞り込み条件に合致するレースの統計集計（auth必須）。"""
    body = request.get_json(silent=True) or {}
    venue = body.get("venue")        # 競馬場コード（任意）
    distance = body.get("distance")  # 距離（任意）
    track = body.get("track")        # 馬場種別（任意）
    grade = body.get("grade")        # グレード（任意）
    limit = min(int(body.get("limit", 100)), 500)

    init_engine()
    from sqlalchemy import select, and_
    from src.db.models import Race

    with get_session() as session:
        q = select(Race)
        filters = []
        if venue:
            filters.append(Race.venue_code == venue)
        if distance:
            filters.append(Race.distance == int(distance))
        if track:
            filters.append(Race.track_type == track)
        if grade:
            filters.append(Race.grade == grade)
        if filters:
            q = q.where(and_(*filters))
        q = q.limit(limit)
        rows = session.scalars(q).all()

    return jsonify({
        "count": len(rows),
        "filters": {"venue": venue, "distance": distance, "track": track, "grade": grade},
        "races": [
            {
                "race_id": r.race_id,
                "race_name": r.race_name,
                "date": str(r.race_date) if r.race_date else None,
                "venue_code": r.venue_code,
                "distance": r.distance,
                "grade": r.grade,
            }
            for r in rows
        ],
    })


# ---------------------------------------------------------------------------
# 2-B: お気に入り馬 API（F-12、認証必須）
# ---------------------------------------------------------------------------

@bp.get("/favorites")
@require_login
def list_favorites():
    """お気に入り馬一覧。"""
    init_engine()
    from sqlalchemy import select
    from src.db.models import UserFavorite, User

    # 現在のログインユーザー ID を取得（シングルユーザー想定: users テーブル先頭）
    with get_session() as session:
        user = session.scalars(select(User).limit(1)).first()
        if not user:
            return jsonify({"favorites": []})
        rows = session.scalars(
            select(UserFavorite).where(UserFavorite.user_id == user.id)
        ).all()
    return jsonify({
        "favorites": [
            {"horse_id": r.horse_id, "horse_name": r.horse_name, "created_at": str(r.created_at)}
            for r in rows
        ]
    })


@bp.post("/favorites")
@require_login
def add_favorite():
    """お気に入り馬を登録。"""
    body = request.get_json(silent=True) or {}
    horse_id = body.get("horse_id", "").strip()
    if not horse_id:
        return jsonify({"error": "horse_id required"}), 400

    init_engine()
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError
    from src.db.models import UserFavorite, User

    with get_session() as session:
        user = session.scalars(select(User).limit(1)).first()
        if not user:
            return jsonify({"error": "user not found"}), 404
        fav = UserFavorite(
            user_id=user.id,
            horse_id=horse_id,
            horse_name=body.get("horse_name"),
        )
        try:
            session.add(fav)
            session.flush()
            fav_id = fav.id
        except IntegrityError:
            return jsonify({"error": "already registered"}), 409
    return jsonify({"id": fav_id, "horse_id": horse_id}), 201


@bp.delete("/favorites/<horse_id>")
@require_login
def remove_favorite(horse_id: str):
    """お気に入り馬を削除。"""
    init_engine()
    from sqlalchemy import select, delete
    from src.db.models import UserFavorite, User

    with get_session() as session:
        user = session.scalars(select(User).limit(1)).first()
        if not user:
            return jsonify({"error": "user not found"}), 404
        session.execute(
            delete(UserFavorite).where(
                UserFavorite.user_id == user.id,
                UserFavorite.horse_id == horse_id,
            )
        )
    return jsonify({"status": "deleted", "horse_id": horse_id})


# ---------------------------------------------------------------------------
# 2-C: 通知設定 API（F-09、認証必須）
# ---------------------------------------------------------------------------

@bp.get("/notifications/settings")
@require_login
def get_notification_settings():
    """通知設定を取得。"""
    init_engine()
    from sqlalchemy import select
    from src.db.models import NotificationSetting, User

    with get_session() as session:
        user = session.scalars(select(User).limit(1)).first()
        if not user:
            return jsonify({"email": None, "notify_favorite_race": False})
        setting = session.scalars(
            select(NotificationSetting).where(NotificationSetting.user_id == user.id)
        ).first()
    if not setting:
        return jsonify({"email": None, "notify_favorite_race": False})
    return jsonify({
        "email": setting.email,
        "notify_favorite_race": setting.notify_favorite_race,
    })


@bp.post("/notifications/settings")
@require_login
def update_notification_settings():
    """通知設定を更新。"""
    body = request.get_json(silent=True) or {}
    init_engine()
    from sqlalchemy import select
    from src.db.models import NotificationSetting, User

    with get_session() as session:
        user = session.scalars(select(User).limit(1)).first()
        if not user:
            return jsonify({"error": "user not found"}), 404
        setting = session.scalars(
            select(NotificationSetting).where(NotificationSetting.user_id == user.id)
        ).first()
        if setting is None:
            setting = NotificationSetting(user_id=user.id)
            session.add(setting)
        if "email" in body:
            setting.email = body["email"]
        if "notify_favorite_race" in body:
            setting.notify_favorite_race = bool(body["notify_favorite_race"])
        session.flush()
    return jsonify({"status": "ok"})


# ── 種牡馬メモ 上書き保存 ─────────────────────────────────────────────────

import json as _json
from pathlib import Path as _Path

_STALLION_OVERRIDES_PATH = _Path(__file__).parents[4] / "data" / "local" / "stallion_notes_overrides.json"


def _load_stallion_overrides() -> dict:
    try:
        if _STALLION_OVERRIDES_PATH.exists():
            return _json.loads(_STALLION_OVERRIDES_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_stallion_overrides(data: dict) -> None:
    _STALLION_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STALLION_OVERRIDES_PATH.write_text(
        _json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@bp.get("/stallion-notes/overrides")
def get_stallion_overrides():
    """サーバー側に保存された種牡馬メモ上書きを返す。"""
    return jsonify(_load_stallion_overrides())


@bp.post("/stallion-notes/overrides")
@require_login
def save_stallion_override():
    """特定エントリの content を上書き保存（管理者のみ）。"""
    body = request.get_json(silent=True) or {}
    entry_id = body.get("id", "").strip()
    content = body.get("content")
    if not entry_id:
        return jsonify({"error": "id が必要です"}), 400
    overrides = _load_stallion_overrides()
    if content is None:
        overrides.pop(entry_id, None)
    else:
        overrides[entry_id] = str(content)
    _save_stallion_overrides(overrides)
    return jsonify({"status": "ok", "id": entry_id, "saved": content is not None})

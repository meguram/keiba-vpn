"""Flask REST API — /api/v1（AREA-03 / MASTER §4-3）。"""

from __future__ import annotations

import os

from flask import Flask, jsonify, request

from src.utils.project_env import load_project_dotenv

load_project_dotenv()

from src.api.auth import COOKIE_NAME, _verify_token
from src.api.flask_auth import require_internal
from src.api.v1.services import (
    DEFAULT_MODEL_VERSION,
    build_laps_response,
    get_predictions_cached,
    get_race_detail,
    list_races,
)
from src.api.v1.routes.analytics import bp as analytics_bp
from src.api.v1.routes.data_analysis import bp as data_analysis_bp
from src.db.session import get_session, init_engine


def _is_logged_in() -> bool:
    """Flask request からセッション Cookie を読んでログイン状態を返す。"""
    token = request.cookies.get(COOKIE_NAME, "")
    return _verify_token(token)


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(analytics_bp, url_prefix="/api/v1")
    app.register_blueprint(data_analysis_bp, url_prefix="/api/v1")

    @app.get("/api/v1/health")
    def health():
        return jsonify({"status": "ok", "service": "keiba-vpn-api"})

    @app.get("/api/v1/races")
    def api_races():
        date_str = request.args.get("date")
        try:
            init_engine()
            with get_session() as session:
                return jsonify({"races": list_races(session, date_str)})
        except Exception as exc:
            return jsonify({"races": [], "error": "database unavailable", "detail": str(exc)}), 503

    @app.get("/api/v1/races/today")
    def api_races_today():
        """当日レース一覧（認証不要）。"""
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        try:
            init_engine()
            with get_session() as session:
                return jsonify({"races": list_races(session, today), "date": today})
        except Exception as exc:
            return jsonify({"races": [], "error": "database unavailable", "detail": str(exc)}), 503

    @app.get("/api/v1/races/<race_id>")
    def api_race_detail(race_id: str):
        init_engine()
        with get_session() as session:
            detail = get_race_detail(session, race_id)
        if detail is None:
            return jsonify({"error": "race not found"}), 404
        return jsonify(detail)

    @app.get("/api/v1/races/<race_id>/entries")
    def api_race_entries(race_id: str):
        init_engine()
        with get_session() as session:
            detail = get_race_detail(session, race_id)
        if detail is None:
            return jsonify({"error": "race not found"}), 404
        return jsonify({"race_id": race_id, "entries": detail.get("entries", [])})

    @app.get("/api/v1/races/<race_id>/predictions")
    def api_predictions(race_id: str):
        init_engine()
        model_version = request.args.get("model_version", DEFAULT_MODEL_VERSION)
        with get_session() as session:
            payload = get_predictions_cached(session, race_id, model_version)
        if payload is None:
            return jsonify({"error": "predictions not found"}), 404

        horses: list = payload.get("horses") or []
        total_horses = len(horses)
        logged_in = _is_logged_in()
        if not logged_in:
            # 非メンバー: 全馬を返すが AI 予測フィールドを null に隠蔽し馬番順にソート
            _REDACTED_FIELDS = (
                "win_prob", "place_prob", "show_prob",
                "predicted_win_odds", "predicted_place_odds",
                "win_roi", "show_roi",
                "predicted_position", "predicted_running_style",
            )
            redacted_horses = []
            for h in horses:
                h2 = dict(h)
                for field in _REDACTED_FIELDS:
                    if field in h2:
                        h2[field] = None
                h2["redacted"] = True
                redacted_horses.append(h2)
            horses = sorted(redacted_horses, key=lambda x: x.get("post_position") or 0)

        return jsonify({
            **payload,
            "horses": horses,
            "is_guest": not logged_in,
            "total_horses": total_horses,
        })

    @app.get("/api/v1/races/<race_id>/predictions/laps")
    def api_prediction_laps(race_id: str):
        init_engine()
        model_version = request.args.get("model_version", DEFAULT_MODEL_VERSION)
        with get_session() as session:
            payload = build_laps_response(session, race_id, model_version)
        if payload is None:
            return jsonify({"error": "lap predictions not found"}), 404
        return jsonify(payload)

    @app.get("/api/v1/races/<race_id>/results")
    def api_race_results(race_id: str):
        init_engine()
        from sqlalchemy import select
        from src.db.models import RaceResult

        with get_session() as session:
            rows = session.scalars(select(RaceResult).where(RaceResult.race_id == race_id)).all()
            result_payload = {
                "race_id": race_id,
                "results": [
                    {
                        "horse_id": r.horse_id,
                        "finish_pos": r.finish_pos,
                        "finish_time_sec": float(r.finish_time_sec) if r.finish_time_sec else None,
                        "last_3f_sec": float(r.last_3f_sec) if r.last_3f_sec else None,
                    }
                    for r in rows
                ],
            } if rows else None
        if not rows:
            storage = __import__("src.scraper.storage", fromlist=["HybridStorage"]).HybridStorage()
            try:
                data = storage.load("race_result", race_id) or storage.load("race_result_on_time", race_id)
            except Exception:
                data = None
            if not data:
                return jsonify({"error": "results not found"}), 404
            return jsonify(data)
        return jsonify(result_payload)

    @app.get("/api/v1/races/<race_id>/megu-index")
    def api_megu_index(race_id: str):
        """レース内全馬のめぐ指数を返す。"""
        init_engine()
        from sqlalchemy import select
        from src.db.models import MeguIndex

        with get_session() as session:
            rows = session.scalars(
                select(MeguIndex)
                .where(MeguIndex.race_id == race_id)
                .order_by(MeguIndex.megu_index.desc())
            ).all()
            if not rows:
                return jsonify({"race_id": race_id, "megu_index": [], "source": "none"}), 404
            return jsonify({
                "race_id": race_id,
                "model_version": rows[0].model_version,
                "source": "db",
                "megu_index": [
                    {
                        "horse_id": r.horse_id,
                        "megu_index": float(r.megu_index),
                        "finish_time_sec": float(r.finish_time_sec) if r.finish_time_sec else None,
                        "par_time_sec": float(r.par_time_sec) if r.par_time_sec else None,
                        "adjusted_time_sec": float(r.adjusted_time_sec) if r.adjusted_time_sec else None,
                        "delta_pace_sec": float(r.delta_pace_sec),
                        "delta_track_sec": float(r.delta_track_sec),
                        "delta_weight_sec": float(r.delta_weight_sec),
                        "delta_level_sec": float(r.delta_level_sec),
                    }
                    for r in rows
                ],
            })

    @app.get("/api/v1/horse/<horse_id>/megu-index-history")
    def api_horse_megu_index_history(horse_id: str):
        """馬の直近めぐ指数履歴（最大20走）。"""
        init_engine()
        from sqlalchemy import select
        from src.db.models import MeguIndex, Race

        with get_session() as session:
            rows = session.execute(
                select(MeguIndex, Race.race_date, Race.venue, Race.surface,
                       Race.distance, Race.track_condition)
                .join(Race, Race.race_id == MeguIndex.race_id)
                .where(MeguIndex.horse_id == horse_id)
                .order_by(Race.race_date.desc())
                .limit(20)
            ).all()
            return jsonify({
                "horse_id": horse_id,
                "history": [
                    {
                        "race_id": r.MeguIndex.race_id,
                        "race_date": str(r.race_date),
                        "venue": r.venue,
                        "surface": r.surface,
                        "distance": r.distance,
                        "track_condition": r.track_condition,
                        "megu_index": float(r.MeguIndex.megu_index),
                        "finish_time_sec": float(r.MeguIndex.finish_time_sec) if r.MeguIndex.finish_time_sec else None,
                        "par_time_sec": float(r.MeguIndex.par_time_sec) if r.MeguIndex.par_time_sec else None,
                    }
                    for r in rows
                ],
            })

    @app.get("/api/v1/races/<race_id>/tracking-difficulty")
    def api_tracking_difficulty(race_id: str):
        try:
            from src.pipeline.inference.tracking_difficulty_service import get_or_compute
            from src.scraper.storage import HybridStorage

            storage = HybridStorage()
            payload = get_or_compute(storage, race_id)
            if payload.get("status") == "not_precomputed":
                return jsonify(payload), 404
            return jsonify(payload)
        except Exception as exc:
            return jsonify({"error": "tracking difficulty unavailable", "detail": str(exc)}), 503

    @app.get("/api/v1/horse/<horse_id>/growth-curve")
    def api_growth_curve(horse_id: str):
        from src.pipeline.inference.growth_curve_service import get_growth_curve
        from src.scraper.storage import HybridStorage

        payload = get_growth_curve(HybridStorage(), horse_id)
        if not payload or payload.get("error"):
            return jsonify(payload or {"error": "growth curve not found"}), 404
        return jsonify(payload)

    @app.post("/api/v1/admin/users/<user_id>/member")
    @require_internal
    def admin_set_member(user_id: str):
        """管理者用: is_member を手動で ON/OFF する（MVP 用）。"""
        body = request.get_json(silent=True) or {}
        if "is_member" not in body:
            return jsonify({"error": "is_member field required"}), 400
        is_member_val = bool(body["is_member"])
        try:
            from uuid import UUID as _UUID
            from sqlalchemy import select as _select
            from src.db.models import User as _User
            user_uuid = _UUID(user_id)
            init_engine()
            with get_session() as session:
                user = session.scalars(_select(_User).where(_User.id == user_uuid)).first()
                if user is None:
                    return jsonify({"error": "user not found"}), 404
                user.is_member = is_member_val
                session.commit()
            return jsonify({"user_id": user_id, "is_member": is_member_val})
        except ValueError:
            return jsonify({"error": "invalid user_id format"}), 400
        except Exception as exc:
            return jsonify({"error": "database error", "detail": str(exc)}), 503

    @app.get("/api/v1/track-speed/day")
    def api_track_speed_day():
        date_str = request.args.get("date")
        venue = request.args.get("venue")
        if not date_str or not venue:
            return jsonify({"error": "date and venue are required"}), 400
        from src.config.data_paths import TRACK_SPEED_RACES_DIR
        import json as _json

        path = TRACK_SPEED_RACES_DIR / f"{date_str.replace('-', '')}_{venue}.json"
        if not path.is_file():
            return jsonify({"error": "track speed data not found", "path": str(path)}), 404
        return jsonify(_json.loads(path.read_text(encoding="utf-8")))

    return app


def main():
    port = int(os.environ.get("FLASK_PORT", os.environ.get("PORT", "5000")))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    main()

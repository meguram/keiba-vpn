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
from pathlib import Path

from src.db.session import get_session, init_engine

# race_result_flat.parquet ディレクトリ（sex_age 補完用）
_FLAT_DIR = Path(__file__).resolve().parents[2] / "data/page_reference/tables"


def _is_logged_in() -> bool:
    """Flask request からセッション Cookie を読んでログイン状態を返す。"""
    token = request.cookies.get(COOKIE_NAME, "")
    return _verify_token(token)


def create_app() -> Flask:
    app = Flask(__name__)
    # UTF-8 日本語を \uXXXX にエスケープしない（文字化け防止）
    app.config["JSON_ENSURE_ASCII"] = False
    app.json.ensure_ascii = False

    @app.after_request
    def set_charset(response):
        """JSONレスポンスに charset=utf-8 を明示する。"""
        ct = response.content_type or ""
        if "application/json" in ct and "charset" not in ct:
            response.content_type = "application/json; charset=utf-8"
        return response

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
        """馬の直近めぐ指数履歴（最大20走）。想定・実測の両方を返す。"""
        init_engine()
        from sqlalchemy import text as sa_text
        from src.api.megu_predict_race import (
            load_beta_weight,
            load_transfer_map,
            megu_predict_enabled,
            predict_megu_final_for_horse_race,
        )
        from src.db.models import Race

        _MODEL_VERSION = "v2"

        with get_session() as session:
            full_rows = session.execute(
                sa_text("""
                    SELECT
                        mi.race_id, mi.megu_index, mi.par_time_sec, mi.adjusted_time_sec,
                        r.race_date, r.venue, r.surface, r.distance, r.track_condition,
                        rr.finish_time_sec
                    FROM megu_index mi
                    JOIN races r ON r.race_id = mi.race_id
                    LEFT JOIN race_results rr
                        ON rr.race_id = mi.race_id AND rr.horse_id = mi.horse_id
                    WHERE mi.horse_id = :horse_id
                      AND mi.model_version = :mv
                      AND mi.computation_status = 'valid'
                      AND mi.megu_index IS NOT NULL
                    ORDER BY r.race_date DESC, mi.race_id DESC
                """),
                {"horse_id": horse_id, "mv": _MODEL_VERSION},
            ).fetchall()

            if not full_rows:
                return jsonify({"horse_id": horse_id, "history": []})

            beta_weight = load_beta_weight(session, _MODEL_VERSION) if megu_predict_enabled() else 0.0
            transfer_map = load_transfer_map(session, _MODEL_VERSION) if megu_predict_enabled() else {}
            display_rows = full_rows[:20]
            history = []

            for i, row in enumerate(display_rows):
                race = session.get(Race, row.race_id)
                prior_rows = full_rows[i + 1:]
                actual = float(row.megu_index)

                entry = session.execute(
                    sa_text("""
                        SELECT e.jockey_weight,
                               COALESCE(NULLIF(e.sex_age, ''), h.sex) AS sex_age
                        FROM entries e
                        LEFT JOIN horses h ON h.horse_id = e.horse_id
                        WHERE e.race_id = :race_id AND e.horse_id = :horse_id
                        LIMIT 1
                    """),
                    {"race_id": row.race_id, "horse_id": horse_id},
                ).fetchone()

                jockey_weight = float(entry.jockey_weight) if entry and entry.jockey_weight is not None else None
                sex_age = entry.sex_age if entry and entry.sex_age else None
                megu_final = None
                if megu_predict_enabled() and race:
                    megu_final = predict_megu_final_for_horse_race(
                        session,
                        horse_id=horse_id,
                        race_id=row.race_id,
                        prior_hist_rows=prior_rows,
                        transfer_map=transfer_map,
                        beta_weight=beta_weight,
                        jockey_weight=jockey_weight,
                        sex_age=sex_age,
                        race=race,
                        model_version=_MODEL_VERSION,
                    )

                history.append({
                    "race_id": row.race_id,
                    "race_date": str(row.race_date),
                    "venue": row.venue,
                    "surface": row.surface,
                    "distance": row.distance,
                    "track_condition": row.track_condition,
                    "megu_index": actual,
                    "actual_megu": actual,
                    "megu_final": megu_final,
                    "finish_time_sec": float(row.finish_time_sec) if row.finish_time_sec else None,
                    "par_time_sec": float(row.par_time_sec) if row.par_time_sec else None,
                })

            return jsonify({"horse_id": horse_id, "history": history})

    @app.get("/api/v1/races/<race_id>/megu-index-predicted")
    def api_megu_index_predicted(race_id: str):
        """
        出走馬ごとの予測めぐ指数。条件代わりがある場合は転換係数を適用する。

        レスポンス構造:
          race_info: レース条件（surface, distance, dist_band, race_date）
          horses[]: 出走馬ごとの予測データ
            actual_megu   : このレースの実測値（計算済みの場合のみ）
            base_megu     : 直近3走の megu 平均（予測ベース）
            megu_adjusted : 条件代わり補正後の予測値（変更なし時は base_megu と同値）
            weight_megu_delta: 今回斤量に対する指数補正（点）
            megu_final    : 条件＋斤量補正後の最終予測得点
            condition_change: 条件代わりの種類・転換係数情報
            history       : 直近5走の megu 履歴
        """
        init_engine()
        from src.api.megu_predict_race import build_race_megu_predictions

        with get_session() as session:
            payload = build_race_megu_predictions(session, race_id)
            if not payload:
                return jsonify({"error": "race not found or no entries"}), 404
            return jsonify(payload)

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
    # リローダ子プロセスは DATABASE_URL を引き継がないため use_reloader=False
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1", use_reloader=False)


if __name__ == "__main__":
    main()

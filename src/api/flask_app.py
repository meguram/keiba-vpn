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
            condition_change: 条件代わりの種類・転換係数情報
            history       : 直近5走の megu 履歴
        """
        init_engine()
        from sqlalchemy import text as sa_text
        from src.db.models import MeguConditionTransfer, Race  # noqa: F401

        _MODEL_VERSION = "v1"
        _BASE_RACES = 3
        _HISTORY_LIMIT = 5
        _MAJOR_DIST = 600

        def _dist_band(d: int) -> str:
            if d <= 1400:
                return "sprint"
            if d <= 1800:
                return "mile"
            if d <= 2200:
                return "middle"
            return "long"

        def _change_label(sf: str, st: str, dist_change: int) -> str:
            if sf != st:
                return f"{sf}→{st}"
            if dist_change <= -_MAJOR_DIST:
                return "距離大幅短縮"
            return "距離大幅延長"

        with get_session() as session:
            # 1. レース情報
            race = session.get(Race, race_id)
            if not race:
                return jsonify({"error": "race not found"}), 404

            race_dist_band = _dist_band(race.distance)

            # 2. 出走馬一覧（horse_name・斤量を horses/entries テーブルから JOIN）
            entry_rows = session.execute(
                sa_text("""
                    SELECT e.horse_id, e.post_no, h.horse_name, e.jockey_weight, e.sex_age
                    FROM entries e
                    LEFT JOIN horses h ON h.horse_id = e.horse_id
                    WHERE e.race_id = :race_id
                    ORDER BY e.post_no
                """),
                {"race_id": race_id},
            ).fetchall()

            if not entry_rows:
                return jsonify({"error": "no entries found"}), 404

            horse_ids = [r.horse_id for r in entry_rows]
            horse_meta = {
                r.horse_id: {
                    "number": r.post_no,
                    "name": r.horse_name,
                    "jockey_weight": float(r.jockey_weight) if r.jockey_weight is not None else None,
                    "sex_age": r.sex_age,
                }
                for r in entry_rows
            }

            # 3. 各馬の megu 履歴を一括取得（このレース自身を除く直近）
            hist_rows = session.execute(
                sa_text("""
                    SELECT
                        mi.horse_id, mi.race_id, mi.megu_index,
                        r.race_date, r.venue, r.surface, r.distance,
                        rr.finish_pos
                    FROM megu_index mi
                    JOIN races r ON r.race_id = mi.race_id
                    LEFT JOIN race_results rr
                        ON rr.race_id = mi.race_id AND rr.horse_id = mi.horse_id
                    WHERE mi.horse_id = ANY(:horse_ids)
                      AND mi.model_version = :mv
                      AND mi.race_id != :race_id
                    ORDER BY mi.horse_id, r.race_date DESC, mi.race_id DESC
                """),
                {"horse_ids": horse_ids, "mv": _MODEL_VERSION, "race_id": race_id},
            ).fetchall()

            # 4. このレースの実測 megu と走破タイム（計算済みなら存在する）
            actual_rows = session.execute(
                sa_text("""
                    SELECT mi.horse_id, mi.megu_index,
                           rr.finish_time_sec, rr.finish_pos
                    FROM megu_index mi
                    LEFT JOIN race_results rr
                        ON rr.race_id = mi.race_id AND rr.horse_id = mi.horse_id
                    WHERE mi.race_id = :race_id AND mi.model_version = :mv
                """),
                {"race_id": race_id, "mv": _MODEL_VERSION},
            ).fetchall()
            actual_map = {
                r.horse_id: {
                    "megu_index": float(r.megu_index) if r.megu_index is not None else None,
                    "finish_time_sec": float(r.finish_time_sec) if r.finish_time_sec is not None else None,
                    "finish_pos": int(r.finish_pos) if r.finish_pos is not None else None,
                }
                for r in actual_rows
            }

            # 4b. レース結果のみ（megu未計算でも走破タイムは取れる）
            result_rows = session.execute(
                sa_text("""
                    SELECT horse_id, finish_time_sec, finish_pos
                    FROM race_results
                    WHERE race_id = :race_id
                """),
                {"race_id": race_id},
            ).fetchall()
            result_map = {
                r.horse_id: {
                    "finish_time_sec": float(r.finish_time_sec) if r.finish_time_sec is not None else None,
                    "finish_pos": int(r.finish_pos) if r.finish_pos is not None else None,
                }
                for r in result_rows
            }

            # 5. 馬ごとに履歴を整理（horse_id → list[row]）
            from collections import defaultdict
            hist_by_horse: dict = defaultdict(list)
            for r in hist_rows:
                hist_by_horse[r.horse_id].append(r)

            # 6. 必要な転換ペアを特定して一括 DB 取得
            transfer_keys: set = set()
            prev_cond: dict = {}  # horse_id -> (surface, distance, dist_band)
            for hid in horse_ids:
                hist = hist_by_horse[hid]
                if hist:
                    p = hist[0]
                    db_from = _dist_band(int(p.distance))
                    prev_cond[hid] = (p.surface, int(p.distance), db_from)
                    if p.surface != race.surface or abs(int(p.distance) - race.distance) >= _MAJOR_DIST:
                        transfer_keys.add((p.surface, db_from, race.surface, race_dist_band))

            transfer_map: dict = {}
            for (sf, dbf, st, dbt) in transfer_keys:
                tr = session.execute(
                    sa_text("""
                        SELECT delta_mean, delta_std, sample_count
                        FROM megu_condition_transfer
                        WHERE surface_from = :sf AND dist_band_from = :dbf
                          AND surface_to = :st AND dist_band_to = :dbt
                          AND model_version = :mv
                    """),
                    {"sf": sf, "dbf": dbf, "st": st, "dbt": dbt, "mv": _MODEL_VERSION},
                ).fetchone()
                if tr:
                    transfer_map[(sf, dbf, st, dbt)] = {
                        "delta_mean": float(tr.delta_mean),
                        "delta_std": float(tr.delta_std) if tr.delta_std is not None else None,
                        "sample_count": int(tr.sample_count),
                    }

            # 7. 各馬の予測値を組み立て
            result_horses = []
            for hid in horse_ids:
                meta = horse_meta[hid]
                hist = hist_by_horse[hid]
                actual = actual_map.get(hid, {})
                result = result_map.get(hid, {})
                actual_megu = actual.get("megu_index")
                # 走破タイムは megu テーブル優先、なければ race_results 直接
                finish_time_sec = actual.get("finish_time_sec") or result.get("finish_time_sec")

                recent = hist[:_BASE_RACES]
                recent_valid = [r for r in recent if r.megu_index is not None]
                base_megu = round(
                    sum(float(r.megu_index) for r in recent_valid) / len(recent_valid), 1
                ) if recent_valid else None

                cond_change: dict = {"type": "none", "label": None}
                megu_adjusted = base_megu

                if hid in prev_cond:
                    sf, pd_dist, db_from = prev_cond[hid]
                    dist_change = race.distance - pd_dist
                    surface_changed = sf != race.surface
                    dist_major = abs(dist_change) >= _MAJOR_DIST

                    if surface_changed or dist_major:
                        change_type = (
                            "both" if surface_changed and dist_major
                            else ("surface" if surface_changed else "distance")
                        )
                        cond_change = {
                            "type": change_type,
                            "label": _change_label(sf, race.surface, dist_change),
                            "surface_from": sf,
                            "surface_to": race.surface,
                            "dist_band_from": db_from,
                            "dist_band_to": race_dist_band,
                            "dist_change": dist_change,
                            "delta_mean": None,
                            "delta_std": None,
                            "transfer_sample_count": 0,
                        }
                        tr_key = (sf, db_from, race.surface, race_dist_band)
                        if tr_key in transfer_map and base_megu is not None:
                            tr = transfer_map[tr_key]
                            megu_adjusted = round(base_megu + tr["delta_mean"], 1)
                            cond_change["delta_mean"] = tr["delta_mean"]
                            cond_change["delta_std"] = tr["delta_std"]
                            cond_change["transfer_sample_count"] = tr["sample_count"]

                result_horses.append({
                    "horse_id": hid,
                    "horse_name": meta["name"],
                    "horse_number": meta["number"],
                    "sex_age": meta.get("sex_age"),
                    "jockey_weight": meta["jockey_weight"],
                    "finish_time_sec": finish_time_sec,
                    "actual_megu": actual_megu,
                    "base_megu": base_megu,
                    "megu_adjusted": megu_adjusted,
                    "condition_change": cond_change,
                    "history": [
                        {
                            "race_id": r.race_id,
                            "race_date": str(r.race_date),
                            "venue": r.venue,
                            "surface": r.surface,
                            "distance": r.distance,
                            "megu_index": float(r.megu_index) if r.megu_index is not None else None,
                            "finish_pos": r.finish_pos,
                        }
                        for r in hist[:_HISTORY_LIMIT]
                    ],
                })

            result_horses.sort(
                key=lambda h: h["megu_adjusted"] if h["megu_adjusted"] is not None else -999,
                reverse=True,
            )

            # ── レースレベル（2着馬の実測めぐ指数で分類）──
            def _race_level(megu: float | None) -> str:
                if megu is None:
                    return "?"
                if megu >= 120:
                    return "S"
                if megu >= 108:
                    return "A"
                if megu >= 95:
                    return "B"
                if megu >= 85:
                    return "C"
                return "D"

            second_megu: float | None = None
            for a in actual_map.values():
                if a.get("finish_pos") == 2 and a.get("megu_index") is not None:
                    second_megu = a["megu_index"]
                    break

            race_level_info = {
                "class": _race_level(second_megu),
                "megu_2nd": second_megu,
            }

            return jsonify({
                "race_id": race_id,
                "race_info": {
                    "race_name": race.race_name,
                    "venue": race.venue,
                    "surface": race.surface,
                    "distance": race.distance,
                    "dist_band": race_dist_band,
                    "track_condition": race.track_condition,
                    "grade": race.grade,
                    "race_date": str(race.race_date) if race.race_date else None,
                },
                "race_level": race_level_info,
                "model_version": _MODEL_VERSION,
                "horses": result_horses,
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

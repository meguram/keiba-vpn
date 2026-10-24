"""詳細データ分析 API — /api/v1/data-analysis/query"""
from __future__ import annotations

from datetime import date as dt

from flask import Blueprint, jsonify, request
from sqlalchemy import Float, Integer, and_, cast, extract, func, select

from src.db.models import Entry, Horse, PredictionResult, Race, RaceResult, RacePaceSummary
from src.db.session import get_session, init_engine

bp = Blueprint("data_analysis", __name__)

# ---------------------------------------------------------------------------
# Whitelisted fields (prevents SQL injection via field names)
# ---------------------------------------------------------------------------

NUMERIC_FIELDS: dict[str, dict] = {
    "finish_pos":        {"label": "着順",            "lower_is_better": True},
    "finish_time_sec":   {"label": "タイム（秒）",    "lower_is_better": True},
    "last_3f_sec":       {"label": "上がり3F（秒）",  "lower_is_better": True},
    "weight":            {"label": "馬体重（kg）",    "lower_is_better": False},
    "jockey_weight":     {"label": "斤量",            "lower_is_better": False},
    "win_prob":          {"label": "AI勝率予測",      "lower_is_better": False},
    "place_prob":        {"label": "AI複勝率予測",    "lower_is_better": False},
    "expected_win_roi":  {"label": "期待ROI（単勝）", "lower_is_better": False},
    "expected_show_roi": {"label": "期待ROI（複勝）", "lower_is_better": False},
}

CATEGORICAL_FIELDS: dict[str, dict] = {
    "surface":         {"label": "馬場（芝/ダート）"},
    "track_condition": {"label": "馬場状態"},
    "grade":           {"label": "グレード"},
    "venue":           {"label": "競馬場"},
    "race_class":      {"label": "クラス"},
    "pace_category":   {"label": "ペース（H/M/S）"},
    "distance_bucket": {"label": "距離帯（200m単位）"},
    "month":           {"label": "月"},
    "year":            {"label": "年"},
}

# ---------------------------------------------------------------------------
# Column expression resolvers
# ---------------------------------------------------------------------------

def _numeric_col(field: str):
    if field == "finish_pos":
        return cast(RaceResult.finish_pos, Float)
    if field == "finish_time_sec":
        return cast(RaceResult.finish_time_sec, Float)
    if field == "last_3f_sec":
        return cast(RaceResult.last_3f_sec, Float)
    if field == "weight":
        return cast(Entry.weight, Float)
    if field == "jockey_weight":
        return cast(Entry.jockey_weight, Float)
    if field == "win_prob":
        return cast(PredictionResult.win_prob, Float)
    if field == "place_prob":
        return cast(PredictionResult.place_prob, Float)
    if field == "expected_win_roi":
        return cast(PredictionResult.expected_win_roi, Float)
    if field == "expected_show_roi":
        return cast(PredictionResult.expected_show_roi, Float)
    raise ValueError(f"Unknown numeric field: {field!r}")


def _cat_col(field: str):
    if field == "surface":
        return Race.surface
    if field == "track_condition":
        return Race.track_condition
    if field == "grade":
        return Race.grade
    if field == "venue":
        return Race.venue
    if field == "race_class":
        return Race.race_class
    if field == "pace_category":
        return RacePaceSummary.pace_category
    if field == "distance_bucket":
        # PostgreSQL integer arithmetic: floor to 200m bands
        return (Race.distance / 200 * 200).label("distance_bucket")
    if field == "month":
        return cast(extract("month", Race.race_date), Integer).label("month")
    if field == "year":
        return cast(extract("year", Race.race_date), Integer).label("year")
    raise ValueError(f"Unknown categorical field: {field!r}")


def _extra_tables(field: str) -> str | None:
    if field in ("weight", "jockey_weight"):
        return "entry"
    if field in ("win_prob", "place_prob", "expected_win_roi", "expected_show_roi"):
        return "prediction"
    if field == "pace_category":
        return "pace"
    return None


def _apply_joins(q, needed: set[str]):
    if "entry" in needed:
        q = q.outerjoin(
            Entry,
            and_(Entry.race_id == RaceResult.race_id, Entry.horse_id == RaceResult.horse_id),
        )
    if "prediction" in needed:
        q = q.outerjoin(
            PredictionResult,
            and_(
                PredictionResult.race_id == RaceResult.race_id,
                PredictionResult.horse_id == RaceResult.horse_id,
            ),
        )
    if "pace" in needed:
        q = q.outerjoin(RacePaceSummary, RacePaceSummary.race_id == RaceResult.race_id)
    if "horse" in needed:
        q = q.outerjoin(Horse, Horse.horse_id == RaceResult.horse_id)
    return q


def _apply_filters(q, args: dict):
    filters = []
    if args.get("date_from"):
        filters.append(Race.race_date >= dt.fromisoformat(args["date_from"]))
    if args.get("date_to"):
        filters.append(Race.race_date <= dt.fromisoformat(args["date_to"]))
    if args.get("surface"):
        vals = [v.strip() for v in args["surface"].split(",") if v.strip()]
        if vals:
            filters.append(Race.surface.in_(vals))
    if args.get("venue"):
        vals = [v.strip() for v in args["venue"].split(",") if v.strip()]
        if vals:
            filters.append(Race.venue.in_(vals))
    if args.get("grade"):
        vals = [v.strip() for v in args["grade"].split(",") if v.strip()]
        if vals:
            filters.append(Race.grade.in_(vals))
    if args.get("distance_min"):
        filters.append(Race.distance >= int(args["distance_min"]))
    if args.get("distance_max"):
        filters.append(Race.distance <= int(args["distance_max"]))
    if filters:
        q = q.where(and_(*filters))
    return q


def _agg_expr(y_col, agg: str):
    agg = agg.lower()
    if agg == "sum":
        return func.sum(y_col)
    if agg == "min":
        return func.min(y_col)
    if agg == "max":
        return func.max(y_col)
    if agg == "count":
        return func.count(y_col)
    return func.avg(y_col)


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _s(v) -> str:
    return str(v) if v is not None else "NULL"


# ---------------------------------------------------------------------------
# Schema endpoint (for frontend field selectors)
# ---------------------------------------------------------------------------

@bp.route("/data-analysis/schema", methods=["GET"])
def data_analysis_schema():
    return jsonify({
        "numeric_fields": [{"key": k, **v} for k, v in NUMERIC_FIELDS.items()],
        "categorical_fields": [{"key": k, **v} for k, v in CATEGORICAL_FIELDS.items()],
    })


# ---------------------------------------------------------------------------
# Main query endpoint
# ---------------------------------------------------------------------------

@bp.route("/data-analysis/query", methods=["GET"])
def data_analysis_query():
    args = request.args
    analysis_type = args.get("analysis_type", "distribution")
    x_field = args.get("x_field", "surface")
    y_field = args.get("y_field", "finish_pos")
    y_agg = args.get("y_agg", "avg")
    group_by = args.get("group_by") or None
    limit = min(max(1, int(args.get("limit", 2000))), 5000)

    all_fields = {**NUMERIC_FIELDS, **CATEGORICAL_FIELDS}

    if y_field not in NUMERIC_FIELDS:
        return jsonify({"error": f"Invalid y_field: {y_field!r}. Valid: {list(NUMERIC_FIELDS)}"}), 400
    if x_field not in all_fields:
        return jsonify({"error": f"Invalid x_field: {x_field!r}. Valid: {list(all_fields)}"}), 400
    if group_by and group_by not in CATEGORICAL_FIELDS:
        return jsonify({"error": f"Invalid group_by: {group_by!r}. Valid: {list(CATEGORICAL_FIELDS)}"}), 400

    needed: set[str] = set()
    for f in [y_field, x_field] + ([group_by] if group_by else []):
        t = _extra_tables(f)
        if t:
            needed.add(t)

    try:
        init_engine()
        with get_session() as session:
            if analysis_type == "distribution":
                rows = _distribution(session, x_field, y_field, y_agg, group_by, needed, args, limit)
            elif analysis_type == "scatter":
                rows = _scatter(session, x_field, y_field, group_by, needed, args, limit)
            elif analysis_type == "ranking":
                rows = _ranking(session, y_field, y_agg, needed, args, limit)
            elif analysis_type == "time_series":
                rows = _time_series(session, y_field, y_agg, group_by, needed, args)
            else:
                return jsonify({"error": f"Unknown analysis_type: {analysis_type!r}"}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "query failed", "detail": str(exc)}), 503

    y_meta = NUMERIC_FIELDS[y_field]
    agg_labels = {"avg": "平均", "count": "件数", "sum": "合計", "min": "最小", "max": "最大"}
    y_label = f"{y_meta['label']}（{agg_labels.get(y_agg, y_agg)}）"

    return jsonify({
        "analysis_type": analysis_type,
        "x_field": x_field,
        "y_field": y_field,
        "y_agg": y_agg,
        "group_by": group_by,
        "rows": rows,
        "total_rows": len(rows),
        "meta": {
            "x_label": all_fields.get(x_field, {}).get("label", x_field),
            "y_label": y_label,
            "group_label": CATEGORICAL_FIELDS.get(group_by, {}).get("label") if group_by else None,
        },
    })


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def _distribution(session, x_field, y_field, y_agg, group_by, needed, args, limit):
    x_col = _cat_col(x_field) if x_field in CATEGORICAL_FIELDS else _numeric_col(x_field)
    y_col = _numeric_col(y_field)
    agg = _agg_expr(y_col, y_agg).label("y_val")
    cnt = func.count().label("n")

    grp_cols = [x_col]
    sel_cols = [x_col.label("x_val"), agg, cnt]
    if group_by:
        g = _cat_col(group_by)
        sel_cols.append(g.label("g_val"))
        grp_cols.append(g)

    q = select(*sel_cols)
    q = q.select_from(RaceResult).join(Race, Race.race_id == RaceResult.race_id)
    q = _apply_joins(q, needed)
    q = _apply_filters(q, args)
    q = q.where(y_col.isnot(None))
    q = q.group_by(*grp_cols).order_by(*grp_cols).limit(limit)

    result = session.execute(q).all()
    if group_by:
        return [{"x": _s(r.x_val), "y": _f(r.y_val), "count": r.n, "group": _s(r.g_val)} for r in result]
    return [{"x": _s(r.x_val), "y": _f(r.y_val), "count": r.n} for r in result]


def _scatter(session, x_field, y_field, group_by, needed, args, limit):
    x_col = _numeric_col(x_field) if x_field in NUMERIC_FIELDS else _cat_col(x_field)
    y_col = _numeric_col(y_field)
    sel_cols = [x_col.label("x_val"), y_col.label("y_val")]
    if group_by:
        g = _cat_col(group_by)
        sel_cols.append(g.label("g_val"))

    q = select(*sel_cols)
    q = q.select_from(RaceResult).join(Race, Race.race_id == RaceResult.race_id)
    q = _apply_joins(q, needed)
    q = _apply_filters(q, args)
    q = q.where(x_col.isnot(None)).where(y_col.isnot(None)).limit(limit)

    result = session.execute(q).all()
    if group_by:
        return [{"x": _f(r.x_val), "y": _f(r.y_val), "group": _s(r.g_val)} for r in result]
    return [{"x": _f(r.x_val), "y": _f(r.y_val)} for r in result]


def _ranking(session, y_field, y_agg, needed, args, limit):
    y_col = _numeric_col(y_field)
    agg = _agg_expr(y_col, y_agg).label("y_val")
    cnt = func.count().label("n")

    needed = needed | {"horse"}
    q = select(Horse.horse_name.label("label"), agg, cnt)
    q = q.select_from(RaceResult).join(Race, Race.race_id == RaceResult.race_id)
    q = _apply_joins(q, needed)
    q = _apply_filters(q, args)
    q = q.where(y_col.isnot(None)).where(Horse.horse_name.isnot(None))
    q = q.group_by(RaceResult.horse_id, Horse.horse_name)
    q = q.having(func.count() >= 3)

    lower_better = NUMERIC_FIELDS[y_field]["lower_is_better"]
    q = q.order_by(agg if lower_better else agg.desc())
    q = q.limit(min(limit, 100))

    result = session.execute(q).all()
    return [{"label": r.label, "y": _f(r.y_val), "count": r.n} for r in result]


def _time_series(session, y_field, y_agg, group_by, needed, args):
    y_col = _numeric_col(y_field)
    agg = _agg_expr(y_col, y_agg).label("y_val")
    cnt = func.count().label("n")
    period = func.date_trunc("month", Race.race_date).label("period")

    grp_cols = [period]
    sel_cols = [period, agg, cnt]
    if group_by:
        g = _cat_col(group_by)
        sel_cols.append(g.label("g_val"))
        grp_cols.append(g)

    q = select(*sel_cols)
    q = q.select_from(RaceResult).join(Race, Race.race_id == RaceResult.race_id)
    q = _apply_joins(q, needed)
    q = _apply_filters(q, args)
    q = q.where(y_col.isnot(None))
    q = q.group_by(*grp_cols).order_by(period).limit(200)

    result = session.execute(q).all()
    if group_by:
        return [
            {"x": r.period.strftime("%Y-%m") if r.period else None, "y": _f(r.y_val), "count": r.n, "group": _s(r.g_val)}
            for r in result
        ]
    return [
        {"x": r.period.strftime("%Y-%m") if r.period else None, "y": _f(r.y_val), "count": r.n}
        for r in result
    ]

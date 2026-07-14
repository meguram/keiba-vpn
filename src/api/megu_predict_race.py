"""レース向け想定めぐ指数の共通ロジック（Flask API 共用）。"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from src.pipeline.megu_index.common import dist_band
from src.pipeline.megu_index.predict import predict_megu_scores, megu_margin_sec
from src.pipeline.megu_index.predict_params import load_predict_params
from src.utils.race_card_merge import (
    is_plausible_sex_age,
    load_merged_race_card,
    patch_race_object,
)

_MODEL_VERSION = "v2"
_HISTORY_LIMIT = 5
_EMPTY_PRED: dict[str, Any] = {
    "base_megu": None,
    "megu_adjusted": None,
    "megu_final": None,
    "weight_megu_delta": None,
    "condition_change": {"type": "none", "label": None},
}


def megu_predict_enabled() -> bool:
    """想定めぐ指数の API 計算を有効化するか（MEGU_PREDICT_ENABLED=0 で無効）。"""
    return os.environ.get("MEGU_PREDICT_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
_TRACK_CAT_MAP = {"良": "良", "稍重": "稍重", "重": "重・不良", "不良": "重・不良"}
_FLAT_DIR = Path(__file__).resolve().parents[2] / "data/page_reference/tables"


def load_beta_weight(session: Session, model_version: str = _MODEL_VERSION) -> float:
    row = session.execute(
        sa_text("""
            SELECT param_value FROM megu_regression_params
            WHERE param_name = 'beta_weight' AND model_version = :mv
            LIMIT 1
        """),
        {"mv": model_version},
    ).fetchone()
    return float(row.param_value) if row and row.param_value is not None else 0.0


def load_transfer_map(session: Session, model_version: str = _MODEL_VERSION) -> dict:
    transfer_map: dict = {}
    rows = session.execute(
        sa_text("""
            SELECT surface_from, dist_band_from, surface_to, dist_band_to,
                   delta_mean, delta_std, sample_count
            FROM megu_condition_transfer
            WHERE model_version = :mv
        """),
        {"mv": model_version},
    ).fetchall()
    for tr in rows:
        transfer_map[(tr.surface_from, tr.dist_band_from, tr.surface_to, tr.dist_band_to)] = {
            "delta_sec": float(tr.delta_mean),
            "delta_std": float(tr.delta_std) if tr.delta_std is not None else None,
            "sample_count": int(tr.sample_count),
        }
    return transfer_map


from src.pipeline.megu_index.class_bucket import par_class_bucket


def lookup_par_time(session: Session, race, model_version: str = _MODEL_VERSION) -> float | None:
    direction = race.direction or race.course or ""
    track_cat = _TRACK_CAT_MAP.get(race.track_condition or "良", "良")
    cb = par_class_bucket(
        getattr(race, "grade", None),
        getattr(race, "race_name", None),
        getattr(race, "race_class", None),
    )
    row = session.execute(
        sa_text("""
            SELECT par_time_sec FROM megu_par_time
            WHERE model_version = :mv
              AND distance = :dist AND surface = :surf
              AND course = :course AND track_condition = :tc
              AND class_bucket = :cb
            LIMIT 1
        """),
        {
            "mv": model_version,
            "dist": race.distance,
            "surf": race.surface,
            "course": direction,
            "tc": track_cat,
            "cb": cb,
        },
    ).fetchone()
    if row and row.par_time_sec is not None:
        return float(row.par_time_sec)
    row = session.execute(
        sa_text("""
            SELECT par_time_sec FROM megu_par_time
            WHERE model_version = :mv
              AND distance = :dist AND surface = :surf
              AND course = :course AND track_condition = :tc
              AND class_bucket = ''
            LIMIT 1
        """),
        {
            "mv": model_version,
            "dist": race.distance,
            "surf": race.surface,
            "course": direction,
            "tc": track_cat,
        },
    ).fetchone()
    if row and row.par_time_sec is not None:
        return float(row.par_time_sec)
    fb = session.execute(
        sa_text("""
            SELECT AVG(par_time_sec) AS pt FROM megu_par_time
            WHERE model_version = :mv
              AND distance = :dist AND surface = :surf
              AND track_condition = :tc
              AND class_bucket = ''
        """),
        {"mv": model_version, "dist": race.distance, "surf": race.surface, "tc": track_cat},
    ).fetchone()
    return float(fb.pt) if fb and fb.pt is not None else None


def _hist_row_to_dict(row) -> dict[str, Any]:
    return {
        "race_id": row.race_id,
        "megu_index": float(row.megu_index),
        "par_time_sec": float(row.par_time_sec),
        "adjusted_time_sec": float(row.adjusted_time_sec) if row.adjusted_time_sec else None,
        "race_date": str(row.race_date),
        "surface": row.surface,
        "distance": row.distance,
    }


def predict_megu_final_for_horse_race(
    session: Session,
    *,
    horse_id: str,
    race_id: str,
    prior_hist_rows: list,
    transfer_map: dict,
    beta_weight: float,
    jockey_weight: float | None,
    sex_age: str | None,
    race,
    model_version: str = _MODEL_VERSION,
) -> float | None:
    """当該レース出走前時点の想定めぐ指数（megu_final）。"""
    par_time_target = lookup_par_time(session, race, model_version)
    if par_time_target is None:
        return None
    hist_dicts = [_hist_row_to_dict(r) for r in prior_hist_rows[:_HISTORY_LIMIT]]
    pred = predict_megu_scores(
        hist_dicts,
        par_time_target=par_time_target,
        surface_target=str(race.surface),
        distance_target=int(race.distance),
        jockey_weight=jockey_weight,
        sex_age=sex_age,
        beta_weight=beta_weight,
        transfer_map=transfer_map,
        max_races=_HISTORY_LIMIT,
    )
    return pred.get("megu_final")


def _enrich_horse_meta_from_gcs(race_id: str, horse_meta: dict) -> None:
    try:
        card = load_merged_race_card(race_id)
        if not card:
            return
        for entry in card.get("entries") or []:
            hid = str(entry.get("horse_id") or "")
            if not hid or hid not in horse_meta:
                continue
            meta = horse_meta[hid]
            horse_name = entry.get("horse_name")
            if horse_name and (not meta.get("name") or meta["name"] == hid):
                meta["name"] = str(horse_name)
            sex_age = entry.get("sex_age")
            if sex_age and not is_plausible_sex_age(meta.get("sex_age")):
                meta["sex_age"] = str(sex_age)
    except Exception:
        pass


def _enrich_horse_meta_from_flat(race_id: str, horse_meta: dict) -> None:
    try:
        year = int(race_id[:4])
        flat_path = _FLAT_DIR / str(year) / "race_result_flat.parquet"
        if not flat_path.exists():
            return
        import pandas as pd

        df_flat = pd.read_parquet(
            flat_path,
            columns=["race_id", "horse_id", "horse_number", "bracket_number", "sex_age", "horse_name"],
        )
        race_flat = df_flat[df_flat["race_id"] == race_id]
        for _, row in race_flat.iterrows():
            hid = str(row["horse_id"])
            if hid not in horse_meta:
                continue
            m = horse_meta[hid]
            if m["number"] is None and pd.notna(row.get("horse_number")):
                m["number"] = int(row["horse_number"])
            if m["bracket"] is None and pd.notna(row.get("bracket_number")):
                m["bracket"] = int(row["bracket_number"])
            if row.get("sex_age") and not is_plausible_sex_age(m.get("sex_age")):
                m["sex_age"] = str(row["sex_age"])
            if row.get("horse_name") and (not m.get("name") or m["name"] == hid):
                m["name"] = str(row["horse_name"])
    except Exception:
        pass


def build_race_megu_predictions(session: Session, race_id: str, model_version: str = _MODEL_VERSION) -> dict | None:
    """
    レース全出走馬の想定・実測めぐ指数を返す。

    Returns:
        {"race_id", "race_info", "race_level", "model_version", "horses": [...]} または None
    """
    from src.db.models import Race

    race = session.get(Race, race_id)
    if not race:
        return None

    merged_card = load_merged_race_card(race_id)
    if merged_card:
        patch_race_object(race, merged_card)

    beta_weight = load_beta_weight(session, model_version)
    par_time_target = lookup_par_time(session, race, model_version)
    transfer_map = load_transfer_map(session, model_version) if megu_predict_enabled() else {}
    predict_params = load_predict_params() if megu_predict_enabled() else None

    entry_rows = session.execute(
        sa_text("""
            SELECT e.horse_id, e.post_no, e.bracket_number, h.horse_name, e.jockey_weight,
                   COALESCE(NULLIF(e.sex_age, ''), h.sex) AS sex_age
            FROM entries e
            LEFT JOIN horses h ON h.horse_id = e.horse_id
            WHERE e.race_id = :race_id
            ORDER BY e.bracket_number NULLS LAST, e.post_no NULLS LAST, e.horse_id
        """),
        {"race_id": race_id},
    ).fetchall()

    if not entry_rows:
        entry_rows = session.execute(
            sa_text("""
                SELECT
                    mi.horse_id,
                    NULL::smallint AS post_no,
                    NULL::smallint AS bracket_number,
                    h.horse_name,
                    NULL::numeric   AS jockey_weight,
                    h.sex           AS sex_age
                FROM megu_index mi
                LEFT JOIN horses h ON h.horse_id = mi.horse_id
                WHERE mi.race_id = :race_id AND mi.model_version = :mv
                ORDER BY mi.horse_id
            """),
            {"race_id": race_id, "mv": model_version},
        ).fetchall()

    if not entry_rows:
        return None

    horse_ids = [r.horse_id for r in entry_rows]
    horse_meta = {
        r.horse_id: {
            "number": int(r.post_no) if r.post_no is not None else None,
            "bracket": int(r.bracket_number) if r.bracket_number is not None else None,
            "name": r.horse_name,
            "jockey_weight": float(r.jockey_weight) if r.jockey_weight is not None else None,
            "sex_age": r.sex_age if r.sex_age else None,
        }
        for r in entry_rows
    }
    _enrich_horse_meta_from_gcs(race_id, horse_meta)
    _enrich_horse_meta_from_flat(race_id, horse_meta)

    hist_rows = session.execute(
        sa_text("""
            SELECT
                mi.horse_id, mi.race_id, mi.megu_index, mi.par_time_sec,
                mi.adjusted_time_sec,
                r.race_date, r.venue, r.surface, r.distance,
                rr.finish_pos
            FROM megu_index mi
            JOIN races r ON r.race_id = mi.race_id
            LEFT JOIN race_results rr
                ON rr.race_id = mi.race_id AND rr.horse_id = mi.horse_id
            WHERE mi.horse_id = ANY(:horse_ids)
              AND mi.model_version = :mv
              AND mi.computation_status = 'valid'
              AND mi.megu_index IS NOT NULL
              AND mi.par_time_sec IS NOT NULL
              AND mi.race_id != :race_id
            ORDER BY mi.horse_id, r.race_date DESC, mi.race_id DESC
        """),
        {"horse_ids": horse_ids, "mv": model_version, "race_id": race_id},
    ).fetchall()

    actual_rows = session.execute(
        sa_text("""
            SELECT mi.horse_id, mi.megu_index, mi.computation_status,
                   rr.finish_time_sec, rr.finish_pos
            FROM megu_index mi
            LEFT JOIN race_results rr
                ON rr.race_id = mi.race_id AND rr.horse_id = mi.horse_id
            WHERE mi.race_id = :race_id AND mi.model_version = :mv
        """),
        {"race_id": race_id, "mv": model_version},
    ).fetchall()
    actual_map = {
        r.horse_id: {
            "megu_index": float(r.megu_index) if r.megu_index is not None else None,
            "computation_status": r.computation_status,
            "finish_time_sec": float(r.finish_time_sec) if r.finish_time_sec is not None else None,
            "finish_pos": int(r.finish_pos) if r.finish_pos is not None else None,
        }
        for r in actual_rows
    }

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

    hist_by_horse: dict = defaultdict(list)
    for r in hist_rows:
        hist_by_horse[r.horse_id].append(r)

    result_horses = []
    for hid in horse_ids:
        meta = horse_meta[hid]
        hist = hist_by_horse[hid]
        actual = actual_map.get(hid, {})
        result = result_map.get(hid, {})
        actual_megu = actual.get("megu_index")
        finish_time_sec = actual.get("finish_time_sec") or result.get("finish_time_sec")

        hist_dicts = [_hist_row_to_dict(r) for r in hist[:_HISTORY_LIMIT]]

        if not megu_predict_enabled():
            pred = dict(_EMPTY_PRED)
        elif par_time_target is None:
            pred = dict(_EMPTY_PRED)
        else:
            pred = predict_megu_scores(
                hist_dicts,
                par_time_target=par_time_target,
                surface_target=str(race.surface),
                distance_target=int(race.distance),
                jockey_weight=meta["jockey_weight"],
                sex_age=meta.get("sex_age"),
                beta_weight=beta_weight,
                transfer_map=transfer_map,
                max_races=_HISTORY_LIMIT,
                weights=predict_params.normalized_history_weights(_HISTORY_LIMIT),
                condition_weights=predict_params.condition_weights,
                tuning=predict_params.tuning,
            )

        cc = pred.get("condition_change") or {"type": "none", "label": None}

        result_horses.append({
            "horse_id": hid,
            "horse_name": meta["name"],
            "horse_number": meta["number"],
            "bracket_number": meta.get("bracket"),
            "sex_age": meta.get("sex_age"),
            "jockey_weight": meta["jockey_weight"],
            "finish_time_sec": finish_time_sec,
            "finish_pos": (
                int(actual.get("finish_pos") or result.get("finish_pos"))
                if (actual.get("finish_pos") or result.get("finish_pos")) is not None
                else None
            ),
            "actual_megu": actual_megu,
            "actual_status": actual.get("computation_status"),
            "base_megu": pred.get("base_megu"),
            "megu_adjusted": pred.get("megu_adjusted"),
            "weight_megu_delta": pred.get("weight_megu_delta"),
            "megu_final": pred.get("megu_final"),
            "condition_change": cc,
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

    if megu_predict_enabled():
        result_horses.sort(
            key=lambda h: h["megu_final"] if h["megu_final"] is not None else -999,
            reverse=True,
        )
    else:
        result_horses.sort(
            key=lambda h: h["actual_megu"] if h["actual_megu"] is not None else -999,
            reverse=True,
        )

    top_pred = (
        next((h["megu_final"] for h in result_horses if h["megu_final"] is not None), None)
        if megu_predict_enabled()
        else None
    )
    top_actual = next((h["actual_megu"] for h in result_horses if h["actual_megu"] is not None), None)
    for h in result_horses:
        pf, af = h.get("megu_final"), h.get("actual_megu")
        h["megu_gap"] = round(af - pf, 1) if af is not None and pf is not None else None
        h["pred_margin_sec"] = megu_margin_sec(top_pred, pf) if top_pred is not None else None
        h["actual_margin_sec"] = megu_margin_sec(top_actual, af) if top_actual is not None else None

    # めぐ指数フォールバック閾値（古馬＝3歳以上/4歳以上の実績中央値から算出、2022-2026）
    # 芝古馬: 未勝利p50≈73, 1勝p50≈103, 2勝p50≈105, 3勝p50≈104, OPp50≈100
    # ダート古馬: 未勝利p50≈71, 1勝p50≈96, 2勝p50≈101, 3勝p50≈101, OPp50≈103
    _MEGU_THRESHOLDS: dict[str, list[tuple[float, str]]] = {
        "芝": [
            (115, "G1級"),
            (108, "重賞級"),
            (103, "3勝級"),
            (99, "2勝級"),
            (93, "1勝級"),
        ],
        "ダート": [
            (112, "G1級"),
            (106, "重賞級"),
            (101, "3勝級"),
            (97, "2勝級"),
            (88, "1勝級"),
        ],
    }

    def _megu_fallback_label(field_avg: float, surface: str) -> str:
        """公式クラス不明時、古馬基準のめぐ指数閾値で分類。"""
        thresholds = _MEGU_THRESHOLDS.get(surface, _MEGU_THRESHOLDS["ダート"])
        for bound, cls in thresholds:
            if field_avg >= bound:
                return cls
        return "未勝利級"

    def _classify_race_level(
        race_name: str, race_class: str, grade: str,
        surface: str, field_avg: float | None,
    ) -> dict:
        """
        全レースにレベルラベルを付与。めぐ指数フォールバック閾値は古馬（3歳以上/4歳以上）実績基準。
        """
        rn = (race_name or "").strip()
        rc = (race_class or "").strip()
        gr = (grade or "").strip()
        sf = (surface or "").strip()

        # ── 1. 公式クラス優先判定（馬齢問わず） ──────────────────────────
        if "未勝利" in rc or "未勝利" in rn or gr == "未勝利":
            label = "未勝利級"
        elif "新馬" in rn or gr == "新馬":
            label = "未勝利級"
        elif "(G1)" in rn or gr == "G1":
            label = "G1級"
        elif "(G2)" in rn or gr == "G2":
            label = "重賞級"
        elif "(G3)" in rn or gr == "G3" or "(L)" in rn or gr == "L" \
                or "JGII" in rn or "JGIII" in rn:
            label = "重賞級"
        elif "(OP)" in rn or "オープン" in rc or gr == "OP":
            label = "3勝級"
        elif "３勝" in rc or "(3勝)" in rn:
            label = "3勝級"
        elif "２勝" in rc or "(2勝)" in rn:
            label = "2勝級"
        elif "１勝" in rc or "(1勝)" in rn or gr == "1勝":
            label = "1勝級"
        # ── 2. meguフォールバック（古馬基準閾値） ────────────────────────
        elif field_avg is not None:
            label = _megu_fallback_label(field_avg, sf)
        else:
            label = "?"

        return {
            "label": label,
            "field_avg_megu": round(field_avg, 1) if field_avg is not None else None,
        }

    # フィールド平均実測めぐ指数を計算
    actual_megu_vals = [
        a["megu_index"] for a in actual_map.values()
        if a.get("megu_index") is not None
    ]
    field_avg_megu: float | None = (
        sum(actual_megu_vals) / len(actual_megu_vals) if actual_megu_vals else None
    )

    race_level = _classify_race_level(
        race.race_name or "",
        getattr(race, "race_class", "") or "",
        race.grade or "",
        race.surface or "",
        field_avg_megu,
    )

    return {
        "race_id": race_id,
        "race_info": {
            "race_name": race.race_name,
            "venue": race.venue,
            "surface": race.surface,
            "distance": race.distance,
            "dist_band": dist_band(race.distance),
            "track_condition": race.track_condition,
            "grade": race.grade,
            "race_class": getattr(race, "race_class", None),
            "race_date": str(race.race_date) if race.race_date else None,
        },
        "race_level": race_level,
        "model_version": model_version,
        "index_note": (
            "実測めぐ指数はペース・馬場・斤量・レースレベル補正後（1点=0.1秒）。"
            + (
                " 想定・実測とも補正後。想定着差は想定めぐ1位との差。"
                if megu_predict_enabled()
                else " 想定めぐ指数は現在リセット中のため表示しません。"
            )
        ),
        "horses": result_horses,
    }

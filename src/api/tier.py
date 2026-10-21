"""公開（匿名）と会員の出し分け。

匿名で見られるのは、出馬表の事実と『勝率・複勝率』まで。想定単勝・期待値・めぐAI印（旧『推奨』）・推奨馬券・
その他の付随情報は会員（または開発者）だけ。UI の隠しではなく、サーバが匿名の応答から項目を落とす（許可リスト方式）。
仕様・項目表: docs/operations/tiered_access.md
"""

from __future__ import annotations

import re
from typing import Any

# ── 予測 API（/api/race/{id}/predictions, /predict）──

PUBLIC_ENTRY_FIELDS = (
    "horse_number", "horse_name", "horse_id", "bracket_number", "sex_age",
    "jockey_name", "jockey_weight", "trainer_name", "weight", "weight_change",
    "win_prob", "top3_prob", "show_prob",          # 勝率・複勝率
    "pred_rank",                                   # 匿名では勝率の順位に振り直す
)

PUBLIC_TOP_FIELDS = (
    "race_id", "race_name", "venue", "round", "date", "start_time", "grade", "surface", "distance",
    "track_condition", "status", "has_prediction", "total_horses", "predictions",
    "model_type", "model_version", "model_description", "prediction_source", "prediction_note",
    "entries_status", "provisional_entries", "freshness", "error", "message", "_mock",
)

# 匿名の応答から落とす（= 会員限定）の説明。UI の鍵表示と docs の表の元
LOCKED_GROUPS = {
    "連対率": ["top2_prob", "place_prob"],
    "想定オッズ": ["win_odds", "place_odds_min", "place_odds_max", "place_odds_avg", "place_odds_eff", "top2_odds_est", "popularity"],
    "期待値": ["ev_win", "ev_place", "ev_top2", "ev_win_cal", "ev_place_cal", "ev_top2_cal", "expected_value", "estimated_prob"],
    "めぐAI印・買い判断": ["mark_type", "recommendation", "buy_tier", "megu_mark", "kesi_reason", "bet_suggestion"],
    "展開・ラップ・追走難度": ["position_forecast", "tracking_v2", "l3f_coherent", "pace_forecast", "gate_bias", "pace_unified", "pace_type", "scenario_forecast", "lap_forecast"],
    "めぐ指数・成長度・クラス基準": ["expected_megu", "expected_megu_meta", "growth", "growth_meta", "class_benchmark"],
    "総合スコア": ["pred_score", "normalized_score", "composite_score", "composite_rank", "prob_calibrated", "prob_calibration"],
}


def _rank_by_win_prob(rows: list[dict]) -> list[dict]:
    if rows and all(isinstance(r.get("win_prob"), (int, float)) for r in rows):
        rows = sorted(rows, key=lambda r: -float(r["win_prob"]))
        for i, r in enumerate(rows, 1):
            r["pred_rank"] = i
    else:
        for r in rows:
            r.pop("pred_rank", None)
    return rows


def redact_predictions(resp: dict[str, Any]) -> dict[str, Any]:
    """匿名向けの予測応答。許可リストの項目だけ残し、会員限定の項目は『無い』ものとして返す。"""
    out = {k: v for k, v in resp.items() if k in PUBLIC_TOP_FIELDS}
    rows = []
    for e in resp.get("predictions") or []:
        if isinstance(e, dict):
            rows.append({k: e[k] for k in PUBLIC_ENTRY_FIELDS if k in e})
    if "predictions" in resp:
        out["predictions"] = _rank_by_win_prob(rows)
    out["access"] = {"tier": "public", "member_only": sorted(LOCKED_GROUPS)}
    return out


# ── レース詳細（/api/race/{id}）──

RACE_DETAIL_MEMBER_KEYS = (
    "race_odds", "race_pair_odds", "race_index", "race_paddock", "race_barometer", "race_trainer_comment", "smartrc_race",
)
SHUTUBA_MEMBER_ENTRY_KEYS = ("odds", "popularity")


def redact_race_detail(resp: dict[str, Any]) -> dict[str, Any]:
    """匿名向けのレース詳細。出馬表・結果・馬の事実は残し、オッズ・指数・パドック等の付随情報は落とす。"""
    out = {k: v for k, v in resp.items() if k not in RACE_DETAIL_MEMBER_KEYS}
    card = out.get("race_shutuba")
    if isinstance(card, dict) and isinstance(card.get("entries"), list):
        card = dict(card)
        card["entries"] = [
            {k: v for k, v in e.items() if k not in SHUTUBA_MEMBER_ENTRY_KEYS} if isinstance(e, dict) else e
            for e in card["entries"]
        ]
        out["race_shutuba"] = card
    avail = out.get("data_availability")
    if isinstance(avail, dict):
        out["data_availability"] = {k: v for k, v in avail.items() if k not in RACE_DETAIL_MEMBER_KEYS}
    out["access"] = {"tier": "public", "member_only": sorted(LOCKED_GROUPS)}
    return out


# ── 会員限定のパス（匿名は 401）──

_MEMBER_ONLY_PATTERNS = [re.compile(p) for p in (
    r"^/api/race/[^/]+/(tracking-difficulty|bundle|bloodline-aptitude)",
    r"^/api/predictions$",
    r"^/api/growth-curve/(?!status)",
    r"^/api/race-quality/",
    r"^/api/v1/betting/",
    r"^/api/v1/races/[^/]+/(final-odds|megu-index|megu-index-predicted|tracking-difficulty|predictions/laps)",
    r"^/api/v1/horse/[^/]+/(megu-index-history|growth-curve)",
    r"^/api/v1/(pedigree-race-stats|data-analysis)/",
    r"^/api/v1/(bloodline-cluster/horse-aptitude|pedigree/race-note-3d-v2)",
)]


def member_only_path(path: str) -> bool:
    return any(p.search(path) for p in _MEMBER_ONLY_PATTERNS)


MEMBER_REQUIRED_BODY = {
    "error": "この情報は会員限定です。ログインしてください。",
    "reason": "member_required",
    "login_url": "/login",
}

"""ホストのローカルに残した予測（shadow / バックフィル）を、画面向けの予測として返す。

GCS の race_predictions が無いレースで、推論経路が作った確率（勝率・連対率・複勝率）がローカルの parquet にあるときに使う。
優先順位は GCS > shadow（``predictions_v3``）> バックフィル（``predictions_backfill/<モデル名>``）。応答には
``prediction_source`` と注記を付ける。確率はモデルのもので、期待値(EV)は保存済みの最新（過去のレースは確定）オッズから計算する。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_NOTE = {
    "shadow": "検証用（shadow）。モデルの確率は前日夕方〜馬体重確定後に作った実モデルの出力ですが、画面向けに確定した予測ではありません。期待値は最新オッズで計算しています。",
    "backfill": "過去分の算出（バックフィル）。レース当日の確定した予測ではありません。その日より前の情報だけで後から算出したモデルの出力で、期待値は最新（確定）オッズで計算しています。",
}
_MARK_LABELS = {
    "honmei": "◎ 1着優位", "pair": "○ 連対候補", "anchor": "△ 馬券内の候補", "show_val": "▲ 単穴", "star": "★ 中穴妙味", "kesi": "消 3着内に来る確率が低い", "none": "",
}


def inference_dir() -> Path:
    return Path(os.environ.get("KEIBA_INFERENCE_DIR") or REPO_ROOT / "data" / "local" / "inference")


def _day_of(card: dict) -> str:
    return str(card.get("date") or "").replace("-", "").replace("/", "")[:8]


def shadow_model_label(day: str, inf: Path | None = None) -> str:
    """shadow の予測を作ったモデル名。shadow のファイルにはモデル名が無い（設定は後から変わる）ので、
    日付ごとの対応表 ``predictions_backfill/_shadow_models.json``（{"20261018": "bundle_2025"}）があればそれを使い、無ければ現在の設定。"""
    import json

    inf = inf or inference_dir()
    try:
        m = json.loads((inf / "predictions_backfill" / "_shadow_models.json").read_text(encoding="utf-8"))
        if m.get(day):
            return str(m[day])
    except (OSError, ValueError):
        pass
    return Path(os.environ.get("KEIBA_V3_BUNDLE") or "").name or "shadow"


PRIMARY_LABEL = "bundle_v6_2025"


def primary_model() -> str:
    """画面・集計・推奨が参照する予測モデル。既定は v6e（最新の親モデル）。KEIBA_PRED_MODEL=legacy で従来（shadow=v4 > バックフィル）に戻す。"""
    m = (os.environ.get("KEIBA_PRED_MODEL") or "v6e").strip().lower()
    return "legacy" if m in ("legacy", "v4", "v5", "off") else "v6e"


def find_primary_prediction(race_id: str, day: str, inf: Path | None = None) -> tuple[Path, str, str] | None:
    """v6e の予測（当日の shadow=predictions_v6、無ければ v6e のバックフィル）。KEIBA_PRED_MODEL=legacy のときは常に None。"""
    if primary_model() != "v6e" or not day or len(day) != 8:
        return None
    inf = inf or inference_dir()
    live = inf / "predictions_v6" / day / f"{race_id}.parquet"
    if live.exists():
        return live, "shadow", PRIMARY_LABEL
    back = inf / "predictions_backfill" / PRIMARY_LABEL / day / f"{race_id}.parquet"
    if back.exists():
        return back, "backfill", PRIMARY_LABEL
    return None


def find_local_prediction(race_id: str, day: str, inf: Path | None = None) -> tuple[Path, str, str] | None:
    """(parquet のパス, source, モデル名)。優先順位は v6e（既定）> shadow(v4) > バックフィル。"""
    inf = inf or inference_dir()
    if not day or len(day) != 8:
        return None
    prim = find_primary_prediction(race_id, day, inf)
    if prim:
        return prim
    shadow = inf / "predictions_v3" / day / f"{race_id}.parquet"
    if shadow.exists():
        return shadow, "shadow", shadow_model_label(day, inf)
    cands = sorted((inf / "predictions_backfill").glob(f"*/{day}/{race_id}.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        return cands[0], "backfill", cands[0].parent.parent.name
    return None


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def score_entries(entries: list[dict], odds: dict[int, dict], alpha: float = 0.55) -> list[dict]:
    """確率つきの出走馬に、オッズ・期待値・合成スコア・印・買い推奨度を付ける（in-place ではなく新しい一覧を返す）。"""
    from src.utils.race_probabilities import (
        assign_kesi_mark,
        assign_mece_marks,
        attach_calibrated_ev,
        buy_recommendation_tier,
        market_top2_fair_odds,
        place_effective_odds,
    )

    out: list[dict] = []
    top2_fair = market_top2_fair_odds({int(e.get("horse_number") or 0): _num((odds.get(int(e.get("horse_number") or 0)) or {}).get("win_odds")) or 0.0 for e in entries})
    for e in entries:
        o = odds.get(int(e.get("horse_number") or 0), {})
        win_odds = _num(o.get("win_odds"))
        pmin, pmax = _num(o.get("place_odds_min")), _num(o.get("place_odds_max"))
        place_avg = (pmin + pmax) / 2 if (pmin and pmax) else None
        top2_odds = top2_fair.get(int(e.get("horse_number") or 0))
        place_eff = place_effective_odds(pmin, pmax)
        pw, p2, p3 = float(e["win_prob"]), float(e["top2_prob"]), float(e["top3_prob"])
        ev_win = round(pw * win_odds, 3) if win_odds and pw > 0 else None
        ev_top2 = round(p2 * top2_odds, 3) if top2_odds and p2 > 0 else None
        ev_place = round(p3 * place_eff, 3) if place_eff and p3 > 0 else None
        if p3 > 0 and ev_place:
            composite = (p3 ** alpha) * (ev_place ** (1 - alpha))
        else:
            composite = p3 if p3 > 0 else 0.0
        out.append({
            **e, "win_odds": round(win_odds, 1) if win_odds else None,
            "place_odds_min": round(pmin, 1) if pmin else None, "place_odds_max": round(pmax, 1) if pmax else None,
            "place_odds_avg": round(place_avg, 1) if place_avg else None, "place_odds_eff": round(place_eff, 2) if place_eff else None, "top2_odds_est": top2_odds,
            "popularity": o.get("popularity") or None, "estimated_prob": p3,
            "ev_win": ev_win, "ev_top2": ev_top2, "ev_place": ev_place, "expected_value": ev_place,
            "composite_score": round(composite, 4),
        })
    out.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, s in enumerate(out):
        s["composite_rank"] = i + 1
    assign_mece_marks(out)
    assign_kesi_mark(out)
    for s in out:
        s["recommendation"] = _MARK_LABELS.get(s.get("mark_type") or "none", "")
        attach_calibrated_ev(s)
        s["buy_tier"] = buy_recommendation_tier(s)
    return out


def build_payload(race_id: str, card: dict, odds_doc: dict | None, probs, source: str, label: str, alpha: float = 0.55, market: bool = True) -> dict:
    """probs: race_id・horse_id・p_win・p_top2・p_top3 の表（pandas）。card: race_shutuba の保存内容。"""
    info = {str(e.get("horse_id") or ""): e for e in (card.get("entries") or []) if isinstance(e, dict)}
    odds: dict[int, dict] = {}
    for o in (odds_doc or {}).get("entries") or []:
        if isinstance(o, dict) and o.get("horse_number") is not None:
            odds[int(o["horse_number"])] = o
    df = probs.sort_values("p_win", ascending=False).reset_index(drop=True)
    entries: list[dict] = []
    for i, r in df.iterrows():
        e = info.get(str(r["horse_id"]), {})
        pw = float(r["p_win"])
        w = e.get("weight") or None
        entries.append({
            "pred_rank": i + 1, "horse_number": int(e.get("horse_number") or 0), "horse_name": e.get("horse_name", ""), "horse_id": str(r["horse_id"]),
            "bracket_number": e.get("bracket_number") or None, "sex_age": e.get("sex_age") or None, "jockey_name": e.get("jockey_name") or None,
            "jockey_weight": e.get("jockey_weight") or None, "trainer_name": e.get("trainer_name") or None,
            "weight": w, "weight_change": e.get("weight_change") if w else None,
            "pred_score": round(float(np.log(max(pw, 1e-9) / max(1 - pw, 1e-9))), 4), "normalized_score": round(pw, 4),
            "win_prob": round(pw, 4), "top2_prob": round(float(r["p_top2"]), 4), "top3_prob": round(float(r["p_top3"]), 4),
            "place_prob": round(float(r["p_top2"]), 4), "show_prob": round(float(r["p_top3"]), 4),
        })
    scored = score_entries(entries, odds, alpha)
    note = SOURCE_NOTE.get(source, "")
    mtype = "ensemble_v6e" if "v6" in label else "ensemble_v5" if "v5" in label else "ensemble_v4" if "v4" in label else "ensemble_v3"
    payload = {
        "race_id": race_id, "race_name": card.get("race_name", ""), "venue": card.get("venue", ""), "round": card.get("round", 0),
        "surface": card.get("surface", ""), "distance": card.get("distance", 0), "track_condition": card.get("track_condition", ""),
        "status": "success", "has_prediction": True, "model_type": mtype, "model_version": label,
        "model_description": (f"アンサンブル v6e（{label}）の確率 — {'算出済み（過去分）' if source == 'backfill' else '当日の予測'}" if "v6" in label
                              else f"アンサンブル（{label}）の確率 — {'算出済み（参考）' if source == 'backfill' else '検証用（shadow）'}"),
        "total_horses": len(scored), "predictions": scored,
        "prediction_source": source, "prediction_note": note,
        "_compute_meta": {"source": source, "model": label},
    }
    # 市場補正（KEIBA_PROB_CALIBRATION=1 のときだけ）: 保存済みの予測は書き換えず、読み出し時に確率・EV・印を付け直す
    from src.utils.prob_calibration import apply_prob_calibration

    if not market:  # 実オッズを使わない（有料ゾーン案A・B）: 市場補正をかけない
        return payload
    return apply_prob_calibration(payload, odds_phase=(odds_doc or {}).get("phase"), alpha=alpha)


def load_primary_prediction_payload(race_id: str, storage: Any, alpha: float = 0.55, inf: Path | None = None, market: bool = True) -> dict | None:
    """v6e の予測がローカルにあるレースだけ、その応答を返す（GCS の古い予測より優先するために使う）。無ければ None。"""
    if primary_model() != "v6e":
        return None
    try:
        card = storage.load("race_shutuba", race_id) or {}
        if not find_primary_prediction(race_id, _day_of(card), inf):
            return None
    except Exception:  # noqa: BLE001
        return None
    return load_local_prediction_payload(race_id, storage, alpha=alpha, inf=inf, market=market)


def load_local_prediction_payload(race_id: str, storage: Any, alpha: float = 0.55, inf: Path | None = None, market: bool = True) -> dict | None:
    """ローカルに予測があれば画面向けの応答にして返す。無い・読めないときは None（呼び出し側は従来の応答に進む）。"""
    try:
        card = storage.load("race_shutuba", race_id) or {}
        found = find_local_prediction(race_id, _day_of(card), inf)
        if not found:
            return None
        import pandas as pd

        path, source, label = found
        probs = pd.read_parquet(path)
        if probs.empty or not {"horse_id", "p_win", "p_top2", "p_top3"} <= set(probs.columns):
            return None
        odds_doc = None
        if market:  # market=False は実オッズに触れない
            try:
                odds_doc = storage.load("race_odds", race_id)
            except Exception:
                odds_doc = None
        return build_payload(race_id, card, odds_doc, probs, source, label, alpha, market=market)
    except Exception:  # noqa: BLE001
        import logging

        logging.getLogger("api.local_predictions").exception("ローカルの予測を読めなかった: %s", race_id)
        return None

"""予測がまだ無いレースの『出馬表の馬 + 取得済みの最新オッズ』だけの応答。

stg は実データ（GCS・ホスト）を表示する環境。実モデルの予測（race_predictions）が無いレースに、これまではモックを
別の形（horses[]）で返していたため、画面（predictions[] を読む）では馬が一覧に出ず、予測が無いことも分からなかった。
ここでは予測を作り出さず、実データの事実だけを、実際の予測と同じ形（predictions[]）で返す。
"""

from __future__ import annotations

from typing import Any

NO_PREDICTION_MESSAGE = "このレースの AI 予測はまだ生成されていません（実モデルの予測が出力されていない）。出馬表の馬と、取得済みの最新オッズだけを表示しています。"
PROVISIONAL_MESSAGE = "出馬表は公示の途中です（頭数・騎手・枠順は確定前で、変わることがあります）。AI 予測は、出走馬と枠順が確定した前日以降に表示します。"


def _entries(storage: Any, race_id: str) -> list[dict]:
    for cat, key in (("race_shutuba", "entries"), ("race_result", "results")):
        try:
            card = storage.load(cat, race_id)
        except Exception:
            card = None
        rows = (card or {}).get(key) or (card or {}).get("entries") or []
        rows = [r for r in rows if isinstance(r, dict) and r.get("horse_number") is not None]
        if rows:
            return rows
    return []


def _provisional_entries(storage: Any, race_id: str) -> list[dict]:
    """馬番がまだ無い出馬表（特別登録・出走馬決定の段階）の馬。登録順（ページの並び）のまま返す。"""
    try:
        card = storage.load("race_shutuba", race_id)
    except Exception:
        card = None
    return [r for r in ((card or {}).get("entries") or []) if isinstance(r, dict) and (r.get("horse_name") or r.get("horse_id"))]


def entries_only_predictions(race_id: str, storage: Any) -> dict[str, Any]:
    rows = _entries(storage, race_id)
    numbered = [r for r in rows if int(r.get("horse_number") or 0) > 0]
    if len({int(r["horse_number"]) for r in numbered}) != len(rows) or len(numbered) != len(rows):
        rows = []                                       # 馬番が欠けている・重複している → 確定前として扱う
    if not rows:
        prov = _provisional_entries(storage, race_id)
        if prov:
            return _provisional_response(race_id, prov)
    odds: dict[int, dict] = {}
    try:
        for o in (storage.load("race_odds", race_id) or {}).get("entries") or []:
            if isinstance(o, dict) and o.get("horse_number") is not None:
                odds[int(o["horse_number"])] = o
    except Exception:
        pass
    preds = []
    for r in sorted(rows, key=lambda x: int(x["horse_number"])):
        n = int(r["horse_number"])
        o = odds.get(n, {})
        w = r.get("weight") or None                      # 馬体重は当日（発走の約 1 時間前）の発表後に入る。前日は 0 → None
        preds.append({
            "horse_number": n,
            "horse_name": r.get("horse_name"),
            "horse_id": r.get("horse_id"),
            "bracket_number": r.get("bracket_number") or None,
            "sex_age": r.get("sex_age") or None,
            "jockey_name": r.get("jockey_name") or None,
            "jockey_weight": r.get("jockey_weight") or None,
            "trainer_name": r.get("trainer_name") or None,
            "weight": w,
            "weight_change": r.get("weight_change") if w else None,     # 増減 0 は正当な値。馬体重が無いときだけ None
            "win_odds": o.get("win_odds") or None,
            "place_odds_min": o.get("place_odds_min") or None,
            "place_odds_max": o.get("place_odds_max") or None,
            "popularity": o.get("popularity") or None,
        })
    return {
        "race_id": race_id,
        "status": "no_prediction" if preds else "no_entries",
        "has_prediction": False,
        "model_type": None,
        "total_horses": len(preds),
        "predictions": preds,
        "message": NO_PREDICTION_MESSAGE,
    }


def _provisional_response(race_id: str, rows: list[dict]) -> dict[str, Any]:
    """馬番が無い段階の応答。馬番は null（画面は『-』を出す）、horse_number_provisional=true を付ける。"""
    preds = []
    for r in rows:
        w = r.get("weight") or None
        preds.append({
            "horse_number": None,
            "horse_number_provisional": True,
            "horse_name": r.get("horse_name"),
            "horse_id": r.get("horse_id"),
            "bracket_number": None,
            "sex_age": r.get("sex_age") or None,
            "jockey_name": r.get("jockey_name") or None,
            "jockey_weight": r.get("jockey_weight") or None,
            "trainer_name": r.get("trainer_name") or None,
            "weight": w,
            "weight_change": r.get("weight_change") if w else None,
            "win_odds": None,
            "place_odds_min": None,
            "place_odds_max": None,
            "popularity": None,
        })
    return {
        "race_id": race_id,
        "status": "no_prediction",
        "has_prediction": False,
        "provisional_entries": True,
        "model_type": None,
        "total_horses": len(preds),
        "predictions": preds,
        "message": PROVISIONAL_MESSAGE,
    }

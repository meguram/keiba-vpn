"""過去走の通過順位から作る『位置取り予想（展開）』。実モデルの予測ではなく、馬柱（race_shutuba_past）だけで決まる集計。

前日夕方に確定する出馬表と馬柱だけで作れる（当日の追加取得なし）。対象レースの日付より前の過去走だけを使う（先読みなし）。
脚質の境界は pace_predictor（前半ラップの予測モデル）の『逃げ・先行』の数え方と同じ:
  最初のコーナーの通過順位を (順位-1)/(頭数-1) に直し、≤0.15 を逃げ、≤0.35 を先行とする。差し(≤0.65)・追込はそれより後ろ。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

FRONT_MAX = 0.15
STALKER_MAX = 0.35
MID_MAX = 0.65
WEIGHTS = (0.35, 0.25, 0.20, 0.12, 0.08)      # 直近の走ほど重い（新しい順）

STYLES = [("front", "逃げ"), ("stalker", "先行"), ("mid", "差し"), ("closer", "追込")]
_LABEL = dict(STYLES)

_PASS = re.compile(r"(?<![\d.])(\d{1,2}(?:-\d{1,2})+)(?![\d.])")
_FIELD = re.compile(r"(\d+)頭")

NOTES = {
    "high": "逃げ・先行を主張する馬が多く、前半から流れやすい見込み。差し・追込に展開が向きやすい。",
    "slow": "前に行きたい馬が少なく、前半は落ち着く見込み。逃げ・先行が残りやすい。",
    "mid": "前に行く馬の数は平均的で、平均的なペースの見込み。",
    "unknown": "過去走の通過順位が揃っていない馬が多く、ペースは判定できない。",
}


def _digits(s: Any) -> str:
    return re.sub(r"\D", "", str(s or ""))


def parse_run(raw: str) -> tuple[float, int, int, str] | None:
    """馬柱の1行 → (最初のコーナーの正規化位置, その順位, 頭数, 通過順位の文字列)。読めなければ None。"""
    m = _PASS.search(raw or "")
    f = _FIELD.search(raw or "")
    if not m or not f:
        return None
    field = int(f.group(1))
    first = int(m.group(1).split("-")[0])
    if field < 4 or not 1 <= first <= field:
        return None
    return (first - 1) / (field - 1), first, field, m.group(1)


def style_of(norm: float) -> str:
    if norm <= FRONT_MAX:
        return "front"
    if norm <= STALKER_MAX:
        return "stalker"
    if norm <= MID_MAX:
        return "mid"
    return "closer"


def horse_forecast(past_races: Iterable[dict], race_date: str | None) -> dict | None:
    """1 頭分。race_date（YYYYMMDD）より前の近5走から、重み付き平均の位置取りと脚質を返す。使える走が 0 なら None。"""
    rd = _digits(race_date)
    runs = []
    for p in past_races or []:
        if not isinstance(p, dict):
            continue
        d = _digits(p.get("date"))
        if rd and d and d >= rd:
            continue
        r = parse_run(str(p.get("raw") or ""))
        if r:
            runs.append((d, r))
    runs.sort(key=lambda x: x[0], reverse=True)
    runs = runs[: len(WEIGHTS)]
    if not runs:
        return None
    w = WEIGHTS[: len(runs)]
    norm = sum(wi * r[1][0] for wi, r in zip(w, runs)) / sum(w)
    code = style_of(norm)
    n = len(runs)
    return {
        "style": _LABEL[code], "style_code": code, "early_pos_norm": round(norm, 3),
        "avg_first_corner": round(sum(r[1][1] for r in runs) / n, 1), "runs_used": n,
        "confidence": "high" if n >= 4 else "mid" if n >= 2 else "low", "last_passing": runs[0][1][3], "source": "past_runs",
    }


def pace_forecast(forecasts: Iterable[dict | None], field_size: int) -> dict:
    """レース単位。pace_predictor の特徴量（front_runner_count / stalker_count / early_pressure）と同じ定義で数え、ハイ/ミドル/スローを見込む。"""
    fs = [f for f in forecasts if f]
    field = max(int(field_size or 0), len(fs))
    front = sum(1 for f in fs if f["style_code"] == "front")
    stalker = sum(1 for f in fs if f["style_code"] == "stalker")
    pressure = round((front + stalker) / field, 3) if field else 0.0
    if field and len(fs) < max(4, field * 0.5):
        level, label = "unknown", "判定不可"
    elif front >= 3 or (front >= 2 and pressure >= 0.4):
        level, label = "high", "ハイ"
    elif front == 0 or (front == 1 and pressure < 0.35):
        level, label = "slow", "スロー"
    else:
        level, label = "mid", "ミドル"
    note = NOTES[level]
    if front == 1 and level != "unknown":
        note += " 単騎逃げの形。"
    return {
        "level": level, "label": label, "note": note, "front_runner_count": front, "stalker_count": stalker,
        "early_pressure": pressure, "field_size": field, "known_horses": len(fs),
        "source": "past_runs_heuristic",
    }


def build_position_forecast(past_doc: dict | None, entries: Iterable[dict], race_date: str | None) -> tuple[dict[int, dict | None], dict]:
    """entries: horse_number / horse_id を持つ行。返り値: ({馬番: 予想 or None}, レースのペース見込み)。"""
    rows = {}
    for e in (past_doc or {}).get("entries") or []:
        if isinstance(e, dict):
            rows[str(e.get("horse_id") or "")] = e
            if e.get("horse_number") is not None:
                rows.setdefault(f"#{int(e['horse_number'])}", e)
    by_no: dict[int, dict | None] = {}
    for e in entries:
        if e.get("horse_number") is None:
            continue
        n = int(e["horse_number"])
        row = rows.get(str(e.get("horse_id") or "")) or rows.get(f"#{n}") or {}
        by_no[n] = horse_forecast(row.get("past_races") or [], race_date)
    known = sorted((n for n, f in by_no.items() if f), key=lambda n: (by_no[n]["early_pos_norm"], n))
    for rank, n in enumerate(known, 1):
        by_no[n]["early_pos_est"] = rank
    return by_no, pace_forecast(by_no.values(), len(by_no))


def attach_position_forecast(resp: dict, past_doc: dict | None, race_date: str | None) -> dict:
    """予測 API の応答（predictions[]）に position_forecast と pace_forecast を足す。応答の他の項目は変えない。"""
    preds = resp.get("predictions") or []
    if not preds:
        return resp
    by_no, pace = build_position_forecast(past_doc, preds, race_date)
    for p in preds:
        if p.get("horse_number") is not None:
            p["position_forecast"] = by_no.get(int(p["horse_number"]))
    resp["pace_forecast"] = pace
    return resp

"""過去走の通過順位から作る『位置取り予想（展開）』。実モデルの予測ではなく、馬柱（race_shutuba_past）だけで決まる集計。

前日夕方に確定する出馬表と馬柱だけで作れる（当日の追加取得なし）。対象レースの日付より前の過去走だけを使う（先読みなし）。
脚質の境界は pace_predictor（前半ラップの予測モデル）の『逃げ・先行』の数え方と同じ:
  最初のコーナーの通過順位を (順位-1)/(頭数-1) に直し、≤0.15 を逃げ、≤0.35 を先行とする。差し(≤0.65)・追込はそれより後ろ。

枠順の補正: コースごとの『内枠が前に出やすいか』の係数（config/gate_bias.json、2020-2025 の実データ集計）で、
  early_pos_adj = clip(early_pos_norm + k × (gate_rel − 0.5), 0, 1)   gate_rel = (馬番-1)/(最終馬番-1)（0=最内, 1=最外）
と補正する。k は面ごとの平均へ縮約済みの値をそのまま使う（枠順で説明できる位置取りのばらつきは小さく、倍率は掛けない）。
ペース（逃げ・先行の数）は馬の脚質そのもので数えるため、補正前の過去走ベースの脚質を使う。
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
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


GATE_BIAS_PATH = Path(__file__).resolve().parents[3] / "config" / "gate_bias.json"
GATE_MIN_FIELD = 8          # 係数を作った条件（8頭立以上）。これより少ない頭数では補正しない
_CLS_LEVEL = {"内が前に出やすい": "inner", "外が前に出やすい": "outer", "差なし": "flat", "データ不足": "low"}


@functools.lru_cache(maxsize=1)
def load_gate_bias() -> dict:
    try:
        return json.loads(GATE_BIAS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 係数が読めなくても位置取り予想自体は出す
        return {"_meta": {"surface_mean": {}}, "course": {}}


def course_key(venue: Any, surface: Any, distance: Any) -> str | None:
    v = str(venue or "").strip()
    s = "ダート" if str(surface or "").startswith("ダ") else "芝" if str(surface or "").startswith("芝") else ""
    d = _digits(distance)
    return f"{v}|{s}|{int(d)}" if v and s and d else None


def gate_coeff(venue: Any, surface: Any, distance: Any) -> dict:
    """コースの枠順係数。返り値: {k, level, cls, n_races, key, source}。level は inner/outer/flat/low/na。
    『データ不足』のコースと、コースが特定できない場合は面（芝/ダート）の平均を使う。面も分からなければ補正なし（k=0, level=na）。"""
    gb = load_gate_bias()
    key = course_key(venue, surface, distance)
    surf = "ダート" if str(surface or "").startswith("ダ") else "芝" if str(surface or "").startswith("芝") else ""
    mean = (gb.get("_meta") or {}).get("surface_mean", {}).get(surf)
    row = (gb.get("course") or {}).get(key or "")
    if row and row.get("cls") != "データ不足":
        return {"k": float(row["k_shrunk"]), "level": _CLS_LEVEL.get(row["cls"], "flat"), "cls": row["cls"],
                "n_races": int(row.get("n_races") or 0), "key": key, "source": "course"}
    if mean is None:
        return {"k": 0.0, "level": "na", "cls": "適用外", "n_races": 0, "key": key, "source": "none"}
    return {"k": float(mean), "level": "low", "cls": "データ不足", "n_races": int((row or {}).get("n_races") or 0), "key": key, "source": "surface_mean"}


def gate_rel(horse_number: Any, last_number: int) -> float | None:
    """馬番の相対位置（0=最内, 1=最外）。馬番が無い・頭数が少なすぎるときは None。"""
    try:
        n = int(horse_number)
    except (TypeError, ValueError):
        return None
    if last_number < 2 or not 1 <= n <= last_number:
        return None
    return (n - 1) / (last_number - 1)


def apply_gate(f: dict, rel: float | None, k: float) -> dict:
    """1 頭分の予想に枠順を反映する。early_pos_norm を補正後の値にし、補正前は early_pos_raw に残す。"""
    raw = f["early_pos_norm"]
    f["early_pos_raw"] = raw
    f["style_raw_code"] = f["style_code"]
    f["gate_rel"] = None if rel is None else round(rel, 3)
    shift = 0.0 if rel is None else k * (rel - 0.5)
    adj = min(1.0, max(0.0, raw + shift))
    f["gate_shift"] = round(adj - raw, 3)
    f["early_pos_norm"] = round(adj, 3)
    f["style_code"] = style_of(adj)
    f["style"] = _LABEL[f["style_code"]]
    return f


def gate_note(c: dict) -> str:
    n = c.get("n_races") or 0
    if c["level"] == "inner":
        return f"このコースは内枠が前に出やすい傾向（過去 {n} レース）。同じ位置取りなら内枠の馬の方が前・ラチ寄りになりやすい。"
    if c["level"] == "outer":
        return f"このコースは外枠が前に出やすい傾向（過去 {n} レース）。内枠が不利になりやすい形。"
    if c["level"] == "flat":
        return f"このコースは内外の枠による序盤の差はほぼない（過去 {n} レース）。"
    if c["level"] == "low":
        return "このコースは過去のレースが少ないため、芝・ダート全体の平均傾向で補正しています（参考程度）。"
    return "枠順の補正は行っていません。"


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
    code = lambda f: f.get("style_raw_code") or f["style_code"]   # noqa: E731 - 枠順補正前の脚質で数える
    front = sum(1 for f in fs if code(f) == "front")
    stalker = sum(1 for f in fs if code(f) == "stalker")
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


def build_position_forecast(past_doc: dict | None, entries: Iterable[dict], race_date: str | None,
                           course: dict | None = None) -> tuple[dict[int, dict | None], dict, dict]:
    """entries: horse_number / horse_id を持つ行。course: {venue, surface, distance}。
    返り値: ({馬番: 予想 or None}, レースのペース見込み, 枠順バイアス)。"""
    entries = list(entries)
    rows = {}
    for e in (past_doc or {}).get("entries") or []:
        if isinstance(e, dict):
            rows[str(e.get("horse_id") or "")] = e
            if e.get("horse_number") is not None:
                rows.setdefault(f"#{int(e['horse_number'])}", e)
    numbers = [int(e["horse_number"]) for e in entries if e.get("horse_number") is not None]
    last_no = max(numbers) if numbers else 0
    c = course or {}
    coeff = gate_coeff(c.get("venue"), c.get("surface"), c.get("distance"))
    use_gate = coeff["level"] != "na" and last_no >= GATE_MIN_FIELD
    by_no: dict[int, dict | None] = {}
    for e in entries:
        if e.get("horse_number") is None:
            continue
        n = int(e["horse_number"])
        row = rows.get(str(e.get("horse_id") or "")) or rows.get(f"#{n}") or {}
        by_no[n] = horse_forecast(row.get("past_races") or [], race_date)
    pace = pace_forecast(by_no.values(), len(by_no))      # ペースは補正前の脚質で数える
    for n, f in by_no.items():
        if f:
            apply_gate(f, gate_rel(n, last_no) if use_gate else None, coeff["k"] if use_gate else 0.0)
    known = sorted((n for n, f in by_no.items() if f), key=lambda n: (by_no[n]["early_pos_norm"], n))
    for rank, n in enumerate(known, 1):
        by_no[n]["early_pos_est"] = rank
    gate = {k: coeff[k] for k in ("level", "cls", "n_races", "key", "source")}
    gate.update({"k": round(coeff["k"], 4) if use_gate else 0.0, "applied": use_gate, "field_last_number": last_no,
                 "note": gate_note(coeff) if use_gate else ("頭数が少ないため枠順の補正は行っていません。" if coeff["level"] != "na" else gate_note(coeff))})
    return by_no, pace, gate


def attach_position_forecast(resp: dict, past_doc: dict | None, race_date: str | None, course: dict | None = None) -> dict:
    """予測 API の応答（predictions[]）に position_forecast・pace_forecast・gate_bias を足す。応答の他の項目は変えない。
    course 省略時は応答自身の venue / surface / distance を使う。"""
    preds = resp.get("predictions") or []
    if not preds:
        return resp
    course = {**{k: resp.get(k) for k in ("venue", "surface", "distance")}, **{k: v for k, v in (course or {}).items() if v}}
    by_no, pace, gate = build_position_forecast(past_doc, preds, race_date, course)
    for p in preds:
        if p.get("horse_number") is not None:
            p["position_forecast"] = by_no.get(int(p["horse_number"]))
    resp["pace_forecast"] = pace
    resp["gate_bias"] = gate
    return resp

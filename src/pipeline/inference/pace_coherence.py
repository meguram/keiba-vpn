"""想定位置取り・ペース・想定上がりを、結果から読み取れる関係で結びつける整合層（画面表示用、既定オフ）。

背景（tp-53）: 位置取り側（position_forecast）と追走難度側（tracking_difficulty）は独立に作られ、共通のペース想定が無い。
追走難度側のペース型は 91% が『ミドル』で実測と無相関、想定上がりは位置取りにほぼ反応しない（実測は後ろの馬ほど上がりが速く、
ペースが速いほどその差が大きい）。ここでは次の 3 つを 1 つの流れにする。
  1. 位置取り側の先行勢（逃げ・先行の頭数、圧力）から、前半 3F の速さ z（同条件の中央値からの標準化）を推定 → ペース（ハイ/ミドル/スロー）
  2. 位置取り（0=先頭, 1=最後方）とペース z で、上がりの偏差を決める（後ろの馬ほど速い。ペースが速いほど差が大きい）
  3. 同条件の上がり中央値（場×面×距離×馬場）に、上の偏差と自己の過去の上がり偏差を足して想定上がり（秒）にする
係数と同条件の表は config/pace_coherence.json（tp53_analysis が作る）。有効化は環境変数 KEIBA_PACE_COHERENCE=1（既定は無効）。
先読みなし: 自己の過去の上がり偏差は、対象日より前の走だけを使う。
"""

from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "pace_coherence.json"
ENV_FLAG = "KEIBA_PACE_COHERENCE"
ENV_FLAG_V2 = "KEIBA_PACE_COHERENCE_V2"        # ペース想定 v2（tp-54）。既定は無効＝現行のまま
CONFIG_V2_PATH = Path(__file__).resolve().parents[3] / "config" / "pace_coherence_v2.json"

_DATE = re.compile(r"^(\d{4})[./](\d{1,2})[./](\d{1,2})")
_VENUE = re.compile(r"^\d{4}[./]\d{1,2}[./]\d{1,2}\s*(?: )?\s*(?:\d+)?\s*([^\d\s ]{1,3})")
_COURSE = re.compile(r"(芝|ダ|障)\s*(\d{3,4})")
_GOING = re.compile(r"(?<![^\s ])(良|稍重|稍|重|不良|不)(?![^\s ])")
_L3F = re.compile(r"\((\d{2}\.\d)\)")
_FIELD = re.compile(r"(\d+)頭")
_SURF = {"芝": "芝", "ダ": "ダート", "障": "障害"}


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0").strip().lower() in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=2)
def load_config(path: str | None = None) -> dict | None:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def enabled_v2() -> bool:
    return os.environ.get(ENV_FLAG_V2, "0").strip().lower() in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=2)
def load_pace_v2(path: str | None = None) -> dict | None:
    p = Path(path) if path else CONFIG_V2_PATH
    if not p.exists():
        return None
    try:
        return (json.loads(p.read_text(encoding="utf-8")) or {}).get("pace_v2")
    except Exception:
        return None


def pace_v2_features(front: float, stalker: float, pressure: float, known: float, field: float, surface: Any, distance: Any, going: Any) -> dict[str, float]:
    """tp54_pace.design と同じ定義（テストで一致を確認）。出走表の段階で分かる量だけ。"""
    fr = min(float(front or 0), 6.0)
    st = min(float(stalker or 0), 10.0)
    fd = float(field or 14.0)
    turf = 1.0 if surface == "芝" else 0.0
    heavy = 1.0 if going_group(going) == "道悪" else 0.0
    dist = (float(distance or 1600) - 1600.0) / 1000.0
    return {"f0": float(fr == 0), "f1": float(fr == 1), "f2": float(fr == 2), "f3p": float(fr >= 3), "stalk": st / 5.0, "stalk2": (st / 5.0) ** 2,
            "f0_stalk": float(fr == 0) * st / 5.0, "pressure": float(pressure or 0.0), "known_frac": min(float(known or 0) / max(fd, 1.0), 1.0),
            "field": (fd - 14.0) / 4.0, "turf": turf, "dist": dist, "dist2": dist ** 2, "heavy": heavy, "turf_dist": turf * dist, "heavy_turf": heavy * turf}


def predict_pace_z_v2(m: dict, front: float, stalker: float, pressure: float, known: float, field: float, surface: Any, distance: Any, going: Any) -> float:
    x = pace_v2_features(front, stalker, pressure, known, field, surface, distance, going)
    return float(m["intercept"] + sum(c * x[k] for k, c in zip(m["features"], m["coef"])))


def going_group(going: Any) -> str:
    return "良" if str(going or "").strip() == "良" else "道悪"


def _dist(distance: Any) -> int:
    try:
        v = float(distance)
    except (TypeError, ValueError):
        return 0
    return 0 if v != v else int(v)


def cohort_key(venue: Any, surface: Any, distance: Any, going: Any) -> str:
    return f"{venue}|{surface}|{_dist(distance)}|{going_group(going)}"


def fallback_key(surface: Any, distance: Any, going: Any) -> str:
    return f"{surface}|{_dist(distance)}|{going_group(going)}"


def parse_past_run(raw: str) -> dict | None:
    """馬柱の 1 行 → {date, venue, surface, distance, going, l3f}。読めない行は None。"""
    s = str(raw or "")
    m = _DATE.search(s)
    c = _COURSE.search(s)
    l = _L3F.search(s)
    if not (m and c and l):
        return None
    v = _VENUE.search(s)
    gm = _GOING.search(s)
    l3f = float(l.group(1))
    if not 30.0 < l3f < 50.0:
        return None
    return {"date": f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}", "venue": v.group(1) if v else "", "surface": _SURF.get(c.group(1), ""),
            "distance": int(c.group(2)), "going": gm.group(1) if gm else "良", "l3f": l3f}


def cohort_l3f_median(cfg: dict, venue: Any, surface: Any, distance: Any, going: Any) -> float | None:
    t = cfg.get("cohort_l3f") or {}
    v = t.get(cohort_key(venue, surface, distance, going))
    if v is None:
        v = (cfg.get("cohort_l3f_fallback") or {}).get(fallback_key(surface, distance, going))
    return None if v is None else float(v)


def prior_l3f_dev(past_races: Iterable[dict], race_date: str | None, cfg: dict, n: int = 5) -> float | None:
    """直近 n 走（対象日より前）の『自分の上がり − 同条件の上がり中央値』の平均。使える走が無ければ None。"""
    rd = re.sub(r"\D", "", str(race_date or ""))
    runs = []
    for p in past_races or []:
        if not isinstance(p, dict):
            continue
        r = parse_past_run(str(p.get("raw") or ""))
        if not r or (rd and r["date"] >= rd):
            continue
        runs.append(r)
    runs.sort(key=lambda r: r["date"], reverse=True)
    devs = []
    for r in runs[:n]:
        med = cohort_l3f_median(cfg, r["venue"], r["surface"], r["distance"], r["going"])
        if med is not None:
            devs.append(r["l3f"] - med)
    return sum(devs) / len(devs) if devs else None


def pace_features(front: int, stalker: int, pressure: float, surface: Any, distance: Any, going: Any) -> list[float]:
    return [min(float(front), 5.0), min(float(stalker), 8.0), float(pressure), 1.0 if surface == "芝" else 0.0,
            (float(distance or 1600) - 1600.0) / 1000.0, 1.0 if going_group(going) == "道悪" else 0.0]


def predict_pace_z(cfg: dict, front: int, stalker: int, pressure: float, surface: Any, distance: Any, going: Any) -> float:
    m = cfg["pace"]
    x = pace_features(front, stalker, pressure, surface, distance, going)
    return float(m["intercept"] + sum(c * v for c, v in zip(m["coef"], x)))


def pace_label(z: float, cfg: dict) -> tuple[str, str]:
    th = cfg["pace"]["thresholds"]
    if z <= th["fast"]:
        return "high", "ハイ"
    if z >= th["slow"]:
        return "slow", "スロー"
    return "mid", "ミドル"


SLOPE_MAX = -0.05      # 位置取りが後ろ1.0あたりの上がり偏差(秒)の上限。実測はどのペースでも負（後ろの馬ほど速い）なので、正になる外挿は抑える


def l3f_slope(cfg: dict, z: float, distance: Any, surface: Any) -> float:
    m = cfg["l3f"]
    c = dict(zip(m["features"], m["coef"]))
    zc = max(-2.0, min(2.0, z))
    s = c["e"] + c["e_z"] * zc + c["e_dist"] * (float(distance or 1600) - 1600.0) / 1000.0 + c["e_surf"] * (1.0 if surface == "芝" else 0.0)
    return min(s, float(m.get("slope_max", SLOPE_MAX)))


def l3f_dev(cfg: dict, e_d: float, z: float, distance: Any, surface: Any, prior_d: float) -> float:
    """位置取り(レース内の平均との差 e_d)・ペース z・距離・面・自己の過去偏差(レース内の平均との差)から、上がり偏差(秒)を出す。"""
    m = cfg["l3f"]
    c = dict(zip(m["features"], m["coef"]))
    return float(m["intercept"] + l3f_slope(cfg, z, distance, surface) * e_d + c["prior"] * prior_d)


def coherent_forecast(cfg: dict, horses: list[dict], pf_pace: dict, course: dict, race_date: str | None = None) -> dict:
    """horses: [{horse_number, early_pos_norm, past_races(任意)}]。返り値: {'pace': {...}, 'horses': {馬番: {...}}}。
    位置取りが無い馬には上がりを出さない。上がり(秒) = 同条件の中央値 + 偏差（ペースが速いほど全体が遅くなる補正は、実測で弱いので入れない）。"""
    venue, surface, distance, going = course.get("venue"), course.get("surface"), course.get("distance"), course.get("track_condition") or course.get("going")
    z = predict_pace_z(cfg, int(pf_pace.get("front_runner_count") or 0), int(pf_pace.get("stalker_count") or 0), float(pf_pace.get("early_pressure") or 0.0), surface, distance, going)
    v2 = load_pace_v2() if enabled_v2() else None
    if v2:       # tp-54: 逃げ0頭の有無・逃げ×先行の交互作用・頭数・既知馬の割合を入れた較正（既定は無効）
        z = predict_pace_z_v2(v2, pf_pace.get("front_runner_count") or 0, pf_pace.get("stalker_count") or 0, pf_pace.get("early_pressure") or 0.0,
                              pf_pace.get("known_horses") or 0, pf_pace.get("field_size") or len(horses), surface, distance, going)
    level, label = pace_label(z, cfg)
    t = (cfg.get("cohort_t3f") or {}).get(cohort_key(venue, surface, distance, going)) or {}
    first3f = (t.get("median") + z * t.get("std")) if t.get("median") is not None and t.get("std") else None
    base = cohort_l3f_median(cfg, venue, surface, distance, going)
    known = [h for h in horses if h.get("early_pos_norm") is not None]
    ebar = sum(h["early_pos_norm"] for h in known) / len(known) if known else 0.0
    priors = {h["horse_number"]: prior_l3f_dev(h.get("past_races") or [], race_date, cfg) for h in known}
    pvals = [v for v in priors.values() if v is not None]
    pbar = sum(pvals) / len(pvals) if pvals else 0.0
    out: dict[int, dict] = {}
    for h in known:
        pr = priors.get(h["horse_number"])
        dev = l3f_dev(cfg, h["early_pos_norm"] - ebar, z, distance, surface, (pr - pbar) if pr is not None else 0.0)
        out[h["horse_number"]] = {"dev_sec": round(dev, 2), "seconds": round(base + dev, 1) if base is not None else None,
                                  "prior_dev": None if pr is None else round(pr, 2), "has_prior": pr is not None}
    ranked = sorted(out.items(), key=lambda kv: kv[1]["dev_sec"])
    for i, (hn, row) in enumerate(ranked, 1):
        row["rank"] = i
    return {"pace": {"z": round(z, 2), "level": level, "label": label, "first_3f_sec": None if first3f is None else round(first3f, 1),
                     "last_3f_baseline_sec": None if base is None else round(base, 1), "source": "pace_coherence"},
            "horses": out}


def attach_pace_coherence(resp: dict, past_doc: dict | None = None, race_date: str | None = None, course: dict | None = None) -> dict:
    """予測 API の応答（position_forecast / pace_forecast が付いたもの）に pace_unified と各馬の l3f_coherent を足す。無効・設定なし・材料不足は何もしない。"""
    if not enabled():
        return resp
    cfg = load_config()
    preds = resp.get("predictions") or []
    pf = resp.get("pace_forecast")
    if not cfg or not preds or not pf or pf.get("level") == "unknown":
        return resp
    rows = {}
    for e in (past_doc or {}).get("entries") or []:
        if isinstance(e, dict):
            rows[str(e.get("horse_id") or "")] = e
    horses = []
    for p in preds:
        f = p.get("position_forecast") or {}
        if p.get("horse_number") is None or f.get("early_pos_norm") is None:
            continue
        horses.append({"horse_number": int(p["horse_number"]), "early_pos_norm": float(f["early_pos_norm"]),
                       "past_races": (rows.get(str(p.get("horse_id") or "")) or {}).get("past_races") or []})
    if not horses:
        return resp
    course = {**{"venue": resp.get("venue"), "surface": resp.get("surface"), "distance": resp.get("distance"), "track_condition": resp.get("track_condition")},
              **{k: v for k, v in (course or {}).items() if v}}
    res = coherent_forecast(cfg, horses, pf, course, race_date)
    resp["pace_unified"] = res["pace"]
    for p in preds:
        if p.get("horse_number") is not None:
            p["l3f_coherent"] = res["horses"].get(int(p["horse_number"]))
    return resp

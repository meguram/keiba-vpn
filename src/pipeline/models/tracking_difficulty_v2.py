"""追走難度 v2（tp-54）: 『想定位置取りを取るのに必要な通過ラップが、その馬にとって楽か大変か』。

要求（demand）= 想定位置取り p̂ × 今回の想定ペース z_hat（先行勢の数・圧力から）とコース・距離・馬場・頭数。
供給（capacity）= その馬の過去（as-of）の、位置取りの効果を除いた上がりの力、前で運んだ/速いペースで運んだときの終いの落ち方、走る位置の癖。
難度 = 要求 − 供給 を、実測の『終いの落ち』Y（上がり偏差 − 位置取りの効果を除いた過去の上がり偏差、秒）への回帰で表す。
ease_pct_v2 は −(予測 Y) を、学習期間の予測値の分布で 0〜100 の分位に直したもの（50 が標準、60 以上=楽、30 未満=大変）。
drain_v2 は予測 Y（秒）。前走比（ease_vs_prev）は『前走と同じ位置を取るなら、今回の条件は前走の条件より楽か』。

先読みなし: 入力はすべて出走表段階（想定位置・想定ペース）と、対象日より前の走からの集計だけ。
有効化は環境変数 KEIBA_TRACKING_V2=1（既定は無効）。既存の ease_pct・消耗は変えない。
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config" / "tracking_difficulty_v2"
ENV_FLAG = "KEIBA_TRACKING_V2"
Y_CLIP = 4.0

VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]

# 学習・推論で同じ列を使う。すべて対象日より前に分かる量。
FEATURES = [
    # 要求（想定位置取り・想定ペース・コース）
    "pf_early", "pf_rank_norm", "gate_norm", "field_size", "z_hat", "pf_front", "pf_stalker", "pf_pressure", "pf_known_frac",
    "turf", "distance_km", "heavy", "venue_code", "demand_idx", "demand_idx_z",
    # 供給（その馬の過去 5 走。馬柱から推論時にも作れる量だけ）
    "w_adj", "w_dev", "w_dev_front", "w_dev_rear", "w_early", "w_n", "w_nfront", "w_nrear", "age", "weight", "weight_change", "rest_days",
    # 前走との関係
    "prev_early", "prev_dev", "d_dist", "early_shift", "same_style_run",
]


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0").strip().lower() in ("1", "true", "yes", "on")


def make_features(h: pd.DataFrame) -> pd.DataFrame:
    """学習表（tp54_dataset.build の出力）または推論用に同じ列名で作った表から、FEATURES の表を作る。

    必要な列: pf_early, pf_rank, field_size, horse_number, z_hat, pf_front, pf_stalker, pf_pressure, pf_known, surface, distance,
    track_condition または going_group, venue, w_adj, w_dev*, w_n*, age, weight, weight_change, rest_days, prev_*。
    """
    fs = h["field_size"].astype(float).clip(lower=1)
    pe = h["pf_early"].astype(float)
    z = h["z_hat"].astype(float)
    going = h["going_group"] if "going_group" in h.columns else h["track_condition"].map(lambda t: "良" if str(t) == "良" else "道悪")
    f = pd.DataFrame(index=h.index)
    f["pf_early"] = pe
    f["pf_rank_norm"] = (h["pf_rank"].astype(float) - 1.0) / (fs - 1.0).clip(lower=1)
    f["gate_norm"] = h["horse_number"].astype(float) / fs
    f["field_size"] = fs
    f["z_hat"] = z
    f["pf_front"] = h["pf_front"].astype(float)
    f["pf_stalker"] = h["pf_stalker"].astype(float)
    f["pf_pressure"] = h["pf_pressure"].astype(float)
    f["pf_known_frac"] = (h["pf_known"].astype(float) / fs).clip(upper=1.0)
    f["turf"] = (h["surface"] == "芝").astype(float)
    f["distance_km"] = h["distance"].astype(float) / 1000.0
    f["heavy"] = (going == "道悪").astype(float)
    f["venue_code"] = h["venue"].map({v: i for i, v in enumerate(VENUES)}).astype(float)
    f["demand_idx"] = (1.0 - pe) * (-z)                  # 前にいるほど、ペースが速いほど大きい
    f["demand_idx_z"] = (1.0 - pe) * (-z.clip(-2, 2)) * (f["distance_km"] - 0.6)
    for c in ("w_adj", "w_dev", "w_dev_front", "w_dev_rear", "w_early", "w_n", "w_nfront", "w_nrear", "age", "weight", "weight_change", "rest_days"):
        f[c] = h[c].astype(float) if c in h.columns else np.nan
    f["prev_early"] = h["prev_act_early_norm"].astype(float)
    f["prev_dev"] = h["prev_c_dev"].astype(float)
    f["d_dist"] = (h["distance"].astype(float) - h["prev_distance"].astype(float)) / 1000.0
    f["early_shift"] = pe - h["w_early"].astype(float)       # いつもの位置より前/後ろに行く想定か
    f["same_style_run"] = (np.abs(pe - h["prev_act_early_norm"].astype(float)) < 0.15).astype(float)
    return f[FEATURES]


@functools.lru_cache(maxsize=1)
def load_model(path: str | None = None) -> dict | None:
    d = Path(path) if path else CONFIG_DIR
    meta, mfile = d / "meta.json", d / "model.txt"
    if not (meta.exists() and mfile.exists()):
        return None
    try:
        import lightgbm as lgb

        m = json.loads(meta.read_text(encoding="utf-8"))
        m["booster"] = lgb.Booster(model_file=str(mfile))
        if (d / "model_demand.txt").exists():
            m["booster_demand"] = lgb.Booster(model_file=str(d / "model_demand.txt"))
        return m
    except Exception:
        return None


def predict_drain(model: dict, feats: pd.DataFrame) -> np.ndarray:
    """予測 Y（秒）。大きい＝終いが落ちる（大変）。"""
    return model["booster"].predict(feats[model["features"]], num_iteration=model["booster"].best_iteration or None)


def to_ease_pct(drain: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """予測 Y を、学習期間の予測値の分布（101 分位）での位置に直す。Y が小さい（楽）ほど 100 に近い。"""
    q = np.asarray(quantiles, float)
    pos = np.searchsorted(q, np.asarray(drain, float), side="right") / len(q) * 100.0
    return np.clip(100.0 - pos, 0.0, 100.0)


def ease_label(p: float) -> str:
    if p >= 75:
        return "非常に楽"
    if p >= 60:
        return "楽"
    if p >= 45:
        return "普通"
    if p >= 30:
        return "やや大変"
    return "大変"


def attach_v2(rows: pd.DataFrame, model: dict | None = None) -> pd.DataFrame:
    """推論用: rows（1 頭 1 行、make_features が要る列）に drain_v2 / ease_pct_v2 / ease_label_v2 を足して返す。"""
    model = model or load_model()
    if model is None or rows.empty:
        return rows
    feats = make_features(rows)
    drain = predict_drain(model, feats)
    out = rows.copy()
    out["drain_v2"] = np.round(drain, 3)
    if model.get("booster_demand") is not None:
        dem = model["booster_demand"].predict(feats[model["demand_features"]])
        out["demand_v2"] = np.round(dem, 3)
        out["capacity_v2"] = np.round(drain - dem, 3)
    out["ease_pct_v2"] = np.round(to_ease_pct(drain, model["quantiles"]), 1)
    out["ease_label_v2"] = out["ease_pct_v2"].map(ease_label)
    return out


# ───────────────────────── 推論（予測 API の応答に足す。KEIBA_TRACKING_V2=1 のときだけ）

W = 5                  # 直近 5 走（馬柱で見える範囲）
FRONT_E, REAR_E = 0.35, 0.65


def _position_effect(pe: dict, early: float, distance: float, turf: bool) -> float:
    c = dict(zip(pe["features"], pe["coef"]))
    e = early - 0.5
    return c["e"] * e + c["e_dist"] * e * (distance - 1600.0) / 1000.0 + c["e_turf"] * e * (1.0 if turf else 0.0)


def capacity_from_past(past_races: list[dict], race_date: str | None, cfg: dict, pe: dict) -> dict:
    """馬柱の raw 行（対象日より前）から、学習表（tp54_dataset）と同じ定義で直近 5 走の集計を作る。"""
    import re

    from src.pipeline.inference.pace_coherence import cohort_l3f_median, parse_past_run
    from src.pipeline.inference.position_forecast import parse_run

    rd = re.sub(r"\D", "", str(race_date or ""))
    runs = []
    for p in past_races or []:
        raw = str((p or {}).get("raw") or "")
        r = parse_past_run(raw)
        if not r or (rd and r["date"] >= rd):
            continue
        pr = parse_run(raw)
        med = cohort_l3f_median(cfg, r["venue"], r["surface"], r["distance"], r["going"])
        c_dev = None if med is None else r["l3f"] - med
        early = None if pr is None else pr[0]
        runs.append({"date": r["date"], "early": early, "c_dev": c_dev, "distance": r["distance"], "surface": r["surface"], "field": None if pr is None else pr[2]})
    runs.sort(key=lambda x: x["date"], reverse=True)
    last = runs[:W]

    def mean(vals: list[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else float("nan")

    adj = [x["c_dev"] - _position_effect(pe, x["early"], x["distance"], x["surface"] == "芝") for x in last if x["c_dev"] is not None and x["early"] is not None]
    dev = [x["c_dev"] for x in last if x["c_dev"] is not None]
    front = [x["c_dev"] for x in last if x["c_dev"] is not None and x["early"] is not None and x["early"] < FRONT_E]
    rear = [x["c_dev"] for x in last if x["c_dev"] is not None and x["early"] is not None and x["early"] > REAR_E]
    early_all = [x["early"] for x in last if x["early"] is not None]
    prev = runs[0] if runs else None
    rest = None
    if prev and rd:
        try:
            a = pd.Timestamp(f"{rd[:4]}-{rd[4:6]}-{rd[6:8]}")
            b = pd.Timestamp(f"{prev['date'][:4]}-{prev['date'][4:6]}-{prev['date'][6:8]}")
            rest = float((a - b).days)
        except Exception:
            rest = None
    nan = float("nan")
    return {
        "w_adj": mean(adj), "w_dev": mean(dev), "w_dev_front": mean(front), "w_dev_rear": mean(rear), "w_early": mean(early_all),
        "w_n": float(len(early_all)), "w_nfront": float(sum(1 for x in last if x["early"] is not None and x["early"] < FRONT_E)),
        "w_nrear": float(sum(1 for x in last if x["early"] is not None and x["early"] > REAR_E)),
        "prev_act_early_norm": nan if not prev or prev["early"] is None else prev["early"], "prev_c_dev": nan if not prev or prev["c_dev"] is None else prev["c_dev"],
        "prev_distance": nan if not prev else float(prev["distance"]), "rest_days": nan if rest is None else rest,
    }


def attach_tracking_v2_to_response(resp: dict, past_doc: dict | None = None, race_date: str | None = None, course: dict | None = None, card: dict | None = None) -> dict:
    """予測 API の応答（position_forecast / pace_forecast が付いたもの）に、各馬の tracking_v2 を足す。無効・材料不足は何もしない。

    tracking_v2 = {ease_pct, label, drain_sec, demand_sec?}。既存の追走難度（/tracking-difficulty の ease_pct・消耗）は変えない。
    """
    if not enabled():
        return resp
    model = load_model()
    preds = resp.get("predictions") or []
    pf = resp.get("pace_forecast")
    if model is None or not preds or not pf or pf.get("level") == "unknown":
        return resp
    from src.pipeline.inference import pace_coherence as pcoh

    cfg = pcoh.load_config()
    v2 = pcoh.load_pace_v2()
    pe_path = CONFIG_DIR / "position_effect.json"
    if not (cfg and v2 and pe_path.exists()):
        return resp
    pe = json.loads(pe_path.read_text(encoding="utf-8"))
    course = {**{"venue": resp.get("venue"), "surface": resp.get("surface"), "distance": resp.get("distance"), "track_condition": resp.get("track_condition")},
              **{k: v for k, v in (course or {}).items() if v}}
    venue, surface, distance, going = course.get("venue"), course.get("surface"), course.get("distance"), course.get("track_condition") or course.get("going")
    field = int(pf.get("field_size") or len(preds))
    z_hat = pcoh.predict_pace_z_v2(v2, pf.get("front_runner_count") or 0, pf.get("stalker_count") or 0, pf.get("early_pressure") or 0.0, pf.get("known_horses") or 0, field, surface, distance, going)
    entries = {}
    for src in ((past_doc or {}).get("entries") or [], (card or {}).get("entries") or []):
        for e in src:
            if isinstance(e, dict):
                entries.setdefault(str(e.get("horse_id") or ""), {}).update(e)
    rows, idx = [], []
    for p in preds:
        f = p.get("position_forecast") or {}
        if p.get("horse_number") is None or f.get("early_pos_norm") is None:
            continue
        e = entries.get(str(p.get("horse_id") or ""), {})
        cap = capacity_from_past(e.get("past_races") or [], race_date, cfg, pe)
        age = None
        m = __import__("re").search(r"(\d+)", str(e.get("sex_age") or ""))
        if m:
            age = float(m.group(1))
        rows.append({"horse_number": int(p["horse_number"]), "pf_early": float(f["early_pos_norm"]), "pf_rank": f.get("rank") or float("nan"), "field_size": field, "z_hat": z_hat,
                     "pf_front": pf.get("front_runner_count") or 0, "pf_stalker": pf.get("stalker_count") or 0, "pf_pressure": pf.get("early_pressure") or 0.0,
                     "pf_known": pf.get("known_horses") or 0, "surface": surface, "distance": distance, "track_condition": going, "venue": venue,
                     "age": age if age is not None else float("nan"), "weight": float(e.get("weight") or float("nan")) if str(e.get("weight") or "").replace(".", "").isdigit() else float("nan"),
                     "weight_change": float("nan"), **cap})
        idx.append(p)
    if not rows:
        return resp
    df = pd.DataFrame(rows)
    out = attach_v2(df, model)
    for p, (_, r) in zip(idx, out.iterrows()):
        p["tracking_v2"] = {"ease_pct": float(r["ease_pct_v2"]), "label": r["ease_label_v2"], "drain_sec": float(r["drain_v2"])}
        if "demand_v2" in r:
            p["tracking_v2"].update({"demand_sec": float(r["demand_v2"]), "capacity_sec": float(r["capacity_v2"])})
    resp["tracking_v2_meta"] = {"version": model.get("version"), "pace_z": round(float(z_hat), 2),
                                "definition": "想定位置取りを取るのに必要な通過ラップが、その馬にとって楽か大変か（終いの落ちの予測。50=標準、60以上=楽、30未満=大変）"}
    return resp

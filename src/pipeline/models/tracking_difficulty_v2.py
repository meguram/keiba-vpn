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
    # 供給（その馬の過去）
    "ability_adj", "prior_l3f_adj", "prior_l3f_dev_front", "prior_l3f_dev_rear", "prior_l3f_adj_fast", "prior_early", "prior_tpre_dev", "prior_finq",
    "n_prior", "n_prior_front", "n_prior_fast", "age", "weight", "weight_change", "rest_days",
    # 前走との関係
    "prev_early", "prev_l3f_dev", "prev_z", "d_dist", "early_shift", "same_style_run",
]


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0").strip().lower() in ("1", "true", "yes", "on")


def make_features(h: pd.DataFrame) -> pd.DataFrame:
    """学習表（tp54_dataset.build の出力）または推論用に同じ列名で作った表から、FEATURES の表を作る。

    必要な列: pf_early, pf_rank, field_size, horse_number, z_hat, pf_front, pf_stalker, pf_pressure, pf_known, surface, distance,
    track_condition または going_group, venue, ability_adj, prior_*, n_prior*, age, weight, weight_change, rest_days, prev_*。
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
    for c in ("ability_adj", "prior_l3f_adj", "prior_l3f_dev_front", "prior_l3f_dev_rear", "prior_l3f_adj_fast", "prior_early", "prior_tpre_dev", "prior_finq",
              "n_prior", "n_prior_front", "n_prior_fast", "age", "weight", "weight_change", "rest_days"):
        f[c] = h[c].astype(float) if c in h.columns else np.nan
    f["prev_early"] = h["prev_act_early_norm"].astype(float)
    f["prev_l3f_dev"] = h["prev_act_l3f_dev"].astype(float)
    f["prev_z"] = h["prev_z_act"].astype(float)
    f["d_dist"] = (h["distance"].astype(float) - h["prev_distance"].astype(float)) / 1000.0
    f["early_shift"] = pe - h["prior_early"].astype(float)       # いつもの位置より前/後ろに行く想定か
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
    out["ease_pct_v2"] = np.round(to_ease_pct(drain, model["quantiles"]), 1)
    out["ease_label_v2"] = out["ease_pct_v2"].map(ease_label)
    return out

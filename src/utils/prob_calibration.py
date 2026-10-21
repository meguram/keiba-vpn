"""確率の較正（市場情報を使う）。モデルの勝率・連対率・複勝率を、単勝オッズから作る市場の確率と対数線形にブレンドして、実績に合わせる。

背景と検証: docs/html/modeling/chapters/prob_calibration_gap.html（tp-70）。要点: 実績はモデルと市場の“間”にあり（モデルが市場から離れるほど自信過剰）、
logit(p_cal) = a + w1×logit(p_model) + w2×logit(p_market) をレース内で合計 1 / 2 / 3 に合わせると、LogLoss・人気別のずれ・EV の信頼性が改善する。
市場オッズは『モデルの特徴量』ではなく、較正（と馬券の判断）にだけ使う。係数は config/prob_calibration.json（src/scripts/data/tp70_fit.py export で再作成）。

有効化は環境変数 KEIBA_PROB_CALIBRATION=1（既定オフ）。次のときは何もしない（None を返し、呼び出し側は従来の確率を使う）:
  無効 / 設定ファイル無し / 8 頭未満 / 単勝オッズが全頭揃っていない / 市場が予測オッズ（モデルの出力から作った値で独立な市場ではない）。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.race_probabilities import harville_top2_prob, harville_top3_prob

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "prob_calibration.json"
EPS = 1e-6
MIN_FIELD = 8
KSUM = {"win": 1.0, "top2": 2.0, "top3": 3.0}
POP_BINS = ((1, "1人気"), (2, "2人気"), (3, "3人気"), (5, "4-5人気"), (9, "6-9人気"), (10**6, "10人気以下"))


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def scale_to_sum(p: np.ndarray, k: float) -> np.ndarray:
    """レース内の確率の合計を k に合わせる（logit に共通のオフセットを足して二分法）。"""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    k = min(k, len(p) - 1e-3)
    z = _logit(p)
    lo, hi = -12.0, 12.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).sum() > k:
            hi = mid
        else:
            lo = mid
    return _sigmoid(z + (lo + hi) / 2)


@lru_cache(maxsize=1)
def load_config() -> dict:
    try:
        d = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def is_enabled() -> bool:
    return os.environ.get("KEIBA_PROB_CALIBRATION", "") == "1" and bool(load_config().get("targets"))


def market_probs(win_odds: Sequence[float]) -> dict[str, np.ndarray]:
    """単勝オッズ → 市場の勝率（正規化 1/odds）と、Harville の連対率・複勝率。"""
    inv = 1.0 / np.asarray(win_odds, float)
    q = inv / inv.sum()
    return {"win": q, "top2": harville_top2_prob(q), "top3": harville_top3_prob(q)}


def _pop_classes(win_odds: Sequence[float]) -> list[str]:
    order = np.argsort(np.argsort(np.asarray(win_odds, float), kind="stable"), kind="stable") + 1   # 1 始まりの人気（オッズ昇順、同じなら並び順）
    out = []
    for r in order:
        for hi, name in POP_BINS:
            if r <= hi:
                out.append(name)
                break
    return out


def calibrate_arrays(p_model: dict[str, Sequence[float]], win_odds: Sequence[float], method: dict[str, str] | None = None) -> dict[str, np.ndarray] | None:
    """p_model = {'win','top2','top3': 配列}。較正後の同じ形の辞書、または None（適用しない）。"""
    cfg = load_config()
    n = len(win_odds)
    if not cfg.get("targets") or n < MIN_FIELD:
        return None
    wo = np.asarray(win_odds, float)
    if not np.all(np.isfinite(wo)) or np.any(wo <= 0):
        return None
    mk = market_probs(wo)
    meth = {**(cfg.get("default_method") or {}), **(method or {})}
    classes = _pop_classes(wo)
    out: dict[str, np.ndarray] = {}
    for t in ("win", "top2", "top3"):
        spec = cfg["targets"][t]
        lp, lm = _logit(p_model[t]), _logit(mk[t])
        if meth.get(t, "blend") == "blend_pop":
            z = np.array([spec["blend_pop"][c]["intercept"] + spec["blend_pop"][c]["coef"][0] * a + spec["blend_pop"][c]["coef"][1] * b for c, a, b in zip(classes, lp, lm)])
        else:
            s = spec["blend"]
            z = s["intercept"] + s["coef"][0] * lp + s["coef"][1] * lm
        out[t] = scale_to_sum(_sigmoid(z), KSUM[t])
    return out


def calibrate_entries(entries: Sequence[dict], odds_by_horse: dict[int, float], *, market_is_real: bool = True) -> list[dict] | None:
    """画面・API 用。entries は horse_number と win_prob / top2_prob / top3_prob を持つ辞書の並び。
    戻り値は entries と同順の [{'win_prob','top2_prob','top3_prob'}（較正後）]、適用しないときは None。
    呼び出し側は、有効なとき元の確率を *_model に退避し、較正後を win_prob などに入れて EV・印・推奨に使う。"""
    if not is_enabled() or not market_is_real:
        return None
    try:
        odds = [float(odds_by_horse.get(int(e.get("horse_number") or 0)) or 0) for e in entries]
        pm = {"win": [float(e["win_prob"]) for e in entries], "top2": [float(e["top2_prob"]) for e in entries], "top3": [float(e["top3_prob"]) for e in entries]}
    except (KeyError, TypeError, ValueError):
        return None
    cal = calibrate_arrays(pm, odds)
    if cal is None:
        return None
    return [{"win_prob": round(float(cal["win"][i]), 4), "top2_prob": round(float(cal["top2"][i]), 4), "top3_prob": round(float(cal["top3"][i]), 4)} for i in range(len(entries))]

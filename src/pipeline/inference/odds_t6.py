"""想定単勝・想定複勝（再学習の試行 6）の推論。学習は ``src.scripts.data.fit_final_odds_t6``。

- 単勝: L2 とレース内ソフトマックスの 2 本の log 占有率（各々レース内で正規化）を平均 → 較正の直線 → レース内で正規化 → 0.797 / 占有率。
- 複勝: 直接予測（place_raw）と、占有率の並びから複勝を写す 2 段目（実際の市場の占有率で学んだ写像を、予測した占有率に当てる）を半々。
  → 較正の直線 → 複勝オッズの中央。
``odds_sec77.predict_frame`` と同じ入出力（特徴量の表 → race_id・horse_id・win_odds・place_mid・place_min・place_max）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEYS = ["race_id", "horse_id"]
DEFAULT_PLACE_HALF_RANGE = 0.12
_cache: dict[str, Any] = {}


def model_dir() -> Path:
    return Path(os.environ.get("ODDS_T6_DIR") or Path(__file__).resolve().parents[3] / "data/local/modeling/odds_pred/serving_t6")


def available(d: Path | None = None) -> bool:
    d = d or model_dir()
    return all((d / f).exists() for f in ("meta.json", "share_l2.txt", "share_softmax.txt", "place_direct.txt", "place_struct.txt"))


def normalize_log_share(logv: np.ndarray, rid: np.ndarray) -> np.ndarray:
    """生のスコア → レース内で合計 1 になる log 占有率。"""
    s = pd.Series(np.asarray(logv, float))
    g = s.groupby(rid)
    e = np.exp(s - g.transform("max"))
    return (s - g.transform("max") - np.log(e.groupby(rid).transform("sum"))).to_numpy()


def place_struct_features(logp: np.ndarray, rid: np.ndarray) -> pd.DataFrame:
    """レース内の占有率の並びから作る量（自分の log 占有率・順位・上位の占有率・頭数・エントロピー・自分より上位の占有率の合計）。"""
    s = pd.Series(np.asarray(logp, float))
    g = s.groupby(rid)
    p = np.exp(s - g.transform("max"))
    p = p / p.groupby(rid).transform("sum")
    gp = p.groupby(rid)
    f = pd.DataFrame({"lp": np.log(p), "rk": gp.rank(ascending=False), "n": gp.transform("size"),
                      "ent": (-(p * np.log(p))).groupby(rid).transform("sum"),
                      "p1": gp.transform("max"), "p2": gp.transform(lambda x: np.sort(x.to_numpy())[-2] if len(x) > 1 else 0),
                      "p3": gp.transform(lambda x: np.sort(x.to_numpy())[-3] if len(x) > 2 else 0)})
    f["cum_above"] = p.groupby(rid).transform(lambda x: x.sort_values(ascending=False).cumsum().reindex(x.index)) - p
    f["rk_n"] = f["rk"] / f["n"]
    return f


def _load(d: Path) -> dict[str, Any]:
    mt = (d / "meta.json").stat().st_mtime
    ent = _cache.get(str(d))
    if ent and ent[0] == mt:
        return ent[1]
    import lightgbm as lgb

    pack = {"meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
            **{k: lgb.Booster(model_file=str(d / f"{k}.txt")) for k in ("share_l2", "share_softmax", "place_direct", "place_struct")}}
    _cache[str(d)] = (mt, pack)
    return pack


def reload() -> None:
    _cache.clear()


def feature_columns(d: Path | None = None) -> list[str]:
    return list(_load(d or model_dir())["meta"]["cols"])


def predict_frame(X: pd.DataFrame, d: Path | None = None) -> pd.DataFrame:
    pack = _load(d or model_dir())
    meta = pack["meta"]
    cols = meta["cols"]
    miss = [c for c in cols if c not in X.columns]
    if miss:
        raise KeyError(f"odds_t6 の特徴量が足りない列: {len(miss)} 列 {miss[:8]}")
    A = X[cols].astype("float32")
    rid = X["race_id"].astype(str).to_numpy()
    ls = (normalize_log_share(pack["share_l2"].predict(A), rid) + normalize_log_share(pack["share_softmax"].predict(A, raw_score=True), rid)) / 2
    place_direct = pack["place_direct"].predict(A)
    st = pack["place_struct"].predict(place_struct_features(ls, rid))
    w = float(meta["place_struct_weight"])
    place_raw = w * st + (1 - w) * place_direct
    a1, b1 = meta["win_line"]
    a2, b2 = meta["place_line"]
    share = np.exp(normalize_log_share(a1 + b1 * ls, rid))
    win = np.clip(float(meta["payback_win"]) / np.clip(share, 1e-6, None), 1.0, None)
    mid = np.clip(np.exp(a2 + b2 * place_raw), float(meta.get("place_floor", 1.05)), None)
    h = float(meta.get("place_half_range", DEFAULT_PLACE_HALF_RANGE))
    out = pd.DataFrame({"race_id": rid, "horse_id": X["horse_id"].astype(str).to_numpy(), "win_odds": win, "place_mid": mid})
    out["place_min"] = np.maximum(out["place_mid"] * (1 - h), 1.0)
    out["place_max"] = out["place_mid"] * (1 + h)
    return out

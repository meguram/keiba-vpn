"""想定単勝・想定複勝（決定ログ §77 のモデル）の推論。

出走馬の当日特徴量（V3Serving v6 の ``features()`` の戻り値）→ LightGBM（対数シェア・対数複勝）→ 較正の直線 → レース内で正規化 → 想定オッズ。
保存は最終オッズ予測のキャッシュ（FinalOddsCache。カテゴリ final_odds_prediction）と同じ形で行い、entries の source を ``sec77`` にする。
これにより predicted_single_odds（odds_source）を通るすべての表示（レース詳細・race-values・推奨馬券・予測成績）が同じ想定オッズを使う。
モデルが無い・特徴量が足りないときは何もしない（現行の想定オッズに自動で戻る）。KEIBA_ODDS_SEC77=0 で無効。

学習と保存: ``python -m src.scripts.data.odds_sec77_fit fit``（data/local/modeling/odds_pred/serving/）。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(os.environ.get("ODDS_SEC77_DIR") or ROOT / "data/local/modeling/odds_pred/serving")
SOURCE = "sec77"
KEYS = ["race_id", "horse_id"]
DEFAULT_PLACE_HALF_RANGE = 0.12   # 想定複勝の中央から min/max を作る幅（実データの (max-min)/(2·mid) の中央値で上書きされる）
_cache: dict[str, Any] = {}


def enabled() -> bool:
    return (os.environ.get("KEIBA_ODDS_SEC77") or "1").strip().lower() not in ("0", "false", "no", "off")


def available() -> bool:
    return enabled() and (MODEL_DIR / "meta.json").exists() and (MODEL_DIR / "share_model.txt").exists() and (MODEL_DIR / "place_model.txt").exists()


def reload() -> None:
    _cache.clear()


def _load() -> dict[str, Any]:
    key = str(MODEL_DIR)
    mt = (MODEL_DIR / "meta.json").stat().st_mtime
    ent = _cache.get(key)
    if ent and ent[0] == mt:
        return ent[1]
    import lightgbm as lgb

    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    pack = {"meta": meta, "share": lgb.Booster(model_file=str(MODEL_DIR / "share_model.txt")), "place": lgb.Booster(model_file=str(MODEL_DIR / "place_model.txt"))}
    _cache[key] = (mt, pack)
    return pack


def feature_columns() -> list[str]:
    return list(_load()["meta"]["cols"])


def predict_frame(X: pd.DataFrame) -> pd.DataFrame:
    """X: race_id・horse_id + 特徴量。戻り値: race_id・horse_id・win_odds・place_mid・place_min・place_max（odds_pred_eval の win_B / place_B と同じ変換）。"""
    pack = _load()
    meta = pack["meta"]
    cols = meta["cols"]
    miss = [c for c in cols if c not in X.columns]
    if miss:
        raise KeyError(f"sec77 の特徴量が足りない列: {len(miss)} 列 {miss[:8]}")
    A = X[cols].astype("float32")
    share_raw = pack["share"].predict(A)
    place_raw = pack["place"].predict(A)
    a1, b1 = meta["win_line"]
    a2, b2 = meta["place_line"]
    rid = X["race_id"].astype(str).to_numpy()
    logv = a1 + b1 * share_raw
    v = np.exp(logv - logv.max())
    s = pd.Series(v).groupby(rid).transform("sum").to_numpy()
    share = v / s
    win = np.clip(float(meta["payback_win"]) / np.clip(share, 1e-6, None), 1.0, None)
    mid = np.clip(np.exp(a2 + b2 * place_raw), float(meta.get("place_floor", 1.05)), None)
    h = float(meta.get("place_half_range", DEFAULT_PLACE_HALF_RANGE))
    out = pd.DataFrame({"race_id": rid, "horse_id": X["horse_id"].astype(str).to_numpy(), "win_odds": win, "place_mid": mid})
    out["place_min"] = np.maximum(out["place_mid"] * (1 - h), 1.0)
    out["place_max"] = out["place_mid"] * (1 + h)
    return out


def payload_for_race(rows: pd.DataFrame, horse_numbers: dict[str, int], race_name: str, elapsed: float = 0.0) -> dict | None:
    """1 レース分の予測行 → 最終オッズ予測のキャッシュと同じ形の応答。馬番が分からない馬がいれば None。"""
    entries = []
    for r in rows.itertuples(index=False):
        hn = horse_numbers.get(str(r.horse_id))
        if not hn:
            return None
        entries.append({"horse_number": int(hn), "predicted_win_odds": round(float(r.win_odds), 2),
                        "predicted_place_odds_min": round(float(r.place_min), 2), "predicted_place_odds_max": round(float(r.place_max), 2), "source": SOURCE})
    if not entries:
        return None
    entries.sort(key=lambda e: e["horse_number"])
    return {"race_id": str(rows["race_id"].iloc[0]), "race_name": race_name or "-", "elapsed_sec": round(float(elapsed), 2), "entries": entries}


def save_for_race(storage, payload: dict) -> bool:
    from src.pipeline.inference.final_odds_service import FinalOddsCache

    return bool(FinalOddsCache.save_cached(storage, payload["race_id"], payload, source=SOURCE))


def is_sec77(single: dict | None) -> bool:
    """predicted_single_odds の結果が sec77 由来か。"""
    return bool(single) and (single.get("odds_model") == SOURCE)

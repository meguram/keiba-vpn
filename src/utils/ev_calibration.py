"""期待値（EV）の較正。モデルの EV は『モデル確率が市場より尖っている』『EV の高い買い目を選ぶ』ぶん、実現回収率より高く出る
（検証: docs/html/modeling/chapters/ev_calibration_audit.html、ev_calibration_adopt.html）。そこで、表示する EV を過去の実績
（確定オッズ・確定払戻）に合わせて較正し、『その EV で買い続けたときの実現回収率に収束する』値にする。

方式は券種（kind）ごとに config/ev_calibration.json に持つ（当てはめは src/scripts/data/calibrate_ev.py）:
  isotonic … EV → 実現回収率の単調な折れ線（結び目の外は打ち切り）
  shrink   … base + k × (EV − base)
  linear   … a + b × EV
  const    … EV の高低では見分けられない券種の、実現回収率の平均
kind は 2 系統:
  馬ごとの EV（画面の勝期待値・複期待値と買い推奨度）: win / top2 / place
  推奨ポートフォリオの買い目の EV: tansho / fukusho / umaren / wide（設定に無い券種は、払戻率＝市場並みの保守値）
config の enabled が false のときは何も変えない（従来の EV のまま）。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ev_calibration.json"
DEFAULT_BASE = {"win": 0.80, "top2": 0.775, "place": 0.80, "tansho": 0.80, "fukusho": 0.80, "umaren": 0.775, "wide": 0.775, "umatan": 0.75,
                "sanrenpuku": 0.75, "trio": 0.75, "sanrentan": 0.725, "tierce": 0.725, "portfolio": 0.80}
CAND_KINDS = {"tansho", "fukusho", "umaren", "wide"}
LEGACY_TIERS = {"strong": 1.2, "recommend": 1.05, "watch": 0.9}
LEGACY_LEVELS: tuple[tuple[str, float], ...] = (("hot", 1.5), ("good", 1.25), ("ok", 1.05))
LABEL = "期待回収率（較正後）"


@lru_cache(maxsize=1)
def load_config() -> dict:
    try:
        d = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


from src.utils import cache_epoch as _cache_epoch  # noqa: E402

_cache_epoch.register("ev_calibration.config", lambda: (load_config.cache_clear(), load_config_pc.cache_clear()))

PC_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ev_calibration_pc.json"


@lru_cache(maxsize=1)
def load_config_pc() -> dict:
    """確率を市場補正（KEIBA_PROB_CALIBRATION=1）したレース用の EV 較正。補正後の確率から作った EV に当てはめ直したもの（tp-70）。無ければ空。"""
    try:
        d = json.loads(PC_CONFIG_PATH.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def active_config(prob_calibrated: bool = False) -> dict:
    """確率を市場補正したレースは PC 版（あれば）、それ以外は従来の設定。二重に補正しないための切り替え。"""
    if prob_calibrated:
        pc = load_config_pc()
        if pc:
            return pc
    return load_config()


def is_enabled(config: dict | None = None) -> bool:
    cfg = load_config() if config is None else config
    return bool(cfg.get("enabled"))


def apply_spec(spec: dict, ev):
    """較正の仕様 1 つを EV（スカラーまたは配列）に適用する。"""
    scalar = np.isscalar(ev)
    v = np.atleast_1d(np.asarray(ev, float))
    m = spec["method"]
    if m == "market":
        out = np.full_like(v, float(spec["base"]))
    elif m == "shrink":
        out = spec["base"] + spec["k"] * (v - spec["base"])
    elif m == "linear":
        out = spec["a"] + spec["b"] * v
    elif m == "isotonic":
        out = np.interp(v, spec["x"], spec["y"])
    elif m == "const":
        out = np.full_like(v, float(spec["value"]))
    else:
        raise ValueError(m)
    return float(out[0]) if scalar else out


def _spec_for(kind: str, cfg: dict) -> dict | None:
    kinds = cfg.get("kinds") or {}
    spec = kinds.get(kind) or kinds.get(f"cand_{kind}")
    if spec:
        return spec
    # 旧形式（k / base だけ）
    k = (cfg.get("k") or {}).get(kind)
    if k is not None:
        return {"method": "shrink", "base": (cfg.get("base") or {}).get(kind, DEFAULT_BASE.get(kind, 0.8)), "k": float(k)}
    return None


def calibrated_ev(ev: float | None, kind: str, config: dict | None = None) -> float | None:
    """較正した EV。無効のとき、EV が None のときはそのまま返す。設定に無い買い目の券種（馬単・3連複など）は払戻率（市場並み）に置く。"""
    if ev is None:
        return None
    cfg = load_config() if config is None else config
    if not cfg.get("enabled"):
        return ev
    spec = _spec_for(kind, cfg)
    if spec is None:
        if kind in CAND_KINDS or kind not in ("win", "top2", "place", "portfolio"):
            return DEFAULT_BASE.get(kind, 0.8)         # 検証していない券種: 市場並みの保守値
        return ev
    cap = float((cfg.get("cap") or {}).get("ev", 1.3))
    return float(min(max(apply_spec(spec, float(ev)), 0.0), cap))


def calibrate_portfolio(candidates: Iterable[dict], config: dict | None = None) -> dict | None:
    """推奨ポートフォリオの買い目（bet_type, ev, bet_amount）から、較正後の買い目ごとの EV と、購入額加重の期待回収率を出す。
    無効のときは None。"""
    cfg = load_config() if config is None else config
    if not cfg.get("enabled"):
        return None
    cs = list(candidates)
    evs = [calibrated_ev(float(c["ev"]), str(c["bet_type"]), cfg) for c in cs]
    stake = sum(float(c.get("bet_amount") or 0) for c in cs)
    if not cs or stake <= 0:
        return {"evs": evs, "expected_roi": None, "expected_return": 0.0}
    ret = sum(e * float(c.get("bet_amount") or 0) for e, c in zip(evs, cs))
    cap = float((cfg.get("cap") or {}).get("roi", 1.15))
    roi = min(ret / stake, cap)
    return {"evs": evs, "expected_roi": round(roi, 3), "expected_return": round(roi * stake, 0), "capped": ret / stake > cap}


def tier_thresholds(config: dict | None = None) -> dict:
    """買い推奨度の閾値（較正後の EV の最大に対する）。無効のときは従来の 1.2 / 1.05 / 0.9。"""
    cfg = load_config() if config is None else config
    if not cfg.get("enabled"):
        return dict(LEGACY_TIERS)
    return {**LEGACY_TIERS, **(cfg.get("tiers") or {})}


def value_levels(config: dict | None = None) -> tuple[tuple[str, float], ...]:
    """レース価値の段階（hot / good / ok）の下限（較正後の期待回収率）。無効のときは従来の 1.5 / 1.25 / 1.05。"""
    cfg = load_config() if config is None else config
    if not cfg.get("enabled"):
        return LEGACY_LEVELS
    lv = cfg.get("levels") or {}
    return tuple((n, float(lv.get(n, f))) for n, f in LEGACY_LEVELS)

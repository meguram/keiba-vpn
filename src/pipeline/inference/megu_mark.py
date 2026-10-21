"""めぐ指数の印（好走に値する馬）。過去走のめぐ指数の集約と、マーキング規則。

評価（どの集約が最も効くか）は tp-66（docs/html/modeling/chapters/megu_marking.html、src/scripts/data/tp66_*.py）。
指数は絶対指数 megu_index_v3a_abs_robust（status ok・低信頼を除く）。今走の日付より前の走だけを使う。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

DECAY = 0.65
LOOKBACK = 8
NS = (1, 2, 3, 5, 8)
DIST_SCALE = 600.0
SURF_MISMATCH = 0.5


def dist_tol(dist_today: float | None) -> float:
    return 400.0 if dist_today and dist_today >= 2400 else 200.0


def winsor_mean(v: np.ndarray, k: float = 1.5) -> float:
    """中央値 ± k×MAD の外を丸めた平均（3 走未満は単純な平均）。"""
    if len(v) < 3:
        return float(np.mean(v))
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    if mad <= 0:
        return med
    return float(np.mean(np.clip(v, med - k * mad, med + k * mad)))


FLOOR_BELOW_WINNER = 40.0     # 大差負けの下限: 同じレースの勝ち馬の指数 − この点（tp-65: めぐ・netkeiba とも +0.006 改善、−20 は悪化、−30〜60 は同程度）


def floored(value: float, winner_value: float | None, below: float = FLOOR_BELOW_WINNER) -> float:
    return value if winner_value is None or winner_value != winner_value else max(value, winner_value - below)


def weighted_huber(v: np.ndarray, base_w: np.ndarray, k: float = 1.345, iters: int = 6) -> float:
    """直近ほど重い重み(base_w)つきの Huber 位置推定。外れ値の走は重みを下げる（尺度は MAD）。"""
    mu = float(np.sum(v * base_w) / base_w.sum())
    if len(v) < 3:
        return mu
    scale = max(float(np.median(np.abs(v - np.median(v)))) * 1.4826, 3.0)
    for _ in range(iters):
        r = np.abs(v - mu)
        hw = np.where(r <= k * scale, 1.0, k * scale / np.maximum(r, 1e-9))
        w = base_w * hw
        mu = float(np.sum(v * w) / w.sum())
    return mu


def features_for(hist: list[tuple], surf: int, dist: float | None, line: float | None) -> dict[str, float]:
    """hist: 古い順の (指数, 面[0 芝/1 ダ/-1 不明], 距離, 下限つき指数[省略可])。直近 LOOKBACK 走。今走の面・距離・突破ライン(line)を与える。"""
    h = hist[-LOOKBACK:][::-1]               # 新しい順
    v = np.array([x[0] for x in h], dtype=float)
    vf = np.array([(x[3] if len(x) > 3 and x[3] is not None else x[0]) for x in h], dtype=float)
    s = np.array([x[1] for x in h], dtype=int)
    d = np.array([x[2] if x[2] is not None else np.nan for x in h], dtype=float)
    out: dict[str, float] = {}
    out["last1"] = float(v[0])
    for n in NS[1:]:
        w = v[:n]
        out[f"mean{n}"] = float(w.mean())
        out[f"max{n}"] = float(w.max())
        out[f"med{n}"] = float(np.median(w))
    out["hub5"] = winsor_mean(v[:5])
    wts = DECAY ** np.arange(min(5, len(v)))
    out["ew5"] = float((v[:5] * wts).sum() / wts.sum())
    out["ew5_fl"] = float((vf[:5] * wts).sum() / wts.sum())              # 大差負けに下限
    out["mean5_fl"] = float(vf[:5].mean())
    out["ewhuber5"] = weighted_huber(v[:5], wts)
    out["ewhuber5_fl"] = weighted_huber(vf[:5], wts)
    top = np.sort(v)[::-1]
    out["top2of8"] = float(top[:2].mean())
    out["top3of8"] = float(top[:3].mean())
    if len(v) >= 4:
        out["trend"] = float(v[0] - v[1:4].mean())
    # 同条件走（同じ面・距離が近い）
    if surf >= 0 and dist:
        tol = dist_tol(dist)
        m = (s == surf) & (np.abs(d - dist) <= tol)
        out["cond_n"] = float(m.sum())
        if m.any():
            out["cond_mean"] = float(v[m].mean())
            out["cond_max"] = float(v[m].max())
            out["cond_mean3"] = float(v[m][:3].mean())
            out["cond_max3"] = float(v[m][:3].max())
        # 条件適合の重み付き平均（直近 5 走）
        k = min(5, len(v))
        wf = DECAY ** np.arange(k)
        wf = wf * np.where(s[:k] == surf, 1.0, SURF_MISMATCH) * np.exp(-np.nan_to_num(np.abs(d[:k] - dist), nan=DIST_SCALE) / DIST_SCALE)
        out["fitw5"] = float((v[:k] * wf).sum() / wf.sum())
    if line is not None:
        out["line"] = float(line)
        out["max5_margin"] = out["max5"] - line
        out["mean5_margin"] = out["mean5"] - line
        out["ew5_margin"] = out["ew5"] - line
        out["ew5_fl_margin"] = out["ew5_fl"] - line
        out["ewhuber5_fl_margin"] = out["ewhuber5_fl"] - line
        out["n_above5"] = float((v[:5] >= line).sum())
        out["frac_above5"] = float((v[:5] >= line).mean())
        if "cond_max" in out:
            out["cond_max_margin"] = out["cond_max"] - line
            out["cond_mean_margin"] = out["cond_mean"] - line
        out["last1_margin"] = out["last1"] - line
        out["top2of8_margin"] = out["top2of8"] - line
    return out


# ── めぐ印（API 用）──────────────────────────────────────────────────

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "megu_mark.json"
_cfg_cache: dict[str, Any] = {}


def load_config(path: Path | None = None) -> dict | None:
    p = path or CONFIG_PATH
    try:
        mt = p.stat().st_mtime
    except OSError:
        return None
    hit = _cfg_cache.get(str(p))
    if hit and hit[0] == mt:
        return hit[1]
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    _cfg_cache[str(p)] = (mt, cfg)
    return cfg


def v3b_value(runs: list[tuple], race_date: str | None) -> tuple[float | None, int]:
    """runs: expected_megu の走の記録 (YYYYMMDD, 指数, 低信頼, ...)。今走の日付より前の直近 5 走を v3b で集約する。
    大差負けの下限（勝ち馬−40点）は、絶対指数（robust）ではほとんど効かない（0.007%）ので API では使わない。"""
    from src.pipeline.megu_index import v3b

    rd = "".join(ch for ch in str(race_date or "") if ch.isdigit())[:8]
    use = [r for r in runs if r[0] and r[1] == r[1] and not (rd and str(r[0]) >= rd)]
    use.sort(key=lambda r: str(r[0]), reverse=True)
    use = use[: v3b.N_RUNS]
    if not use:
        return None, 0
    val = v3b.aggregate([float(r[1]) for r in use], low_conf=[bool(r[2]) for r in use])
    return val, len(use)


def compute_marks(entries: list[dict], line: float | None, cfg: dict | None = None) -> dict[Any, dict]:
    """entries: [{"key": 馬を区別する値, "horse_number": 馬番, "runs": 走の記録, "race_date": YYYYMMDD}]。
    返り値: key → めぐ印の辞書（key/label は印が付くときだけ。付かないときも value/rank/unavailable などの根拠は返す）。
    規則（tp-66）: v3b の値がレース内で 1 位かつ突破ライン以上=めぐ◎、2〜3 位かつ突破ライン以上=めぐ○。
    突破ラインが無いクラス、出走が min_runners 頭未満のレース、過去走が無い馬は印を付けない（理由を unavailable に書く）。"""
    cfg = cfg if cfg is not None else load_config()
    if not cfg:
        return {}
    scored = []
    for e in entries:
        val, n = v3b_value(e.get("runs") or [], e.get("race_date"))
        scored.append((e, val, n))
    valid = [(e, v, n) for e, v, n in scored if v is not None]
    order = sorted(valid, key=lambda t: (-t[1], float(t[0].get("horse_number") or 0)))
    rank = {id(t[0]): i + 1 for i, t in enumerate(order)}
    vals = np.array([t[1] for t in valid], dtype=float)
    mu, sd = (float(vals.mean()), float(vals.std(ddof=1))) if len(vals) >= 3 else (None, None)
    race_ok = len(entries) >= int(cfg.get("min_runners", 6))
    out: dict[Any, dict] = {}
    for e, val, n in scored:
        d: dict[str, Any] = {"version": cfg.get("version", 1), "value": None if val is None else round(val, 1), "n_runs": n, "line": line,
                             "margin": None, "rank": None, "z": None, "key": None, "label": None, "unavailable": None, "reason": None}
        if val is None:
            d["unavailable"] = "過去走のめぐ指数がありません（初出走など）"
        else:
            d["rank"] = rank[id(e)]
            d["z"] = None if not sd else round((val - mu) / sd, 2)
            d["margin"] = None if line is None else round(val - line, 1)
            if not race_ok:
                d["unavailable"] = f"出走が{cfg.get('min_runners', 6)}頭未満のレースは印を付けません"
            elif cfg.get("require_line", True) and line is None:
                d["unavailable"] = "このクラスの突破ライン（基準）がないため印を付けません"
            elif d["margin"] is not None and d["margin"] < 0:
                d["reason"] = f"集約 {d['value']:.1f} は突破ライン {line:.1f} 未満（{d['margin']:+.1f}）"
            else:
                for t in cfg.get("tiers", []):
                    if int(t["rank_min"]) <= d["rank"] <= int(t["rank_max"]):
                        d["key"], d["label"] = t["key"], t["label"]
                        d["reason"] = f"レース内{d['rank']}位・集約 {d['value']:.1f}" + (f"（突破ライン {line:.1f} 比 {d['margin']:+.1f}）" if line is not None else "")
                        break
                if d["key"] is None:
                    top_rank = max((int(t["rank_max"]) for t in cfg.get("tiers", [])), default=0)
                    d["reason"] = f"レース内{d['rank']}位（印は上位{top_rank}位まで）・集約 {d['value']:.1f}"
        out[e.get("key")] = d
    return out

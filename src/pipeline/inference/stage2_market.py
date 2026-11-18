"""第 2 段のモデル（市場とのずれを学習した xgb、noise_fit2025）の当日の推論。

入力は、第 1 段（新しい候補の束）の推論の経路が作る特徴量の行列 X6（V3Serving.features の出力。xgb の 1,193 列）と、
T-60 の単勝オッズ。市場の勝率 q（1/オッズを出走馬で合計 1）の logit を base_margin（offset）にして、
win・top2・top3 の 3 seed の確率を平均し、レース内で合計 1・2・3 にそろえる。r = log(p_win / q)。

計算は学習・評価の道具 ``src/scripts/data/model_cycles_market.py`` の predict_full・q_from_odds と同じ
（列の選びと並びは meta.json の feature_cols、型は cat_levels。学習の表は読まない）。

事象と根拠: 第 2 段は「市場がすでに織り込んだ分」を offset で固定し、モデルが市場から上下にずれる分だけを学ぶ。
r が大きい馬は、市場より強いとモデルが見る馬（model-improve-cycles の第 2 段: T-60 の雑音つき offset を採用、
2023〜24 年の確かめで ◎ の単勝 103.6%）。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ODDS_SUM_RANGE = (1.0, 1.6)        # Σ1/単勝オッズ（払戻率の逆数）の正常の幅。外はオッズの取り込みの崩れとみなして使わない
TARGETS = (("win", 1.0), ("top2", 2.0), ("top3", 3.0))


class OddsBroken(ValueError):
    """T-60 のオッズが第 2 段に使えない（Σ1/オッズが正常の幅の外など）。"""


def model_dir() -> Path:
    """第 2 段のモデルの置き場（KEIBA_STAGE2_MODEL_DIR。影のランナーの設定 config/candidate_runtime.json が当てる。既定は無い）。"""
    v = os.environ.get("KEIBA_STAGE2_MODEL_DIR")
    if not v:
        raise RuntimeError("KEIBA_STAGE2_MODEL_DIR が無い（第 2 段のモデルの置き場）")
    return Path(v)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def market_q(odds) -> np.ndarray:
    """1 レースの単勝オッズ → 市場の勝率（合計 1）。オッズの無い馬・0 以下は NaN（model_cycles_market.q_from_odds の beta=1 と同じ）。"""
    o = np.asarray(odds, dtype=float)
    inv = np.where(np.isfinite(o) & (o > 0), 1.0 / np.where(o > 0, o, 1.0), np.nan)
    s = np.nansum(inv)
    return inv / s if s > 0 else inv


def check_odds(odds) -> float:
    """Σ1/オッズを返す。正常の幅の外・有効なオッズが 3 頭未満なら OddsBroken。"""
    o = np.asarray(odds, dtype=float)
    ok = np.isfinite(o) & (o > 0)
    if ok.sum() < 3:
        raise OddsBroken(f"有効な単勝オッズが {int(ok.sum())} 頭")
    s = float(np.sum(1.0 / o[ok]))
    if not (ODDS_SUM_RANGE[0] <= s <= ODDS_SUM_RANGE[1]):
        raise OddsBroken(f"Σ1/オッズ={s:.3f} が {ODDS_SUM_RANGE} の外")
    return s


class Stage2Model:
    """保存した 9 本（win/top2/top3 × 3 seed）を読み込んで持つ。1 プロセスで 1 回だけ読む。"""

    def __init__(self, path: Path | str | None = None):
        import xgboost as xgb

        self.path = Path(path or model_dir())
        self.meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        self.feature_cols: list[str] = list(self.meta["feature_cols"])
        self.cat_levels: dict[str, list] = dict(self.meta.get("cat_levels") or {})
        self.offset = (self.meta.get("spec") or {}).get("offset", "none")
        self.models: dict[str, list] = {}
        for t, _ in TARGETS:
            ms = []
            for s in range(int(self.meta["seeds"])):
                m = xgb.XGBClassifier(enable_categorical=True)
                m.load_model(str(self.path / f"{t}_s{s}.json"))
                ms.append(m)
            self.models[t] = ms
        logger.info("第 2 段のモデルを読んだ: %s（%d 列・%d seed・offset=%s）", self.path, len(self.feature_cols), self.meta["seeds"], self.offset)

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or self.path.name)

    def cast(self, X: pd.DataFrame) -> pd.DataFrame:
        """列の選びと並び（feature_cols）と型（カテゴリは cat_levels の水準、欠けは「(なし)」があればそれ）。足りない列があれば止める。"""
        miss = [c for c in self.feature_cols if c not in X.columns]
        if miss:
            raise KeyError(f"第 2 段の入力に足りない列: {len(miss)} 列 {miss[:12]}")
        out = X[self.feature_cols].copy()
        for c in self.feature_cols:
            if c in self.cat_levels:
                cats = list(self.cat_levels[c])
                s = out[c].astype(str).where(out[c].notna())
                if "(なし)" in cats:
                    s = s.fillna("(なし)")
                out[c] = pd.Categorical(s, categories=cats)
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def predict_race(self, X: pd.DataFrame, odds) -> pd.DataFrame:
        """1 レースの出走馬（取消を除く）の X と、同じ並びの T-60 の単勝オッズ → p_win・p_top2・p_top3・q・r。

        q はオッズのある馬だけで合計 1（無い馬は NaN、r も NaN）。offset は、q の無い馬を 0.5/頭数で埋めて合計 1 に
        直し 1e-5〜1 に切った値（学習の q の埋め方と同じ）。"""
        from src.pipeline.models import zoo

        n = len(X)
        if n == 0:
            raise ValueError("出走馬が 0 頭")
        o = np.asarray(odds, dtype=float)
        if len(o) != n:
            raise ValueError(f"X {n} 行とオッズ {len(o)} 頭の数が違う")
        check_odds(o)
        q = market_q(o)
        qm = np.where(np.isfinite(q) & (q > 0), q, 0.5 / n)
        qm = np.clip(qm / qm.sum(), 1e-5, 1.0)
        race = np.zeros(n, dtype=int)
        q2, q3 = zoo.harville_topk(race, qm, 1.0, 1.0)
        q2, q3 = 2.0 * q2 / q2.sum(), 3.0 * q3 / q3.sum()
        margins = {"win": logit(qm), "top2": logit(np.clip(q2, 1e-5, 0.999)), "top3": logit(np.clip(q3, 1e-5, 0.999))}
        Xc = self.cast(X)
        out = {}
        for t, tot in TARGETS:
            acc = np.zeros(n)
            for m in self.models[t]:
                acc += m.predict_proba(Xc, base_margin=None if self.offset == "none" else margins[t])[:, 1]
            acc /= len(self.models[t])
            out[f"p_{t}"] = tot * acc / acc.sum()
        res = pd.DataFrame({"p_win": out["p_win"], "p_top2": out["p_top2"], "p_top3": out["p_top3"], "q": q})
        with np.errstate(divide="ignore", invalid="ignore"):
            res["r"] = np.log(res["p_win"].to_numpy() / q)
        res["odds"] = o
        return res

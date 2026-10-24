"""tp-64: 展開利の特徴量（pb_*）が V5 の OOF 確率の残差を説明できるか（再学習なしの簡易評価。tp-58 の残差ブースティングと同じ作法）。

python -m src.scripts.data.tp64_resid   # 出力: $TP64_OUT/resid.json
V5 相当の OOF（valid 2022〜2024 年）のロジットを init_score に、浅い LightGBM で pb_* から補正を学び、1 年を外して（LOYO）
外した年の LogLoss の変化を測る（負 = 改善）。対照: 同じ形のノイズ列、直近の走の展開利を入れ替えた列。レース単位のブートストラップで区間。
full 再学習の増分（tp64_eval）の代わりではなく、列に情報があるかの先行指標。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.scripts.data import tp64_eval as ev

ROOT = Path(__file__).resolve().parents[3]
T61 = Path(os.environ.get("TP64_T61") or ROOT / "data/local/modeling/tp61")
OUT = Path(os.environ.get("TP64_OUT") or "/home/jovyan/work/tp64_out")
OOF = Path(os.environ.get("TP64_OOF") or ROOT / "data/local/modeling/training_pipeline_v4/v6/eval/oof_d_CY_BEST.parquet")
THREADS = int(os.environ.get("TP64_THREADS", "2"))
B = 400
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_data_in_leaf=200, lambda_l2=20.0, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, num_threads=THREADS, verbose=-1, seed=0)
ROUNDS = 150
TARGETS = {"win": ("p_win", "is_win"), "top2": ("p_top2", "is_top2"), "top3": ("p_top3", "is_top3")}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load() -> pd.DataFrame:
    o = pd.read_parquet(OOF)
    f = pd.read_parquet(ev.FEATS)
    h = pd.read_parquet(T61 / "horse.parquet")[["race_id", "horse_id", "is_win", "is_top2", "is_top3"]]
    for d in (o, f, h):
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    return o.merge(f, on=["race_id", "horse_id"], how="left").merge(h, on=["race_id", "horse_id"], how="left")


def loyo(d: pd.DataFrame, cols: list[str], tgt: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """戻り値: (行ごとの LogLoss の差 = 補正後 − 基準, race_id)。"""
    pcol, ycol = TARGETS[tgt]
    y = d[ycol].to_numpy(float)
    base = _logit(d[pcol].to_numpy(float))
    X = d[cols].to_numpy(np.float32)
    new = base.copy()
    for yr in sorted(d["year"].unique()):
        te = (d["year"] == yr).to_numpy()
        tr = ~te
        params = PARAMS | {"seed": seed}
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr], init_score=base[tr]), num_boost_round=ROUNDS)
        new[te] = base[te] + m.predict(X[te], raw_score=True)

    def ll(z):
        return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))

    return ll(new) - ll(base), d["race_id"].to_numpy()


def summarize(diff: np.ndarray, races: np.ndarray, years: np.ndarray) -> dict:
    ids, inv = np.unique(races, return_inverse=True)
    s = np.zeros(len(ids))
    n = np.zeros(len(ids))
    np.add.at(s, inv, diff)
    np.add.at(n, inv, 1.0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), size=(B, len(ids)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    out = {"delta": float(s.sum() / n.sum()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
    out["by_year"] = {str(y): float(diff[years == y].mean()) for y in np.unique(years)}
    return out


def main() -> int:
    d = load()
    d = d.dropna(subset=["is_win", "is_top2", "is_top3"]).reset_index(drop=True)
    rng = np.random.default_rng(1)
    for i in range(len(ev.HIST + ev.GAP)):
        d[f"noise_{i}"] = rng.normal(size=len(d)).astype("float32")
    sets = {
        "PB_ALL": ev.HIST + ev.GAP,
        "PB_HIST": ev.HIST,
        "PB_GAP": ev.GAP,
        "BEN_ONLY": ["pb_ben_1", "pb_ben_m3", "pb_ben_m5", "pb_ben_max3", "pb_lucky_good", "pb_hard_good", "pb_lucky_bad", "pb_unlucky_bad"],
        "NOISE(対照)": [f"noise_{i}" for i in range(len(ev.HIST + ev.GAP))],
    }
    res: dict = {"rows": int(len(d))}
    years = d["year"].to_numpy()
    for nm, cols in sets.items():
        res[nm] = {}
        for tgt in TARGETS:
            diff, races = loyo(d, cols, tgt)
            res[nm][tgt] = summarize(diff, races, years)
            r = res[nm][tgt]
            print(f"{nm:12s} {tgt:5s} Δ{r['delta']:+.5f} [{r['lo']:+.5f}, {r['hi']:+.5f}]  by_year {r['by_year']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "resid.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

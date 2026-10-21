"""tp-64: 仮説『展開に恵まれて好走した馬は、次走で展開が苦しくなりそうなときに好走率が下がる』の検証。

python -m src.scripts.data.tp64_hyp   # 出力: $TP64_OUT/hyp.json
入力: tp64_build の pb_features.parquet / tp61 の horse.parquet / V5 相当の OOF（v6/eval/oof_d_CY_BEST.parquet。valid 2022〜2024）
評価は『V5 の OOF 確率に対する実績の残差』（残差 = 実績 − 予測確率。V5 が既に織り込んだ分を引く）を、レース単位のブートストラップで。
グループのしきい値は 2022〜2023 年の分位点で固定し、2024 年は同じしきい値での確認に使う。2025・2026 年は使わない（OOF がない／テストセット②）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
T61 = Path(os.environ.get("TP64_T61") or ROOT / "data/local/modeling/tp61")
OUT = Path(os.environ.get("TP64_OUT") or ROOT / "data/local/modeling/tp64")
OOF = Path(os.environ.get("TP64_OOF") or ROOT / "data/local/modeling/training_pipeline_v4/v6/eval/oof_d_CY_BEST.parquet")
B = 1000
GOOD = 0.30
LUCKY = 0.03


def load() -> pd.DataFrame:
    f = pd.read_parquet(OUT / "pb_features.parquet")
    h = pd.read_parquet(T61 / "horse.parquet")[["race_id", "horse_id", "date", "is_win", "is_top3", "fin_pct", "field_size"]]
    for d in (f, h):
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    h = h.sort_values(["horse_id", "date", "race_id"], kind="stable")
    h["prev_fin"] = h.groupby("horse_id", sort=False)["fin_pct"].shift(1)
    o = pd.read_parquet(OOF)
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    d = o.merge(f, on=["race_id", "horse_id"], how="left").merge(h.drop(columns=["date"]), on=["race_id", "horse_id"], how="left")
    d["r3"] = d["is_top3"] - d["p_top3"]
    d["rw"] = d["is_win"] - d["p_win"]
    return d


class Agg:
    """レース単位の合計（残差・件数）をグループごとに持ち、レースを復元抽出して平均残差とその差の区間を出す。"""

    def __init__(self, d: pd.DataFrame, groups: dict[str, np.ndarray], cols=("r3", "rw")):
        self.races = d["race_id"].to_numpy()
        self.ids, inv = np.unique(self.races, return_inverse=True)
        self.cols = cols
        self.names = list(groups)
        R, G = len(self.ids), len(groups)
        self.n = np.zeros((R, G))
        self.s = {c: np.zeros((R, G)) for c in cols}
        for gi, nm in enumerate(self.names):
            m = groups[nm]
            np.add.at(self.n[:, gi], inv[m], 1.0)
            for c in cols:
                np.add.at(self.s[c][:, gi], inv[m], d[c].to_numpy()[m])
        rng = np.random.default_rng(0)
        self.idx = rng.integers(0, R, size=(B, R))

    def mean(self, c: str, g: str) -> dict:
        gi = self.names.index(g)
        n, s = self.n[:, gi], self.s[c][:, gi]
        bs = s[self.idx].sum(1) / np.maximum(n[self.idx].sum(1), 1e-9)
        return {"mean": float(s.sum() / max(n.sum(), 1e-9)), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                "n": int(n.sum()), "races": int((n > 0).sum())}

    def diff(self, c: str, g1: str, g0: str) -> dict:
        i1, i0 = self.names.index(g1), self.names.index(g0)
        def m(i):
            n, s = self.n[:, i], self.s[c][:, i]
            return s[self.idx].sum(1) / np.maximum(n[self.idx].sum(1), 1e-9), s.sum() / max(n.sum(), 1e-9)
        b1, p1 = m(i1)
        b0, p0 = m(i0)
        bs = b1 - b0
        return {"diff": float(p1 - p0), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}

    def did(self, c: str, a: str, b: str, cc: str, dd: str) -> dict:
        """(a − b) − (cc − dd) の差の差。"""
        def m(g):
            i = self.names.index(g)
            n, s = self.n[:, i], self.s[c][:, i]
            return s[self.idx].sum(1) / np.maximum(n[self.idx].sum(1), 1e-9), s.sum() / max(n.sum(), 1e-9)
        (ba, pa), (bb, pb_), (bc, pc), (bd, pd_) = m(a), m(b), m(cc), m(dd)
        bs = (ba - bb) - (bc - bd)
        return {"diff": float((pa - pb_) - (pc - pd_)), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def table(d: pd.DataFrame, lo_exp: float, hi_exp: float, label: str) -> dict:
    """前走が好走（fin_pct ≤ 0.30）の馬を、前走の展開利（恵まれ/不利）× 今走の期待展開利（高/低）の 2×2 に分ける。"""
    g = d[(d["prev_fin"] <= GOOD) & d["pb_ben_1"].notna() & d["pb_exp_today"].notna()].copy()
    lucky, hard = g["pb_ben_1"] >= LUCKY, g["pb_ben_1"] <= -LUCKY
    ehi, elo = g["pb_exp_today"] >= hi_exp, g["pb_exp_today"] <= lo_exp
    groups = {"lucky_exp_low": (lucky & elo).to_numpy(), "lucky_exp_high": (lucky & ehi).to_numpy(),
              "hard_exp_low": (hard & elo).to_numpy(), "hard_exp_high": (hard & ehi).to_numpy(),
              "all_good_prev": np.ones(len(g), dtype=bool)}
    a = Agg(g.reset_index(drop=True), groups)
    out = {"label": label, "thresholds": {"exp_low": lo_exp, "exp_high": hi_exp}}
    for c in ("r3", "rw"):
        out[c] = {k: a.mean(c, k) for k in groups}
        out[c]["lucky: low−high"] = a.diff(c, "lucky_exp_low", "lucky_exp_high")
        out[c]["hard: high−low"] = a.diff(c, "hard_exp_high", "hard_exp_low")
        out[c]["did(lucky low−high) − (hard low−high)"] = a.did(c, "lucky_exp_low", "lucky_exp_high", "hard_exp_low", "hard_exp_high")
    return out


def slopes(d: pd.DataFrame, label: str) -> dict:
    """残差を pb_gap_1（前走の展開利 − 今走の期待展開利）の十分位ごとに見る。前走好走の馬と全馬。"""
    out = {"label": label}
    for nm, g in (("prev_good", d[(d["prev_fin"] <= GOOD)]), ("all", d)):
        g = g.dropna(subset=["pb_gap_1"]).reset_index(drop=True)
        q = pd.qcut(g["pb_gap_1"], 5, labels=False, duplicates="drop")
        groups = {f"Q{int(i) + 1}": (q == i).to_numpy() for i in sorted(q.dropna().unique())}
        a = Agg(g, groups)
        top = max(groups)
        out[nm] = {c: {k: a.mean(c, k) for k in groups} | {f"{top}−Q1": a.diff(c, top, "Q1")} for c in ("r3", "rw")}
        out[nm]["corr_gap_r3"] = float(np.corrcoef(g["pb_gap_1"], g["r3"])[0, 1])
    return out


def mains(d: pd.DataFrame, label: str) -> dict:
    """gap が効くのは、前走の展開利（V5 が過大評価？）か、今走の期待展開利（V5 が拾っていない？）か。5 分位の残差と、残差への回帰係数（レース単位ブートストラップ）。"""
    out = {"label": label}
    g = d.dropna(subset=["pb_ben_1", "pb_exp_today", "pb_ben_m3"]).reset_index(drop=True)
    for col in ("pb_ben_1", "pb_ben_m3", "pb_exp_today", "pb_exp_rel"):
        gg = g.dropna(subset=[col]).reset_index(drop=True)
        q = pd.qcut(gg[col], 5, labels=False, duplicates="drop")
        groups = {f"Q{int(i) + 1}": (q == i).to_numpy() for i in sorted(q.dropna().unique())}
        a = Agg(gg, groups)
        top = max(groups)
        out[col] = {c: {k: a.mean(c, k) for k in groups} | {f"{top}−Q1": a.diff(c, top, "Q1")} for c in ("r3", "rw")}
        # 5 分位ごとの残差を完全に補正できたと仮定したときの LogLoss の改善の目安（二項分布の 2 次近似: Σ w·r²/(2·p(1−p))）。
        # 上限の目安であり、実際の補正はこれより小さい。複勝・勝率それぞれ。
        gb = {}
        for rc, pc in (("r3", "p_top3"), ("rw", "p_win")):
            tot = 0.0
            for i in sorted(q.dropna().unique()):
                m = (q == i).to_numpy()
                r, pbar = float(gg.loc[m, rc].mean()), float(gg.loc[m, pc].mean())
                tot += m.mean() * r * r / (2 * pbar * (1 - pbar))
            gb[rc] = tot
        out[col]["gain_bound"] = gb
    # 多変量: r3 ~ ben_1 + ben_m3 + exp_today + exp_rel（標準化した係数）
    cols = ["pb_ben_1", "pb_ben_m3", "pb_exp_today", "pb_exp_rel"]
    z = g[cols].apply(lambda s: (s - s.mean()) / s.std())
    X = np.column_stack([np.ones(len(z)), z.to_numpy()])
    y = g["r3"].to_numpy()
    ids, inv = np.unique(g["race_id"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(1)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    bs = []
    order = np.argsort(inv)
    starts = np.searchsorted(inv[order], np.arange(len(ids)))
    ends = np.append(starts[1:], len(inv))
    for _ in range(300):
        pick = rng.integers(0, len(ids), len(ids))
        rows = np.concatenate([order[starts[i]:ends[i]] for i in pick])
        bs.append(np.linalg.lstsq(X[rows], y[rows], rcond=None)[0])
    bs = np.array(bs)
    out["ols_r3"] = {c: {"beta_per_sd": float(beta[i + 1]), "lo": float(np.percentile(bs[:, i + 1], 2.5)), "hi": float(np.percentile(bs[:, i + 1], 97.5))} for i, c in enumerate(cols)}
    return out


def main() -> int:
    d = load()
    d = d[d["year"].isin([2022, 2023, 2024])]
    ref = d[d["year"].isin([2022, 2023]) & d["pb_exp_today"].notna()]
    lo, hi = float(ref["pb_exp_today"].quantile(1 / 3)), float(ref["pb_exp_today"].quantile(2 / 3))
    res = {"n_rows": int(len(d)), "exp_terciles_2022_23": [lo, hi]}
    res["tab_2022_23"] = table(d[d["year"].isin([2022, 2023])], lo, hi, "2022-23")
    res["tab_2024"] = table(d[d["year"] == 2024], lo, hi, "2024（同じしきい値）")
    res["tab_all"] = table(d, lo, hi, "2022-24")
    res["gap"] = slopes(d, "2022-24")
    res["gap_2024"] = slopes(d[d["year"] == 2024], "2024")
    res["mains"] = mains(d, "2022-24")
    res["mains_2024"] = mains(d[d["year"] == 2024], "2024")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hyp.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    t = res["tab_all"]
    for c in ("r3", "rw"):
        print(c, {k: (round(v["mean"], 4), v["n"]) for k, v in t[c].items() if "mean" in v})
        for k in ("lucky: low−high", "hard: high−low", "did(lucky low−high) − (hard low−high)"):
            print(" ", k, {x: round(y, 4) for x, y in t[c][k].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

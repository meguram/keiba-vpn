"""tp-61: 展開シナリオごとの各馬の勝率・連対率・複勝率（scenario_forecast）を作り、実測の展開を知っていた場合（オラクル）と事前の確率で重み付けした場合を比べる。

python -m src.scripts.data.tp61_scen eval   # 学習 2022〜2023（選択 2024、確認 2025）→ 2026（テストセット②は最終確認のみ）
入力: tp36 の frame_all（最終モデルの勝率・連対率・複勝率 p_win / p_top2 / p_top3）、tp61 の horse / pre、tp53 の想定位置取り。
各シナリオ s=(ペース p, 隊列 q) の下で、 logit p_s = logit p_base + φ(想定位置取り) · (B_pace[p] + B_shape[q]) を最終モデルの出力に足し、レース内で合計が 1（連対 2・複勝 3）になるよう戻す。
オッズ・人気は使わない（評価の『実際の回収率』にだけ使う場合も、ここでは使わない）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.scripts.data import tp61_pre as pre

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/local/modeling/tp61"
FRAME = ROOT / "data/local/modeling/betting/tp36/frame_all.parquet"
PACE, SHAPE = pre.PACE, pre.SHAPE
TARGETS = {"win": ("p_win", "is_win", 1.0), "top2": ("p_top2", "is_top2", 2.0), "top3": ("p_top3", "is_top3", 3.0)}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigm(z):
    return 1.0 / (1.0 + np.exp(-z))


def phi(df: pd.DataFrame) -> np.ndarray:
    e = df["pf_early"].to_numpy(float)
    unk = np.isnan(e).astype(float)
    e = np.where(np.isnan(e), 0.5, e) - 0.5
    return np.column_stack([e, e ** 2, unk])


def design(df: pd.DataFrame, pace_idx: np.ndarray, shape_idx: np.ndarray, mode: str = "add") -> np.ndarray:
    F = phi(df)
    n, k = F.shape
    if mode == "none":
        return F
    X = np.zeros((n, 3 * k + 3 * k + (2 if mode == "apt" else 0)))
    for i in range(3):
        m = pace_idx == i
        X[m, i * k:(i + 1) * k] = F[m]
    for j in range(3):
        m = shape_idx == j
        X[m, 3 * k + j * k:3 * k + (j + 1) * k] = F[m]
    if mode == "apt":
        ap = np.column_stack([df[f"apt_pace_{c}"].fillna(0.0).to_numpy(float) for c in PACE])
        aq = np.column_stack([df[f"apt_shape_{c}"].fillna(0.0).to_numpy(float) for c in SHAPE])
        X[:, 6 * k] = ap[np.arange(n), pace_idx]
        X[:, 6 * k + 1] = aq[np.arange(n), shape_idx]
    return X


def fit_offset_logit(X: np.ndarray, y: np.ndarray, off: np.ndarray, lam: float = 5.0) -> np.ndarray:
    def f(th):
        z = off + X @ th
        loss = np.logaddexp(0, z).sum() - (y * z).sum() + 0.5 * lam * (th ** 2).sum()
        g = X.T @ (sigm(z) - y) + lam * th
        return loss, g

    th0 = np.zeros(X.shape[1])
    return minimize(f, th0, jac=True, method="L-BFGS-B").x


def renorm(p: np.ndarray, race: np.ndarray, k: float) -> np.ndarray:
    """レース内の合計が k になるように戻す。勝率は比例配分、連対・複勝は logit のずらし（二分法）。"""
    codes, uniq = pd.factorize(race)
    if k == 1.0:
        s = np.bincount(codes, weights=p)
        return p / s[codes]
    z = logit(p)
    lo, hi = np.full(len(uniq), -6.0), np.full(len(uniq), 6.0)
    for _ in range(30):
        mid = (lo + hi) / 2
        s = np.bincount(codes, weights=sigm(z + mid[codes]))
        up = s < k
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return sigm(z + ((lo + hi) / 2)[codes])


def bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def apt_features() -> pd.DataFrame:
    """馬ごとの『展開への適性』（as-of）: 過去走の着順パーセンタイル（0=1着）の、同じペース／隊列の走での平均 − 全走の平均（件数で縮約）。
    負＝そのペース／隊列で成績が良い。"""
    h = pd.read_parquet(D / "horse.parquet", columns=["race_id", "date", "horse_id", "horse_number", "finish_position", "field_size"])
    r = pd.read_parquet(D / "pre.parquet")[["race_id", "pace_cls", "shape_cls"]]
    h = h.merge(r, on="race_id", how="left")
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    v = ((h["finish_position"] - 1) / (h["field_size"] - 1)).to_numpy(float)
    ok = ~np.isnan(v)
    g = h["horse_id"].to_numpy()

    def prior_sum(x):
        cs = pd.Series(x).groupby(g).cumsum().to_numpy() - x
        return cs

    v0 = np.where(ok, v, 0.0)
    cs_all, n_all = prior_sum(v0), prior_sum(ok.astype(float))
    mean_all = np.where(n_all > 0, cs_all / np.where(n_all > 0, n_all, 1), 0.5)
    out = {"race_id": h["race_id"].to_numpy(), "horse_number": h["horse_number"].to_numpy()}
    for col, classes, pref in (("pace_cls", PACE, "apt_pace_"), ("shape_cls", SHAPE, "apt_shape_")):
        for c in classes:
            ind = ((h[col] == c).to_numpy() & ok).astype(float)
            n_c = prior_sum(ind)
            cs_c = prior_sum(v0 * ind)
            m_c = np.where(n_c > 0, cs_c / np.where(n_c > 0, n_c, 1), mean_all)
            out[pref + c] = (m_c - mean_all) * n_c / (n_c + 3.0)
    return pd.DataFrame(out)


def load_horses() -> pd.DataFrame:
    base = pd.read_parquet(FRAME, columns=["race_id", "horse_number", "p_win", "p_top2", "p_top3", "year"])
    hp, _ = pre._pf_all()
    out = pd.read_parquet(D / "horse.parquet", columns=["race_id", "horse_number", "finish_position", "is_win", "is_top2", "is_top3", "early_norm"])
    r = pd.read_parquet(D / "pre.parquet")[["race_id", "pace_cls", "shape_cls", "field_size", "distance", "surface"]]
    h = base.merge(out, on=["race_id", "horse_number"], how="inner").merge(apt_features(), on=["race_id", "horse_number"], how="left")
    h = h.merge(hp, on=["race_id", "horse_number"], how="left").merge(r, on="race_id", how="inner")
    return h[h["pace_cls"].isin(PACE) & h["shape_cls"].isin(SHAPE)].reset_index(drop=True)


def class_probs(train_years: list[int], pred_years: list[int], use_hana: bool) -> pd.DataFrame:
    """事前の展開の確率（ペース 3 × 隊列 3 の独立な積）。学習 train_years → pred_years のレース。"""
    d = pd.read_parquet(D / "pre.parquet")
    d = d[d["pace_cls"].isin(PACE) & d["shape_cls"].isin(SHAPE)]
    feats = [c for c in pre.RACE_FEATS if c in d.columns] + ([c for c in pre.HANA if c in d.columns] if use_hana else [])
    tr, pr = d[d["year"].isin(train_years)], d[d["year"].isin(pred_years)]
    out = pd.DataFrame({"race_id": pr["race_id"].to_numpy()})
    for name, classes, col in (("pace", PACE, "pace_cls"), ("shape", SHAPE, "shape_cls")):
        m = pre._fit_lgb(tr[feats], tr[col].map({c: i for i, c in enumerate(classes)}).to_numpy(), 3)
        p = m.predict_proba(pr[feats])
        for i, c in enumerate(classes):
            out[f"pi_{name}_{c}"] = p[:, i]
    return out


def scenario_probs(h: pd.DataFrame, th: dict, kind: str, mode: str = "add") -> dict:
    """全 9 シナリオでの各馬の確率。戻り値 {(pace, shape): array}。"""
    col, _, k = TARGETS[kind][0], TARGETS[kind][1], TARGETS[kind][2]
    base = h[col].to_numpy(float)
    z0 = logit(base)
    out = {}
    for pi, pc in enumerate(PACE):
        for qi, sc in enumerate(SHAPE):
            X = design(h, np.full(len(h), pi), np.full(len(h), qi), mode)
            p = sigm(z0 + X @ th[kind])
            out[(pc, sc)] = renorm(p, h["race_id"].to_numpy(), k)
    return out


def fit_params(h: pd.DataFrame, mode: str = "add", lam: float = 5.0) -> dict:
    pi = h["pace_cls"].map({c: i for i, c in enumerate(PACE)}).to_numpy()
    qi = h["shape_cls"].map({c: i for i, c in enumerate(SHAPE)}).to_numpy()
    X = design(h, pi, qi, mode)
    return {kind: fit_offset_logit(X, h[tg[1]].to_numpy(float), logit(h[tg[0]].to_numpy(float)), lam) for kind, tg in TARGETS.items()}


def evaluate(h: pd.DataFrame, th_sc: dict, th_none: dict, probs: pd.DataFrame, label: str, mode: str = "add") -> dict:
    """h: 評価年の馬。th_sc: シナリオ付きの係数、th_none: シナリオなし（位置のみ）の係数。probs: レースごとの pi_*。"""
    res = {}
    m = h.merge(probs, on="race_id", how="left")
    ok = m[[f"pi_pace_{c}" for c in PACE] + [f"pi_shape_{c}" for c in SHAPE]].notna().all(axis=1).to_numpy()
    m = m[ok].reset_index(drop=True)
    pi_idx = m["pace_cls"].map({c: i for i, c in enumerate(PACE)}).to_numpy()
    qi_idx = m["shape_cls"].map({c: i for i, c in enumerate(SHAPE)}).to_numpy()
    for kind, (col, ycol, k) in TARGETS.items():
        y = m[ycol].to_numpy(float)
        base = m[col].to_numpy(float)
        res[f"{label}/{kind}/base"] = bce(y, base)
        none = renorm(sigm(logit(base) + design(m, pi_idx, qi_idx, "none") @ th_none[kind]), m["race_id"].to_numpy(), k)
        res[f"{label}/{kind}/position_only"] = bce(y, none)
        sp = scenario_probs(m, th_sc, kind, mode)
        oracle = np.zeros(len(m))
        marg = np.zeros(len(m))
        for (pc, sc), p in sp.items():
            mask = (m["pace_cls"].to_numpy() == pc) & (m["shape_cls"].to_numpy() == sc)
            oracle[mask] = p[mask]
            marg += m[f"pi_pace_{pc}"].to_numpy() * m[f"pi_shape_{sc}"].to_numpy() * p
        res[f"{label}/{kind}/oracle"] = bce(y, oracle)
        res[f"{label}/{kind}/marginal"] = bce(y, marg)
        # ペースだけを知っている場合（隊列は事前の確率）
        pace_only = np.zeros(len(m))
        for (pc, sc), p in sp.items():
            mask = m["pace_cls"].to_numpy() == pc
            pace_only[mask] += (m[f"pi_shape_{sc}"].to_numpy() * p)[mask]
        res[f"{label}/{kind}/oracle_pace_only"] = bce(y, pace_only)
    res[f"{label}/n_horses"] = float(len(m))
    res[f"{label}/n_races"] = float(m["race_id"].nunique())
    return res


def cmd_eval() -> int:
    use_hana = True
    h = load_horses()
    res: dict = {}
    # 選択: 学習 2022〜2023 → 検証 2024、確認 2025（正式な 2025 は『確認』）
    tr = h[h["year"].isin([2022, 2023])]
    th_sc = fit_params(tr, "add")
    th_none = fit_params(tr, "none")
    probs = class_probs([2020, 2021, 2022, 2023], [2024, 2025], use_hana)
    th_apt = fit_params(tr, "apt")
    for yr in (2024, 2025):
        res.update(evaluate(h[h["year"] == yr], th_sc, th_none, probs, f"y{yr}"))
        res.update(evaluate(h[h["year"] == yr], th_apt, th_none, probs, f"y{yr}apt", "apt"))
    # 最終: 学習 2022〜2025 → 2026（テストセット②）
    tr2 = h[h["year"].isin([2022, 2023, 2024, 2025])]
    th_sc2, th_none2 = fit_params(tr2, "add"), fit_params(tr2, "none")
    probs2 = class_probs([2020, 2021, 2022, 2023, 2024, 2025], [2026], use_hana)
    res.update(evaluate(h[h["year"] == 2026], th_sc2, th_none2, probs2, "y2026"))
    th_apt2 = fit_params(tr2, "apt")
    res.update(evaluate(h[h["year"] == 2026], th_apt2, th_none2, probs2, "y2026apt", "apt"))
    print("apt coefficients (2022-2025):", {k: v[-2:].round(3).tolist() for k, v in th_apt2.items()})
    for k, v in res.items():
        print(f"{k:34s} {v:.5f}")
    (D / "scen_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    names = [f"pace_{c}_{f}" for c in PACE for f in ("e", "e2", "unk")] + [f"shape_{c}_{f}" for c in SHAPE for f in ("e", "e2", "unk")]
    (D / "scen_params.json").write_text(json.dumps({"names": names, "fit_2022_2025": {k: v.round(5).tolist() for k, v in th_sc2.items()},
                                                    "fit_2022_2023": {k: v.round(5).tolist() for k, v in th_sc.items()}}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval"])
    a = ap.parse_args()
    return cmd_eval()


if __name__ == "__main__":
    sys.exit(main())

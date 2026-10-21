"""tp-70 較正器の比較（ウォークフォワード）。

python -m src.scripts.data.tp70_fit compare --base base.parquet --out DIR      # 評価年 2023/2024/2025 は前年まで、2026 は 2025 年までで学習
python -m src.scripts.data.tp70_fit export  --base base.parquet --method C3 --out config/prob_calibration.json   # 2022〜2025 で当てはめた係数

較正器（目的 = 勝率 / 連対 / 複勝 ごとに別）:
  C0 raw        モデルの確率そのまま
  C1 market     単勝オッズから作った市場の確率（Harville）
  C2 platt      a + b×logit(p)         … 市場を使わない通常の較正
  C3 blend      a + b1×logit(p) + b2×logit(m)               … Benter 型の対数線形ブレンド
  C4 blend_x    C3 + 食い違い d=logit(p)−logit(m) の絶対値・d×logit(m)・頭数
  C5 gbm        単調制約つき浅い GBM [logit(p), logit(m), 頭数]
  C6 blend_pop  人気区分ごとの切片・傾き（C3 を 6 区分で別々に）
  C3r           C3 の“直近 52 週”ローリング再当てはめ（評価月の前月まで）。ドリフト対策の候補
いずれもレース内で合計（勝率 1・連対 2・複勝 3）に合わせる。市場の確率が無いとき（オッズ未確定）は C2 にフォールバック。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.scripts.data.tp70_lib import HIT, MKT, PRED, TARGETS, ece, logit, logloss, normalize_races, pop_class, race_boot_ci, sigmoid

METHODS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C3r"]
LR_C = 100.0


def feats(df: pd.DataFrame, t: str, kind: str) -> np.ndarray:
    lp = logit(df[PRED[t]])
    lm = logit(df[MKT[t]])
    d = lp - lm
    if kind == "C2":
        return lp.reshape(-1, 1)
    if kind in ("C3", "C3r", "C6"):
        return np.c_[lp, lm]
    if kind == "C4":
        return np.c_[lp, lm, np.abs(d), d * lm, np.log(df["n"].to_numpy(float))]
    if kind == "C5":
        return np.c_[lp, lm, df["n"].to_numpy(float)]
    raise ValueError(kind)


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, t: str, kind: str) -> np.ndarray:
    y = tr[HIT[t]].to_numpy()
    if kind == "C0":
        return te[PRED[t]].to_numpy()
    if kind == "C1":
        return te[MKT[t]].to_numpy()
    if kind == "C6":
        out = np.full(len(te), np.nan)
        ptr = pop_class(tr["pop"]).astype(str).to_numpy()
        pte = pop_class(te["pop"]).astype(str).to_numpy()
        for c in np.unique(pte):
            a, b = ptr == c, pte == c
            if a.sum() < 300 or y[a].min() == y[a].max():
                lr = LogisticRegression(C=LR_C, max_iter=500).fit(feats(tr, t, "C3"), y)
                out[b] = lr.predict_proba(feats(te[b], t, "C3"))[:, 1]
            else:
                lr = LogisticRegression(C=LR_C, max_iter=500).fit(feats(tr[a], t, "C3"), y[a])
                out[b] = lr.predict_proba(feats(te[b], t, "C3"))[:, 1]
        return out
    X, Xt = feats(tr, t, kind), feats(te, t, kind)
    if kind == "C5":
        g = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=300, l2_regularization=1.0, monotonic_cst=[1, 1, 0], random_state=0)
        return g.fit(X, y).predict_proba(Xt)[:, 1]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(C=LR_C, max_iter=1000).fit((X - mu) / sd, y)
    return lr.predict_proba((Xt - mu) / sd)[:, 1]


def evaluate(df: pd.DataFrame, years: list[int], kinds: list[str], train_end_of: dict[int, int]) -> pd.DataFrame:
    """各評価年 y を、学習 = year<=train_end_of[y] で当てはめ。C3r は評価月ごとに直近 52 週。"""
    preds = {k: {t: np.full(len(df), np.nan) for t in TARGETS} for k in kinds}
    for y in years:
        te_idx = np.where(df["year"].to_numpy() == y)[0]
        tr = df[df["year"] <= train_end_of[y]]
        te = df.iloc[te_idx]
        for t in TARGETS:
            for k in kinds:
                if k == "C3r":
                    continue
                preds[k][t][te_idx] = fit_predict(tr, te, t, k)
        if "C3r" in kinds:
            dts = pd.to_datetime(df["date"])
            for mo in sorted(te["month"].unique()):
                sel = np.where((df["year"].to_numpy() == y) & (df["month"].to_numpy() == mo))[0]
                start = pd.Timestamp(year=y, month=mo, day=1)
                trr = df[(dts < start) & (dts >= start - pd.Timedelta(weeks=52))]
                if len(trr) < 20000:
                    trr = df[dts < start]
                for t in TARGETS:
                    preds["C3r"][t][sel] = fit_predict(trr, df.iloc[sel], t, "C3")
    rows = []
    for k in kinds:
        for t in TARGETS:
            col = f"cal_{k}_{t}"
            df[col] = normalize_races(df.assign(_x=preds[k][t]).fillna({"_x": 0.5}), "_x", t) if k not in ("C0", "C1") else preds[k][t]
    return df


def summarize(df: pd.DataFrame, years: list[int], kinds: list[str]) -> list[dict]:
    rows = []
    for period, ys in ([(str(y), [y]) for y in years] + [("2023-25", [y for y in years if y <= 2025])]):
        x = df[df["year"].isin(ys)]
        if x.empty:
            continue
        x = x.assign(pc=pop_class(x["pop"]).astype(str))
        for t in TARGETS:
            ll_raw = None
            for k in kinds:
                col = f"cal_{k}_{t}" if k not in ("C0", "C1") else (PRED[t] if k == "C0" else MKT[t])
                ll = logloss(x[HIT[t]], x[col])
                r = {"period": period, "target": t, "method": k, "n": len(x), "ll": ll, "ece": ece(x[HIT[t]], x[col]), "bias": float(x[HIT[t]].mean() - x[col].mean())}
                for c, g in x.groupby("pc"):
                    r[f"gap_{c}"] = float(g[HIT[t]].mean() - g[col].mean())
                    r[f"ece_{c}"] = ece(g[HIT[t]], g[col])
                rows.append(r)
    return rows


def compare(a) -> None:
    df = pd.read_parquet(a.base).reset_index(drop=True)
    years = [2023, 2024, 2025, 2026]
    train_end = {2023: 2022, 2024: 2023, 2025: 2024, 2026: 2025}
    df = evaluate(df, years, METHODS, train_end)
    rows = summarize(df, years, METHODS)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(6).to_json(out / "tp70_compare.json", orient="records", force_ascii=False, indent=1)
    df.to_parquet(out / "tp70_cal_preds.parquet")
    r = pd.DataFrame(rows)
    for t in TARGETS:
        print(t)
        print(r[(r["target"] == t) & (r["period"].isin(["2023-25", "2026"]))][["period", "method", "ll", "ece", "bias", "gap_1人気", "gap_2人気", "gap_3人気", "gap_4-5人気", "gap_6-9人気", "gap_10人気以下"]].round(4).to_string(index=False))


def _coef(X: np.ndarray, y: np.ndarray) -> dict:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(C=LR_C, max_iter=1000).fit((X - mu) / sd, y)
    w = lr.coef_[0] / sd
    return {"intercept": float(lr.intercept_[0] - (w * mu).sum()), "coef": [float(v) for v in w]}


def export(a) -> None:
    """2022〜2025 年で当てはめた係数を JSON に出す。blend = C3、blend_pop = C6（人気区分ごと。件数不足の区分は blend の係数）。"""
    df = pd.read_parquet(a.base).reset_index(drop=True)
    tr = df[df["year"] <= 2025]
    pc = pop_class(tr["pop"]).astype(str).to_numpy()
    spec: dict = {
        "version": 1,
        "fit": "2022-2025（V4 の leave-one-year-out OOF、確定オッズ）",
        "market": "単勝オッズ → 正規化 1/odds → Harville（連対・複勝）",
        "features": ["logit(p_model)", "logit(p_market)"],
        "pop_classes": ["1人気", "2人気", "3人気", "4-5人気", "6-9人気", "10人気以下"],
        "default_method": {"win": "blend", "top2": "blend_pop", "top3": "blend_pop"},
        "targets": {},
    }
    for t in TARGETS:
        y = tr[HIT[t]].to_numpy()
        base = _coef(feats(tr, t, "C3"), y)
        by = {}
        for c in spec["pop_classes"]:
            m = pc == c
            by[c] = _coef(feats(tr[m], t, "C3"), y[m]) if m.sum() >= 300 and y[m].min() != y[m].max() else base
        spec["targets"][t] = {"blend": base, "blend_pop": by}
    Path(a.out).write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    print(json.dumps({t: spec["targets"][t]["blend"] for t in TARGETS}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("compare")
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=compare)
    p = sp.add_parser("export")
    p.add_argument("--base", required=True)
    p.add_argument("--method", default="C3")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=export)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

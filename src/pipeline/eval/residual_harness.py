"""親モデルの予測（勝率）の残差に、候補の列が増分を持つかを測る共通部品（再学習の候補評価）。

apt_bias_rq_eval.py・interaction_te_search.py・wind_score_test.py に書き写されていた処理（スコア検定・残差の木・test 2025 の 1 回評価）を、
関数として 1 か所にまとめたもの。既存のスクリプトは結果の再現性のため書き換えない。新しい候補（レース環境 ENV など）はこちらを使う。

合格線は測る前に docs/operations/retrain_trials/README.md へ固定し、ここでは定数にする（呼び出し側で緩めない）。
  - 第 1 段（スコア検定）: |z| ≥ Bonferroni の値、かつ年ごとの z の符号がそろう。
  - 第 1 段（残差の木）: 対照（CTX のみ）との LL の差が、平均 ≤ -0.0002 かつ検証年（2022〜2024）の全部で負。
  - 第 2 段: 全年の OOF で木を学習し、test 2025 に 1 回だけ適用。対照との差 ≤ -0.0002 で採用。95% 区間はレース単位のブートストラップ。
2026 年と市場オッズは使わない。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

PASS_LINE = -0.0002
Z_LINE = 3.48   # README に固定した Bonferroni 後の線。これより緩い線は受け付けない
BOOT_YEARS = (2022, 2023, 2024)   # 第 1 段の検証年（固定）
SURFACES = ("芝", "ダート")   # 平地のみ（障害は別）
CTX = ["venue_code", "surface", "distance", "grade", "track_condition", "field_size", "a_style", "a_l3fz_ewm2", "gate"]
CAT_COLS = ("venue_code", "surface", "grade", "track_condition")
PRM = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_data_in_leaf=400, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=50.0, verbose=-1, num_threads=4, seed=7)
ROUNDS = 300


def logloss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    """p_win・finish_position を持つ表に y・res（残差）と、掛ける列 s・k・g（a_style・a_l3fz_ewm2・gate を全体平均で中心化）を足す。

    2026 年の行が混ざっていたら止める（選択に 2026 年は使わない）。
    """
    d = d.copy()
    if "year" in d and (d["year"] >= 2026).any():
        raise ValueError("2026 年の行が含まれている（候補の選択には使わない）")
    for new, src in (("s", "a_style"), ("k", "a_l3fz_ewm2"), ("g", "gate")):
        if src in d:
            d[new] = d[src] - d[src].mean()
    d["y"] = (d["finish_position"] == 1).astype(float)
    d["res"] = d["y"] - d["p_win"]
    return d


def score_test(d: pd.DataFrame, candidates: Sequence[tuple[str, str, str | None]], z_line: float = Z_LINE) -> list[dict]:
    """候補 (名前, 列, 掛ける列 or None) ごとのスコア検定。レース単位で合計して z を出す（レース内の相関に頑健）。

    d: prepare() 済み（race_id・year・p_win・res と候補の列）。z_line は Z_LINE より緩くできない（厳しくするのは可）。
    符号は検証年（BOOT_YEARS）の z だけで見る。
    """
    z_line = max(z_line, Z_LINE)
    rows = []
    for name, col, mult in candidates:
        x = d[col] * (d[mult].fillna(0) if mult else 1.0)
        ok = d[col].notna() & (d[mult].notna() if mult else True)
        s = d[ok].assign(x=x[ok])
        s["xc"] = s["x"] - s["x"].mean()
        pr = (s["res"] * s["xc"]).groupby(s["race_id"]).sum()
        z = float(pr.sum() / np.sqrt((pr ** 2).sum()))
        beta = float((s["res"] * s["xc"]).sum() / (s["p_win"] * (1 - s["p_win"]) * s["xc"] ** 2).sum())
        yz = {}
        for y, sy in s[s["year"].isin(BOOT_YEARS)].groupby("year"):
            p2 = (sy["res"] * sy["xc"]).groupby(sy["race_id"]).sum()
            yz[int(y)] = round(float(p2.sum() / np.sqrt((p2 ** 2).sum())), 2)
        rows.append({"name": name, "n": int(len(s)), "z": round(z, 2), "beta": round(beta, 4), "z_by_year": yz,
                     "pass": bool(abs(z) >= z_line and len(yz) == len(BOOT_YEARS) and len({np.sign(v) for v in yz.values()}) == 1)})
    return rows


def _logit(p: pd.Series) -> pd.Series:
    p = p.clip(1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """p_win の logit を初期値にした小さな木を残差に当て、te の予測確率を返す。"""
    import lightgbm as lgb

    cols = list(cols)
    a, b = tr[cols].copy(), te[cols].copy()
    for c in (c for c in CAT_COLS if c in cols):
        cat = sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))
        a[c] = pd.Categorical(tr[c].astype(str), categories=cat)
        b[c] = pd.Categorical(te[c].astype(str), categories=cat)
    m = lgb.train(PRM, lgb.Dataset(a, tr["y"], init_score=_logit(tr["p_win"]).to_numpy()), num_boost_round=ROUNDS)
    return 1 / (1 + np.exp(-(_logit(te["p_win"]).to_numpy() + m.predict(b, raw_score=True))))


def _flat(d: pd.DataFrame) -> pd.DataFrame:
    return d[d["surface"].isin(SURFACES)].reset_index(drop=True)


def boost_test(d: pd.DataFrame, srcs: Mapping[str, Sequence[str]], ctx: Sequence[str] = CTX, years: Sequence[int] = BOOT_YEARS) -> dict:
    """第 1 段の残差の木。年を 1 つずつ外し、対照（ctx のみ）との LL の差を見る。srcs は {供給元: 列のリスト}。

    平地（芝・ダート）だけに絞る。供給元を全部足した組は "ALL" として返す。years は検証用に変えられるが、採否は BOOT_YEARS で出した値だけを使う。
    """
    if {"control", "ALL"} & set(srcs):
        raise ValueError("供給元の名前に control・ALL は使えない")
    d = _flat(d)
    allcols = [c for v in srcs.values() for c in v]
    srcs = {**srcs, "ALL": allcols} if len(srcs) > 1 else dict(srcs)
    sets = {"control": list(ctx), **{k: list(ctx) + list(v) for k, v in srcs.items()}}
    gains: dict[str, list[float]] = {}
    for name, cols in sets.items():
        g = []
        for hold in years:
            tr, te = d[d["year"] != hold], d[d["year"] == hold]
            pr = fit_predict(tr, te, cols)
            g.append(logloss(te["y"], pr) - logloss(te["y"], te["p_win"]))
        gains[name] = g
    out: dict = {"control": [round(x, 6) for x in gains["control"]]}
    for name in srcs:
        diff = [a - b for a, b in zip(gains[name], gains["control"])]
        out[name] = {"gain_vs_base": [round(x, 6) for x in gains[name]], "diff_vs_control": [round(x, 6) for x in diff],
                     "mean_diff": round(float(np.mean(diff)), 6), "pass": bool(np.mean(diff) <= PASS_LINE and all(x < 0 for x in diff))}
    return out


def tier2(tr: pd.DataFrame, te: pd.DataFrame, srcs: Mapping[str, Sequence[str]], ctx: Sequence[str] = CTX, n_boot: int = 2000, seed: int = 0) -> dict:
    """第 2 段。全年の OOF（tr）で木を学習し、test（te）に 1 回だけ適用する。test は選択に使わない。

    平地だけに絞る。供給元ごとに乱数を seed から作り直すので、供給元の並びで区間が変わらない。n_boot の既定 2000 を採否の記録には使う。
    """
    tr, te = _flat(tr), _flat(te)
    if "year" in tr and (tr["year"] >= 2026).any():
        raise ValueError("学習側に 2026 年の行が含まれている")
    y = te["y"].to_numpy()
    ctrl = fit_predict(tr, te, ctx)
    res: dict = {"rows": int(len(te)), "ll_base": logloss(y, te["p_win"]), "ll_control": logloss(y, ctrl)}
    uniq, inv = np.unique(te["race_id"].to_numpy(), return_inverse=True)
    w = np.bincount(inv)
    li = lambda p: -(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
    for name, cols in srcs.items():
        pr = fit_predict(tr, te, list(ctx) + list(cols))
        dif = li(pr) - li(ctrl)
        per = np.bincount(inv, weights=dif)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            ix = rng.integers(0, len(uniq), len(uniq))
            boots.append(per[ix].sum() / w[ix].sum())
        res[name] = {"ll": logloss(y, pr), "diff_vs_control": float(dif.mean()),
                     "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], "adopt": bool(dif.mean() <= PASS_LINE)}
    return res

"""tp-05b〜e の部品: モデル族（LightGBM・CatBoost・XGBoost・線形・NN）を同じ入出力でそろえ、OOF 予測を保存・合成する。

どのモデルも ``fit_predict(recipe, X_tr, y_tr, race_tr, X_va, race_va, threads, seed)`` で
``{'win': 確率(レース内 Σ=1), 'top2': 確率(Σ=2), 'top3': 確率(Σ=3)}`` を返す（正規化は共通部で行う）。
softmax 系のように p_top2・p_top3 を直接出さないモデルは、割引 Harville で p_win から導く（指数 γ は OOF で当てはめる）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.pipeline.models import win_models as wm

EPS = 1e-12


# ── 割引 Harville: p_win から p_top2・p_top3 を作る
def harville_topk(race_ids: np.ndarray, p_win: np.ndarray, gamma2: float = 1.0, gamma3: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """勝率（レース内 Σ=1）から連対率・複勝率を導く。2 着以降は p^γ に比例して選ばれるとみなす（γ<1 で2着以降が荒れやすい）。
    18 頭までなら全順列を列挙せず、2 着・3 着の周辺化で計算する。"""
    p2 = np.zeros(len(p_win))
    p3 = np.zeros(len(p_win))
    codes = pd.factorize(race_ids)[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    for idx in np.split(order, bounds):
        p = np.clip(p_win[idx], EPS, None)
        p = p / p.sum()
        n = len(p)
        q2 = p ** gamma2
        q3 = p ** gamma3
        # 連対: i が 1 着、または j が 1 着で i が 2 着
        S2 = q2.sum()
        top2 = p.copy()
        for j in range(n):
            denom = S2 - q2[j]
            if denom <= EPS:
                continue
            w = q2 / denom
            w[j] = 0.0
            top2 += p[j] * w
        p2[idx] = np.minimum(top2, 1.0)
        if n <= 3:
            p3[idx] = 1.0
            continue
        # 複勝: i が 1・2・3 着のいずれか。2 着以降の選ばれやすさは p^γ3 に比例（連対率の γ2 とは別に当てはめる）。
        # 1 着 j・2 着 k の周辺化を k について先に足し込む（O(n^2)）
        S3 = q3.sum()
        top3 = p.copy()
        for j in range(n):
            d1 = S3 - q3[j]
            if d1 <= EPS:
                continue
            w2 = q3 / d1
            w2[j] = 0.0
            top3 += p[j] * w2
            dk = d1 - q3
            a = np.where(dk > EPS, q3 / (d1 * np.maximum(dk, EPS)), 0.0)
            a[j] = 0.0
            contrib = p[j] * q3 * (a.sum() - a)
            contrib[j] = 0.0
            top3 += contrib
        p3[idx] = np.minimum(top3, 1.0)
    return p2, p3


def fit_gamma(race_ids: np.ndarray, p_win: np.ndarray, finish: np.ndarray, which: int, grid: Sequence[float] = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)) -> float:
    """OOF で p_top{which} の LogLoss を最小にする γ（1 パラメータ）。"""
    y = (finish <= which).astype(float)
    best, best_g = 9.0, 1.0
    for g in grid:
        p2, p3 = harville_topk(race_ids, p_win, g if which == 2 else 1.0, g if which == 3 else 1.0)
        p = wm.normalize_to_total(race_ids, p2 if which == 2 else p3, float(which))
        ll = wm.binary_logloss(p, y)
        if ll < best:
            best, best_g = ll, g
    return best_g


# ── 前処理（NN・線形用）
class Prep:
    """数値は欠損を中央値で埋めて標準化（欠損の多い列には欠損フラグを足す）、カテゴリは整数化して埋め込み用に返す。"""

    def __init__(self, cat_cols: Sequence[str], num_cols: Sequence[str]):
        self.cat_cols, self.num_cols = list(cat_cols), list(num_cols)

    def fit(self, X: pd.DataFrame) -> "Prep":
        num = X[self.num_cols].astype("float64")
        self.med = num.median().fillna(0.0).to_numpy()
        self.flag_cols = [c for c in self.num_cols if num[c].isna().mean() > 0.02]
        filled = num.fillna(pd.Series(self.med, index=self.num_cols))
        self.mu = filled.mean().to_numpy()
        self.sd = filled.std().replace(0, 1.0).fillna(1.0).to_numpy()
        self.cats = {c: {v: i + 1 for i, v in enumerate(sorted(map(str, X[c].astype(str).unique())))} for c in self.cat_cols}
        return self

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        num = X[self.num_cols].astype("float64")
        flags = num[self.flag_cols].isna().to_numpy().astype("float32") if self.flag_cols else np.zeros((len(X), 0), "float32")
        z = (num.fillna(pd.Series(self.med, index=self.num_cols)).to_numpy() - self.mu) / self.sd
        z = np.clip(z, -5, 5).astype("float32")
        cat = np.stack([X[c].astype(str).map(self.cats[c]).fillna(0).astype("int64").to_numpy() for c in self.cat_cols], axis=1) if self.cat_cols else np.zeros((len(X), 0), "int64")
        return np.concatenate([z, flags], axis=1), cat


def _pad_races(race_ids: np.ndarray, *arrays: np.ndarray, max_n: int = 18):
    """行をレースごとにまとめて (R, max_n, ...) に詰める。戻り値: (インデックス表 (R, max_n)（無ければ -1）, 各配列を詰めたもの)。"""
    codes = pd.factorize(race_ids)[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    groups = np.split(order, bounds)
    idx = np.full((len(groups), max_n), -1, dtype="int64")
    for r, g in enumerate(groups):
        g = g[:max_n]
        idx[r, : len(g)] = g
    return idx


def nn_fit_predict(X_tr: pd.DataFrame, y_tr: dict, race_tr: np.ndarray, X_va: pd.DataFrame, race_va: np.ndarray, seed: int = 0,
                   hidden: Sequence[int] = (256, 128), epochs: int = 14, lr: float = 2e-3, wd: float = 1e-4, dropout: float = 0.15,
                   batch_races: int = 128, device: str | None = None) -> dict:
    """レース単位の NN。馬ごとの MLP の出力を、同レースの平均（文脈）と連結してスコアを出す。
    損失 = 勝ち馬の listwise softmax 交差エントロピー + p_top2・p_top3 の BCE。hidden=() で線形（条件付きロジット + BCE）。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
    num_cols = [c for c in X_tr.columns if c not in cat_cols]
    prep = Prep(cat_cols, num_cols).fit(X_tr)
    Ztr, Ctr = prep.transform(X_tr)
    Zva, Cva = prep.transform(X_va)
    idx_tr, idx_va = _pad_races(race_tr), _pad_races(race_va)
    n_cat = [len(prep.cats[c]) + 1 for c in cat_cols]
    emb_dim = [min(8, (n + 1) // 2) for n in n_cat]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(n, d) for n, d in zip(n_cat, emb_dim)])
            d_in = Ztr.shape[1] + sum(emb_dim)
            layers, d = [], d_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            self.body = nn.Sequential(*layers) if layers else nn.Identity()
            self.d = d
            self.head = nn.Linear(d * 2, 3)

        def forward(self, z, c, mask):
            e = [emb(c[..., i]) for i, emb in enumerate(self.embs)]
            h = self.body(torch.cat([z] + e, dim=-1))
            m = mask.unsqueeze(-1).float()
            ctx = (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
            return self.head(torch.cat([h, ctx.expand_as(h)], dim=-1))     # (R, n, 3): win score, top2 logit, top3 logit

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(idx_tr) / batch_races)))

    def batch(idx_rows, Z, C):
        valid = idx_rows >= 0
        safe = np.where(valid, idx_rows, 0)
        z = torch.from_numpy(Z[safe]).to(dev)
        c = torch.from_numpy(C[safe]).to(dev)
        return z, c, torch.from_numpy(valid).to(dev), safe, valid

    yw, y2, y3 = (np.asarray(y_tr[k], dtype="float32") for k in ("win", "top2", "top3"))
    n_win_race = np.array([yw[r[r >= 0]].sum() for r in idx_tr])
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(len(idx_tr))
        for s in range(0, len(perm), batch_races):
            rows = perm[s:s + batch_races]
            z, c, mask, safe, valid = batch(idx_tr[rows], Ztr, Ctr)
            out = net(z, c, mask)
            score = out[..., 0].masked_fill(~mask, -1e9)
            tw = torch.from_numpy(np.where(valid, yw[safe], 0.0)).to(dev)
            use = torch.from_numpy((n_win_race[rows] == 1).astype("float32")).to(dev)
            logp = torch.log_softmax(score, dim=1)
            loss_w = -((logp * tw).sum(1) * use).sum() / use.sum().clamp(min=1)
            bce = nn.functional.binary_cross_entropy_with_logits
            t2 = torch.from_numpy(np.where(valid, y2[safe], 0.0)).to(dev)
            t3 = torch.from_numpy(np.where(valid, y3[safe], 0.0)).to(dev)
            m = mask.float()
            loss_23 = ((bce(out[..., 1], t2, reduction="none") * m).sum() + (bce(out[..., 2], t3, reduction="none") * m).sum()) / m.sum()
            loss = loss_w + loss_23
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
    net.eval()
    sw, s2, s3 = (np.zeros(len(X_va)) for _ in range(3))
    with torch.no_grad():
        for s in range(0, len(idx_va), 512):
            z, c, mask, safe, valid = batch(idx_va[s:s + 512], Zva, Cva)
            out = net(z, c, mask).cpu().numpy()
            rr, cc = np.nonzero(valid)
            rows = safe[rr, cc]
            sw[rows], s2[rows], s3[rows] = out[rr, cc, 0], out[rr, cc, 1], out[rr, cc, 2]
    return {"win_score": sw, "top2_logit": s2, "top3_logit": s3}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def cat_to_str(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].astype(str)
    return X


def fit_predict(recipe: str, X_tr: pd.DataFrame, y_tr: dict, race_tr: np.ndarray, X_va: pd.DataFrame, race_va: np.ndarray, threads: int, seed: int = 0,
                params: dict | None = None) -> dict:
    """recipe: lgb_binary / lgb_softmax / cat_binary / xgb_binary / linear / nn。
    戻り値: 'win'（確率）, 'top2'・'top3'（確率）, 'win_score'（あれば）。正規化前の値。"""
    params = params or {}
    if recipe == "lgb_binary":
        o = wm.fit_predict("binary", X_tr, y_tr, race_tr, X_va, race_va, threads, seed=seed, params=params.get("lgb"), n_rounds=params.get("n_rounds", wm.N_ROUNDS))
        return {"win": o["win"], "top2": o["top2"], "top3": o["top3"]}
    if recipe == "lgb_softmax":
        o = wm.fit_predict("softmax", X_tr, y_tr, race_tr, X_va, race_va, threads, seed=seed, params=params.get("lgb"), n_rounds=params.get("n_rounds", wm.N_ROUNDS))
        return {"win": wm.softmax_by_race(race_va, o["win_score"]), "win_score": o["win_score"]}
    if recipe == "cat_binary":
        from catboost import CatBoostClassifier

        Xt, Xv = cat_to_str(X_tr), cat_to_str(X_va)
        cats = [c for c in Xt.columns if Xt[c].dtype == object]
        out = {}
        for tgt in ("win", "top2", "top3"):
            m = CatBoostClassifier(iterations=params.get("iterations", 600), depth=params.get("depth", 6), learning_rate=params.get("learning_rate", 0.06),
                                   l2_leaf_reg=params.get("l2_leaf_reg", 5.0), border_count=254, random_seed=seed, verbose=0, thread_count=threads,
                                   task_type=params.get("task_type", "CPU"), cat_features=cats)
            m.fit(Xt, y_tr[tgt])
            out[tgt] = m.predict_proba(Xv)[:, 1]
        return out
    if recipe == "xgb_binary":
        import xgboost as xgb

        out = {}
        for tgt in ("win", "top2", "top3"):
            m = xgb.XGBClassifier(n_estimators=params.get("n_estimators", 400), learning_rate=params.get("learning_rate", 0.05), max_depth=params.get("max_depth", 6),
                                  subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, min_child_weight=20, tree_method="hist", device=params.get("device", "cuda"),
                                  enable_categorical=True, random_state=seed, n_jobs=threads, verbosity=0)
            m.fit(X_tr, y_tr[tgt])
            out[tgt] = m.predict_proba(X_va)[:, 1]
        return out
    if recipe in ("linear", "nn"):
        hidden = () if recipe == "linear" else tuple(params.get("hidden", (256, 128)))
        o = nn_fit_predict(X_tr, y_tr, race_tr, X_va, race_va, seed=seed, hidden=hidden, epochs=params.get("epochs", 14), lr=params.get("lr", 2e-3),
                           wd=params.get("wd", 1e-4), dropout=params.get("dropout", 0.15), batch_races=params.get("batch_races", 128))
        return {"win": wm.softmax_by_race(race_va, o["win_score"]), "win_score": o["win_score"], "top2": sigmoid(o["top2_logit"]), "top3": sigmoid(o["top3_logit"])}
    raise ValueError(recipe)


def finalize(race_va: np.ndarray, out: dict, gammas: tuple[float, float] | None = None) -> dict:
    """生の出力を、レース内の合計がそろった確率にする。top2・top3 が無いモデルは割引 Harville で導く。"""
    win = wm.normalize_to_total(race_va, out["win"], 1.0)
    if "top2" in out:
        p2, p3 = out["top2"], out["top3"]
    else:
        g2, g3 = gammas or (1.0, 1.0)
        p2, p3 = harville_topk(race_va, win, g2, g3)
    return {"win": win, "top2": wm.normalize_to_total(race_va, p2, 2.0), "top3": wm.normalize_to_total(race_va, p3, 3.0)}

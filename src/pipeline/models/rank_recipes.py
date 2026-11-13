"""上位 k 頭の並びに集中する学習（ラベル × 学習モデルの比較用の recipe）。

6 着以下を同じ扱いにするラベル（関連度 = k+1-着順、k 着より下は 0）で、LightGBM・XGBoost・CatBoost のランキング学習、
二値分類（k 着以内 = 1）、着順を丸めた回帰、NN の Top-k Plackett-Luce・ListNet を、zoo.fit_predict と同じ形で学習・予測する。
戻り値はどれも {"win": レース内 softmax, "win_score": スコア}（大きいほど上位）。確率としての較正はしない（順位の比較用）。
木の設定（学習率・木の数など）は土台のメンバー（勝ちの二値向けに調整した値）をそのまま使う。目的関数ごとの調整はしていない。

使い方: patch_zoo() で zoo.fit_predict に recipe "rk" を足し、params["rk"] に設定（kind・k・gain など）を入れる。
既存の recipe の動作は変えない。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.pipeline.models import win_models as wm

NAN_FINISH = 99      # 着順が無い（競走中止など）は最下位として扱う


def clean_finish(finish) -> np.ndarray:
    f = np.asarray(finish, dtype="float64")
    return np.where(np.isfinite(f) & (f > 0), f, NAN_FINISH)


def topk_relevance(finish, k: int = 5) -> np.ndarray:
    """関連度: 1 着 = k、2 着 = k-1、…、k 着 = 1、k+1 着以下 = 0（k+1 着以下はすべて同じ扱い）。"""
    f = clean_finish(finish)
    return np.clip(k + 1 - f, 0, None).astype(int)


def topk_binary(finish, k: int = 3) -> np.ndarray:
    """k 着以内 = 1、それ以外 = 0。"""
    return (clean_finish(finish) <= k).astype(int)


def capped_finish(finish, cap: int = 6) -> np.ndarray:
    """着順を cap で丸める（cap 着以下はすべて cap）。"""
    return np.minimum(clean_finish(finish), cap).astype("float64")


def label_gain(kind: str, k: int) -> list[float]:
    """LightGBM の label_gain。exp = 2^r-1（既定と同じ）、linear = r。"""
    if kind == "linear":
        return [float(r) for r in range(k + 1)]
    return [float(2 ** r - 1) for r in range(k + 1)]


# ───────────────────────── 木のモデル
def _cats(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if str(X[c].dtype) == "category"]


def _lgb(cfg: dict, X_tr, y_tr, race_tr, X_va, threads, seed, lgb_params, n_rounds) -> np.ndarray:
    import lightgbm as lgb

    p = {**wm.BASE_PARAMS, **(lgb_params or {}), **(cfg.get("lgb_over") or {}), "seed": seed, "num_threads": threads}   # lgb_over: 2 段目の木の設定の調整（後勝ち）
    n_rounds = int(cfg.get("n_rounds", n_rounds))
    kind, k = cfg["kind"], int(cfg.get("k", 5))
    fin = y_tr["finish"]
    if kind in ("lambdarank", "xendcg"):
        rel = topk_relevance(fin, k)
        ds = lgb.Dataset(X_tr, label=rel, group=wm._groups(race_tr), categorical_feature=_cats(X_tr) or "auto")
        if kind == "lambdarank":
            p.update(objective="lambdarank", label_gain=label_gain(cfg.get("gain", "exp"), k),
                     lambdarank_truncation_level=int(cfg.get("truncation", 10)), eval_at=[k])
        else:
            p.update(objective="rank_xendcg")
        b = lgb.train(p, ds, num_boost_round=n_rounds)
        return b.predict(X_va, raw_score=True)
    if kind == "binary":
        ds = lgb.Dataset(X_tr, label=topk_binary(fin, k), categorical_feature=_cats(X_tr) or "auto")
        b = lgb.train(dict(p, objective="binary"), ds, num_boost_round=n_rounds)
        return b.predict(X_va, raw_score=True)                      # logit。大きいほど上位
    if kind == "regression":
        ds = lgb.Dataset(X_tr, label=capped_finish(fin, int(cfg.get("cap", 6))), categorical_feature=_cats(X_tr) or "auto")
        b = lgb.train(dict(p, objective="regression"), ds, num_boost_round=n_rounds)
        return -b.predict(X_va)                                     # 着順が小さいほど上位なので符号を反転
    raise ValueError(kind)


def _qid(race: np.ndarray) -> np.ndarray:
    return pd.factorize(race)[0]


def _xgb(cfg: dict, X_tr, y_tr, race_tr, X_va, threads, seed, xp: dict) -> np.ndarray:
    import xgboost as xgb

    k = int(cfg.get("k", 5))
    xp = {**xp, **(cfg.get("xgb_over") or {})}       # xgb_over: 2 段目の木の設定の調整
    if cfg.get("kind") == "binary":                  # k 着以内 = 1 の二値（GPU で回す。設定は xgb_binary と同じ）
        c = xgb.XGBClassifier(n_estimators=xp.get("n_estimators", 400), learning_rate=xp.get("learning_rate", 0.05), max_depth=xp.get("max_depth", 6),
                              subsample=xp.get("subsample", 0.8), colsample_bytree=xp.get("colsample_bytree", 0.7), reg_lambda=xp.get("reg_lambda", 5.0),
                              min_child_weight=xp.get("min_child_weight", 20), tree_method="hist", device=xp.get("device", "cuda"),
                              enable_categorical=True, random_state=seed, n_jobs=threads, verbosity=0)
        c.fit(X_tr, topk_binary(y_tr["finish"], k))
        return c.predict(X_va, output_margin=True)          # logit。大きいほど上位
    m = xgb.XGBRanker(objective=cfg["objective"], n_estimators=xp.get("n_estimators", 400), learning_rate=xp.get("learning_rate", 0.05),
                      max_depth=xp.get("max_depth", 6), subsample=xp.get("subsample", 0.8), colsample_bytree=xp.get("colsample_bytree", 0.7),
                      reg_lambda=xp.get("reg_lambda", 5.0), min_child_weight=xp.get("min_child_weight", 20), tree_method="hist",
                      device=xp.get("device", "cuda"), enable_categorical=True, random_state=seed, n_jobs=threads, verbosity=0,
                      lambdarank_pair_method=cfg.get("pair_method", "topk" if cfg["objective"] == "rank:ndcg" else "mean"),
                      lambdarank_num_pair_per_sample=int(cfg.get("pairs", k if cfg["objective"] == "rank:ndcg" else 1)))
    m.fit(X_tr, topk_relevance(y_tr["finish"], k), qid=_qid(race_tr))
    return m.predict(X_va)


def _cat(cfg: dict, X_tr, y_tr, race_tr, X_va, race_va, threads, seed) -> np.ndarray:
    from catboost import CatBoost, Pool

    from src.pipeline.models.zoo import cat_to_str

    k = int(cfg.get("k", 5))
    Xt, Xv = cat_to_str(X_tr), cat_to_str(X_va)
    cats = [c for c in Xt.columns if Xt[c].dtype == object]
    gtr, gva = _qid(race_tr), _qid(race_va)
    m = CatBoost({"loss_function": cfg.get("loss", "YetiRank"), "iterations": int(cfg.get("iterations", 800)), "depth": int(cfg.get("depth", 6)),
                  "learning_rate": float(cfg.get("learning_rate", 0.08)), "l2_leaf_reg": 5.0, "random_seed": seed, "verbose": 0,
                  "thread_count": threads, "task_type": cfg.get("task_type", "GPU"), "border_count": 254})
    m.fit(Pool(Xt, label=topk_relevance(y_tr["finish"], k), group_id=gtr, cat_features=cats))
    return m.predict(Pool(Xv, group_id=gva, cat_features=cats))


# ───────────────────────── NN（レース単位）
def plackett_luce_topk_loss(score, finish, mask, k: int = 5):
    """Top-k Plackett-Luce の負の対数尤度（レースあたりの平均）。
    1 着・2 着・…・k 着の順に「残っている馬の中から選ばれる」確率の積。k 着より下の並びは式に入らない。
    score・finish: (R, n)、mask: (R, n) の bool。finish は 1 始まり（無いものは大きな値）。
    同着は近似: 同じ着順の馬を 1 段でまとめて選び（スコアの平均を使う）、次の着順が無い分だけ項が減る。"""
    import torch

    neg = torch.finfo(score.dtype).min / 4
    s = score.masked_fill(~mask, neg)
    remaining = mask.clone()
    total = torch.zeros(s.shape[0], device=s.device, dtype=s.dtype)
    for pos in range(1, k + 1):
        chosen = (finish == pos) & mask                                   # 同着は両方を「選ばれた」として平均する
        has = chosen.any(1)
        lse = torch.logsumexp(s.masked_fill(~remaining, neg), dim=1)
        n_ch = chosen.sum(1).clamp(min=1)
        pick = (s.masked_fill(~chosen, 0.0)).sum(1) / n_ch
        total = total + torch.where(has, lse - pick, torch.zeros_like(lse))
        remaining = remaining & ~chosen
    return total.mean()


def listnet_topk_loss(score, finish, mask, k: int = 5):
    """ListNet: 目標分布（関連度 k+1-着順 を softmax ではなく比例で正規化）と予測の softmax の交差エントロピー。"""
    import torch

    rel = (k + 1 - finish).clamp(min=0).to(score.dtype) * mask.to(score.dtype)
    tot = rel.sum(1, keepdim=True)
    ok = (tot[:, 0] > 0)
    tgt = rel / tot.clamp(min=1e-9)
    logp = torch.log_softmax(score.masked_fill(~mask, torch.finfo(score.dtype).min / 4), dim=1)
    ce = -(tgt * logp.masked_fill(~mask, 0.0)).sum(1)
    return ce[ok].mean() if ok.any() else ce.sum() * 0


def _nn(cfg: dict, X_tr, y_tr, race_tr, X_va, race_va, seed, threads: int = 5) -> np.ndarray:
    import torch
    import torch.nn as nn

    from src.pipeline.models import zoo

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    k = int(cfg.get("k", 5))
    hidden: Sequence[int] = tuple(cfg.get("hidden", (128, 64)))
    epochs, lr, wd, dropout, batch_races = int(cfg.get("epochs", 14)), float(cfg.get("lr", 2e-3)), float(cfg.get("wd", 1e-4)), float(cfg.get("dropout", 0.15)), 128
    cat_cols = _cats(X_tr)
    num_cols = [c for c in X_tr.columns if c not in cat_cols]
    prep = zoo.Prep(cat_cols, num_cols, (), 30).fit(X_tr)
    Ztr, Ctr = prep.transform(X_tr)
    idx_tr = zoo._pad_races(race_tr)
    n_cat = [len(prep.cats[c]) + 1 for c in prep.all_cat]
    emb_dim = [min(8, (n + 1) // 2) for n in n_cat]
    net = zoo._build_net(Ztr.shape[1], n_cat, emb_dim, hidden, dropout, 0, 64, 0).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(idx_tr) / batch_races)))
    fin = clean_finish(y_tr["finish"]).astype("float32")
    loss_fn = plackett_luce_topk_loss if cfg["kind"] == "pl" else listnet_topk_loss
    for _ in range(epochs):
        net.train()
        perm = np.random.permutation(len(idx_tr))
        for s in range(0, len(perm), batch_races):
            rows = idx_tr[perm[s:s + batch_races]]
            valid = rows >= 0
            safe = np.where(valid, rows, 0)
            mask = torch.from_numpy(valid).to(dev)
            out = net(torch.from_numpy(Ztr[safe]).to(dev), torch.from_numpy(Ctr[safe]).to(dev), mask)
            f = torch.from_numpy(np.where(valid, fin[safe], NAN_FINISH)).to(dev)
            loss = loss_fn(out[..., 0], f, mask, k)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
    return zoo._nn_infer(net, prep, X_va, race_va, 0, dev)["win_score"]


# ───────────────────────── 入口
def fit_predict(cfg: dict, X_tr, y_tr, race_tr, X_va, race_va, threads, seed=0, params=None) -> dict:
    """cfg["model"]: lgb / xgb / cat / nn。params は zoo の member の params（lgb: {"lgb":..,"n_rounds":..}、xgb: XGB の設定）。"""
    params = params or {}
    model = cfg["model"]
    if model == "lgb":
        score = _lgb(cfg, X_tr, y_tr, race_tr, X_va, threads, seed, params.get("lgb"), params.get("n_rounds", wm.N_ROUNDS))
    elif model == "xgb":
        score = _xgb(cfg, X_tr, y_tr, race_tr, X_va, threads, seed, params)
    elif model == "cat":
        score = _cat(cfg, X_tr, y_tr, race_tr, X_va, race_va, threads, seed)
    elif model == "nn":
        score = _nn(cfg, X_tr, y_tr, race_tr, X_va, race_va, seed, threads)
    else:
        raise ValueError(model)
    score = np.asarray(score, dtype="float64")
    return {"win": wm.softmax_by_race(race_va, score), "win_score": score}


def patch_zoo() -> None:
    """zoo.fit_predict に recipe "rk" を足す（params["rk"] に cfg、残りはモデルの設定）。何度呼んでもよい。"""
    from src.pipeline.models import zoo

    if getattr(zoo.fit_predict, "_rk_patched", False):
        return
    orig = zoo.fit_predict

    def fp(recipe, X_tr, y_tr, race_tr, X_va, race_va, threads, seed=0, params=None, model_out=None):
        if recipe == "rk":
            p = dict(params or {})
            cfg = p.pop("rk")
            return fit_predict(cfg, X_tr, y_tr, race_tr, X_va, race_va, threads, seed=seed, params=p)
        return orig(recipe, X_tr, y_tr, race_tr, X_va, race_va, threads, seed=seed, params=params, model_out=model_out)

    fp._rk_patched = True
    zoo.fit_predict = fp

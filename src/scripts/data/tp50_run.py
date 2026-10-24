"""tp-50: 馬券最適化（ルール・教師あり・オフライン強化学習・方策勾配・PPO・リーグ・進化戦略・決定木）の walk-forward 比較。

python -m src.scripts.data.tp50_run wf --year 2024 --classes rules,lgbm,mlp,pg,ppo,league,es,tree   # 年 Y: 学習 = Y より前の年、評価 = Y
python -m src.scripts.data.tp50_run summarize                                                       # 2024+2025 のプール（主）と年別
python -m src.scripts.data.tp50_run register                                                        # 2026 年で評価する方策を 2024-25 の結果だけから固定してハッシュを記録
python -m src.scripts.data.tp50_run final2026                                                       # 固定した方策だけを 2026 年で 1 回評価

方策はすべて『各行にいくら賭けるか（stake）』で表す。報酬は確定払戻（100 円あたり倍率 ret）。学習に評価年の情報は一切使わない
（早期停止は学習年の末尾 15% のレースで決める）。ret は学習時のみ 30 倍で頭打ち（評価は頭打ちなし）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data.tp50_lib import (BT_NAMES, FEATS, PRIMARY_YEARS, RET_CAP, TP50, Stats, load_years, pad_races, race_starts, reality_check, roi_ci,
                                       standardize)

RES = TP50 / "res"
THREADS = 3
WANT: set[str] | None = None     # 2026 年の最終評価で、固定した方策に必要な構成だけを学習するための絞り込み
CLASS_ALIAS = {"rule": "rules"}


def want(prefix: str) -> bool:
    return WANT is None or any(w.startswith(prefix) for w in WANT)
SEEDS = (0, 1, 2)
KS = 320


def dev():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ───────────────────────── ルール ─────────────────────────
def cls_rules(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    from sklearn.isotonic import IsotonicRegression

    out: dict = {}
    bt_ev = ev["bt"].to_numpy()
    pos_bt = ev.groupby(["ridx", "bt"], sort=False).cumcount().to_numpy()
    for bt in range(6):
        mb = bt_ev == bt
        mt = tr["bt"].to_numpy() == bt
        if mb.sum() == 0:
            continue
        for col in ("ev_mix", "ev_model"):
            v = ev[col].to_numpy()
            for th in (1.0, 1.1, 1.2, 1.5, 2.0, 3.0):
                out[f"rule|{BT_NAMES[bt]}|{col}>={th}"] = (mb & (v >= th)).astype(np.float32)
            for th in (1.0, 1.2):
                out[f"rule|{BT_NAMES[bt]}|{col}>={th}&top1"] = (mb & (v >= th) & (pos_bt < 1)).astype(np.float32)
            if mt.sum() >= 2000:
                ir = IsotonicRegression(y_min=0, out_of_bounds="clip").fit(tr.loc[mt, col].clip(upper=50), tr.loc[mt, "ret"].clip(upper=RET_CAP))
                c = ir.predict(np.clip(v, 0, 50))
                for th in (0.9, 0.95, 1.0, 1.05):
                    out[f"rule|{BT_NAMES[bt]}|cal_{col}>={th}"] = (mb & (c >= th)).astype(np.float32)
        for k in (1, 3):
            out[f"rule|{BT_NAMES[bt]}|top{k}_ev_mix"] = (mb & (pos_bt < k)).astype(np.float32)
    return out


# ───────────────────────── 教師あり（LightGBM） ─────────────────────────
def hit_scale(tr: pd.DataFrame) -> np.ndarray:
    """的中時の ret / odds の中央値（券種別）。ワイド・複勝は下限+0.3×幅の目安オッズなので 1 でない。"""
    c = np.ones(6)
    h = tr[tr["ret"] > 0]
    for bt in range(6):
        m = h["bt"].to_numpy() == bt
        if m.sum() > 50:
            c[bt] = float(np.median(h.loc[m, "ret"].to_numpy() / h.loc[m, "odds"].to_numpy()))
    return c


def kelly_stake(p: np.ndarray, o: np.ndarray, f: float, ridx: np.ndarray, starts: np.ndarray, cap: float = 0.2) -> np.ndarray:
    edge = p * o - 1.0
    st = np.where(edge > 0, f * edge / np.maximum(o - 1.0, 1e-6), 0.0)
    st = np.clip(st, 0, 0.1)
    tot = np.add.reduceat(st, starts)
    scale = np.where(tot > cap, cap / np.maximum(tot, 1e-12), 1.0)
    return (st * np.repeat(scale, np.diff(np.append(starts, len(st))))).astype(np.float32)


def cls_lgbm(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    import lightgbm as lgb

    Xtr, Xev = tr[FEATS].to_numpy(np.float32), ev[FEATS].to_numpy(np.float32)
    c = hit_scale(tr)
    odds = ev["odds"].to_numpy(np.float64)
    bt = ev["bt"].to_numpy()
    starts, counts = race_starts(ev["ridx"].to_numpy())
    pos_race = np.arange(len(ev)) - np.repeat(starts, counts)
    out: dict = {}
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             reg_lambda=5.0, n_jobs=THREADS, verbose=-1)
    clf.fit(Xtr, (tr["ret"].to_numpy() > 0).astype(int))
    p = clf.predict_proba(Xev)[:, 1]
    reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            reg_lambda=5.0, n_jobs=THREADS, verbose=-1)
    reg.fit(Xtr, np.minimum(tr["ret"].to_numpy(), RET_CAP))
    evh = p * odds * c[bt]
    evr = reg.predict(Xev)
    for name, v, ths in (("clf", evh, (1.0, 1.05, 1.1, 1.2, 1.3, 1.5)), ("reg", evr, (1.0, 1.1, 1.2))):
        for th in ths:
            out[f"lgbm|{name}|pooled>={th}"] = (v >= th).astype(np.float32)
            for b in range(6):
                out[f"lgbm|{name}|{BT_NAMES[b]}>={th}"] = ((bt == b) & (v >= th)).astype(np.float32)
        if name == "clf":
            # レース内順位
            order = pd.Series(v).groupby(ev["ridx"].to_numpy()).rank(ascending=False, method="first").to_numpy()
            for k in (1, 3):
                out[f"lgbm|clf|top{k}&ev>=1.0"] = ((order <= k) & (v >= 1.0)).astype(np.float32)
    for f in (0.1, 0.25):
        out[f"lgbm|kelly{f}"] = kelly_stake(p * c[bt], odds, f, ev["ridx"].to_numpy(), starts)
    return out


# ───────────────────────── MLP（ブートストラップ・アンサンブルの悲観的な価値推定） ─────────────────────────
def _mlp(F: int, H: int = 128):
    import torch.nn as nn

    return nn.Sequential(nn.Linear(F, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1))


def train_hit_ensemble(Xtr: np.ndarray, ytr: np.ndarray, race_tr: np.ndarray, K: int = 8, epochs: int = 6, seed: int = 0):
    import torch

    D = dev()
    X = torch.tensor(Xtr, dtype=torch.float32, device=D)
    y = torch.tensor(ytr, dtype=torch.float32, device=D)
    nr = int(race_tr.max()) + 1
    rid = torch.tensor(race_tr, dtype=torch.long, device=D)
    nets = []
    for k in range(K):
        g = torch.Generator(device="cpu").manual_seed(seed * 100 + k)
        torch.manual_seed(seed * 100 + k)
        wr = torch.poisson(torch.ones(nr), generator=g).to(D)
        net = _mlp(X.shape[1]).to(D)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        n = X.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n, device=D)
            for b in range(0, n, 8192):
                ib = perm[b:b + 8192]
                logit = net(X[ib]).squeeze(-1)
                w = wr[rid[ib]]
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, y[ib], reduction="none") * w).sum() / w.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
        nets.append(net.eval())
    return nets


def predict_ensemble(nets, Xev: np.ndarray) -> np.ndarray:
    import torch

    D = dev()
    X = torch.tensor(Xev, dtype=torch.float32, device=D)
    outs = []
    with torch.no_grad():
        for net in nets:
            ps = [torch.sigmoid(net(X[b:b + 65536]).squeeze(-1)).cpu().numpy() for b in range(0, len(X), 65536)]
            outs.append(np.concatenate(ps))
    return np.stack(outs, 1)       # [行, K]


def cls_mlp(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    Xtr, Xev = standardize(tr[FEATS].to_numpy(np.float32), ev[FEATS].to_numpy(np.float32))
    nets = train_hit_ensemble(Xtr, (tr["ret"].to_numpy() > 0).astype(np.float32), pd.factorize(tr["ridx"])[0])
    P = predict_ensemble(nets, Xev)
    c = hit_scale(tr)
    odds = ev["odds"].to_numpy(np.float64)
    bt = ev["bt"].to_numpy()
    EV = P * (odds * c[bt])[:, None]
    mu, sd = EV.mean(1), EV.std(1)
    out: dict = {}
    order = None
    for lam in (0.0, 0.5, 1.0, 2.0):
        lcb = mu - lam * sd
        for th in (1.0, 1.05, 1.1, 1.2):
            out[f"mlp|lcb{lam}>={th}"] = (lcb >= th).astype(np.float32)
        r = pd.Series(lcb).groupby(ev["ridx"].to_numpy()).rank(ascending=False, method="first").to_numpy()
        out[f"mlp|lcb{lam}|top1&>=1.0"] = ((r <= 1) & (lcb >= 1.0)).astype(np.float32)
    return out


# ───────────────────────── 方策勾配（完全情報の文脈つきバンディット）と PPO ─────────────────────────
def make_race_net(F: int, H: int):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(F, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
            self.t = nn.Linear(H, 1)
            self.n = nn.Linear(H, 1)
            self.v = nn.Linear(H, 1)

        def forward(self, X, mask):
            h = self.f(X)
            lt = self.t(h).squeeze(-1).masked_fill(~mask, -1e9)
            m = mask.float().unsqueeze(-1)
            hm = (h * m).sum(1) / m.sum(1).clamp(min=1)
            ln = self.n(hm)
            return torch.cat([lt, ln], 1), self.v(hm).squeeze(-1)

    return Net()


def pg_loss(obj: str, profit, w, ent: float):
    import torch

    if obj == "lin":
        base = -profit.mean()
    elif obj == "log":
        base = -torch.log1p(0.5 * profit).mean()
    else:     # cvar: 利益の平均 + 下位 20% の平均
        k = max(1, int(0.2 * len(profit)))
        low = torch.topk(-profit, k).values.mean()
        base = -(profit.mean() - 0.5 * low)
    H = -(w * torch.log(w.clamp(min=1e-9))).sum(1).mean()
    return base - ent * H


def train_pg(Xp, retp, maskp, obj: str, ent: float, seed: int, epochs: int = 30, lr: float = 1e-3, H: int = 64, wd: float = 1e-4, boot: bool = False):
    import torch

    D = dev()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    R = Xp.shape[0]
    ntr = int(R * 0.85)
    Xt = torch.tensor(Xp, device=D)
    rt = torch.tensor(retp, device=D)
    mt = torch.tensor(maskp, device=D)
    net = make_race_net(Xp.shape[2], H).to(D)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    tr_idx = np.arange(ntr)
    if boot:
        tr_idx = rng.integers(0, ntr, ntr)
    va = torch.arange(ntr, R, device=D)
    best, best_state = 1e18, None
    for ep in range(epochs):
        perm = rng.permutation(tr_idx)
        for b in range(0, len(perm), 256):
            ib = torch.tensor(perm[b:b + 256], device=D)
            logits, _ = net(Xt[ib], mt[ib])
            w = torch.softmax(logits, 1)
            profit = (w[:, :-1] * (rt[ib] - 1.0)).sum(1)
            loss = pg_loss(obj, profit, w, ent)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            lv = []
            for b in range(0, len(va), 512):
                ib = va[b:b + 512]
                logits, _ = net(Xt[ib], mt[ib])
                w = torch.softmax(logits, 1)
                lv.append(pg_loss(obj, (w[:, :-1] * (rt[ib] - 1.0)).sum(1), w, 0.0).item())
            v = float(np.mean(lv))
        if v < best:
            best, best_state = v, {k: x.detach().clone() for k, x in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net.eval()


def train_ppo(Xp, retp, maskp, ent: float, seed: int, epochs: int = 25, lr: float = 5e-4, H: int = 64):
    import torch

    D = dev()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    R = Xp.shape[0]
    ntr = int(R * 0.85)
    Xt = torch.tensor(Xp, device=D)
    rt = torch.tensor(retp, device=D)
    mt = torch.tensor(maskp, device=D)
    net = make_race_net(Xp.shape[2], H).to(D)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    va = torch.arange(ntr, R, device=D)
    best, best_state = 1e18, None
    K = Xp.shape[1]
    for ep in range(epochs):
        perm = rng.permutation(ntr)
        for b in range(0, ntr, 256):
            ib = torch.tensor(perm[b:b + 256], device=D)
            with torch.no_grad():
                logits, v0 = net(Xt[ib], mt[ib])
                lp0 = torch.log_softmax(logits, 1)
                a = torch.multinomial(lp0.exp(), 1).squeeze(1)
                rew = torch.where(a < K, rt[ib].gather(1, a.clamp(max=K - 1).unsqueeze(1)).squeeze(1) - 1.0, torch.zeros_like(v0))
                adv = rew - v0
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
                old = lp0.gather(1, a.unsqueeze(1)).squeeze(1)
            for _ in range(4):
                logits, v = net(Xt[ib], mt[ib])
                lp = torch.log_softmax(logits, 1)
                ratio = (lp.gather(1, a.unsqueeze(1)).squeeze(1) - old).exp()
                pol = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
                H = -(lp.exp() * lp).sum(1).mean()
                loss = pol + 0.5 * ((v - rew) ** 2).mean() - ent * H
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            lv = []
            for b in range(0, len(va), 512):
                ib = va[b:b + 512]
                logits, _ = net(Xt[ib], mt[ib])
                w = torch.softmax(logits, 1)
                lv.append(-(w[:, :-1] * (rt[ib] - 1.0)).sum(1).mean().item())
            v = float(np.mean(lv))
        if v < best:
            best, best_state = v, {k: x.detach().clone() for k, x in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net.eval()


def net_weights(net, Xp, maskp) -> np.ndarray:
    """評価レースの [R, K+1] の重み（最後がベットしない）。"""
    import torch

    D = dev()
    out = []
    with torch.no_grad():
        for b in range(0, Xp.shape[0], 256):
            X = torch.tensor(Xp[b:b + 256], device=D)
            m = torch.tensor(maskp[b:b + 256], device=D)
            logits, _ = net(X, m)
            out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(out)


def rows_from_padded(w_t: np.ndarray, rid_local, pos, keep, n_rows: int) -> np.ndarray:
    s = np.zeros(n_rows, np.float32)
    s[keep] = w_t[rid_local[keep], pos[keep]]
    return s


def policies_from_weights(prefix: str, W: np.ndarray, rid_local, pos, keep, n_rows: int, ret_rows: np.ndarray | None = None) -> dict:
    w_t, w_null = W[:, :-1], W[:, -1]
    out = {f"{prefix}|stake": rows_from_padded(w_t, rid_local, pos, keep, n_rows)}
    am = w_t.argmax(1)
    top = w_t[np.arange(len(am)), am]
    sel = top > np.maximum(w_null, 0.2)
    pick = np.zeros_like(w_t)
    pick[np.arange(len(am))[sel], am[sel]] = 1.0
    out[f"{prefix}|argmax"] = rows_from_padded(pick, rid_local, pos, keep, n_rows)
    return out


def padded(tr: pd.DataFrame, ev: pd.DataFrame):
    Xtr, Xev = standardize(tr[FEATS].to_numpy(np.float32), ev[FEATS].to_numpy(np.float32))
    return pad_races(tr, Xtr, KS), pad_races(ev, Xev, KS)


def cls_pg(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    (Xtr, rtr, mtr, *_), (Xev, rev, mev, rid, pos, keep) = padded(tr, ev)
    out: dict = {}
    for obj in ("lin", "log", "cvar"):
        for ent in (0.0, 0.02):
            for seed in SEEDS:
                net = train_pg(Xtr, rtr, mtr, obj, ent, seed)
                out.update(policies_from_weights(f"pg|{obj}|ent{ent}|s{seed}", net_weights(net, Xev, mev), rid, pos, keep, len(ev)))
                log("pg", obj, ent, seed)
    return out


def cls_ppo(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    (Xtr, rtr, mtr, *_), (Xev, rev, mev, rid, pos, keep) = padded(tr, ev)
    out: dict = {}
    for ent in (0.01, 0.05):
        for seed in (0, 1):
            net = train_ppo(Xtr, rtr, mtr, ent, seed)
            out.update(policies_from_weights(f"ppo|ent{ent}|s{seed}", net_weights(net, Xev, mev), rid, pos, keep, len(ev)))
            log("ppo", ent, seed)
    return out


# ───────────────────────── リーグ戦（集団の競争＋批評役） ─────────────────────────
def cls_league(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    if Y < 2024:
        return {}
    inner = tr[tr["year"] < Y - 1].reset_index(drop=True)
    val = tr[tr["year"] == Y - 1].reset_index(drop=True)
    rng = np.random.default_rng(Y)
    Xin, Xva = standardize(inner[FEATS].to_numpy(np.float32), val[FEATS].to_numpy(np.float32))
    (Xi, ri, mi, *_) = pad_races(inner, Xin, KS)
    (Xv, rv, mv, rid_v, pos_v, keep_v) = pad_races(val, Xva, KS)
    # 評価年の入力は inner の統計で標準化
    mu, sd = inner[FEATS].to_numpy(np.float32).mean(0), inner[FEATS].to_numpy(np.float32).std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Xev = (ev[FEATS].to_numpy(np.float32) - mu) / sd
    (Xe, re_, me, rid_e, pos_e, keep_e) = pad_races(ev, Xev, KS)
    pop = []
    for i in range(12):
        obj = ("lin", "log", "cvar")[i % 3]
        ent = float(rng.choice([0.0, 0.01, 0.03]))
        H = int(rng.choice([32, 64, 128]))
        lr = float(rng.choice([5e-4, 1e-3, 2e-3]))
        net = train_pg(Xi, ri, mi, obj, ent, 1000 + i, H=H, lr=lr, boot=True)
        Wv = net_weights(net, Xv, mv)
        sv = rows_from_padded(Wv[:, :-1], rid_v, pos_v, keep_v, len(val))
        S = float(sv.sum())
        roi = float((sv * val["ret"].to_numpy()).sum() / S) if S > 0 else 0.0
        pop.append((roi, S, net))
        log("league member", i, obj, ent, H, lr, "val ROI", round(roi, 3), "stake", round(S, 1))
    pop.sort(key=lambda x: -x[0] if x[1] > 50 else 1e9)       # 賭けが極端に少ないものは後ろへ
    top1 = net_weights(pop[0][2], Xe, me)
    top3 = np.mean([net_weights(p[2], Xe, me) for p in pop[:3]], 0)
    out: dict = {}
    out.update(policies_from_weights("league|top1", top1, rid_e, pos_e, keep_e, len(ev)))
    out.update(policies_from_weights("league|top3", top3, rid_e, pos_e, keep_e, len(ev)))
    # 批評役: 悲観的な価値推定（ブートストラップ・アンサンブルの LCB）が 1.0 以上の買い目だけ許す
    nets = train_hit_ensemble(Xin, (inner["ret"].to_numpy() > 0).astype(np.float32), pd.factorize(inner["ridx"])[0], K=6, seed=Y)
    P = predict_ensemble(nets, Xev)
    c = hit_scale(inner)
    EV = P * (ev["odds"].to_numpy() * c[ev["bt"].to_numpy()])[:, None]
    ok = ((EV.mean(1) - EV.std(1)) >= 1.0).astype(np.float32)
    for k, (nm, W) in enumerate((("top1", top1), ("top3", top3))):
        st = rows_from_padded(W[:, :-1], rid_e, pos_e, keep_e, len(ev)) * ok
        out[f"league|{nm}|critic"] = st
    return out


# ───────────────────────── 進化戦略（線形スコア） ─────────────────────────
def cls_es(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    import torch

    D = dev()
    Xtr, Xev = standardize(tr[FEATS].to_numpy(np.float32), ev[FEATS].to_numpy(np.float32))
    n = len(Xtr)
    rng = np.random.default_rng(Y)
    sub = rng.choice(n, size=min(n, 1_500_000), replace=False)
    X = torch.tensor(Xtr[sub], device=D)
    X = torch.cat([X, torch.ones(len(X), 1, device=D)], 1)
    ret = torch.tensor(np.minimum(tr["ret"].to_numpy(np.float32)[sub], RET_CAP), device=D)
    Xe = torch.tensor(Xev, device=D)
    Xe = torch.cat([Xe, torch.ones(len(Xe), 1, device=D)], 1)
    out: dict = {}
    for z in (1.0, 2.0):
        for seed in (0, 1):
            torch.manual_seed(seed)
            th = torch.zeros(X.shape[1], device=D)
            th[-1] = -2.0
            sigma = 0.5

            def fit(t):
                sel = (X @ t > 0).float()
                m = sel.sum()
                if m < 1500:
                    return -(1500 - m) / 1500
                mean = (ret * sel).sum() / m
                var = ((ret - mean) ** 2 * sel).sum() / m
                return mean - 1.0 - z * torch.sqrt(var / m)

            for gen in range(120):
                eps = torch.randn(64, X.shape[1], device=D)
                f = torch.stack([fit(th + sigma * e) for e in torch.cat([eps, -eps])])
                f = (f - f.mean()) / (f.std() + 1e-9)
                grad = (torch.cat([eps, -eps]) * f[:, None]).mean(0) / sigma
                th = th + 0.1 * grad
                sigma = max(0.05, sigma * 0.98)
            out[f"es|z{z}|s{seed}"] = ((Xe @ th) > 0).float().cpu().numpy()
            log("es", z, seed, "train sel", int((X @ th > 0).sum()))
    return out


# ───────────────────────── 決定木のルール抽出 ─────────────────────────
def cls_tree(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    from sklearn.tree import DecisionTreeRegressor

    Xtr, Xev = tr[FEATS].to_numpy(np.float32), ev[FEATS].to_numpy(np.float32)
    rng = np.random.default_rng(Y)
    sub = rng.choice(len(Xtr), size=min(len(Xtr), 1_500_000), replace=False)
    y = np.minimum(tr["ret"].to_numpy(), RET_CAP)[sub]
    out: dict = {}
    for depth in (4, 6):
        t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=3000, random_state=0).fit(Xtr[sub], y)
        leaf_tr = t.apply(Xtr[sub])
        leaf_ev = t.apply(Xev)
        stat = pd.DataFrame({"l": leaf_tr, "y": y}).groupby("l")["y"].agg(["mean", "size"])
        for th in (1.0, 1.05, 1.1):
            good = stat.index[(stat["mean"] >= th) & (stat["size"] >= 3000)]
            out[f"tree|d{depth}|leaf>={th}"] = np.isin(leaf_ev, good).astype(np.float32)
    return out


# ───────────────────────── 教師あり（LightGBM）の拡張: ハイパーパラメータ・特徴量集合・乱数シード・券種別モデル ─────────────────────────
HPARAMS = {
    "H1": dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200),
    "H2": dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=300),
    "H3": dict(n_estimators=200, learning_rate=0.05, num_leaves=255, min_child_samples=500),
}
FSETS = {
    "full": FEATS,
    "nomix": [f for f in FEATS if f not in ("lp_mix", "lev_mix", "pq_mix")],
    "min": ["bt0", "bt1", "bt2", "bt3", "bt4", "bt5", "n", "lodds", "lp_model", "lq", "pq_model", "pop_min", "rk_p", "n_c", "r_p1", "r_ent"],
}
THS = (1.0, 1.05, 1.1, 1.2, 1.3, 1.5)


def _clf(h: dict, seed: int):
    import lightgbm as lgb

    return lgb.LGBMClassifier(**h, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=THREADS, verbose=-1, random_state=seed)


def cls_lgbm2(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    c = hit_scale(tr)
    odds = ev["odds"].to_numpy(np.float64)
    bt = ev["bt"].to_numpy()
    ytr = (tr["ret"].to_numpy() > 0).astype(int)
    out: dict = {}
    for hn, h in HPARAMS.items():
        for fn, fs in FSETS.items():
            if hn != "H1" and fn == "min":
                continue
            if not want(f"lgbm2|{hn}|{fn}|"):
                continue
            Xtr, Xev = tr[fs].to_numpy(np.float32), ev[fs].to_numpy(np.float32)
            ps = []
            for seed in (0, 1, 2):
                ps.append(_clf(h, seed).fit(Xtr, ytr).predict_proba(Xev)[:, 1])
            v = np.mean(ps, 0) * odds * c[bt]
            tag = f"lgbm2|{hn}|{fn}"
            for th in THS:
                out[f"{tag}|pooled>={th}"] = (v >= th).astype(np.float32)
                for b in range(5):
                    out[f"{tag}|{BT_NAMES[b]}>={th}"] = ((bt == b) & (v >= th)).astype(np.float32)
            log("lgbm2", hn, fn)
    return out


def cls_lgbmb(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    """券種別に専用モデルを学習（その券種の行だけ）。"""
    c = hit_scale(tr)
    out: dict = {}
    tb = tr["bt"].to_numpy()
    eb = ev["bt"].to_numpy()
    odds = ev["odds"].to_numpy(np.float64)
    for b in range(6):
        mt, me = tb == b, eb == b
        if mt.sum() < 20000 or me.sum() == 0:
            continue
        Xtr, Xev = tr.loc[mt, FEATS].to_numpy(np.float32), ev.loc[me, FEATS].to_numpy(np.float32)
        y = (tr.loc[mt, "ret"].to_numpy() > 0).astype(int)
        p = np.mean([_clf(HPARAMS["H1"], s).fit(Xtr, y).predict_proba(Xev)[:, 1] for s in (0, 1, 2)], 0)
        v = np.zeros(len(ev))
        v[me] = p * odds[me] * c[b]
        for th in THS:
            out[f"lgbmb|{BT_NAMES[b]}>={th}"] = ((eb == b) & (v >= th)).astype(np.float32)
        log("lgbmb", BT_NAMES[b])
    return out


def cls_pg2(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    """方策勾配の大規模探索（目的 3 × エントロピー 2 × 隠れ層 2 × 学習率 2 × シード 5）。"""
    (Xtr, rtr, mtr, *_), (Xev, rev, mev, rid, pos, keep) = padded(tr, ev)
    out: dict = {}
    rng = np.random.default_rng(Y + 7)
    n = 0
    for obj in ("lin", "log", "cvar"):
        for ent in (0.0, 0.02):
            for H in (32, 128):
                for lr in (5e-4, 2e-3):
                    for seed in range(5):
                        if WANT is None and rng.random() > 0.5:      # 組合せの半分を乱択（時間の都合）。最終評価では固定した方策の構成だけ学習する
                            continue
                        if not want(f"pg2|{obj}|ent{ent}|H{H}|lr{lr}|s{seed}"):
                            continue
                        net = train_pg(Xtr, rtr, mtr, obj, ent, 100 + seed, H=H, lr=lr)
                        out.update(policies_from_weights(f"pg2|{obj}|ent{ent}|H{H}|lr{lr}|s{seed}", net_weights(net, Xev, mev), rid, pos, keep, len(ev)))
                        n += 1
    log("pg2 nets", n)
    return out


def cls_lgbm3(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    """lgbm2 で良かった小さい木の周辺を探索（木の葉数 × 本数 × 特徴量集合）。券種は単勝・複勝・馬連・ワイド。"""
    c = hit_scale(tr)
    odds = ev["odds"].to_numpy(np.float64)
    bt = ev["bt"].to_numpy()
    ytr = (tr["ret"].to_numpy() > 0).astype(int)
    out: dict = {}
    for leaves in (7, 15, 31):
        for est in (200, 400):
            for fn in ("nomix", "full"):
                fs = FSETS[fn]
                if not want(f"lgbm3|L{leaves}|E{est}|{fn}|"):
                    continue
                h = dict(n_estimators=est, learning_rate=0.03, num_leaves=leaves, min_child_samples=300)
                Xtr, Xev = tr[fs].to_numpy(np.float32), ev[fs].to_numpy(np.float32)
                p = np.mean([_clf(h, sd).fit(Xtr, ytr).predict_proba(Xev)[:, 1] for sd in (0, 1)], 0)
                v = p * odds * c[bt]
                for th in (1.0, 1.05, 1.1, 1.2):
                    for b in range(4):
                        out[f"lgbm3|L{leaves}|E{est}|{fn}|{BT_NAMES[b]}>={th}"] = ((bt == b) & (v >= th)).astype(np.float32)
                log("lgbm3", leaves, est, fn)
    return out


def cls_ens(tr: pd.DataFrame, ev: pd.DataFrame, Y: int) -> dict:
    """LightGBM（H1 full・H2 nomix）と MLP アンサンブルの的中確率を平均して EV を作る。"""
    c = hit_scale(tr)
    odds = ev["odds"].to_numpy(np.float64)
    bt = ev["bt"].to_numpy()
    ytr = (tr["ret"].to_numpy() > 0).astype(int)
    ps = [
        np.mean([_clf(HPARAMS["H1"], sd).fit(tr[FSETS["full"]].to_numpy(np.float32), ytr).predict_proba(ev[FSETS["full"]].to_numpy(np.float32))[:, 1] for sd in (0, 1)], 0),
        np.mean([_clf(HPARAMS["H2"], sd).fit(tr[FSETS["nomix"]].to_numpy(np.float32), ytr).predict_proba(ev[FSETS["nomix"]].to_numpy(np.float32))[:, 1] for sd in (0, 1)], 0),
    ]
    Xtr, Xev = standardize(tr[FEATS].to_numpy(np.float32), ev[FEATS].to_numpy(np.float32))
    nets = train_hit_ensemble(Xtr, ytr.astype(np.float32), pd.factorize(tr["ridx"])[0], K=4, seed=Y)
    ps.append(predict_ensemble(nets, Xev).mean(1))
    out: dict = {}
    for nm, p in (("lgbm", np.mean(ps[:2], 0)), ("all3", np.mean(ps, 0))):
        v = p * odds * c[bt]
        for th in (1.0, 1.05, 1.1, 1.2, 1.3):
            out[f"ens|{nm}|pooled>={th}"] = (v >= th).astype(np.float32)
            for b in range(5):
                out[f"ens|{nm}|{BT_NAMES[b]}>={th}"] = ((bt == b) & (v >= th)).astype(np.float32)
    return out


CLASSES = {"rules": cls_rules, "lgbm": cls_lgbm, "mlp": cls_mlp, "pg": cls_pg, "ppo": cls_ppo, "league": cls_league, "es": cls_es, "tree": cls_tree,
           "lgbm2": cls_lgbm2, "lgbmb": cls_lgbmb, "pg2": cls_pg2, "lgbm3": cls_lgbm3, "ens": cls_ens}


# ───────────────────────── 実行・集計 ─────────────────────────
def cmd_wf(a) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    Y = a.year
    d = load_years(range(2022, Y + 1))
    tr = d[d["year"] < Y].reset_index(drop=True)
    ev = d[d["year"] == Y].reset_index(drop=True)
    del d
    log("train", len(tr), "eval", len(ev), "races", ev["ridx"].nunique())
    st = Stats(ev)
    for cname in a.classes.split(","):
        f = RES / f"{cname}_{Y}.pkl"
        if f.exists() and not a.force:
            log("skip", cname)
            continue
        t0 = time.time()
        pols = CLASSES[cname](tr, ev, Y)
        res = {k: st.summarize(v) for k, v in pols.items()}
        res["__meta__"] = {"race_ids": st.race_ids, "year": st.year}
        f.write_bytes(pickle.dumps(res))
        log("saved", cname, Y, len(pols), "policies", round(time.time() - t0), "s")


def load_res(years, classes=None) -> tuple[dict, dict]:
    pol: dict = {}
    races: dict = {}
    for f in sorted(RES.glob("*.pkl")):
        cname, y = f.stem.rsplit("_", 1)
        if not y.isdigit() or int(y) not in years or (classes and cname not in classes) or cname == "final":
            continue
        r = pickle.loads(f.read_bytes())
        meta = r.pop("__meta__")
        races[int(y)] = meta["race_ids"]
        for k, v in r.items():
            pol.setdefault(k, {})[int(y)] = v
    return pol, races


def pooled(pol_k: dict, years) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not all(y in pol_k for y in years):
        return None
    S = np.concatenate([pol_k[y]["S"] for y in years]).astype(np.float64)
    R = np.concatenate([pol_k[y]["R"] for y in years]).astype(np.float64)
    nb = np.concatenate([pol_k[y]["nb"] for y in years])
    nh = np.concatenate([pol_k[y]["nh"] for y in years])
    return S, R, nb, nh


def cmd_summarize(a) -> None:
    years = PRIMARY_YEARS
    pol, races = load_res(set(years) | {2023})
    rows, profits, names = [], [], []
    for k, v in pol.items():
        p = pooled(v, years)
        if p is None:
            continue
        S, R, nb, nh = p
        row = {"policy": k, "cls": k.split("|")[0], "stake": float(S.sum()), "bets": int(nb.sum()), "races_bet": int((S > 0).sum()), "hit_rate": float(nh.sum() / max(nb.sum(), 1))}
        if S.sum() > 0:
            roi, lo, hi = roi_ci(S, R, 400)
            idx = np.where(S > 0)[0]
            rr = R[idx] / S[idx]
            top = np.argsort(-R[idx])[:3]
            keep = np.ones(len(idx), bool)
            keep[top] = False
            row.update({"roi": roi, "lo": lo, "hi": hi, "roi_wo_top3": float(R[idx][keep].sum() / S[idx][keep].sum()) if S[idx][keep].sum() > 0 else None})
            for y in years + (2023,):
                if y in v:
                    s_, r_ = v[y]["S"].astype(np.float64), v[y]["R"].astype(np.float64)
                    row[f"y{y}"] = float(r_.sum() / s_.sum()) if s_.sum() > 0 else None
            if S.std() > 0 and (S > 0).sum() >= 100:
                profits.append(((R - S) / max(S.sum() / len(S), 1e-9)).astype(np.float32))
                names.append(k)
        rows.append(row)
    df = pd.DataFrame(rows)
    rc = None
    if profits:
        P = np.stack(profits, 1)
        rc = reality_check(P, 1000)
        rc["best_name"] = names[rc["best"]]
    out = {"n_policies": int(len(df)), "n_with_bets": int(df["roi"].notna().sum()), "reality_check": rc, "classes": df.groupby("cls").size().to_dict()}
    df.to_csv(TP50 / "summary.csv", index=False)
    (TP50 / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ok = df[df["races_bet"] >= 300].sort_values("lo", ascending=False)
    print(json.dumps(out, ensure_ascii=False, default=float))
    print(ok.head(25)[["policy", "roi", "lo", "hi", "roi_wo_top3", "races_bet", "bets", "hit_rate", "y2024", "y2025"]].to_string())
    print("by class (best lo):")
    print(ok.groupby("cls").head(1)[["policy", "roi", "lo", "hi", "races_bet"]].to_string())


def cmd_register(a) -> None:
    """2026 年で評価する方策を、2024-25 の結果だけから事前に固定する（下限上位 N と ROI 上位 N、方策クラスごとに最良 1 本）。"""
    df = pd.read_csv(TP50 / "summary.csv")
    ok = df[(df["races_bet"] >= 300) & df["roi"].notna()].copy()
    pick: list[str] = []
    pick += list(ok.sort_values("lo", ascending=False).head(5)["policy"])
    pick += list(ok.sort_values("roi", ascending=False).head(5)["policy"])
    # 2023・2024・2025 のどの年も回収率 100% 超で、賭けたレースが多い方策（頑健性の基準。2026 年は見ていない）
    rob = ok[(ok["races_bet"] >= 2000) & (ok["y2023"] > 1.0) & (ok["y2024"] > 1.0) & (ok["y2025"] > 1.0)]
    pick += list(rob.sort_values("lo", ascending=False).head(5)["policy"])
    for c, g in ok.groupby("cls"):
        pick.append(g.sort_values("lo", ascending=False).iloc[0]["policy"])
    pick = list(dict.fromkeys(pick))
    payload = {"registered_at": time.strftime("%Y-%m-%d %H:%M:%S"), "policies": pick, "n_tried": int(len(df))}
    payload["hash"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    f = TP50 / "rules_2026.json"
    if f.exists() and not a.force:
        raise SystemExit("既に固定済み: " + str(f))
    f.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("registered", payload["hash"], len(pick), "policies")


def cmd_final2026(a) -> None:
    f = TP50 / "rules_2026.json"
    payload = json.loads(f.read_text(encoding="utf-8"))
    h = payload.pop("hash")
    chk = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    if h != chk:
        raise SystemExit("固定した方策のハッシュが一致しない")
    out_f = TP50 / "final2026.json"
    if out_f.exists() and not a.force:
        raise SystemExit("2026 年の評価は 1 回だけ")
    wanted = payload["policies"]
    global WANT
    WANT = set(wanted)
    classes = sorted({CLASS_ALIAS.get(p.split("|")[0], p.split("|")[0]) for p in wanted})
    d = load_years(range(2022, 2027))
    tr = d[d["year"] < 2026].reset_index(drop=True)
    ev = d[d["year"] == 2026].reset_index(drop=True)
    st = Stats(ev)
    res: dict = {"hash": h, "registered_at": payload["registered_at"], "policies": {}}
    pn: dict = {}
    RES.mkdir(parents=True, exist_ok=True)
    for cname in classes:
        cf = RES / f"final_{cname}.pkl"            # 途中で失敗しても再計算しないための保存
        if cf.exists():
            got = pickle.loads(cf.read_bytes())
        else:
            pols = CLASSES[cname](tr, ev, 2026)
            got = {k: st.summarize(v) for k, v in pols.items() if k in WANT}
            cf.write_bytes(pickle.dumps(got))
        log("final", cname, len(got))
        pn.update(got)
    want = wanted
    profits = []
    for k in want:
        if k not in pn:
            res["policies"][k] = None
            continue
        s = pn[k]
        S, R = s["S"].astype(np.float64), s["R"].astype(np.float64)
        row = {"races_bet": int((S > 0).sum()), "bets": int(s["nb"].sum())}
        if S.sum() > 0:
            roi, lo, hi = roi_ci(S, R, 1000)
            row.update({"roi": roi, "lo": lo, "hi": hi})
            profits.append((R - S) / max(S.sum() / len(S), 1e-9))
        res["policies"][k] = row
    if profits:
        res["reality_check"] = reality_check(np.stack(profits, 1), 1000)
    out_f.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, default=float)[:3000])


# ───────────────────────── 購入時点オッズの感度 ─────────────────────────
def refresh_odds(ev: pd.DataFrame, odds_p: np.ndarray) -> pd.DataFrame:
    e = ev.copy()
    e["odds"] = odds_p.astype(np.float32)
    e["ev_model"] = (e["p_model"] * e["odds"]).astype(np.float32)
    e["ev_mix"] = (e["p_mix"] * e["odds"]).astype(np.float32)
    e["lodds"] = np.log(e["odds"].clip(lower=1.0)).astype(np.float32)
    e["lev_model"] = np.log1p(e["ev_model"].clip(0, 50)).astype(np.float32)
    e["lev_mix"] = np.log1p(e["ev_mix"].clip(0, 50)).astype(np.float32)
    return e


def cmd_sens(a) -> None:
    """判断用のオッズに対数正規の誤差（購入時点と確定の差の代理）を加えて選び、払戻は確定で精算する。"""
    d = load_years(range(2022, 2026))
    sig = (0.0, 0.05, 0.1, 0.2)
    res: dict = {}
    pols = [("fukusho", 1.1), ("fukusho", 1.3), ("wide", 1.2), ("pooled", 1.1), ("tansho", 1.3)]
    acc: dict = {}
    for Y in PRIMARY_YEARS:
        tr = d[d["year"] < Y].reset_index(drop=True)
        ev = d[d["year"] == Y].reset_index(drop=True)
        c = hit_scale(tr)
        ytr = (tr["ret"].to_numpy() > 0).astype(int)
        models = {"H1full": ([_clf(HPARAMS["H1"], 0).fit(tr[FEATS].to_numpy(np.float32), ytr)], FEATS),
                  "L15nomix": ([_clf(dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=300), sd).fit(tr[FSETS["nomix"]].to_numpy(np.float32), ytr) for sd in (0, 1)], FSETS["nomix"])}
        st = Stats(ev)
        rng = np.random.default_rng(Y)
        for sg in sig:
            for rep in range(1 if sg == 0 else 5):
                op = ev["odds"].to_numpy(np.float64) * np.exp(sg * rng.standard_normal(len(ev))) if sg > 0 else ev["odds"].to_numpy(np.float64)
                e2 = refresh_odds(ev, op)
                for mn, (ms, fs) in models.items():
                    p = np.mean([m_.predict_proba(e2[fs].to_numpy(np.float32))[:, 1] for m_ in ms], 0)
                    v = p * op * c[ev["bt"].to_numpy()]
                    for bt, th in (pols if mn == "H1full" else [("wide", 1.0), ("wide", 1.05), ("wide", 1.1), ("fukusho", 1.0)]):
                        m = (v >= th) & (np.ones(len(ev), bool) if bt == "pooled" else (ev["bt"].to_numpy() == BT_NAMES.index(bt)))
                        s_ = st.summarize(m.astype(np.float32))
                        acc.setdefault((f"{mn}|{bt}", th, sg, rep), []).append(s_)
        log("sens", Y)
    for (bt, th, sg, rep), lst in acc.items():
        S = np.concatenate([x["S"] for x in lst]).astype(np.float64)
        R = np.concatenate([x["R"] for x in lst]).astype(np.float64)
        res.setdefault(f"{bt}>={th}", {}).setdefault(str(sg), []).append({"roi": float(R.sum() / S.sum()) if S.sum() > 0 else None, "bets": float(S.sum())})
    out = {k: {sg: {"roi_mean": float(np.mean([r["roi"] for r in v if r["roi"] is not None])), "roi_min": float(np.min([r["roi"] for r in v if r["roi"] is not None])),
                    "bets_mean": float(np.mean([r["bets"] for r in v]))} for sg, v in d2.items()} for k, d2 in res.items()}
    (TP50 / "sens.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_cond(a) -> None:
    """最有力のワイド方策（小さい木・特徴量 nomix・EV≥1.0）がどんな買い目を選び、どこで稼いでいるかを年・オッズ帯・人気・頭数・月で分解する。"""
    d = load_years(range(2022, 2026))
    h = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=300)
    fs = FSETS["nomix"]
    rows = []
    for Y in PRIMARY_YEARS:
        tr = d[d["year"] < Y].reset_index(drop=True)
        ev = d[(d["year"] == Y) & (d["bt"] == 3)].reset_index(drop=True)
        c = hit_scale(tr)
        ytr = (tr["ret"].to_numpy() > 0).astype(int)
        p = np.mean([_clf(h, sd).fit(tr[fs].to_numpy(np.float32), ytr).predict_proba(ev[fs].to_numpy(np.float32))[:, 1] for sd in (0, 1)], 0)
        ev["ev_hat"] = p * ev["odds"].to_numpy() * c[3]
        ev["p_hat"] = p
        rows.append(ev[ev["ev_hat"] >= 1.0])
        log("cond", Y)
    s_ = pd.concat(rows, ignore_index=True)
    s_["month"] = s_["date"].astype(str).str[5:7]
    s_["odds_band"] = pd.cut(s_["odds"], [0, 3, 6, 10, 20, 50, 1e9], labels=["~3", "3~6", "6~10", "10~20", "20~50", "50~"])
    s_["pop_band"] = pd.cut(s_["pop_max"], [0, 3, 6, 9, 12, 99], labels=["~3", "4~6", "7~9", "10~12", "13~"])
    s_["n_band"] = pd.cut(s_["n"], [0, 10, 14, 16, 18], labels=["~10", "11~14", "15~16", "17~18"])

    def agg(g):
        return pd.Series({"bets": len(g), "roi": g["ret"].mean(), "hit": (g["ret"] > 0).mean(), "mean_ev_hat": g["ev_hat"].mean()})

    out = {"n_bets": int(len(s_)), "overall_roi": float(s_["ret"].mean())}
    for col in ("year", "odds_band", "pop_band", "n_band", "month"):
        g = s_.groupby(col, observed=True).apply(agg)
        out[col] = {str(k): {kk: float(vv) for kk, vv in r.items()} for k, r in g.iterrows()}
    # EV̂ の水準別（予測どおりに回収されるか）
    s_["evb"] = pd.cut(s_["ev_hat"], [1.0, 1.1, 1.2, 1.4, 1.8, 1e9], right=False)
    g = s_.groupby("evb", observed=True).apply(agg)
    out["ev_hat_band"] = {str(k): {kk: float(vv) for kk, vv in r.items()} for k, r in g.iterrows()}
    (TP50 / "cond.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False)[:2500])


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("wf")
    w.add_argument("--year", type=int, required=True)
    w.add_argument("--classes", default=",".join(CLASSES))
    w.add_argument("--force", action="store_true")
    sp.add_parser("summarize")
    r = sp.add_parser("register")
    r.add_argument("--force", action="store_true")
    sp.add_parser("sens")
    sp.add_parser("cond")
    f = sp.add_parser("final2026")
    f.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"wf": cmd_wf, "summarize": cmd_summarize, "register": cmd_register, "final2026": cmd_final2026, "sens": cmd_sens, "cond": cmd_cond}[a.cmd](a)


if __name__ == "__main__":
    main()

"""馬券最適化の方策（強化学習）: 候補の買い目の表（``ticket_env``）から、レースごとの賭け金の配分を学習する。

状態 = レースの候補の買い目（特徴量の集合）、行動 = 各買い目への賭け金の割合（合計 ≤ 1 = 1 レースの予算、残りは「見送り」）、
報酬 = 払戻 − 賭け金（実際の結果）。1 レース 1 ステップの文脈つきバンディット（結果は購入に影響されない）。
全買い目の報酬が結果から分かる（full information）ので、方策勾配を期待値で厳密に計算できる（expected）。
サンプリングして REINFORCE（reinforce）で学習する方式も実装し、比べる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.pipeline.betting.ticket_env import TYPE_IDX, TYPES

FEATURE_NAMES = (["type_" + t for t in TYPES] + ["log_p", "log_odds", "log_pxo", "ev", "log_ratio", "rank_p", "rank_ev", "p_rel", "ev_max_race", "field", "entropy", "win_max"])


def ticket_features(df: pd.DataFrame) -> np.ndarray:
    """候補の買い目の表 → 特徴量（行数 × len(FEATURE_NAMES)）。"""
    t = df["type"].map(TYPE_IDX).to_numpy()
    oh = np.eye(len(TYPES), dtype="float32")[t]
    p, o, pm = df["p_comb"].to_numpy(), df["odds"].to_numpy(), df["p_model"].to_numpy()
    ev = np.clip(p * o - 1.0, -1.0, 5.0)
    g = df.groupby(["race_id", "type"], sort=False)
    rank_p = g["p_comb"].rank(ascending=False, pct=True).to_numpy()
    rank_ev = df.assign(_ev=p * o).groupby(["race_id", "type"], sort=False)["_ev"].rank(ascending=False, pct=True).to_numpy()
    p_rel = (df["p_comb"] / g["p_comb"].transform("max")).to_numpy()
    ev_max = df.assign(_ev=np.clip(p * o - 1.0, -1.0, 5.0)).groupby("race_id", sort=False)["_ev"].transform("max").to_numpy()
    rest = np.stack([np.log(p), np.log(o), np.log(p * o), ev, np.log(np.clip(p, 1e-12, None) / np.clip(pm, 1e-12, None)), rank_p, rank_ev, p_rel, ev_max,
                     df["field_size"].to_numpy() / 18.0, df["win_entropy"].to_numpy() / 3.0, df["win_max"].to_numpy()], axis=1).astype("float32")
    return np.concatenate([oh, rest], axis=1)


@dataclass
class Packed:
    """レース × 買い目の詰め込み（足りない所は mask=False）。"""
    X: np.ndarray            # (R, N, F)
    mask: np.ndarray         # (R, N) bool
    mult: np.ndarray         # (R, N) 的中時の払戻倍率（外れ 0）
    odds: np.ndarray         # (R, N)
    p: np.ndarray            # (R, N) 判断に使う確率
    ttype: np.ndarray        # (R, N) 券種の番号（足りない所は -1）
    race_ids: np.ndarray     # (R,)
    index: np.ndarray = field(default=None)   # (R, N) 元の表の行番号（無ければ -1）

    @property
    def n_races(self) -> int:
        return len(self.race_ids)


def pack(df: pd.DataFrame, mu: np.ndarray | None = None, sd: np.ndarray | None = None) -> tuple[Packed, np.ndarray, np.ndarray]:
    df = df.reset_index(drop=True)
    X = ticket_features(df)
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        mu[: len(TYPES)], sd[: len(TYPES)] = 0.0, 1.0       # 券種の one-hot は標準化しない
    Z = ((X - mu) / sd).astype("float32")
    codes, uniq = pd.factorize(df["race_id"])
    n_per = np.bincount(codes)
    R, N = len(uniq), int(n_per.max())
    pos = df.groupby(codes).cumcount().to_numpy()
    out = {k: np.zeros((R, N), "float32") for k in ("mult", "odds", "p")}
    Xp = np.zeros((R, N, Z.shape[1]), "float32")
    mask = np.zeros((R, N), bool)
    tt = np.full((R, N), -1, "int8")
    idx = np.full((R, N), -1, "int64")
    Xp[codes, pos] = Z
    mask[codes, pos] = True
    out["mult"][codes, pos] = df["mult"].to_numpy()
    out["odds"][codes, pos] = df["odds"].to_numpy()
    out["p"][codes, pos] = df["p_comb"].to_numpy()
    tt[codes, pos] = df["type"].map(TYPE_IDX).to_numpy()
    idx[codes, pos] = np.arange(len(df))
    return Packed(Xp, mask, out["mult"], out["odds"], out["p"], tt, np.asarray(uniq), idx), mu, sd


# ── 評価: 賭け金の配分 w (R, N)（各レースの予算 = 1。合計 ≤ 1）から
def settle(pk: Packed, w: np.ndarray) -> pd.DataFrame:
    """レースごとの賭け金・払戻・損益（予算 1 に対する割合）。"""
    w = np.where(pk.mask, w, 0.0)
    stake = w.sum(1)
    ret = (w * pk.mult).sum(1)
    return pd.DataFrame({"race_id": pk.race_ids, "stake": stake, "ret": ret, "profit": ret - stake})


def summarize(s: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """ROI = 総払戻 / 総賭け金。95% 信頼区間はレース単位の bootstrap。"""
    st, rt = s["stake"].to_numpy(), s["ret"].to_numpy()
    n = len(s)
    roi = rt.sum() / max(st.sum(), 1e-12)
    rng = np.random.default_rng(seed)
    rois = np.empty(n_boot)
    for b in range(n_boot):
        ii = rng.integers(0, n, n)
        rois[b] = rt[ii].sum() / max(st[ii].sum(), 1e-12)
    return {"roi": float(roi), "roi_lo": float(np.quantile(rois, 0.025)), "roi_hi": float(np.quantile(rois, 0.975)),
            "profit_per_race": float((rt - st).mean()), "stake_per_race": float(st.mean()), "participation": float((st > 1e-6).mean()),
            "races": int(n), "max_drawdown": float(_max_drawdown(np.cumsum(rt - st)))}


def _max_drawdown(c: np.ndarray) -> float:
    peak = np.maximum.accumulate(np.concatenate([[0.0], c]))
    return float((peak[1:] - c).max()) if len(c) else 0.0


# ── 基準となる賭け方（どれも予算 1 以内）
def alloc_top_p(pk: Packed, ttype: str) -> np.ndarray:
    """その券種で、モデルの確率が最も高い 1 点に予算全額。"""
    w = np.zeros(pk.mask.shape)
    sc = np.where(pk.ttype == TYPE_IDX[ttype], pk.p, -1.0)
    j = sc.argmax(1)
    ok = sc.max(1) > 0
    w[np.arange(len(j))[ok], j[ok]] = 1.0
    return w


def alloc_favorite(pk: Packed, ttype: str = "tansho") -> np.ndarray:
    """その券種で、オッズが最も低い（市場の人気が最も高い）1 点に予算全額。"""
    w = np.zeros(pk.mask.shape)
    sc = np.where(pk.ttype == TYPE_IDX[ttype], -pk.odds, -1e9)
    j = sc.argmax(1)
    ok = sc.max(1) > -1e8
    w[np.arange(len(j))[ok], j[ok]] = 1.0
    return w


def alloc_ev_threshold(pk: Packed, tau: float, top: int | None = None) -> np.ndarray:
    """期待回収率 p×odds ≥ 1+τ の買い目に等分（top を指定すれば EV 上位 top 点まで）。"""
    ev = pk.p * pk.odds - 1.0
    ok = pk.mask & (ev >= tau)
    if top is not None:
        sc = np.where(ok, ev, -np.inf)
        thr = -np.sort(-sc, axis=1)[:, min(top, sc.shape[1]) - 1: min(top, sc.shape[1])]
        ok = ok & (sc >= thr)
    n = ok.sum(1, keepdims=True)
    return np.where(ok, 1.0 / np.maximum(n, 1), 0.0)


def alloc_kelly(pk: Packed, fraction: float = 0.25, cap_total: float = 1.0) -> np.ndarray:
    """買い目ごとの Kelly 割合 f = (p×odds − 1)/(odds − 1) を fraction 倍して賭ける（同じレースの買い目は独立とみなす近似）。合計が cap を超えたら縮める。"""
    f = np.clip((pk.p * pk.odds - 1.0) / np.maximum(pk.odds - 1.0, 1e-6), 0.0, None) * fraction
    f = np.where(pk.mask, f, 0.0)
    tot = f.sum(1, keepdims=True)
    return np.where(tot > cap_total, f * cap_total / np.maximum(tot, 1e-12), f)


def alloc_random(pk: Packed, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.random(pk.mask.shape)
    r = np.where(pk.mask, r, -1.0)
    w = np.zeros(pk.mask.shape)
    j = r.argmax(1)
    w[np.arange(len(j)), j] = 1.0
    return w


# ── 方策ネットワーク
def build_policy(n_feat: int, hidden: int = 64, dropout: float = 0.1):
    import torch
    import torch.nn as nn

    class Policy(nn.Module):
        """買い目ごとの表現 + レース全体の表現（平均・最大のプール）→ 買い目のスコア。見送りのスコアはレース全体の表現から出す。"""

        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
            self.score = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.skip = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, x, mask):
            h = self.enc(x)
            m = mask.unsqueeze(-1).float()
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
            mx = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
            pool = torch.cat([mean, mx], dim=-1)
            s = self.score(torch.cat([h, pool.unsqueeze(1).expand(-1, h.shape[1], -1)], dim=-1)).squeeze(-1).masked_fill(~mask, -1e9)
            k = self.skip(pool)
            return torch.cat([s, k], dim=1)      # (R, N+1): 最後が見送り

    return Policy()


@dataclass
class TrainConfig:
    mode: str = "expected"          # expected: 期待値の厳密な勾配 / reinforce: サンプリングした行動の方策勾配
    hidden: int = 64
    dropout: float = 0.1
    lr: float = 2e-3
    wd: float = 1e-4
    epochs: int = 30
    batch_races: int = 256
    entropy: float = 0.01           # 配分のエントロピー報酬（0 に近いほど集中）
    risk: float = 0.0               # レースごとの損益の分散への罰則
    samples: int = 8                # reinforce で 1 レースから引く行動数
    max_ticket: float = 1.0         # 1 点への賭け金の上限（予算比）。超えた分は見送りに回す
    seed: int = 0
    threads: int = 4


def _alloc(logits, mask, max_ticket: float):
    import torch

    p = torch.softmax(logits, dim=1)
    w = p[:, :-1]
    if max_ticket < 1.0:
        w = w.clamp(max=max_ticket)
    return w, p


def train_policy(pk: Packed, cfg: TrainConfig):
    import torch

    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = build_policy(pk.X.shape[2], cfg.hidden, cfg.dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    R = pk.n_races
    steps = cfg.epochs * int(np.ceil(R / cfg.batch_races))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    X, M = torch.from_numpy(pk.X), torch.from_numpy(pk.mask)
    G = torch.from_numpy(pk.mult - 1.0) * M      # 買い目 1 点あたりの損益（予算 1 を 1 点に賭けた場合）
    hist = []
    for ep in range(cfg.epochs):
        net.train()
        perm = rng.permutation(R)
        tot = 0.0
        for s in range(0, R, cfg.batch_races):
            ii = torch.from_numpy(perm[s: s + cfg.batch_races])
            logits = net(X[ii], M[ii])
            w, p = _alloc(logits, M[ii], cfg.max_ticket)
            g = G[ii]
            ent = -(p * torch.log(p.clamp(min=1e-12))).sum(1)
            if cfg.mode == "expected":
                r = (w * g).sum(1)
                loss = -r.mean() + cfg.risk * r.var(unbiased=False) - cfg.entropy * ent.mean()
            else:
                k = cfg.samples
                a = torch.multinomial(p.detach(), k, replacement=True)      # (B, k) 行動（最後 = 見送り）
                pad = torch.zeros(g.shape[0], 1)
                gg = torch.cat([g, pad], dim=1)
                r_a = gg.gather(1, a)
                base = (p.detach() * gg).sum(1, keepdim=True)               # 厳密な期待値を基準線にする
                logp = torch.log(p.clamp(min=1e-12)).gather(1, a)
                loss = -((r_a - base) * logp).mean() - cfg.entropy * ent.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(ii)
        hist.append(tot / R)
    net.eval()
    return net, hist


def predict_alloc(net, pk: Packed, max_ticket: float = 1.0) -> np.ndarray:
    import torch

    with torch.no_grad():
        logits = net(torch.from_numpy(pk.X), torch.from_numpy(pk.mask))
        w, _ = _alloc(logits, torch.from_numpy(pk.mask), max_ticket)
    return w.numpy().astype("float64")

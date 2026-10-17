"""tp-25c: 補助タスク（テン1F・位置取り・上がり・走破タイム・ペース）つきのレース単位マルチタスク NN。

zoo.nn_fit_predict と同じ前処理（Prep）・損失（勝ち馬の listwise softmax + 連対/複勝 BCE）に、
共有の隠れ層から出す補助ヘッド（標準化した目的変数への masked MSE）を足す。補助の目的変数が欠損の行は損失から外す。
interact=True なら、補助ヘッドの予測と『同じレースの平均との差』を勝ち馬ヘッドの入力に足す
（自分の位置取りが他馬との相対でどうなるか、を主タスクに伝える）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.pipeline.models.zoo import Prep, _pad_races


def _build(d_z, n_cat, emb_dim, hidden, dropout, n_aux, interact):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(n, d) for n, d in zip(n_cat, emb_dim)])
            d, layers = d_z + sum(emb_dim), []
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            self.body = nn.Sequential(*layers)
            self.d = d
            self.aux = nn.Linear(d, n_aux) if n_aux > 0 else None
            extra = 2 * n_aux if (interact and n_aux > 0) else 0
            self.head = nn.Linear(d * 2 + extra, 3)

        def forward(self, z, c, mask):
            e = [emb(c[..., i]) for i, emb in enumerate(self.embs)]
            h = self.body(torch.cat([z] + e, dim=-1))
            m = mask.unsqueeze(-1).float()
            ctx = (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
            a = self.aux(h) if self.aux is not None else None
            parts = [h, ctx.expand_as(h)]
            if interact and a is not None:
                amean = (a * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
                parts += [a, a - amean]
            return self.head(torch.cat(parts, dim=-1)), a, h

    return Net()


def mtl_fit_predict(X_tr: pd.DataFrame, y_tr: dict, A_tr: np.ndarray | None, race_tr: np.ndarray, X_va: pd.DataFrame, race_va: np.ndarray,
                    seed: int = 0, hidden: Sequence[int] = (128, 64), epochs: int = 14, lr: float = 2e-3, wd: float = 1e-4, dropout: float = 0.15,
                    batch_races: int = 128, aux_w: float = 0.5, interact: bool = False, device: str | None = None, want_hidden: bool = False, want_model: bool = False) -> dict:
    """A_tr: (n_train, k) の補助目的変数（欠損は NaN）。戻り値: win_score, top2_logit, top3_logit, aux（標準化した補助の予測 (n_va, k)）, hidden（want_hidden のとき）。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
    num_cols = [c for c in X_tr.columns if c not in cat_cols]
    prep = Prep(cat_cols, num_cols, [], 30).fit(X_tr)
    Ztr, Ctr = prep.transform(X_tr)
    Zva, Cva = prep.transform(X_va)
    idx_tr, idx_va = _pad_races(race_tr), _pad_races(race_va)
    n_cat = [len(prep.cats[c]) + 1 for c in prep.all_cat]
    emb_dim = [min(8, (n + 1) // 2) for n in n_cat]
    k = 0 if A_tr is None else A_tr.shape[1]
    if k:
        mu, sd = np.nanmean(A_tr, axis=0), np.nanstd(A_tr, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        As = np.clip((A_tr - mu) / sd, -5, 5).astype("float32")
        Aok = np.isfinite(As)
        As = np.where(Aok, As, 0.0).astype("float32")
    net = _build(Ztr.shape[1], n_cat, emb_dim, tuple(hidden), dropout, k, interact).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(idx_tr) / batch_races)))
    yw, y2, y3 = (np.asarray(y_tr[q], dtype="float32") for q in ("win", "top2", "top3"))
    n_win_race = np.array([yw[r[r >= 0]].sum() for r in idx_tr])
    bce = nn.functional.binary_cross_entropy_with_logits
    for _ in range(epochs):
        net.train()
        perm = np.random.permutation(len(idx_tr))
        for s in range(0, len(perm), batch_races):
            rows = perm[s:s + batch_races]
            ir = idx_tr[rows]
            valid = ir >= 0
            safe = np.where(valid, ir, 0)
            z = torch.from_numpy(Ztr[safe]).to(dev)
            c = torch.from_numpy(Ctr[safe]).to(dev)
            mask = torch.from_numpy(valid).to(dev)
            out, a, _h = net(z, c, mask)
            m = mask.float()
            tw = torch.from_numpy(np.where(valid, yw[safe], 0.0)).to(dev)
            use = torch.from_numpy((n_win_race[rows] == 1).astype("float32")).to(dev)
            t2 = torch.from_numpy(np.where(valid, y2[safe], 0.0)).to(dev)
            t3 = torch.from_numpy(np.where(valid, y3[safe], 0.0)).to(dev)
            logp = torch.log_softmax(out[..., 0].masked_fill(~mask, -1e9), dim=1)
            loss_w = -((logp * tw).sum(1) * use).sum() / use.sum().clamp(min=1)
            loss_23 = ((bce(out[..., 1], t2, reduction="none") * m).sum() + (bce(out[..., 2], t3, reduction="none") * m).sum()) / m.sum()
            loss = loss_w + loss_23
            if k and aux_w > 0:
                ta = torch.from_numpy(As[safe]).to(dev)
                ok = torch.from_numpy(Aok[safe] & valid[..., None]).to(dev).float()
                per = (((a - ta) ** 2) * ok).sum((0, 1)) / ok.sum((0, 1)).clamp(min=1)
                loss = loss + aux_w * per.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
    net.eval()
    n = len(X_va)
    model = None
    if want_model:     # 推論用に保存する（nn_predict と同じ作り: 構造の引数・学習済みの重み・前処理・補助の標準化）
        model = {"arch": {"d_z": int(Ztr.shape[1]), "n_cat": [int(x) for x in n_cat], "emb_dim": [int(x) for x in emb_dim], "hidden": [int(x) for x in hidden],
                          "dropout": float(dropout), "n_aux": int(k), "interact": bool(interact)},
                 "state": {kk: vv.detach().cpu() for kk, vv in net.state_dict().items()}, "prep": prep,
                 "mu": None if not k else mu, "sd": None if not k else sd}
    res = {"win_score": np.zeros(n), "top2_logit": np.zeros(n), "top3_logit": np.zeros(n)}
    if k:
        res["aux"] = np.zeros((n, k), dtype="float32")
    if want_hidden:
        res["hidden"] = np.zeros((n, net.d), dtype="float32")
    with torch.no_grad():
        for s in range(0, len(idx_va), 512):
            ir = idx_va[s:s + 512]
            valid = ir >= 0
            safe = np.where(valid, ir, 0)
            o, a, h = net(torch.from_numpy(Zva[safe]).to(dev), torch.from_numpy(Cva[safe]).to(dev), torch.from_numpy(valid).to(dev))
            o = o.cpu().numpy()
            rr, cc = np.nonzero(valid)
            rows_ = safe[rr, cc]
            res["win_score"][rows_], res["top2_logit"][rows_], res["top3_logit"][rows_] = o[rr, cc, 0], o[rr, cc, 1], o[rr, cc, 2]
            if k:
                res["aux"][rows_] = a.cpu().numpy()[rr, cc]
            if want_hidden:
                res["hidden"][rows_] = h.cpu().numpy()[rr, cc]
    if model is not None:
        res["model"] = model
    return res


def mtl_infer(model: dict, X: pd.DataFrame, race: np.ndarray, device: str | None = None) -> np.ndarray:
    """mtl_fit_predict(want_model=True) で保存したモデルの補助ヘッド予測（標準化した単位）を返す。形は (len(X), n_aux)。"""
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    a = model["arch"]
    net = _build(a["d_z"], a["n_cat"], a["emb_dim"], tuple(a["hidden"]), a["dropout"], a["n_aux"], a["interact"]).to(dev)
    net.load_state_dict(model["state"])
    net.eval()
    Z, C = model["prep"].transform(X)
    idx = _pad_races(np.asarray(race))
    out = np.zeros((len(X), a["n_aux"]), dtype="float32")
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            ir = idx[s:s + 512]
            valid = ir >= 0
            safe = np.where(valid, ir, 0)
            _o, aux, _h = net(torch.from_numpy(Z[safe]).to(dev), torch.from_numpy(C[safe]).to(dev), torch.from_numpy(valid).to(dev))
            rr, cc = np.nonzero(valid)
            out[safe[rr, cc]] = aux.cpu().numpy()[rr, cc]
    return out

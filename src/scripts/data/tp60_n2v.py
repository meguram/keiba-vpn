"""tp-60 対照: 血統グラフ上の Node2Vec（DeepWalk 型、p=q=1）。節点の固有 ID に引きずられるかを、既知ペアの近さで確かめる。

python -m src.scripts.data.tp60_n2v train      # embed/Efull_N_node2vec.npy（節点 × 64 次元）を作る
その後 python -m src.scripts.data.tp60_embed known で既知ペア（全兄弟・同系・父だけ違う産駒）の近さの順位・AUC を他の方式と並べる。
結果を使わない自己教師（グラフ構造のみ）。特徴量には使わない（近さの定義として制約を満たすかの検査用）。
"""

from __future__ import annotations

import os
import time

import numpy as np

from src.scripts.data import tp60_embed as te

DIM = int(os.environ.get("TP60_N2V_DIM", 64))
WALKS, LENGTH, WINDOW, NEG, STEPS, BATCH = 3, 15, 4, 5, 1200, 65536


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_csr(sire: np.ndarray, dam: np.ndarray):
    n = len(sire)
    child = np.concatenate([np.flatnonzero(sire >= 0), np.flatnonzero(dam >= 0)])
    par = np.concatenate([sire[sire >= 0], dam[dam >= 0]])
    src = np.concatenate([child, par])
    dst = np.concatenate([par, child])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, dst


def walks(indptr: np.ndarray, indices: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    deg = np.diff(indptr)
    out = []
    for _ in range(WALKS):
        cur = np.arange(n)
        w = [cur]
        for _ in range(LENGTH - 1):
            d = deg[cur]
            pick = (rng.random(len(cur)) * np.maximum(d, 1)).astype(np.int64)
            nxt = np.where(d > 0, indices[np.minimum(indptr[cur] + pick, len(indices) - 1)], cur)
            w.append(nxt)
            cur = nxt
        out.append(np.stack(w, 1))
    return np.concatenate(out, 0)


def main() -> int:
    import torch

    G, code, sire, dam = te.load_graph()
    n = len(G)
    indptr, indices = build_csr(sire, dam)
    W = walks(indptr, indices, n)
    _log(f"歩行 {W.shape}")
    centers, ctxs = [], []
    for o in range(1, WINDOW + 1):
        centers += [W[:, :-o].reshape(-1), W[:, o:].reshape(-1)]
        ctxs += [W[:, o:].reshape(-1), W[:, :-o].reshape(-1)]
    c, x = np.concatenate(centers), np.concatenate(ctxs)
    _log(f"ペア {len(c):,}")
    torch.manual_seed(0)
    torch.set_num_threads(2)
    emb_in = torch.nn.Embedding(n, DIM)
    emb_out = torch.nn.Embedding(n, DIM)
    torch.nn.init.normal_(emb_in.weight, std=0.1)
    torch.nn.init.zeros_(emb_out.weight)
    opt = torch.optim.SparseAdam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=0.01) if False else torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=0.003)
    rng = np.random.default_rng(1)
    deg = np.diff(indptr).astype(np.float64)
    pn = deg ** 0.75
    pn /= pn.sum()
    for step in range(STEPS):
        idx = rng.integers(0, len(c), BATCH)
        ci, xi = torch.from_numpy(c[idx]), torch.from_numpy(x[idx])
        neg = torch.from_numpy(rng.choice(n, size=(BATCH, NEG), p=pn))
        v = emb_in(ci)
        pos = (v * emb_out(xi)).sum(1)
        ng = torch.bmm(emb_out(neg), v.unsqueeze(2)).squeeze(2)
        loss = -(torch.nn.functional.logsigmoid(pos).mean() + torch.nn.functional.logsigmoid(-ng).mean() * NEG)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            _log(f"step {step} loss {loss.item():.4f}")
    E = emb_in.weight.detach().numpy().astype(np.float16)
    np.save(te.EMB / "Efull_N_node2vec.npy", E)
    _log("保存 Efull_N_node2vec.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

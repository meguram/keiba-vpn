"""tp-58 交互作用を表現できる NN（MLP 対照 / DCN-v2 / FM / DeepFM）を V5 の入力で学習し、V5 の LightGBM OOF とロジット空間でブレンドして上乗せを見る。

python -m src.scripts.data.tp58_nn_eval train --variant dcn [--seeds 0,1]
python -m src.scripts.data.tp58_nn_eval blend --variants mlp,dcn,fm2,deepfm

学習: 検証年 2022/2023/2024 の expanding window（検証年の 10 日前より前が学習）。1 段目は直近 1 年を早期終了用に取り置いて最良エポックを決め、2 段目で全学習期間を最良エポックだけ学習し直す。
ブレンド重みは『検証年以外の 2 年』で選び、検証年へ当てる（LOYO）。test 2025・2026 は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
WORKDIR = Path(os.environ.get("TP58_DIR", ROOT / "data/local/modeling/shared_cache/tp58"))
CACHE, OUT = WORKDIR / "cache", WORKDIR / "out"
OOF = ROOT / "data/local/modeling/training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet"
KEY = ["race_id", "horse_id"]
YEARS = (2022, 2023, 2024)
THREADS = int(os.environ.get("TP58_THREADS", 3))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs():
    F = pd.read_parquet(CACHE / "v5_frame_num.parquet")
    F["race_id"], F["horse_id"] = F["race_id"].astype(str), F["horse_id"].astype(str)
    cols = json.loads((CACHE / "columns.json").read_text(encoding="utf-8"))["num"]
    X = F[cols].to_numpy(dtype="float32")
    tr = (F["year"] <= 2021).to_numpy()
    mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    miss = np.isnan(X[tr]).mean(0)
    ind_cols = np.flatnonzero((miss > 0.005) & (miss < 0.995))
    ind = np.isnan(X[:, ind_cols]).astype("float32")
    X = np.nan_to_num(np.clip((X - mu) / sd, -5, 5), nan=0.0).astype("float32")
    parts = [X, ind]
    cat_file = CACHE / "v5_frame_cat.parquet"
    if cat_file.exists():
        C = pd.read_parquet(cat_file)
        oh = []
        for c in [c for c in C.columns if c not in KEY and C[c].nunique() <= 40]:
            oh.append(pd.get_dummies(C[c], prefix=c).to_numpy(dtype="float32"))
        if oh:
            parts.append(np.hstack(oh))
    return F[KEY + ["finish_position", "year", "date"]].reset_index(drop=True), np.hstack(parts)


def _fold_masks(meta: pd.DataFrame, vy: int):
    d = pd.to_datetime(meta["date"])
    valid = (meta["year"] == vy).to_numpy()
    train = ((d < pd.Timestamp(f"{vy}-01-01") - pd.Timedelta(days=10)) & (meta["year"] >= 2020)).to_numpy()
    es = train & (d >= pd.Timestamp(f"{vy - 1}-01-01")).to_numpy()
    return train, es, valid


def _train_once(net_fn, Xt, yw, y2, y3, ok, race_codes, tr_mask, es_mask, epochs, seed, device, lr=1e-3, wd=1e-4):
    import torch

    from src.pipeline.models.tp58_nn import listwise_loss, race_index

    torch.manual_seed(seed)
    np.random.seed(seed)
    rows = np.flatnonzero(tr_mask)
    idx_l, _ = race_index(race_codes[rows])
    idx_g = np.where(idx_l >= 0, rows[np.maximum(idx_l, 0)], -1)
    net = net_fn().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Yw, Y2, Y3 = (torch.as_tensor(a, device=device) for a in (yw, y2, y3))
    OK = torch.as_tensor(ok, device=device)
    bs = 128

    def batch_loss(ib):
        ib = torch.as_tensor(ib, device=device)
        m = ib >= 0
        safe = ib.clamp(min=0).reshape(-1)
        logits = net(Xt[safe])
        local = torch.where(m.reshape(-1), torch.arange(len(safe), device=device), torch.full_like(safe, -1)).reshape(ib.shape)
        return listwise_loss(logits, local, Yw[safe], Y2[safe], Y3[safe], OK[safe] & m.reshape(-1))

    def es_eval():
        if es_mask is None:
            return None
        net.eval()
        r = np.flatnonzero(es_mask)
        il, _ = race_index(race_codes[r])
        ig = np.where(il >= 0, r[np.maximum(il, 0)], -1)
        tot, n = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(ig), 512):
                tot += float(batch_loss(ig[s : s + 512])) * len(ig[s : s + 512])
                n += len(ig[s : s + 512])
        net.train()
        return tot / n

    hist, best, best_ep, state = [], 1e9, 0, None
    net.train()
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(len(idx_g))
        for s in range(0, len(perm), bs):
            loss = batch_loss(idx_g[perm[s : s + bs]])
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = es_eval()
        hist.append(v)
        if v is not None:
            if v < best:
                best, best_ep = v, ep
            elif ep - best_ep >= 2:
                break
    return net, best_ep, hist


def _predict(net, X, rows, race_codes, device):
    import torch

    net.eval()
    out = np.zeros((len(rows), 3), dtype="float64")
    with torch.no_grad():
        for s in range(0, len(rows), 8192):
            out[s : s + 8192] = net(torch.as_tensor(X[rows[s : s + 8192]], device=device)).cpu().numpy()
    return out


def cmd_train(a) -> None:
    import torch

    from src.pipeline.models.tp58_nn import Net

    torch.set_num_threads(THREADS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    meta, X = load_inputs()
    _log(f"入力 {X.shape} device={device} variant={a.variant}")
    fp = pd.to_numeric(meta["finish_position"], errors="coerce").to_numpy()
    ok = ~np.isnan(fp)
    fpf = np.nan_to_num(fp, nan=99)
    yw, y2, y3 = (fpf == 1).astype("float32"), (fpf <= 2).astype("float32"), (fpf <= 3).astype("float32")
    race_codes = meta["race_id"].to_numpy()
    seeds = [int(s) for s in a.seeds.split(",")]
    Xt = torch.as_tensor(X, device=device)
    preds = []
    info = {}
    for vy in YEARS:
        tr, es, va = _fold_masks(meta, vy)
        vrows = np.flatnonzero(va)
        lg = np.zeros((len(vrows), 3))
        for sd in seeds:
            mk = lambda: Net(X.shape[1], a.variant, hidden=tuple(int(h) for h in a.hidden.split(",")), drop=a.drop, in_drop=a.in_drop)  # noqa: E731
            net1, ep, hist = _train_once(mk, Xt, yw, y2, y3, ok, race_codes, tr & ~es, es, a.epochs, sd, device, a.lr, a.wd)
            ep = max(ep, 1)
            net2, _, _ = _train_once(mk, Xt, yw, y2, y3, ok, race_codes, tr, None, ep, sd, device, a.lr, a.wd)
            lg += _predict(net2, X, vrows, race_codes, device) / len(seeds)
            info[f"{vy}_seed{sd}"] = {"best_epoch": ep, "es_hist": [None if h is None else round(h, 5) for h in hist]}
            _log(f"{a.variant} {vy} seed{sd}: best_epoch={ep} es={['%.4f' % h for h in hist if h is not None]}")
        df = meta.loc[vrows, KEY + ["year"]].copy()
        w = lg[:, 0] - lg[:, 0].max()
        e = np.exp(w)
        rc = race_codes[vrows]
        starts = np.flatnonzero(np.r_[True, rc[1:] != rc[:-1]])
        s = np.add.reduceat(e, starts)
        df["p_win"] = e / np.repeat(s, np.diff(np.r_[starts, len(rc)]))
        df["p_top2"] = 1 / (1 + np.exp(-lg[:, 1]))
        df["p_top3"] = 1 / (1 + np.exp(-lg[:, 2]))
        preds.append(df)
    pd.concat(preds).to_parquet(OUT / f"nn_{a.variant}{a.tag}.parquet", index=False)
    (OUT / f"nn_{a.variant}{a.tag}_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    _log("保存")


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _blend_frame(base: pd.DataFrame, nn: pd.DataFrame, w: dict) -> pd.DataFrame:
    m = base.merge(nn, on=KEY, suffixes=("", "_nn"))
    out = m[KEY + ["year"]].copy()
    rc = m["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rc[1:] != rc[:-1]])
    cnt = np.diff(np.r_[starts, len(rc)])
    s = (1 - w["win"]) * np.log(np.clip(m["p_win"].to_numpy(), 1e-12, None)) + w["win"] * np.log(np.clip(m["p_win_nn"].to_numpy(), 1e-12, None))
    e = np.exp(s - s.max())
    out["p_win"] = e / np.repeat(np.add.reduceat(e, starts), cnt)
    for k in ("top2", "top3"):
        z = (1 - w[k]) * _logit(m[f"p_{k}"].to_numpy()) + w[k] * _logit(m[f"p_{k}_nn"].to_numpy())
        out[f"p_{k}"] = 1 / (1 + np.exp(-z))
    return out


def _load_nn(v: str) -> pd.DataFrame:
    """'a+b' は NN 出力を平均（勝ちは対数確率、連対/複勝はロジット空間）。"""
    parts = []
    for n in v.split("+"):
        d = pd.read_parquet(OUT / f"nn_{n}.parquet")
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
        parts.append(d.sort_values(KEY).reset_index(drop=True))
    if len(parts) == 1:
        return parts[0]
    out = parts[0][KEY + ["year"]].copy()
    lw = np.mean([np.log(np.clip(d["p_win"].to_numpy(), 1e-12, None)) for d in parts], 0)
    out["p_win"] = lw
    rc = out["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rc[1:] != rc[:-1]])
    e = np.exp(lw - lw.max())
    out["p_win"] = e / np.repeat(np.add.reduceat(e, starts), np.diff(np.r_[starts, len(rc)]))
    for k in ("top2", "top3"):
        z = np.mean([_logit(d[f"p_{k}"].to_numpy()) for d in parts], 0)
        out[f"p_{k}"] = 1 / (1 + np.exp(-z))
    return out


def cmd_blend(a) -> None:
    from src.pipeline.models import feature_bakeoff as fb

    base = pd.read_parquet(a.base or OOF)
    base["race_id"], base["horse_id"] = base["race_id"].astype(str), base["horse_id"].astype(str)
    meta = pd.read_parquet(CACHE / "v5_frame_num.parquet", columns=KEY + ["finish_position"])
    meta["race_id"], meta["horse_id"] = meta["race_id"].astype(str), meta["horse_id"].astype(str)
    fin = meta.set_index(KEY)["finish_position"]
    base = base.sort_values(["race_id", "horse_id"], kind="stable").reset_index(drop=True)
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    res = {}
    for v in a.variants.split(","):
        nn = _load_nn(v)
        b = base.merge(nn[KEY], on=KEY).sort_values(["race_id", "horse_id"], kind="stable").reset_index(drop=True)
        nn = b[KEY].merge(nn, on=KEY, how="left")
        ll = {}
        # 年ごと・重みごとの logloss
        def ll_of(fr, year):
            f = fr[fr["year"] == year]
            fp = fin.loc[list(zip(f["race_id"], f["horse_id"]))].to_numpy()
            r = {}
            for k, kk in (("win", 1), ("top2", 2), ("top3", 3)):
                y = (fp <= kk).astype(float)
                p = np.clip(f[f"p_{k}"].to_numpy(), 1e-9, 1 - 1e-9)
                r[k] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
            return r

        table = {}
        for w in grid:
            fr = _blend_frame(b, nn, {"win": w, "top2": w, "top3": w})
            table[w] = {y: ll_of(fr, y) for y in YEARS}
        chosen = {}
        pooled = []
        for hy in YEARS:
            others = [y for y in YEARS if y != hy]
            wsel = {}
            for k in ("win", "top2", "top3"):
                wsel[k] = min(grid, key=lambda w: sum(table[w][y][k] for y in others))
            chosen[hy] = wsel
            pooled.append(_blend_frame(b, nn, wsel).query("year == @hy"))
        P = pd.concat(pooled)
        pc = KEY + ["p_win", "p_top2", "p_top3"]
        cmpr = fb.compare_oof(b[pc], P[pc], fin, f"{v}_LOYO_blend")
        res[v] = {"chosen_w_by_year": {str(k): w for k, w in chosen.items()}, "compare": {k: (float(x) if isinstance(x, (float, np.floating)) else x) for k, x in cmpr.items()},
                  "nn_alone": {y: ll_of(nn, y) for y in YEARS}, "base": {y: ll_of(b, y) for y in YEARS}, "table_fixed_w": {str(w): {str(y): t for y, t in d.items()} for w, d in table.items()}}
        _log(f"{v}: Δwin={cmpr['Δwin']:.5f} 上限={cmpr['win_hi']:.5f} Δtop2={cmpr['Δtop2']:.5f} Δtop3={cmpr['Δtop3']:.5f} w={chosen}")
    (OUT / f"nn_blend{a.out_tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--variant", required=True)
    t.add_argument("--seeds", default="0,1")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--drop", type=float, default=0.2)
    t.add_argument("--in-drop", dest="in_drop", type=float, default=0.0)
    t.add_argument("--hidden", default="512,256")
    t.add_argument("--tag", default="")
    b = sub.add_parser("blend")
    b.add_argument("--variants", required=True)
    b.add_argument("--base", default=None)
    b.add_argument("--out-tag", dest="out_tag", default="")
    a = ap.parse_args()
    {"train": cmd_train, "blend": cmd_blend}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

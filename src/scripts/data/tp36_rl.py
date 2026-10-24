"""tp-36 (4) 強化学習の方策探索（単勝）: 単一エージェント、リーグ戦（集団ベースの進化・乗法的重み）、批評役による拒否、多目的（パレート）。

python -m src.scripts.data.tp36_rl --out data/local/modeling/betting/tp36/rl.json
環境: 過去のレースの再生。1 レースの行動 = 各馬への賭け金の割合（合計はレースの上限 B 以内、見送りもできる）。報酬 = そのレースの対数成長 log(1+Σ f_i (払戻倍率_i − 1))。
オッズは確定オッズ（購入時点の値ではない。感度は tp36_edge2 の誤差実験）。モデル確率は v4（2022〜24 は walk-forward の OOF、2025 は bundle_v4_2024）。
評価は walk-forward: 評価年 Y について、学習は Y−2 年以前、方策の選択（リーグ戦の勝ち抜き・しきい値）は Y−1 年、評価は Y 年。評価年のデータは学習・選択に使わない。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
TP = ROOT / "data/local/modeling/betting/tp36"
NMAX = 18
A0, B0 = 0.262, 0.833        # Benter の定数（2022 年までで当てはめた値。固定）


def features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    lp = np.log(np.clip(d["p_win"], 1e-6, 1))
    lq = np.log(np.clip(d["q_win"], 1e-6, 1))
    z = A0 * lp + B0 * lq
    z = z - z.groupby(d["race_id"]).transform("max")
    pb = np.exp(z) / np.exp(z).groupby(d["race_id"]).transform("sum")
    d["f_lp"], d["f_lq"] = lp, lq
    d["f_logo"] = np.log(d["win_odds"])
    d["f_ratio"] = np.log(np.clip(d["p_win"] / d["q_win"].clip(lower=1e-6), 1e-3, 1e3))
    d["f_ev_m"] = np.log(np.clip(d["p_win"] * d["win_odds"], 1e-3, 1e3))
    d["f_ev_b"] = np.log(np.clip(pb * d["win_odds"], 1e-3, 1e3))
    d["f_p3"] = np.log(np.clip(d["p_top3"], 1e-6, 1))
    d["f_rank_q"] = d.groupby("race_id")["q_win"].rank(ascending=False)
    d["f_rank_p"] = d.groupby("race_id")["p_win"].rank(ascending=False)
    d["f_n"] = d.groupby("race_id")["q_win"].transform("size")
    d["p_b"] = pb
    return d


FEATS = ["f_lp", "f_lq", "f_logo", "f_ratio", "f_ev_m", "f_ev_b", "f_p3", "f_rank_q", "f_rank_p", "f_n"]


def pack(d: pd.DataFrame, mu=None, sd=None):
    rids = d["race_id"].unique()
    idx = {r: i for i, r in enumerate(rids)}
    R = len(rids)
    X = np.zeros((R, NMAX, len(FEATS)), np.float32)
    M = np.zeros((R, NMAX), bool)
    RET = np.zeros((R, NMAX), np.float32)
    pos = d.groupby("race_id").cumcount().to_numpy()
    ri = d["race_id"].map(idx).to_numpy()
    ok = pos < NMAX
    X[ri[ok], pos[ok]] = d.loc[ok, FEATS].to_numpy(np.float32)
    M[ri[ok], pos[ok]] = True
    RET[ri[ok], pos[ok]] = (d.loc[ok, "ret_win"].to_numpy() / 100.0).astype(np.float32)
    if mu is None:
        flat = X[M]
        mu, sd = flat.mean(0), flat.std(0) + 1e-6
    Xn = ((X - mu) / sd) * M[..., None]
    return {"X": torch.tensor(Xn), "M": torch.tensor(M), "RET": torch.tensor(RET), "rids": rids, "mu": mu, "sd": sd}


class Policy(nn.Module):
    """各馬のスコア z = 線形のスキップ接続（初期値は基準ルール＝混合の期待値）+ 小さな NN。見送りロジット z0 との softmax で資金割合を決める。"""

    def __init__(self, nf: int, hidden: int = 16, drop: float = 0.1, skip: float = 1.0, forced: int = 0, single: int = 0):
        super().__init__()
        self.hidden, self.forced, self.single = hidden, forced, single
        self.skip = nn.Linear(nf, 1)
        with torch.no_grad():
            self.skip.weight.zero_()
            self.skip.bias.zero_()
            self.skip.weight[0, FEATS.index("f_ev_b")] = skip
        if hidden > 0:
            self.enc = nn.Sequential(nn.Linear(nf, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, hidden), nn.ReLU())
            self.head = nn.Linear(hidden, 1)
            self.gate = nn.Linear(hidden, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        self.z0 = nn.Parameter(torch.tensor(1.0))

    def forward(self, X, M, B: float):
        z = self.skip(X).squeeze(-1)
        z0 = self.z0.expand(X.shape[0], 1)
        if self.hidden > 0:
            h = self.enc(X) * M[..., None]
            z = z + self.head(h).squeeze(-1)
            hm = (h * M[..., None]).sum(1) / M.sum(1, keepdim=True).clamp(min=1)
            z0 = z0 + self.gate(hm)
        z = z.masked_fill(~M, -1e9)
        if self.forced:
            z0 = torch.full_like(z0, -1e9)
        a = torch.softmax(torch.cat([z0, z], dim=1), dim=1)
        if self.single:                                   # 1 点だけ買う（または見送り）: 順伝播は one-hot、逆伝播は softmax（straight-through）
            hard = torch.zeros_like(a).scatter_(1, a.argmax(1, keepdim=True), 1.0)
            a = hard + a - a.detach()
        return B * a[:, 1:]


def restrict_single(f: torch.Tensor) -> torch.Tensor:
    """配分 f（[レース, 馬]）を、最大の馬 1 頭だけに合計額を寄せた配分にする（賭けていないレースは 0 のまま）。"""
    j = f.argmax(1, keepdim=True)
    out = torch.zeros_like(f)
    out.scatter_(1, j, f.sum(1, keepdim=True))
    return out


def growth(f, RET, M):
    return torch.log1p((f * (RET - 1.0) * M).sum(1).clamp(min=-0.999))


def train(pk, hp: dict, epochs: int, seed: int, net: Policy | None = None, val=None) -> Policy:
    torch.manual_seed(seed)
    net = net or Policy(pk["X"].shape[-1], hp["hidden"], hp.get("drop", 0.1), hp.get("skip", 1.0), hp.get("forced", 0), hp.get("single", 0))
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    R = pk["X"].shape[0]
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(R)
        for s in range(0, R, 512):
            ii = perm[s:s + 512]
            f = net(pk["X"][ii], pk["M"][ii], hp["B"])
            g = growth(f, pk["RET"][ii], pk["M"][ii])
            ent = -(f / hp["B"]).clamp(min=1e-9).log().mul(f / hp["B"]).sum(1)      # 分散（エントロピー）の報酬
            loss = -(g.mean() + hp.get("ent", 0.0) * ent.mean()) + hp.get("risk", 0.0) * g.var()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


@torch.no_grad()
def alloc(net, pk, B):
    net.eval()
    return net(pk["X"], pk["M"], B)


def evaluate(f: torch.Tensor, pk, n_boot: int = 2000, seed: int = 0, thr: float = 0.002) -> dict:
    F, RET, M = f.numpy(), pk["RET"].numpy(), pk["M"].numpy()
    stake = (F * M).sum(1)
    ret = (F * RET * M).sum(1)
    g = np.log1p(np.clip(ret - stake, -0.999, None))
    W = float(np.exp(g.sum()))
    cum = np.cumsum(g)
    peak = np.maximum.accumulate(cum)
    mdd = float(1 - np.exp((cum - peak).min()))
    roi = float(ret.sum() / max(stake.sum(), 1e-12))
    bets = (F > thr) & M
    hit = float(((RET > 0) & bets).sum() / max(bets.sum(), 1))
    rng = np.random.default_rng(seed)
    n = len(stake)
    idx = rng.integers(0, n, size=(n_boot, n))
    ss, rr = stake[idx].sum(1), ret[idx].sum(1)
    ok = ss > 0
    b = rr[ok] / ss[ok]
    if len(b) == 0:
        b = np.array([np.nan])
    return {"roi": roi, "lo": float(np.nanquantile(b, 0.025)), "hi": float(np.nanquantile(b, 0.975)), "lo_adj": float(np.nanquantile(b, 0.00125)), "wealth": W, "max_dd": mdd,
            "n_bets": int(bets.sum()), "bet_races": int((stake > thr).sum()), "races": int(n), "bet_hit": hit, "mean_stake": float(stake.mean())}


def baseline_blend(pk, d_split: pd.DataFrame, tau: float, B: float) -> torch.Tensor:
    """基準: 混合の EV ≥ τ の馬へ、(EV−1) に比例した割合（レース合計 ≤ B）。"""
    rids = pk["rids"]
    ev = d_split.pivot_table(index="race_id", columns=d_split.groupby("race_id").cumcount(), values="p_b", aggfunc="first")
    odds = d_split.pivot_table(index="race_id", columns=d_split.groupby("race_id").cumcount(), values="win_odds", aggfunc="first")
    ev = (ev * odds).reindex(rids).fillna(0).to_numpy()[:, :NMAX]
    w = np.where(ev >= tau, np.clip(ev - 1 + 0.1, 0, None), 0.0)
    tot = w.sum(1, keepdims=True)
    f = np.where(tot > 0, w / np.maximum(tot, 1e-9) * B * np.minimum(1.0, tot), 0.0)
    out = np.zeros((len(rids), NMAX), np.float32)
    out[:, : f.shape[1]] = f
    return torch.tensor(out)


def baseline_single(pk, d_split: pd.DataFrame, col: str, tau: float, B: float) -> torch.Tensor:
    """基準（1 点買い）: EV=col×オッズ が最大の 1 頭に、EV≥τ のときだけ B を賭ける。col は p_b（混合）か p_win（市場なし）。"""
    rids = pk["rids"]
    cc = d_split.groupby("race_id").cumcount()
    ev = (d_split[col] * d_split["win_odds"]).groupby([d_split["race_id"], cc]).first().unstack().reindex(rids).fillna(-1).to_numpy()[:, :NMAX]
    out = np.zeros((len(rids), NMAX), np.float32)
    j = ev.argmax(1)
    ok = ev[np.arange(len(rids)), j] >= tau
    out[np.arange(len(rids))[ok], j[ok]] = B
    return torch.tensor(out)


HP_SPACE = {"hidden": [0, 8, 16, 32], "skip": [0.0, 0.5, 1.0, 2.0], "forced": [0, 1], "lr": [3e-4, 1e-3, 3e-3], "wd": [1e-5, 1e-4, 1e-3], "B": [0.05, 0.1, 0.2], "ent": [0.0, 0.01, 0.05], "risk": [0.0, 1.0, 5.0], "drop": [0.0, 0.1, 0.3]}


def sample_hp(rng) -> dict:
    return {k: (v[int(rng.integers(len(v)))]) for k, v in HP_SPACE.items()}


def perturb(hp: dict, rng) -> dict:
    h = dict(hp)
    for k in rng.choice(list(HP_SPACE), size=2, replace=False):
        vals = HP_SPACE[k]
        i = vals.index(h[k]) + int(rng.choice([-1, 1]))
        h[k] = vals[min(max(i, 0), len(vals) - 1)]
    return h


def league(tr, va, K: int = 12, gens: int = 3, epochs: int = 8, seed: int = 0, single: int = 0) -> list[dict]:
    """集団ベースの進化: 方策の集団が同じ過去レースで競い（検証年の対数成長）、下位は上位のコピーを摂動して置き換える。"""
    rng = np.random.default_rng(seed)
    pop = []
    for k in range(K):
        hp = {**sample_hp(rng), "single": single}
        pop.append({"hp": hp, "net": train(tr, hp, epochs, seed * 100 + k)})
    for g in range(gens):
        for p in pop:
            p["score"] = float(growth(alloc(p["net"], va, p["hp"]["B"]), va["RET"], va["M"]).mean())
        pop.sort(key=lambda p: -p["score"])
        for j in range(K // 4):                          # 下位 25% を入れ替え
            src = pop[j]
            hp = {**perturb(src["hp"], rng), "single": single}
            net = copy.deepcopy(src["net"]) if (hp["hidden"], hp["forced"]) == (src["hp"]["hidden"], src["hp"]["forced"]) else None
            pop[-1 - j] = {"hp": hp, "net": train(tr, hp, epochs, seed * 100 + 50 + g * 10 + j, net=net)}
    for p in pop:
        p["score"] = float(growth(alloc(p["net"], va, p["hp"]["B"]), va["RET"], va["M"]).mean())
    pop.sort(key=lambda p: -p["score"])
    return pop


def exp3_mix(pop: list[dict], va, topn: int = 6, eta: float = 5.0) -> np.ndarray:
    """乗法的重み: 検証年のレースを時系列順に再生し、各方策の対数成長で重みを更新（検証年でだけ学習し、以後は固定）。"""
    w = np.ones(topn)
    for r in range(va["X"].shape[0]):
        gs = []
        for p in pop[:topn]:
            f = alloc(p["net"], {"X": va["X"][r:r + 1], "M": va["M"][r:r + 1]}, p["hp"]["B"])
            gs.append(float(growth(f, va["RET"][r:r + 1], va["M"][r:r + 1])))
        w = w * np.exp(eta * np.array(gs))
        w = w / w.sum()
    return w


def critic_fit(d_tr: pd.DataFrame):
    import lightgbm as lgb

    X = d_tr[FEATS]
    y = d_tr["ret_win"].to_numpy() / 100.0
    m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, num_leaves=8, min_child_samples=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=10, verbose=-1)
    m.fit(X, y)
    return m


def split_years(d: pd.DataFrame, eval_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """学習 = 評価年の 2 年前まで、方策の選択 = 前年、評価 = 評価年。評価年以降のデータは学習・選択に入らない。"""
    return d[d["year"] <= eval_year - 2], d[d["year"] == eval_year - 1], d[d["year"] == eval_year]


def arm_results(tag: str, pop: list[dict], tr_d, te_d, tr, va, te, crit: np.ndarray, single: int) -> dict:
    """1 つのアーム（制限なし or 1 点制限）のリーグ戦結果: 最上位・乗法的重みの混合・批評役との併用・多目的の一覧。"""
    res: dict = {}
    post = restrict_single if single else (lambda f: f)
    res[f"{tag}league_top1"] = evaluate(post(alloc(pop[0]["net"], te, pop[0]["hp"]["B"])), te)
    res[f"{tag}league_top1_hp"] = pop[0]["hp"]
    res[f"{tag}league_val_scores"] = [round(p["score"], 5) for p in pop]
    w = exp3_mix(pop, va, topn=min(6, len(pop)))
    fmix = post(sum(w[i] * alloc(pop[i]["net"], te, pop[i]["hp"]["B"]) for i in range(len(w))))
    res[f"{tag}league_exp3_mix"] = evaluate(fmix, te)
    res[f"{tag}league_exp3_weights"] = [round(float(x), 3) for x in w]
    top = post(alloc(pop[0]["net"], te, pop[0]["hp"]["B"]))
    for th in (0.9, 1.0, 1.1):
        gate = torch.tensor((crit >= th).astype(np.float32))
        res[f"{tag}league_top1_x_critic{th}"] = evaluate(top * gate, te)
        res[f"{tag}mix_x_critic{th}"] = evaluate(fmix * gate, te)
    res[f"{tag}league_members_test"] = [{"hp": p["hp"], "val_growth": round(p["score"], 5),
                                        **{k: v for k, v in evaluate(post(alloc(p["net"], te, p["hp"]["B"])), te, n_boot=300).items() if k in ("roi", "bet_hit", "wealth", "max_dd", "n_bets")}} for p in pop[:8]]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TP / "rl.json"))
    ap.add_argument("--frame", default=str(TP / "edge_frame.parquet"))
    ap.add_argument("--eval-years", nargs="+", type=int, default=[2024, 2025])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(4)
    d = pd.read_parquet(a.frame)
    d = features(d[d["year"] >= 2022].sort_values(["date", "race_id", "horse_number"]).reset_index(drop=True))
    out: dict = {"eval": {}, "n_policies_tried": 0}
    for Y in a.eval_years:
        tr_d, va_d, te_d = split_years(d, Y)
        tr = pack(tr_d)
        va = pack(va_d, tr["mu"], tr["sd"])
        te = pack(te_d, tr["mu"], tr["sd"])
        res: dict = {}
        # 基準（複数点）
        for tau, B in [(0.95, 0.05), (1.0, 0.05), (1.05, 0.1)]:
            res[f"rule_blend_ev{tau}_B{B}"] = evaluate(baseline_blend(te, te_d, tau, B), te)
        # 基準（1 点買い）: 混合 EV 最大の 1 頭 / 市場なし EV 最大の 1 頭
        for tau in (0.0, 1.0, 1.1, 1.2):
            res[f"rule_single_mixed_ev{tau}_B0.05"] = evaluate(baseline_single(te, te_d, "p_b", tau, 0.05), te)
        res["rule_single_model_B0.02"] = evaluate(baseline_single(te, te_d, "p_win", 0.0, 0.02), te)
        # 対照: 人気 1 位に常に 5%
        fav = np.zeros((te["X"].shape[0], NMAX), np.float32)
        rank_idx = te["X"][..., FEATS.index("f_rank_q")]
        j = torch.where(te["M"], rank_idx, torch.full_like(rank_idx, 1e9)).argmin(1)
        fav[np.arange(len(fav)), j.numpy()] = 0.05
        res["rule_favorite_B0.05"] = evaluate(torch.tensor(fav), te)
        # 単一エージェント（制限なし / 1 点制限）
        hp0 = {"hidden": 16, "skip": 1.0, "forced": 0, "single": 0, "lr": 1e-3, "wd": 1e-4, "B": 0.1, "ent": 0.01, "risk": 0.0, "drop": 0.1}
        res["free_single_agent"] = evaluate(alloc(train(tr, hp0, 25, seed=0), te, hp0["B"]), te)
        hp1 = {**hp0, "single": 1}
        res["single_single_agent"] = evaluate(alloc(train(tr, hp1, 25, seed=0), te, hp1["B"]), te)
        out["n_policies_tried"] += 2
        # 批評役
        cm = critic_fit(tr_d)
        te_d2 = te_d.copy()
        te_d2["critic"] = cm.predict(te_d2[FEATS])
        crit = np.zeros((te["X"].shape[0], NMAX), np.float32)
        ci = te_d2.groupby("race_id").cumcount().to_numpy()
        ri = te_d2["race_id"].map({r: i for i, r in enumerate(te["rids"])}).to_numpy()
        ok = ci < NMAX
        crit[ri[ok], ci[ok]] = te_d2.loc[ok, "critic"].to_numpy()
        # リーグ戦（制限なし / 1 点制限）
        for single, tag in ((0, "free_"), (1, "single_")):
            pop = league(tr, va, K=a.k, gens=a.gens, epochs=a.epochs, seed=Y + 1000 * single, single=single)
            out["n_policies_tried"] += a.k + a.k // 4 * a.gens
            res.update(arm_results(tag, pop, tr_d, te_d, tr, va, te, crit, single))
        out["eval"][str(Y)] = res
        print("year", Y, "done", flush=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()

"""人気の順を基準にした比べ（ユーザーの目的: 人気馬をそのまま買うより、血統の予測で選ぶ方が優位か。開発の行・時間の順・記述）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.pop_compare

- 行: 時間の順の予測（oof_v4_pure: 年 t を t より前で学ぶ）がある 2020〜24 年の開発の行（split 0・1）で、全頭に予測があるレース。
- 選び方（各レースの 1 番手と上位 3 頭）:
  (i) 1 番人気（確定オッズ）・1 番人気（T-60 の再現オッズ。5 回それぞれで順位と払戻を出して平均。critic の指摘 1）
  (ii) 血統だけのモデル（時間の順の p_win）
  (iii) 市場 ＋ 血統（Benter の 2 段: α log p + β log q。年 t の α・β は 2019〜t−1 年で当てる）。q は確定と T-60 の再現の 2 つ。
- 物差し: 1 番手の勝率・3 着内率、上位 3 頭の 3 着内率、1 番手を単勝・複勝で 100 円ずつ買った回収率（payoff の確定の払戻）。
- 基準線: 1 番人気を全部買う・同じオッズの帯（確定オッズの 6 帯）を全部買う。帯の平均はブートストラップの回ごとに作り直す（critic の指摘 3・4）。
- 区間は開催で束ねたブートストラップ 2,000 回。T-60 の再現の揺れの元は 2026 年 10 月以降の T-60 と確定のオッズ（着順は読まない）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import build_base, data, pre_odds
from src.research.pedigree_lab.hyp_tests import clogit

BANDS = [0, 3, 5, 10, 20, 50, 1e9]


def payouts() -> tuple[dict, dict]:
    r = build_base.load_runs()[["race_id", "payoff"]].drop_duplicates("race_id")
    win, pl = {}, {}
    for rid, p in zip(r["race_id"], r["payoff"]):
        try:
            d = json.loads(p) if isinstance(p, str) else {}
        except Exception:
            continue
        w = d.get("単勝")
        ws = w if isinstance(w, list) else ([w] if w else [])
        win[rid] = {int(x["numbers"]): float(str(x["payout"]).replace(",", "")) for x in ws if str(x.get("numbers", "")).isdigit()}
        pl[rid] = {int(x["numbers"]): float(str(x["payout"]).replace(",", "")) for x in (d.get("複勝") or []) if str(x.get("numbers", "")).isdigit()}
    return win, pl


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "pop_compare",
                                   "人気の順を基準にした比べ（1 番手の的中と回収率の記述・合否にしない・critic の直し）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "win", "top3", "odds", "horse_number", "n_run"])
    WF = pd.read_parquet(data.OUT / "oof_v4_pure.parquet")[["race_id", "horse_id", "p_win"]]
    S = X[X["split"].isin([0, 1])].merge(WF, on=["race_id", "horse_id"], how="inner")
    S = S[S["odds"] > 0]
    S = S[S.groupby("race_id")["horse_id"].transform("size") == S["n_run"]].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    rid = S["race_id"].to_numpy()
    yr = S["date"].dt.year.to_numpy()
    inv = 1 / S["odds"].to_numpy(float)
    q = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
    pool = pre_odds.load_drift("t60")
    rk = S.groupby("race_id")["odds"].rank(method="first").to_numpy()
    sims = pre_odds.simulate(rid, S["odds"].to_numpy(float), rk, pool, n=5)
    q60s = [(1 / o) / pd.Series(1 / o).groupby(rid).transform("sum").to_numpy() for o in sims]
    lp = np.log(np.clip(S["p_win"].to_numpy(), 1e-6, 1))
    y = S["win"].to_numpy(float)

    def benter(qv):
        lq = np.log(np.clip(qv, 1e-6, 1))
        sc = np.full(len(S), np.nan)
        for t in range(2020, 2025):
            tr, te = (yr >= 2019) & (yr < t), yr == t
            b = clogit(np.c_[lp[tr], lq[tr]], y[tr], pd.factorize(rid[tr])[0])
            sc[te] = b[0] * lp[te] + b[1] * lq[te]
        return sc
    # 選び方ごとの点（T-60 の 2 つは 5 回分）
    scores = {"1 番人気（確定）": [-S["odds"].to_numpy(float)], "1 番人気（T-60 の再現）": [-o for o in sims],
              "血統だけ": [S["p_win"].to_numpy()], "市場（確定）＋ 血統": [benter(q)], "市場（T-60）＋ 血統": [benter(qq) for qq in q60s]}
    keep = yr >= 2020
    S = S[keep].reset_index(drop=True)
    scores = {k: [v[keep] for v in vs] for k, vs in scores.items()}
    rid = S["race_id"].to_numpy()
    wp, pp = payouts()
    S["win_pay"] = [wp.get(r, {}).get(int(h), 0.0) for r, h in zip(S["race_id"], S["horse_number"])]
    S["place_pay"] = [pp.get(r, {}).get(int(h), 0.0) for r, h in zip(S["race_id"], S["horse_number"])]
    band = np.digitize(S["odds"].to_numpy(float), BANDS[1:-1])            # 確定オッズの 6 帯
    mt = S["race_id"].str[:8].to_numpy()
    mu, minv = np.unique(mt, return_inverse=True)
    rng = np.random.default_rng(12)
    Wb = np.stack([np.bincount(rng.integers(0, len(mu), len(mu)), minlength=len(mu)) for _ in range(2000)]).astype(float)
    nb = len(BANDS) - 1
    # 帯 × 開催の和（帯の基準線を回ごとに作り直すため）
    MB = {}
    for col in ("win_pay", "place_pay"):
        v = S[col].to_numpy()
        sm = np.zeros((len(mu), nb)); cn = np.zeros((len(mu), nb))
        np.add.at(sm, (minv, band), v); np.add.at(cn, (minv, band), 1)
        MB[col] = (sm, cn)

    def stats(mask_list):
        """mask_list: 5 回分（または 1 本）の 1 番手の印。回収率・帯との差を、回ごとに平均した点と区間で返す。"""
        out = {}
        for col in ("win_pay", "place_pay"):
            pts, bs, dpts, dbs = [], [], [], []
            sm, cn = MB[col]
            bmean = (Wb @ sm) / np.maximum(Wb @ cn, 1)                       # [B, 帯]
            bmean0 = sm.sum(0) / np.maximum(cn.sum(0), 1)
            for mk in mask_list:
                n_ = np.bincount(minv[mk], minlength=len(mu)).astype(float)
                s_ = np.bincount(minv[mk], weights=S[col].to_numpy()[mk], minlength=len(mu))
                pb = np.zeros((len(mu), nb)); np.add.at(pb, (minv[mk], band[mk]), 1)
                roi_b = (Wb @ s_) / (Wb @ n_)
                base_b = ((Wb @ pb) * bmean).sum(1) / (Wb @ n_)
                pts.append(s_.sum() / n_.sum()); bs.append(roi_b)
                dpts.append(s_.sum() / n_.sum() - (pb.sum(0) * bmean0).sum() / n_.sum()); dbs.append(roi_b - base_b)
            k = "win" if col == "win_pay" else "place"
            out[f"{k}_roi"] = [round(float(np.mean(pts)), 1), np.percentile(np.mean(bs, 0), [2.5, 97.5]).round(1).tolist()]
            out[f"{k}_minus_same_odds_band"] = [round(float(np.mean(dpts)), 1), np.percentile(np.mean(dbs, 0), [2.5, 97.5]).round(1).tolist()]
        return out
    res = {"rows": int(len(S)), "races": int(S["race_id"].nunique()), "years": "2020〜2024（開発の行・時間の順）", "table": {}}
    top1 = {}
    fav = (S.groupby("race_id")["odds"].rank(method="first") == 1).to_numpy()
    for nm, vs in scores.items():
        ms, m3 = [], []
        for v in vs:
            r1 = pd.Series(v).groupby(rid).rank(ascending=False, method="first").to_numpy()
            ms.append(r1 == 1); m3.append(r1 <= 3)
        top1[nm] = ms
        row = {"top1_win": round(float(np.mean([S.loc[m, "win"].mean() for m in ms])), 4),
               "top1_top3": round(float(np.mean([S.loc[m, "top3"].mean() for m in ms])), 4),
               "top3_top3": round(float(np.mean([S.loc[m, "top3"].mean() for m in m3])), 4),
               "share_not_final_fav": round(float(np.mean([(m & ~fav).sum() / m.sum() for m in ms])), 4)}
        row |= stats(ms)
        res["table"][nm] = row
        print(nm, row, flush=True)
    # (iii) − 1 番人気（同じオッズの見方どうし）と、血統だけ − 1 番人気（確定）
    for bn, fn in (("市場（確定）＋ 血統", "1 番人気（確定）"), ("市場（T-60）＋ 血統", "1 番人気（T-60 の再現）"), ("血統だけ", "1 番人気（確定）")):
        out = {}
        for col in ("win_pay", "place_pay"):
            v = S[col].to_numpy()
            pts, bs = [], []
            for b1, f1 in zip(top1[bn] if len(top1[bn]) > 1 else top1[bn] * len(top1[fn]), top1[fn] if len(top1[fn]) > 1 else top1[fn] * len(top1[bn])):
                d = pd.Series(np.where(b1, v, 0.0) - np.where(f1, v, 0.0)).groupby(mt).sum()
                n_ = pd.Series(np.where(f1, 1.0, 0.0)).groupby(mt).sum()
                d, n_ = d.reindex(mu).to_numpy(), n_.reindex(mu).to_numpy()
                pts.append(d.sum() / n_.sum()); bs.append((Wb @ d) / (Wb @ n_))
            out["win" if col == "win_pay" else "place"] = [round(float(np.mean(pts)), 2), np.percentile(np.mean(bs, 0), [2.5, 97.5]).round(2).tolist()]
        res[f"roi_diff|{bn}|{fn}"] = out
        print("diff", bn, fn, out, flush=True)
    path = data.OUT / "pop_compare.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()

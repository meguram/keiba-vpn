"""馬券最適化（強化学習）の環境: レースごとの候補の買い目（券種 × 組）に、モデル確率・市場オッズ・実際の払戻を付けて並べる。

- 券種: 単勝・複勝・馬連・ワイド・馬単（三連複・三連単はオッズが無く、払戻だけあるため対象外）
- 確率: モデルの勝率・連対率・複勝率（tp-05/06）から割引 Harville の着順分布（``exotic_probabilities``）で各組の確率を出す。
  市場オッズ（確定オッズ）の暗黙の確率とのブレンド（Benter 型: p ∝ p_model^a · p_market^b）も作る。
- 判断に使うオッズ: 単勝・馬連・馬単はそのもの、複勝・ワイドは下限（保守的）。実際の払戻（報酬）は払戻表の確定値（的中した組のみ）。
- 注意: 確定オッズは締切後の値で、購入判断の時点（T-15/T-60）のオッズとは違う。回収率は楽観側に偏りうる。
"""

from __future__ import annotations

from itertools import combinations, permutations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.pipeline.inference.exotic_probabilities import RankDistribution, fit_rank_distribution
from src.pipeline.models.win_models import normalize_to_total

TYPES = ["tansho", "fukusho", "umaren", "wide", "umatan"]
TYPE_IDX = {t: i for i, t in enumerate(TYPES)}
EPS = 1e-12


def market_probs(odds: np.ndarray) -> np.ndarray:
    """単勝オッズの暗黙の確率（控除を除いて合計 1 にそろえる）。"""
    inv = 1.0 / np.clip(odds, 1.0, None)
    return inv / inv.sum()


def benter_blend(log_pm: np.ndarray, log_pk: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * log_pm + b * log_pk
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def fit_benter(groups: list[tuple[np.ndarray, np.ndarray, int]], x0=(0.5, 0.5)) -> tuple[float, float]:
    """groups: [(log p_model, log p_market, 勝ち馬の位置)] のリスト。勝ち馬の対数尤度が最大になる (a, b)。"""
    def nll(ab):
        a, b = ab
        s = 0.0
        for lm, lk, w in groups:
            z = a * lm + b * lk
            m = z.max()
            s -= z[w] - m - np.log(np.exp(z - m).sum())
        return s / len(groups)

    r = minimize(nll, np.array(x0), method="L-BFGS-B", bounds=[(0.0, 2.0), (0.0, 2.0)])
    return float(r.x[0]), float(r.x[1])


def _tickets_for_race(hn: np.ndarray, dist: RankDistribution, dist_model: RankDistribution, dist_mkt: RankDistribution, odds: dict, K: int, payout: dict) -> list[dict]:
    """1 レースの候補の買い目。dist = 判断に使う確率（ブレンド後）、dist_model = モデル単独の確率。"""
    n = len(hn)
    rows: list[dict] = []

    def add(t, combo, pc, pm, pk, o):
        if o is None or not np.isfinite(o) or o <= 0 or pc <= 0:
            return
        rows.append({"type": t, "combo": combo, "p_comb": pc, "p_model": pm, "p_mkt": pk, "odds": o, "mult": payout.get((t, combo), 0.0)})

    t3c, t3m, t3k = dist.top3(), dist_model.top3(), dist_mkt.top3()
    wide_c, wide_m, wide_k = dist.all_wide(), dist_model.all_wide(), dist_mkt.all_wide()
    for i in range(n):
        h = int(hn[i])
        add("tansho", str(h), float(dist.win[i]), float(dist_model.win[i]), float(dist_mkt.win[i]), odds["win"].get(h))
        add("fukusho", str(h), float(t3c[i]), float(t3m[i]), float(t3k[i]), odds["place_min"].get(h))
    if n >= 2:
        for i, j in combinations(range(n), 2):
            a, b = sorted((int(hn[i]), int(hn[j])))
            c = f"{a}-{b}"
            add("umaren", c, dist.umaren(i, j), dist_model.umaren(i, j), dist_mkt.umaren(i, j), odds["umaren"].get(c))
            if n >= 3:
                add("wide", c, wide_c[(i, j)], wide_m[(i, j)], wide_k[(i, j)], odds["wide_min"].get(c))
        for i, j in permutations(range(n), 2):
            c = f"{int(hn[i])}>{int(hn[j])}"
            add("umatan", c, dist.umatan(i, j), dist_model.umatan(i, j), dist_mkt.umatan(i, j), odds["umatan"].get(c))
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    df["ev"] = df["p_comb"] * df["odds"] - 1.0
    keep = []
    for t, g in df.groupby("type"):
        top_p = g.nlargest(K, "p_comb").index
        top_ev = g.nlargest(max(1, K // 2), "ev").index
        top_x = g.assign(_x=g["p_mkt"] * g["odds"]).nlargest(max(1, K // 2), "_x").index       # 市場の暗黙確率に対して配当が高い順（券種間の整合ずれ）
        keep += list(top_p) + list(top_ev) + list(top_x)
    return df.loc[sorted(set(keep))].to_dict("records")


def impute_missing_runners(pred: pd.DataFrame, win_place: pd.DataFrame) -> pd.DataFrame:
    """オッズ表にいるが予測に無い馬（中止・失格などで着順が無く学習データから外れた馬）に、市場の確率から作った予測を補う。
    買う時点ではその馬が中止するかは分からないので、除くと結果を知った上でレースを選ぶことになる（楽観側の偏り）。
    p_win = 市場の暗黙の確率、p_top2・p_top3 = 予測全体での p_win との対応（ビンの平均）から。補ったあとレース内の合計を 1・2・3 にそろえる。"""
    key = ["race_id", "horse_id"]
    w = win_place[(win_place["win_odds"] > 0) & win_place["race_id"].isin(pred["race_id"].unique())][key + ["horse_number", "win_odds"]].copy()
    w["q"] = (1.0 / np.clip(w["win_odds"], 1.0, None)) / (1.0 / np.clip(w["win_odds"], 1.0, None)).groupby(w["race_id"]).transform("sum")
    miss = w.merge(pred[key].assign(_h=1), on=key, how="left")
    miss = miss[miss["_h"].isna()]
    if miss.empty:
        return pred
    b = pd.qcut(pred["p_win"], 20, duplicates="drop").rename("bin")
    g = pred.groupby(b, observed=True)[["p_win", "p_top2", "p_top3"]].mean().sort_values("p_win")
    add = pd.DataFrame({"race_id": miss["race_id"].to_numpy(), "horse_id": miss["horse_id"].to_numpy(), "horse_number": miss["horse_number"].to_numpy(),
                        "p_win": miss["q"].to_numpy(), "p_top2": np.interp(miss["q"], g["p_win"], g["p_top2"]), "p_top3": np.interp(miss["q"], g["p_win"], g["p_top3"])})
    if "year" in pred.columns:
        add["year"] = pred.groupby("race_id")["year"].first().reindex(add["race_id"]).to_numpy()
    out = pd.concat([pred, add], ignore_index=True)
    hit = out["race_id"].isin(set(add["race_id"]))
    sub = out[hit].copy()
    for c, tot in (("p_win", 1.0), ("p_top2", 2.0), ("p_top3", 3.0)):
        sub[c] = normalize_to_total(sub["race_id"].to_numpy(), sub[c].to_numpy(), tot)
    out.loc[hit, ["p_win", "p_top2", "p_top3"]] = sub[["p_win", "p_top2", "p_top3"]].to_numpy()
    return out


def build_race_tickets(pred: pd.DataFrame, win_place: pd.DataFrame, pair: pd.DataFrame, payoff: pd.DataFrame, a: float, b: float, K: int = 12,
                       alphas: tuple[float, float] | None = None, mkt_alphas: tuple[float, float] = (1.0, 1.0)) -> pd.DataFrame:
    """pred: race_id, horse_number, p_win, p_top2, p_top3（レース内で合計 1・2・3）。他は整備済みの表。
    戻り値: 候補の買い目（race_id, type, combo, p_comb, p_model, odds, mult, ev, race 特徴）。"""
    if "horse_number" not in pred.columns:
        pred = pred.merge(win_place[["race_id", "horse_id", "horse_number"]].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="inner")
    pred = impute_missing_runners(pred, win_place)
    wp = {rid: g for rid, g in win_place[win_place["win_odds"] > 0].groupby("race_id")}      # 取消・除外は win_odds = -3.0（人気 9999）。出走しないので除く
    pr = {rid: g for rid, g in pair.groupby("race_id")}
    po = {rid: g for rid, g in payoff.groupby("race_id")}
    out: list[pd.DataFrame] = []
    for rid, g in pred.groupby("race_id", sort=False):
        w = wp.get(rid)
        if w is None or len(g) < 3:
            continue
        g = g.sort_values("horse_number")
        w = w.drop_duplicates("horse_number").set_index("horse_number")
        hn = g["horse_number"].astype(int).to_numpy()
        if set(hn) != set(w.index):
            continue
        win_odds = w.loc[hn, "win_odds"].to_numpy(dtype=float)
        pm = g[["p_win"]].to_numpy().ravel()
        pk = market_probs(win_odds)
        win_c = benter_blend(np.log(np.clip(pm, EPS, None)), np.log(pk), a, b)
        t2, t3 = g["p_top2"].to_numpy(), g["p_top3"].to_numpy()
        dist_m, _ = fit_rank_distribution(pm / pm.sum(), t2, t3)
        al2, al3 = (dist_m.alpha2, dist_m.alpha3) if alphas is None else alphas
        dist_c = RankDistribution(win_c, al2, al3)
        dist_k = RankDistribution(pk, mkt_alphas[0], mkt_alphas[1])      # 市場の単勝確率だけから作った着順分布（券種間の整合の比較用）
        odds = {"win": dict(zip(hn, win_odds)), "place_min": dict(zip(hn, w.loc[hn, "place_odds_min"].to_numpy(dtype=float)))}
        for t, col, name in (("umaren", "odds", "umaren"), ("wide", "odds_min", "wide_min"), ("umatan", "odds", "umatan")):
            sub = pr.get(rid)
            sub = sub[sub["bet_type"] == t] if sub is not None else None
            odds[name] = dict(zip(sub["combo"], sub[col].astype(float))) if sub is not None and len(sub) else {}
        pay = {}
        p_rows = po.get(rid)
        if p_rows is not None:
            ok = p_rows[p_rows["bet_type"].isin(TYPES) & p_rows["combo"].notna() & p_rows["payout"].notna()]
            for t, c, v in zip(ok["bet_type"], ok["combo"], ok["payout"]):
                pay[(t, str(c))] = float(v) / 100.0
        rows = _tickets_for_race(hn, dist_c, dist_m, dist_k, odds, K, pay)
        if not rows:
            continue
        d = pd.DataFrame(rows)
        d.insert(0, "race_id", rid)
        d["field_size"] = len(hn)
        d["win_entropy"] = float(-(win_c * np.log(np.clip(win_c, EPS, None))).sum())
        d["win_max"] = float(win_c.max())
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _build_chunk(args):
    return build_race_tickets(*args)


def build_parallel(pred: pd.DataFrame, win_place: pd.DataFrame, pair: pd.DataFrame, payoff: pd.DataFrame, a: float, b: float, K: int = 12,
                   procs: int = 4, chunks: int = 24, mkt_alphas: tuple[float, float] = (1.0, 1.0)) -> pd.DataFrame:
    """build_race_tickets をレースの塊に分けて並列に実行する（procs を超えない）。"""
    from concurrent.futures import ProcessPoolExecutor

    rids = pred["race_id"].unique()
    jobs = []
    for ch in np.array_split(rids, max(1, min(chunks, len(rids)))):
        s = set(ch)
        jobs.append((pred[pred["race_id"].isin(s)], win_place[win_place["race_id"].isin(s)], pair[pair["race_id"].isin(s)], payoff[payoff["race_id"].isin(s)], a, b, K, None, mkt_alphas))
    with ProcessPoolExecutor(procs) as ex:
        parts = [p for p in ex.map(_build_chunk, jobs) if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def race_dists(pred: pd.DataFrame, win_place: pd.DataFrame, a: float, b: float) -> dict[str, tuple]:
    """build_race_tickets と同じ手順で、レースごとの（馬番, ブレンド後の勝率, α2, α3）を返す（的中確率の厳密計算用）。"""
    if "horse_number" not in pred.columns:
        pred = pred.merge(win_place[["race_id", "horse_id", "horse_number"]].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="inner")
    pred = impute_missing_runners(pred, win_place)
    wp = {rid: g for rid, g in win_place[win_place["win_odds"] > 0].groupby("race_id")}
    out = {}
    for rid, g in pred.groupby("race_id", sort=False):
        w = wp.get(rid)
        if w is None or len(g) < 3:
            continue
        g = g.sort_values("horse_number")
        w = w.drop_duplicates("horse_number").set_index("horse_number")
        hn = g["horse_number"].astype(int).to_numpy()
        if set(hn) != set(w.index):
            continue
        pm = g["p_win"].to_numpy()
        pk = market_probs(w.loc[hn, "win_odds"].to_numpy(dtype=float))
        win_c = benter_blend(np.log(np.clip(pm, EPS, None)), np.log(pk), a, b)
        dist_m, _ = fit_rank_distribution(pm / pm.sum(), g["p_top2"].to_numpy(), g["p_top3"].to_numpy())
        out[rid] = (hn, win_c, float(dist_m.alpha2), float(dist_m.alpha3))
    return out


def _dists_chunk(args):
    return race_dists(*args)


def race_dists_parallel(pred: pd.DataFrame, win_place: pd.DataFrame, a: float, b: float, procs: int = 4, chunks: int = 24) -> dict[str, tuple]:
    from concurrent.futures import ProcessPoolExecutor

    rids = pred["race_id"].unique()
    jobs = []
    for ch in np.array_split(rids, max(1, min(chunks, len(rids)))):
        s = set(ch)
        jobs.append((pred[pred["race_id"].isin(s)], win_place[win_place["race_id"].isin(s)], a, b))
    out: dict = {}
    with ProcessPoolExecutor(procs) as ex:
        for part in ex.map(_dists_chunk, jobs):
            out.update(part)
    return out


def _build_pred_race(hn, p_model, dist_m, dist_k, pred_win, pred_place, c, payout, K):
    """予測オッズだけで判断する 1 レース分の買い目。odds = 予測オッズ（判断用）、mult = 実際の払戻倍率（精算用）。"""
    n = len(hn)
    rows = []

    def add(t, combo, pc, pk, o):
        if not np.isfinite(o) or o <= 0 or pc <= 0:
            return
        rows.append({"type": t, "combo": combo, "p_comb": pc, "p_model": pc, "p_mkt": pk, "odds": float(o), "mult": payout.get((t, combo), 0.0)})

    t3m, t3k = dist_m.top3(), dist_k.top3()
    wide_m, wide_k = dist_m.all_wide(), dist_k.all_wide()
    for i in range(n):
        h = int(hn[i])
        add("tansho", str(h), float(dist_m.win[i]), float(dist_k.win[i]), pred_win[i])
        add("fukusho", str(h), float(t3m[i]), float(t3k[i]), pred_place[i])
    for i, j in combinations(range(n), 2):
        a_, b_ = sorted((int(hn[i]), int(hn[j])))
        cmb = f"{a_}-{b_}"
        pk = dist_k.umaren(i, j)
        add("umaren", cmb, dist_m.umaren(i, j), pk, c["umaren"] / max(pk, 1e-9))
        pkw = wide_k[(i, j)]
        add("wide", cmb, wide_m[(i, j)], pkw, c["wide"] / max(pkw, 1e-9))
    for i, j in permutations(range(n), 2):
        pk = dist_k.umatan(i, j)
        add("umatan", f"{int(hn[i])}>{int(hn[j])}", dist_m.umatan(i, j), pk, c["umatan"] / max(pk, 1e-9))
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    df["ev"] = df["p_comb"] * df["odds"] - 1.0
    keep = []
    for t, g in df.groupby("type"):
        keep += list(g.nlargest(K, "p_comb").index) + list(g.nlargest(max(1, K // 2), "ev").index)
    return df.loc[sorted(set(keep))].to_dict("records")


def build_tickets_predicted(pred: pd.DataFrame, win_place: pd.DataFrame, payoff: pd.DataFrame, podds: pd.DataFrame, consts: dict, mkt_alphas: tuple[float, float], K: int = 12) -> pd.DataFrame:
    """予測オッズで判断する買い目の表。pred = モデルの p_win・p_top2・p_top3、podds = race_id・horse_id・win_pred・place_pred（予測オッズ。そのレースの実際のオッズは使っていない）。
    win_place は出走馬の確認（取消・除外の除外）と馬番の対応にだけ使い、オッズの値は使わない。精算用の払戻 mult だけが実際の結果。"""
    wp = win_place[win_place["win_odds"] > 0][["race_id", "horse_id", "horse_number"]].drop_duplicates(["race_id", "horse_id"])
    d = pred.drop(columns=["horse_number"], errors="ignore").merge(wp, on=["race_id", "horse_id"], how="inner").merge(podds[["race_id", "horse_id", "win_pred", "place_pred"]], on=["race_id", "horse_id"], how="inner")
    runners = {rid: set(g["horse_number"]) for rid, g in wp.groupby("race_id")}
    po = {rid: g for rid, g in payoff.groupby("race_id")}
    out = []
    for rid, g in d.groupby("race_id", sort=False):
        if len(g) < 3 or set(g["horse_number"]) != runners.get(rid):
            continue
        g = g.sort_values("horse_number")
        hn = g["horse_number"].astype(int).to_numpy()
        pm = g["p_win"].to_numpy()
        dist_m, _ = fit_rank_distribution(pm / pm.sum(), g["p_top2"].to_numpy(), g["p_top3"].to_numpy())
        q = (1.0 / g["win_pred"].to_numpy()); q = q / q.sum()
        dist_k = RankDistribution(q, mkt_alphas[0], mkt_alphas[1])
        pay = {}
        pr = po.get(rid)
        if pr is not None:
            ok = pr[pr["bet_type"].isin(TYPES) & pr["combo"].notna() & pr["payout"].notna()]
            for t, cm, v in zip(ok["bet_type"], ok["combo"], ok["payout"]):
                pay[(t, str(cm))] = float(v) / 100.0
        rows = _build_pred_race(hn, pm, dist_m, dist_k, g["win_pred"].to_numpy(), g["place_pred"].to_numpy(), consts, pay, K)
        if not rows:
            continue
        r = pd.DataFrame(rows)
        r.insert(0, "race_id", rid)
        r["field_size"] = len(hn)
        w_c = dist_m.win
        r["win_entropy"] = float(-(w_c * np.log(np.clip(w_c, EPS, None))).sum())
        r["win_max"] = float(w_c.max())
        out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _pred_chunk(args):
    return build_tickets_predicted(*args)


def build_tickets_predicted_parallel(pred, win_place, payoff, podds, consts, mkt_alphas, K=12, procs=4, chunks=24) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor

    rids = pred["race_id"].unique()
    jobs = []
    for ch in np.array_split(rids, max(1, min(chunks, len(rids)))):
        s = set(ch)
        jobs.append((pred[pred["race_id"].isin(s)], win_place[win_place["race_id"].isin(s)], payoff[payoff["race_id"].isin(s)], podds[podds["race_id"].isin(s)], consts, mkt_alphas, K))
    with ProcessPoolExecutor(procs) as ex:
        parts = [p for p in ex.map(_pred_chunk, jobs) if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

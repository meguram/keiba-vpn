"""tp-30(4): 荒れ度に応じた馬券の幅（3連複のフォーメーション・流し・点数、券種、見送り）の評価。

python -m src.scripts.data.tp30_bet_eval prep   # v3 の OOF（2022〜2024）→ レースごとの順位・3連複の確率順位・払戻の対応を作る
python -m src.scripts.data.tp30_bet_eval eval   # 固定戦略と、荒れ度で幅を変える戦略の回収率・的中率・最大ドローダウン・点数。reality check つき

荒れ度の帯の閾値は『評価する年より前の年』の予測荒れ度の三分位で決める。確率は v3 の OOF（valid 2022〜2024）。
買い目の選択は予測確率の順位だけで決める（オッズは使わない）。馬券用の荒れ度（market）だけが購入前のオッズ構造（確定オッズで代用）を使う。
1 点 100 円。回収 = その組が的中したときの払戻（100 円あたり）。test 2025・2026 年は使わない。
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "data/local/modeling/training_pipeline"
OUT = WORK / "tp30"
VALID = [2022, 2023, 2024]
TOPK_CUM = 80


def _jsonable(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def prep() -> None:
    from src.pipeline.features import tp30_volatility as tv
    from src.pipeline.inference.exotic_probabilities import fit_rank_distribution

    t0 = time.time()
    o = pd.read_parquet(WORK / "oof/model_oof/ensemble_v3.parquet")
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    res = tv.load_results([2022, 2023, 2024])
    res = res[res["race_id"].isin(set(o["race_id"]))]
    info = res.groupby("race_id").agg(date=("date", "first"), payoff=("payoff", "first")).reset_index()
    hn = res.set_index(["race_id", "horse_id"])["horse_number"]
    fin = res.set_index(["race_id", "horse_id"])["finish_position"]
    o["horse_number"] = [hn.get((r, h), np.nan) for r, h in zip(o["race_id"], o["horse_id"])]
    o["finish_position"] = [fin.get((r, h), np.nan) for r, h in zip(o["race_id"], o["horse_id"])]
    o = o.dropna(subset=["horse_number"])
    vo = pd.read_parquet(OUT / "vol_oof.parquet").set_index("race_id")
    races = []
    for k, (rid, g) in enumerate(o.groupby("race_id", sort=False)):
        if len(g) < 8:
            continue
        pay = tv.parse_payoff(info.set_index("race_id")["payoff"].get(rid))
        if "sanrenpuku" not in pay:
            continue
        g = g.sort_values("horse_number")
        nums = g["horse_number"].astype(int).to_numpy()
        pw, p2, p3 = g["p_win"].to_numpy(float), g["p_top2"].to_numpy(float), g["p_top3"].to_numpy(float)
        try:
            dist, _ = fit_rank_distribution(pw, p2, p3)
        except Exception:
            continue
        trio = dist.all_sanrenpuku()
        keys = list(trio.keys())
        pr = np.array([trio[x] for x in keys])
        order = np.argsort(-pr)
        pos = {nums[i]: i for i in range(len(nums))}
        (wn, wpay) = pay["sanrenpuku"]
        wset = tuple(sorted(pos[x] for x in wn if x in pos))
        rank_of_winner = None
        if len(wset) == 3:
            idx = keys.index(wset)
            rank_of_winner = int(np.where(order == idx)[0][0]) + 1
        def map1(t):
            return tuple(sorted(pos[x] for x in t if x in pos))
        races.append({
            "race_id": rid, "date": str(info.set_index("race_id")["date"].get(rid)), "year": int(str(info.set_index("race_id")["date"].get(rid))[:4]), "n": len(g), "nums": nums,
            "p_win": pw, "p_top2": p2, "p_top3": p3,
            "rk_win": np.argsort(-pw), "rk_top2": np.argsort(-p2), "rk_top3": np.argsort(-p3),
            "trio_win": wset, "trio_pay": wpay, "trio_rank": rank_of_winner,
            "trio_cum": np.cumsum(pr[order])[:TOPK_CUM], "trio_keys_sorted": [keys[i] for i in order[:TOPK_CUM]],
            "tansho": (map1(pay["tansho"][0]), pay["tansho"][1]) if "tansho" in pay else None,
            "fukusho": [(map1(a), b) for a, b in pay.get("fukusho", [])], "umaren": (map1(pay["umaren"][0]), pay["umaren"][1]) if "umaren" in pay else None,
            "wide": [(map1(a), b) for a, b in pay.get("wide", [])],
            "vol_model": float(vo["model_vol"].get(rid, np.nan)), "vol_market": float(vo["market_vol"].get(rid, np.nan)),
        })
        if k % 2000 == 0:
            print(f"[{time.time() - t0:.0f}s] {k} races", flush=True)
    with open(OUT / "bet_races.pkl", "wb") as f:
        pickle.dump(races, f)
    print(f"[{time.time() - t0:.0f}s] 保存 {len(races)} レース", flush=True)


# ───────────────────────── 戦略
def _top(rk, m):
    return [int(x) for x in rk[:m]]


def cols_formation(r, a, b, c):
    from src.pipeline.features.tp30_volatility import formation_combos

    return formation_combos([_top(r["rk_win"], a), _top(r["rk_top2"], b), _top(r["rk_top3"], c)])


def cols_box(r, m):
    from src.pipeline.features.tp30_volatility import box_combos

    return box_combos(_top(r["rk_win"], m))


def cols_nagashi(r, partners):
    from src.pipeline.features.tp30_volatility import nagashi_combos

    return nagashi_combos(_top(r["rk_win"], 1), _top(r["rk_win"], 1 + partners))


def cols_cum(r, th):
    """累積確率で列を切るフォーメーション: 1 列目 = 勝率の累積が th[0] に達するまで（1〜3 頭）、2 列目 = 連対率の累積が th[1]（3〜7 頭）、3 列目 = 複勝率の累積が th[2]（5〜12 頭）。"""
    from src.pipeline.features.tp30_volatility import formation_combos

    def cut(rk, p, k, th_, lo, hi):
        pp = p[rk] / k
        c = np.cumsum(pp) / pp.sum()
        m = int(np.searchsorted(c, th_) + 1)
        return [int(x) for x in rk[:max(lo, min(hi, m))]]

    return formation_combos([cut(r["rk_win"], r["p_win"], 1.0, th[0], 1, 3), cut(r["rk_top2"], r["p_top2"], 2.0, th[1], 3, 7), cut(r["rk_top3"], r["p_top3"], 3.0, th[2], 5, 12)])


def cols_topn(r, n):
    """3連複の予測確率の上位 n 点。"""
    return set(r["trio_keys_sorted"][:n])


def settle(r, combos) -> tuple[int, float]:
    if not combos:
        return 0, 0.0
    return len(combos), float(r["trio_pay"]) if r["trio_win"] in combos else 0.0


def build_strategies():
    """名前 → (関数 r,band→組の集合)。band: 0 堅い / 1 中 / 2 荒れ。None は見送り。"""
    S: dict = {}
    # 固定
    for name, fn in {"F_box4": lambda r: cols_box(r, 4), "F_box5": lambda r: cols_box(r, 5), "F_box6": lambda r: cols_box(r, 6), "F_box7": lambda r: cols_box(r, 7),
                     "F_nag5": lambda r: cols_nagashi(r, 5), "F_nag7": lambda r: cols_nagashi(r, 7),
                     "F_form_2_5_7": lambda r: cols_formation(r, 2, 5, 7), "F_form_2_6_9": lambda r: cols_formation(r, 2, 6, 9), "F_form_3_7_11": lambda r: cols_formation(r, 3, 7, 11),
                     **{f"F_top{n}": (lambda n: (lambda r: cols_topn(r, n)))(n) for n in (6, 8, 10, 12, 15, 20, 25, 30, 40, 50)}}.items():
        S[name] = (lambda fn: (lambda r, band: fn(r)))(fn)
    # 荒れ度で幅を変える（band: 0 堅い 1 中 2 荒れ）
    ladder = {
        "A1_nag4|form2_5_7|form3_7_11": [lambda r: cols_nagashi(r, 4), lambda r: cols_formation(r, 2, 5, 7), lambda r: cols_formation(r, 3, 7, 11)],
        "A2_skip|form2_5_7|form3_7_11": [lambda r: None, lambda r: cols_formation(r, 2, 5, 7), lambda r: cols_formation(r, 3, 7, 11)],
        "A3_nag4|form2_6_9|form3_8_12": [lambda r: cols_nagashi(r, 4), lambda r: cols_formation(r, 2, 6, 9), lambda r: cols_formation(r, 3, 8, 12)],
        "A4_cum(.3,.4,.5)|(.4,.55,.7)|(.6,.7,.8)": [lambda r: cols_cum(r, (.3, .4, .5)), lambda r: cols_cum(r, (.4, .55, .7)), lambda r: cols_cum(r, (.6, .7, .8))],
        "A5_top6|top15|top30": [lambda r: cols_topn(r, 6), lambda r: cols_topn(r, 15), lambda r: cols_topn(r, 30)],
        "A6_nag4|nag6|form3_7_11": [lambda r: cols_nagashi(r, 4), lambda r: cols_nagashi(r, 6), lambda r: cols_formation(r, 3, 7, 11)],
        "R2_top30|top15|top6(逆)": [lambda r: cols_topn(r, 30), lambda r: cols_topn(r, 15), lambda r: cols_topn(r, 6)],
        "R1_wide_when_calm(逆)": [lambda r: cols_formation(r, 3, 7, 11), lambda r: cols_formation(r, 2, 5, 7), lambda r: cols_nagashi(r, 4)],   # 逆向きの対照
    }
    for name, fns in ladder.items():
        for src in ("model", "market"):
            S[f"{name}@{src}"] = (lambda fns, src: (lambda r, band: fns[band](r)))(fns, src)
    return S


def thresholds(races, Y, src):
    """評価年 Y より前の年（2021 以降。OOF のある年）の予測荒れ度の三分位。"""
    vo = pd.read_parquet(OUT / "vol_oof.parquet")
    v = vo.loc[(vo["year"] >= 2021) & (vo["year"] < Y), f"{src}_vol"].dropna().to_numpy()
    return np.quantile(v, [1 / 3, 2 / 3])


def run(races, S):
    names = list(S)
    stake = np.zeros((len(races), len(names)))
    ret = np.zeros_like(stake)
    for i, r in enumerate(races):
        th = {src: thresholds_cache[(r["year"], src)] for src in ("model", "market")}
        for j, nm in enumerate(names):
            src = "market" if nm.endswith("@market") else "model"
            v = r[f"vol_{src}"]
            band = 0 if v < th[src][0] else (1 if v < th[src][1] else 2)
            combos = S[nm](r, band)
            n, pay = settle(r, combos)
            stake[i, j] = 100 * n
            ret[i, j] = pay
    return names, stake, ret


thresholds_cache: dict = {}


def maxdd(pl_series: np.ndarray) -> float:
    c = np.cumsum(pl_series)
    peak = np.maximum.accumulate(np.concatenate([[0.0], c]))[1:]
    return float((peak - c).max())


def boot_roi_diff(days, st_a, rt_a, st_b, rt_b, n_boot=1000, seed=0):
    """日単位のブートストラップで ROI 差（a − b）の分布を返す。"""
    rng = np.random.default_rng(seed)
    ud, inv = np.unique(days, return_inverse=True)
    d_st_a, d_rt_a, d_st_b, d_rt_b = [np.bincount(inv, weights=x, minlength=len(ud)) for x in (st_a, rt_a, st_b, rt_b)]
    obs = rt_a.sum() / max(st_a.sum(), 1) - rt_b.sum() / max(st_b.sum(), 1)
    out = np.empty(n_boot)
    for b in range(n_boot):
        k = rng.integers(0, len(ud), len(ud))
        out[b] = d_rt_a[k].sum() / max(d_st_a[k].sum(), 1) - d_rt_b[k].sum() / max(d_st_b[k].sum(), 1)
    return obs, out


def evaluate() -> None:
    with open(OUT / "bet_races.pkl", "rb") as f:
        races = pickle.load(f)
    races = [r for r in races if r["year"] in VALID and r["trio_rank"] is not None]
    races.sort(key=lambda r: (r["date"], r["race_id"]))
    for Y in VALID:
        for src in ("model", "market"):
            thresholds_cache[(Y, src)] = thresholds(races, Y, src)
    S = build_strategies()
    names, stake, ret = run(races, S)
    days = np.array([r["date"] for r in races])
    y = np.array([r["year"] for r in races])
    rep: dict = {"n_races": len(races), "n_strategies": len(names), "thresholds": {f"{k[0]}_{k[1]}": [float(x) for x in v] for k, v in thresholds_cache.items()}}
    table = []
    for j, nm in enumerate(names):
        s, rt = stake[:, j], ret[:, j]
        bought = s > 0
        pl = rt - s
        row = {"name": nm, "races_bought": int(bought.sum()), "points_per_bought_race": float(s[bought].mean() / 100) if bought.any() else 0.0, "stake_total": float(s.sum()),
               "roi": float(rt.sum() / max(s.sum(), 1)), "hit_rate": float((rt[bought] > 0).mean()) if bought.any() else 0.0, "max_dd": maxdd(pl), "max_dd_per_stake": maxdd(pl) / max(s.sum(), 1),
               "pl_sd_per_race": float(pl.std()), "roi_by_year": {int(Y): float(rt[y == Y].sum() / max(s[y == Y].sum(), 1)) for Y in VALID}}
        table.append(row)
    rep["table"] = table
    # 同じ平均点数の固定戦略と比べる（点数が近いもの）
    fixed = [t for t in table if t["name"].startswith("F_")]
    comps = []
    adaptive = [t for t in table if not t["name"].startswith("F_")]
    for t in adaptive:
        j = names.index(t["name"])
        fam = "F_top" if t["name"].startswith(("A5", "R2")) else "F_"
        same = [f for f in fixed if (f["name"].startswith("F_top") if fam == "F_top" else not f["name"].startswith("F_top"))]
        b = min(same, key=lambda f: abs(f["points_per_bought_race"] - t["points_per_bought_race"]))
        near = [f for f in fixed if abs(f["points_per_bought_race"] - t["points_per_bought_race"]) <= 0.2 * t["points_per_bought_race"]] or [b]
        strong = max(near, key=lambda f: f["roi"])           # 点数が近い固定のうち回収率が最良のもの（選択の甘さは固定側に有利）
        jb, js = names.index(b["name"]), names.index(strong["name"])
        obs, bs = boot_roi_diff(days, stake[:, j], ret[:, j], stake[:, jb], ret[:, jb])
        obs_s, bs_s = boot_roi_diff(days, stake[:, j], ret[:, j], stake[:, js], ret[:, js])
        comps.append({"adaptive": t["name"], "bench": b["name"], "pts_a": t["points_per_bought_race"], "pts_b": b["points_per_bought_race"], "roi_a": t["roi"], "roi_b": b["roi"],
                      "diff": float(obs), "ci_lo": float(np.quantile(bs, 0.025)), "ci_hi": float(np.quantile(bs, 0.975)), "maxdd_a": t["max_dd_per_stake"], "maxdd_b": b["max_dd_per_stake"],
                      "hit_a": t["hit_rate"], "hit_b": b["hit_rate"], "obs": float(obs),
                      "strong_bench": strong["name"], "strong_pts": strong["points_per_bought_race"], "strong_roi": strong["roi"], "diff_strong": float(obs_s),
                      "ci_lo_strong": float(np.quantile(bs_s, 0.025)), "ci_hi_strong": float(np.quantile(bs_s, 0.975)), "obs_s": float(obs_s), "_boot": bs, "_boot_s": bs_s})
    # reality check（max 統計）。各戦略の差を観測値で中心化した分布の最大値と比較（同系統の点数が近い固定 / 点数が近い最強の固定、の 2 通り）
    rc = {}
    for key, ob, bk in (("same_family", "obs", "_boot"), ("strongest_matched", "obs_s", "_boot_s")):
        obs_max = max(c[ob] for c in comps)
        mx = np.vstack([c[bk] - c[ob] for c in comps]).max(axis=0)
        rc[key] = {"n_tested": len(comps), "max_obs_diff": float(obs_max), "p": float((mx >= obs_max).mean())}
    for c in comps:
        c.pop("_boot"), c.pop("_boot_s")
    rep["compare_adaptive_vs_matched_fixed"] = comps
    rep["reality_check"] = rc
    # 帯別 ROI（固定戦略）。荒れ度の帯ごとに回収率がどう変わるか
    band_rows = []
    for src in ("model", "market"):
        b = np.array([0 if r[f"vol_{src}"] < thresholds_cache[(r["year"], src)][0] else (1 if r[f"vol_{src}"] < thresholds_cache[(r["year"], src)][1] else 2) for r in races])
        for nm in ("F_box5", "F_nag5", "F_form_2_5_7", "F_form_3_7_11", "F_top6", "F_top10", "F_top20", "F_top40"):
            j = names.index(nm)
            for bd in (0, 1, 2):
                m = b == bd
                obs, bs = boot_roi_diff(days[m], stake[m, j], ret[m, j], np.ones(m.sum()) * 0 + 1, np.ones(m.sum()) * 0, n_boot=500)
                # 上は b 側を ROI=0 にしたダミー（a の ROI の CI を取る）
                band_rows.append({"vol": src, "strategy": nm, "band": bd, "n_races": int(m.sum()), "roi": float(ret[m, j].sum() / max(stake[m, j].sum(), 1)),
                                  "ci": [float(np.quantile(bs + (0.0), 0.025) + 0.0), float(np.quantile(bs, 0.975))],
                                  "hit": float((ret[m, j] > 0).mean())})
    rep["roi_by_band_fixed"] = band_rows
    # 3連複の確率の較正: 上位 n 点の累積確率と実際の的中率
    cal = []
    for n in (6, 10, 20, 40):
        cum = np.array([r["trio_cum"][n - 1] for r in races])
        hit = np.array([1.0 if (r["trio_rank"] <= n) else 0.0 for r in races])
        cal.append({"top_n": n, "mean_pred_cum_prob": float(cum.mean()), "actual_hit": float(hit.mean())})
    rep["trio_calibration"] = cal
    # 券種の選択: 荒れ度の帯別 ROI（単勝1点・複勝1点・馬連3頭ボックス・ワイド3頭ボックス・3連複5頭ボックス）
    kinds = {"tansho_top1": 0, "fukusho_top1": 0, "umaren_box3": 0, "wide_box3": 0, "sanrenpuku_box5": 0}
    kind_rows = []
    for src in ("model", "market"):
        for bd in (0, 1, 2):
            acc = {k: [0.0, 0.0, 0] for k in kinds}
            for r in races:
                th = thresholds_cache[(r["year"], src)]
                v = r[f"vol_{src}"]
                if (0 if v < th[0] else (1 if v < th[1] else 2)) != bd:
                    continue
                t1 = int(r["rk_win"][0])
                top3 = _top(r["rk_win"], 3)
                pairs = {tuple(sorted(p)) for p in [(top3[0], top3[1]), (top3[0], top3[2]), (top3[1], top3[2])]}
                # 単勝
                acc["tansho_top1"][0] += 100
                if r["tansho"] and r["tansho"][0] == (t1,):
                    acc["tansho_top1"][1] += r["tansho"][1]
                # 複勝（払戻のセルが連結されて分解できないレースは除く）
                if len(r["fukusho"]) >= 2:
                    acc["fukusho_top1"][0] += 100
                    for a, pay in r["fukusho"]:
                        if a == (t1,):
                            acc["fukusho_top1"][1] += pay
                # 馬連
                acc["umaren_box3"][0] += 300
                if r["umaren"] and r["umaren"][0] in pairs:
                    acc["umaren_box3"][1] += r["umaren"][1]
                # ワイド
                if len(r["wide"]) >= 2:
                    acc["wide_box3"][0] += 300
                    for a, pay in r["wide"]:
                        if a in pairs:
                            acc["wide_box3"][1] += pay
                # 3連複
                acc["sanrenpuku_box5"][0] += 1000
                if r["trio_win"] in cols_box(r, 5):
                    acc["sanrenpuku_box5"][1] += r["trio_pay"]
                for k in kinds:
                    acc[k][2] += 1 if True else 0
            for k, (s_, r_, n_) in acc.items():
                kind_rows.append({"vol": src, "band": bd, "ticket": k, "n_races": n_, "roi": float(r_ / max(s_, 1))})
    rep["roi_by_ticket_band"] = kind_rows
    (OUT / "bet.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")
    pd.set_option("display.width", 250)
    tb = pd.DataFrame(table)[["name", "races_bought", "points_per_bought_race", "roi", "hit_rate", "max_dd_per_stake", "pl_sd_per_race"]]
    print(tb.round(4).to_string(index=False))
    print(pd.DataFrame(comps)[["adaptive", "bench", "pts_a", "pts_b", "roi_a", "roi_b", "diff", "ci_lo", "ci_hi", "maxdd_a", "maxdd_b"]].round(4).to_string(index=False))
    print(pd.DataFrame(comps)[["adaptive", "strong_bench", "strong_pts", "strong_roi", "diff_strong", "ci_lo_strong", "ci_hi_strong"]].round(4).to_string(index=False))
    print("reality check", rep["reality_check"])
    print("trio calibration", cal)


if __name__ == "__main__":
    {"prep": prep, "eval": evaluate}[sys.argv[1]]()

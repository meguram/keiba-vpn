"""rules-bet-opt の探す・評価・凍結の後の 1 回（config/rules_bet_opt/search_spec.json のとおり。仕様を変えたら別の登録として始める）。

入力: <COURSE_OUT>/bet_opt/wf_Y.parquet・field_Y.parquet（bet_opt wf の出力。Y = 2020〜2024）。
発走前のオッズは bet_opt.sim_t60（bet_cycles の作法）で 20 通り。指標は bet_cycles.metrics・poisson_weights・paired_diff・decompose。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.course import bet_opt as BO
from src.scripts.data import bet_cycles as BC

N_DRAWS = 20
N_BOOT = 1000


def spec() -> dict:
    return json.loads(BO.SPEC.read_text(encoding="utf-8"))


def load_years(years) -> tuple[pd.DataFrame, pd.DataFrame]:
    hm = pd.concat([pd.read_parquet(BO.OUT / f"wf_{y}.parquet") for y in years], ignore_index=True)
    fl = pd.concat([pd.read_parquet(BO.OUT / f"field_{y}.parquet") for y in years], ignore_index=True)
    return prep(hm, fl)


def prep(hm: pd.DataFrame, fl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """払戻の解けたレースだけに絞り、型をそろえる。hm = ◎ の行（1 レース 1 行）、fl = そのレースの全頭。"""
    for x in (hm, fl):
        x["race_id"] = x["race_id"].astype(str)
        x["hn"] = pd.to_numeric(x["hn"], errors="coerce")
        for c in ("odds_final", "pop_final", "fin", "ret_tan", "ret_fuku"):
            x[c] = pd.to_numeric(x[c], errors="coerce")
    hm = hm[hm["pay_ok"].fillna(False).astype(bool)].drop_duplicates("race_id").sort_values(["date", "race_id"]).reset_index(drop=True)
    fl = fl[fl["race_id"].isin(set(hm["race_id"]))].reset_index(drop=True)
    for c in ("ret_tan", "ret_fuku"):
        hm[c], fl[c] = hm[c].fillna(0.0), fl[c].fillna(0.0)
    return hm, fl


class Draws:
    """20 通りの T-60（γ・基準の券種ごと）。o[v] = ◎ の T-60 の単勝オッズ（hm の行の順）、p[v] = その人気、fav[v] = T-60 の 1 番人気の
    (単勝の払戻, 複勝の払戻)、band[v] = 帯ごとの (頭数, 単勝の払戻の和, 複勝の払戻の和)。"""

    def __init__(self, hm: pd.DataFrame, fl: pd.DataFrame, pool, gamma: float, basis: str, bands: list, k: int = N_DRAWS, real: dict | None = None):
        self.o, self.p, self.fav, self.band = [], [], [], []
        key_h = list(zip(hm["race_id"], hm["hn"]))
        for v in range(k):
            if real is not None:                          # 実物の T-60（1 通り）
                s = fl[["race_id", "hn"]].copy()
                s["pre"] = [real.get((r, int(h)), (np.nan, np.nan))[0] if pd.notna(h) else np.nan for r, h in zip(s["race_id"], s["hn"])]
                s["pre_pop"] = s.groupby("race_id")["pre"].rank(method="min").fillna(99.0)
            else:
                s = BO.sim_t60(fl, pool, seed=v, gamma=gamma, basis=basis)
            m = fl.merge(s, on=["race_id", "hn"], how="left")
            idx = dict(zip(zip(m["race_id"], m["hn"]), range(len(m))))
            pre, pp = m["pre"].to_numpy(float), m["pre_pop"].to_numpy(float)
            j = np.array([idx.get(x, -1) for x in key_h])
            self.o.append(np.where(j >= 0, pre[np.maximum(j, 0)], np.nan))
            self.p.append(np.where(j >= 0, pp[np.maximum(j, 0)], 99.0))
            f = m[m["pre_pop"] == 1].drop_duplicates("race_id").set_index("race_id")
            self.fav.append(np.column_stack([f["ret_tan"].reindex(hm["race_id"]).fillna(0).to_numpy(), f["ret_fuku"].reindex(hm["race_id"]).fillna(0).to_numpy(),
                                             pd.Series(hm["race_id"].isin(f.index).to_numpy(), dtype=float).to_numpy()]))
            bb = []
            for lo, hi in bands:
                inb = (m["pre"] > lo) & (m["pre"] <= hi)
                g = m[inb].groupby("race_id").agg(n=("hn", "size"), t=("ret_tan", "sum"), f=("ret_fuku", "sum")).reindex(hm["race_id"]).fillna(0)
                bb.append(g.to_numpy())
            self.band.append(bb)
            if real is not None:
                break
        self.k = len(self.o)


def date_weights(dates: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """日ごとのまとまりのブートストラップの重み（bet_cycles.poisson_weights を日に掛け、k 通りに同じ重みを並べる）。"""
    ud, inv = np.unique(dates, return_inverse=True)
    Wd = BC.poisson_weights(len(ud), N_BOOT, seed)
    return np.tile(Wd[:, inv], (1, k))


def stack(arrs: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrs)


def cand_arrays(hm: pd.DataFrame, D: Draws, c: dict, thr: dict, logit: dict | None = None) -> dict:
    """候補 c の (cost, ret) を k 通り重ねた配列。c = {kind: grid|logit|B0|B_all|B_fav|B_band, band: [lo,hi], strength, bet}。"""
    col = "ret_tan" if c["bet"] == "tan" else "ret_fuku"
    r1 = hm[col].to_numpy(float)
    sc = hm["score"].to_numpy(float)
    cs, rs = [], []
    for v in range(D.k):
        o, p = D.o[v], D.p[v]
        if c["kind"] == "grid":
            lo, hi = c["band"]
            b = np.isfinite(o) & (o > lo) & (o <= hi) & (sc >= thr[c["strength"]])
            cost, ret = b.astype(float), np.where(b, r1, 0.0)
        elif c["kind"] == "logit":
            ph = logit_p(logit, o, sc, D, v, hm)
            b = np.isfinite(o) & (ph * o > logit["margin"])
            cost, ret = b.astype(float), np.where(b, r1, 0.0)
        elif c["kind"] == "B0":
            b = np.isfinite(o) & (p <= 3)
            cost, ret = b.astype(float), np.where(b, r1, 0.0)
        elif c["kind"] == "B_all":
            cost, ret = np.ones(len(hm)), r1.copy()
        elif c["kind"] == "B_fav":
            f = D.fav[v]
            cost, ret = f[:, 2].astype(float), f[:, 0 if c["bet"] == "tan" else 1]
        elif c["kind"] == "B_band":
            g = D.band[v][c["band_i"]]
            cost, ret = g[:, 0], g[:, 1 if c["bet"] == "tan" else 2]
        else:
            raise ValueError(c["kind"])
        cs.append(cost)
        rs.append(ret)
    return {"cost": stack(cs), "ret": stack(rs)}


def market_q_honmei(hm: pd.DataFrame, D: Draws, v: int) -> np.ndarray:
    """◎ の T-60 の市場の勝率（1/オッズ を控除込みで 0.8 倍に近い値に直さず、レースの 1/オッズの合計で割る）。合計は確定と同じに合わせてあるので、
    確定の 1/オッズの合計で割る。"""
    return (1.0 / D.o[v]) / hm["inv_sum"].to_numpy(float)


def logit_p(m: dict, o: np.ndarray, sc: np.ndarray, D: Draws, v: int, hm: pd.DataFrame) -> np.ndarray:
    q = np.clip(market_q_honmei(hm, D, v), 1e-4, 0.999)
    z = m["b"][0] + m["b"][1] * np.log(q) + m["b"][2] * np.log1p(sc)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logit(hm: pd.DataFrame, D: Draws) -> dict:
    """勝ち ~ log q_T60 + log(1 + 点) の IRLS（20 通りを重ねた行）。"""
    X, y = [], []
    for v in range(D.k):
        q = np.clip(market_q_honmei(hm, D, v), 1e-4, 0.999)
        ok = np.isfinite(q)
        X.append(np.column_stack([np.ones(ok.sum()), np.log(q[ok]), np.log1p(hm["score"].to_numpy(float)[ok])]))
        y.append((hm["fin"].to_numpy(float)[ok] == 1).astype(float))
    X, y = np.vstack(X), np.concatenate(y)
    b = np.zeros(3)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        w = p * (1 - p)
        g = X.T @ (y - p)
        Hm = X.T @ (X * w[:, None]) + 1e-6 * np.eye(3)
        st = np.linalg.solve(Hm, g)
        b = b + st
        if np.abs(st).max() < 1e-8:
            break
    return {"b": b.tolist(), "margin": 1.05, "n": int(len(y))}


def with_inv_sum(hm: pd.DataFrame, fl: pd.DataFrame) -> pd.DataFrame:
    s = fl.assign(inv=1.0 / fl["odds_final"].where(fl["odds_final"] > 0)).groupby("race_id")["inv"].sum()
    return hm.assign(inv_sum=s.reindex(hm["race_id"]).to_numpy())


def grid(sp: dict) -> list[dict]:
    out = []
    for bi, band in enumerate(sp["grid"]["odds_bands"]):
        for st in sp["grid"]["strength"]:
            for bet in sp["grid"]["bet"]:
                out.append({"id": f"g{bi}_{st}_{bet}", "kind": "grid", "band": band, "band_i": bi, "strength": st, "bet": bet})
    out.append({"id": "logit_tan", "kind": "logit", "bet": "tan"})
    return out


def mets(a: dict, dates: np.ndarray, k: int, W: np.ndarray | None) -> dict:
    BC._REP[len(a["cost"])] = k
    return BC.metrics(a["cost"], a["ret"], np.tile(dates, k), W)


def rw_adjust(arrs: dict[str, dict], W: np.ndarray, ok: set[str]) -> dict:
    """H0: 回収率 ≤ 1。ブートストラップの回収率 − 実測（中心化）の最大で、Romano–Wolf のステップダウン・White の 1 段目・Bonferroni を出す。"""
    ids = [i for i in arrs if i in ok]
    if not ids:
        return {}
    roi = np.array([arrs[i]["ret"].sum() / max(arrs[i]["cost"].sum(), 1e-9) for i in ids])
    bs = np.column_stack([(W @ arrs[i]["ret"]) / np.maximum(W @ arrs[i]["cost"], 1e-9) for i in ids])
    t = roi - 1.0
    cen = bs - roi[None, :]
    order = np.argsort(-t)
    adj = np.ones(len(ids))
    active = list(order)
    prev = 0.0
    for i in order:                                       # ステップダウン: 残りの仮説の最大で、いちばん大きい t の p を出し、外していく
        mx = cen[:, active].max(1)
        p = float((mx >= t[i]).mean())
        prev = max(prev, p)
        adj[i] = prev
        active.remove(i)
    white = float((cen.max(1) >= t.max()).mean())
    raw = (bs <= 1.0).mean(0)
    return {i: {"roi": float(roi[j]), "p_raw": float(raw[j]), "p_bonf": float(min(1.0, raw[j] * len(ids))), "p_rw": float(adj[j])} for j, i in enumerate(ids)} | {"_white_p": white, "_n": len(ids)}


def thresholds(hm: pd.DataFrame) -> dict:
    s = hm["score"].to_numpy(float)
    return {"all": -np.inf, "top_half": float(np.quantile(s, 0.5)), "top_third": float(np.quantile(s, 2 / 3))}


def evaluate(hm, fl, pool, cands, thr, logit, years_label, gammas=(0.0,), W_seed=0) -> dict:
    """期間の値: 候補ごとの指標（γ ごと）と、基準との対の差。"""
    dates = hm["date"].astype(str).to_numpy()
    W = date_weights(dates, N_DRAWS, W_seed)
    out = {"period": years_label, "races": int(len(hm)), "by_gamma": {}}
    sp = spec()
    for g in gammas:
        res = {}
        Ds = {"win": Draws(hm, fl, pool, g, "win", sp["grid"]["odds_bands"])}
        Ds["place"] = Ds["win"] if g == 0 else Draws(hm, fl, pool, g, "place", sp["grid"]["odds_bands"])
        arrs = {}
        for c in cands:
            D = Ds["win" if c["bet"] == "tan" else "place"]
            arrs[c["id"]] = cand_arrays(hm, D, c, thr, logit)
        for bet in ("tan", "fuku"):
            D = Ds["win" if bet == "tan" else "place"]
            for b in ("B0", "B_all", "B_fav"):
                arrs[f"{b}_{bet}"] = cand_arrays(hm, D, {"kind": b, "bet": bet}, thr)
            for bi in range(len(sp["grid"]["odds_bands"])):
                arrs[f"B_band{bi}_{bet}"] = cand_arrays(hm, D, {"kind": "B_band", "band_i": bi, "bet": bet}, thr)
        for i, a in arrs.items():
            res[i] = mets(a, dates, N_DRAWS, W)
        for c in cands:
            a = arrs[c["id"]]
            res[c["id"]]["diff"] = {}
            for b in ("B0", "B_all", "B_fav") + ((f"B_band{c['band_i']}",) if c["kind"] == "grid" else ()):
                res[c["id"]]["diff"][b] = BC.paired_diff(a, arrs[f"{b}_{c['bet']}"], W)
            res[c["id"]]["decompose_vs_B0"] = BC.decompose(res[c["id"]], res[f"B0_{c['bet']}"])
        if g == 0:
            out["multiple"] = rw_adjust({c["id"]: arrs[c["id"]] for c in cands}, W, {c["id"] for c in cands})
            out["_arrays"] = arrs
        out["by_gamma"][str(g)] = res
    return out


def optimistic(hm: pd.DataFrame, fl: pd.DataFrame, cands: list[dict], thr: dict, logit: dict | None) -> dict:
    """確定オッズで判定した値（楽観）: T-60 の代わりに確定オッズ・人気をそのまま使う。"""
    class _D:  # 1 通り
        pass
    D = _D()
    D.k = 1
    D.o = [hm["odds_final"].to_numpy(float)]
    D.p = [hm["pop_final"].to_numpy(float)]
    out = {}
    dates = hm["date"].astype(str).to_numpy()
    for c in cands + [{"id": "B0_tan", "kind": "B0", "bet": "tan"}, {"id": "B0_fuku", "kind": "B0", "bet": "fuku"}]:
        a = cand_arrays(hm, D, c, thr, logit)
        BC._REP[len(a["cost"])] = 1
        out[c["id"]] = BC.metrics(a["cost"], a["ret"], dates, None)
    return out


def _slim(m: dict) -> dict:
    keys = ("races", "bet_races", "bet_rate", "hit_rate", "roi", "roi_lo", "roi_hi", "roi_ex_top5", "avg_return", "median_return", "sim_roi_min", "sim_roi_max")
    o = {k: m.get(k) for k in keys if k in m}
    if "diff" in m:
        o["diff"] = m["diff"]
    if "decompose_vs_B0" in m:
        o["decompose_vs_B0"] = m["decompose_vs_B0"]
    return o


def neighbors(c: dict, cands: list[dict]) -> list[str]:
    """閾値の 1 段ずらし: 同じ券種で、帯の 1 つ隣（並びの前後）か、確からしさの 1 つ隣。"""
    if c["kind"] != "grid":
        return []
    st = ["all", "top_half", "top_third"]
    out = []
    for x in cands:
        if x["kind"] != "grid" or x["bet"] != c["bet"] or x["id"] == c["id"]:
            continue
        if x["strength"] == c["strength"] and abs(x["band_i"] - c["band_i"]) == 1:
            out.append(x["id"])
        if x["band_i"] == c["band_i"] and abs(st.index(x["strength"]) - st.index(c["strength"])) == 1:
            out.append(x["id"])
    return out


def per_year(hm, fl, pool, c, thr, logit) -> dict:
    out = {}
    for y in sorted(hm["year"].unique()):
        m = (hm["year"] == y).to_numpy()
        h2 = hm[m].reset_index(drop=True)
        f2 = fl[fl["race_id"].isin(set(h2["race_id"]))].reset_index(drop=True)
        D = Draws(h2, f2, pool, 0.0, "win", spec()["grid"]["odds_bands"])
        a = cand_arrays(h2, D, c, thr, logit)
        b = cand_arrays(h2, D, {"kind": "B0", "bet": c["bet"]}, thr)
        out[str(int(y))] = {"cand": _slim(mets(a, h2["date"].astype(str).to_numpy(), N_DRAWS, None)),
                            "B0": _slim(mets(b, h2["date"].astype(str).to_numpy(), N_DRAWS, None))}
    return out


def similar_units(hm, fl, pool, c, thr, logit, min_bets: int = 30) -> dict:
    """似た単位（芝ダ × 距離帯 × クラスの群）ごとに、候補が B0 を上回るか。"""
    fam = hm["surf"].astype(str) + "|" + pd.cut(pd.to_numeric(hm["dist"], errors="coerce"), [0, 1400, 1800, 2200, 9999], labels=["〜1400", "〜1800", "〜2200", "2200〜"]).astype(str) + "|" + hm["group"].astype(str)
    D = Draws(hm, fl, pool, 0.0, "win", spec()["grid"]["odds_bands"])
    a = cand_arrays(hm, D, c, thr, logit)
    b = cand_arrays(hm, D, {"kind": "B0", "bet": c["bet"]}, thr)
    n = len(hm)
    rows = {}
    for f in sorted(fam.unique()):
        m = np.tile((fam == f).to_numpy(), N_DRAWS)
        ca, ra, cb, rb = a["cost"][m].sum(), a["ret"][m].sum(), b["cost"][m].sum(), b["ret"][m].sum()
        if ca / N_DRAWS < min_bets:
            continue
        rows[f] = {"bets": ca / N_DRAWS, "roi": ra / ca, "B0_roi": (rb / cb) if cb else None}
    better = [r["roi"] > (r["B0_roi"] or 0) for r in rows.values()]
    return {"families": rows, "n": len(rows), "share_better_than_B0": float(np.mean(better)) if better else None, "min_bets": min_bets, "_n_races": n}


POOL_FILE = BO.OUT / "drift_pool_t60.npz"


def frozen_pool() -> list[np.ndarray]:
    """探す・評価で使ったずれの元（凍結）。無ければ今の pre_odds から作って残す（探す前だけ）。"""
    if POOL_FILE.exists():
        z = np.load(POOL_FILE)
        return [z[k] for k in sorted(z.files, key=int)]
    pool = BO.drift_pool("t60")
    np.savez(POOL_FILE, **{str(i): x for i, x in enumerate(pool)})
    return pool


def cmd_search() -> None:
    sp = spec()
    pool = frozen_pool()
    hs, fs = load_years(range(*[sp["periods"]["search"][0], sp["periods"]["search"][1] + 1]))
    he, fe = load_years(range(sp["periods"]["evaluate"][0], sp["periods"]["evaluate"][1] + 1))
    hs, he = with_inv_sum(hs, fs), with_inv_sum(he, fe)
    thr = thresholds(hs)
    cands = grid(sp)
    D0 = Draws(hs, fs, pool, 0.0, "win", sp["grid"]["odds_bands"])
    logit = fit_logit(hs, D0)
    S = evaluate(hs, fs, pool, cands, thr, logit, "2020-2022")
    res0 = S["by_gamma"]["0.0"]
    ok = [c for c in cands if (res0[c["id"]].get("bet_races") or 0) >= sp["min_bets"]["search"]]
    best = max(ok, key=lambda c: res0[c["id"]].get("roi") or -1) if ok else None
    out = {"spec_sha": BO.hashlib.sha256(BO.SPEC.read_bytes()).hexdigest()[:12], "thresholds": thr, "logit": logit,
           "search": {"races": S["races"], "cands": {c["id"]: _slim(res0[c["id"]]) for c in cands},
                      "baselines": {k: _slim(v) for k, v in res0.items() if k.startswith("B")}, "multiple": S["multiple"]},
           "eligible": [c["id"] for c in ok], "best": best}
    if best:
        E = evaluate(he, fe, pool, cands, thr, logit, "2023-2024", gammas=tuple(sp["gamma"]["grid"]), W_seed=1)
        e0 = E["by_gamma"]["0.0"]
        out["evaluate"] = {"races": E["races"], "best": _slim(e0[best["id"]]),
                           "gamma": {g: {"cand": _slim(r[best["id"]]), "B0": _slim(r[f"B0_{best['bet']}"])} for g, r in E["by_gamma"].items()},
                           "neighbors": {n: _slim(e0[n]) for n in neighbors(best, cands)},
                           "neighbors_search": {n: _slim(res0[n]) for n in neighbors(best, cands)},
                           "all_cands": {c["id"]: _slim(e0[c["id"]]) for c in cands},
                           "baselines": {k: _slim(v) for k, v in e0.items() if k.startswith("B")},
                           "optimistic": {k: _slim(v) for k, v in optimistic(he, fe, cands, thr, logit).items()}}
        hall = pd.concat([hs, he], ignore_index=True)
        fall = pd.concat([fs, fe], ignore_index=True)
        out["per_year"] = per_year(hall, fall, pool, best, thr, logit)
        out["similar_units_eval"] = similar_units(he, fe, pool, best, thr, logit)
        ev = out["evaluate"]["best"]
        gam = [out["evaluate"]["gamma"][str(g)]["cand"]["diff"]["B0"][0] for g in sp["gamma"]["grid"] if sp["gamma"]["required"][0] <= g <= sp["gamma"]["required"][1]]
        out["adopt"] = {"bets_ok": (ev.get("bet_races") or 0) >= sp["min_bets"]["evaluate"], "roi_ok": (ev.get("roi") or 0) >= 1.0,
                        "trim_ok": (ev.get("roi_ex_top5") or 0) >= 1.0, "diff_B0_lo_ok": ev["diff"]["B0"][1] > 0, "gamma_ok": all(x > 0 for x in gam)}
        out["adopt"]["decision"] = "採用" if all(out["adopt"].values()) else "保留"
    out["search_optimistic"] = {k: _slim(v) for k, v in optimistic(hs, fs, cands, thr, logit).items()}
    (BO.OUT / "search_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"best": best, "search": out["search"]["cands"].get(best["id"]) if best else None,
                      "evaluate": out.get("evaluate", {}).get("best"), "adopt": out.get("adopt")}, ensure_ascii=False, indent=1, default=float))


# ---------------------------------------------------------------- 凍結の後に 1 回だけ（config/rules_bet_opt/prereg_final.json）
LEVELS = ("否定", "100% 超えを確かめた", "100% 超えの向き（確かめは弱い）", "基準より上（100% は確かめられず）", "確かめられず")


def judge(m: dict, diff: tuple, min_bets: int) -> str:
    """prereg_final.json の 5 段（bet-optimizer の prereg_2026_r2.json と同じ形。B0 との対の差）。順は 否定 → 上から。"""
    if (m.get("bet_races") or 0) < min_bets:
        return "判定なし（件数が足りない）"
    d, lo, hi = diff
    roi, tr, rhi, rlo = m.get("roi") or 0, m.get("roi_ex_top5") or 0, m.get("roi_hi") or 0, m.get("roi_lo") or 0
    if d <= 0 or rhi < 1.0:
        return LEVELS[0]
    if rlo >= 1.0 and tr >= 1.0 and lo > 0:
        return LEVELS[1]
    if roi >= 1.0 and tr >= 1.0 and lo > 0:
        return LEVELS[2]
    if lo > 0:
        return LEVELS[3]
    return LEVELS[4]


def graded_frames(years: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """重賞: 凍結した B の ◎（history.json）と、そのレースの全頭の確定オッズ（race_result_flat）。払戻は ◎ の分だけ（単複、100 円あたり → 1 あたり）。"""
    from src.research.graded import picks as PK  # noqa: F401  （置き場の既定を合わせるため）
    H = json.loads((Path(os.environ.get("GRADED_PICKS_DIR") or (BO.CR.REPO / "data/local/graded_picks")) / "history.json").read_text(encoding="utf-8"))
    from src.research.graded.versions import V0_ID
    rows, other = [], 0
    for v in H["rkeys"].values():
        for x in v["races"]:
            y = int(str(x["date"])[:4])
            b = (x.get("B") or {}).get("honmei")
            vid = (x.get("version") or {}).get("id") or V0_ID
            if y in years and b and x.get("pay_ok") and vid != V0_ID:
                other += 1                                 # 前日の計算の新しい版の ◎（凍結した B ではない）は使わない
                continue
            if y in years and b and x.get("pay_ok"):
                rows.append({"race_id": str(x["race_id"]), "date": x["date"], "year": y, "hn": b["hn"], "ret_tan": b["tan"] / 100.0,
                             "ret_fuku": b["fuku"] / 100.0, "pay_ok": True, "score": 0.0, "surf": "", "dist": np.nan, "group": "重賞"})
    hm = pd.DataFrame(rows).drop_duplicates("race_id")
    fl = pd.concat([pd.read_parquet(BO.CR.REPO / "data/page_reference/tables" / str(y) / "race_result_flat.parquet",
                                    columns=["race_id", "horse_number", "odds", "popularity", "finish_position"]) for y in years])
    fl = fl.rename(columns={"horse_number": "hn", "odds": "odds_final", "popularity": "pop_final", "finish_position": "fin"})
    fl["race_id"] = fl["race_id"].astype(str)
    fl = fl[fl["race_id"].isin(set(hm["race_id"]))].copy()
    fl["hn"] = pd.to_numeric(fl["hn"], errors="coerce")
    fl = fl.drop_duplicates(["race_id", "hn"])
    k = fl.merge(hm[["race_id", "hn", "ret_tan", "ret_fuku"]], on=["race_id", "hn"], how="left")
    fl["ret_tan"], fl["ret_fuku"] = k["ret_tan"].fillna(0.0).to_numpy(), k["ret_fuku"].fillna(0.0).to_numpy()
    o = fl[["race_id", "hn", "odds_final", "pop_final", "fin"]]
    hm = hm.merge(o, on=["race_id", "hn"], how="left")
    hm, fl = prep(hm, fl)
    hm.attrs["excluded_other_version"] = other
    return hm, fl


def run_group(name: str, hm: pd.DataFrame, fl: pd.DataFrame, pool, rule: dict, thr: dict, min_bets: int, real: dict | None = None,
              gammas=(0.0,), with_fav: bool = True) -> dict:
    cov = {"honmei_in": int(len(hm)), "honmei_no_final_odds": int(hm["odds_final"].isna().sum()),
           "excluded_other_version": int(hm.attrs.get("excluded_other_version", 0))}
    if real is not None:                                  # 実物の T-60: 全頭の 90% 以上に T-60 があり、◎ にもあるレースだけ（基準の人気をそろえる）
        has = np.array([(r, int(h)) in real if pd.notna(h) else False for r, h in zip(fl["race_id"], fl["hn"])])
        share = pd.Series(has).groupby(fl["race_id"].to_numpy()).mean()
        hh = np.array([(r, int(h)) in real for r, h in zip(hm["race_id"], hm["hn"])])
        keep = hm["race_id"].map(share).fillna(0).to_numpy() >= 0.9
        cov |= {"races_t60_cover_ok": int((keep & hh).sum()), "races_t60_partial": int((~keep).sum()), "honmei_no_t60": int((keep & ~hh).sum())}
        hm = hm[keep & hh].reset_index(drop=True)
        fl = fl[fl["race_id"].isin(set(hm["race_id"]))].reset_index(drop=True)
    hm = with_inv_sum(hm, fl)
    dates = hm["date"].astype(str).to_numpy()
    sp = spec()
    out = {"group": name, "races_with_honmei": int(len(hm)), "coverage": cov, "by_gamma": {}}
    for g in gammas:
        D = Draws(hm, fl, pool, g, "place" if rule["bet"] == "fuku" else "win", sp["grid"]["odds_bands"], real=real)
        W = date_weights(dates, D.k, 2)
        a = cand_arrays(hm, D, rule, thr)
        base = {b: cand_arrays(hm, D, {"kind": b, "bet": rule["bet"]}, thr) for b in (("B0", "B_all", "B_fav") if with_fav else ("B0", "B_all"))}
        m = mets(a, dates, D.k, W)
        res = {"cand": _slim(m) | {"diff": {b: BC.paired_diff(a, x, W) for b, x in base.items()}},
               "baselines": {b: _slim(mets(x, dates, D.k, W)) for b, x in base.items()}}
        res["judge"] = judge(m, res["cand"]["diff"]["B0"], min_bets)
        out["by_gamma"][str(g)] = res
    out["judge"] = out["by_gamma"]["0.0"]["judge"]
    out["optimistic"] = {k: _slim(v) for k, v in optimistic(with_inv_sum(hm, fl), fl, [rule], thr, None).items()}
    return out


def cmd_final() -> None:
    """凍結の後に 1 回だけ。prereg_final.json が commit 済みで、凍結した仕様・入力の sha が一致するときだけ流す。印は計算の前に立てる。"""
    import hashlib
    import secrets
    import subprocess
    import sys
    from datetime import datetime
    if BO.MARK.exists():
        sys.exit("評価は済んでいる（1 回だけ）")
    P = json.loads(BO.PREREG.read_text(encoding="utf-8"))
    tools = [BO.CR.REPO / "src/research/course/bet_opt.py", BO.CR.REPO / "src/research/course/bet_opt_search.py"]
    for f in (BO.PREREG, BO.SPEC, *tools):
        r = subprocess.run(["git", "-C", str(BO.CR.REPO), "ls-files", "--error-unmatch", str(f)], capture_output=True)
        if r.returncode != 0 or subprocess.run(["git", "-C", str(BO.CR.REPO), "diff", "--quiet", "HEAD", "--", str(f)]).returncode != 0:
            sys.exit(f"{f.name} が commit されていない")
    if hashlib.sha256(BO.SPEC.read_bytes()).hexdigest()[:12] != P["spec_sha"]:
        sys.exit("search_spec.json が凍結の sha と違う")
    rs = BO.OUT / "search_result.json"
    if hashlib.sha256(rs.read_bytes()).hexdigest()[:12] != P["search_result_sha"]:
        sys.exit("search_result.json が凍結の sha と違う")
    if hashlib.sha256(POOL_FILE.read_bytes()).hexdigest()[:12] != P["drift_pool_sha"]:
        sys.exit("ずれの元（drift_pool_t60.npz）が凍結の sha と違う")
    SR = json.loads(rs.read_text(encoding="utf-8"))
    if {k: SR["best"][k] for k in ("id", "kind", "band", "strength", "bet")} != {k: P["rule"][k] for k in ("id", "kind", "band", "strength", "bet")}:
        sys.exit("prereg の rule が search_result の best と違う")
    P3 = json.loads(BO.V3.PREREG3.read_text(encoding="utf-8"))
    if BO.CR._sha_file(BO.V3.OUT3 / P3["cells_file"]) != P3["sha256_cells"]:
        sys.exit("第 3 版の凍結の条件の表が sha と違う")
    need = [BO.CR.OUT / "eval" / "master" / "master.parquet", BO.V3.OUT3 / "frame.parquet", BO.V3.OUT3 / "levels.parquet", BC.PRE_ODDS,
            BO.CR.REPO / "data/local/graded_picks/history.json"] + \
           [BO.CR.REPO / "data/page_reference/tables" / str(y) / f for y in (2025, 2026) for f in ("race_result_flat.parquet", "race_result_race.parquet")]
    miss = [str(x) for x in need if not x.exists()]
    if miss:
        sys.exit(f"評価に要る表が無い（印を立てる前に止める）: {miss}")
    head = subprocess.run(["git", "-C", str(BO.CR.REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    tok = secrets.token_hex(16)
    BO.MARK.write_text(json.dumps({"started_at": datetime.now(BO.CR.JST).isoformat(timespec="seconds"), "prereg_spec_sha": P["spec_sha"], "head": head,
                                   "token_sha": hashlib.sha256(tok.encode()).hexdigest()}), encoding="utf-8")
    # 通常 2026 年の ◎（子の処理。2026 年の単位の表は CR.env(2026) の子の中だけで作る。重賞の確定オッズと pre_odds は親で読む）
    e = BO.CR.env(2026) | {"FOLKLORE_OUT": str(BO.CR.OUT / "eval" / "master"), "GRADED_OUT": str(BO.CR.OUT / "eval"), "RBO_FINAL_TOKEN": tok}
    subprocess.run([sys.executable, "-m", "src.research.course.bet_opt", "_final2026"], env=e, cwd=BO.CR.REPO, check=True)
    rule = P["rule"]
    thr = {k: (-np.inf if v is None or (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in SR["thresholds"].items()}
    thr["all"] = -np.inf
    pool = frozen_pool()
    pre = pd.read_parquet(BC.PRE_ODDS)
    pre = pre[pre["phase"] == "t60"].sort_values("snapshot_at")
    real = {(str(r), int(h)): (float(o), float(pp)) for r, h, o, pp in pre[["race_id", "horse_number", "win_odds", "popularity"]].itertuples(index=False)
            if pd.notna(o) and float(o) > 0}
    g = {x["id"]: x for x in P["groups"]}
    out = {"evaluated_at": None, "head": head, "prereg": P["registered_at"], "rule": rule, "groups": {}}
    # 通常 2025 年（凍結の ◎ を frame の 2025 年に当てる。第 3 版を選んだ年）
    d, L, _, _ = BO.V3.load3()
    kept = json.loads((BO.V3.OUT3 / P3["cells_file"]).read_text(encoding="utf-8"))
    y = pd.to_numeric(d["year"]).to_numpy() == 2025
    h25, f25 = BO.honmei_rows(d[y].reset_index(drop=True), L[y].reset_index(drop=True), kept, BO.frozen_c3(), "frozen2025")
    h26, f26 = pd.read_parquet(BO.OUT / "final_hm_2026.parquet"), pd.read_parquet(BO.OUT / "final_field_2026.parquet")
    h25, f25 = prep(h25, f25)
    h26, f26 = prep(h26, f26)
    ds = pd.to_datetime(h26["date"].astype(str))
    lo, hi = pd.Timestamp(g["course_main"]["dates"][0]), pd.Timestamp(g["course_main"]["dates"][1])
    sep_lo = pd.Timestamp(g["course_separate"]["dates"][0])
    m_main, m_sep = ((ds >= lo) & (ds <= hi)).to_numpy(), (ds >= sep_lo).to_numpy()
    sub = lambda h, f, m: (h[m].reset_index(drop=True), f[f["race_id"].isin(set(h[m]["race_id"]))].reset_index(drop=True))  # noqa: E731
    gam = tuple(spec()["gamma"]["grid"])
    out["groups"]["course_main"] = run_group("course_main", *sub(h26, f26, m_main), pool, rule, thr, g["course_main"]["min_bets"], gammas=gam)
    out["groups"]["course_separate"] = run_group("course_separate", *sub(h26, f26, m_sep), pool, rule, thr, g["course_separate"]["min_bets"], real=real)
    out["groups"]["course_separate_repro"] = run_group("course_separate_repro", *sub(h26, f26, m_sep), pool, rule, thr, g["course_separate"]["min_bets"])
    out["groups"]["course_2025"] = run_group("course_2025", h25, f25, pool, rule, thr, g["course_2025"]["min_bets"], gammas=gam)
    for yy in (2025, 2026):
        hg, fg = graded_frames((yy,))
        out["groups"][f"graded_{yy}"] = run_group(f"graded_{yy}", hg, fg, pool, rule, thr, g["graded"]["min_bets_per_year"], gammas=gam, with_fav=False)
    hg, fg = graded_frames((2025, 2026))
    out["groups"]["graded_2025_2026"] = run_group("graded_2025_2026", hg, fg, pool, rule, thr, g["graded"]["min_bets"], gammas=gam, with_fav=False)
    out["evaluated_at"] = datetime.now(BO.CR.JST).isoformat(timespec="seconds")
    (BO.OUT / "final_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    mk = json.loads(BO.MARK.read_text(encoding="utf-8"))
    BO.MARK.write_text(json.dumps(mk | {"evaluated_at": out["evaluated_at"]}), encoding="utf-8")
    for k, v in out["groups"].items():
        c = v["by_gamma"]["0.0"]["cand"]
        print(k, v["judge"], "参戦", c.get("bet_races"), "回収", c.get("roi"), "区間", c.get("roi_lo"), c.get("roi_hi"), "上位5除き", c.get("roi_ex_top5"),
              "B0 との差", c["diff"]["B0"], flush=True)

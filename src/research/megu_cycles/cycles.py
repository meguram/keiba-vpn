"""めぐサイクルの各サイクル（仮説・設計・選ぶ期間の中の探し方）。結果は docs/operations/megu_cycles/cycle_NN.json に書く。

進め方（docs/operations/megu_cycles/README.md）:
- 各サイクルは 2017〜19（select_a）だけで設定を選ぶ。試した設定の数を数える。
- 選んだ設定を 2020〜21（select_b）に当て、関門（mc_eval.gate）と並べ替えの帰無を見る。
- 関門を通った案だけ 2022〜24 に当てる（mc_eval.confirm。台帳に confirm、通算 10 候補まで）。

  KEIBA_SESSION=megu-cycles python -m src.research.megu_cycles.cycles run 1
費用 0 円（手元の表だけ）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.research.megu_cycles import mc_data as D
from src.research.megu_cycles import mc_eval as E

DOCS = D.ROOT / "docs/operations/megu_cycles"
SEGS = ("flat", "graded")


# ---------------------------------------------------------------- 共通の部品
# サイクルごとの確率の版（再現のため）。11 以降は v5。13 は v5 の学習の平均（cycle13 の中で差し替える）
CYCLE_RICH = {4: "rich_pred.parquet", 5: "rich_pred_v2.parquet", 6: "rich_pred_v4.parquet", 7: "rich_pred_v4.parquet", 8: "rich_pred_v4.parquet",
              9: "rich_pred_v4.parquet", 10: "rich_pred_v4.parquet", 13: "rich_pred_v5bag.parquet", 15: "rich_pred_v5bag.parquet",
              16: "rich_pred_v6bag.parquet", 17: "rich_pred_v5bag.parquet", 18: "rich_pred_v5bag.parquet"}
RICH_FILE = D.RICH


def _prep(d: pd.DataFrame, rich: bool) -> pd.DataFrame:
    d = d.sort_values(["race_id", "hn"]).reset_index(drop=True)
    d["p_model"] = E.race_norm(d, d["pm_win"].to_numpy(float))
    if rich and RICH_FILE.exists():
        r = pd.read_parquet(RICH_FILE, filters=[("year", ">=", int(d["year"].min())), ("year", "<=", int(d["year"].max()))])
        r = r.drop_duplicates(["race_id", "horse_id"])
        d = d.merge(r[["race_id", "horse_id", "pr_win", "pr_top3"]], on=["race_id", "horse_id"], how="left")
        d["p_rich"] = E.race_norm(d, d["pr_win"].fillna(1e-4).to_numpy(float))      # 欠けは固定の小さな値（読む年の範囲に依らない）
    return d


def load_select(rich: bool = True) -> pd.DataFrame:
    """選ぶ期間（2017〜21）だけ。2022〜24 の行は読まない。"""
    return _prep(D.load_table((2017, 2021)), rich)


def load_all(rich: bool = True) -> pd.DataFrame:
    """確かめ（mc_eval.confirm）に渡す表（2017〜24）。"""
    return _prep(D.load_table((2017, 2024)), rich)


def fit_combo(d: pd.DataFrame, k: int | None = None, col: str = "p_model") -> tuple[float, float]:
    """Benter の第 2 段: p ∝ exp(a·log p_model + b·log q)。q は判定のオッズ（k=None なら 5 通りを縦に並べる）。条件付き logit の最尤。"""
    ks = range(D.N_PRE) if k is None else [k]
    lp = np.log(np.clip(d[col].to_numpy(float), 1e-6, 1))
    y = d["y1"].fillna(0).to_numpy(float)
    rid = pd.factorize(d["race_id"])[0]
    lqs = [np.log(np.clip(E.q_pre(d, kk), 1e-6, 1)) for kk in ks]

    def nll(w):
        tot = 0.0
        for lq in lqs:
            z = w[0] * lp + w[1] * lq
            mx = pd.Series(z).groupby(rid).transform("max").to_numpy()
            lse = np.log(pd.Series(np.exp(z - mx)).groupby(rid).transform("sum").to_numpy()) + mx
            tot -= np.sum(y * (z - lse))
        return tot

    r = minimize(nll, x0=[0.5, 0.5], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-2})
    return float(r.x[0]), float(r.x[1])


def combo_p(d: pd.DataFrame, k: int, a: float, b: float, col: str = "p_model") -> np.ndarray:
    lp = np.log(np.clip(d[col].to_numpy(float), 1e-6, 1))
    lq = np.log(np.clip(E.q_pre(d, k), 1e-6, 1))
    z = a * lp + b * lq
    rid = d["race_id"].to_numpy()
    z = z - pd.Series(z).groupby(rid).transform("max").to_numpy()
    return E.race_norm(d, np.exp(z))


def sub(d: pd.DataFrame, period: str, seg: str) -> pd.DataFrame:
    return d[E.period_mask(d, period) & E.seg_mask(d, seg)].reset_index(drop=True)


def pick_best(rows: list[dict], min_bpy: float = 50) -> dict | None:
    ok = [r for r in rows if r["a"]["roi"] is not None and r["a"]["bets_per_year"] >= min_bpy]
    if not ok:
        return None
    return max(ok, key=lambda r: r["a"]["roi_ci"][0] if r["a"].get("roi_ci") else r["a"]["roi"])


def write_cycle(n: int, body: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / f"cycle_{n:02d}.json").write_text(json.dumps(body, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def short(m: dict | None) -> str:
    if not m:
        return "-"
    ci = m.get("roi_ci")
    mr = m.get("mr_ci")
    return (f"回収率 {m['roi']:.1%}" + (f"（{ci[0]:.0%}〜{ci[1]:.0%}）" if ci else "") + f"・{m['bets_per_year']:.0f} 件/年・市場比 {m['market_ratio']:.2f}"
            + (f"（{mr[0]:.2f}〜{mr[1]:.2f}）" if mr else "") + f"・基準線 {m['base_same_pop']:.1%}") if m.get("roi") is not None else "買い無し"


# ---------------------------------------------------------------- サイクル 1: 自前の確率 × T-60 の再現オッズの期待値
def strat_ev(p_fn, t: float, omax: float, omin: float = 1.0):
    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        p = p_fn(d, k)
        o = d[f"odds_pre{k}"].to_numpy(float)
        return ((p * o >= t) & (o <= omax) & (o >= omin)).astype(float)
    return s


def cycle1(d: pd.DataFrame) -> dict:
    out = {"cycle": 1, "hypothesis": "市場の列を使わない自前の確率（as-of の列の LightGBM）が市場の知らない情報を持つなら、"
           "その確率 × T-60 の再現オッズの期待値が高い馬は、市場どおりより多く勝つ（Benter 1994 の第 1 段）。第 2 段（市場との組み合わせ）も並べる。",
           "segments": {}}
    n_trials = 0
    for seg in SEGS:
        a_d = sub(d, "select_a", seg)
        b_d = sub(d, "select_b", seg)
        ab = fit_combo(a_d)
        variants = {"model": lambda e, k: e["p_model"].to_numpy(float),
                    "combo": lambda e, k, ab=ab: combo_p(e, k, *ab)}
        rows = []
        for (vn, pf), t, omax in itertools.product(variants.items(), (1.0, 1.1, 1.2, 1.3, 1.5), (10.0, 20.0, 50.0)):
            st = strat_ev(pf, t, omax)
            rows.append({"variant": vn, "t": t, "omax": omax, "a": E.evaluate(a_d, st, "win", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"combo_ab_2017_19": ab, "trials": len(rows),
                   "grid": [{k: r[k] for k in ("variant", "t", "omax")} | {"roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"],
                                                                         "mr_a": r["a"]["market_ratio"]} for r in rows]}
        if best:
            pf = variants[best["variant"]]
            st = strat_ev(pf, best["t"], best["omax"])
            b = E.evaluate(b_d, st, "win")
            nl = E.null_perm(b_d, st, ["p_model"], "win", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {k: best[k] for k in ("variant", "t", "omax")}, "select_a": best["a"], "select_b": b, "null_b": nl,
                        "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES = {1: cycle1}


def cmd_run(n: int) -> None:
    global RICH_FILE
    RICH_FILE = D.OUT / CYCLE_RICH.get(n, D.RICH.name)
    t0 = time.time()
    d = load_select()
    body = CYCLES[n](d)
    body["sec"] = round(time.time() - t0)
    body["rich_file"] = str(RICH_FILE)
    write_cycle(n, body)
    for seg, s in body["segments"].items():
        print(seg, s.get("best"), "\n  2017〜19:", short(s.get("select_a")), "\n  2020〜21:", short(s.get("select_b")),
              "\n  帰無:", s.get("null_b"), "\n  関門:", s.get("gate"), s.get("gate_why"), flush=True)





# ---------------------------------------------------------------- サイクル 2: 人気の偏り（T-60 の人気 × クラスの群 × 頭数）
def group_of(d: pd.DataFrame) -> np.ndarray:
    c = pd.to_numeric(d["cls"], errors="coerce").to_numpy()
    return np.where(d["graded"].to_numpy(bool), "重賞", np.where(c <= 1, "新馬未勝利", np.where(c <= 3, "1-2勝", "3勝OP")))


def pop_pre(d: pd.DataFrame, k: int) -> np.ndarray:
    return d.groupby("race_id")[f"odds_pre{k}"].rank(method="first").to_numpy()


def field_band(d: pd.DataFrame) -> np.ndarray:
    f = pd.to_numeric(d["field"], errors="coerce").to_numpy()
    return np.where(f <= 10, "≤10", np.where(f <= 14, "11-14", "15+"))


def strat_cell(grp: str, pp: int, fb: str | None):
    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        m = (group_of(d) == grp) & (pop_pre(d, k) == pp)
        if fb:
            m &= field_band(d) == fb
        return m.astype(float)
    return s


def cycle2(d: pd.DataFrame) -> dict:
    out = {"cycle": 2, "hypothesis": "本命・大穴の偏り（公開の知見: 人気薄が過大に買われる）が、クラスの群・頭数によって強まり、"
           "ある人気の順位（T-60 の再現の人気）の馬が払戻率の壁を越える区画がある。", "segments": {}}
    n_trials = 0
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        grps = ["重賞"] if seg == "graded" else ["新馬未勝利", "1-2勝", "3勝OP"]
        rows = []
        for grp, pp, fb, kind in itertools.product(grps, range(1, 8), (None, "≤10", "11-14", "15+"), ("win", "place")):
            st = strat_cell(grp, pp, fb)
            r = E.evaluate(a_d, st, kind, n_boot=300)
            rows.append({"grp": grp, "pp": pp, "fb": fb, "kind": kind, "a": r})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"trials": len(rows), "top5_a": sorted([{k: r[k] for k in ("grp", "pp", "fb", "kind")} | {"roi_a": r["a"]["roi"], "bpy": r["a"]["bets_per_year"]}
                                                          for r in rows if r["a"]["roi"] is not None and r["a"]["bets_per_year"] >= 50],
                                                         key=lambda x: -x["roi_a"])[:5]}
        if best:
            st = strat_cell(best["grp"], best["pp"], best["fb"])
            b = E.evaluate(b_d, st, best["kind"])
            nl = E.null_perm(b_d, st, ["odds_pre0", "odds_pre1", "odds_pre2", "odds_pre3", "odds_pre4"], best["kind"], n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out["null_b"] = nl
            seg_out |= {"best": {k: best[k] for k in ("grp", "pp", "fb", "kind")}, "select_a": best["a"], "select_b": b, "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[2] = cycle2



# ---------------------------------------------------------------- サイクル 4: 豊富な特徴量の自前の確率（1,231 列の XGBoost）× T-60 の期待値
def cycle4(d: pd.DataFrame) -> dict:
    out = {"cycle": 4, "hypothesis": "サイクル 1 の確率は列が少なく（約 120 列）、市場に足す情報が無かった。本番のモデルと同じ 1,231 列の特徴量で作り直せば、"
           "確定の市場に足す情報（第 2 段の重み）が増え、期待値の判定が効く。", "segments": {}}
    n_trials = 0
    yv = d["y1"].fillna(0).to_numpy()
    ll = {c: float(-np.mean(np.log(np.clip(d[c].to_numpy(float)[yv == 1], 1e-9, 1)))) for c in ("p_model", "p_rich", "q_final")}
    ll["q_pre0"] = float(-np.mean(np.log(np.clip(E.q_pre(d, 0)[yv == 1], 1e-9, 1))))
    out["logloss_2017_21"] = ll
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        ab = fit_combo(a_d, col="p_rich")
        f = a_d.copy()
        for k in range(D.N_PRE):
            f[f"odds_pre{k}"] = f["odds_final"]
        ab_final = fit_combo(f, 0, col="p_rich")
        variants = {"rich": lambda e, k: e["p_rich"].to_numpy(float),
                    "combo": lambda e, k, ab=ab: combo_p(e, k, *ab, col="p_rich")}
        rows = []
        for (vn, pf), t, omax in itertools.product(variants.items(), (1.0, 1.1, 1.2, 1.3, 1.5), (10.0, 20.0, 50.0)):
            st = strat_ev(pf, t, omax)
            rows.append({"variant": vn, "t": t, "omax": omax, "a": E.evaluate(a_d, st, "win", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"combo_ab_t60": ab, "combo_ab_final_ref": ab_final, "trials": len(rows),
                   "grid": [{k: r[k] for k in ("variant", "t", "omax")} | {"roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"],
                                                                         "mr_a": r["a"]["market_ratio"]} for r in rows]}
        if best:
            pf = variants[best["variant"]]
            st = strat_ev(pf, best["t"], best["omax"])
            b = E.evaluate(b_d, st, "win")
            nl = E.null_perm(b_d, st, ["p_rich"], "win", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {k: best[k] for k in ("variant", "t", "omax")}, "select_a": best["a"], "select_b": b, "null_b": nl,
                        "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[4] = cycle4


def cycle5(d: pd.DataFrame) -> dict:
    """サイクル 4 と同じ設計を、リークの列（tl_*・wc_*・rb_nk_*）を除いた確率（rich_pred_v2）で流し直す。年ごとの第 2 段の重みも残す。"""
    out = cycle4(d)
    out["cycle"] = 5
    out["hypothesis"] = "サイクル 4 の作り直し: 調教の記録の有無に依る列を除いた 1,0xx 列の確率でも、確定の市場に足す情報があり、期待値の判定が効く。"
    yw = {}
    for y in range(2017, 2022):
        x = d[d["year"] == y].reset_index(drop=True)
        for k in range(D.N_PRE):
            x[f"odds_pre{k}"] = x["odds_final"]
        yw[y] = fit_combo(x, 0, col="p_rich")
    out["yearly_ab_final"] = yw
    return out


CYCLES[5] = cycle5


def cycle6(d: pd.DataFrame) -> dict:
    """サイクル 5 と同じ設計を、結果を使わない決まりで列を外した確率で流す（実際に流したのは v4。CYCLE_RICH[6]）。2017〜19 で選び直す。"""
    out = cycle5(d)
    out["cycle"] = 6
    out["hypothesis"] = "記録の取れ方が年で変わる列を結果を使わずに外した確率（v3）でも、確定の市場に足す情報があり、期待値の判定が効く。"
    return out


CYCLES[6] = cycle6


# ---------------------------------------------------------------- サイクル 7: レースの見送り × 年の揺れに強い選び方
def race_feat(d: pd.DataFrame, k: int, ab: tuple[float, float], col: str = "p_rich") -> dict[str, np.ndarray]:
    """発走前に分かるレースの値（走ごとに並べる）。favq: T-60 の再現の 1 番人気の q。dis: 自前と市場の確率の差の大きさ（レースの和）。"""
    rid = d["race_id"].to_numpy()
    q = E.q_pre(d, k)
    p = combo_p(d, k, *ab, col=col)
    favq = pd.Series(q).groupby(rid).transform("max").to_numpy()
    dis = pd.Series(np.abs(p - q)).groupby(rid).transform("sum").to_numpy()
    return {"favq": favq, "dis": dis}


def keep_mask(d: pd.DataFrame, k: int, ab, rule: tuple | None, cuts: dict) -> np.ndarray:
    if rule is None:
        return np.ones(len(d), bool)
    name, lv = rule
    if name in ("favq", "dis"):
        v = race_feat(d, k, ab)[name]
        b = np.searchsorted(cuts[name], v, side="right")
        return b == lv
    if name == "field":
        return field_band(d) == lv
    if name == "grp":
        return group_of(d) == lv
    if name == "surf":
        return d["surf"].astype(str).to_numpy() == lv
    if name == "track":
        return (d["track"].astype(str).to_numpy() == "良") == (lv == "良")
    raise ValueError(name)


def strat_ev_skip(ab, t: float, omax: float, rule, cuts, col: str = "p_rich"):
    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        p = combo_p(d, k, *ab, col=col)
        o = d[f"odds_pre{k}"].to_numpy(float)
        return ((p * o >= t) & (o <= omax) & keep_mask(d, k, ab, rule, cuts)).astype(float)
    return s


def yearly_min(d: pd.DataFrame, st, kind: str = "win") -> tuple[float, float]:
    vals, bpy = [], []
    for y in sorted(d["year"].unique()):
        r = E.evaluate(d[d["year"] == y].reset_index(drop=True), st, kind, n_boot=0)
        vals.append(r["roi"] if r["roi"] is not None else 0.0)
        bpy.append(r["bets"])
    return float(min(vals)), float(min(bpy))


def cycle7(d: pd.DataFrame) -> dict:
    out = {"cycle": 7, "hypothesis": "自前の確率 × T-60 の期待値（サイクル 6 の combo）に、発走前に分かるレースの値での見送り（頭数・1 番人気の強さ・"
           "自前と市場の差の大きさ・クラス・芝ダ・馬場）を足すと、期待値の高く見えても当たりにくい・控除負けするレースを避けられる。"
           "選び方は 2017〜19 の年ごとの最小の回収率（年の揺れに強い）。", "segments": {}}
    n_trials = 0
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        ab = fit_combo(a_d, col="p_rich")
        rf = race_feat(a_d, 0, ab)
        first = ~pd.Series(a_d["race_id"]).duplicated().to_numpy()
        cuts = {n: list(np.quantile(v[first], [1 / 3, 2 / 3])) for n, v in rf.items()}
        rules = [None] + [("favq", i) for i in range(3)] + [("dis", i) for i in range(3)] + [("field", x) for x in ("≤10", "11-14", "15+")]
        rules += [("surf", x) for x in ("芝", "ダ")] + [("track", x) for x in ("良", "良以外")]
        if seg == "flat":
            rules += [("grp", x) for x in ("新馬未勝利", "1-2勝", "3勝OP")]
        rows = []
        for t, omax, rule in itertools.product((1.1, 1.2, 1.3, 1.5, 1.7, 2.0), (20.0, 50.0), rules):
            st = strat_ev_skip(ab, t, omax, rule, cuts)
            ymin, bmin = yearly_min(a_d, st)
            rows.append({"t": t, "omax": omax, "rule": rule, "ymin": ymin, "bmin": bmin})
            n_trials += 1
        ok = [r for r in rows if r["bmin"] >= 50]
        seg_out = {"combo_ab": ab, "cuts": cuts, "trials": len(rows),
                   "top5": sorted(ok, key=lambda r: -r["ymin"])[:5]}
        if ok:
            best = max(ok, key=lambda r: r["ymin"])
            st = strat_ev_skip(ab, best["t"], best["omax"], best["rule"], cuts)
            a = E.evaluate(a_d, st, "win")
            b = E.evaluate(b_d, st, "win")
            nl = E.null_perm(b_d, st, ["p_rich"], "win", n=30)
            g_ok, why = E.gate(a, b, nl)
            seg_out |= {"best": {k: best[k] for k in ("t", "omax", "rule")}, "select_a": a, "select_b": b, "null_b": nl, "gate": g_ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[7] = cycle7


# ---------------------------------------------------------------- サイクル 8: 第 2 段の重みの交差の当てはめ
def crossfit_combo(a_d: pd.DataFrame, col: str = "p_rich") -> dict:
    """2017〜19 の年ごとに、その年を抜いた 2 年で a・b を当てはめる（同じ年で当てはめて数える楽観を除く）。"""
    out = {}
    for y in sorted(a_d["year"].unique()):
        out[int(y)] = fit_combo(a_d[a_d["year"] != y].reset_index(drop=True), col=col)
    return out


def strat_ev_cf(abs_by_year: dict | None, ab_all, t: float, omax: float, col: str = "p_rich"):
    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        p = np.zeros(len(d))
        yrs = d["year"].to_numpy()
        for y in np.unique(yrs):
            ab = abs_by_year.get(int(y), ab_all) if abs_by_year else ab_all
            m = yrs == y
            p[m] = combo_p(d[m].reset_index(drop=True), k, *ab, col=col)
        o = d[f"odds_pre{k}"].to_numpy(float)
        return ((p * o >= t) & (o <= omax)).astype(float)
    return s


def cycle8(d: pd.DataFrame) -> dict:
    out = {"cycle": 8, "hypothesis": "サイクル 6 の 2017〜19 の高さの一部は、第 2 段の重み a・b を同じ 2017〜19 で当てはめて数えた楽観。"
           "年を 1 つずつ抜いて当てはめる（交差の当てはめ）と、2017〜19 の値は下がり、2020〜21 との差が縮む。", "segments": {}}
    n_trials = 0
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        ab_all = fit_combo(a_d, col="p_rich")
        cf = crossfit_combo(a_d)
        rows = []
        for t, omax in itertools.product((1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0), (10.0, 20.0, 50.0)):
            rows.append({"t": t, "omax": omax, "a": E.evaluate(a_d, strat_ev_cf(cf, ab_all, t, omax), "win", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"ab_all": ab_all, "ab_crossfit": {str(k): v for k, v in cf.items()}, "trials": len(rows),
                   "grid": [{"t": r["t"], "omax": r["omax"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
        if best:
            st_b = strat_ev_cf(None, ab_all, best["t"], best["omax"])
            b = E.evaluate(b_d, st_b, "win")
            nl = E.null_perm(b_d, st_b, ["p_rich"], "win", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {"t": best["t"], "omax": best["omax"]}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[8] = cycle8


# ---------------------------------------------------------------- サイクル 9: T-60 の市場を offset にした学習の確率
def attach_offset(d: pd.DataFrame) -> pd.DataFrame:
    o = pd.read_parquet(D.OFFSET, filters=[("year", ">=", int(d["year"].min())), ("year", "<=", int(d["year"].max()))]).drop_duplicates(["race_id", "horse_id"])
    d = d.merge(o[["race_id", "horse_id"] + [f"po{k}" for k in range(D.N_PRE)]], on=["race_id", "horse_id"], how="left")
    for k in range(D.N_PRE):
        q = E.q_pre(d, k)
        d[f"po{k}"] = d[f"po{k}"].fillna(pd.Series(q))          # 欠け（特徴量の表に無い走）は市場の確率のまま
        d[f"pk{k}"] = E.race_norm(d, d[f"po{k}"].to_numpy(float))
    return d


def strat_ev_off(t: float, omax: float):
    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        p = d[f"pk{k}"].to_numpy(float)
        o = d[f"odds_pre{k}"].to_numpy(float)
        return ((p * o >= t) & (o <= omax)).astype(float)
    return s


def cycle9(d: pd.DataFrame) -> dict:
    out = {"cycle": 9, "hypothesis": "第 2 段を a・b の 2 つの数でなく、T-60 の再現の市場を offset にした学習（市場の知らない分だけを特徴量から学ぶ）にすると、"
           "確率が市場に対して較正され、期待値の判定が安定する（2017〜19 と 2020〜21 の差が縮む）。", "segments": {}}
    d = attach_offset(d)
    yv = d["y1"].fillna(0).to_numpy()
    out["logloss_2017_21"] = {"pk0": float(-np.mean(np.log(np.clip(d["pk0"].to_numpy()[yv == 1], 1e-9, 1)))),
                              "q_pre0": float(-np.mean(np.log(np.clip(E.q_pre(d, 0)[yv == 1], 1e-9, 1))))}
    n_trials = 0
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        rows = []
        for t, omax in itertools.product((1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0), (10.0, 20.0, 50.0)):
            rows.append({"t": t, "omax": omax, "a": E.evaluate(a_d, strat_ev_off(t, omax), "win", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"trials": len(rows), "grid": [{"t": r["t"], "omax": r["omax"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
        if best:
            st = strat_ev_off(best["t"], best["omax"])
            b = E.evaluate(b_d, st, "win")
            nl = E.null_perm(b_d, st, [f"pk{k}" for k in range(D.N_PRE)], "win", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {"t": best["t"], "omax": best["omax"]}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[9] = cycle9


# ---------------------------------------------------------------- サイクル 10: 複勝（自前の 3 着内の確率 × 想定の複勝の払戻）
def place_payout_map(a_d: pd.DataFrame) -> dict:
    """2017〜19 の、当たった複勝の払戻の平均を、T-60 の再現の単勝オッズの帯 × 頭数の帯ごとに（版 0）。発走前の想定の複勝の払戻に使う。"""
    hit = E._hit(a_d, "place") > 0
    o = a_d["odds_pre0"].to_numpy(float)
    band = np.searchsorted(E.ODDS_BANDS[1:-1], o, side="right")
    fb = field_band(a_d)
    t = pd.DataFrame({"b": band, "f": fb, "r": a_d["ret_fuku"].to_numpy(float)})[hit]
    by_b = t.groupby("b")["r"].mean()
    out = {}
    for (b, f), g in t.groupby(["b", "f"])["r"]:
        out[f"{b}|{f}"] = float(g.mean()) if len(g) >= 30 else float(by_b[b])     # 当たり 30 件未満の枠は帯だけの平均（跳ねを抑える）
    return out


def p3_combo(d: pd.DataFrame, k: int, ab3: tuple[float, float]) -> np.ndarray:
    """3 着内の確率: 自前の 3 着内（pr_top3 をレースで払われる着数の和に直す）と、市場（T-60）の Harville の 3 着内の logit の組み合わせ。"""
    slots = E.place_slots(d)
    pm = d["pr_top3"].fillna(1e-3).to_numpy(float)
    pm = np.clip(E.race_norm(d, pm) * slots, 1e-4, 1 - 1e-4)
    hq = np.clip(E.harville_top3_cached(d, k), 1e-4, 1 - 1e-4)
    z = ab3[0] * np.log(pm / (1 - pm)) + ab3[1] * np.log(hq / (1 - hq))
    return 1 / (1 + np.exp(-z))


def fit_p3(a_d: pd.DataFrame) -> tuple[float, float]:
    y = E._hit(a_d, "place")
    slots = E.place_slots(a_d)
    pm = np.clip(E.race_norm(a_d, a_d["pr_top3"].fillna(1e-3).to_numpy(float)) * slots, 1e-4, 1 - 1e-4)
    hs = [np.clip(E.harville_top3_cached(a_d, k), 1e-4, 1 - 1e-4) for k in range(D.N_PRE)]

    def nll(w):
        tot = 0.0
        for hq in hs:
            z = w[0] * np.log(pm / (1 - pm)) + w[1] * np.log(hq / (1 - hq))
            p = np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
            tot -= np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return tot

    r = minimize(nll, x0=[0.5, 0.5], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-2})
    return float(r.x[0]), float(r.x[1])


def strat_place(ab3, pmap: dict, t: float, omax: float):
    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        p3 = p3_combo(d, k, ab3)
        o = d[f"odds_pre{k}"].to_numpy(float)
        band = np.searchsorted(E.ODDS_BANDS[1:-1], o, side="right")
        fb = field_band(d)
        pay = np.array([pmap.get(f"{b}|{f}", np.nan) for b, f in zip(band, fb)])
        return ((p3 * pay >= t) & (o <= omax)).astype(float)
    return s


def cycle10(d: pd.DataFrame) -> dict:
    out = {"cycle": 10, "hypothesis": "複勝は当たりが多く、年の揺れが小さい。自前の 3 着内の確率（豊富な特徴量）と市場の Harville を組み合わせ、"
           "想定の複勝の払戻（T-60 の単勝オッズの帯 × 頭数の帯の、2017〜19 の当たりの平均）との期待値で買うと、単勝より差の小さい買い方になる。", "segments": {}}
    n_trials = 0
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        ab3 = fit_p3(a_d)
        pmap = place_payout_map(a_d)
        rows = []
        for t, omax in itertools.product((1.0, 1.05, 1.1, 1.2, 1.3, 1.5), (5.0, 10.0, 20.0, 50.0)):
            rows.append({"t": t, "omax": omax, "a": E.evaluate(a_d, strat_place(ab3, pmap, t, omax), "place", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"ab3": ab3, "trials": len(rows), "grid": [{"t": r["t"], "omax": r["omax"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
        if best:
            st = strat_place(ab3, pmap, best["t"], best["omax"])
            b = E.evaluate(b_d, st, "place")
            nl = E.null_perm(b_d, st, ["pr_top3"], "place", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {"t": best["t"], "omax": best["omax"]}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[10] = cycle10


# ---------------------------------------------------------------- 判定のオッズの感度: SmartRC の想定人気（2021 年だけ値がある）
PRERACE = D.Path("/home/jovyan/work/folklore_out/prerace.parquet")


def smartrc_check(b_d: pd.DataFrame, strat, kind: str = "win", year: int | None = 2021) -> dict | None:
    """判定のオッズを SmartRC の想定の勝率（E_q）から作った想定オッズ（0.8 / E_q）に替えて数える（確定の並びを使わない判定）。
    既定は 2021 年（選ぶ期間）。year=None なら b_d の年すべて（2024 年まで）。"""
    if not PRERACE.exists():
        return None
    flt = [("year", "==", year)] if year else [("year", ">=", int(b_d["year"].min())), ("year", "<=", min(int(b_d["year"].max()), D.YEAR_END))]
    pr = pd.read_parquet(PRERACE, columns=["race_id", "horse_id", "year", "E_q"], filters=flt).dropna(subset=["E_q"])
    pr["race_id"], pr["horse_id"] = pr["race_id"].astype(str), pr["horse_id"].astype(str)
    e = (b_d[b_d["year"] == year] if year else b_d).merge(pr[["race_id", "horse_id", "E_q"]], on=["race_id", "horse_id"], how="inner")
    e = e[e.groupby("race_id")["race_id"].transform("size") == e.groupby("race_id")["field"].transform("first")]   # 全頭そろったレースだけ
    if e.empty:
        return None
    e = e.sort_values(["race_id", "hn"]).reset_index(drop=True)
    q = E.race_norm(e, e["E_q"].to_numpy(float))
    for k in range(D.N_PRE):
        e[f"odds_pre{k}"] = 0.8 / np.clip(q, 1e-4, None)
    r = E.evaluate(e, strat, kind, n_boot=1000)
    r["races_covered"] = int(e["race_id"].nunique())
    return r


def cycle11(d: pd.DataFrame) -> dict:
    """サイクル 6 の設計（combo × 期待値、2017〜19 の区間の下限で選ぶ）を、mtl_* と水準の段の列も外した確率 v5 で 1 回だけ流す。"""
    out = cycle5(d)
    out["cycle"] = 11
    out["hypothesis"] = ("2018・19 の上積みの山は mtl_*（上流の補助モデル）が作る。これと水準の段の列を外した確率（v5）なら、"
                         "2017〜19 と 2020〜21 の差が縮み、関門の 0.9 倍を越える。縮まなければ、この確率の族は壁として打ち切る。")
    for seg, v in out["segments"].items():
        if v.get("best"):
            ab = v["combo_ab_t60"]
            st = strat_ev(lambda e, k, ab=tuple(ab): combo_p(e, k, *ab, col="p_rich"), v["best"]["t"], v["best"]["omax"]) if v["best"]["variant"] == "combo" \
                else strat_ev(lambda e, k: e["p_rich"].to_numpy(float), v["best"]["t"], v["best"]["omax"])
            v["smartrc_2021"] = smartrc_check(sub(d, "select_b", seg), st)
    return out


CYCLES[11] = cycle11


# ---------------------------------------------------------------- サイクル 12: 3 つの確率の組み合わせ（v5・約 120 列の確率・T-60 の市場）
def fit_combo3(d: pd.DataFrame) -> tuple[float, float, float]:
    l1 = np.log(np.clip(d["p_rich"].to_numpy(float), 1e-6, 1))
    l2 = np.log(np.clip(d["p_model"].to_numpy(float), 1e-6, 1))
    y = d["y1"].fillna(0).to_numpy(float)
    rid = pd.factorize(d["race_id"])[0]
    lqs = [np.log(np.clip(E.q_pre(d, k), 1e-6, 1)) for k in range(D.N_PRE)]

    def nll(w):
        tot = 0.0
        for lq in lqs:
            z = w[0] * l1 + w[1] * l2 + w[2] * lq
            mx = pd.Series(z).groupby(rid).transform("max").to_numpy()
            lse = np.log(pd.Series(np.exp(z - mx)).groupby(rid).transform("sum").to_numpy()) + mx
            tot -= np.sum(y * (z - lse))
        return tot

    r = minimize(nll, x0=[0.4, 0.1, 0.5], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-2})
    return tuple(float(x) for x in r.x)


def combo3_p(d: pd.DataFrame, k: int, w) -> np.ndarray:
    z = (w[0] * np.log(np.clip(d["p_rich"].to_numpy(float), 1e-6, 1)) + w[1] * np.log(np.clip(d["p_model"].to_numpy(float), 1e-6, 1))
         + w[2] * np.log(np.clip(E.q_pre(d, k), 1e-6, 1)))
    z = z - pd.Series(z).groupby(d["race_id"].to_numpy()).transform("max").to_numpy()
    return E.race_norm(d, np.exp(z))


def cycle12(d: pd.DataFrame) -> dict:
    out = {"cycle": 12, "hypothesis": "v5 の確率と、別の列から作った約 120 列の確率（frame の as-of の列）は誤差の向きが違う。3 つ（v5・120 列・T-60 の市場）を"
           "条件付き logit で組むと確率が鋭くなり、同じ期待値の水準で件数が増えて、2020〜21 の区間が狭まる。", "segments": {}}
    n_trials = 0
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        w = fit_combo3(a_d)
        rows = []
        for t, omax in itertools.product((1.1, 1.2, 1.3, 1.5, 1.7), (20.0, 50.0)):
            st = strat_ev(lambda e, k, w=w: combo3_p(e, k, w), t, omax)
            rows.append({"t": t, "omax": omax, "a": E.evaluate(a_d, st, "win", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"w": w, "trials": len(rows), "grid": [{"t": r["t"], "omax": r["omax"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
        if best:
            st = strat_ev(lambda e, k, w=w: combo3_p(e, k, w), best["t"], best["omax"])
            b = E.evaluate(b_d, st, "win")
            nl = E.null_perm(b_d, st, ["p_rich", "p_model"], "win", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {"t": best["t"], "omax": best["omax"]}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why,
                        "smartrc_2021": smartrc_check(b_d, st)}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[12] = cycle12


# ---------------------------------------------------------------- サイクル 14: 単勝＋複勝（同じ馬に半分ずつ）
def cycle14(d: pd.DataFrame) -> dict:
    out = {"cycle": 14, "hypothesis": "単勝の期待値で選んだ馬（v5 の combo）に、単勝と複勝を半分ずつ賭けると、当たりの数が増えて区間が狭まる。"
           "複勝の払戻率の壁（約 80%）で回収率は下がるが、選んだ馬は複勝でも市場より当たる（サイクル 10 の市場比 1.2〜1.3）。", "segments": {}}
    n_trials = 0
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        ab = fit_combo(a_d, col="p_rich")
        rows = []
        for t, omax in itertools.product((1.2, 1.3, 1.5, 1.7), (20.0, 50.0)):
            st = strat_ev(lambda e, k, ab=ab: combo_p(e, k, *ab, col="p_rich"), t, omax)
            rows.append({"t": t, "omax": omax, "a": E.evaluate(a_d, st, "winplace", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"ab": ab, "trials": len(rows), "grid": [{"t": r["t"], "omax": r["omax"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
        if best:
            st = strat_ev(lambda e, k, ab=ab: combo_p(e, k, *ab, col="p_rich"), best["t"], best["omax"])
            b = E.evaluate(b_d, st, "winplace")
            nl = E.null_perm(b_d, st, ["p_rich"], "winplace", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {"t": best["t"], "omax": best["omax"]}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[14] = cycle14


def cycle13(d: pd.DataFrame) -> dict:
    """サイクル 11 と同じ設計を、v5 の列で設定の違う XGBoost 3 本の平均（rich_pred_v5bag）で流す。"""
    out = cycle11(d)
    out["cycle"] = 13
    out["hypothesis"] = ("v5 の確率を設定の違う 3 本の学習の平均にすると、1 本の学習の雑音が減って確率が鋭くなり、"
                         "同じ期待値の水準で市場比が上がり、2020〜21 の市場比の区間の下限が 1 を越える。")
    return out


CYCLES[13] = cycle13


def cycle15(d: pd.DataFrame) -> dict:
    """サイクル 12 の設計（3 つの確率の組み合わせ）を、学習の平均（v5bag）で流す。重賞で件数と区間を狙う。"""
    out = cycle12(d)
    out["cycle"] = 15
    out["hypothesis"] = ("サイクル 12 の重賞（103% → 104%、市場比の下限 0.93 だけで不合格）は、確率の雑音が区間を広げていた。"
                         "学習の平均（v5bag）で組むと、重賞でも市場比が上がり区間の下限が 1 を越える。")
    return out


CYCLES[15] = cycle15


def cycle16(d: pd.DataFrame) -> dict:
    """サイクル 13 の設計を、汚れの少ない列（v6: s_*・sr・f3・f2・e・pc も外す）の学習の平均で流す。"""
    out = cycle11(d)
    out["cycle"] = 16
    out["hypothesis"] = ("c13_flat の効きは、汚れのある族（未来の年を含む上流の s_*、2021 年末までで当てはめた sr・f3 など）に頼っていない。"
                         "それらを外した学習の平均でも、2017〜19・2020〜21 で同じ水準が出る。")
    return out


# ---------------------------------------------------------------- サイクル 17: 期待値に応じた賭け金（上限付きの Kelly）
def strat_kelly(ab, t: float, omax: float, frac: float, cap: float):
    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        p = combo_p(d, k, *ab, col="p_rich")
        o = d[f"odds_pre{k}"].to_numpy(float)
        f = (p * o - 1) / np.clip(o - 1, 1e-6, None)           # Kelly の賭けの割合
        w = np.clip(frac * f / 0.01, 0, cap)                     # 1% の Kelly を 1 単位に。上限 cap 単位
        return np.where((p * o >= t) & (o <= omax), np.maximum(w, 1e-9), 0.0)
    return s


def cycle17(d: pd.DataFrame) -> dict:
    out = {"cycle": 17, "hypothesis": "期待値が高い馬ほど多く賭ける（上限付きの分数 Kelly）と、上積みの大きい馬の重みが増えて回収率が上がる。"
           "期待値の推定が粗いなら、かえって下がる（その確かめ）。確率は v5bag、第 2 段は c13 と同じ当てはめ。", "segments": {}}
    n_trials = 0
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        ab = fit_combo(a_d, col="p_rich")
        rows = []
        for t, frac, cap in itertools.product((1.1, 1.2, 1.3), (0.25, 0.5, 1.0), (3.0, 10.0)):
            st = strat_kelly(ab, t, 50.0, frac, cap)
            rows.append({"t": t, "frac": frac, "cap": cap, "a": E.evaluate(a_d, st, "win", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"ab": ab, "trials": len(rows), "grid": [{k: r[k] for k in ("t", "frac", "cap")} | {"roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
        if best:
            st = strat_kelly(ab, best["t"], 50.0, best["frac"], best["cap"])
            b = E.evaluate(b_d, st, "win")
            nl = E.null_perm(b_d, st, ["p_rich"], "win", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {k: best[k] for k in ("t", "frac", "cap")}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[16] = cycle16
CYCLES[17] = cycle17


# ---------------------------------------------------------------- サイクル 18: 判定の時点を T-5 に（発走 5 分前の再現オッズ）
def swap_phase(d: pd.DataFrame, phase: str) -> pd.DataFrame:
    o = pd.read_parquet(D.pre_phase_path(phase)).drop_duplicates(["race_id", "horse_id"])
    d = d.drop(columns=[f"odds_pre{k}" for k in range(D.N_PRE)]).merge(o, on=["race_id", "horse_id"], how="left")
    assert d[[f"odds_pre{k}" for k in range(D.N_PRE)]].notna().all().all()
    return d.sort_values(["race_id", "hn"]).reset_index(drop=True)


def cycle18(d: pd.DataFrame) -> dict:
    out = cycle11(swap_phase(d, "t05"))
    out["cycle"] = 18
    out["phase"] = "t05"
    out["hypothesis"] = ("判定の時点を T-60 から T-5 に遅らせると、判定のオッズが確定に近づき（終盤の資金の情報が入る）、"
                         "期待値の判定の雑音が減って回収率が上がる。運用は発走 5 分前の購入（自動の購入なら可能）。")
    return out


CYCLES[18] = cycle18


# ---------------------------------------------------------------- 2022〜24 に当てる（関門を通り、凍結した候補だけ）
CAND = D.ROOT / "config/megu_cycles/candidates"


def cmd_confirm(name: str) -> None:
    global RICH_FILE
    spec = json.loads((CAND / f"{name}.json").read_text(encoding="utf-8"))
    RICH_FILE = D.OUT / spec["probability"]["file"]
    import hashlib
    for key, path in (("probability", RICH_FILE), ("table", D.TABLE)):
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if h != spec[key]["sha256_16"]:
            raise SystemExit(f"{path} の sha が凍結と違う")
    d_all = load_all()
    if spec.get("phase", "t60") != "t60":
        d_all = swap_phase(d_all, spec["phase"])
        h = hashlib.sha256(D.pre_phase_path(spec["phase"]).read_bytes()).hexdigest()[:16]
        if h != spec["phase_file_sha256_16"]:
            raise SystemExit("判定の時点の再現オッズの表の sha が凍結と違う")
    assert RICH_FILE.name == spec["probability"]["file"] and d_all["pr_win"].notna().mean() > 0.99
    ab = tuple(spec["combo_ab_t60"])
    st = strat_ev(lambda e, k: combo_p(e, k, *ab, col="p_rich"), spec["t"], spec["omax"])

    def extra(e: pd.DataFrame) -> dict:
        """副の読み: 判定のオッズを SmartRC の想定（0.8 / E_q、全頭そろったレースだけ）に替えて、2022〜24 を年ごと・通しで数える。"""
        out = {}
        for y in (2022, 2023, 2024):
            r = smartrc_check(e[e["year"] == y].reset_index(drop=True), st, spec["kind"], year=y)
            out[str(y)] = r
        r = smartrc_check(e, st, spec["kind"], year=None)
        out["all"] = r
        return out

    row = E.confirm(d_all, st, spec, spec["kind"], spec["segment"], extra=extra,
                    reason=f"候補 {name} を 2022〜24 に 1 回だけ当てる（関門・帰無 200 回・SmartRC・別のエージェントの見直しの後）")
    print(json.dumps({k: row[k] for k in ("version", "goal_met", "why")}, ensure_ascii=False))
    print("選ぶ期間（2017〜21）:", short(row["select"]))
    print("確かめ（2022〜24）:", short(row["confirm"]))
    sm = row["confirm"].get("extra", {})
    for k, v in sm.items():
        print("  SmartRC", k, short(v) if v else "-")
    sec = (sm.get("all") or {}).get("roi")
    final = bool(row["goal_met"] and sec is not None and sec >= 0.95)       # 当てる前に決めた合格の線（主と副の両方）
    log = E.confirm_log()
    log[-1]["secondary_smartrc_roi"] = sec
    log[-1]["pass_final"] = final
    E.CONFIRM_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("合格（主と副）:", final)


def cmd_md(n: int, title: str) -> None:
    """cycle_NN.json から cycle_NN.md の頭（結論・仮説・結果の表）を作る。課題・次に生かすこと・打ち切りは手で足す。"""
    b = json.loads((DOCS / f"cycle_{n:02d}.json").read_text(encoding="utf-8"))
    segs = b["segments"]
    passed = [k for k, v in segs.items() if v.get("gate")]
    head = (f"**結論: 関門を通った区分 {'・'.join('平場' if k == 'flat' else '重賞' for k in passed)}。**" if passed
            else "**結論: 不合格（関門で落ちた）。2022〜24 には当てていない。**")
    L = [f"# サイクル {n}: {title}", "", head, "", "## 仮説", b.get("hypothesis", ""), "", f"## 試した数: {b.get('n_trials')}", "", "## 結果",
         "| 区分 | 選んだ設定 | 2017〜19 | 2020〜21 | 帰無 p | 関門 |", "|---|---|---|---|---|---|"]
    for k, v in segs.items():
        nl = v.get("null_b") or {}
        L.append(f"| {'平場' if k == 'flat' else '重賞'} | {json.dumps(v.get('best'), ensure_ascii=False)} | {short(v.get('select_a'))} | {short(v.get('select_b'))} | "
                 f"{nl.get('p', '-') if not isinstance(nl.get('p'), float) else round(nl['p'], 3)} | {'通過' if v.get('gate') else ('不合格: ' + '・'.join(v['gate_why']) if v.get('gate_why') else '判定なし（2017〜19 で年 50 件以上の設定が無い）')} |")
    (DOCS / f"cycle_{n:02d}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "md", "confirm"])
    ap.add_argument("n")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(int(a.n))
    elif a.cmd == "md":
        cmd_md(int(a.n), a.title)
    else:
        cmd_confirm(a.n)


if __name__ == "__main__":
    sys.exit(main())

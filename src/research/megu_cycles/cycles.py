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
              16: "rich_pred_v6bag.parquet", 17: "rich_pred_v5bag.parquet", 18: "rich_pred_v5bag.parquet", 19: "rich_pred_v5bag.parquet", 20: "rich_pred_v5bag.parquet", 21: "rich_pred_v5bag.parquet", 22: "rich_pred_v5bag.parquet", 23: "rich_pred_v5bag.parquet", 24: "rich_pred_v7bag.parquet", 29: "rich_pred_v5bag.parquet", 44: "rich_pred_v5bag.parquet", 48: "rich_pred_v5bag.parquet", 49: "rich_pred_v5bag.parquet"}
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
    cfgp = DOCS / f"cycle_{n:02d}.cfg.json"
    cfg = json.loads(cfgp.read_text(encoding="utf-8")) if cfgp.exists() else {}
    RICH_FILE = D.OUT / cfg.get("rich", CYCLE_RICH.get(n, D.RICH.name))
    t0 = time.time()
    d = load_select()
    body = CYCLES[n](d) if n in CYCLES else cycle_generic(d, n)
    body["sec"] = round(time.time() - t0)
    body["rich_file"] = str(RICH_FILE) if n not in (20, 43, 50) else "候補ごと（config/megu_cycles/candidates の probability.file）"
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


# ---------------------------------------------------------------- サイクル 19: 重賞を、平場で当てはめた第 2 段で買う（件数の壁への対応）
def cycle19(d: pd.DataFrame) -> dict:
    out = {"cycle": 19, "hypothesis": "重賞だけで第 2 段の重みを当てはめると件数が少なく不安定（サイクル 11〜15）。平場の 2017〜19 で当てはめた重み"
           "（同じ自前の確率・同じ市場）をそのまま重賞に使い、判定の時点を T-60・T-5 の 2 通りで、単勝・単勝＋複勝を見る。", "segments": {}}
    n_trials = 0
    rows_all = {}
    for phase in ("t60", "t05"):
        e = d if phase == "t60" else swap_phase(d, phase)
        ab = fit_combo(sub(e, "select_a", "flat"), col="p_rich")
        a_d, b_d = sub(e, "select_a", "graded"), sub(e, "select_b", "graded")
        for kind, t, omax in itertools.product(("win", "winplace"), (1.0, 1.1, 1.2, 1.3), (20.0, 50.0)):
            st = strat_ev(lambda x, k, ab=ab: combo_p(x, k, *ab, col="p_rich"), t, omax)
            rows_all[(phase, kind, t, omax)] = {"phase": phase, "kind": kind, "t": t, "omax": omax, "ab": ab, "a": E.evaluate(a_d, st, kind, n_boot=500)}
            n_trials += 1
    rows = list(rows_all.values())
    best = pick_best(rows)
    seg_out = {"trials": len(rows), "grid": [{k: r[k] for k in ("phase", "kind", "t", "omax")} | {"roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
    if best:
        e = d if best["phase"] == "t60" else swap_phase(d, best["phase"])
        b_d = sub(e, "select_b", "graded")
        ab = best["ab"]
        st = strat_ev(lambda x, k, ab=ab: combo_p(x, k, *ab, col="p_rich"), best["t"], best["omax"])
        b = E.evaluate(b_d, st, best["kind"])
        nl = E.null_perm(b_d, st, ["p_rich"], best["kind"], n=30)
        ok, why = E.gate(best["a"], b, nl)
        seg_out |= {"best": {k: best[k] for k in ("phase", "kind", "t", "omax", "ab")}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why}
    out["segments"]["graded"] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[19] = cycle19


# ---------------------------------------------------------------- サイクル 20: 候補の中身の一貫性（選ぶ期間 2017〜21 だけ。探さない）
def cycle20(d: pd.DataFrame) -> dict:
    out = {"cycle": 20, "hypothesis": "c13_flat・c18_flat_t05 の効きが本物なら、選ぶ期間（2017〜21）の中で、年・芝ダ・クラスの群・頭数・オッズの帯のどこでも"
           "市場比が 1 を超え、一部の区画だけの偶然ではない。探し直しはしない（設定は凍結のまま）。", "segments": {}, "n_trials": 0}
    cands = {}
    for name in ("c13_flat", "c18_flat_t05"):
        spec = json.loads((CAND / f"{name}.json").read_text(encoding="utf-8"))
        e = d if spec.get("phase", "t60") == "t60" else swap_phase(d, spec["phase"])
        e = e[E.seg_mask(e, "flat")].reset_index(drop=True)
        ab = tuple(spec["combo_ab_t60"])
        st = strat_ev(lambda x, k, ab=ab: combo_p(x, k, *ab, col="p_rich"), spec["t"], spec["omax"])
        S = np.vstack([st(e, k) for k in range(D.N_PRE)]).mean(0)
        e = e.assign(_s=S, _grp=group_of(e), _fb=field_band(e),
                     _ob=pd.cut(e["odds_final"], [0, 3, 5, 10, 20, 50, 1e9]).astype(str))
        tab = {}
        # 馬ごとの値（オッズの帯）は、レースの中の正規化が崩れるので、全体で決めた賭け金（_s）をそのまま帯で足す
        rows = {}
        for lv, g in e.groupby("_ob"):
            st_ = g["_s"].sum()
            rows[str(lv)] = {"roi": float((g["_s"] * g["ret_tan"].fillna(0)).sum() / st_) if st_ > 0 else None, "bets": float((g["_s"] > 0).sum()),
                             "mr": float((g["_s"] * g["y1"].fillna(0)).sum() / (g["_s"] * g["q_final"]).sum()) if st_ > 0 else None, "base": None}
        tab["_ob"] = rows
        for col in ("year", "surf", "_grp", "_fb"):        # レース単位の値だけ（レースを丸ごと含む）
            rows = {}
            for lv, g in e.groupby(col):
                g = g.reset_index(drop=True)
                r = E.evaluate(g, st, "win", n_boot=300)
                rows[str(lv)] = {"roi": r["roi"], "bets": r["bets"], "mr": r["market_ratio"], "mr_ci": r.get("mr_ci"), "base": r["base_same_pop"]}
            tab[col] = rows
        cands[name] = tab
    out["segments"] = {}
    out["consistency"] = cands
    return out


CYCLES[20] = cycle20


# ---------------------------------------------------------------- サイクル 21: 新馬・未勝利だけ（情報の薄い市場）
def cycle21(d: pd.DataFrame) -> dict:
    out = {"cycle": 21, "hypothesis": "新馬・未勝利は走った記録が少なく、市場の情報が薄い（公開の知見）。自前の確率の上積みはそこで大きい。"
           "新馬・未勝利だけで第 2 段を当てはめ、期待値で買うと、平場全体より回収率が高く、年 50 件以上を保てる。"
           "注: サイクル 20 で 2017〜21 の群ごとの値を見たので、2020〜21 の関門は汚れている。", "segments": {}}
    n_trials = 0
    for phase in ("t60", "t05"):
        e = d if phase == "t60" else swap_phase(d, phase)
        e = e[group_of(e) == "新馬未勝利"].reset_index(drop=True)
        a_d, b_d = sub(e, "select_a", "flat"), sub(e, "select_b", "flat")
        ab = fit_combo(a_d, col="p_rich")
        rows = []
        for t, omax in itertools.product((1.1, 1.2, 1.3, 1.5), (20.0, 50.0)):
            st = strat_ev(lambda x, k, ab=ab: combo_p(x, k, *ab, col="p_rich"), t, omax)
            rows.append({"t": t, "omax": omax, "a": E.evaluate(a_d, st, "win", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"ab": ab, "trials": len(rows), "grid": [{"t": r["t"], "omax": r["omax"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
        if best:
            st = strat_ev(lambda x, k, ab=ab: combo_p(x, k, *ab, col="p_rich"), best["t"], best["omax"])
            b = E.evaluate(b_d, st, "win")
            nl = E.null_perm(b_d, st, ["p_rich"], "win", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {"t": best["t"], "omax": best["omax"], "phase": phase}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why}
        out["segments"][f"flat_maiden_{phase}"] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[21] = cycle21


# ---------------------------------------------------------------- サイクル 22: 1 レース 1 点（期待値が最大の馬だけ）
def strat_ev_top1(ab, t: float, omax: float):
    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        p = combo_p(d, k, *ab, col="p_rich")
        o = d[f"odds_pre{k}"].to_numpy(float)
        ev = np.where(o <= omax, p * o, 0.0)
        mx = pd.Series(ev).groupby(d["race_id"].to_numpy()).transform("max").to_numpy()
        return ((ev >= t) & (ev == mx)).astype(float)
    return s


def cycle22(d: pd.DataFrame) -> dict:
    out = {"cycle": 22, "hypothesis": "同じレースで複数の馬を買うと、当たりは高々 1 頭で、期待値の低い側の馬が控除の負けを足す。"
           "1 レース 1 点（期待値が最大の馬だけ）にすると回収率が上がる。", "segments": {}}
    n_trials = 0
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        ab = fit_combo(a_d, col="p_rich")
        rows = []
        for t, omax in itertools.product((1.1, 1.2, 1.3, 1.5), (20.0, 50.0)):
            rows.append({"t": t, "omax": omax, "a": E.evaluate(a_d, strat_ev_top1(ab, t, omax), "win", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"ab": ab, "trials": len(rows), "grid": [{"t": r["t"], "omax": r["omax"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
        if best:
            st = strat_ev_top1(ab, best["t"], best["omax"])
            b = E.evaluate(b_d, st, "win")
            nl = E.null_perm(b_d, st, ["p_rich"], "win", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {"t": best["t"], "omax": best["omax"]}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[22] = cycle22


def cycle23(d: pd.DataFrame) -> dict:
    out = cycle11(swap_phase(d, "t15"))
    out["cycle"] = 23
    out["phase"] = "t15"
    out["hypothesis"] = "判定の時点を T-15 にすると、T-60 と T-5 の間の成績になり、手での購入でも間に合う時点で 100% を越える。"
    return out


CYCLES[23] = cycle23


def cycle24(d: pd.DataFrame) -> dict:
    out = cycle11(d)
    out["cycle"] = 24
    out["hypothesis"] = ("サイクル 16 の落ち込みは、汚れた s_*（未来の年を含む上流の学習）ではなく、sr・f3 など（値は as-of、定数の当てはめが 2021 年末まで）"
                         "を外したことによる。s_* だけを外した v7 でも c13 と同じ水準が出る。")
    return out


CYCLES[24] = cycle24


# ---------------------------------------------------------------- サイクル 29: 確定オッズの見込みで期待値を計る（T-60 の人気の順位ごとの平均のずれ）
def mean_drift_by_rank() -> np.ndarray:
    """2026 年の発走前（T-60）と確定のオッズの対数の比の、T-60 の人気の順位ごとの平均（オッズの列だけ。着順・払戻は読まない）。"""
    from src.scripts.data.bet_cycles import load_drift
    pool = load_drift("t60")["races"]
    mx = max(len(r) for r in pool)
    sums, cnt = np.zeros(mx), np.zeros(mx)
    for r in pool:
        sums[: len(r)] += r
        cnt[: len(r)] += 1
    return sums / np.maximum(cnt, 1)


def strat_ev_adj(ab, t: float, omax: float, md: np.ndarray):
    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        p = combo_p(d, k, *ab, col="p_rich")
        o = d[f"odds_pre{k}"].to_numpy(float)
        rk = np.clip(pop_pre(d, k).astype(int) - 1, 0, len(md) - 1)
        o_final = o * np.exp(md[rk])                        # 確定のオッズの見込み（人気馬は短く、人気薄は長くなる）
        return ((p * o_final >= t) & (o <= omax)).astype(float)
    return s


def cycle29(d: pd.DataFrame) -> dict:
    out = {"cycle": 29, "hypothesis": "T-60 のオッズは人気馬で確定より長く、人気薄で短い（終盤の資金）。期待値を、T-60 の人気の順位ごとの平均のずれで"
           "確定のオッズの見込みに直してから計ると、人気馬の過大な期待値を避けられ、回収率が上がる。", "segments": {}}
    md = mean_drift_by_rank()
    out["mean_drift_by_rank"] = md[:18].tolist()
    n_trials = 0
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        ab = fit_combo(a_d, col="p_rich")
        rows = []
        for t, omax in itertools.product((1.2, 1.3, 1.5), (20.0, 50.0)):
            rows.append({"t": t, "omax": omax, "a": E.evaluate(a_d, strat_ev_adj(ab, t, omax, md), "win", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"combo_ab_t60": ab, "trials": len(rows), "grid": [{"t": r["t"], "omax": r["omax"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
        if best:
            st = strat_ev_adj(ab, best["t"], best["omax"], md)
            b = E.evaluate(b_d, st, "win")
            nl = E.null_perm(b_d, st, ["p_rich"], "win", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {"t": best["t"], "omax": best["omax"]}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[29] = cycle29


# ---------------------------------------------------------------- サイクル 41: 最終レースの効果（市場の値だけ）
def strat_lastrace(pp_max: int, last_only: bool = True):
    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        rno = d["race_id"].astype(str).str[-2:].astype(int).to_numpy()
        m = pop_pre(d, k) <= pp_max
        if last_only:
            m &= rno == 12
        return m.astype(float)
    return s


def cycle41(d: pd.DataFrame) -> dict:
    out = {"cycle": 41, "hypothesis": "最終レースの効果（McGlothlin 1956・Ali 1977）: その日の負けを取り返そうとする買い手が最終レースで大穴を過剰に買い、"
           "人気馬が割安になる。最終レース（12R）の T-60 の人気上位を買うと、ほかのレースの同じ人気より回収率が高く、100% を越える。", "segments": {}}
    n_trials = 0
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        rows = []
        for pp, kind in itertools.product((1, 2, 3), ("win", "place")):
            rows.append({"pp": pp, "kind": kind, "a": E.evaluate(a_d, strat_lastrace(pp), kind, n_boot=500),
                         "ref_all": E.evaluate(a_d, strat_lastrace(pp, False), kind, n_boot=0)["roi"]})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"trials": len(rows), "grid": [{"pp": r["pp"], "kind": r["kind"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"], "roi_all_races": r["ref_all"]} for r in rows]}
        if best:
            st = strat_lastrace(best["pp"])
            b = E.evaluate(b_d, st, best["kind"])
            nl = E.null_perm(b_d, st, ["odds_pre0", "odds_pre1", "odds_pre2", "odds_pre3", "odds_pre4"], best["kind"], n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {"pp": best["pp"], "kind": best["kind"]}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[41] = cycle41


# ---------------------------------------------------------------- サイクル 42: 人気騎手の過剰人気（市場 + 騎手の成績だけの第 2 段）
def attach_frame_cols(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    f = pd.read_parquet(D.FRAME, columns=["race_id", "horse_id"] + cols, filters=[("year", ">=", int(d["year"].min())), ("year", "<=", int(d["year"].max()))])
    f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    return d.merge(f.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")


def fit_cov(d: pd.DataFrame, covs: list[str]) -> np.ndarray:
    y = d["y1"].fillna(0).to_numpy(float)
    rid = pd.factorize(d["race_id"])[0]
    X = np.column_stack([d[c].fillna(d[c].median()).to_numpy(float) for c in covs])
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    lqs = [np.log(np.clip(E.q_pre(d, k), 1e-6, 1)) for k in range(D.N_PRE)]

    def nll(w):
        tot = 0.0
        for lq in lqs:
            z = w[0] * lq + X @ w[1:]
            mx = pd.Series(z).groupby(rid).transform("max").to_numpy()
            lse = np.log(pd.Series(np.exp(z - mx)).groupby(rid).transform("sum").to_numpy()) + mx
            tot -= np.sum(y * (z - lse))
        return tot

    r = minimize(nll, x0=np.r_[1.0, np.zeros(len(covs))], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 4000})
    return r.x, X.mean(0), X.std(0)


def strat_cov(w, covs, stats, t: float, omax: float):
    mu, sd = stats

    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        X = np.column_stack([d[c].fillna(np.nanmedian(d[c].to_numpy(float))).to_numpy(float) for c in covs])
        Xs = X - np.nanmean(X, 0)
        Xs = Xs / np.where(np.nanstd(X, 0) > 0, np.nanstd(X, 0), 1)
        z = w[0] * np.log(np.clip(E.q_pre(d, k), 1e-6, 1)) + Xs @ w[1:]
        z = z - pd.Series(z).groupby(d["race_id"].to_numpy()).transform("max").to_numpy()
        p = E.race_norm(d, np.exp(z))
        o = d[f"odds_pre{k}"].to_numpy(float)
        return ((p * o >= t) & (o <= omax)).astype(float)
    return s


def cycle42(d: pd.DataFrame) -> dict:
    covs = ["j_rate", "t_rate"]
    out = {"cycle": 42, "hypothesis": "人気騎手・人気厩舎の過剰人気（公開の知見: 有名な騎手の馬は売れ過ぎる）。自前の確率を使わず、T-60 の市場と、"
           "騎手・調教師の as-of の勝率だけの第 2 段（条件付き logit）で、市場が騎手・厩舎の名前に払い過ぎている分を取り除くと、期待値の判定が効く。",
           "covs": covs, "segments": {}}
    d = attach_frame_cols(d, covs)
    n_trials = 0
    for seg in SEGS:
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        w, mu, sd = fit_cov(a_d, covs)
        rows = []
        for t, omax in itertools.product((1.05, 1.1, 1.2, 1.3), (20.0, 50.0)):
            rows.append({"t": t, "omax": omax, "a": E.evaluate(a_d, strat_cov(w, covs, (mu, sd), t, omax), "win", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"w": list(map(float, w)), "trials": len(rows), "grid": [{"t": r["t"], "omax": r["omax"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
        if best:
            st = strat_cov(w, covs, (mu, sd), best["t"], best["omax"])
            b = E.evaluate(b_d, st, "win")
            nl = E.null_perm(b_d, st, covs, "win", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {"t": best["t"], "omax": best["omax"]}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[42] = cycle42


# ---------------------------------------------------------------- サイクル 43: 判定のオッズのずれの大きさへの感度（候補の凍結値。探さない）
def sim_scaled(d: pd.DataFrame, phase: str, scale: float) -> pd.DataFrame:
    from src.scripts.data.bet_cycles import load_drift
    pool = load_drift(phase)["races"]
    rid = d["race_id"].to_numpy()
    odds = d["odds_final"].to_numpy(float)
    pop = np.nan_to_num(d["pop_final"].to_numpy(float), nan=99.0)
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], len(d)]
    e = d.copy()
    for k in range(D.N_PRE):
        rng = np.random.default_rng(1000 + k)
        out = np.empty(len(d))
        for s_, e_ in zip(starts, ends):
            src = pool[int(rng.integers(len(pool)))]
            rk = np.clip(np.where(pop[s_:e_] < 99, pop[s_:e_], len(src)).astype(int) - 1, 0, len(src) - 1)
            o = odds[s_:e_] * np.exp(-scale * src[rk])
            out[s_:e_] = o * (np.sum(1.0 / o) / np.sum(1.0 / odds[s_:e_]))
        e[f"odds_pre{k}"] = out
    return e


def cycle43(d: pd.DataFrame) -> dict:
    out = {"cycle": 43, "hypothesis": "候補の回収率は、判定のオッズの再現のずれの大きさ（2026 年の 1 年から借りた値）に強く依る。ずれを 0.5・1・1.5・2 倍にして、"
           "凍結した設定のまま 2017〜21 で数え、どの候補がどれだけ楽観に頼るかを測る（探さない）。", "segments": {}, "n_trials": 0, "sensitivity": {}}
    d = d.sort_values(["race_id", "hn"]).reset_index(drop=True)
    for name in ("c13_flat", "c18_flat_t05", "c21_maiden_t60", "c23_flat_t15", "c24_flat_v7"):
        spec = json.loads((CAND / f"{name}.json").read_text(encoding="utf-8"))
        global RICH_FILE
        RICH_FILE = D.OUT / spec["probability"]["file"]
        e0 = _prep(D.load_table((2017, 2021)), True)
        if spec.get("group"):
            e0 = e0[group_of(e0) == spec["group"]].reset_index(drop=True)
        e0 = e0[E.seg_mask(e0, "flat")].reset_index(drop=True)
        ab = tuple(spec["combo_ab_t60"])
        st = strat_ev(lambda x, k, ab=ab: combo_p(x, k, *ab, col="p_rich"), spec["t"], spec["omax"])
        res = {}
        for sc in (0.0, 0.5, 1.0, 1.5, 2.0):
            r = E.evaluate(sim_scaled(e0, spec.get("phase", "t60"), sc), st, "win", n_boot=300)
            res[str(sc)] = {"roi": r["roi"], "bets_per_year": r["bets_per_year"], "roi_ci": r.get("roi_ci"), "mr": r["market_ratio"]}
        out["sensitivity"][name] = res
    return out


CYCLES[43] = cycle43


# ---------------------------------------------------------------- サイクル 44: ワイド（期待値の上位 2 頭の組）
def wide_payoffs(years) -> dict:
    from src.scripts.data import bet_cycles as bc
    ys = set(years)
    rids = {r for r in pd.read_parquet(D.TABLE, columns=["race_id", "year"], filters=[("year", ">=", min(ys)), ("year", "<=", max(ys))])["race_id"].astype(str)}
    pay = bc.load_payoffs(rids)                      # 読む年のレースだけ（2022 年以降の払戻を読まない）
    bc.fill_payoffs_from_flat(pay, years)
    return {rid: v.get("wide", {}) for rid, v in pay.items() if str(rid)[:4].isdigit() and int(str(rid)[:4]) in set(years)}


def harville_pair_top3(q: np.ndarray, i: int, j: int) -> float:
    n = len(q)

    def p3(a, b, c):
        return q[a] * q[b] / (1 - q[a]) * q[c] / (1 - q[a] - q[b])
    tot = 0.0
    for x in range(n):
        if x in (i, j):
            continue
        tot += p3(i, j, x) + p3(j, i, x) + p3(i, x, j) + p3(j, x, i) + p3(x, i, j) + p3(x, j, i)
    return tot


def eval_wide(d: pd.DataFrame, ab, t: float, omax: float, wpay: dict, n_boot: int = 500) -> dict:
    """各版で、期待値 ≥ t の馬が 2 頭以上いるレースの、期待値の上位 2 頭のワイドを 1 点。5 版の平均。"""
    rows = []
    for k in range(D.N_PRE):
        p = combo_p(d, k, *ab, col="p_rich")
        o = d[f"odds_pre{k}"].to_numpy(float)
        ev = np.where((o <= omax), p * o, 0.0)
        for rid, idx in pd.Series(np.arange(len(d))).groupby(d["race_id"].to_numpy()).indices.items():
            sel = idx[ev[idx] >= t]
            if len(sel) < 2:
                continue
            top = sel[np.argsort(-ev[sel])[:2]]
            hn = d["hn"].to_numpy()[top].astype(int)
            key = "-".join(map(str, sorted(hn)))
            fin = pd.to_numeric(d["fin"], errors="coerce").to_numpy()[top]
            hit = bool((fin <= 3).all())
            ret = (wpay.get(rid, {}).get(key, 0) / 100.0) if hit else 0.0
            qf = d["q_final"].to_numpy()[idx]
            ii = [int(np.flatnonzero(idx == x)[0]) for x in top]
            rows.append({"k": k, "rid": rid, "date": str(d["date"].iloc[top[0]])[:10], "ret": ret, "hit": float(hit),
                         "exp": harville_pair_top3(qf, ii[0], ii[1])})
    if not rows:
        return {"roi": None, "bets_per_year": 0, "market_ratio": None, "base_same_pop": None}
    r = pd.DataFrame(rows)
    years = d["year"].nunique()
    out = {"kind": "wide", "bets": len(r) / D.N_PRE, "bets_per_year": len(r) / D.N_PRE / years, "roi": float(r["ret"].mean()),
           "market_ratio": float(r["hit"].sum() / r["exp"].sum()), "base_same_pop": None, "hit_rate": float(r["hit"].mean())}
    g = r.groupby("date")[["ret", "hit", "exp"]].sum().assign(n=r.groupby("date").size())
    A = g[["n", "ret", "hit", "exp"]].to_numpy()
    rng = np.random.default_rng(0)
    B = A[rng.integers(0, len(A), size=(n_boot, len(A)))].sum(1)
    out["roi_ci"] = [float(np.percentile(B[:, 1] / B[:, 0], 2.5)), float(np.percentile(B[:, 1] / B[:, 0], 97.5))]
    out["mr_ci"] = [float(np.percentile(B[:, 2] / B[:, 3], 2.5)), float(np.percentile(B[:, 2] / B[:, 3], 97.5))]
    return out


def cycle44(d: pd.DataFrame) -> dict:
    out = {"cycle": 44, "hypothesis": "券種を変える: 単勝の期待値で選んだ馬が同じレースに 2 頭以上いるとき、上位 2 頭のワイドは当たりが多く、"
           "2 頭とも市場より 3 着内に来やすい（自前の確率の上積み）なら、ワイドの控除（22.5%）を越える。", "segments": {}}
    wpay = wide_payoffs(range(2017, 2022))
    n_trials = 0
    seg = "flat"
    a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
    ab = fit_combo(a_d, col="p_rich")
    rows = []
    for t, omax in itertools.product((1.0, 1.1, 1.2, 1.3), (20.0, 50.0)):
        rows.append({"t": t, "omax": omax, "a": eval_wide(a_d, ab, t, omax, wpay)})
        n_trials += 1
    best = pick_best(rows)
    seg_out = {"combo_ab_t60": ab, "trials": len(rows), "grid": [{"t": r["t"], "omax": r["omax"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
    if best:
        b = eval_wide(b_d, ab, best["t"], best["omax"], wpay)
        b["base_same_pop"] = 0.0
        best["a"]["base_same_pop"] = 0.0
        ok, why = E.gate(best["a"], b, None)
        seg_out |= {"best": {"t": best["t"], "omax": best["omax"]}, "select_a": best["a"], "select_b": b, "gate": ok, "gate_why": why}
    out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[44] = cycle44


def cycle48(d: pd.DataFrame) -> dict:
    """サイクル 12 の 3 つの確率の組み合わせを、学習の平均（v5bag）と約 120 列の確率と T-60 の市場で、平場だけ流す。"""
    out = cycle12(d)
    out["cycle"] = 48
    out["hypothesis"] = "学習の平均（v5bag）に、別の列の約 120 列の確率を足して 3 つで組むと、平場の確率が鋭くなり c13 より回収率・件数が上がる。"
    out["segments"].pop("graded", None)
    return out


# ---------------------------------------------------------------- サイクル 49: 勝ち目の薄い馬を避ける（自前の確率の順位も上位）
def strat_ev_rank(ab, t: float, omax: float, rmax: int):
    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        p = combo_p(d, k, *ab, col="p_rich")
        o = d[f"odds_pre{k}"].to_numpy(float)
        rk = pd.Series(p).groupby(d["race_id"].to_numpy()).rank(ascending=False, method="first").to_numpy()
        return ((p * o >= t) & (o <= omax) & (rk <= rmax)).astype(float)
    return s


def cycle49(d: pd.DataFrame) -> dict:
    out = {"cycle": 49, "hypothesis": "期待値が高く見えても、組んだ確率のレースの中の順位が低い馬（オッズの長さだけで期待値が高い馬）は、確率の推定の誤差の影響が大きい。"
           "組んだ確率の順位が上位（3・5 位以内）の馬だけを買うと、誤差に強くなり回収率が上がる。", "segments": {}}
    n_trials = 0
    for seg in ("flat",):
        a_d, b_d = sub(d, "select_a", seg), sub(d, "select_b", seg)
        ab = fit_combo(a_d, col="p_rich")
        rows = []
        for t, rmax in itertools.product((1.1, 1.2, 1.3), (3, 5)):
            rows.append({"t": t, "rmax": rmax, "a": E.evaluate(a_d, strat_ev_rank(ab, t, 50.0, rmax), "win", n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"combo_ab_t60": ab, "trials": len(rows), "grid": [{"t": r["t"], "rmax": r["rmax"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
        if best:
            st = strat_ev_rank(ab, best["t"], 50.0, best["rmax"])
            b = E.evaluate(b_d, st, "win")
            nl = E.null_perm(b_d, st, ["p_rich"], "win", n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {"t": best["t"], "rmax": best["rmax"]}, "select_a": best["a"], "select_b": b, "null_b": nl, "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


CYCLES[48] = cycle48
CYCLES[49] = cycle49


# ---------------------------------------------------------------- サイクル 50: 割引きの評価（2022〜24 に当てた候補を同じ開催日で同時にブートストラップ）
def candidate_daily(name: str, d_all_cache: dict) -> pd.DataFrame:
    """候補の凍結した設定で、2022〜24 の開催日ごとの賭け金・払戻（5 版の平均）を返す。"""
    global RICH_FILE
    spec = json.loads((CAND / f"{name}.json").read_text(encoding="utf-8"))
    RICH_FILE = D.OUT / spec["probability"]["file"]
    key = spec["probability"]["file"]
    import hashlib
    if hashlib.sha256(RICH_FILE.read_bytes()).hexdigest()[:16] != spec["probability"]["sha256_16"]:
        raise SystemExit(f"{RICH_FILE} の sha が凍結と違う")
    if key not in d_all_cache:
        d_all_cache[key] = load_all()
    d = d_all_cache[key]
    if spec.get("phase", "t60") != "t60":
        d = swap_phase(d, spec["phase"])
    if spec.get("group"):
        d = d[group_of(d) == spec["group"]].reset_index(drop=True)
    d = d[E.period_mask(d, "confirm") & E.seg_mask(d, spec["segment"])].reset_index(drop=True)
    ab = tuple(spec["combo_ab_t60"])
    st = strat_ev(lambda x, k: combo_p(x, k, *ab, col="p_rich"), spec["t"], spec["omax"])
    S = np.vstack([st(d, k) for k in range(D.N_PRE)]).mean(0)
    g = pd.DataFrame({"date": d["date"].astype(str).str[:10], "s": S, "p": S * d["ret_tan"].fillna(0).to_numpy()}).groupby("date").sum()
    return g


def cycle50(d: pd.DataFrame) -> dict:
    names = [r["spec"]["name"] for r in E.confirm_log()]
    out = {"cycle": 50, "hypothesis": "2022〜24 に当てた候補は同じ考えの変種で強く相関する。同じ開催日で同時にブートストラップし、全候補の同時の 95% 下限"
           "（標準化した最大の偏差の分位で割り引く。Romano-Wolf 型）と、相関を無視した Holm を並べて、割引きの後も 100% を超えるかを見る。",
           "segments": {}, "n_trials": 0, "candidates": names}
    with E.eval_only_guard("src/research/megu_cycles/cycles.py:cycle50", "discount-v1", ("2022-01-01", "2024-12-31"),
                           "2022〜24 に当てた候補の多重比較の割引き（同時のブートストラップ）。設定は選び直さない") as tok:
        cache: dict = {}
        G = {n: candidate_daily(n, cache) for n in names}
        tok.finish()
    dates = sorted(set().union(*[set(g.index) for g in G.values()]))
    A = np.stack([G[n].reindex(dates).fillna(0)[["s", "p"]].to_numpy() for n in names])      # (候補, 日, 2)
    roi = A[:, :, 1].sum(1) / A[:, :, 0].sum(1)
    rng = np.random.default_rng(20261124)
    nb = 4000
    idx = rng.integers(0, len(dates), size=(nb, len(dates)))
    B = A[:, idx, :].sum(2)                                   # (候補, nb, 2)
    rb = B[:, :, 1] / B[:, :, 0]
    sd = rb.std(1)
    z = (rb - roi[:, None]) / sd[:, None]
    cmax = float(np.quantile((-z).max(0), 0.95))              # 全候補の同時の片側 95%
    p_raw = (rb <= 1.0).mean(1)                               # 片側の p（回収率 ≤ 1 の割合）
    order = np.argsort(p_raw)
    holm = np.empty(len(names))
    run = 0.0
    for r_, i in enumerate(order):
        run = max(run, min(1.0, (len(names) - r_) * p_raw[i]))
        holm[i] = run
    corr = np.corrcoef(rb)
    out["discount"] = {n: {"roi_2022_24": float(roi[i]), "sd": float(sd[i]), "lb_simultaneous_95": float(roi[i] - cmax * sd[i]),
                           "lb_single_95": float(np.quantile(rb[i], 0.05)), "p_raw": float(p_raw[i]), "p_holm": float(holm[i])}
                       for i, n in enumerate(names)}
    out["c_max"] = cmax
    out["corr"] = {n: {m: float(corr[i, j]) for j, m in enumerate(names)} for i, n in enumerate(names)}
    t = json.loads((DOCS / "trials.json").read_text(encoding="utf-8"))
    out["n_total_trials"] = t["total_trials"]
    out["select_b_looks"] = t["select_b_looks"]["count"]
    return out


CYCLES[50] = cycle50


# ---------------------------------------------------------------- 汎用のサイクル（設定は docs/operations/megu_cycles/cycle_NN.cfg.json）
def strat_generic(ab, t: float, omax: float, omin: float = 1.0, field_max: int | None = None, age2: bool | None = None):
    def s(d: pd.DataFrame, k: int) -> np.ndarray:
        p = combo_p(d, k, *ab, col="p_rich")
        o = d[f"odds_pre{k}"].to_numpy(float)
        m = (p * o >= t) & (o <= omax) & (o >= omin)
        if field_max:
            m &= pd.to_numeric(d["field"], errors="coerce").to_numpy() <= field_max
        if age2 is not None:
            m &= (pd.to_numeric(d["age"], errors="coerce").to_numpy() == 2) == age2
        return m.astype(float)
    return s


def cycle_generic(d: pd.DataFrame, n: int) -> dict:
    cfg = json.loads((DOCS / f"cycle_{n:02d}.cfg.json").read_text(encoding="utf-8"))
    out = {"cycle": n, "hypothesis": cfg["hypothesis"], "cfg": cfg, "segments": {}}
    e0 = d if cfg.get("phase", "t60") == "t60" else swap_phase(d, cfg["phase"])
    if cfg.get("group"):
        e0 = e0[group_of(e0) == cfg["group"]].reset_index(drop=True)
    if cfg.get("surf"):
        e0 = e0[e0["surf"].astype(str) == cfg["surf"]].reset_index(drop=True)
    if cfg.get("venues"):
        e0 = e0[e0["venue"].astype(str).isin(cfg["venues"])].reset_index(drop=True)
    if cfg.get("cls"):
        e0 = e0[pd.to_numeric(e0["cls"], errors="coerce").isin(cfg["cls"])].reset_index(drop=True)
    if cfg.get("field_min"):
        e0 = e0[pd.to_numeric(e0["field"], errors="coerce") >= cfg["field_min"]].reset_index(drop=True)
    n_trials = 0
    for seg in cfg.get("segs", ["flat", "graded"]):
        a_d, b_d = sub(e0, "select_a", seg), sub(e0, "select_b", seg)
        ab = fit_combo(sub(e0, "select_a", cfg.get("fit_seg", seg)), col="p_rich")
        rows = []
        grid = list(itertools.product(cfg.get("t", [1.2, 1.3, 1.5]), cfg.get("omax", [20.0, 50.0]), cfg.get("omin", [1.0])))
        for t, omax, omin in grid:
            st = strat_generic(ab, t, omax, omin, cfg.get("field_max"), cfg.get("age2"))
            rows.append({"t": t, "omax": omax, "omin": omin, "a": E.evaluate(a_d, st, cfg.get("kind", "win"), n_boot=500)})
            n_trials += 1
        best = pick_best(rows)
        seg_out = {"combo_ab_t60": ab, "trials": len(rows),
                   "grid": [{"t": r["t"], "omax": r["omax"], "omin": r["omin"], "roi_a": r["a"]["roi"], "bpy_a": r["a"]["bets_per_year"]} for r in rows]}
        if best:
            st = strat_generic(ab, best["t"], best["omax"], best["omin"], cfg.get("field_max"), cfg.get("age2"))
            b = E.evaluate(b_d, st, cfg.get("kind", "win"))
            nl = E.null_perm(b_d, st, ["p_rich"], cfg.get("kind", "win"), n=30)
            ok, why = E.gate(best["a"], b, nl)
            seg_out |= {"best": {"t": best["t"], "omax": best["omax"], "omin": best["omin"]}, "select_a": best["a"], "select_b": b, "null_b": nl,
                        "gate": ok, "gate_why": why}
        out["segments"][seg] = seg_out
    out["n_trials"] = n_trials
    return out


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
    if spec.get("group"):
        d_all = d_all[group_of(d_all) == spec["group"]].reset_index(drop=True)
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
        L.append(f"| {'平場' if k == 'flat' else '重賞' if k == 'graded' else k} | {json.dumps(v.get('best'), ensure_ascii=False)} | {short(v.get('select_a'))} | {short(v.get('select_b'))} | "
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

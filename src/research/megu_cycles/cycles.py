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
def _prep(d: pd.DataFrame, rich: bool) -> pd.DataFrame:
    d = d.sort_values(["race_id", "hn"]).reset_index(drop=True)
    d["p_model"] = E.race_norm(d, d["pm_win"].to_numpy(float))
    if rich and D.RICH.exists():
        r = pd.read_parquet(D.RICH, filters=[("year", ">=", int(d["year"].min())), ("year", "<=", int(d["year"].max()))])
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
    t0 = time.time()
    d = load_select()
    body = CYCLES[n](d)
    body["sec"] = round(time.time() - t0)
    write_cycle(n, body)
    for seg, s in body["segments"].items():
        print(seg, s.get("best"), "\n  2017〜19:", short(s.get("select_a")), "\n  2020〜21:", short(s.get("select_b")),
              "\n  帰無:", s.get("null_b"), "\n  関門:", s.get("gate"), s.get("gate_why"), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("n", type=int)
    a = ap.parse_args()
    cmd_run(a.n)



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
    """サイクル 5 と同じ設計を、結果を使わない決まり（年別の欠けの率）で列を外した確率 v3 で流す。2017〜19 で選び直す。"""
    out = cycle5(d)
    out["cycle"] = 6
    out["hypothesis"] = "記録の取れ方が年で変わる列を結果を使わずに外した確率（v3）でも、確定の市場に足す情報があり、期待値の判定が効く。"
    out["rich_file"] = str(D.RICH)
    return out


CYCLES[6] = cycle6


if __name__ == "__main__":
    sys.exit(main())

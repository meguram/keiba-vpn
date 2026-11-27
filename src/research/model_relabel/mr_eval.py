"""拡張ラベルの評価（担当 model-relabel）: 較正・入れ子の選び（L_λ）・本物のラベルの物差し・Benter の 2 段・対のブートストラップ。

- 予測は mr_train.py の pred/time/*.parquet（本数ごとの列 p_r*）。物差しは本物の 1 着（y1）・3 着内（y3）だけ。
- 候補の族:
  - base（元のラベル）: 本数 × 較正
  - pw（拡張・大敗の重み w ∈ {1,2,4}）: w × 本数 × 較正（w ごとの固定の列も並べる）
  - pord（拡張 w=1 の m_b と notbig の m_a、s = logit(m_b) + κ·log(m_a)）: κ × 本数 × 較正
- 時間の順の入れ子: 外側の年 t の選びは、t−2 年で当てた較正を t−1 年の予測に当てて L_λ を数える。年 t の較正は t−1 年で当てる。
- L_λ = LL_true(p̂) + λ · mean_{big}(−log(1 − p̂))（spec の metric）。
- 回収率は計算しない（rules-council 1 号の条件）。Benter の 2 段は確定の市場 q と組んだ勝ち馬の −log p。
- 期間: select は外側 2019〜21（台帳 select。読む行は 2021 年まで）。confirm は外側 2022〜24 を 1 回だけ（台帳 confirm）。
- meeting: 開催単位の 5 分割（報告だけ。設定は時間の順の選びに固定。confirm の後に 1 回、台帳 eval_only）。
  KEIBA_SESSION=model-relabel nice -n 19 python -m src.research.model_relabel.mr_eval select
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.research.model_relabel import mr_labels as LB
from src.research.model_relabel import mr_train as TR

ROOT = Path(__file__).resolve().parents[3]
OUT = LB.OUT
RUNNERS = Path("/home/jovyan/work/megu_cycles_out/runners.parquet")
DOCS = ROOT / "docs/operations/model_relabel"
EPS = 1e-6


# ---------------------------------------------------------------- 物差し
def race_normalize(p: np.ndarray, rid: np.ndarray, total: float) -> np.ndarray:
    """レースの中で和を total（勝ち 1・複勝 3）にそろえる。上限 1 を超えた分は他の馬に配り直す（数回の反復）。"""
    p = np.clip(np.asarray(p, float), EPS, None)
    s = pd.Series(p)
    n = pd.Series(1.0, index=s.index).groupby(rid).transform("sum").to_numpy()
    tgt = np.minimum(total, n)
    for it in range(20):
        sm = s.groupby(rid).transform("sum").to_numpy()
        q = s.to_numpy() * tgt / sm
        if total == 1 or (q <= 1 + 1e-9).all():
            s = pd.Series(q)
            break
        if it == 19:
            print(f"[警告] race_normalize が 20 回で収まらない（上限を超える馬 {int((q > 1 + 1e-9).sum())} 頭）", flush=True)
        capped = q >= 1
        # 上限の馬は 1 に固定し、残りの和で他を伸ばす
        fixed = pd.Series(np.where(capped, 1.0, 0.0)).groupby(rid).transform("sum").to_numpy()
        free = pd.Series(np.where(capped, 0.0, s.to_numpy())).groupby(rid).transform("sum").to_numpy()
        q = np.where(capped, 1.0, s.to_numpy() * (tgt - fixed) / np.maximum(free, EPS))
        s = pd.Series(q)
    return np.clip(s.to_numpy(), EPS, 1 - EPS)


def ll(p, y) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def L_metric(p, y, big, lam: float) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    pen = float(np.mean(-np.log(1 - p[big]))) if big.any() else 0.0
    return float(np.mean(ll(p, y)) + lam * pen)


def auc(score, y) -> float:
    r = pd.Series(score).rank().to_numpy()
    npos = y.sum()
    nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def race_auc(score, y, rid) -> float:
    d = pd.DataFrame({"s": score, "y": y.astype(float), "r": rid})
    d["rk"] = d.groupby("r")["s"].rank()
    g = d.groupby("r").agg(npos=("y", "sum"), n=("y", "size"))
    sp = d[d["y"] == 1].groupby("r")["rk"].sum()
    g = g.join(sp.rename("sp")).fillna({"sp": 0})
    g = g[(g["npos"] > 0) & (g["npos"] < g["n"])]
    a = (g["sp"] - g["npos"] * (g["npos"] + 1) / 2) / (g["npos"] * (g["n"] - g["npos"]))
    return float(a.mean())


def ece(p, y, bins=10) -> tuple[float, list]:
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    t = pd.DataFrame({"p": p, "y": y, "b": q}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    e = float((t["n"] * (t["p"] - t["y"]).abs()).sum() / t["n"].sum())
    return e, [[round(float(a), 4), round(float(b), 4)] for a, b in zip(t["p"], t["y"])]


def top_rates(p, d: pd.DataFrame) -> dict:
    x = d[["race_id", "y1", "y3"]].copy()
    x["p"] = p
    x["rk"] = x.groupby("race_id")["p"].rank(ascending=False, method="first")
    t1 = x[x["rk"] == 1]
    t3 = x[x["rk"] <= 3]
    return {"top1_win": round(float(t1["y1"].mean()), 4), "top1_top3": round(float(t1["y3"].mean()), 4),
            "top3_top3": round(float(t3["y3"].mean()), 4)}


# ---------------------------------------------------------------- 較正
def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_calibrator(kind: str, s, y):
    if kind == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        m = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(s, y)
        return lambda z: m.predict(z)
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(s).reshape(-1, 1), y)
    return lambda z: m.predict_proba(_logit(z).reshape(-1, 1))[:, 1]


def calibrate(kind, s_fit, y_fit, rid_fit, s_app, rid_app, total) -> np.ndarray:
    """較正（s_fit → 本物のラベル）を当てて、レース内で和をそろえる。"""
    f = fit_calibrator(kind, s_fit, y_fit)
    return race_normalize(f(s_app), rid_app, total)


# ---------------------------------------------------------------- 表の準備
META = Path("/home/jovyan/work/retrain_out/cycles16_fix3/meta.parquet")


def base_frame(max_year: int) -> pd.DataFrame:
    """meta（max_year 年まで。読むときに絞る）に本物・拡張のラベル・開催日・確定の市場 q を付ける。"""
    assert max_year <= 2024
    meta = pd.read_parquet(META, columns=["race_id", "horse_id", "year", "date", "finish_position"], filters=[("year", "<=", max_year)])
    assert pd.to_numeric(meta["year"]).max() <= max_year
    meta = meta.reset_index(drop=True)
    m = TR.attach_labels(meta, max_year=max_year)
    m["date"] = pd.to_datetime(meta["date"]).dt.strftime("%Y%m%d").to_numpy()
    q = pd.read_parquet(RUNNERS, columns=["race_id", "horse_id", "year", "q_final"], filters=[("year", "<=", max_year)])
    q["race_id"], q["horse_id"] = q["race_id"].astype(str), q["horse_id"].astype(str)
    m = m.merge(q[["race_id", "horse_id", "q_final"]].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    assert len(m) == len(meta)
    return m


def load_pred(fs, target, variant, base: pd.DataFrame, split="time", conv="convert_A") -> pd.DataFrame:
    """完了の名前の予測の表だけを読む（未完は .partial で、ここでは読まない）。time は base と行の順が同じことを確かめる。"""
    p = OUT / "pred" / split / f"{fs}_{target}_{variant}_{conv}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    x = pd.read_parquet(p, filters=[("year", "<=", int(base["year"].max()))])
    if split == "time":
        assert (x["race_id"].to_numpy() == base["race_id"].to_numpy()).all() and (x["horse_id"].to_numpy() == base["horse_id"].to_numpy()).all()
    return x


def _ord_score(pb, pa, k):
    return 1 / (1 + np.exp(-(_logit(pb) + k * np.log(np.clip(pa, EPS, 1)))))


def candidates(fs: str, target: str, sp: dict, base: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """族ごとに {候補名: 点数の列（全行）}。"""
    rounds = sp["select"]["rounds_v6e" if fs == "v6e" else "rounds_v5bag"]
    fam: dict[str, dict[str, np.ndarray]] = {"base": {}, "pw": {}, "pord": {}}
    b = load_pred(fs, target, "orig", base)
    for r in rounds:
        fam["base"][f"orig|r{r}"] = b[f"p_r{r}"].to_numpy(float)
    for w in sp["penalty"]["P_w"]["w"]:
        e = load_pred(fs, target, f"ext_w{w}", base)
        for r in rounds:
            fam["pw"][f"w{w}|r{r}"] = e[f"p_r{r}"].to_numpy(float)
    nb = load_pred(fs, "notbig", "orig", base)
    e1 = load_pred(fs, target, "ext_w1", base)
    for k in sp["penalty"]["P_ord"]["kappa"]:
        for r in rounds:
            fam["pord"][f"k{k}|r{r}"] = _ord_score(e1[f"p_r{r}"].to_numpy(float), nb[f"p_r{r}"].to_numpy(float), k)
    return fam


# ---------------------------------------------------------------- 入れ子の選びと外側の物差し
def _clogit_nll(w, lp, lq, y, rid):
    z = w[0] * lp + w[1] * lq
    mx = pd.Series(z).groupby(rid).transform("max").to_numpy()
    lse = np.log(pd.Series(np.exp(z - mx)).groupby(rid).transform("sum").to_numpy()) + mx
    return -np.sum(y * (z - lse)) / max(1.0, y.sum())


def _clogit_prep(p, d):
    rid = pd.factorize(d["race_id"])[0]
    ok = pd.Series(d["q_final"].notna().to_numpy()).groupby(rid).transform("all").to_numpy()      # q の欠けたレースは丸ごと落とす
    lp = np.log(np.clip(p, EPS, 1))[ok]
    lq = np.log(np.clip(d["q_final"].to_numpy(float), EPS, 1))[ok]
    return lp, lq, d["y1"].to_numpy(float)[ok], pd.factorize(rid[ok])[0]


_MARKET: dict = {}


def benter(p_fit, d_fit, p_app, d_app, year_key) -> dict:
    """p ∝ exp(a·log p + b·log q)。年 t−1 で当てて年 t で数える。勝ち馬の −log p の平均と、市場だけ（a=0）の値（年ごとに 1 回だけ当てる）。"""
    f = _clogit_prep(p_fit, d_fit)
    g = _clogit_prep(p_app, d_app)
    a, b = minimize(_clogit_nll, x0=[0.5, 0.5], args=f, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6}).x
    if year_key not in _MARKET:
        b0 = minimize(lambda w: _clogit_nll([0.0, w[0]], *f), x0=[1.0], method="Nelder-Mead").x[0]
        _MARKET[year_key] = float(_clogit_nll([0.0, b0], *g))
    return {"a": round(float(a), 4), "b": round(float(b), 4), "nll_combo": float(_clogit_nll([a, b], *g)),
            "nll_market": _MARKET[year_key], "races": int(g[3].max() + 1)}


def outer_metrics(p, s_raw, d: pd.DataFrame, target: str, sp: dict) -> dict:
    """確率の物差しは較正の後の p、順位の物差し（AUC・1 番手）は較正の前の点数 s_raw（isotonic の階段で同点にしない）。"""
    y = d["y1" if target == "win" else "y3"].to_numpy(float)
    big = d["big"].to_numpy()
    rid = d["race_id"].to_numpy()
    e, rel = ece(p, y)
    out = {"n": int(len(d)), "ll": round(float(np.mean(ll(p, y))), 6),
           "L_main": round(L_metric(p, y, big, sp["metric"]["lambda_main"]), 6),
           **{f"L_{lam}": round(L_metric(p, y, big, lam), 6) for lam in sp["metric"]["lambda_sensitivity"]},
           "auc": round(auc(s_raw, y), 5), "race_auc": round(race_auc(s_raw, y, rid), 5), "ece": round(e, 5), "reliability": rel,
           "p_big_mean": round(float(np.mean(p[big])), 5), **top_rates(s_raw, d)}
    if target == "win":
        out["winner_nlp"] = round(float(-np.mean(np.log(np.clip(p[y == 1], EPS, 1)))), 6)
    return out


def nested_time(fs: str, target: str, years, lams, sp: dict, base: pd.DataFrame) -> dict:
    """外側の年ごと・λ ごとに、族ごとの選び（L_λ）と、選んだ候補の外側の物差し。固定の w の列も並べる。
    較正・外側の物差しは (候補, 較正) ごとに 1 回だけ作り、λ ごとには選びだけを変える。"""
    total = 1.0 if target == "win" else 3.0
    yname = "y1" if target == "win" else "y3"
    fam = candidates(fs, target, sp, base)
    yr = base["year"].to_numpy()
    groups = dict(fam)
    for w in sp["penalty"]["P_w"]["w"]:
        groups[f"pw_fixed_w{w}"] = {k: v for k, v in fam["pw"].items() if k.startswith(f"w{w}|")}
    res = {lam: {"years": {}} for lam in lams}
    for t in years:
        i2, i1, i0 = yr == t - 2, yr == t - 1, yr == t
        d2, d1, d0 = (base[i].reset_index(drop=True) for i in (i2, i1, i0))
        inner, outer = {}, {}
        for cands in fam.values():
            for cname, s in cands.items():
                for cal in sp["select"]["calibration"]:
                    p1 = calibrate(cal, s[i2], d2[yname].to_numpy(float), d2["race_id"].to_numpy(), s[i1], d1["race_id"].to_numpy(), total)
                    inner[(cname, cal)] = (p1, {lam: L_metric(p1, d1[yname].to_numpy(float), d1["big"].to_numpy(), lam) for lam in lams})

        def outer_of(key):
            if key not in outer:
                cname, cal = key
                s = next(c[cname] for c in fam.values() if cname in c)
                p0 = calibrate(cal, s[i1], d1[yname].to_numpy(float), d1["race_id"].to_numpy(), s[i0], d0["race_id"].to_numpy(), total)
                m = outer_metrics(p0, s[i0], d0, target, sp)
                if target == "win":
                    m["benter"] = benter(inner[key][0], d1, p0, d0, (fs, t))
                outer[key] = (p0, m)
            return outer[key]

        for lam in lams:
            rec = {}
            for gname, cands in groups.items():
                keys = [(c, cal) for c in cands for cal in sp["select"]["calibration"]]
                key = min(keys, key=lambda k: inner[k][1][lam])
                p0, m = outer_of(key)
                rec[gname] = {"chosen": key[0], "cal": key[1], "L_inner": round(inner[key][1][lam], 6), "outer": m, "_p": p0}
            res[lam]["years"][t] = rec
            print(f"  {fs} {target} λ={lam} {t}: " + "・".join(f"{k} {v['chosen']}/{v['cal']}" for k, v in rec.items()), flush=True)
    return res


def paired_boot(res: dict, base: pd.DataFrame, target: str, fam_a: str, fam_b: str, years, n_boot=2000, seed=20261128,
                res_b: dict | None = None) -> dict:
    """族 a − 族 b の差の、開催日で束ねた対のブートストラップ。res_b を渡すと、族 b は別の特徴量の版（res_b）から取る。
    勝ち: 勝ち馬の −log p の平均。複勝: y3 の馬ごとの二値の対数損失の平均（全頭）。"""
    yname = "y1" if target == "win" else "y3"
    res_b = res_b or res
    rows = []
    for t in years:
        d0 = base[base["year"] == t].reset_index(drop=True)
        pa, pb = res["years"][t][fam_a]["_p"], res_b["years"][t][fam_b]["_p"]
        y = d0[yname].to_numpy(float)
        if target == "win":
            m = y == 1
            la, lb_ = -np.log(np.clip(pa[m], EPS, 1)), -np.log(np.clip(pb[m], EPS, 1))
            dd = d0.loc[m, "date"].to_numpy()
        else:
            la, lb_, dd = ll(pa, y), ll(pb, y), d0["date"].to_numpy()
        rows.append(pd.DataFrame({"date": dd, "a": la, "b": lb_}))
    return _boot(pd.concat(rows), n_boot, seed)


def _boot(rows: pd.DataFrame, n_boot: int, seed: int) -> dict:
    x = rows.groupby("date").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    rng = np.random.default_rng(seed)
    A, B, N = x["a"].to_numpy(), x["b"].to_numpy(), x["n"].to_numpy()
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        diffs[k] = (A[i].sum() - B[i].sum()) / N[i].sum()
    est = (A.sum() - B.sum()) / N.sum()
    return {"diff": round(float(est), 6), "ci95": [round(float(np.quantile(diffs, 0.025)), 6), round(float(np.quantile(diffs, 0.975)), 6)],
            "days": int(len(x))}


def summarize(res: dict, years) -> dict:
    """外側の年を通した単純な平均と、年ごとの選び。"""
    out = {}
    for f in next(iter(res["years"].values())).keys():
        ms = [res["years"][t][f]["outer"] for t in years]
        keys = [k for k in ms[0] if isinstance(ms[0][k], (int, float)) and k != "n"]
        out[f] = {k: round(float(np.mean([m[k] for m in ms])), 6) for k in keys}
        out[f]["chosen"] = {str(t): f"{res['years'][t][f]['chosen']}/{res['years'][t][f]['cal']}" for t in years}
        if "benter" in ms[0]:
            out[f]["benter_a"] = round(float(np.mean([m["benter"]["a"] for m in ms])), 4)
            out[f]["benter_gain"] = round(float(np.mean([m["benter"]["nll_market"] - m["benter"]["nll_combo"] for m in ms])), 6)
    return out


def _ledger(period: str, lo: str, hi: str, purpose: str):
    from src.ops import data_ledger as DL
    sp = TR.spec()
    return DL.guard(tool="src/research/model_relabel/mr_eval.py", family="model_relabel", version=f"eval-{period}-{sp['version']}",
                    period=(lo, hi), purpose=purpose, reads=("着順",), frozen_paths=[TR.SPEC],
                    reason="拡張ラベルのモデルの本物のラベルでの当てはまり（回収率は数えない）")


def cmd(period: str) -> None:
    sp = TR.spec()
    key = "outer_main" if period == "select" else "outer_confirm_once"
    years = range(sp["years"][key][0], sp["years"][key][1] + 1)
    tok = _ledger(period, f"{min(years) - 2}-01-01", f"{max(years)}-12-31", "select" if period == "select" else "confirm")
    t0 = time.time()
    base = base_frame(max(years))
    lams = [sp["metric"]["lambda_main"]] + list(sp["metric"]["lambda_sensitivity"])
    report = {"spec_version": sp["version"], "period": period, "years": list(years), "results": {}, "feature_axis": {}}
    kept: dict = {}
    for fs in ("v5bag", "v5bag_ped", "v6e"):
        for target in ("win", "top3"):
            try:
                allres = nested_time(fs, target, years, lams, sp, base)
            except FileNotFoundError as e:
                print("予測がまだ揃っていない", fs, target, e, flush=True)
                continue
            if fs in ("v5bag", "v5bag_ped"):
                kept[(fs, target)] = allres
            for lam, res in allres.items():
                summ = summarize(res, years)
                boots = {f"{a}-base": paired_boot(res, base, target, a, "base", years)
                         for a in ("pw", "pord", *[f"pw_fixed_w{w}" for w in sp["penalty"]["P_w"]["w"]])}
                per_year = {str(t): {f: {k: v for k, v in r.items() if k != "_p"} for f, r in res["years"][t].items()} for t in years}
                report["results"][f"{fs}|{target}|lam{lam}"] = {"summary": summ, "boot": boots, "per_year": per_year}
                print(f"{fs}|{target}|lam{lam}", json.dumps({f: {k: summ[f].get(k) for k in ("ll", "L_main", "winner_nlp", "race_auc", "benter_gain")}
                                                             for f in summ}, ensure_ascii=False), json.dumps(boots, ensure_ascii=False), flush=True)
    # 特徴量の軸（2 × 2）: 同じ族の v5bag_ped − v5bag（spec の judge_feature・selection_caveat）
    for target in ("win", "top3"):
        if ("v5bag", target) in kept and ("v5bag_ped", target) in kept:
            for lam in lams:
                ra, rb = kept[("v5bag_ped", target)][lam], kept[("v5bag", target)][lam]
                report["feature_axis"][f"{target}|lam{lam}"] = {f"{f}:ped-v5bag": paired_boot(ra, base, target, f, f, years, res_b=rb)
                                                                for f in ("base", "pw", "pord")}
            print("feature_axis", target, json.dumps(report["feature_axis"].get(f"{target}|lam{lams[0]}"), ensure_ascii=False), flush=True)
    dest = DOCS / f"eval_{period}.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tok.finish(result_path=dest)
    print("done", dest, round(time.time() - t0), "秒", flush=True)


def meeting_configs(sel: dict, fs: str, target: str, lam: float) -> dict:
    """開催単位の分けに使う設定: 時間の順の選び（λ 主・外側 2019〜21）で、族ごとに最も多く選ばれた (候補, 較正)。"""
    py = sel["results"][f"{fs}|{target}|lam{lam}"]["per_year"]
    out = {}
    for fam in ("base", "pw", "pord"):
        c = pd.Series([(py[t][fam]["chosen"], py[t][fam]["cal"]) for t in sorted(py)]).value_counts()
        out[fam] = c.index[0]
    return out


def cmd_meeting() -> None:
    """開催単位の 5 分割（報告だけ）。設定は時間の順で選んだものに固定。折 j の較正の側で当てた較正を、外側の行に当てて数える。
    rules-council 1 号の条件: 買い方の入力・版の選び・設定の選び直しに使わない。2022〜24 も含むので、時間の順の confirm の後に流す。"""
    sp = TR.spec()
    sel = json.loads((DOCS / "eval_select.json").read_text(encoding="utf-8"))
    assert (DOCS / "eval_confirm.json").exists(), "時間の順の 2022〜24 の確かめ（confirm）の後に流す"
    tok = _ledger("meeting", "2016-01-01", "2024-12-31", "eval_only")
    base = base_frame(2024)
    lam = sp["metric"]["lambda_main"]
    keyed = base.set_index(["race_id", "horse_id"])
    report = {"spec_version": sp["version"], "lambda": lam, "results": {}}
    for fs in ("v5bag", "v5bag_ped", "v6e"):
        for target in ("win", "top3"):
            total = 1.0 if target == "win" else 3.0
            yname = "y1" if target == "win" else "y3"
            try:
                cfg = meeting_configs(sel, fs, target, lam)
                pooled = {}
                for fam, (cname, cal) in cfg.items():
                    r = int(cname.split("|r")[1])
                    if fam == "pord":
                        k = float(cname.split("|")[0][1:])
                        a = load_pred(fs, target, "ext_w1", base, "meeting")
                        nb = load_pred(fs, "notbig", "orig", base, "meeting")
                        assert (a[["race_id", "horse_id", "fold"]].to_numpy() == nb[["race_id", "horse_id", "fold"]].to_numpy()).all()
                        a["s"] = _ord_score(a[f"p_r{r}"].to_numpy(float), nb[f"p_r{r}"].to_numpy(float), k)
                    else:
                        var = "orig" if fam == "base" else f"ext_w{cname.split('|')[0][1:]}"
                        a = load_pred(fs, target, var, base, "meeting")
                        a["s"] = a[f"p_r{r}"].to_numpy(float)
                    outs = []
                    for j in range(5):
                        c = a[(a["fold"] == j) & (a["role"] == "calib")]
                        o = a[(a["fold"] == j) & (a["role"] == "outer")].reset_index(drop=True)
                        yc = keyed.loc[list(zip(c["race_id"], c["horse_id"])), yname].to_numpy(float)
                        p = calibrate(cal, c["s"].to_numpy(), yc, c["race_id"].to_numpy(), o["s"].to_numpy(), o["race_id"].to_numpy(), total)
                        o["p"] = p
                        outs.append(o)
                    o = pd.concat(outs, ignore_index=True)
                    d = keyed.loc[list(zip(o["race_id"], o["horse_id"]))].reset_index()
                    m = outer_metrics(o["p"].to_numpy(), o["s"].to_numpy(), d, target, sp)
                    pooled[fam] = (o.assign(**{yname: d[yname].to_numpy(), "date": d["date"].to_numpy()}), m, f"{cname}/{cal}")
            except (FileNotFoundError, KeyError) as e:
                print("予測か時間の順の選びがまだ揃っていない", fs, target, e, flush=True)
                continue
            boots = {}
            for fam in ("pw", "pord"):
                a_, b_ = pooled[fam][0], pooled["base"][0]
                mg = a_.merge(b_[["race_id", "horse_id", "p"]], on=["race_id", "horse_id"], suffixes=("", "_b"))
                y = mg[yname].to_numpy(float)
                if target == "win":
                    mg = mg[y == 1]
                    la, lb_ = -np.log(np.clip(mg["p"], EPS, 1)), -np.log(np.clip(mg["p_b"], EPS, 1))
                else:
                    la, lb_ = ll(mg["p"].to_numpy(), y), ll(mg["p_b"].to_numpy(), y)
                boots[f"{fam}-base"] = _boot(pd.DataFrame({"date": mg["date"].to_numpy(), "a": np.asarray(la), "b": np.asarray(lb_)}), 2000, 20261128)
            report["results"][f"{fs}|{target}"] = {f: {"config": v[2], "metrics": v[1]} for f, v in pooled.items()} | {"boot": boots}
            print(fs, target, json.dumps({f: v[1]["ll"] for f, v in pooled.items()}), json.dumps(boots), flush=True)
    dest = DOCS / "eval_meeting.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tok.finish(result_path=dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "confirm", "meeting"])
    a = ap.parse_args()
    cmd_meeting() if a.cmd == "meeting" else cmd(a.cmd)


if __name__ == "__main__":
    main()

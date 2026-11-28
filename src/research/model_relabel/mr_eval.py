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
    # 評価は平地のレースだけ（spec mr-7。障害は市場の q が無く、学習には今の v5bag と同じく入れる）。予測の表は鍵で行をそろえる
    lab = pd.read_parquet(LB.OUT / "labels_convert_A.parquet", columns=["race_id", "surface", "year"], filters=[("year", "<=", max_year)])
    obst = set(lab.loc[lab["surface"] == "障", "race_id"])
    m["_row"] = np.arange(len(m))
    qmiss = set(m.loc[m["q_final"].isna(), "race_id"])           # 平地にも q の欠けたレースがある（2023 年に 1 行）。勝ち・複勝・Benter の行をそろえる
    drop = obst | qmiss
    print(f"評価の集合: 障害 {len(obst)} レース・q の欠けた平地 {len(qmiss - obst)} レースを除く", flush=True)
    m = m[~m["race_id"].isin(drop)].reset_index(drop=True)
    assert m["q_final"].notna().all()
    return m


def load_pred(fs, target, variant, base: pd.DataFrame, split="time", conv="convert_A") -> pd.DataFrame:
    """完了の名前の予測の表だけを読む（未完は .partial で、ここでは読まない）。time は base と行の順が同じことを確かめる。"""
    p = OUT / "pred" / split / f"{fs}_{target}_{variant}_{conv}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    x = pd.read_parquet(p, filters=[("year", "<=", int(base["year"].max()))])
    if split == "time":
        x = x.iloc[base["_row"].to_numpy()].reset_index(drop=True)          # 平地の行だけ（base の _row は meta の行の番号）
        assert (x["race_id"].to_numpy() == base["race_id"].to_numpy()).all() and (x["horse_id"].to_numpy() == base["horse_id"].to_numpy()).all()
    else:
        x = x[x["race_id"].isin(set(base["race_id"]))].reset_index(drop=True)
    return x


def _ord_score(pb, pa, k):
    return 1 / (1 + np.exp(-(_logit(pb) + k * np.log(np.clip(pa, EPS, 1)))))


MISSING: list[str] = []       # 予測が未完で飛ばした族（report に残す。試した数と結果を合わせるため）


def candidates(fs: str, target: str, sp: dict, base: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """族ごとに {候補名: 点数の列（全行）}。"""
    rounds = sp["select"]["rounds_v6e" if fs == "v6e" else "rounds_v5bag"]
    if fs in ("v5bag_mfs", "v5bag_mfc05", "v5bag_mfc03"):
        # 市場と相関の高い列を外す版（spec の market_corr）: 元のラベルと、規則で決めた 1 つの w だけ（market_choice.json）
        w = json.loads((DOCS / "market_choice.json").read_text(encoding="utf-8"))[target]
        b, e = load_pred(fs, target, "orig", base), load_pred(fs, target, f"ext_w{w}", base)
        return {"base": {f"orig|r{r}": b[f"p_r{r}"].to_numpy(float) for r in rounds},
                "pw": {f"w{w}|r{r}": e[f"p_r{r}"].to_numpy(float) for r in rounds}}
    fam: dict[str, dict[str, np.ndarray]] = {"base": {}, "pw": {}, "pord": {}}
    with_p1 = target == "top3" and fs in ("v5bag", "v5bag_ped", "v6e")      # 1 馬身の版は v5bag_mf・mfc では作らない（mr-5・6）
    if with_p1:
        fam["pw1"], fam["pord1"] = {}, {}               # 複勝の 1 馬身の版（候補の名前は x1 で始めて、1/2 馬身の版と重ねない）
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
    if fs == "v5bag":                              # 本命決着のレースを学習から除いた版（_nc。候補の名前は nc で始める）
        try:
            nc = {v: load_pred(fs, target, f"{v}_nc", base) for v in ["orig"] + [f"ext_w{w}" for w in sp["penalty"]["P_w"]["w"]]}
            fam["base_nc"] = {f"ncorig|r{r}": nc["orig"][f"p_r{r}"].to_numpy(float) for r in rounds}
            fam["pw_nc"] = {f"ncw{w}|r{r}": nc[f"ext_w{w}"][f"p_r{r}"].to_numpy(float) for w in sp["penalty"]["P_w"]["w"] for r in rounds}
        except FileNotFoundError as err:
            print("本命決着を除いた版の予測がまだ無いので base_nc・pw_nc を飛ばす", err, flush=True)
            MISSING.append(f"{fs}|{target}|base_nc・pw_nc")
    if with_p1:
        try:
            ext1 = {w: load_pred(fs, target, f"ext1_w{w}", base) for w in sp["penalty"]["P_w"]["w"]}
        except FileNotFoundError as err:          # 1 馬身の版が未完でも、ほかの族の評価は続ける
            print("1 馬身の版の予測がまだ無いので pw1・pord1 を飛ばす", fs, err, flush=True)
            MISSING.append(f"{fs}|{target}|pw1・pord1")
            del fam["pw1"], fam["pord1"]
            return fam
        for w in sp["penalty"]["P_w"]["w"]:
            e = ext1[w]
            for r in rounds:
                fam["pw1"][f"x1w{w}|r{r}"] = e[f"p_r{r}"].to_numpy(float)
        e1 = ext1[1]
        for k in sp["penalty"]["P_ord"]["kappa"]:
            for r in rounds:
                fam["pord1"][f"x1k{k}|r{r}"] = _ord_score(e1[f"p_r{r}"].to_numpy(float), nb[f"p_r{r}"].to_numpy(float), k)
    return fam


def variant_of(cname: str) -> tuple[str, float | None]:
    """候補の名前から、読む予測の変種と κ（pord のとき）を返す。"""
    head = cname.split("|")[0]
    if head.startswith("nc"):
        v, k = variant_of(head[2:] + "|r0")
        return f"{v}_nc", k
    if head == "orig":
        return "orig", None
    if head.startswith("x1w"):
        return f"ext1_w{head[3:]}", None
    if head.startswith("x1k"):
        return "ext1_w1", float(head[3:])
    if head.startswith("w"):
        return f"ext_w{head[1:]}", None
    return "ext_w1", float(head[1:])


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
    return {"a": round(float(a), 4), "b": round(float(b), 4), "a_raw": float(a), "b_raw": float(b), "nll_combo": float(_clogit_nll([a, b], *g)),
            "nll_market": _MARKET[year_key], "races": int(g[3].max() + 1)}


def benter_rows(p_app, d_app, a, b) -> pd.DataFrame:
    """Benter の 2 段の、年 t の勝ち馬ごとの −log p（q の欠けたレースは落とす）。対のブートストラップ用（鍵 race_id・date）。"""
    rid = pd.factorize(d_app["race_id"])[0]
    ok = pd.Series(d_app["q_final"].notna().to_numpy()).groupby(rid).transform("all").to_numpy()
    d = d_app[ok].reset_index(drop=True)
    z = a * np.log(np.clip(np.asarray(p_app)[ok], EPS, 1)) + b * np.log(np.clip(d["q_final"].to_numpy(float), EPS, 1))
    r = d["race_id"].to_numpy()
    mx = pd.Series(z).groupby(r).transform("max").to_numpy()
    lse = np.log(pd.Series(np.exp(z - mx)).groupby(r).transform("sum").to_numpy()) + mx
    w = d["y1"].to_numpy(bool)          # 1 着同着のレースは勝ち馬が 2 頭（鍵は race_id＋horse_id）
    return pd.DataFrame({"key": (d["race_id"] + "_" + d["horse_id"]).to_numpy()[w], "date": d.loc[w, "date"].to_numpy(), "nll": -(z - lse)[w]})


_MARKET3: dict = {}


def benter3(p_fit, d_fit, p_app, d_app, year_key) -> tuple[dict, pd.DataFrame]:
    """複勝の 2 段: y3 を logit(p̂) と logit(q_final) の二値のロジスティック回帰で当てる（年 t−1 で当て、年 t で数える）。市場だけの回帰と比べる。"""
    from sklearn.linear_model import LogisticRegression

    def prep(p, d):
        ok = d["q_final"].notna().to_numpy()
        Z = np.column_stack([_logit(np.asarray(p)[ok]), _logit(d["q_final"].to_numpy(float)[ok])])
        return Z, d["y3"].to_numpy(float)[ok], d[ok]
    Zf, yf, _ = prep(p_fit, d_fit)
    Za, ya, da = prep(p_app, d_app)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(Zf, yf)
    pa = m.predict_proba(Za)[:, 1]
    if year_key not in _MARKET3:
        m0 = LogisticRegression(C=1e6, max_iter=1000).fit(Zf[:, 1:], yf)
        _MARKET3[year_key] = float(np.mean(ll(m0.predict_proba(Za[:, 1:])[:, 1], ya)))
    rows = pd.DataFrame({"key": (da["race_id"] + "_" + da["horse_id"]).to_numpy(), "date": da["date"].to_numpy(), "nll": ll(pa, ya)})
    return {"a": round(float(m.coef_[0][0]), 4), "b": round(float(m.coef_[0][1]), 4), "nll_combo": float(rows["nll"].mean()),
            "nll_market": _MARKET3[year_key], "n": int(len(rows))}, rows


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
        if any(k.startswith(f"w{w}|") for k in fam["pw"]) and fs not in ("v5bag_mfs", "v5bag_mfc05", "v5bag_mfc03"):     # w が 1 つの版は固定の群を作らない
            groups[f"pw_fixed_w{w}"] = {k: v for k, v in fam["pw"].items() if k.startswith(f"w{w}|")}
        if "pw1" in fam:
            groups[f"pw1_fixed_w{w}"] = {k: v for k, v in fam["pw1"].items() if k.startswith(f"x1w{w}|")}
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
                    m["benter"] = benter(inner[key][0], d1, p0, d0, t)
                    brows = benter_rows(p0, d0, m["benter"]["a_raw"], m["benter"]["b_raw"])
                else:
                    m["benter"], brows = benter3(inner[key][0], d1, p0, d0, t)
                outer[key] = (p0, m, brows)
            return outer[key]

        for lam in lams:
            rec = {}
            for gname, cands in groups.items():
                keys = [(c, cal) for c in cands for cal in sp["select"]["calibration"]]
                key = min(keys, key=lambda k: inner[k][1][lam])
                p0, m, brows = outer_of(key)
                rec[gname] = {"chosen": key[0], "cal": key[1], "L_inner": round(inner[key][1][lam], 6), "outer": m, "_p": p0, "_b": brows}
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


def paired_boot_benter(res_a: dict, fam_a: str, res_b: dict, fam_b: str, years, n_boot=2000, seed=20261128) -> dict:
    """Benter の 2 段の上乗せの差（a − b、勝ち: 勝ち馬の −log p・複勝: y3 の対数損失）の、開催日で束ねた対のブートストラップ。
    市場だけの値は両者で同じなので、上乗せの差 ＝ 2 段の損失の差。"""
    rows = []
    for t in years:
        ra, rb = res_a["years"][t][fam_a]["_b"], res_b["years"][t][fam_b]["_b"]
        m = ra.merge(rb[["key", "nll"]], on="key", suffixes=("", "_b"))
        assert len(m) == len(ra) == len(rb), "2 段の行がそろわない"
        rows.append(pd.DataFrame({"date": m["date"].to_numpy(), "a": m["nll"].to_numpy(), "b": m["nll_b"].to_numpy()}))
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
    a2 = 0.05 / 12 / 2              # Bonferroni（族 12 個・両側。spec の confirm_plan）
    return {"diff": round(float(est), 6), "ci95": [round(float(np.quantile(diffs, 0.025)), 6), round(float(np.quantile(diffs, 0.975)), 6)],
            "ci_bonf12": [round(float(np.quantile(diffs, a2)), 6), round(float(np.quantile(diffs, 1 - a2)), 6)],
            "days": int(len(x))}


_DNF: dict = {}


def dnf_rows(base: pd.DataFrame) -> pd.DataFrame:
    """学習の表 X に行の無い、評価のレースの中止・失格などの馬（runners にあり着順が無い馬。点検役の指摘 1・11/27）。ラベルは 0。"""
    if "rows" not in _DNF:
        mx = int(base["year"].max())
        rn = pd.read_parquet(RUNNERS, columns=["race_id", "horse_id", "year", "fin", "odds_final"], filters=[("year", "<=", mx)])
        rn["race_id"], rn["horse_id"] = rn["race_id"].astype(str), rn["horse_id"].astype(str)
        meta = pd.read_parquet(META, columns=["race_id", "horse_id", "year"], filters=[("year", "<=", mx)])
        k = set(zip(meta["race_id"].astype(str), meta["horse_id"].astype(str)))
        notin = np.array([(a, b) not in k for a, b in zip(rn["race_id"], rn["horse_id"])])
        x = rn[notin & rn["fin"].isna().to_numpy()]      # runners は出走した馬だけ（取消・除外は入っていない。確定オッズの欠けは 0 行）
        _DNF["rows"] = x[x["race_id"].isin(set(base["race_id"]))].reset_index(drop=True)
    return _DNF["rows"]


def dnf_sensitivity(res: dict, base: pd.DataFrame, target: str, years) -> dict:
    """感度: 評価のレースに、X に行の無い中止・失格の馬をラベル 0 で戻す。予測はレースの最小（min）と中央（median）の 2 通りで置き、
    レース内で和をそろえ直して、勝ち馬の −log p（勝ち）・y3 の対数損失（複勝）を数える。族ごとの値と、族 − base の差。"""
    yname = "y1" if target == "win" else "y3"
    total = 1.0 if target == "win" else 3.0
    dn = dnf_rows(base)
    out = {"dnf_horses": int(dn["year"].isin(list(years)).sum())}
    fams = list(res["years"][years[0]])          # 固定の w の群も含める（mfs − mf の対は mf の pw_fixed_w{w} を使う）
    for how in ("none", "min", "median"):           # none = 戻さない（同じ数え方の比べ先）
        vals = {}
        for f in fams:
            per_year = []
            for t in years:
                d0 = base[base["year"] == t].reset_index(drop=True)
                p = pd.Series(res["years"][t][f]["_p"])
                fill = p.groupby(d0["race_id"].to_numpy()).agg(how) if how != "none" else pd.Series(dtype=float)
                add = dn[dn["year"] == t] if how != "none" else dn.iloc[0:0]
                rid = np.concatenate([d0["race_id"].to_numpy(), add["race_id"].to_numpy()])
                pp = np.concatenate([p.to_numpy(), add["race_id"].map(fill).to_numpy(float)])
                y = np.concatenate([d0[yname].to_numpy(float), np.zeros(len(add))])
                pp = race_normalize(pp, rid, total)
                if target == "win":
                    per_year.append(float(-np.log(np.clip(pp[y == 1], EPS, 1)).mean()))
                else:
                    per_year.append(float(ll(pp, y).mean()))
            vals[f] = float(np.mean(per_year))          # summarize と同じく年ごとの平均の単純な平均
        out[how] = {f: round(v, 6) for f, v in vals.items()}
        out[f"{how}_minus_base"] = {f: round(v - vals["base"], 6) for f, v in vals.items() if f != "base"}
    # spec の dnf_rule: 3 通りで族 − base の差の符号がそろうか（そろわない比べは「決まらない」）
    out["direction_agree"] = {f: len({np.sign(out[f"{h}_minus_base"][f]) for h in ("none", "min", "median")}) == 1
                              for f in out["none_minus_base"]}
    return out


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
                    period=(lo, hi), purpose=purpose, reads=("着順",),
                    frozen_paths=[TR.SPEC] + [p for p in (TR.MARKET_CORR, TR.MARKET_CHOICE) if p.exists()],
                    reason="拡張ラベルのモデルの本物のラベルでの当てはまり（回収率は数えない）")


def cmd(period: str, only: str | None = None) -> None:
    sp = TR.spec()
    if only:          # 試しは select だけ（confirm の 1 回を使い切らない）。組の名前も確かめる
        assert period == "select", "--only は select だけ"
        fs_, tg_ = only.split("|")
        assert fs_ in ("v5bag", "v5bag_ped", "v5bag_mf", "v5bag_mfs", "v5bag_mfc05", "v5bag_mfc03", "v6e") and tg_ in ("win", "top3"), only
    key = "outer_main" if period == "select" else "outer_confirm_once"
    years = range(sp["years"][key][0], sp["years"][key][1] + 1)
    tok = _ledger(period if not only else f"{period}-smoke", f"{min(years) - 2}-01-01", f"{max(years)}-12-31",
                  "select" if period == "select" else "confirm")
    t0 = time.time()
    base = base_frame(max(years))
    lams = [sp["metric"]["lambda_main"]] + list(sp["metric"]["lambda_sensitivity"])
    report = {"spec_version": sp["version"], "period": period, "years": list(years), "results": {}, "feature_axis": {}}
    kept: dict = {}
    for fs in ("v5bag", "v5bag_ped", "v5bag_mf", "v5bag_mfs", "v5bag_mfc05", "v5bag_mfc03", "v6e"):
        for target in ("win", "top3"):
            if only and f"{fs}|{target}" != only:
                continue
            try:
                allres = nested_time(fs, target, years, lams, sp, base)
            except FileNotFoundError as e:
                print("予測がまだ揃っていない", fs, target, e, flush=True)
                continue
            if fs != "v6e":
                kept[(fs, target)] = allres
            if fs == "v5bag_mf" and (DOCS / "market_choice.json").exists():
                # market_choice.json が、この回の v5bag_mf の選びから同じ規則で作った値と一致するか（古い w のまま使わない）
                chk = _modal_w(allres[sp["metric"]["lambda_main"]], years)
                got = json.loads((DOCS / "market_choice.json").read_text(encoding="utf-8"))
                assert got.get(target) == chk, f"market_choice.json（{got}）が今回の v5bag_mf の選び（{target}: {chk}）と違う"
            for lam, res in allres.items():
                summ = summarize(res, years)
                boots = {f"{a}-base": paired_boot(res, base, target, a, "base", years)
                         for a in res["years"][years[0]] if a != "base"}
                for a in [f for f in res["years"][years[0]] if f != "base"]:      # Benter の 2 段の上乗せの差（族 − base、同じ特徴量の版の中）
                    boots[f"{a}-base:benter"] = paired_boot_benter(res, a, res, "base", years)
                for a, b in (("pw_nc", "pw"), ("base_nc", "base")):          # 除きの効き目（同じ族どうし）
                    if a in res["years"][years[0]]:
                        boots[f"{a}-{b}"] = paired_boot(res, base, target, a, b, years)
                per_year = {str(t): {f: {k: v for k, v in r.items() if not k.startswith("_")} for f, r in res["years"][t].items()} for t in years}
                report["results"][f"{fs}|{target}|lam{lam}"] = {"summary": summ, "boot": boots, "per_year": per_year,
                                                                "dnf_sensitivity": dnf_sensitivity(res, base, target, years)}
                print(f"{fs}|{target}|lam{lam}", json.dumps({f: {k: summ[f].get(k) for k in ("ll", "L_main", "winner_nlp", "race_auc", "benter_gain")}
                                                             for f in summ}, ensure_ascii=False), json.dumps(boots, ensure_ascii=False), flush=True)
    # 特徴量の軸: 同じ族の v5bag_ped − v5bag（spec の judge_feature・selection_caveat）と v5bag_mf − v5bag（judge_market_free）
    for target in ("win", "top3"):
        for alt, ref in (("v5bag_ped", "v5bag"), ("v5bag_mf", "v5bag"), ("v5bag_mfs", "v5bag_mf"),
                         ("v5bag_mfc05", "v5bag_mfs"), ("v5bag_mfc03", "v5bag_mfs")):     # mfc は s_* を外した mfs と比べる（漏れの列の効きを分ける。mr-8）
            if (ref, target) not in kept or (alt, target) not in kept:
                continue
            for lam in lams:
                ra, rb = kept[(alt, target)][lam], kept[(ref, target)][lam]
                fams = [f for f in ra["years"][years[0]] if "_fixed_" not in f and f in rb["years"][years[0]]]
                pairs = [(f, f) for f in fams]
                if alt == "v5bag_mfs":     # mfs の pw は w を 1 つに決めた版なので、mf の側は同じ w の固定の群と対にする（同じラベルどうし）。mfc − mfs は族どうしのまま
                    w = json.loads((DOCS / "market_choice.json").read_text(encoding="utf-8"))[target]
                    pairs = [("base", "base"), ("pw", f"pw_fixed_w{w}")]
                da = report["results"][f"{alt}|{target}|lam{lam}"]["dnf_sensitivity"]
                db = report["results"][f"{ref}|{target}|lam{lam}"]["dnf_sensitivity"]
                report["feature_axis"][f"{alt}|{target}|lam{lam}"] = {
                    **{f"{a}:{alt}-{ref}:{b}": paired_boot(ra, base, target, a, b, years, res_b=rb) for a, b in pairs},
                    **{f"{a}:{alt}-{ref}:{b}:benter": paired_boot_benter(ra, a, rb, b, years) for a, b in pairs},
                    # spec の dnf_rule: 中止の馬の扱いの 3 通りで、差の符号がそろうか
                    **{f"{a}:{alt}-{ref}:{b}:dnf": {h: round(da[h][a] - db[h][b], 6) for h in ("none", "min", "median")} | {
                        "direction_agree": len({np.sign(da[h][a] - db[h][b]) for h in ("none", "min", "median")}) == 1}
                       for a, b in pairs}}
            print("feature_axis", alt, target, json.dumps(report["feature_axis"].get(f"{alt}|{target}|lam{lams[0]}"), ensure_ascii=False), flush=True)
    report["missing_families"] = MISSING
    if MISSING:
        print("[警告] 飛ばした族がある（揃ってから流し直す）:", MISSING, flush=True)
    dest = DOCS / f"eval_{period}.json" if not only else LB.OUT / f"eval_{period}_smoke.json"     # 試しは docs に書かない
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tok.finish(result_path=dest)
    print("done", dest, round(time.time() - t0), "秒", flush=True)


def meeting_configs(sel: dict, fs: str, target: str, lam: float) -> dict:
    """開催単位の分けに使う設定: 時間の順の選び（λ 主・外側 2019〜21）で、族ごとに最も多く選ばれた (候補, 較正)。"""
    py = sel["results"][f"{fs}|{target}|lam{lam}"]["per_year"]
    out = {}
    for fam in [f for f in next(iter(py.values())) if "_fixed_" not in f and not f.endswith("_nc")]:   # _nc の版は時間の順だけ
        c = pd.Series([(py[t][fam]["chosen"], py[t][fam]["cal"]) for t in sorted(py)]).value_counts()
        out[fam] = c.index[0]
    return out


def cmd_meeting() -> None:
    """開催単位の 5 分割（報告だけ）。設定は時間の順で選んだものに固定。折 j の較正の側で当てた較正を、外側の行に当てて数える。
    rules-council 1 号の条件: 買い方の入力・版の選び・設定の選び直しに使わない。2022〜24 も含むので、時間の順の confirm の後に流す。"""
    sp = TR.spec()
    sel = json.loads((DOCS / "eval_select.json").read_text(encoding="utf-8"))
    assert sel["spec_version"] == sp["version"], f"eval_select.json の spec の版（{sel['spec_version']}）が今の版（{sp['version']}）と違う"
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
                    var, k = variant_of(cname)
                    if k is not None:
                        a = load_pred(fs, target, var, base, "meeting")
                        nb = load_pred(fs, "notbig", "orig", base, "meeting")
                        assert (a[["race_id", "horse_id", "fold"]].to_numpy() == nb[["race_id", "horse_id", "fold"]].to_numpy()).all()
                        a["s"] = _ord_score(a[f"p_r{r}"].to_numpy(float), nb[f"p_r{r}"].to_numpy(float), k)
                    else:
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
            for fam in [f for f in pooled if f != "base"]:
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


def _modal_w(res: dict, years) -> int:
    ws = pd.Series([int(res["years"][t]["pw"]["chosen"].split("|")[0][1:]) for t in years])
    c = ws.value_counts()
    return int(min(c.index[c == c.max()]))


def market_choice() -> dict:
    """spec の market_corr.labels の規則: v5bag_mf の時間の順の内側の選び（λ 主・外側 2019〜21）で、族 pw の中で最も多く選ばれた w（同数なら小さい w）。"""
    sp = TR.spec()
    sel = json.loads((DOCS / "eval_select.json").read_text(encoding="utf-8"))
    assert sel["spec_version"] == sp["version"], f"eval_select.json の spec の版（{sel['spec_version']}）が今の版（{sp['version']}）と違う"
    out = {"source": {"spec_version": sel["spec_version"], "eval_select": "v5bag_mf の λ 主・外側 2019〜21 の族 pw の選び"}}
    for target in ("win", "top3"):
        py = sel["results"][f"v5bag_mf|{target}|lam{sp['metric']['lambda_main']}"]["per_year"]
        ws = pd.Series([int(py[t]["pw"]["chosen"].split("|")[0][1:]) for t in sorted(py)])
        c = ws.value_counts()
        out[target] = int(min(c.index[c == c.max()]))
    (DOCS / "market_choice.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("market_choice", out, flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "confirm", "meeting", "market_choice"])
    ap.add_argument("--only", default=None, help="試し: fs|target の 1 組だけ（結果は model_relabel_out に書く）")
    a = ap.parse_args()
    if a.cmd == "market_choice":
        market_choice()
    else:
        cmd_meeting() if a.cmd == "meeting" else cmd(a.cmd, a.only)


if __name__ == "__main__":
    main()

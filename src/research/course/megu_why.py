"""めぐ分析の過去データの条件（凍結の第 3 版。成績表 B）が、回収率 100% を超えない理由を確かめる（評価だけ。条件は選び直さない）。

- collect: 訓練 2016〜2024・2025・2026 年の走ごとの表（市場の勝率・買う印・当たった条件の数）と、条件ごとの期間の値を書く。
  買い方は course_rule_bets と同じ（表示の条件のうち訓練の回収率 100% 以上に当てはまる馬を、馬ごとに重複を除いて 1 点）。
  2025・2026 年の着順・払戻を読むので、台帳に eval_only を申告する（family course）。
- perm: 並べ替えの帰無（訓練 2016〜2024 だけ。2025・2026 年は読まない）。単位 × 確定の人気（13 番人気以下はまとめる）の中で
  切り口の水準の行を入れ替え、凍結と同じ探し方・絞り方・表示の上位 5・訓練 100% 以上の判定を流し直す。
  人気と着順・払戻の関係はそのまま残り、切り口と結果の関係だけが消える（「条件は人気より先の情報を持たない」の帰無）。
  消しの条件は B に使わないので探さない（速さのため）。--n 0 の 1 回目は入れ替え無し（凍結の再現の確かめ）。
- report: 上の 2 つから、仮説ごとの数字を docs/operations/retrain_trials/megu_why_flat.json に書く。

使い方（重いので nice -n 19）:
  KEIBA_SESSION=megu-why python -m src.research.course.megu_why collect
  python -m src.research.course.megu_why perm --seeds 0,1,2
  python -m src.research.course.megu_why report
出力の置き場: <COURSE_OUT>/megu_why。費用 0 円（手元の表だけ。GCS を読まない）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.research.course import run as CR
from src.research.course import search as SE
from src.research.course import v2 as V2

OUTW = CR.OUT / "megu_why"
OUT_REPO = CR.REPO / "docs/operations/retrain_trials/megu_why_flat.json"
POP_CAP = 13
ODDS_BANDS = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 1e9]


# ---------------------------------------------------------------- 共通
def market_q(d: pd.DataFrame) -> np.ndarray:
    """確定オッズの 1/オッズ をレースの中で和 1 に直した市場の勝率（v3.market_q と同じ）。"""
    from src.research.course.v3 import market_q as MQ
    return MQ(d)


def shown_rules(kept: dict, roi_of) -> dict:
    """単位ごとに、めぐ分析に出る買いの条件（ordered_conds の上位 RULE_TOP。訓練で 1 件以上当たるもの）と、その訓練の単・複の回収率。
    roi_of(u, j, b) は (n, roi_t, roi_f) を返す。"""
    from src.research.course.v3 import RULE_TOP, ordered_conds
    out = {}
    for u, cu in kept.items():
        buy, _ = ordered_conds(cu)
        rows = []
        for j, b in enumerate(buy):
            n, rt, rf = roi_of(u, j, b)
            if not n:
                continue
            rows.append({"i": j, "n": n, "roi_t": rt, "roi_f": rf})
            if len(rows) >= RULE_TOP:
                break
        out[u] = rows
    return out


def bet_masks(d: pd.DataFrame, L: pd.DataFrame, kept: dict, c, rules: dict) -> dict:
    """走ごとの買う印（単・複）と、当たった買いの条件の数、条件ごとの当たりの行。"""
    from src.research.course.v3 import ordered_conds, units_of
    X, types, code = V2._codes(L, kept)
    unit = units_of(d, c)
    res = {"buy_t": np.zeros(len(d), bool), "buy_f": np.zeros(len(d), bool), "nr_t": np.zeros(len(d), int), "nr_f": np.zeros(len(d), int),
           "rule_rows": {}}
    for u, cu in kept.items():
        idx = np.flatnonzero(unit == u)
        if not len(idx) or u not in rules:
            continue
        conds, _ = ordered_conds(cu)
        Xu = X[idx]
        for r in rules[u]:
            m = V2._mask_names(Xu, types, code, conds[r["i"]])
            rows = idx[m]
            res["rule_rows"][(u, r["i"])] = rows
            if r.get("roi_t") is not None and r["roi_t"] >= 1.0:
                res["buy_t"][rows] = True
                res["nr_t"][rows] += 1
            if r.get("roi_f") is not None and r["roi_f"] >= 1.0:
                res["buy_f"][rows] = True
                res["nr_f"][rows] += 1
    return res


def roi_of(Y: np.ndarray, m: np.ndarray) -> tuple[int, float | None, float | None]:
    s = Y[m].sum(0)
    return int(s[0]), (float(s[6] / s[5]) if s[5] else None), (float(s[7] / s[5]) if s[5] else None)


# ---------------------------------------------------------------- collect
def runner_table(d: pd.DataFrame, L: pd.DataFrame, kept: dict, c, rules: dict, period: str) -> tuple[pd.DataFrame, list[dict]]:
    from src.research.course.v3 import units_of
    Y = SE.y_matrix(d)
    bm = bet_masks(d, L, kept, c, rules)
    t = pd.DataFrame({"period": period, "race_id": d["race_id"].astype(str).to_numpy(), "date": pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
                      "year": pd.to_numeric(d["year"]).to_numpy(), "unit": units_of(d, c), "surf": d["surf"].astype(str).to_numpy(),
                      "group": d["group"].astype(str).to_numpy(), "horse_id": d["horse_id"].astype(str).to_numpy(),
                      "odds": pd.to_numeric(d["odds_final"], errors="coerce").to_numpy(), "pop": pd.to_numeric(d["pop_final"], errors="coerce").to_numpy(),
                      "q": market_q(d), "field": pd.to_numeric(d["field"], errors="coerce").to_numpy(),
                      "win": Y[:, 2], "top3": Y[:, 1], "pay": Y[:, 5], "rt": Y[:, 6], "rf": Y[:, 7],
                      "buy_t": bm["buy_t"], "buy_f": bm["buy_f"], "nr_t": bm["nr_t"], "nr_f": bm["nr_f"]})
    rr = []
    yr = t["year"].to_numpy()
    q = t["q"].to_numpy()
    for (u, i), rows in bm["rule_rows"].items():
        r = next(x for x in rules[u] if x["i"] == i)
        row = {"period": period, "unit": u, "i": i, "n_train": r["n"], "roi_t_train": r["roi_t"], "roi_f_train": r["roi_f"]}
        for tag, sel in (("", np.ones(len(rows), bool)), ("_a", yr[rows] <= 2021), ("_b", yr[rows] >= 2022)):
            rs = rows[sel]
            s = Y[rs].sum(0)
            row |= {f"n{tag}": int(s[0]), f"w{tag}": int(s[2]), f"k{tag}": int(s[1]), f"np{tag}": int(s[5]), f"tan{tag}": float(s[6]),
                    f"fuku{tag}": float(s[7]), f"q{tag}": float(q[rs].sum()), f"fh{tag}": int((Y[rs, 7] > 0).sum())}
        rr.append(row)
    return t, rr


def cmd_collect() -> None:
    from src.research.course.v3 import EVAL_MARK3, OUT3, PREREG3
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    if CR._sha_file(OUT3 / P["cells_file"]) != P["sha256_cells"]:
        sys.exit("条件の表が凍結の sha と違う")
    mark = json.loads(EVAL_MARK3.read_text(encoding="utf-8")) if EVAL_MARK3.exists() else {}
    if "evaluated_at" not in mark:
        sys.exit("2026 年の 1 回だけの評価が済んでいない")
    CR.wait_resources()
    e = CR.env(2026) | {"FOLKLORE_OUT": str(CR.OUT / "eval" / "master"), "GRADED_OUT": str(CR.OUT / "eval")}
    subprocess.run([sys.executable, "-m", "src.research.course.megu_why", "_collect_inner"], env=e, cwd=CR.REPO, check=True)
    # 注: 2026 年は _frame_2026 の表のまま（2026-11-23 の collect では 11-15 まで）。前向きの期間（11-21〜）が入る表になったら、日付で切る


def _collect_inner() -> None:
    from src.ops import data_ledger as DL
    from src.research.course.course_rule_bets import STATS, chosen_rules
    from src.research.course.v3 import C3, OUT3, PREREG3, _frame_2026, load3
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    if stats["sha256_cells"] != P["sha256_cells"] or CR._sha_file(OUT3 / P["cells_file"]) != P["sha256_cells"]:
        sys.exit("めぐ分析の条件の表・凍結の sha が合わない")
    c = C3(**{k: v for k, v in P["params"].items() if k != "name"})
    kept = json.loads((OUT3 / P["cells_file"]).read_text(encoding="utf-8"))
    rules = chosen_rules(stats, kept)
    OUTW.mkdir(parents=True, exist_ok=True)
    with DL.guard(tool="src/research/course/megu_why.py:_collect_inner", family="course", version=P["sha256_cells"][:12],
                  period=("2025-01-01", "2026-11-20"), purpose="eval_only",
                  reason="めぐ分析の条件（成績表 B）が回収率 100% を超えない理由の診断（較正・縮小・年ごと・配当の偏り・重複）。条件・閾値は選び直さない",
                  frozen_paths=[PREREG3, OUT3 / P["cells_file"]]) as tok:
        d, L, tr, ev = load3()
        tabs, rr = [], []
        for nm, m in (("train", tr), ("2025", ev)):
            if nm == "2025":
                tok.check_dates(d.loc[m, "date"])
            t, r = runner_table(d[m].reset_index(drop=True), L[m].reset_index(drop=True), kept, c, rules, nm)
            tabs.append(t)
            rr += r
        del d, L
        d6, L6 = _frame_2026()
        tok.check_dates(d6["date"])
        t, r = runner_table(d6, L6, kept, c, rules, "2026")
        tabs.append(t)
        rr += r
        T = pd.concat(tabs, ignore_index=True)
        T.to_parquet(OUTW / "runners.parquet", index=False)
        pd.DataFrame(rr).to_parquet(OUTW / "rules.parquet", index=False)
        tok.finish(result_path=OUTW / "runners.parquet")
    print("collect", {p: (int(g["buy_t"].sum()), round(float(g.loc[g.buy_t, "rt"].sum() / g.loc[g.buy_t, "pay"].sum()), 4))
                      for p, g in T.groupby("period")}, flush=True)


# ---------------------------------------------------------------- perm（訓練だけ）
def permute_levels(d: pd.DataFrame, L: pd.DataFrame, c, seed: int) -> pd.DataFrame:
    """単位 × 確定の人気の中で、L の行を入れ替える（seed 0 は入れ替えない）。"""
    from src.research.course.v3 import units_of
    if seed == 0:
        return L
    rng = np.random.default_rng(20261123 + seed)
    pop = np.clip(np.nan_to_num(pd.to_numeric(d["pop_final"], errors="coerce").to_numpy(float), nan=99), 0, POP_CAP).astype(int)
    key = pd.Series(units_of(d, c)).astype(str) + "|" + pd.Series(pop).astype(str)
    perm = np.arange(len(L))
    for _, idx in key.groupby(key).indices.items():
        if len(idx) > 1:
            perm[idx] = idx[rng.permutation(len(idx))]
    return L.iloc[perm].reset_index(drop=True)


def load_train(year_end: int = 2024):
    """load3 と同じ表（良・芝ダ・重賞以外・確定オッズ付き）を、year_end 年までの行だけ読む（2025 年の行は開かない）。"""
    from src.research.course import v3 as V3
    import pyarrow.parquet as pq
    yr = pd.to_numeric(pq.read_table(V3.OUT3 / "frame.parquet", columns=["year"]).column("year").to_pandas()).to_numpy()
    keep_rows = np.flatnonzero(yr <= year_end)
    d = pd.read_parquet(V3.OUT3 / "frame.parquet", filters=[("year", "<=", year_end)])
    L = pd.read_parquet(V3.OUT3 / "levels.parquet").iloc[keep_rows].reset_index(drop=True)
    if len(d) != len(L) or not ((d["race_id"].astype(str).to_numpy() == L["__race_id"].to_numpy())
                                & (d["horse_id"].astype(str).to_numpy() == L["__horse_id"].to_numpy())).all():
        raise ValueError("frame と levels の行がずれている")
    keep = ((d["track"].astype(str) == "良") & d["cls"].map(CR.group_of).notna()).to_numpy()
    d, L = d[keep].reset_index(drop=True), L[keep].reset_index(drop=True).drop(columns=["__race_id", "__horse_id"])
    d["group"] = d["cls"].map(CR.group_of)
    return V2.attach_final_odds(d, range(2016, year_end + 1)), L


def cmd_perm(seeds: list[int], fit_end: int) -> None:
    """fit_end < 2024 のときは、fit_end+1〜2024 年の着順・払戻を確かめに使うので、台帳に申告する（2016〜24 年は開発・開発の検証の期間）。"""
    from src.ops import data_ledger as DL
    OUTW.mkdir(parents=True, exist_ok=True)

    def run() -> None:
        for s in seeds:
            CR.wait_resources()
            e = CR.env(2024)
            subprocess.run([sys.executable, "-m", "src.research.course.megu_why", "_perm_inner", str(s), str(fit_end)], env=e, cwd=CR.REPO, check=True)
    if fit_end >= 2024:
        return run()
    with DL.guard(tool="src/research/course/megu_why.py:cmd_perm", family="course", version="megu_why_refit",
                  period=(f"{fit_end + 1}-01-01", "2024-12-31"), purpose="confirm",
                  reason=f"凍結の第 3 版と同じ設定を 2016〜{fit_end} 年で探し直し、{fit_end + 1}〜2024 年に当てる（seed {seeds}。0 は本物、他は帰無）。選び直しには使わない",
                  frozen_paths=[]) as tok:
        run()
        tok.finish(result_path=OUTW)


def _bets_metrics(Y: np.ndarray, d: pd.DataFrame, bm: dict) -> dict:
    out = {}
    for kind, col, hitc in (("tansho", 6, 2), ("fukusho", 7, 7)):
        m = bm["buy_t" if kind == "tansho" else "buy_f"]
        Yb = Y[m]
        out[kind] = {"bets": int(m.sum()), "roi": float(Yb[:, col].sum() / Yb[:, 5].sum()) if Yb[:, 5].sum() else None,
                     "hit": float((Yb[:, hitc] > 0).mean()) if len(Yb) else None, "win": float(Yb[:, 2].mean()) if len(Yb) else None,
                     "base_same_pop_roi": _same_pop_roi(Y, d, m, col)}
    return out


def _perm_inner(seed: int, fit_end: int) -> None:
    """seed: 0 = 入れ替えなし。fit_end: 条件を探す期間の終わり（2024 = 凍結と同じ、2021 = 2016〜21 で探して 2022〜24 で確かめる）。"""
    from src.research.course import v3 as V3
    P = json.loads(V3.PREREG3.read_text(encoding="utf-8"))
    c = V3.C3(**{k: v for k, v in P["params"].items() if k != "name"})
    t0 = time.time()
    d_all, L_all = load_train(2024)
    yr = pd.to_numeric(d_all["year"]).to_numpy()
    fit = yr <= fit_end
    d, L = d_all[fit].reset_index(drop=True), L_all[fit].reset_index(drop=True)
    orders = V3.level_orders(d, L)                 # 水準の並びは意味で決まる（入れ替えの前の表で 1 回だけ作る）
    V3.level_orders = lambda *a, **k: orders
    Lp = permute_levels(d, L, c, seed)
    SE.search_kill = lambda *a, **k: []          # 消しは B に使わない（探さない）
    cells = V3.fit_cells(d, Lp, c)
    V3.add_market(cells, d, Lp, c)
    kept = V3.select_cells(cells, c)
    Y = SE.y_matrix(d)
    has = any(cu["buy"] for cu in kept.values())
    X, types, code = V2._codes(Lp, kept) if has else (None, [], {})
    unit = V3.units_of(d, c)

    def ro(u, j, b):
        idx = np.flatnonzero(unit == u)
        return roi_of(Y[idx], V2._mask_names(X[idx], types, code, b))
    rules = shown_rules(kept, ro) if has else {}
    empty = {"buy_t": np.zeros(len(d), bool), "buy_f": np.zeros(len(d), bool)}
    bm = bet_masks(d, Lp, kept, c, rules) if has else empty
    res = {"seed": seed, "fit_end": fit_end, "made_at": datetime.now(CR.JST).isoformat(timespec="seconds"),
           "kept_buy": int(sum(len(cu["buy"]) for cu in kept.values())), "shown": int(sum(len(v) for v in rules.values())),
           "rules_t": int(sum(1 for v in rules.values() for r in v if (r["roi_t"] or 0) >= 1)),
           "rules_f": int(sum(1 for v in rules.values() for r in v if (r["roi_f"] or 0) >= 1)),
           "n_rules_median": float(np.median([r["n"] for v in rules.values() for r in v])) if rules else None,
           "in_sample": _bets_metrics(Y, d, bm)}
    if fit_end < 2024:                             # 確かめ: 探した後の年（入れ替えない本物の水準で当てる）
        h = ~fit
        dh, Lh = d_all[h].reset_index(drop=True), L_all[h].reset_index(drop=True)
        Yh = SE.y_matrix(dh)
        bmh = bet_masks(dh, Lh, kept, c, rules) if has else {"buy_t": np.zeros(len(dh), bool), "buy_f": np.zeros(len(dh), bool)}
        res["holdout"] = {"years": [fit_end + 1, 2024], **_bets_metrics(Yh, dh, bmh)}
    res["seconds"] = round(time.time() - t0)
    (OUTW / f"perm_{fit_end}_{seed}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("perm", json.dumps(res, ensure_ascii=False), flush=True)


def _same_pop_roi(Y: np.ndarray, d: pd.DataFrame, m: np.ndarray, col: int) -> float | None:
    """買った馬と同じ人気（単位の芝ダ・群 × 人気）の全馬の回収率を、買った馬の人気の分布で重みづけした値（市場どおりの基準線）。"""
    pop = np.clip(np.nan_to_num(pd.to_numeric(d["pop_final"], errors="coerce").to_numpy(float), nan=99), 0, POP_CAP).astype(int)
    key = pd.Series(d["surf"].astype(str).to_numpy() + "|" + d["group"].astype(str).to_numpy() + "|" + pop.astype(str))
    g = pd.DataFrame({"k": key, "r": Y[:, col], "p": Y[:, 5]}).groupby("k")[["r", "p"]].sum()
    rate = (g["r"] / g["p"].replace(0, np.nan)).to_dict()
    w = key[m].map(rate).dropna()
    return float(w.mean()) if len(w) else None


# ---------------------------------------------------------------- report
def _roi(g: pd.DataFrame, col: str) -> float | None:
    p = g["pay"].sum()
    return float(g[col].sum() / p) if p else None


def _boot(g: pd.DataFrame, col: str, by: str = "race_id", n: int = 1000, seed: int = 20261123) -> tuple[float | None, float | None]:
    """束ね（レース・開催日）のポアソンのブートストラップで回収率の 95% 区間。"""
    if not len(g):
        return None, None
    s = g.groupby(by)[[col, "pay"]].sum()
    w = np.random.default_rng(seed).poisson(1.0, size=(n, len(s)))
    bs = (w @ s[col].to_numpy()) / np.maximum(w @ s["pay"].to_numpy(), 1e-9)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def same_pop(T: pd.DataFrame, m: pd.Series, col: str) -> float | None:
    """同じ期間・芝ダ・群・人気の、買わなかった馬の回収率を、買った馬の人気の分布で重みづけ（市場どおりに買ったときの基準線）。
    買った馬は基準線に入れない（v3._pop_base と同じ決まり）。"""
    pop = T["pop"].fillna(99).clip(0, POP_CAP).astype(int).astype(str)
    key = T["period"] + "|" + T["surf"] + "|" + T["group"] + "|" + pop
    g = pd.DataFrame({"k": key, "r": T[col], "p": T["pay"]})[~m.to_numpy()].groupby("k")[["r", "p"]].sum()
    rate = g["r"] / g["p"].replace(0, np.nan)
    return float(key[m].map(rate).dropna().mean()) if m.any() else None


def edge_ci(w: float, q: float) -> dict:
    """勝ち数 w ÷ 市場の見込みの勝ち数 q と、w をポアソンとみた 95% 区間（正確法）。"""
    from scipy.stats import chi2
    lo = chi2.ppf(0.025, 2 * w) / 2 if w > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * (w + 1)) / 2
    return {"w": float(w), "q": float(q), "ratio": float(w / q) if q else None, "lo": float(lo / q) if q else None, "hi": float(hi / q) if q else None}


def market_cols_check() -> dict:
    """表示の条件が使う切り口の型の列に、市場（オッズ・人気・払戻）の列が無いかを数える（catalog.FORBIDDEN_PATTERN）。"""
    import re
    from src.research.course.course_rule_bets import STATS
    from src.research.course.v3 import OUT3, PREREG3, ordered_conds
    from src.research.graded import catalog as CAT
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    kept = json.loads((OUT3 / P["cells_file"]).read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    T = {t.id: t for t in CAT.TYPES}
    pat = re.compile(CAT.FORBIDDEN_PATTERN)
    types, n = set(), 0
    for u, r in stats["rules"].items():
        buy, _ = ordered_conds(kept[u])
        for b in r["buy"]:
            n += 1
            types |= set(buy[b["i"]]["types"])
    bad = sorted(t for t in types if t not in T or any(pat.search(c) for c in T[t].cols) or t in ("M01", "M02"))
    return {"rules": n, "types": len(types), "market_types": bad, "forbidden_pattern": CAT.FORBIDDEN_PATTERN}


def trim_roi(g: pd.DataFrame, col: str, frac: float) -> float | None:
    """払戻の大きい順に上位 frac の当たりを除いた回収率（分母はそのまま）。"""
    r = np.sort(g[col].to_numpy())[::-1]
    k = int(np.ceil(frac * (r > 0).sum()))
    p = g["pay"].sum()
    return float(r[k:].sum() / p) if p else None


def cmd_report() -> None:
    T = pd.read_parquet(OUTW / "runners.parquet")
    R = pd.read_parquet(OUTW / "rules.parquet")
    T = T[T["pay"] > 0].reset_index(drop=True)
    out = {"made_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "periods": ["train", "2025", "2026"]}

    # H1 控除率と較正
    base, buys = {}, {}
    for p, g in T.groupby("period"):
        base[p] = {"runners": int(len(g)), "roi_t": _roi(g, "rt"), "roi_f": _roi(g, "rf"),
                   "fav_win": float(g.loc[g["pop"] == 1, "win"].mean()), "overround": float((1 / g["odds"]).groupby(g["race_id"]).sum().mean())}
        base[p]["payback"] = 1 / base[p]["overround"]
        for kind, b, col in (("tansho", "buy_t", "rt"), ("fukusho", "buy_f", "rf")):
            gb = g[g[b]]
            lo, hi = _boot(gb, col)
            dlo, dhi = _boot(gb, col, by="date")
            buys.setdefault(kind, {})[p] = {
                "bets": int(len(gb)), "races": int(gb["race_id"].nunique()), "days": int(gb["date"].nunique()),
                "roi": _roi(gb, col), "roi_lo": lo, "roi_hi": hi, "roi_lo_day": dlo, "roi_hi_day": dhi,
                "win": float(gb["win"].mean()), "top3": float(gb["top3"].mean()), "q_mean": float(gb["q"].mean()),
                "edge_win": float(gb["win"].sum() / gb["q"].sum()),           # 勝ち数 ÷ 市場の勝率の和（買い目・馬ごとに重複なし）
                "edge_ci": edge_ci(gb["win"].sum(), gb["q"].sum()),
                "same_pop_roi": same_pop(T, (T["period"] == p) & T[b], col),
                "odds_median": float(gb["odds"].median()), "pop_mean": float(gb["pop"].mean()),
                "pop_dist": gb["pop"].clip(upper=POP_CAP).value_counts(normalize=True).sort_index().round(4).to_dict(),
                "trim1": trim_roi(gb, col, 0.01), "trim5": trim_roi(gb, col, 0.05),
                "top1pct_share": None,
                "multi_rule": float((gb["nr_t" if kind == "tansho" else "nr_f"] >= 2).mean()),
                "rules_per_bet": float(gb["nr_t" if kind == "tansho" else "nr_f"].mean()),
                "bets_per_race": float(len(gb) / max(gb["race_id"].nunique(), 1)),
                "horses": int(gb["horse_id"].nunique())}
            r = np.sort(gb[col].to_numpy())[::-1]
            buys[kind][p]["top1pct_share"] = float(r[:max(int(np.ceil(0.01 * (r > 0).sum())), 1)].sum() / r.sum()) if r.sum() else None
    out["base"], out["buys"] = base, buys

    # 較正（買った単勝の馬・オッズ帯ごと）: 市場の勝率と実際の勝率
    cal = {}
    for p, g in T[T["buy_t"]].groupby("period"):
        bands = pd.cut(g["odds"], ODDS_BANDS, right=False)
        rows = []
        for bnd, h in g.groupby(bands, observed=True):
            gp = T[(T["period"] == p) & (pd.cut(T["odds"], ODDS_BANDS, right=False) == bnd)]
            rows.append({"band": f"{bnd.left:g}〜{bnd.right:g}" if bnd.right < 1e8 else f"{bnd.left:g}〜", "n": int(len(h)), "q": float(h["q"].mean()),
                         "win": float(h["win"].mean()), "roi": _roi(h, "rt"), "pop_win": float(gp["win"].mean()), "pop_q": float(gp["q"].mean()),
                         "pop_roi": _roi(gp, "rt")})
        cal[p] = rows
    out["calibration"] = cal

    # 年ごと（訓練の中も）
    yrs = []
    for y, g in T.groupby("year"):
        row = {"year": int(y), "pop_roi_t": _roi(g, "rt"), "pop_roi_f": _roi(g, "rf")}
        for kind, b, col in (("t", "buy_t", "rt"), ("f", "buy_f", "rf")):
            gb = g[g[b]]
            row |= {f"bets_{kind}": int(len(gb)), f"roi_{kind}": _roi(gb, col), f"win_{kind}": float(gb["win"].mean()) if len(gb) else None,
                    f"edge_{kind}": float(gb["win"].sum() / gb["q"].sum()) if len(gb) else None}
        yrs.append(row)
    out["by_year"] = yrs

    # H2 条件ごと: 訓練の値と後の期間の値（縮小の当てはめ）
    tr = R[R["period"] == "train"].set_index(["unit", "i"])
    post = R[R["period"].isin(["2025", "2026"])].groupby(["unit", "i"])[["n", "w", "k", "np", "tan", "fuku", "q", "fh"]].sum()
    J = tr.join(post, rsuffix="_post", how="left").fillna({c + "_post": 0 for c in ["n", "w", "k", "np", "tan", "fuku", "q", "fh"]})
    J["roi_t_in"] = J["tan"] / J["np"].replace(0, np.nan)
    J["edge_in"] = J["w"] / J["q"].replace(0, np.nan)
    J["edge_a"] = J["w_a"] / J["q_a"].replace(0, np.nan)
    J["edge_b"] = J["w_b"] / J["q_b"].replace(0, np.nan)
    out["rules"] = {"n": int(len(J)), "n_median": float(J["n"].median()), "n_q": J["n"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
                    "buy_t": int((J["roi_t_train"] >= 1).sum()), "buy_f": int((J["roi_f_train"] >= 1).sum()),
                    "in_win": float(J["w"].sum() / J["n"].sum()), "in_edge": float(J["w"].sum() / J["q"].sum()),
                    "post_win": float(J["w_post"].sum() / max(J["n_post"].sum(), 1)), "post_edge": float(J["w_post"].sum() / J["q_post"].sum()),
                    "post_n": int(J["n_post"].sum())}
    # 縮小: 訓練の「市場より多く勝った分」（勝ち数 ÷ 市場の期待勝ち数 − 1）のうち、後の期間に残った割合（保持率）。
    # 0 なら訓練の値は後の期間に何も伝えない（縮小は市場どおりへ全部）。1 なら縮めなくてよい。
    def keep_rate(w_in, q_in, w_post, q_post):
        e_in, e_post = w_in / q_in, w_post / q_post
        return {"edge_in": float(e_in), "edge_post": float(e_post), "keep": float((e_post - 1) / (e_in - 1)) if e_in != 1 else None}
    pw = J[J["q_post"] > 0]
    slope = np.polyfit(pw["edge_in"].to_numpy(), (pw["w_post"] / pw["q_post"]).to_numpy(), 1, w=np.sqrt(pw["q_post"].to_numpy()))[0]
    R26 = R[R["period"] == "2026"].set_index(["unit", "i"])
    e_post = edge_ci(J["w_post"].sum(), J["q_post"].sum())
    e_in = J["w"].sum() / J["q"].sum()
    out["shrink"] = {"all": keep_rate(J["w"].sum(), J["q"].sum(), J["w_post"].sum(), J["q_post"].sum()),
                     "keep_lo": float((e_post["lo"] - 1) / (e_in - 1)), "keep_hi": float((e_post["hi"] - 1) / (e_in - 1)), "edge_post_ci": e_post,
                     "edge_2026_ci": edge_ci(R26["w"].sum(), R26["q"].sum()),
                     "slope_post_on_train_edge": float(slope), "rules_with_post": int(len(pw)),
                     "pred_post_wins_noshrink": float((J["edge_in"] * J["q_post"]).sum()), "market_post_wins": float(J["q_post"].sum()),
                     "obs_post_wins": float(J["w_post"].sum())}
    # 訓練の回収率の帯ごとの後の期間の回収率（訓練の値に予測力があるか）
    J["bin"] = pd.qcut(J["roi_t_in"].rank(method="first"), 5, labels=False)
    out["rules_by_train_roi"] = [{"q": int(b), "rules": int(len(g)), "train_roi_t": float(g["tan"].sum() / g["np"].sum()),
                                  "train_n": int(g["n"].sum()), "post_n": int(g["n_post"].sum()),
                                  "post_roi_t": float(g["tan_post"].sum() / g["np_post"].sum()) if g["np_post"].sum() else None,
                                  "post_edge": float(g["w_post"].sum() / g["q_post"].sum()) if g["q_post"].sum() else None}
                                 for b, g in J.groupby("bin")]
    # 訓練の中の時代: 同じ条件の 2016〜21 と 2022〜24
    out["rules_split"] = {"a": {"n": int(J["n_a"].sum()), "roi_t": float(J["tan_a"].sum() / J["np_a"].sum()), "edge": float(J["w_a"].sum() / J["q_a"].sum())},
                          "b": {"n": int(J["n_b"].sum()), "roi_t": float(J["tan_b"].sum() / J["np_b"].sum()), "edge": float(J["w_b"].sum() / J["q_b"].sum())},
                          "corr_edge_ab": float(np.corrcoef(*J[["edge_a", "edge_b"]].dropna().to_numpy().T)[0, 1]) if J[["edge_a", "edge_b"]].dropna().shape[0] > 2 else None}

    # 並べ替えの帰無
    perms = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUTW.glob("perm_*_*.json"))]
    out["perm"] = perms
    out["perm_seeds"] = {str(fe): sorted(p["seed"] for p in perms if p["fit_end"] == fe) for fe in sorted({p["fit_end"] for p in perms})}
    out["market_cols"] = market_cols_check()
    OUT_REPO.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print("書いた", OUT_REPO, flush=True)


# ---------------------------------------------------------------- 重賞（好走データ予想の主に見る版）
GRADED_OUT_REPO = CR.REPO / "docs/operations/retrain_trials/megu_why_graded.json"


def _graded_period(h: pd.DataFrame, years) -> dict:
    """主に見る版の買いの馬（buy）と、同じレースの全馬を、確定オッズの市場の勝率・人気で比べる。"""
    parts = [pd.read_parquet(CR.REPO / "data/page_reference/tables" / str(y) / "race_result_flat.parquet",
                             columns=["race_id", "horse_number", "odds", "popularity"]) for y in years]
    o = pd.concat(parts, ignore_index=True)
    o["race_id"] = o["race_id"].astype(str)
    o["hn"] = pd.to_numeric(o["horse_number"], errors="coerce")
    h = h.copy()
    h["race_id"] = h["race_id"].astype(str)
    h["hn"] = pd.to_numeric(h["hn"], errors="coerce")
    h = h.merge(o.drop_duplicates(["race_id", "hn"])[["race_id", "hn", "odds", "popularity"]], on=["race_id", "hn"], how="left")
    h = h[h["pay_ok"].fillna(False).astype(bool)].copy()
    inv = 1.0 / pd.to_numeric(h["odds"], errors="coerce").where(lambda x: x > 0)
    h["q"] = (inv / inv.groupby(h["race_id"]).transform("sum")).fillna(0.0)
    h["win"] = (pd.to_numeric(h["fin"], errors="coerce") == 1).astype(float)
    h["top3"] = (pd.to_numeric(h["fin"], errors="coerce") <= 3).astype(float)
    h["pop"] = pd.to_numeric(h["popularity"], errors="coerce").fillna(99).clip(upper=POP_CAP)
    b = h[h["buy"].astype(bool)]
    nb = h[~h["buy"].astype(bool)]                 # 基準線は買わなかった馬だけで作る
    rate_t = nb.groupby("pop")["ret_tan"].mean()
    rate_f = nb.groupby("pop")["ret_fuku"].mean()
    q3 = nb.groupby("pop")["top3"].mean()
    out = {"years": list(years), "races": int(h["race_id"].nunique()), "runners": int(len(h)),
           "all_roi_t": float(h["ret_tan"].mean()), "all_roi_f": float(h["ret_fuku"].mean()),
           "fav_win": float(h.loc[h["pop"] == 1, "win"].mean()), "overround": float(inv.groupby(h["race_id"]).sum().mean()),
           "bets": int(len(b)), "bet_races": int(b["race_id"].nunique()), "win": float(b["win"].mean()), "top3": float(b["top3"].mean()),
           "roi_t": float(b["ret_tan"].mean()), "roi_f": float(b["ret_fuku"].mean()), "q_sum": float(b["q"].sum()), "wins": int(b["win"].sum()),
           "edge_win": float(b["win"].sum() / b["q"].sum()) if b["q"].sum() else None, "pop_mean": float(b["pop"].mean()),
           "same_pop_roi_t": float(b["pop"].map(rate_t).mean()), "same_pop_roi_f": float(b["pop"].map(rate_f).mean()),
           "same_pop_top3": float(b["pop"].map(q3).mean()), "edge_ci": edge_ci(b["win"].sum(), b["q"].sum()),
           "trim5_t": trim_roi(b.assign(pay=1.0), "ret_tan", 0.05), "pop_dist": {str(int(k)): int(v) for k, v in b["pop"].value_counts().sort_index().items()}}
    lo, hi = _boot(b.assign(pay=1.0), "ret_tan")
    flo, fhi = _boot(b.assign(pay=1.0), "ret_fuku")
    out |= {"roi_t_lo": lo, "roi_t_hi": hi, "roi_f_lo": flo, "roi_f_hi": fhi}
    return out


def cmd_graded() -> None:
    """重賞の主に見る版（graded_out_v2）の、探した期間の中（2022〜24 の当てはめ）と 2025・26 年の買いの馬を、市場と比べる。
    2025・2026 年の着順・払戻（凍結の評価の記録）とオッズを読むので、台帳に eval_only を申告する（family graded）。"""
    from pathlib import Path
    from src.ops import data_ledger as DL
    pb = json.loads((CR.REPO / "src/research/graded/prereg_buy.json").read_text(encoding="utf-8"))
    base = Path(os.environ.get("GRADED_V2_OUT") or "/home/jovyan/work/graded_out_v2")
    OUTW.mkdir(parents=True, exist_ok=True)
    res = {"made_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "primary": pb["primary"], "cells_sha": pb["cells"]["sha256"]}
    v = next(x for x in pb["versions"] if x.get("primary"))
    dr = v["dev"]["real"]
    b = dr["buy"]
    res["primary_cells"] = v.get("cells_2016_2024")
    res["null_kind"] = "レースの中で着順・払戻を入れ替える（graded/buy.py の permute_within_race。人気と結果の関係も消える）"
    res["prereg_dev"] = {"source": "prereg_buy.json の versions[primary].dev.real（2016〜21 年で探し 2022〜24 年に当てる）",
                         "n": b["n"], "races": dr["races_with_buy"], "win": b["rec"][0] / b["n"], "top3": b["top3_rate"], "roi_t": b["roi_t"], "roi_f": b["roi_f"],
                         "all_roi_t": dr["all_same_races"]["roi_t"], "all_roi_f": dr["all_same_races"]["roi_f"]}
    perm = pd.read_parquet(base / "buy_cells.parquet", columns=["perm"])["perm"].value_counts()
    res["cells"] = {"real": int(perm.get(-1, 0)), "perm": {str(int(k)): int(n) for k, n in perm.items() if k >= 0}}
    res["versions"] = int(pb["n_versions"])
    res["in_sample_2022_2024"] = _graded_period(pd.read_parquet(base / "smoke_buy/buy_primary_horses.parquet"), (2022, 2023, 2024))
    with DL.guard(tool="src/research/course/megu_why.py:cmd_graded", family="graded", version=pb["cells"]["sha256"][:12],
                  period=("2025-01-01", "2026-11-20"), purpose="eval_only",
                  reason="重賞の好走データ予想（主に見る版）が回収率 100% を超えない理由の診断（市場の勝率・同じ人気の基準線との比較）。規則は選び直さない",
                  frozen_paths=[CR.REPO / "src/research/graded/prereg_buy.json"]) as tok:
        h = pd.read_parquet(base / "buy_primary_horses.parquet")
        tok.check_race_ids(h["race_id"])
        h.to_parquet(OUTW / "graded_eval_horses.parquet", index=False)      # 表を直すときに 2025・26 年を読み直さないための写し
        res["eval_2025_2026"] = _graded_period(h, (2025, 2026))
        for y in (2025, 2026):
            res[f"eval_{y}"] = _graded_period(h[h["race_id"].astype(str).str[:4] == str(y)], (y,))
        tok.finish(result_path=GRADED_OUT_REPO)
    GRADED_OUT_REPO.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: ({x: v[x] for x in ("bets", "roi_t", "roi_f", "edge_win", "same_pop_roi_t", "same_pop_roi_f", "all_roi_t")} if isinstance(v, dict) and "bets" in v else v)
                      for k, v in res.items()}, ensure_ascii=False, default=float), flush=True)


def main() -> None:
    if sys.argv[1:2] == ["_collect_inner"]:
        return _collect_inner()
    if sys.argv[1:2] == ["_perm_inner"]:
        return _perm_inner(int(sys.argv[2]), int(sys.argv[3]))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    s = sub.add_parser("perm")
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--fit-end", type=int, default=2024, help="条件を探す期間の終わり（2021 = 2016〜21 で探し 2022〜24 で確かめる）")
    sub.add_parser("report")
    sub.add_parser("graded")
    a = ap.parse_args()
    if a.cmd == "collect":
        cmd_collect()
    elif a.cmd == "perm":
        cmd_perm([int(x) for x in a.seeds.split(",")], a.fit_end)
    elif a.cmd == "graded":
        cmd_graded()
    else:
        cmd_report()


if __name__ == "__main__":
    main()

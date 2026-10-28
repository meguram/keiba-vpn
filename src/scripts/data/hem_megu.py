"""想定めぐ指数の階層縮約モデル（HEM）。1 走ごとのノイズ（標準偏差約 10 点）を、馬 → 馬×面 → 馬×距離 → 騎手 の階層で部分プーリングして縮約する。

python -m src.scripts.data.hem_megu tune      # 2021 年の次走で超パラメータを選ぶ（2020〜21 年の走だけ）→ OUT/hem_params.json
python -m src.scripts.data.hem_megu eval      # 単独の精度（2022〜2024 年、現行・単純加重平均との比較、走数別・年別）→ OUT/hem_eval.json
python -m src.scripts.data.hem_megu build     # 親に足す群 HEM.parquet / CUR.parquet（行キー race_id, horse_id）
python -m src.scripts.data.hem_megu check     # 先読みの検査（未来の走を削っても同じ値になるか）

構造（正規-正規の経験ベイズ）:
  r_i = y_i − m(セル_i)                  セル = クラス水準×馬齢×面。m は 2020〜2021 年の平均（固定）
  馬の水準  d̂ = B · d̄,  B = 1/(1+k1·q)  d̄ = 直近 N 走の減衰加重平均、q = Σw²/(Σw)²（走数が少ないほど B が小さく 0 へ縮約）
  馬×面     us = Σ w·1[同じ面]·(r−d̂) / (Σ w·1[同じ面] + ks)
  馬×距離   ud = Σ w·κ(距離差)·(r−d̂) / (Σ w·κ + kd)
  騎手      J  = Σ e / (n + kj)           e = その騎乗の実指数 −（セル平均 + その時点の d̂）。今回の日付より前の騎乗だけ
  予測      ŷ = a + b0·m(今回のセル) + b1·d̂ + b2·us + b3·ud + b4·J   （係数は 2021 年の次走で当てはめて固定）
市場オッズ・人気は使わない。2026 年は使わない。特徴は『日付が今回より前の走』だけから作る（check で検査）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TABLES = Path(os.environ.get("HEM_TABLES", ROOT / "data/page_reference/tables"))
OUT = Path(os.environ.get("HEM_OUT", ROOT / "data/local/modeling/hem"))
KEY = ["race_id", "horse_id"]
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
FIT_YEARS = [2020, 2021]
TUNE_YEAR = 2021
EVAL_YEARS = [2022, 2023, 2024]
GRADE_LEVEL = {"新馬": 0, "未勝利": 1, "1勝": 2, "2勝": 3, "3勝": 4, "OP": 5, "L": 6, "G3": 7, "G2": 8, "G1": 9}
VALID = {"ok", "clipped", "restored_out_of_range", "restored_clipped"}
LOW_W = 0.5
DEFAULT = {"rho": 0.65, "N": 5, "k1": 1.0, "ks": 1.0, "D": 400.0, "kd": 1.0, "kj": 100.0}
GRID = {"rho": [0.5, 0.65, 0.8, 0.9, 0.95], "N": [5, 8, 12], "k1": [0.25, 0.5, 1.0, 2.0, 4.0], "ks": [0.5, 1.0, 2.0, 4.0],
        "D": [200.0, 400.0, 800.0, 1600.0], "kd": [0.5, 1.0, 2.0, 4.0], "kj": [30.0, 100.0, 300.0, 1000.0]}   # 縮約の強さは意味のある範囲に限る（大きすぎると係数で打ち消し合う）


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _digits(s) -> int:
    t = re.sub(r"\D", "", str(s or ""))[:8]
    return int(t) if len(t) == 8 else -1


def encode_ctx(df: pd.DataFrame) -> pd.DataFrame:
    """date, surface, distance, grade, sex_age から、数値の文脈列（dnum, surf, dist, lvl, age）を作る。"""
    out = pd.DataFrame(index=df.index)
    out["dnum"] = [_digits(d) for d in df["date"]]
    sf = df["surface"].astype(str)
    out["surf"] = np.where(sf.str.startswith("芝"), 1, np.where(sf.str.startswith("ダ"), 0, -1)).astype("int8")
    out["dist"] = pd.to_numeric(df["distance"], errors="coerce").astype("float64")
    out["lvl"] = df["grade"].map(GRADE_LEVEL).fillna(10).astype("int8")
    age = df["sex_age"].astype(str).str.extract(r"(\d+)")[0]
    out["age"] = pd.to_numeric(age, errors="coerce").fillna(0).clip(0, 7).astype("int8")
    out["jw"] = pd.to_numeric(df["jockey_weight"], errors="coerce").astype("float64") if "jockey_weight" in df.columns else np.nan
    return out


def load_history(years: list[int] = YEARS, tables: Path = TABLES) -> pd.DataFrame:
    """指数のある走（v3a 絶対指数、有効な状態のみ）と文脈。列: race_id, horse_id, jockey_id, y, low, year + encode_ctx の列。"""
    parts = []
    for y in years:
        p = tables / str(y) / "megu_index_flat.parquet"
        rr = tables / str(y) / "race_result_flat.parquet"
        if not p.exists() or not rr.exists():
            continue
        m = pd.read_parquet(p, columns=["race_id", "horse_id", "megu_index_v3a_abs_robust", "megu_v3a_status", "megu_v3a_low_confidence"])
        m = m[m["megu_v3a_status"].isin(VALID) & m["megu_index_v3a_abs_robust"].notna()]
        c = pd.read_parquet(rr, columns=["race_id", "horse_id", "date", "surface", "distance", "grade", "jockey_id", "sex_age", "jockey_weight"]).drop_duplicates(KEY)
        parts.append(m.merge(c, on=KEY, how="left"))
    d = pd.concat(parts, ignore_index=True)
    ctx = encode_ctx(d)
    h = pd.DataFrame({"race_id": d["race_id"].astype(str), "horse_id": d["horse_id"].astype(str), "jockey_id": d["jockey_id"].astype(str),
                      "y": d["megu_index_v3a_abs_robust"].astype(float), "low": d["megu_v3a_low_confidence"].fillna(False).astype(bool)})
    h = pd.concat([h, ctx], axis=1)
    h = h[h["dnum"] > 0].reset_index(drop=True)
    h["year"] = h["dnum"] // 10000
    return h


def cell_id(lvl, age, surf) -> np.ndarray:
    return (np.asarray(lvl).astype(np.int64) * 100 + np.asarray(age).astype(np.int64) * 10 + (np.asarray(surf).astype(np.int64) + 1))


def fit_cells(h: pd.DataFrame, min_n: int = 30) -> dict:
    """セルの平均（2020〜2021 年の走）。少ないセルは (クラス, 面) → 全体の順に代える。"""
    f = h[h["year"].isin(FIT_YEARS)]
    glob = float(f["y"].mean())
    fine = f.groupby(cell_id(f["lvl"], f["age"], f["surf"]))["y"].agg(["mean", "count"])
    mid = f.groupby(f["lvl"].astype(int) * 10 + (f["surf"].astype(int) + 1))["y"].agg(["mean", "count"])
    return {"glob": glob, "fine": {int(k): float(v) for k, v in fine[fine["count"] >= min_n]["mean"].items()},
            "mid": {int(k): float(v) for k, v in mid[mid["count"] >= min_n]["mean"].items()}}


def cell_mean(cells: dict, lvl, age, surf) -> np.ndarray:
    ids = cell_id(lvl, age, surf)
    mids = np.asarray(lvl).astype(np.int64) * 10 + (np.asarray(surf).astype(np.int64) + 1)
    out = np.empty(len(ids))
    for i, (a, b) in enumerate(zip(ids, mids)):
        out[i] = cells["fine"].get(int(a), cells["mid"].get(int(b), cells["glob"]))
    return out


class Hist:
    """履歴（指数のある走）を、馬×日付で並べて、『日付が今回より前の直近 N 走』を行列で引けるようにする。"""

    def __init__(self, h: pd.DataFrame, cells: dict):
        h = h.sort_values(["horse_id", "dnum"], kind="stable").reset_index(drop=True)
        self.h = h
        codes, uniq = pd.factorize(h["horse_id"])
        self.uniq = {u: i for i, u in enumerate(uniq)}
        self.code = codes.astype(np.int64)
        self.key = self.code * 100_000_000 + h["dnum"].to_numpy().astype(np.int64)
        self.y = h["y"].to_numpy(dtype=float)
        self.m = cell_mean(cells, h["lvl"], h["age"], h["surf"])
        self.r = self.y - self.m
        self.low = h["low"].to_numpy()
        self.surf = h["surf"].to_numpy()
        self.dist = h["dist"].to_numpy(dtype=float)
        self.dnum = h["dnum"].to_numpy().astype(np.int64)
        self.lvlf = np.where(h["lvl"].to_numpy() < 10, h["lvl"].to_numpy(), np.nan).astype(float)
        self.jw = h["jw"].to_numpy(dtype=float)
        self.cells = cells

    def window(self, horse_ids: pd.Series, dnum: np.ndarray, N: int):
        """各対象について、前の走の行番号 (M,N)（新しい順）と有効マスク。"""
        hc = np.array([self.uniq.get(x, -1) for x in horse_ids], dtype=np.int64)
        ok = hc >= 0
        tkey = np.where(ok, hc * 100_000_000 + dnum, -1)
        pos = np.searchsorted(self.key, tkey, side="left")
        start = np.searchsorted(self.key, np.where(ok, hc * 100_000_000, -1), side="left")
        idx = pos[:, None] - 1 - np.arange(N)[None, :]
        valid = (idx >= start[:, None]) & ok[:, None]
        return np.where(valid, idx, 0), valid


def horse_terms(hist: Hist, T: pd.DataFrame, p: dict) -> pd.DataFrame:
    """馬・馬×面・馬×距離の項と、補助量。T の列: horse_id, dnum, surf, dist（+ lvl, age）。"""
    N = int(p["N"])
    idx, valid = hist.window(T["horse_id"], T["dnum"].to_numpy().astype(np.int64), N)
    dec = float(p["rho"]) ** np.arange(N)[None, :]
    w = dec * np.where(hist.low[idx], LOW_W, 1.0) * valid
    R = hist.r[idx]
    ws = w.sum(1)
    safe = np.where(ws > 0, ws, 1.0)
    dbar = (w * R).sum(1) / safe
    q = (w ** 2).sum(1) / safe ** 2
    B = 1.0 / (1.0 + float(p["k1"]) * q)
    dhat = np.where(ws > 0, B * dbar, 0.0)
    res = R - dhat[:, None]
    tsurf = T["surf"].to_numpy()[:, None]
    same = (hist.surf[idx] == tsurf) & (tsurf >= 0) & valid
    ws_same = (w * same).sum(1)
    us = (w * same * res).sum(1) / (ws_same + float(p["ks"]))
    tdist = T["dist"].to_numpy(dtype=float)[:, None]
    dd = np.abs(hist.dist[idx] - tdist)
    kap = np.where(np.isnan(dd), 0.0, np.exp(-np.nan_to_num(dd) / float(p["D"])))
    wk = w * kap
    ud = (wk * res).sum(1) / (wk.sum(1) + float(p["kd"]))
    n = valid.sum(1)
    last = np.where(n > 0, hist.dnum[idx[:, 0]], 0)
    mbar = (w * hist.m[idx]).sum(1) / safe
    wm5 = np.where(ws > 0, (w * hist.y[idx]).sum(1) / safe, np.nan)
    chg = _change_terms(hist, T, idx, valid, w, safe, n, last)
    return pd.DataFrame({"dhat": dhat, "us": us, "ud": ud, "n": n, "q": np.where(ws > 0, q, np.nan), "last": last, "mbar": np.where(ws > 0, mbar, np.nan), "wmean": wm5, **chg},
                        index=T.index)


CHG_COLS = ["cls_up", "cls_down", "to_dirt", "to_turf", "dist_up", "dist_down", "rest_long", "rest_short", "age_c", "wdiff"]


def _wmean_nan(vals: np.ndarray, w: np.ndarray) -> np.ndarray:
    ok = np.isfinite(vals)
    ww = w * ok
    s = ww.sum(1)
    return np.where(s > 0, (ww * np.nan_to_num(vals)).sum(1) / np.where(s > 0, s, 1.0), np.nan)


def _change_terms(hist: Hist, T: pd.DataFrame, idx, valid, w, safe, n, last) -> dict:
    """現行式（expected_megu.condition_features）と同じ意味の『条件変化』を、行列で作る。履歴が無い行は 0。"""
    has = (n > 0)
    z = lambda a: np.where(has, np.nan_to_num(a), 0.0)  # noqa: E731
    tsurf = T["surf"].to_numpy()
    share_same = np.where(has, (w * (hist.surf[idx] == tsurf[:, None])).sum(1) / safe, 0.0)
    to_dirt = np.where(tsurf == 0, 1.0 - share_same, 0.0)
    to_turf = np.where(tsurf == 1, 1.0 - share_same, 0.0)
    pd_ = _wmean_nan(hist.dist[idx], w)
    dd = (T["dist"].to_numpy(dtype=float) - pd_) / 100.0
    pl = _wmean_nan(hist.lvlf[idx], w)
    tl = np.where(T["lvl"].to_numpy() < 10, T["lvl"].to_numpy(), np.nan).astype(float)
    cd = tl - pl
    d1 = pd.to_datetime(pd.Series(T["dnum"].to_numpy()).astype(str), format="%Y%m%d", errors="coerce")
    d0 = pd.to_datetime(pd.Series(last).astype(str), format="%Y%m%d", errors="coerce")
    rest = (d1 - d0).dt.days.to_numpy(dtype=float)
    pw = _wmean_nan(hist.jw[idx], w)
    wd = T["jw"].to_numpy(dtype=float) - pw
    age = np.clip(np.where(T["age"].to_numpy() > 0, T["age"].to_numpy(), 4), 2, 8).astype(float)
    return {"cls_up": z(np.maximum(cd, 0)), "cls_down": z(np.maximum(-cd, 0)), "to_dirt": np.where(has, to_dirt, 0.0), "to_turf": np.where(has, to_turf, 0.0),
            "dist_up": z(np.maximum(dd, 0)), "dist_down": z(np.maximum(-dd, 0)), "rest_long": z(np.maximum(0.0, rest - 90) / 30.0),
            "rest_short": np.where(has & (rest < 14), 1.0, 0.0), "age_c": age - 4.0, "wdiff": z(wd)}


def jockey_table(hist: Hist, p: dict) -> pd.DataFrame:
    """騎手ごとの『日付までの累積 e と件数』。e = その騎乗の実指数 −（セル平均 + その時点の馬の d̂）。"""
    h = hist.h
    T = pd.DataFrame({"horse_id": h["horse_id"], "dnum": h["dnum"], "surf": h["surf"], "dist": h["dist"], "lvl": h["lvl"], "age": h["age"], "jw": h["jw"]})
    ht = horse_terms(hist, T, p)
    e = hist.y - (hist.m + ht["dhat"].to_numpy())
    g = pd.DataFrame({"jockey_id": h["jockey_id"].to_numpy(), "dnum": h["dnum"].to_numpy(), "e": e}).groupby(["jockey_id", "dnum"], as_index=False).agg(se=("e", "sum"), cnt=("e", "size"))
    g = g.sort_values(["jockey_id", "dnum"])
    g["cse"] = g.groupby("jockey_id")["se"].cumsum()
    g["ccnt"] = g.groupby("jockey_id")["cnt"].cumsum()
    return g[["jockey_id", "dnum", "cse", "ccnt"]].sort_values("dnum").reset_index(drop=True)


def jockey_effect(jt: pd.DataFrame, T: pd.DataFrame, kj: float) -> np.ndarray:
    q = pd.DataFrame({"jockey_id": T["jockey_id"].astype(str).to_numpy(), "dnum": T["dnum"].to_numpy(), "_i": np.arange(len(T))}).sort_values("dnum")
    m = pd.merge_asof(q, jt, on="dnum", by="jockey_id", allow_exact_matches=False)
    se = m["cse"].fillna(0.0).to_numpy()
    cnt = m["ccnt"].fillna(0.0).to_numpy()
    out = np.empty(len(T))
    out[m["_i"].to_numpy()] = se / (cnt + kj)
    return out


STACK_COLS = ["m_T", "dhat", "us", "ud", "jock"] + CHG_COLS


def features(hist: Hist, T: pd.DataFrame, p: dict, jt: pd.DataFrame | None = None) -> pd.DataFrame:
    """T: race_id, horse_id, jockey_id, dnum, surf, dist, lvl, age。返り値は行ごとの項（予測の前）。"""
    ht = horse_terms(hist, T, p)
    jt = jt if jt is not None else jockey_table(hist, p)
    ht["jock"] = jockey_effect(jt, T, float(p["kj"]))
    ht["m_T"] = cell_mean(hist.cells, T["lvl"], T["age"], T["surf"])
    return ht


def fit_stack(F: pd.DataFrame, y: np.ndarray, cols: list[str], lam: float = 1.0) -> np.ndarray:
    X = np.column_stack([np.ones(len(F))] + [F[c].to_numpy() for c in cols])
    A = X.T @ X + lam * np.eye(X.shape[1])
    A[0, 0] -= lam
    return np.linalg.solve(A, X.T @ y)


def predict_stack(F: pd.DataFrame, coef: np.ndarray, cols: list[str]) -> np.ndarray:
    X = np.column_stack([np.ones(len(F))] + [F[c].to_numpy() for c in cols])
    return X @ coef


def targets_of(h: pd.DataFrame) -> pd.DataFrame:
    return h[["race_id", "horse_id", "jockey_id", "dnum", "surf", "dist", "lvl", "age", "jw"]].copy()


def _mse_tune(hist: Hist, tune: pd.DataFrame, y: np.ndarray, p: dict, cols: list[str]) -> float:
    F = features(hist, tune, p)
    m = (F["n"] >= 1).to_numpy()
    coef = fit_stack(F[m], y[m], cols)
    return float(np.mean((predict_stack(F[m], coef, cols) - y[m]) ** 2))


def cmd_tune(a) -> None:
    h = load_history()
    cells = fit_cells(h)
    hist = Hist(h, cells)
    tune_rows = h[h["year"] == TUNE_YEAR]
    tune = targets_of(tune_rows)
    y = tune_rows["y"].to_numpy()
    p = dict(DEFAULT)
    best = _mse_tune(hist, tune, y, p, STACK_COLS)
    _log(f"初期 MSE {best:.3f} (RMSE {best ** 0.5:.3f}) 件数 {len(tune)}")
    for rnd in range(2):
        changed = False
        for k, vals in GRID.items():
            for v in vals:
                if v == p[k]:
                    continue
                q = {**p, k: v}
                mse = _mse_tune(hist, tune, y, q, STACK_COLS)
                if mse < best - 1e-6:
                    best, p, changed = mse, q, True
            _log(f"round {rnd} {k}={p[k]} MSE {best:.3f}")
        if not changed:
            break
    F = features(hist, tune, p)
    m = (F["n"] >= 1).to_numpy()
    coef = fit_stack(F[m], y[m], STACK_COLS)
    resid_sd = float(np.sqrt(np.mean((predict_stack(F[m], coef, STACK_COLS) - y[m]) ** 2)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hem_params.json").write_text(json.dumps({"params": p, "stack_cols": STACK_COLS, "coef": coef.tolist(), "tune_year": TUNE_YEAR, "tune_rmse": resid_sd,
                                                     "fit_years": FIT_YEARS, "cells": {"glob": cells["glob"], "n_fine": len(cells["fine"]), "n_mid": len(cells["mid"])}},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"完了 params={p} coef={np.round(coef, 3).tolist()} RMSE {resid_sd:.3f}")


def load_params() -> dict:
    return json.loads((OUT / "hem_params.json").read_text(encoding="utf-8"))


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(y)
    d = pred[ok] - y[ok]
    return {"n": int(ok.sum()), "rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.mean(np.abs(d))), "corr": float(np.corrcoef(pred[ok], y[ok])[0, 1])}


def _within_race_corr(race: np.ndarray, pred: np.ndarray, y: np.ndarray) -> float:
    d = pd.DataFrame({"r": race, "p": pred, "y": y}).dropna()
    d = d[d.groupby("r")["p"].transform("size") >= 5]
    d["p"] -= d.groupby("r")["p"].transform("mean")
    d["y"] -= d.groupby("r")["y"].transform("mean")
    return float(np.corrcoef(d["p"], d["y"])[0, 1])


def current_pred(h_rows: pd.DataFrame, tables: Path = TABLES, date_col: str = "dnum") -> pd.DataFrame:
    """現行の想定めぐ指数（src/pipeline/inference/expected_megu.py の expected_from_runs。基準=加重平均、value=補正後）を行ごとに返す。"""
    from src.pipeline.inference import expected_megu as em

    model = em.load_model()
    by_horse: dict[str, list] = {}
    for y in YEARS:
        m, _ = em._load_year(y, tables)
        for hid, rs in m.items():
            by_horse.setdefault(hid, []).extend(rs)
    base_out, val_out = [], []
    for hid, d, su, di, ve, gr, jw, sa in zip(h_rows["horse_id"], h_rows[date_col], h_rows["surface_s"], h_rows["distance_s"], h_rows["venue_s"], h_rows["grade_s"], h_rows["jw_s"], h_rows["sex_age_s"]):
        rd = str(int(d))
        today = {"surface": em.surface_of(su), "distance": (float(di) if di == di and di is not None else None), "venue": (str(ve) if ve == ve and ve is not None else None),
                 "level": em.class_level(gr), "weight": (float(jw) if jw == jw and jw is not None else None), "age": em.age_of(sa)}
        o = em.expected_from_runs(by_horse.get(str(hid), []), rd, today, model)
        base_out.append(o["base_value"] if o else np.nan)
        val_out.append(o["value"] if o else np.nan)
    return pd.DataFrame({"cur_base": base_out, "cur_pred": val_out}, index=h_rows.index)


def _raw_ctx(h: pd.DataFrame, tables: Path = TABLES) -> pd.DataFrame:
    """現行式に渡す生の文脈（面・距離・競馬場・グレード・斤量・性齢）を race_result_flat から付ける。"""
    parts = []
    for y in YEARS:
        rr = tables / str(y) / "race_result_flat.parquet"
        if rr.exists():
            parts.append(pd.read_parquet(rr, columns=["race_id", "horse_id", "surface", "distance", "venue", "grade", "jockey_weight", "sex_age"]).drop_duplicates(KEY))
    c = pd.concat(parts, ignore_index=True)
    c["race_id"], c["horse_id"] = c["race_id"].astype(str), c["horse_id"].astype(str)
    c = c.rename(columns={"surface": "surface_s", "distance": "distance_s", "venue": "venue_s", "grade": "grade_s", "jockey_weight": "jw_s", "sex_age": "sex_age_s"})
    return h[KEY].merge(c, on=KEY, how="left").set_index(h.index)


def cmd_eval(a) -> None:
    h = load_history()
    cells = fit_cells(h)
    hist = Hist(h, cells)
    P = load_params()
    p, cols, coef = P["params"], P["stack_cols"], np.array(P["coef"])
    rows = h[h["year"].isin(EVAL_YEARS)].copy()
    T = targets_of(rows)
    F = features(hist, T, p)
    y = rows["y"].to_numpy()
    rows = rows.join(F[["n", "wmean"]])
    ctx = _raw_ctx(rows)
    cur = current_pred(pd.concat([rows[["horse_id", "dnum"]], ctx.drop(columns=KEY)], axis=1))
    variants = {
        "wmean5（単純な加重平均, 補正なし, 現行の基準値）": cur["cur_base"].to_numpy(),
        "現行（加重平均 + 条件補正）": cur["cur_pred"].to_numpy(),
    }
    tr_rows = h[h["year"] == TUNE_YEAR]
    Ftr = features(hist, targets_of(tr_rows), p)
    mtr = (Ftr["n"] >= 1).to_numpy()
    for name, cs in (("HEM 馬のみ", ["m_T", "dhat"]), ("HEM 馬+面+距離", ["m_T", "dhat", "us", "ud"]), ("HEM +騎手", ["m_T", "dhat", "us", "ud", "jock"]), ("HEM 全部（+条件変化）", STACK_COLS)):
        cf = fit_stack(Ftr[mtr], tr_rows["y"].to_numpy()[mtr], cs)
        variants[name] = predict_stack(F, cf, cs)
    ok = (F["n"] >= 1).to_numpy()
    buckets = {"1走": (F["n"] == 1).to_numpy(), "2走": (F["n"] == 2).to_numpy(), "3〜4走": F["n"].between(3, 4).to_numpy(), "5走以上": (F["n"] >= 5).to_numpy()}
    out = {"params": p, "overall": {}, "by_n": {}, "by_year": {}, "within_race_corr": {}}
    yrs = rows["year"].to_numpy()
    both = ok & np.isfinite(variants["現行（加重平均 + 条件補正）"])
    for name, pr in variants.items():
        out["overall"][name] = _metrics(np.where(both, pr, np.nan), y)
        out["within_race_corr"][name] = _within_race_corr(rows["race_id"].to_numpy(), np.where(both, pr, np.nan), y)
        out["by_n"][name] = {b: _metrics(np.where(both & m, pr, np.nan), y) for b, m in buckets.items()}
        out["by_year"][name] = {str(yv): _metrics(np.where(both & (yrs == yv), pr, np.nan), y) for yv in EVAL_YEARS}
    (OUT / "hem_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for name in variants:
        o = out["overall"][name]
        _log(f"{name}: RMSE {o['rmse']:.3f} MAE {o['mae']:.3f} corr {o['corr']:.4f} レース内相関 {out['within_race_corr'][name]:.4f}（n={o['n']}）")
    for b in buckets:
        _log(f"[{b}] " + " / ".join(f"{k[:6]} {out['by_n'][k][b]['rmse']:.2f}" for k in variants))
    for yv in EVAL_YEARS:
        _log(f"[{yv}] " + " / ".join(f"{k[:6]} {out['by_year'][k][str(yv)]['rmse']:.2f}" for k in variants))


def build_group(df: pd.DataFrame, hist: Hist, P: dict, tables: Path = TABLES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """親の行（race_id, horse_id, date, surface, distance, grade, jockey_id, sex_age, venue, jockey_weight）に対する HEM / CUR の群。"""
    p, cols, coef = P["params"], P["stack_cols"], np.array(P["coef"])
    ctx = encode_ctx(df)
    T = pd.concat([df[["race_id", "horse_id"]].astype(str), df[["jockey_id"]].astype(str), ctx], axis=1)
    F = features(hist, T, p)
    pred = predict_stack(F, coef, cols)
    out = pd.DataFrame({"race_id": T["race_id"].to_numpy(), "horse_id": T["horse_id"].to_numpy()})
    n = F["n"].to_numpy()
    has = n >= 1
    out["hem_pred"] = np.where(has, pred, np.nan)
    out["hem_dhat"] = np.where(has, F["dhat"].to_numpy(), np.nan)
    out["hem_us"] = np.where(has, F["us"].to_numpy(), np.nan)
    out["hem_ud"] = np.where(has, F["ud"].to_numpy(), np.nan)
    out["hem_jock"] = F["jock"].to_numpy()
    out["hem_cell"] = F["m_T"].to_numpy()
    out["hem_n"] = n.astype("float32")
    out["hem_sd"] = np.where(has, float(P["tune_rmse"]) * np.sqrt(np.nan_to_num(F["q"].to_numpy(), nan=1.0)), np.nan)
    out["hem_gap_days"] = np.where(has, (pd.to_datetime(T["dnum"].astype(str), format="%Y%m%d", errors="coerce") - pd.to_datetime(F["last"].astype(str), format="%Y%m%d", errors="coerce")).dt.days, np.nan)
    g = out.groupby("race_id")["hem_pred"]
    out["hem_pred_rel"] = out["hem_pred"] - g.transform("mean")
    out["hem_pred_rank"] = g.rank(ascending=False, method="min")
    out["hem_pred_z"] = out["hem_pred_rel"] / g.transform("std").replace(0, np.nan)
    cu = current_pred(pd.concat([T[["horse_id"]], T[["dnum"]], _raw_ctx(T).drop(columns=KEY)], axis=1))
    cur = pd.DataFrame({"race_id": out["race_id"], "horse_id": out["horse_id"], "cur_pred": cu["cur_pred"].to_numpy(), "cur_base": cu["cur_base"].to_numpy()})
    cur["cur_adj"] = cur["cur_pred"] - cur["cur_base"]
    gc = cur.groupby("race_id")["cur_pred"]
    cur["cur_pred_rel"] = cur["cur_pred"] - gc.transform("mean")
    cur["cur_pred_rank"] = gc.rank(ascending=False, method="min")
    cur["cur_pred_z"] = cur["cur_pred_rel"] / gc.transform("std").replace(0, np.nan)
    return out, cur


def cmd_build(a) -> None:
    from src.scripts.data import tp27_eval as t

    df, base, meta, folds, lp = t.setup()
    h = load_history()
    cells = fit_cells(h)
    hist = Hist(h, cells)
    P = load_params()
    cols = [c for c in ("race_id", "horse_id", "date", "surface", "distance", "grade", "jockey_id", "sex_age", "venue", "jockey_weight") if c in df.columns]
    d = df[cols].copy()
    for c in ("race_id", "horse_id"):
        d[c] = d[c].astype(str)
    hem, cur = build_group(d, hist, P)
    OUT.mkdir(parents=True, exist_ok=True)
    hem.to_parquet(OUT / "HEM.parquet", index=False)
    cur.to_parquet(OUT / "CUR.parquet", index=False)
    _log(f"保存 HEM {hem.shape} CUR {cur.shape}、欠損率 hem_pred {hem['hem_pred'].isna().mean():.3f} cur_pred {cur['cur_pred'].isna().mean():.3f}")


def cmd_check(a) -> None:
    """先読みの検査: 2023 年以降の走をすべて消した履歴で 2022 年の行を作り直しても、全項が一致するか。"""
    h = load_history()
    cells = fit_cells(h)
    P = load_params()
    p, cols, coef = P["params"], P["stack_cols"], np.array(P["coef"])
    rows = h[h["year"] == 2022]
    T = targets_of(rows)
    full = features(Hist(h, cells), T, p)
    cut = h[h["year"] <= 2022]
    part = features(Hist(cut, cells), T, p)
    worst = {c: float(np.nanmax(np.abs(full[c].to_numpy() - part[c].to_numpy()))) for c in ["dhat", "us", "ud", "jock", "m_T", "wmean"] + CHG_COLS}
    # さらに、今回のレース当日以降の走の指数を壊しても変わらないか（当日の自分の走を NaN に）
    h2 = h.copy()
    h2.loc[h2["year"] == 2022, "y"] = h2.loc[h2["year"] == 2022, "y"] + 1000.0       # 2022 年の指数を全部ずらす
    T21 = targets_of(h[h["year"] == 2021])
    a21 = features(Hist(h, cells), T21, p)
    b21 = features(Hist(h2, cells), T21, p)
    worst21 = {c: float(np.nanmax(np.abs(a21[c].to_numpy() - b21[c].to_numpy()))) for c in ["dhat", "us", "ud", "jock", "wmean"] + CHG_COLS}
    _log(f"2023年以降を消した差（最大）: {worst}")
    _log(f"2022年の指数を +1000 したときの 2021 年の項の差（最大。0 であるべき）: {worst21}")
    (OUT / "hem_check.json").write_text(json.dumps({"truncate_diff": worst, "future_shift_diff_2021": worst21}, indent=1), encoding="utf-8")


N_BUCKETS = (("n1", 1, 1), ("n2", 2, 2), ("n3_4", 3, 4), ("n5p", 5, 10_000))
RANGE_FIT_YEARS = [2022, 2023, 2024]
RANGE_CHECK_YEARS = [2025]
EXPORT_PATH = Path(os.environ.get("HEM_EXPORT") or ROOT / "config" / "expected_megu" / "hem.json")


def build_state(years: list[int] | None = None, tables: Path = TABLES) -> pd.DataFrame:
    """騎手の累積表（推論で使う）。固定の超パラメータ・セル平均で、全期間の履歴から作り直す。年を省くと表のある全年。"""
    ys = years or sorted(int(p.name) for p in tables.iterdir() if p.name.isdigit() and int(p.name) >= YEARS[0])
    h = load_history(ys, tables)
    P = load_params()
    hist = Hist(h, fit_cells(h))
    jt = jockey_table(hist, P["params"])
    jt["jockey_id"] = jt["jockey_id"].astype(str)
    return jt


def cmd_export(a) -> None:
    """推論用の係数 config/expected_megu/hem.json（学習済み係数・セル平均・レンジ較正・2025 年の精度）と、騎手の累積表 OUT/jockey_table.parquet を作る。"""
    h = load_history()
    cells = fit_cells(h)
    hist = Hist(h, cells)
    P = load_params()
    p, cols, coef = P["params"], P["stack_cols"], np.array(P["coef"])
    rows = h[h["year"].isin(RANGE_FIT_YEARS + RANGE_CHECK_YEARS)].copy()
    F = features(hist, targets_of(rows), p)
    rows = rows.join(F[["n"]])
    pred = predict_stack(F, coef, cols)
    y = rows["y"].to_numpy()
    n = rows["n"].to_numpy()
    yrs = rows["year"].to_numpy()
    ok = n >= 1
    resid_sd, coverage = {}, {}
    for name, lo, hi in N_BUCKETS:
        m = ok & (n >= lo) & (n <= hi) & np.isin(yrs, RANGE_FIT_YEARS)
        resid_sd[name] = round(float(np.sqrt(np.mean((pred[m] - y[m]) ** 2))), 2)
    for yv in RANGE_CHECK_YEARS:
        cov = {}
        for name, lo, hi in N_BUCKETS:
            m = ok & (n >= lo) & (n <= hi) & (yrs == yv)
            cov[name] = {"n": int(m.sum()), "within_1sd": round(float(np.mean(np.abs(pred[m] - y[m]) <= resid_sd[name])), 3)}
        coverage[str(yv)] = cov
    ev = {}
    cur = current_pred(pd.concat([rows[["horse_id", "dnum"]], _raw_ctx(rows).drop(columns=KEY)], axis=1))
    for yv in RANGE_FIT_YEARS + RANGE_CHECK_YEARS:
        m = ok & (yrs == yv) & np.isfinite(cur["cur_pred"].to_numpy())
        ev[str(yv)] = {"hem": _metrics(np.where(m, pred, np.nan), y), "current": _metrics(np.where(m, cur["cur_pred"].to_numpy(), np.nan), y)}
    doc = {"version": "hem-1", "trained_on": "2020-2021（セル平均）・2021 の次走で係数を当てはめ", "validated_on": "2022-2025",
           "params": p, "stack_cols": cols, "coef": [float(c) for c in coef],
           "cells": {"glob": cells["glob"], "fine": {str(k): v for k, v in cells["fine"].items()}, "mid": {str(k): v for k, v in cells["mid"].items()}},
           "resid_sd": resid_sd, "range_fit_years": RANGE_FIT_YEARS, "coverage": coverage,
           "eval": {"rmse_by_year": {k: {"hem": round(v["hem"]["rmse"], 3), "current": round(v["current"]["rmse"], 3), "corr_hem": round(v["hem"]["corr"], 4), "corr_current": round(v["current"]["corr"], 4), "n": v["hem"]["n"]} for k, v in ev.items()}}}
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    jt = build_state()
    OUT.mkdir(parents=True, exist_ok=True)
    jt.to_parquet(OUT / "jockey_table.parquet", index=False)
    _log(f"保存 {EXPORT_PATH}（レンジ {resid_sd}）、騎手の累積表 {jt.shape}（最新 {int(jt['dnum'].max())}）")
    _log(f"被覆率 {coverage}")
    _log(f"精度 {doc['eval']['rmse_by_year']}")


def cmd_state(a) -> None:
    """騎手の累積表だけを作り直す（新しいレースが増えたあとの定期更新用。係数は変えない）。"""
    jt = build_state()
    OUT.mkdir(parents=True, exist_ok=True)
    jt.to_parquet(OUT / "jockey_table.parquet", index=False)
    _log(f"騎手の累積表 {jt.shape}（最新 {int(jt['dnum'].max())}）")


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("tune", cmd_tune), ("eval", cmd_eval), ("build", cmd_build), ("check", cmd_check), ("export", cmd_export), ("state", cmd_state)):
        sp.add_parser(n).set_defaults(fn=f)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

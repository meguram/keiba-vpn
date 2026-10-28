"""tp-81: 展開利（想定ペース × 想定位置取り [× 追走難度]）が親モデルの好走確率にどれだけ寄与するか、直接測る。

  python -m src.scripts.data.tp81_tenkai oracle     # 実際のペース・位置で親の OOF 残差がどれだけ系統的に動くか（上限）
  python -m src.scripts.data.tp81_tenkai predicted  # 予測したペース分布 × 位置分布での展開利補正（後段のロジット補正）
  python -m src.scripts.data.tp81_tenkai features   # 特徴量版の群 TENKAI を書き出す（v6_eval の外部群）

親 = z:RB@e の OOF（2022〜2024。2025 年の OOF は無い）。補正 = 親ロジット + 展開利 δ（脚質×ペース×芝ダ距離帯×頭数帯の階層縮約ロジスティック、
親ロジットを offset）。δ の当てはめは年を抜く（LOYO）。通過順・ペースは race_result_flat 由来（tp80 の入力）で、市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp80_userpos as U
from src.scripts.data import tp80_eval as T80

ROOT = Path(__file__).resolve().parents[3]
V = ROOT / "data/local/modeling/training_pipeline_v4/v6"
OUT = Path(os.environ.get("TP81_OUT") or ROOT / "data/local/modeling/tp81")
KEY = ["race_id", "horse_id"]
YEARS = [2022, 2023, 2024]
POS_CUTS = [0.2, 0.45, 0.75]          # 初角位置 act_pos（0=先頭〜1=最後方）→ 逃/先/差/追
PACE_FEATS = T80.FEATS["ユーザー手順 + 既存系"]


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ───────────────────────── 入力
def load_all():
    D, career = T80.load()
    D["horse_id"] = D["horse_id"].astype(str)
    Tb = pd.read_parquet(T80.OUT / "pace_table.parquet")
    Tb["race_id"] = Tb["race_id"].astype(str)
    oof = pd.read_parquet(V / "eval_rev/oof_z:RB@e.parquet")
    oof["race_id"], oof["horse_id"] = oof["race_id"].astype(str), oof["horse_id"].astype(str)
    fl = pd.concat([pd.read_parquet(ROOT / f"data/page_reference/tables/{y}/race_result_flat.parquet", columns=KEY + ["finish_position"]) for y in YEARS])
    fl["race_id"], fl["horse_id"] = fl["race_id"].astype(str), fl["horse_id"].astype(str)
    cols = ["race_id", "horse_id", "year", "horse_number", "field_size", "surface", "distance", "act_pos", "act_first", "ts_c1_pred", "date"]
    X = oof.merge(D[cols].drop(columns=["year"]), on=KEY, how="left").merge(fl, on=KEY, how="left")
    X = X.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    X["win"] = (X["finish_position"] == 1).astype(float)
    X["top2"] = X["finish_position"].between(1, 2).astype(float)
    X["top3"] = X["finish_position"].between(1, 3).astype(float)
    X["ok"] = X["finish_position"] > 0
    return D, Tb, X


def strata(X: pd.DataFrame):
    d = X["distance"].to_numpy(dtype="float64")
    band3 = np.select([d <= 1200, d <= 1600], [0, 1], 2)
    st = (X["surface"].eq("芝").to_numpy().astype(int)) * 3 + band3
    fs = X["field_size"].to_numpy(dtype="float64")
    fsb = np.select([fs <= 12, fs <= 15], [0, 1], 2)
    return st, fsb


# ───────────────────────── 実際の位置・ペース、予測の位置・ペース
def actual_axes(X: pd.DataFrame, Tb: pd.DataFrame):
    qs = Tb[Tb["year"] <= 2021]["z_act"].quantile([1 / 3, 2 / 3]).to_numpy()
    pos = np.digitize(X["act_pos"].to_numpy(dtype="float64"), POS_CUTS)
    pos_ok = X["act_pos"].notna().to_numpy()
    za = X["race_id"].map(Tb.set_index("race_id")["z_act"]).to_numpy(dtype="float64")
    pace = np.digitize(za, qs)
    pace_ok = ~np.isnan(za)
    return pos, pos_ok, pace, pace_ok, qs


def onehot(idx, k, ok):
    M = np.zeros((len(idx), k))
    M[np.arange(len(idx))[ok], idx[ok]] = 1.0
    M[~ok] = 1.0 / k
    return M


def _pos_bucket(Z: pd.DataFrame):
    rk = U.order_in_race(Z["race_id"].to_numpy(), Z["ts_c1_pred"].to_numpy(dtype="float64"), Z["horse_number"].to_numpy())
    n = Z["field_size"].to_numpy(dtype="float64")
    ph = np.where(n > 1, (rk - 1) / np.maximum(n - 1, 1), 0.5)
    return ph, np.minimum((ph * 10).astype(int), 9)


def predicted_pos(X: pd.DataFrame, D: pd.DataFrame):
    """前段 ML(c1 予測) の想定順位 → 10 分位 → 年(<y)の実際の脚質区分の分布（2020 年〜、D 全体から）を soft に割り当てる。"""
    ph_d, b_d = _pos_bucket(D)
    apos_d = np.digitize(D["act_pos"].to_numpy(dtype="float64"), POS_CUTS)
    ok_d = D["act_pos"].notna().to_numpy()
    yr_d = D["year"].to_numpy()
    ph_x, b_x = _pos_bucket(X)
    P = np.zeros((len(X), 4))
    yr = X["year"].to_numpy()
    for y in YEARS:
        tr = ok_d & (yr_d < y)
        tab = np.zeros((10, 4)) + 1.0
        np.add.at(tab, (b_d[tr], apos_d[tr]), 1.0)
        tab /= tab.sum(1, keepdims=True)
        m = yr == y
        P[m] = tab[b_x[m]]
    return P, ph_x


def predicted_pace(X: pd.DataFrame, Tb: pd.DataFrame, qs):
    """想定ペース z（tp80 の ユーザー手順＋既存系、年<y で当てはめ）→ 正規分布で 3 区分の確率。"""
    from scipy.stats import norm

    P = np.zeros((len(X), 3))
    zhat = np.zeros(len(X))
    race_of = X["race_id"].to_numpy()
    for y in YEARS:
        te = Tb[Tb["year"] == y]
        p = T80._fit_predict(Tb, PACE_FEATS, y)
        # 残差 sd: 一つ前の年の out-of-sample（2022 は 2021 の in-sample）
        prev = Tb[Tb["year"] == y - 1]
        pp = T80._fit_predict(Tb, PACE_FEATS, y - 1)
        sd = float(np.nanstd(prev["z_act"].to_numpy() - pp))
        m = pd.Series(p, index=te["race_id"].to_numpy())
        zr = pd.Series(race_of).map(m).to_numpy(dtype="float64")
        sel = X["year"].to_numpy() == y
        zhat[sel] = zr[sel]
        c = np.stack([norm.cdf((qs[0] - zr) / sd), norm.cdf((qs[1] - zr) / sd) - norm.cdf((qs[0] - zr) / sd), 1 - norm.cdf((qs[1] - zr) / sd)], 1)
        P[sel] = c[sel]
    P = np.where(np.isnan(P), 1 / 3, P)
    return P, zhat


# ───────────────────────── 設計行列と当てはめ
def design(axes: list[np.ndarray], st: np.ndarray, fsb: np.ndarray):
    n = len(st)
    J = axes[0]
    for A in axes[1:]:
        J = (J[:, :, None] * A[:, None, :]).reshape(n, -1)
    kj = J.shape[1]
    blocks, pen = [], []
    for A in axes:
        blocks.append(A); pen.append(np.full(A.shape[1], 1.0))
    if len(axes) > 1:
        blocks.append(J); pen.append(np.full(kj, 3.0))
    k0 = axes[0].shape[1]
    B = np.zeros((n, 6 * k0)); C = np.zeros((n, 6 * kj)); Dm = np.zeros((n, 3 * k0))
    for s in range(6):
        m = st == s
        B[m, s * k0:(s + 1) * k0] = axes[0][m]
        C[m, s * kj:(s + 1) * kj] = J[m]
    for s in range(3):
        m = fsb == s
        Dm[m, s * k0:(s + 1) * k0] = axes[0][m]
    blocks += [B, C, Dm]; pen += [np.full(B.shape[1], 3.0), np.full(C.shape[1], 10.0), np.full(Dm.shape[1], 3.0)]
    return np.hstack(blocks), np.concatenate(pen)


def fit_offset(X, y, off, pen, lam, iters=8):
    b = np.zeros(X.shape[1])
    P = lam * pen
    for _ in range(iters):
        p = _sig(off + X @ b)
        W = p * (1 - p)
        g = X.T @ (y - p) - P * b
        H = (X * W[:, None]).T @ X + np.diag(P)
        b = b + np.linalg.solve(H, g)
    return b


def loyo_delta(Xd, pen, y, off, ok, year, lam):
    """年を抜いて δ = X b を作る（3 年: 各年は他の 2 年で当てはめ）。"""
    delta = np.zeros(len(y))
    for t in YEARS:
        tr = ok & (year != t); te = year == t
        b = fit_offset(Xd[tr], y[tr], off[tr], pen, lam)
        delta[te] = Xd[te] @ b
    return delta


def renorm_win(race, p):
    s = pd.Series(p).groupby(race).transform("sum").to_numpy()
    return p / s


def adjust(X: pd.DataFrame, d_win, d_top2, d_top3, relative: bool):
    race = X["race_id"].to_numpy()
    if relative:
        mu = lambda d: pd.Series(d).groupby(race).transform("mean").to_numpy()
        d_top2, d_top3 = d_top2 - mu(d_top2), d_top3 - mu(d_top3)
    pw = renorm_win(race, _sig(_logit(X["p_win"].to_numpy()) + d_win))
    return pd.DataFrame({"race_id": X["race_id"], "horse_id": X["horse_id"], "year": X["year"], "p_win": pw,
                          "p_top2": _sig(_logit(X["p_top2"].to_numpy()) + d_top2), "p_top3": _sig(_logit(X["p_top3"].to_numpy()) + d_top3)})


def evaluate(X: pd.DataFrame, new: pd.DataFrame, label: str) -> dict:
    from src.scripts.data import tp27_eval as t

    base = X[["race_id", "horse_id", "year", "p_win", "p_top2", "p_top3"]]
    fin = X.set_index(KEY)["finish_position"]
    out = {"overall": t._cmp(base, new, fin, label), "by_year": {}}
    for y in YEARS:
        ids = set(X.loc[X["year"] == y, "race_id"])
        a, b = base[base["race_id"].isin(ids)], new[new["race_id"].isin(ids)]
        c = t._cmp(a, b, fin, f"{label} {y}")
        out["by_year"][str(y)] = {k: c[k] for k in ("Δwin", "win_hi", "Δtop2", "Δtop3") if k in c}
    return out


def run_variant(X, axes, label, lams=(100.0, 300.0, 1000.0, 3000.0), relative_opts=(False, True)):
    st, fsb = strata(X)
    Xd, pen = design(axes, st, fsb)
    ok = X["ok"].to_numpy(); year = X["year"].to_numpy()
    res = {}
    # λ は『3 年の LogLoss(top3)』で選ぶ（パラメータ 1 つ）
    best = None
    for lam in lams:
        d3 = loyo_delta(Xd, pen, X["top3"].to_numpy(), _logit(X["p_top3"].to_numpy()), ok, year, lam)
        p = _sig(_logit(X["p_top3"].to_numpy()) + d3)
        y = X["top3"].to_numpy()
        ll = -np.mean(np.where(ok, y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)), 0)) * len(y) / ok.sum()
        if best is None or ll < best[0]:
            best = (ll, lam)
    lam = best[1]
    dw = loyo_delta(Xd, pen, X["win"].to_numpy(), _logit(X["p_win"].to_numpy()), ok, year, lam)
    d2 = loyo_delta(Xd, pen, X["top2"].to_numpy(), _logit(X["p_top2"].to_numpy()), ok, year, lam)
    d3 = loyo_delta(Xd, pen, X["top3"].to_numpy(), _logit(X["p_top3"].to_numpy()), ok, year, lam)
    for rel in relative_opts:
        new = adjust(X, dw, d2, d3, rel)
        res["相対" if rel else "絶対"] = evaluate(X, new, f"{label}{'(相対)' if rel else ''}")
    res["lam"] = lam
    res["delta_sd"] = {"win": float(np.std(dw[ok])), "top3": float(np.std(d3[ok]))}
    return res, (dw, d2, d3)


def cell_table(X: pd.DataFrame, pos, pace) -> list[dict]:
    """区分別の実際の複勝圏率・親の予測平均・残差（実際 − 親）。"""
    d = pd.DataFrame({"pos": pos, "pace": pace, "y": X["top3"], "p": X["p_top3"], "yw": X["win"], "pw": X["p_win"], "year": X["year"], "ok": X["ok"]})
    d = d[d["ok"]]
    g = d.groupby(["pos", "pace"]).agg(n=("y", "size"), top3=("y", "mean"), p_top3=("p", "mean"), win=("yw", "mean"), p_win=("pw", "mean")).reset_index()
    g["resid3"] = g["top3"] - g["p_top3"]
    g["resid_w"] = g["win"] - g["p_win"]
    by = d.assign(r3=d["y"] - d["p"]).groupby(["pos", "pace", "year"])["r3"].mean().unstack("year")
    g = g.merge(by.reset_index().rename(columns={y: f"resid3_{y}" for y in YEARS}), on=["pos", "pace"], how="left")
    return g.round(4).to_dict("records")


# ───────────────────────── コマンド
def cmd_oracle(extra_axes: dict | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    D, Tb, X = load_all()
    pos, pos_ok, pace, pace_ok, qs = actual_axes(X, Tb)
    ok_all = pos_ok & pace_ok
    X["ok"] = X["ok"] & ok_all
    res = {"n_rows": int(X["ok"].sum()), "pos_cuts": POS_CUTS, "pace_cuts_z": qs.tolist()}
    res["cells"] = cell_table(X, pos, pace)
    Ppos, Ppace = onehot(pos, 4, pos_ok), onehot(pace, 3, pace_ok)
    U3 = np.full((len(X), 3), 1 / 3)
    U4 = np.full((len(X), 4), 1 / 4)
    for name, axes in {"オラクル: 位置×ペース": [Ppos, Ppace], "オラクル: 位置のみ": [Ppos, U3], "オラクル: ペースのみ": [U4, Ppace]}.items():
        r, _ = run_variant(X, axes, name)
        res[name] = r
        _log(f"{name}: " + json.dumps({k: {m: round(v['overall'].get(m, float('nan')), 5) for m in ('Δwin', 'win_hi', 'Δtop2', 'Δtop3')} for k, v in r.items() if k in ('絶対', '相対')}, ensure_ascii=False))
    (OUT / "oracle.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return res


def cmd_predicted() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    D, Tb, X = load_all()
    pos, pos_ok, pace, pace_ok, qs = actual_axes(X, Tb)
    Ppos, ph = predicted_pos(X, D)
    Ppace, zhat = predicted_pace(X, Tb, qs)
    U3, U4 = np.full((len(X), 3), 1 / 3), np.full((len(X), 4), 1 / 4)
    res = {"n_rows": int(X["ok"].sum())}
    # 予測の質（参考）: 位置の hard 正解率、ペースの 3 分位正解率
    res["quality"] = {"pos_acc": float((Ppos.argmax(1)[pos_ok & X["ok"].to_numpy()] == pos[pos_ok & X["ok"].to_numpy()]).mean()),
                       "pace_acc": float((Ppace.argmax(1)[pace_ok & X["ok"].to_numpy()] == pace[pace_ok & X["ok"].to_numpy()]).mean())}
    for name, axes in {"予測: 位置×ペース": [Ppos, Ppace], "予測: 位置のみ": [Ppos, U3], "予測: ペースのみ": [U4, Ppace]}.items():
        r, _ = run_variant(X, axes, name)
        res[name] = r
        _log(f"{name}: " + json.dumps({k: {m: round(v['overall'].get(m, float('nan')), 5) for m in ('Δwin', 'win_hi', 'Δtop2', 'Δtop3')} for k, v in r.items() if k in ('絶対', '相対')}, ensure_ascii=False))
    (OUT / "predicted.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["oracle", "predicted"])
    a = ap.parse_args()
    {"oracle": cmd_oracle, "predicted": cmd_predicted}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

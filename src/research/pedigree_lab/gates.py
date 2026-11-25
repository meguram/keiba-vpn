"""R0-PL1 の門 0〜3（docs/operations/pedigree_lab/decisions.md の 1 号の決定どおり）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.gates <gate0|gate1|shapes|roi|hold>

- 場面の定義（決定の 2 章）:
  - FD 初ダート: 2014 年以降の生まれ・中央のデビューが新馬戦・前走までがすべて芝（取消・除外は数えない）・今回がダート。
  - SD ダート 2 戦目: 同じ条件の馬が、初ダートを 1 回走った後の次のダート。
  - DX: FD の馬がいるレースの、ダート経験馬（形の確かめ (f) だけ）。
- 規則: R-A（FD の中の 1 位かつ全頭 3 位以内・複勝）・R-B（FD の上位 20%・複勝）・R-C（新馬戦の 1 位・複勝）・R-D（SD の上位 20%・複勝）。
- 点は第 4 版の pure の勝ちのモデル。主は 2 代母の単位の OOF（kf_v4_pure_dd）、副は開催の単位（kf_v4_pure_meeting）。
- as-of は data.asof（その日より厳密に前）で作ってある。同じ日の結果が入っていないことは tests/research/test_pedigree_lab_asof.py で確かめている。
出力: OUT/gate_<名前>.json。費用 0 円。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import bets, data
from src.research.pedigree_lab.hyp_tests import clogit

KF = {"dd": "kf_v4_pure_dd", "meeting": "kf_v4_pure_meeting"}
FCOLS = ["race_id", "horse_id", "meeting", "date", "win", "top3", "fpct", "odds", "popularity", "n_run", "surf", "cls",
         "horse_number", "payoff", "split", "by", "career", "n_surf_before", "sire", "sire_surf_top3", "sire_all_top3",
         "bms_surf_top3", "bms_all_top3"]


def scenes(F: pd.DataFrame) -> pd.DataFrame:
    """場面の印（着順は使わない。本人の成績は (b) の分けにだけ使う列 prev_turf_fpct を別に作る）。"""
    F = F.sort_values(["horse_id", "date"], kind="stable").copy()
    g = F.groupby("horse_id")
    first_cls = g["cls"].transform("first")
    exact = pd.to_numeric(F["by"], errors="coerce").fillna(0) >= 2014
    debut_new = first_cls == "NEW"
    ok = exact & debut_new
    isD = F["surf"] == "D"
    nD_before = isD.astype(int).groupby(F["horse_id"]).cumsum() - isD.astype(int)
    F["FD"] = ok & isD & (F["career"] > 0) & (nD_before == 0)
    # 初ダートを走った馬の、次のダート（ダート 2 戦目）
    had_fd = (F["FD"].astype(int).groupby(F["horse_id"]).cumsum() - F["FD"].astype(int)) > 0
    F["SD"] = ok & isD & (nD_before == 1) & had_fd
    # 前走までの芝の分位の平均（本人の成績。(b) の分けにだけ使う）
    tf = np.where(F["surf"] == "T", F["fpct"], np.nan)
    F["_tf"] = tf
    F["_tn"] = (~np.isnan(tf)).astype(float)
    F["_tfs"] = np.nan_to_num(tf)
    cs = F.groupby("horse_id")["_tfs"].cumsum() - F["_tfs"]
    cn = F.groupby("horse_id")["_tn"].cumsum() - F["_tn"]
    F["prev_turf_fpct"] = cs / cn.replace(0, np.nan)
    F["dirt_lean"] = (F["sire_surf_top3"] - F["sire_all_top3"]) + 0.5 * (F["bms_surf_top3"] - F["bms_all_top3"])
    return F.drop(columns=["_tf", "_tn", "_tfs"]).sort_index()


def load(group: str, part: str = "dev", reason: str = "", purpose: str = "select") -> pd.DataFrame:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), purpose, f"r0pl1_{group}_{part}", reason))
    F = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=FCOLS)
    F = scenes(F)
    O = pd.read_parquet(data.OUT / f"{KF[group]}.parquet")
    O = O[O["part"] == part][["race_id", "horse_id", "p_win"]]
    X = O.merge(F, on=["race_id", "horse_id"], how="left")
    assert X["date"].max() <= pd.Timestamp("2024-12-31")
    X["year"] = X["date"].dt.year
    X = X[X["odds"] > 0].copy()
    inv = 1.0 / X["odds"].to_numpy()
    X["q"] = inv / pd.Series(inv).groupby(X["race_id"].to_numpy()).transform("sum").to_numpy()
    X["lp"] = np.log(np.clip(X["p_win"], 1e-5, 1))
    X["lq"] = np.log(np.clip(X["q"], 1e-5, 1))
    X = X.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    X["rank"] = X.groupby("race_id")["p_win"].rank(ascending=False, method="first")
    X["DX"] = (X["surf"] == "D") & (X["n_surf_before"] > 0) & ~X["FD"] & \
        X.groupby("race_id")["FD"].transform("any")
    return X


def thresholds(X: pd.DataFrame) -> dict:
    return {"FD_q80": float(X.loc[X["FD"], "p_win"].quantile(0.8)), "SD_q80": float(X.loc[X["SD"], "p_win"].quantile(0.8))}


def rules(X: pd.DataFrame, th: dict) -> dict[str, np.ndarray]:
    fd = X["FD"].to_numpy()
    v = np.where(fd, X["p_win"].to_numpy(), -1.0)
    best = pd.Series(v).groupby(X["race_id"].to_numpy()).transform("max").to_numpy()
    ra = fd & (v == best) & (X["rank"].to_numpy() <= 3)
    rb = fd & (X["p_win"].to_numpy() >= th["FD_q80"])
    rc = (X["cls"] == "NEW").to_numpy() & (X["rank"].to_numpy() == 1)
    rd = X["SD"].to_numpy() & (X["p_win"].to_numpy() >= th["SD_q80"])
    return {"R-A": ra, "R-B": rb, "R-C": rc, "R-D": rd}


def centered(X, v):
    return v - pd.Series(v).groupby(X["race_id"].to_numpy()).transform("mean").to_numpy()


def scene_alpha(X: pd.DataFrame, masks: dict[str, np.ndarray], n_boot: int = 200, by_sire: bool = False) -> dict:
    """場面の馬に α の上乗せ（lp・lq を中心化した交互作用 ＋ 水準）を置いた条件付きロジット。区間は開催（と父）の単位。"""
    y = X["win"].to_numpy(float)
    rc = pd.factorize(X["race_id"])[0]
    lpc, lqc = centered(X, X["lp"].to_numpy()), centered(X, X["lq"].to_numpy())
    cols, names = [X["lp"].to_numpy(), X["lq"].to_numpy()], []
    for nm, m in masks.items():
        m = m.astype(float)
        cols += [lpc * m, lqc * m, m]
        names.append(nm)
    Z = np.c_[tuple(cols)]
    b = clogit(Z, y, rc)
    out = {nm: {"alpha_extra": round(float(b[2 + 3 * i]), 4), "rows": int(masks[nm].sum())} for i, nm in enumerate(names)}
    for unit in (["meeting", "sire"] if by_sire else ["meeting"]):
        u = pd.factorize(X[unit].astype(str))[0]
        nu = u.max() + 1
        rng = np.random.default_rng(5)
        bs = np.array([clogit(Z, y, rc, n_iter=8, w=np.bincount(rng.integers(0, nu, nu), minlength=nu)[u].astype(float))
                       for _ in range(n_boot)])
        for i, nm in enumerate(names):
            out[nm][f"ci_{unit}"] = np.percentile(bs[:, 2 + 3 * i], [2.5, 97.5]).round(4).tolist()
    return out


def harville_top(p: np.ndarray, rid: np.ndarray, n_run: np.ndarray) -> np.ndarray:
    """勝ちの確率から、複勝の払いの範囲（8 頭以上は 3 着内・7 頭以下は 2 着内）に入る確率（Harville）。"""
    out = np.zeros_like(p)
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], len(p)]
    for s, e in zip(starts, ends):
        q = p[s:e] / p[s:e].sum()
        n = len(q)
        k = 3 if n_run[s] >= 8 else 2
        A = q[:, None] * q[None, :] / (1 - q[:, None])        # i 1 着・j 2 着
        np.fill_diagonal(A, 0)
        P2 = A.sum(0)
        if k == 2:
            out[s:e] = q + P2
            continue
        den = 1 - q[:, None] - q[None, :]
        np.fill_diagonal(den, 1)
        W = A / np.clip(den, 1e-6, None)                      # 3 着の k に掛ける重み
        P3 = (W.sum() - W.sum(0) - W.sum(1)) * q              # i,j ≠ k の和（i=k・j=k の分を引く）
        out[s:e] = q + P2 + P3
    return np.clip(out, 0, 1)


def gate0(group: str = "dd") -> dict:
    X = load(group, reason="R0-PL1 の門 0（量・上限・新しい定義での α。回収率は見ない）")
    th = thresholds(X)
    R = rules(X, th)
    res = {"group": group, "thresholds": th, "rows": {k: int(v.sum()) for k, v in {"FD": X["FD"], "SD": X["SD"], "DX": X["DX"]}.items()}}
    res["per_year"] = {k: X.loc[v].groupby("year").size().to_dict() for k, v in R.items()}
    res["per_year_mean_dev"] = {k: round(float(np.mean(list(d.values()))), 1) for k, d in res["per_year"].items()}
    res["per_year_mean_full"] = {k: round(v / 0.85, 1) for k, v in res["per_year_mean_dev"].items()}
    res["scratch_races_share"] = None   # 取消は研究の表に残っていない（_prep で落とした）。出馬表の頭数との差で数える: 下
    n_entry = X.groupby("race_id")["n_run"].first()
    res["note_scratch"] = "発走した馬で順位を付けた（研究は楽観）。field_size との差は runs.parquet で数える"
    # 場面の上乗せを当てた Benter（レースの単位の 5 分割の交差）→ 勝ちの確率 → Harville で複勝の範囲へ
    y = X["win"].to_numpy(float)
    rc = pd.factorize(X["race_id"])[0]
    fd = X["FD"].to_numpy().astype(float)
    lpc, lqc = centered(X, X["lp"].to_numpy()), centered(X, X["lq"].to_numpy())
    Z = np.c_[X["lp"], X["lq"], lpc * fd, lqc * fd, fd]
    fo = np.random.default_rng(7).permutation(rc.max() + 1)[rc] % 5
    pb = np.zeros(len(X))
    for k in range(5):
        A = fo != k
        bk = clogit(Z[A], y[A], pd.factorize(X.loc[A, "race_id"])[0])
        eta = Z[~A] @ bk
        rid = X.loc[~A, "race_id"].to_numpy()
        e = np.exp(eta - pd.Series(eta).groupby(rid).transform("max").to_numpy())
        pb[~A] = e / pd.Series(e).groupby(rid).transform("sum").to_numpy()
    rid = X["race_id"].to_numpy()
    p3 = harville_top(pb, rid, X["n_run"].to_numpy())
    q3 = harville_top(X["q"].to_numpy(), rid, X["n_run"].to_numpy())
    res["upper_E_p3_over_q3"] = {k: round(float(np.mean(p3[v] / q3[v])), 4) for k, v in R.items()}
    res["upper_sum_ratio"] = {k: round(float(p3[v].sum() / q3[v].sum()), 4) for k, v in R.items()}
    res["alpha_FD_new_def"] = scene_alpha(X, {"FD": X["FD"].to_numpy()}, n_boot=200, by_sire=True)["FD"]
    return res


def gate1(group: str = "dd", n_sim: int = 200, mult: float = 1.10, years: float = 1.8) -> dict:
    """検出力の模擬: 開発の初ダートの行から開催の単位で引き直し、払戻を倍率に植え、2 本の同時の下限が 100% を越える確率。"""
    X = load(group, reason="R0-PL1 の門 1（検出力の模擬。払戻の分布を使う・回収率の判定はしない）")
    X = bets.attach_pay(X)
    th = thresholds(X)
    R = rules(X, th)
    res = {"group": group, "mult": mult, "years": years}
    mt = pd.factorize(X["meeting"])[0]
    nm_all = mt.max() + 1
    n_years_dev = X["year"].nunique() * 0.85
    rng = np.random.default_rng(11)
    z = 2.24          # 2 本の同時（Bonferroni の片側 2.5%/2 の近似）
    for unit in ("meeting", "sire"):
        hits = 0
        sds = {}
        for _ in range(n_sim):
            ok = True
            for nm in ("R-A", "R-B"):
                sel = R[nm] & X["has_place"].to_numpy()
                pay = X.loc[sel, "pay_place"].to_numpy() / 100.0
                obs = pay.mean()
                pay = pay * (mult / obs)                        # 真の回収率を mult に植える
                sds[nm] = float(pay.std())
                n_target = int(len(pay) / n_years_dev * years)
                u = pd.factorize(X.loc[sel, unit].astype(str))[0]
                nu = u.max() + 1
                # 束ね単位で引き直して n_target 点に達するまで
                order = rng.permutation(nu)
                cnt = np.bincount(u, minlength=nu)
                take = order[np.cumsum(cnt[order]) <= n_target]
                m = np.isin(u, take)
                sm = pay[m]
                # 区間は束ね単位のブートストラップ（近似: 束ねの和で SE）
                su = np.bincount(u[m], sm, minlength=nu)[take]
                cu = cnt[take]
                r = su.sum() / cu.sum()
                se = np.sqrt(np.sum((su - r * cu) ** 2)) / cu.sum()
                if r - z * se <= 1.0:
                    ok = False
            hits += ok
        res[f"power_{unit}"] = round(hits / n_sim, 3)
        res["sd_place_per_bet"] = {k: round(v, 3) for k, v in sds.items()}
    res["points_per_year_full"] = {nm: round(float((R[nm] & X["has_place"].to_numpy()).sum() / n_years_dev), 1) for nm in ("R-A", "R-B")}
    return res


def shapes(group: str = "dd") -> dict:
    """門 2 の形 (a)〜(f)（尤度の α の上乗せ。回収率は見ない）。"""
    X = load(group, reason="R0-PL1 の門 2 の形 (a)〜(f)（α の上乗せの形）")
    fd = X["FD"].to_numpy()
    dl = X["dirt_lean"].to_numpy()
    med_dl = np.nanmedian(dl[fd])
    ptf = X["prev_turf_fpct"].to_numpy()
    med_tf = np.nanmedian(ptf[fd])
    q5 = pd.qcut(pd.Series(np.where(fd, X["p_win"], np.nan)).rank(method="first"), 5, labels=False).to_numpy()
    early = X["year"].to_numpy() <= 2020
    out = {}
    out["a"] = scene_alpha(X, {"FD_lean_hi": fd & (dl >= med_dl), "FD_lean_lo": fd & (dl < med_dl)})
    out["b"] = scene_alpha(X, {"FD_turf_bad": fd & (ptf >= med_tf), "FD_turf_good": fd & (ptf < med_tf)})
    out["d"] = scene_alpha(X, {"FD_2017_20": fd & early, "FD_2021_24": fd & ~early})
    out["e_f"] = scene_alpha(X, {"FD": fd, "SD": X["SD"].to_numpy(), "DX": X["DX"].to_numpy()})
    # (c) 血統の点の 5 分位ごとの A/E（勝ち ÷ 市場の温度付きの確率）と順位相関
    rc = pd.factorize(X["race_id"])[0]
    bm = clogit(X[["lq"]].to_numpy(), X["win"].to_numpy(float), rc)
    eta = X["lq"].to_numpy() * bm[0]
    rid = X["race_id"].to_numpy()
    e = np.exp(eta - pd.Series(eta).groupby(rid).transform("max").to_numpy())
    pm = e / pd.Series(e).groupby(rid).transform("sum").to_numpy()
    ae = {}
    for k in range(5):
        m = q5 == k
        ae[k] = round(float(X.loc[m, "win"].sum() / pm[m].sum()), 4)
    from scipy.stats import spearmanr
    out["c"] = {"AE_by_quintile": ae, "spearman": round(float(spearmanr(list(ae.keys()), list(ae.values()))[0]), 3)}
    a_, b_, d_, ef = out["a"], out["b"], out["d"], out["e_f"]
    fdx = ef["FD"]["alpha_extra"]
    out["judge"] = {
        "a": a_["FD_lean_hi"]["alpha_extra"] > a_["FD_lean_lo"]["alpha_extra"],
        "b": b_["FD_turf_bad"]["alpha_extra"] > b_["FD_turf_good"]["alpha_extra"],
        "c": out["c"]["spearman"] >= 0.8,
        "d": d_["FD_2021_24"]["alpha_extra"] >= 0.5 * d_["FD_2017_20"]["alpha_extra"],
        "e": ef["SD"]["alpha_extra"] <= 0.5 * fdx,
        "f": ef["DX"]["alpha_extra"] <= 0.5 * fdx,
    }
    out["judge"]["misses"] = int(sum(not v for k, v in out["judge"].items()))
    out["medians"] = {"dirt_lean": float(med_dl), "prev_turf_fpct": float(med_tf)}
    return out


def roi_once(part: str = "dev") -> dict:
    """門 2 の回収率（part=dev。1 回だけ）・門 3 の保留（part=hold。1 回だけ）。閾値は開発の行（2 代母の単位）で固定した値。"""
    th_path = data.OUT / "gate_thresholds.json"
    res = {"part": part}
    for group in ("dd", "meeting"):
        if part == "dev":
            Xd = load(group, "dev", reason="R0-PL1 の門 2 の回収率（1 回だけ）")
        else:
            Xd = load(group, "dev", reason="R0-PL1 の門 3 の閾値の読み出し（開発の行の分位だけ）")
        th = thresholds(Xd)
        if group == "dd":
            th_path.write_text(json.dumps(th))
        X = Xd if part == "dev" else load(group, "hold", reason="R0-PL1 の門 3（保留 split 2 を 1 回だけ開ける）",
                                         purpose="confirm")
        X = bets.attach_pay(X)
        R = rules(X, th)
        out = {"thresholds": th}
        nulls = {}
        for nm, sel in R.items():
            r = bets.roi(X, sel, "place", n_boot=500)
            r["by_sire_ci"] = _sire_ci(X, sel)
            r["win_desc"] = bets.roi(X, sel, "win", n_boot=200) if nm == "R-A" else None
            yrs = {}
            for t in sorted(X["year"].unique()):
                m = sel & (X["year"].to_numpy() == t)
                if m.sum() >= 5:
                    yrs[int(t)] = bets.roi(X, m, "place", n_boot=20)["vs_same_pop"]
            r["years_pos"] = f"{sum(v > 0 for v in yrs.values())}/{len(yrs)}"
            out[nm] = r
            if part == "dev" and nm in ("R-A", "R-B"):
                scene = X["FD"].to_numpy()
                S = X[scene].reset_index(drop=True)
                fn = (lambda Y, nm=nm: rules_scene(Y, th, nm))
                nulls[nm] = bets.perm_null(S, "p_win", fn, "place", n_perm=200)
        if nulls:
            names = list(nulls)
            obs = np.array([out[n]["roi"] - nulls[n].mean() for n in names])
            mx = np.max(np.vstack([nulls[n] - nulls[n].mean() for n in names]), axis=0)
            for n, o in zip(names, obs):
                out[n]["null_mean"] = round(float(nulls[n].mean()), 4)
                out[n]["p_maxT"] = round(float((mx >= o).mean()), 4)
        res[group] = out
    return res


def rules_scene(Y: pd.DataFrame, th: dict, nm: str) -> np.ndarray:
    """帰無の並べ替え用: 場面の行だけの表で R-A・R-B を作り直す（R-A の全頭の順位は元の rank を使う）。"""
    if nm == "R-B":
        return Y["p_win"].to_numpy() >= th["FD_q80"]
    v = Y["p_win"].to_numpy()
    best = pd.Series(v).groupby(Y["race_id"].to_numpy()).transform("max").to_numpy()
    return (v == best) & (Y["rank"].to_numpy() <= 3)


def _sire_ci(X: pd.DataFrame, sel: np.ndarray, n_boot: int = 500) -> list[float]:
    S = X[sel & X["has_place"].to_numpy()]
    if len(S) == 0:
        return []
    u = pd.factorize(S["sire"].astype(str))[0]
    nu = u.max() + 1
    pay = np.bincount(u, S["pay_place"].to_numpy() / 100, minlength=nu)
    cnt = np.bincount(u, minlength=nu)
    rng = np.random.default_rng(3)
    bs = []
    for _ in range(n_boot):
        c = np.bincount(rng.integers(0, nu, nu), minlength=nu)
        bs.append((c * pay).sum() / max((c * cnt).sum(), 1))
    return np.percentile(bs, [2.5, 97.5]).round(4).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["gate0", "gate1", "shapes", "roi", "hold"])
    a = ap.parse_args()
    fn = {"gate0": gate0, "gate1": gate1, "shapes": shapes, "roi": lambda: roi_once("dev"), "hold": lambda: roi_once("hold")}[a.gate]
    res = fn()
    path = data.OUT / f"gate_{a.gate}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()

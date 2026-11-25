"""最後の 1 回: 血統だけのモデルの最終の評価（2025・26 年。台帳 eval_only。log.md「2025・26 年の 1 回」の線 M1〜M5）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.final_eval --version v4 [--check-only]

1. 2016〜2026 の全行で as-of の特徴量を作り直す（第 4 版。v5e なら祖先の列も）。k・床は 2019 年・2016 年までの行で決まるので同じ。
2. 2024 年までの行が、凍結した feat_<版>.parquet と一致することを列ごとに確かめる（一致しなければ止める）。
   --check-only は 2024 年までの行だけで作り直して確かめ、ここで止まる（2025・26 年に触れない予行）。
3. 凍結した 2 代母の単位の 5 折のモデル（models_<版>_pure_dd）の予測の平均 → 温度 → レースの中で和 1。
4. M1〜M5 を出す。回収率は出さない。買い方の選びに使わない。
出力: OUT/final_eval_<版>.json。費用 0 円。
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.research.pedigree_lab import data, features4, gates, model
from src.research.pedigree_lab.hyp_tests import clogit, prep

HYP9 = [("H1", ["sire_surf_top3"], ["sire_all_top3"]), ("H2", ["sire_sd_top3"], ["sire_surf_top3"]),
        ("H3", ["bms_all_top3"], ["sire_all_top3"]), ("H4", ["dam_v_win"], ["sire_all_top3", "bms_all_top3"]),
        ("H6'", ["sire_rasurf_top3"], ["sire_surf_top3"]), ("H7", ["dd_v_win"], ["sire_all_top3", "bms_all_top3", "dam_v_win"]),
        ("H9", ["nick_all_top3"], ["sire_all_top3", "bms_all_top3"]),
        ("H12", ["dam_v_quick"], ["sire_all_top3", "bms_all_top3", "dam_v_win"]), ("H14", ["sire_dz"], ["sire_sd_top3"])]
DEV_SIGN = {"H1": 1, "H2": 1, "H3": 1, "H4": 1, "H6'": 1, "H7": 1, "H9": 1, "H12": 1, "H14": -1}


def build_all(version: str, check_only: bool = False) -> pd.DataFrame:
    if check_only:      # 2025・26 年に触れずに、作り直しが凍結の表と同じになるかだけを確かめる（開発の行だけ）
        r = data.load_dev(reason="最終の評価の前の作り直しの確かめ（2024 年まで）", version=f"final_check_{version}")
    else:
        r = data.load_final(version=f"final_{version}", reason="血統だけのモデルの最終の評価（M1〜M5。回収率は出さない）")
    ped = data.load_ped()
    X, _ = features4.build(r, ped, sub=True)
    X["split"] = data.split_meetings(X["meeting"])
    if version == "v5e":
        from src.research.pedigree_lab import features5
        tmp = data.OUT / "_final_feat_tmp.parquet"
        X.to_parquet(tmp, index=False)
        X = features5.build(tmp)
    return X


def check_equal(X: pd.DataFrame, version: str, cols: list[str]) -> dict:
    F = pd.read_parquet(data.OUT / f"feat_{'v4' if version == 'v4' else 'v5e'}.parquet", columns=["race_id", "horse_id", *cols])
    A = X.loc[X["date"] <= data.DEV_END, ["race_id", "horse_id", *cols]]
    M = F.merge(A, on=["race_id", "horse_id"], suffixes=("_f", "_n"), how="outer", indicator=True)
    bad = {}
    for c in cols:
        a, b = M[f"{c}_f"].astype(float).to_numpy(), M[f"{c}_n"].astype(float).to_numpy()
        ok = np.isclose(a, b, rtol=1e-4, atol=1e-6, equal_nan=True)
        if not ok.all():
            bad[c] = int((~ok).sum())
    return {"rows_frozen": len(F), "rows_rebuilt": len(A), "unmatched": int((M["_merge"] != "both").sum()), "bad_cols": bad}


def predict(X: pd.DataFrame, version: str) -> np.ndarray:
    d = data.OUT / f"models_{version}_pure_dd"
    meta = json.loads((d / "win_meta.json").read_text())
    raw = np.zeros(len(X))
    for k in range(5):
        m = lgb.Booster(model_file=str(d / f"win_fold{k}.txt"))
        raw += m.predict(X[meta["cols"]]) / 5
    return model.race_norm(raw, X["race_id"].to_numpy(), 1.0, meta["temp"])


def evaluate(X: pd.DataFrame, p: np.ndarray, oof_gain: float) -> dict:
    T = X.assign(p_win=p)
    T = T[T["date"] > data.DEV_END].reset_index(drop=True)
    w = T["win"].to_numpy() == 1
    n = T["n_run"].to_numpy()
    d_race = np.log(T["p_win"].to_numpy()[w]) - np.log(1 / n[w])
    mt = pd.factorize(T.loc[w, "meeting"])[0]
    nu = mt.max() + 1
    s, c = np.bincount(mt, d_race, minlength=nu), np.bincount(mt, minlength=nu)
    rng = np.random.default_rng(21)
    bs = [((wb := np.bincount(rng.integers(0, nu, nu), minlength=nu)) * s).sum() / (wb * c).sum() for _ in range(1000)]
    res = {"rows": len(T), "races": int(T["race_id"].nunique()), "meetings": int(nu)}
    g = float(d_race.mean())
    lo = float(np.percentile(bs, 2.5))
    res["M1"] = {"gain": round(g, 5), "ci": [round(lo, 5), round(float(np.percentile(bs, 97.5)), 5)], "oof_gain": oof_gain,
                 "ratio": round(g / oof_gain, 3), "pass": bool(g >= 0.75 * oof_gain and lo >= 0.5 * oof_gain)}
    # M2 較正（10 分位のロジットの傾き・平均の比）
    T["dec"] = pd.qcut(T["p_win"].rank(method="first"), 10, labels=False)
    cal = T.groupby("dec").agg(pred=("p_win", "mean"), act=("win", "mean"))
    lg = lambda z: np.log(z / (1 - z))
    slope = float(np.polyfit(lg(cal["pred"].clip(1e-4, 0.999)), lg(cal["act"].clip(1e-4, 0.999)), 1)[0])
    ratio = float(T["p_win"].mean() / T["win"].mean())
    res["M2"] = {"slope": round(slope, 3), "mean_ratio": round(ratio, 4), "deciles": cal.round(4).reset_index().to_dict("records"),
                 "pass": bool(0.85 <= slope <= 1.15 and 0.95 <= ratio <= 1.05)}
    # M3 仮説 9 本の符号
    rc = pd.factorize(T["race_id"])[0]
    signs = {}
    for nm, test, base in HYP9:
        cols = base + test
        b = clogit(prep(T, cols), T["win"].to_numpy(float), rc)[-1]
        signs[nm] = round(float(b), 4)
    same = sum(np.sign(v) == DEV_SIGN[k] for k, v in signs.items())
    res["M3"] = {"beta": signs, "same_sign": int(same), "pass": bool(same >= 8)}
    # M4・M5 Benter（確定オッズ）と初ダートの α の上乗せ
    T = T[T["odds"] > 0].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    inv = 1 / T["odds"].to_numpy()
    T["q"] = inv / pd.Series(inv).groupby(T["race_id"].to_numpy()).transform("sum").to_numpy()
    T["lp"], T["lq"] = np.log(np.clip(T["p_win"], 1e-5, 1)), np.log(np.clip(T["q"], 1e-5, 1))
    rc = pd.factorize(T["race_id"])[0]
    b = clogit(T[["lp", "lq"]].to_numpy(), T["win"].to_numpy(float), rc)
    sc = gates.scene_alpha(T, {"FD": T["FD"].to_numpy()}, n_boot=200, by_sire=True)["FD"]
    res["M4"] = {**sc, "pass": bool(sc["alpha_extra"] >= 0.18)}
    wq = T["win"].to_numpy() == 1
    res["M5"] = {"alpha_all": round(float(b[0]), 4), "beta": round(float(b[1]), 4),
                 "market_race_ll": round(float(-np.mean(np.log(T["q"].to_numpy()[wq]))), 4),
                 "model_race_ll": round(float(-np.mean(np.log(T["p_win"].to_numpy()[wq]))), 4)}
    sub = {"2歳戦": T["rage"] == "A2", "新馬戦": T["cls"] == "NEW", "初年度の産駒の父（as-of 300 走未満）": T["sire_all_n"] < np.log1p(300)}
    for nm, m in sub.items():
        m = m.to_numpy() & wq
        res["M5"][nm] = {"rows_win": int(m.sum()),
                         "gain": round(float(np.mean(np.log(T["p_win"].to_numpy()[m]) - np.log(1 / T["n_run"].to_numpy()[m]))), 5)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v4", choices=["v4", "v5e"])
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    X = build_all(a.version, a.check_only)
    meta = json.loads((data.OUT / f"models_{a.version}_pure_dd" / "win_meta.json").read_text())
    chk = check_equal(X, a.version, meta["cols"])
    print("check", json.dumps(chk, ensure_ascii=False))
    if chk["bad_cols"] or chk["unmatched"]:
        raise SystemExit("2024 年までの行が凍結の表と一致しない（止める）")
    if a.check_only:
        data.finish_all()
        return
    F = gates.scenes(X[gates.FCOLS].copy())
    X["FD"] = F["FD"].to_numpy()
    p = predict(X, a.version)
    oof = {"v4": 0.08152}.get(a.version)
    res = {"version": a.version, "check": chk, **evaluate(X, p, oof)}
    path = data.OUT / f"final_eval_{a.version}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()

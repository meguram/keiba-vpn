"""tp-55: ラップ予想モデルの walk-forward 評価。ホストで実行（重い処理は /tmp/heavy_run.sh 経由）。

python -m src.scripts.data.tp55_eval main --year 2024 --scenario A [--drop ten]       # 学習は評価年より前、評価は評価年
python -m src.scripts.data.tp55_eval ablate --year 2024                              # 特徴量グループの抜き取り（軽い版）
python -m src.scripts.data.tp55_eval final2026                                       # 2026 年を 1 回だけ（学習は 2020〜2025）＋API 用モデルの保存

入力: tp55_build の races_feat.parquet。出力: --out/<label>.json（指標）と <label>_pred.parquet（レースごとの予測）。
2025 年は参考、2026 年は最後に 1 回（評価前に設定を固定）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp55_lapfeat as lf
from src.pipeline.models import lap_forecast as lfm

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("TP55_OUT", "/home/jovyan/work/tp55_out"))
THREADS = int(os.environ.get("TP55_THREADS", "3"))


def load() -> pd.DataFrame:
    d = pd.read_parquet(OUT / "races_feat.parquet")
    d["year"] = d["date"].str[:4].astype(int)
    return d


def _logn(x, mu, sd):
    return -0.5 * np.log(2 * np.pi * sd**2) - 0.5 * ((x - mu) / sd) ** 2


def evaluate(model: lfm.LapForecast, train: pd.DataFrame, test: pd.DataFrame, n_draw: int = 400, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    out = model.predict(test)
    test = test.reset_index(drop=True)
    n = len(test)
    laps_t = [json.loads(s) for s in test["laps_json"]]
    L = lfm.laps_matrix(laps_t)
    n_laps = np.array([len(a) for a in laps_t])
    Pfull = lfm.par_matrix(test, model.par)
    dev_full = L - Pfull
    ok = ~np.isnan(Pfull[np.arange(n), 0])
    lab = model._labels(test, dev_full, n_laps, fit=False)
    K = model.K
    proba = out["proba"]
    J = lfm.MAXJ
    dev = dev_full[:, :J]
    # 分類
    y = lab[ok]
    p = proba[ok]
    ll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))
    prior = model.prior
    ll_prior = float(-np.mean(np.log(np.clip(prior[y], 1e-9, 1))))
    onehot = np.eye(K)[y]
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    brier_prior = float(np.mean(np.sum((prior[None, :] - onehot) ** 2, axis=1)))
    ece = []
    for k in range(K):
        pk = p[:, k]
        qs = np.quantile(pk, np.linspace(0, 1, 6))
        bins = np.clip(np.searchsorted(qs[1:-1], pk, side="right"), 0, 4)
        ece.append(float(np.mean([abs(pk[bins == b].mean() - onehot[bins == b, k].mean()) * (bins == b).mean() for b in range(5) if (bins == b).any()]) * 5))
    rel = []
    for k in range(K):
        pk = p[:, k]
        qs = np.quantile(pk, np.linspace(0, 1, 6))
        bins = np.clip(np.searchsorted(qs[1:-1], pk, side="right"), 0, 4)
        rel.append([{"pred": round(float(pk[bins == b].mean()), 3), "obs": round(float(onehot[bins == b, k].mean()), 3), "n": int((bins == b).sum())} for b in range(5) if (bins == b).any()])
    acc = float(np.mean(np.argmax(p, axis=1) == y))
    # ラップ予測
    mu = out["mu_dev"]
    w = proba[:, :, None]
    num = np.nansum(w * mu, axis=1)
    den = np.nansum(w * (~np.isnan(mu)), axis=1)
    mix = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    direct = out["direct_dev"]
    oracle = np.full_like(mix, np.nan)
    for i in range(n):
        oracle[i] = mu[i, lab[i]]
    mae = {}
    for j in range(J):
        m = ok & (n_laps > j) & ~np.isnan(dev[:, j])
        if m.sum() < 30:
            continue
        mae[str(j)] = {
            "n": int(m.sum()), "mixture": float(np.nanmean(np.abs(mix[m, j] - dev[m, j]))), "par": float(np.mean(np.abs(dev[m, j]))),
            "direct": float(np.nanmean(np.abs(direct[m, j] - dev[m, j]))), "oracle": float(np.nanmean(np.abs(oracle[m, j] - dev[m, j]))),
        }
    # 前半3F・後半3F（秒）
    def agg(pred_mat, which):
        res = np.full(n, np.nan)
        for i in range(n):
            m = n_laps[i]
            if which == "f3":
                res[i] = np.nansum(pred_mat[i, :3]) if not np.isnan(pred_mat[i, :3]).any() else np.nan
            elif m <= J:
                v = pred_mat[i, m - 3:m]
                res[i] = np.nansum(v) if not np.isnan(v).any() else np.nan
        return res
    tf3 = np.array([dev_full[i, :3].sum() for i in range(n)])
    tl3 = np.array([dev_full[i, n_laps[i] - 3:n_laps[i]].sum() for i in range(n)])
    agg_res = {}
    for name, truth, which in (("first3f", tf3, "f3"), ("last3f", tl3, "l3")):
        pm, pd_, po = agg(mix, which), agg(direct, which), agg(oracle, which)
        m = ok & ~np.isnan(pm) & ~np.isnan(truth)
        agg_res[name] = {"n": int(m.sum()), "mixture_mae": float(np.mean(np.abs(pm[m] - truth[m]))), "par_mae": float(np.mean(np.abs(truth[m]))),
                         "direct_mae": float(np.nanmean(np.abs(pd_[m] - truth[m]))), "oracle_mae": float(np.nanmean(np.abs(po[m] - truth[m]))),
                         "mixture_r2": float(1 - np.sum((pm[m] - truth[m]) ** 2) / np.sum((truth[m] - truth[m].mean()) ** 2)),
                         "direct_r2": float(1 - np.nansum((pd_[m] - truth[m]) ** 2) / np.sum((truth[m] - truth[m].mean()) ** 2))}
    # 予測密度（独立ガウスの混合）の対数尤度（ラップあたりの平均）
    sd = out["sd"]
    ll_mix = np.full(n, np.nan)
    ll_dir = np.full(n, np.nan)
    ll_par = np.full(n, np.nan)
    par_sd = _par_sd(model, train)
    nj = np.zeros(n)
    for i in range(n):
        if not ok[i]:
            continue
        js = [j for j in range(min(int(n_laps[i]), J)) if not np.isnan(dev[i, j]) and not np.isnan(direct[i, j]) and j in model.direct_sd and not np.isnan(sd[:, j]).any() and j in par_sd]
        if not js:
            continue
        nj[i] = len(js)
        comp = np.array([math.log(max(proba[i, k], 1e-12)) + sum(_logn(dev[i, j], mu[i, k, j] if not np.isnan(mu[i, k, j]) else 0.0, sd[k, j]) for j in js) for k in range(K)])
        mx = comp.max()
        ll_mix[i] = (mx + math.log(np.exp(comp - mx).sum())) / len(js)
        ll_dir[i] = sum(_logn(dev[i, j], direct[i, j], model.direct_sd[j]) for j in js) / len(js)
        ll_par[i] = sum(_logn(dev[i, j], 0.0, par_sd[j]) for j in js) / len(js)
    # 80% 区間のカバー率（混合からサンプリング。ラップ内は独立と仮定）
    rng = np.random.default_rng(seed)
    cover = {}
    cum = np.cumsum(proba, axis=1)
    pick = []
    for i in range(n):
        r = rng.random(n_draw)
        pick.append(np.minimum((r[:, None] > cum[i][None, :]).sum(axis=1), K - 1))
    pick = np.array(pick)                                         # n × draws
    for name, js in (("lap0", [0]), ("lap1", [1]), ("first3f", [0, 1, 2])):
        inside = []
        for i in range(n):
            if not ok[i] or any(np.isnan(dev[i, j]) for j in js) or any(np.isnan(sd[pick[i], j]).any() for j in js):
                continue
            s = np.zeros(n_draw)
            for j in js:
                mj = np.nan_to_num(mu[i, pick[i], j])
                s += mj + rng.standard_normal(n_draw) * sd[pick[i], j]
            lo, hi = np.quantile(s, [0.1, 0.9])
            t = sum(dev[i, j] for j in js)
            inside.append(lo <= t <= hi)
        cover[name] = {"n": len(inside), "coverage80": float(np.mean(inside)) if inside else None}
    ins = []
    for i in range(n):
        if not ok[i] or n_laps[i] > J or any(np.isnan(sd[pick[i], j]).any() for j in range(n_laps[i] - 3, n_laps[i])):
            continue
        s = np.zeros(n_draw)
        for j in range(n_laps[i] - 3, n_laps[i]):
            s += np.nan_to_num(mu[i, pick[i], j]) + rng.standard_normal(n_draw) * sd[pick[i], j]
        lo, hi = np.quantile(s, [0.1, 0.9])
        ins.append(lo <= tl3[i] <= hi)
    cover["last3f"] = {"n": len(ins), "coverage80": float(np.mean(ins)) if ins else None}
    # 距離帯・面別の 3F 誤差
    seg = {}
    for key, mask in (("sprint<=1400", test["distance"] <= 1400), ("mile1500-1800", (test["distance"] > 1400) & (test["distance"] <= 1800)),
                      ("middle1900-2200", (test["distance"] > 1800) & (test["distance"] <= 2200)), ("long>=2300", test["distance"] >= 2300),
                      ("芝", test["surface"] == "芝"), ("ダート", test["surface"] == "ダート")):
        m = mask.values & ok
        pm = agg(mix, "f3")
        pl = agg(mix, "l3")
        a = m & ~np.isnan(pm)
        b = m & ~np.isnan(pl)
        seg[key] = {"n": int(m.sum()), "f3_mae": float(np.mean(np.abs(pm[a] - tf3[a]))) if a.any() else None, "f3_par_mae": float(np.mean(np.abs(tf3[a]))) if a.any() else None,
                    "l3_mae": float(np.mean(np.abs(pl[b] - tl3[b]))) if b.any() else None, "l3_par_mae": float(np.mean(np.abs(tl3[b]))) if b.any() else None}
    res = {
        "n_test": int(n), "n_ok": int(ok.sum()), "coverage_share": float(ok.mean()), "scenario": model.scenario, "drop": list(model.drop_groups), "temp": model.temp,
        "classify": {"logloss": ll, "logloss_prior": ll_prior, "brier": brier, "brier_prior": brier_prior, "acc": acc, "ece": ece, "reliability": rel,
                     "prior": [float(x) for x in prior], "test_dist": [float(x) for x in np.bincount(y, minlength=K) / len(y)]},
        "mae_by_lap": mae, "agg": agg_res,
        "density": {"loglik_per_lap_mixture": float(np.nanmean(ll_mix)), "loglik_per_lap_direct": float(np.nanmean(ll_dir)), "loglik_per_lap_par": float(np.nanmean(ll_par))},
        "coverage": cover, "segments": seg,
    }
    if hasattr(model, "cen_sorted"):
        res["cluster_centers"] = {"names": model.scenario_names(), "centers_f3_mid_l3_dev": np.asarray(model.cen_sorted).round(3).tolist(),
                                  "share_train": [float(x) for x in model.prior]}
    if model.scenario == "A":
        res["baseline_pressure"] = baseline_pressure(model, train, test)
    pred = pd.DataFrame({"race_id": test["race_id"], **{f"p{k}": proba[:, k] for k in range(K)}, "exp_f3_dev": agg(mix, "f3"), "exp_l3_dev": agg(mix, "l3"),
                         "true_f3_dev": tf3, "true_l3_dev": tl3, "label": lab})
    return res, pred


def _labels_df(model: lfm.LapForecast, df: pd.DataFrame):
    df = df.reset_index(drop=True)
    laps = [json.loads(x) for x in df["laps_json"]]
    L = lfm.laps_matrix(laps)
    nl = np.array([len(a) for a in laps])
    P = lfm.par_matrix(df, model.par)
    ok = ~np.isnan(P[np.arange(len(df)), 0])
    lab = model._labels(df, L - P, nl, fit=False)
    return df, lab, ok


def baseline_pressure(model: lfm.LapForecast, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """比較: 脚質の頭数だけ（逃げ・先行の頭数、圧力、逃げ不在）の多項ロジットでシナリオを当てる簡易モデル（position_forecast の『先行の頭数でペースを決める』発想）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in ("n_front_hard", "n_stalk_hard", "press", "no_leader", "exp_front", "exp_frontstalk") if c in train.columns]
    if not cols:
        return {}
    tr, ytr, oktr = _labels_df(model, train)
    te, yte, okte = _labels_df(model, test)
    Xtr = tr[cols].fillna(tr[cols].median()).values[oktr]
    Xte = te[cols].fillna(tr[cols].median()).values[okte]
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=500).fit(sc.transform(Xtr), ytr[oktr])
    p = lr.predict_proba(sc.transform(Xte))
    y = yte[okte]
    return {"logloss": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)))), "acc": float(np.mean(np.argmax(p, axis=1) == y))}


def _par_sd(model: lfm.LapForecast, train: pd.DataFrame) -> dict:
    laps = [json.loads(s) for s in train["laps_json"]]
    L = lfm.laps_matrix(laps)
    P = lfm.par_matrix(train.reset_index(drop=True), model.par)
    d = (L - P)[:, : lfm.MAXJ]
    return {j: float(np.nanstd(d[:, j])) for j in range(d.shape[1]) if (~np.isnan(d[:, j])).sum() > 30}


def run_main(year: int, scenario: str, drop: tuple[str, ...], k_cluster: int, label: str) -> None:
    d = load()
    train = d[d["year"] < year]
    test = d[d["year"] == year]
    if "ten" not in drop and year >= 2026:
        raise SystemExit("2026 は ten を落とす")
    m = lfm.LapForecast(scenario=scenario, k_cluster=k_cluster, drop_groups=drop, threads=THREADS).fit(train)
    res, pred = evaluate(m, train, test)
    res.update({"year": year, "n_train": int(len(train)), "label": label})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{label}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    pred.to_parquet(OUT / f"{label}_pred.parquet", index=False)
    print(label, "logloss", res["classify"]["logloss"], "prior", res["classify"]["logloss_prior"], "f3 mae", res["agg"]["first3f"]["mixture_mae"], "par", res["agg"]["first3f"]["par_mae"])


def run_ablate(year: int) -> None:
    """特徴量グループの抜き取り（軽い版）: シナリオ A の分類器と、テン1F・前半3F・後半3F の直接回帰だけで比べる。"""
    import lightgbm as lgb

    d = load()
    train = d[d["year"] < year].reset_index(drop=True)
    test = d[d["year"] == year].reset_index(drop=True)
    laps_tr = [json.loads(s) for s in train["laps_json"]]
    par = lf.fit_par(pd.DataFrame({"venue": train["venue"], "surface": train["surface"], "distance": train["distance"], "direction": train["direction"], "laps": laps_tr}))
    laps_te = [json.loads(s) for s in test["laps_json"]]

    def targets(df, laps):
        P = lfm.par_matrix(df, par)
        L = lfm.laps_matrix(laps)
        dev = L - P
        nl = np.array([len(a) for a in laps])
        st = lfm.scenario_stats(dev, nl)
        ok = ~np.isnan(P[:, 0])
        return dev[:, 0], st[:, 0], st[:, 2], st[:, 2] - st[:, 0], ok

    ytr = targets(train, laps_tr)
    yte = targets(test, laps_te)
    q_lo, q_hi = np.nanpercentile(ytr[3][ytr[4]], [25, 75])
    lab_tr = np.where(ytr[3] <= q_lo, 0, np.where(ytr[3] >= q_hi, 2, 1))
    lab_te = np.where(yte[3] <= q_lo, 0, np.where(yte[3] >= q_hi, 2, 1))
    results = {}
    for drop in [(), ("style",), ("gate",), ("ten",), ("jockey",), ("sameday",), ("field",), ("style", "gate", "ten", "jockey"), ("course",), ("cond",), ("geo",), ("recent",)]:
        num, cats = lfm.feature_columns(drop)
        vocab = lfm.Vocab.fit(train, cats)
        Xtr = vocab.apply(train, num, cats)
        Xte = vocab.apply(test, num, cats)
        mtr, mte = ytr[4], yte[4]
        clf = lfm._lgb_clf(THREADS, 0, 3).fit(Xtr[mtr], lab_tr[mtr])
        p = clf.predict_proba(Xte[mte])
        ll = float(-np.mean(np.log(np.clip(p[np.arange(mte.sum()), lab_te[mte]], 1e-9, 1))))
        prior = np.bincount(lab_tr[mtr], minlength=3) / mtr.sum()
        ll0 = float(-np.mean(np.log(prior[lab_te[mte]])))
        r = {"logloss": ll, "logloss_prior": ll0}
        for name, idx in (("lap0", 0), ("first3f", 1), ("last3f", 2)):
            ytrain = ytr[idx]
            ok_tr = mtr & ~np.isnan(ytrain)
            reg = lfm._lgb_reg(THREADS, 0).fit(Xtr[ok_tr], ytrain[ok_tr])
            pr = reg.predict(Xte)
            okte = mte & ~np.isnan(yte[idx])
            r[f"{name}_mae"] = float(np.mean(np.abs(pr[okte] - yte[idx][okte])))
            r[f"{name}_par_mae"] = float(np.mean(np.abs(yte[idx][okte])))
        results["+".join(drop) or "full"] = r
        print("ablate", drop, r)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ablate_{year}.json").write_text(json.dumps({"year": year, "n_train": len(train), "n_test": len(test), "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")


def _dump_examples(m: lfm.LapForecast, test: pd.DataFrame) -> None:
    """章に載せる例: 2026 年のレースから、距離・面の違う 4 つ（結果のラップと並べる）。"""
    t = test.reset_index(drop=True)
    picks = []
    for surf, dist in (("芝", 1200), ("ダート", 1800), ("芝", 1600), ("芝", 2000)):
        sub = t[(t["surface"] == surf) & (t["distance"] == dist)]
        if len(sub):
            picks.append(int(sub.index[len(sub) // 2]))
    out = m.predict(t)
    ex = []
    for i in picks:
        summ = lfm.mixture_summary(out, i, m.scenario_names(), seed=i)
        ex.append({"race_id": t.at[i, "race_id"], "date": t.at[i, "date"], "venue": t.at[i, "venue"], "surface": t.at[i, "surface"], "distance": int(t.at[i, "distance"]),
                   "field": int(t.at[i, "field_size"]) if pd.notna(t.at[i, "field_size"]) else None, "actual_laps": json.loads(t.at[i, "laps_json"]),
                   "par": [None if x != x else round(float(x), 2) for x in out["par"][i][: int(out["n_laps"][i])]], "summary": summ})
    (OUT / "examples_2026.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")


def run_final2026() -> None:
    d = load()
    train = d[d["year"] <= 2025]
    test = d[d["year"] == 2026]
    OUT.mkdir(parents=True, exist_ok=True)
    for scenario, kc, drop, label in (("A", 5, ("ten",), "final2026_A_noten"), ("A", 5, (), "final2026_A_full"), ("B", 5, ("ten",), "final2026_B_noten")):
        m = lfm.LapForecast(scenario=scenario, k_cluster=kc, drop_groups=drop, threads=THREADS).fit(train)
        res, pred = evaluate(m, train, test)
        res.update({"year": 2026, "n_train": int(len(train)), "label": label})
        (OUT / f"{label}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        pred.to_parquet(OUT / f"{label}_pred.parquet", index=False)
        m.save(OUT / f"model_{label}.pkl")
        if label == "final2026_A_noten":
            _dump_examples(m, test)
        print(label, "logloss", res["classify"]["logloss"], "f3", res["agg"]["first3f"]["mixture_mae"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["main", "ablate", "final2026"])
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--scenario", default="A")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--drop", nargs="*", default=[])
    a = ap.parse_args()
    if a.cmd == "main":
        label = f"main_{a.scenario}{a.k if a.scenario == 'B' else ''}_{'-'.join(a.drop) or 'full'}_{a.year}"
        run_main(a.year, a.scenario, tuple(a.drop), a.k, label)
    elif a.cmd == "ablate":
        run_ablate(a.year)
    else:
        run_final2026()


if __name__ == "__main__":
    main()

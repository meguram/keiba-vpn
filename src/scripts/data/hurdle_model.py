"""障害レース専用モデル（実験）。親モデル v6e16 の確率と合成して、障害内の勝率 LogLoss が良くなるかを leave-one-year-out で測る。

    python -m src.scripts.data.hurdle_model --base /home/jovyan/work/retrain_out/base_2016 --out /tmp/hurdle

方針（2026-11-11 の会議）:
- 障害は別の母集団（上がり約11秒・距離約2,900m）。平地用の列の多くが空になる。ゼロから作る専用化は平地では悪化したが（05j）、
  障害は『親の確率と合成する』形で試す。親の OOF は 2022〜2025 年にしか無いため、障害モデルは 2016〜2025 の他の年で学習し（9 年分）、
  合成の重みだけを 2022〜2025 年の他の年で当てはめる。
- 特徴量は、その日より前の走だけから作る（馬・騎手・調教師の障害の成績、平地の成績、間隔）。市場オッズ・人気は使わない。
- 障害の判定は /tmp/hflag.parquet（race_id, horse_id, year, h）。--hflag で指定する。
評価: レース内で確率を正規化し、勝ち馬の −log p の平均（レース単位）。差はレースを再標本化した 95% 区間つき。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

KEY = ["race_id", "horse_id"]
SEED = 20261111


def _asof_by_day(df: pd.DataFrame, key: str, cols: list[str], prefix: str) -> pd.DataFrame:
    """key ごとに、その日より前（同日を含まない）の cols の累計と走数を付ける。df は date 列を持つ。"""
    d = df[[key, "date"] + cols].copy()
    d["_n"] = 1.0
    day = d.groupby([key, "date"], sort=True)[cols + ["_n"]].sum().reset_index().sort_values([key, "date"])
    cs = day.groupby(key)[cols + ["_n"]].cumsum()
    prior = day[[key, "date"]].copy()
    for c in cols + ["_n"]:
        prior[f"{prefix}_{c}"] = cs[c] - day[c]
    return df[[key, "date"]].merge(prior, on=[key, "date"], how="left").drop(columns=[key, "date"])


def build_features(tt: pd.DataFrame, hflag: pd.DataFrame) -> pd.DataFrame:
    """障害の行ごとの特徴量。tt は train_table（全レース）。"""
    t = tt.merge(hflag[KEY + ["h"]], on=KEY, how="left")
    t["h"] = t["h"].fillna(False).astype(bool)
    t["date"] = pd.to_datetime(t["date"])
    t["win"] = (t["finish_position"] == 1).astype(float)
    t["top3"] = (t["finish_position"] <= 3).astype(float)
    t["fin"] = t["finish_position"].astype(float)
    t = t.sort_values(["date", "race_id"]).reset_index(drop=True)

    t["hw"], t["ht3"], t["hfin"] = t["win"] * t["h"], t["top3"] * t["h"], t["fin"].where(t["h"], 0.0).fillna(0.0)
    t["hn"] = t["h"].astype(float)
    t["fw"], t["fn"] = t["win"] * (~t["h"]), (~t["h"]).astype(float)
    t["ft3"] = t["top3"] * (~t["h"])

    f = pd.DataFrame(index=t.index)
    for key, pre in (("horse_id", "hs"), ("jockey_id", "jk"), ("trainer_id", "tr")):
        a = _asof_by_day(t, key, ["hn", "hw", "ht3", "hfin", "fn", "fw", "ft3"], pre)
        a.index = t.index
        f[f"{pre}_hn"] = a[f"{pre}_hn"]
        f[f"{pre}_hw_rate"] = (a[f"{pre}_hw"] + 0.07 * 20) / (a[f"{pre}_hn"] + 20)           # 障害の勝率（平均7%へ縮約）
        f[f"{pre}_ht3_rate"] = (a[f"{pre}_ht3"] + 0.25 * 20) / (a[f"{pre}_hn"] + 20)
        f[f"{pre}_fn"] = a[f"{pre}_fn"]
        f[f"{pre}_fw_rate"] = (a[f"{pre}_fw"] + 0.07 * 20) / (a[f"{pre}_fn"] + 20)
        if pre == "hs":
            f["hs_hfin_mean"] = a["hs_hfin"] / a["hs_hn"].where(a["hs_hn"] > 0)
            f["hs_ft3_rate"] = a["hs_ft3"] / a["hs_fn"].where(a["hs_fn"] > 0)
    g = t.groupby("horse_id")["date"]
    f["days_since"] = (t["date"] - g.shift(1)).dt.days.astype(float)
    f["last_h"] = t.groupby("horse_id")["h"].shift(1).astype(float)
    f["last_fin"] = t.groupby("horse_id")["fin"].shift(1)
    sa = t["sex_age"].astype(str)
    f["age"] = pd.to_numeric(sa.str.extract(r"(\d+)")[0], errors="coerce")
    f["sex"] = sa.str[0].map({"牡": 0, "牝": 1, "セ": 2}).astype(float)
    for c in ("distance", "field_size", "horse_number", "bracket_number", "jockey_weight", "weight", "weight_change",
              "horse_prior_win_rate", "jockey_prior_win_rate", "trainer_prior_win_rate", "horse_prior_races"):
        f[c] = pd.to_numeric(t[c], errors="coerce") if c in t else np.nan
    f["cond"] = t["track_condition"].map({"良": 0, "稍重": 1, "重": 2, "不良": 3}).astype(float)
    f["venue_code"] = pd.to_numeric(t["venue_code"], errors="coerce") if "venue_code" in t else np.nan
    f["rel_weight"] = f["jockey_weight"] - f.groupby(t["race_id"])["jockey_weight"].transform("mean")
    out = pd.concat([t[KEY + ["date", "h", "win", "fin"]], f], axis=1)
    out["year"] = out["date"].dt.year
    return out[out["h"]].reset_index(drop=True)


def race_logloss(df: pd.DataFrame, p: np.ndarray) -> pd.Series:
    """レースごとの勝ち馬の −log p（レース内で p を正規化）。勝ち馬が 1 頭のレースのみ。"""
    x = df[["race_id", "win"]].copy()
    x["p"] = np.clip(p, 1e-9, None)
    x["p"] = x["p"] / x.groupby("race_id")["p"].transform("sum")
    n_win = x.groupby("race_id")["win"].transform("sum")
    w = x[(x["win"] == 1) & (n_win == 1)]
    return pd.Series(-np.log(w["p"].to_numpy()), index=w["race_id"].to_numpy())


def boot_diff(a: pd.Series, b: pd.Series, n: int = 2000) -> tuple[float, float, float]:
    d = (a - b.reindex(a.index)).dropna().to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


FEATS: list[str] = []


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, kind: str, n_est: int) -> np.ndarray:
    X, y = train[FEATS], train["win"]
    if kind == "lgb":
        import lightgbm as lgb

        m = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.03, num_leaves=7, min_child_samples=80, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=10.0, random_state=SEED, verbose=-1, n_jobs=2)
        m.fit(X, y)
        return m.predict_proba(test[FEATS])[:, 1]
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
    m.fit(X, y)
    return m.predict_proba(test[FEATS])[:, 1]


def fit_blend(lp: np.ndarray, lh: np.ndarray, rid: np.ndarray, win: np.ndarray) -> tuple[float, float]:
    """softmax(a·log p_親 + b·log p_障害) の勝者の条件付き対数尤度で (a, b) を当てはめる。"""
    groups = []
    for r in pd.unique(rid):
        m = rid == r
        if win[m].sum() == 1:
            groups.append((lp[m], lh[m], int(np.argmax(win[m]))))

    def nll(x):
        t = 0.0
        for a, b, k in groups:
            z = x[0] * a + x[1] * b
            z = z - z.max()
            t -= z[k] - np.log(np.exp(z).sum())
        return t / len(groups)

    r = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 300})
    return float(r.x[0]), float(r.x[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/jovyan/work/retrain_out/base_2016")
    ap.add_argument("--hflag", default="/tmp/hflag.parquet")
    ap.add_argument("--out", default="/tmp/hurdle")
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    hflag = pd.read_parquet(a.hflag)
    for c in KEY:
        hflag[c] = hflag[c].astype(str)
    tt = pd.read_parquet(base / "features/train_table.parquet")
    for c in KEY:
        tt[c] = tt[c].astype(str)
    H = build_features(tt, hflag)
    H.to_parquet(out / "hurdle_features.parquet", index=False)
    global FEATS
    FEATS = [c for c in H.columns if c not in KEY + ["date", "h", "win", "fin", "year"]]
    print("障害の行", len(H), "レース", H["race_id"].nunique(), "特徴量", len(FEATS), flush=True)

    fr = pd.read_parquet(base / "final_layer/frame_all.parquet", columns=KEY + ["p_win", "win_odds"])
    for c in KEY:
        fr[c] = fr[c].astype(str)
    H = H.merge(fr, on=KEY, how="left")
    H["q"] = np.where(H["win_odds"] >= 1.0, 1.0 / H["win_odds"], np.nan)

    years = sorted(H["year"].unique())
    res: dict = {"n_rows": int(len(H)), "n_races": int(H["race_id"].nunique()), "feats": FEATS, "configs": {}}
    for kind, n_est in (("lgb", 100), ("lgb", 200), ("lgb", 400), ("lr", 0)):
        name = f"{kind}{n_est or ''}"
        H[f"p_{name}"] = np.nan
        for y in years:
            tr, te = H[H["year"] != y], H[H["year"] == y]
            H.loc[te.index, f"p_{name}"] = fit_predict(tr, te, kind, n_est)
        per_year = {}
        for y in years:
            te = H[H["year"] == y]
            per_year[int(y)] = float(race_logloss(te, te[f"p_{name}"].to_numpy()).mean())
        res["configs"][name] = {"hurdle_only_logloss_by_year": per_year}
        print(name, per_year, flush=True)

    # 親・市場・合成の比較（親がある 2022〜2025 年のみ）
    E = H[H["p_win"].notna()].copy()
    E["lp"] = np.log(np.clip(E["p_win"], 1e-9, None))
    E["lq"] = np.log(np.clip(E["q"], 1e-9, None))
    ey = [y for y in years if E["year"].eq(y).any()]
    base_ll = {y: race_logloss(E[E["year"] == y], E.loc[E["year"] == y, "p_win"].to_numpy()) for y in ey}
    mk_ll = {y: race_logloss(E[E["year"] == y], E.loc[E["year"] == y, "q"].fillna(E["q"].median()).to_numpy()) for y in ey}
    for name in res["configs"]:
        E["lh"] = np.log(np.clip(E[f"p_{name}"], 1e-9, None))
        blend_ll, ww = {}, {}
        for y in ey:
            tr, te = E[E["year"] != y], E[E["year"] == y]
            aw, bw = fit_blend(tr["lp"].to_numpy(), tr["lh"].to_numpy(), tr["race_id"].to_numpy(), tr["win"].to_numpy())
            z = aw * te["lp"] + bw * te["lh"]
            blend_ll[y] = race_logloss(te, np.exp(z - z.max()).to_numpy())
            ww[int(y)] = [aw, bw]
        all_b = pd.concat(blend_ll.values())
        all_p = pd.concat(base_ll.values())
        all_h = pd.concat([race_logloss(E[E["year"] == y], E.loc[E["year"] == y, f"p_{name}"].to_numpy()) for y in ey])
        d, lo, hi = boot_diff(all_b, all_p)
        res["configs"][name].update({
            "parent_logloss": float(all_p.mean()), "hurdle_only_logloss_2022_25": float(all_h.mean()), "blend_logloss": float(all_b.mean()),
            "blend_minus_parent": {"mean": d, "lo": lo, "hi": hi},
            "blend_minus_parent_by_year": {int(y): float((blend_ll[y] - base_ll[y].reindex(blend_ll[y].index)).mean()) for y in ey},
            "weights_by_year(parent,hurdle)": ww,
            "races": int(len(all_p)),
        })
        print(name, json.dumps(res["configs"][name], ensure_ascii=False, default=float)[:600], flush=True)
    res["market_logloss_2022_25"] = float(pd.concat(mk_ll.values()).mean())
    (out / "hurdle_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    H.to_parquet(out / "hurdle_preds.parquet", index=False)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()

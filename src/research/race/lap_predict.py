"""実測の馬別ラップを、結果表の情報から予測できるかを検証する（特徴量の群ごとの寄与・年またぎも見る）。

目的変数: 各馬のラスト 5 区間（L1=最終 … L5）のラップ（現行のスケール推定との差を予測し、L1〜L3 の合計は上がり3F に合わせる）。
特徴量群:
  base  距離・芝ダ・頭数・着順・レースラップ・上がり3F/走破タイムとレースの差
  pos0  従来の位置情報（最初/最後の通過順位・n_corner）
  posC  C1〜C4（最終コーナーを C4 として後ろ詰め。無いコーナーは直後の値で埋め、has_Ck を付ける）の頭数比
  posD  C4-C1・C2-C1・C3-C2・C4-C3・着順-C4（頭数比と実順位）・位置の最小/最大/ばらつき
  cond  馬場状態（順序・ダミー・ダートとの交互作用）
  cls   クラス（grade の順序）・年齢区分
  wt    斤量・レース内の斤量差
  rel   レース内の上がり3F/前半の順位・上がり3F/走破タイムの最速馬との差
比較    : 現行(スケール) / ridge / LightGBM / 平均。レース単位の交差検証と、年またぎ（学習と検証の年を分ける）。

python -m src.research.race.lap_predict --laps /home/jovyan/acq/horse_lap/horse_lap_2026.jsonl ... --years 2023 2026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.race import running_profile as rp

K = 5
GRADE_ORD = {"新馬": 0, "未勝利": 1, "1勝": 2, "2勝": 3, "3勝": 4, "OP": 5, "L": 5.5, "G3": 6, "G2": 7, "G1": 8}
COND_ORD = {"良": 0, "稍重": 1, "重": 2, "不良": 3}

GROUPS: dict[str, list[str]] = {
    "base": ["dist", "surf", "fs", "fin_pct", "h3f_gap", "early_gap", "h3f_rel",
             "RL1", "RL2", "RL3", "RL4", "RL5", "race3f"],
    "pos0": ["n_corner", "first_pct", "last_pct", "gain4", "run_up"],
    "posC": ["C1", "C2", "C3", "C4", "has_C1", "has_C2", "has_C3", "n_corner"],
    "posD": ["d41", "d21", "d32", "d43", "dfin4", "r41", "r32", "r43", "rfin4", "pos_min", "pos_max", "pos_std"],
    "cond": ["cond_ord", "cond_yaya", "cond_omo", "cond_furyo", "cond_x_surf"],
    "cls": ["grade_ord", "age2", "age3", "age34", "age4"],
    "wt": ["jw", "jw_rel"],
    "rel": ["h3f_rank", "h3f_margin", "time_margin", "early_rank"],
}


def feats(*groups: str) -> list[str]:
    out: list[str] = []
    for g in groups:
        out += [f for f in GROUPS[g] if f not in out]
    return out


def measured(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            for e in r.get("entries", []):
                laps = [s["lap_time"] for s in e.get("sections", [])]
                if not e.get("finish_order") or len(laps) < 7 or not all(v and v > 0 for v in laps):
                    continue
                row = {"race_id": str(r["race_id"]), "horse_id": str(e["horse_id"]), "n_sec": len(laps)}
                for k in range(1, K + 1):
                    row[f"y{k}"] = laps[-k]
                rows.append(row)
    return pd.DataFrame(rows)


def parse_order(s: object) -> list[int]:
    if not isinstance(s, str):
        return []
    return [int(x) for x in s.replace("→", "-").split("-") if x.strip().isdigit()]


def extra_cols(years: list[int]) -> pd.DataFrame:
    parts = []
    for y in years:
        p = rp.FLAT_DIR / str(y) / "race_result_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["race_id", "horse_id", "grade", "race_class", "jockey_weight"]))
    e = pd.concat(parts, ignore_index=True)
    e["race_id"], e["horse_id"] = e["race_id"].astype(str), e["horse_id"].astype(str)
    return e.drop_duplicates(["race_id", "horse_id"])


def build(years: list[int], m: pd.DataFrame | None) -> pd.DataFrame:
    runs = rp.load_runs(years)
    runs["race_id"] = runs["race_id"].astype(str)
    runs["horse_id"] = runs["horse_id"].astype(str)
    runs = runs.merge(extra_cols(years), on=["race_id", "horse_id"], how="left")
    runs["laps"] = [rp._laps(s) for s in runs["lap_times"]]
    runs = runs[[len(x) >= 6 and all(v > 0 for v in x) for x in runs["laps"]]].dropna(subset=["last_3f", "time_sec"]).copy()
    if m is not None:
        runs = runs.merge(m, on=["race_id", "horse_id"], how="inner")
    runs = runs.reset_index(drop=True)
    L = np.array([x[-K:][::-1] for x in runs["laps"]])        # L1..L5
    for k in range(K):
        runs[f"RL{k + 1}"] = L[:, k]
    runs["race3f"] = L[:, :3].sum(1)
    runs["race_time"] = [sum(x) for x in runs["laps"]]
    runs["dist"] = runs["distance"].astype(float)
    runs["surf"] = (runs["surface"] == "ダート").astype(float)
    runs["fs"] = runs["field_size"].astype(float)
    runs["fin_pct"] = runs["finish_position"] / runs["fs"]
    runs["h3f_gap"] = runs["last_3f"] - runs["race3f"]
    runs["h3f_rel"] = runs["last_3f"] / runs["race3f"]
    runs["early_gap"] = (runs["time_sec"] - runs["last_3f"]) - (runs["race_time"] - runs["race3f"])

    # 通過順位。最終コーナーを C4 として後ろ詰めにし、無い手前のコーナーは直後の値で埋める（has_Ck で区別）
    po = [parse_order(s) for s in runs["passing_order"]]
    runs["n_corner"] = [len(x) for x in po]
    c = np.full((len(runs), 4), np.nan)
    for i, x in enumerate(po):
        for j in range(min(4, len(x))):
            c[i, 3 - j] = x[-1 - j]
    for j in (2, 1, 0):
        c[:, j] = np.where(np.isnan(c[:, j]), c[:, j + 1], c[:, j])
    fs = runs["fs"].to_numpy()
    for j in range(4):
        runs[f"R{j + 1}"] = c[:, j]                           # 実順位
        runs[f"C{j + 1}"] = c[:, j] / fs                      # 頭数比
    for j in (1, 2, 3):
        runs[f"has_C{j}"] = (runs["n_corner"] >= 5 - j).astype(float)
    runs["first_pct"], runs["last_pct"] = runs["C1"], runs["C4"]
    runs["gain4"] = runs["C4"] - runs["fin_pct"]
    runs["run_up"] = runs["C1"] - runs["C4"]
    runs["d41"], runs["d21"], runs["d32"], runs["d43"] = runs["C4"] - runs["C1"], runs["C2"] - runs["C1"], runs["C3"] - runs["C2"], runs["C4"] - runs["C3"]
    runs["dfin4"] = runs["fin_pct"] - runs["C4"]
    runs["r41"], runs["r32"], runs["r43"] = runs["R4"] - runs["R1"], runs["R3"] - runs["R2"], runs["R4"] - runs["R3"]
    runs["rfin4"] = runs["finish_position"] - runs["R4"]
    cc = runs[["C1", "C2", "C3", "C4"]]
    runs["pos_min"], runs["pos_max"], runs["pos_std"] = cc.min(axis=1), cc.max(axis=1), cc.std(axis=1)

    # 馬場状態・クラス・斤量
    runs["cond_ord"] = runs["track_condition"].map(COND_ORD)
    for name, v in (("yaya", 1), ("omo", 2), ("furyo", 3)):
        runs[f"cond_{name}"] = (runs["cond_ord"] == v).astype(float)
    runs["cond_x_surf"] = runs["cond_ord"] * runs["surf"]
    runs["grade_ord"] = runs["grade"].map(GRADE_ORD)
    rc = runs["race_class"].fillna("")
    runs["age2"] = rc.str.contains("２歳").astype(float)
    runs["age34"] = rc.str.contains("３歳以上").astype(float)
    runs["age3"] = (rc.str.contains("３歳") & ~rc.str.contains("３歳以上")).astype(float)
    runs["age4"] = rc.str.contains("４歳以上").astype(float)
    runs["jw"] = pd.to_numeric(runs["jockey_weight"], errors="coerce")
    runs["jw_rel"] = runs["jw"] - runs.groupby("race_id")["jw"].transform("mean")

    # レース内の相対
    g = runs.groupby("race_id")
    runs["early_t"] = runs["time_sec"] - runs["last_3f"]
    runs["h3f_rank"] = g["last_3f"].rank(method="average") / runs["fs"]
    runs["early_rank"] = g["early_t"].rank(method="average") / runs["fs"]
    runs["h3f_margin"] = runs["last_3f"] - g["last_3f"].transform("min")
    runs["time_margin"] = runs["time_sec"] - g["time_sec"].transform("min")
    runs["year"] = pd.to_datetime(runs["date"]).dt.year
    return runs


def baseline(d: pd.DataFrame) -> np.ndarray:
    """現行: L1〜L3 はレースラップ × (馬の上がり3F / レースの上がり3F)、L4・L5 はレースラップのまま。"""
    out = np.zeros((len(d), K))
    for k in range(K):
        out[:, k] = d[f"RL{k + 1}"] * (d["h3f_rel"] if k < 3 else 1.0)
    return out


def targets(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack([d[f"y{k + 1}"] - baseline(d)[:, k] for k in range(K)])


def make_model(kind: str, alpha: float = 10.0):
    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    if kind == "ridge":
        return Ridge(alpha=alpha)
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=8, min_child_samples=30,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def fit_predict_idx(d: pd.DataFrame, cols: list[str], kind: str, tr: np.ndarray, te: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    """tr で学習し te の L1〜L5 推定（L1〜L3 は合計を上がり3Fに合わせる）を返す。ridge は欠損を学習側の中央値で埋めて標準化する。"""
    X = d[cols].to_numpy(dtype=float)
    Y = targets(d)
    if kind == "ridge":
        med = np.nanmedian(X[tr], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        X = np.where(np.isnan(X), med, X)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        X = (X - mu) / sd
    pred = np.zeros((len(te), K))
    for k in range(K):
        pred[:, k] = make_model(kind, alpha).fit(X[tr], Y[tr, k]).predict(X[te])
    sub = d.iloc[te]
    out = baseline(sub) + pred
    adj = (sub["last_3f"].to_numpy() - out[:, :3].sum(1)) / 3.0
    out[:, :3] += adj[:, None]
    return out


def cv_predict(d: pd.DataFrame, cols: list[str], kind: str, folds: int = 5, alpha: float = 10.0) -> np.ndarray:
    from sklearn.model_selection import GroupKFold

    out = np.zeros((len(d), K))
    for tr, te in GroupKFold(folds).split(d, groups=d["race_id"]):
        out[te] = fit_predict_idx(d, cols, kind, tr, te, alpha)
    return out


def score(d: pd.DataFrame, est: np.ndarray) -> dict:
    y = np.column_stack([d[f"y{k + 1}"] for k in range(K)])
    res: dict[str, float] = {}
    for k in range(K):
        res[f"L{k + 1}"] = float(np.sqrt(((est[:, k] - y[:, k]) ** 2).mean()))
    peak_t, peak_e = y[:, :3].min(1), est[:, :3].min(1)
    w = lambda v: v - pd.Series(v).groupby(d["race_id"].to_numpy()).transform("mean").to_numpy()
    res["峰RMSE"] = float(np.sqrt(((peak_e - peak_t) ** 2).mean()))
    res["峰相関(レース内)"] = float(np.corrcoef(w(peak_e), w(peak_t))[0, 1])
    res["峰位置一致"] = float((y[:, :3].argmin(1) == est[:, :3].argmin(1)).mean())
    res["加速度相関"] = float(np.corrcoef((y[:, 4] - peak_t) / y[:, 4], (est[:, 4] - peak_e) / est[:, 4])[0, 1])
    res["L1/峰相関"] = float(np.corrcoef(y[:, 0] / peak_t, est[:, 0] / peak_e)[0, 1])
    return res


def table(rows: dict[str, dict]) -> str:
    return pd.DataFrame(rows).T.round(3).to_string()


def importances(d: pd.DataFrame, cols: list[str]) -> None:
    """ALL の LightGBM（gain 重要度、5 目的の平均）と ridge の L1 への標準化係数。"""
    X = d[cols].to_numpy(dtype=float)
    Y = targets(d)
    imp = np.zeros(len(cols))
    for k in range(K):
        m = make_model("lgbm").fit(X, Y[:, k])
        g = m.booster_.feature_importance("gain")
        imp += g / g.sum()
    s = pd.Series(imp / K, index=cols).sort_values(ascending=False)
    print("\n== LightGBM gain 重要度（上位 15, 5 目的の平均）")
    print(s.head(15).round(3).to_string())
    Xs = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    for k in (0, 2):
        c = pd.Series(make_model("ridge").fit(Xs, Y[:, k]).coef_, index=cols)
        print(f"\n== ridge 標準化係数 L{k + 1}（絶対値の上位 10）")
        print(c.reindex(c.abs().sort_values(ascending=False).index).head(10).round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", nargs="+", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    a = ap.parse_args()
    m = measured([Path(p) for p in a.laps])
    d = build(a.years, m)
    gap = (d[[f"y{k}" for k in (1, 2, 3)]].sum(1) - d["last_3f"]).abs()
    print(f"馬 {len(d)} 頭・レース {d['race_id'].nunique()}  実測L1〜L3の合計と上がり3Fの差: 平均 {gap.mean():.3f} / 0.15超 {(gap > 0.15).mean():.3f}")
    d = d[gap <= 0.15].reset_index(drop=True)
    print("年別の頭数:", d.groupby("year").size().to_dict(), " n_corner の分布:", d["n_corner"].value_counts().sort_index().to_dict())
    print("欠損率(%):", (d[feats("cond", "cls", "wt")].isna().mean() * 100).round(1)[lambda s: s > 0].to_dict())

    sets: dict[str, list[str]] = {
        "①base（位置なし）": feats("base"),
        "②prev=base+pos0": feats("base", "pos0"),
        "③base+posC": feats("base", "posC"),
        "④base+posC+posD": feats("base", "posC", "posD"),
        "⑤base+cond": feats("base", "cond"),
        "⑥base+cls": feats("base", "cls"),
        "⑦base+wt": feats("base", "wt"),
        "⑧base+rel": feats("base", "rel"),
        "ALL": feats("base", "posC", "posD", "cond", "cls", "wt", "rel"),
        "ALL−pos(C+D)": feats("base", "cond", "cls", "wt", "rel"),
        "ALL−posD": feats("base", "posC", "cond", "cls", "wt", "rel"),
        "ALL−cond": feats("base", "posC", "posD", "cls", "wt", "rel"),
        "ALL−cls": feats("base", "posC", "posD", "cond", "wt", "rel"),
        "ALL−wt": feats("base", "posC", "posD", "cond", "cls", "rel"),
        "ALL−rel": feats("base", "posC", "posD", "cond", "cls", "wt"),
    }
    base_rows = {"現行(スケール)": score(d, baseline(d))}
    print("\n== 群ごとの寄与（レース単位 5 分割 CV）")
    cache: dict[tuple[str, str], np.ndarray] = {}
    for kind in ("ridge", "lgbm"):
        rows = dict(base_rows)
        for name, cols in sets.items():
            cache[(kind, name)] = cv_predict(d, cols, kind)
            rows[name] = score(d, cache[(kind, name)])
        print(f"\n-- {kind}")
        print(table(rows))
    rows = dict(base_rows)
    for name in ("②prev=base+pos0", "ALL"):
        rows[f"平均(ridge+lgbm) {name}"] = score(d, (cache[("ridge", name)] + cache[("lgbm", name)]) / 2)
    print("\n-- 平均")
    print(table(rows))

    allc = sets["ALL"]
    print("\n== ridge の alpha（ALL）")
    rows = {}
    for al in (1.0, 10.0, 100.0, 300.0):
        rows[f"alpha={al:g}"] = score(d, cv_predict(d, allc, "ridge", alpha=al))
    print(table(rows))

    print("\n== 部分集合ごと（ALL ridge / ALL lgbm / ②prev ridge / 現行）  峰RMSE・峰相関(レース内)・峰位置一致・L1/峰相関")
    subs = {"芝": d["surf"] == 0, "ダート": d["surf"] == 1, "≤1400m": d["dist"] <= 1400, ">1400m": d["dist"] > 1400,
            "良": d["cond_ord"] == 0, "稍重〜不良": d["cond_ord"] > 0}
    keys = ["峰RMSE", "峰相関(レース内)", "峰位置一致", "L1/峰相関"]
    rows = {}
    for sname, mask in subs.items():
        idx = mask.to_numpy()
        sub = d[idx].reset_index(drop=True)
        for label, est in (("現行", baseline(d)), ("②prev ridge", cache[("ridge", "②prev=base+pos0")]),
                           ("ALL ridge", cache[("ridge", "ALL")]), ("ALL lgbm", cache[("lgbm", "ALL")])):
            r = score(sub, est[idx])
            rows[f"{sname} n={idx.sum()} {label}"] = {k: r[k] for k in keys}
    print(table(rows))

    years = sorted(d["year"].unique())
    if len(years) >= 2:
        print("\n== 年またぎ（学習年→検証年。クラスは年で構成が違う点に注意）")
        rows = {}
        for ytr in years:
            for yte in years:
                if ytr == yte:
                    continue
                tr, te = np.where(d["year"] == ytr)[0], np.where(d["year"] == yte)[0]
                sub = d.iloc[te].reset_index(drop=True)
                rows[f"{ytr}→{yte} n={len(te)} 現行"] = {k: score(sub, baseline(sub))[k] for k in keys}
                for kind in ("ridge", "lgbm"):
                    for name in ("②prev=base+pos0", "ALL", "ALL−cls"):
                        est = fit_predict_idx(d, sets[name], kind, tr, te)
                        rows[f"{ytr}→{yte} {kind} {name}"] = {k: score(sub, est)[k] for k in keys}
        print(table(rows))

    importances(d, allc)


if __name__ == "__main__":
    main()

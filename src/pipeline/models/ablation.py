"""tp-03 特徴量 ablation の部品（学習用テーブルの読み込み、LightGBM の時系列 CV 評価、採否の判定）。

評価は固定条件で行う: LightGBM（n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
subsample=0.8, colsample_bytree=0.7, reg_lambda=5）、valid 2022 / 2023 / 2024 の時系列 CV（purge 7 日 + embargo 3 日）、
複数 seed。主指標は p_win の LogLoss。オッズ・人気は使わない。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import polars as pl

# ── B0（ベースライン）: LayerA の既存列。ID・名前・ラベル・履歴特徴量は含めない
B0_CATEGORICAL = ["surface", "direction", "grade", "race_class", "weather", "track_condition", "venue_code", "weight_rule",
                  "course_type", "sex", "track_bias_cond_bucket", "track_bias_dist_band"]
B0_RACE_NUMERIC = ["distance", "field_size", "round", "horse_number", "bracket_number", "jockey_weight", "weight", "weight_change", "age", "start_hour"]
B0_PRIOR = [f"{p}_prior_{s}" for p in ("jockey", "trainer") for s in ("races", "wins", "top3", "win_rate", "top3_rate")] + \
           [f"horse_prior_{s}" for s in ("races", "wins", "top3", "win_rate", "top3_rate", "avg_finish")]
B0_TRACK_BIAS = ["track_bias_winner_3f_same_day_prior", "track_bias_winner_3f_prev_day", "track_bias_winner_3f_diff", "track_bias_winner_3f_weighted"]
B0_SIRE = ["sire_prior_win_rate_surface", "dam_sire_prior_win_rate_surface", "sire_x_track_bias_diff", "dam_sire_x_track_bias_diff",
           "sire_x_track_bias_weighted", "dam_sire_x_track_bias_weighted"]
B0_SPEED = ["speed_max", "speed_avg", "speed_distance", "speed_course", "speed_recent_1", "speed_recent_2", "speed_recent_3", "has_speed_index"]

# 時代を見分ける上位の特徴量（クラス・相手関係）。年内標準化の対象
ERA_CLASS_COLS = ["c_rq_ewm2", "c_cls_ewm2", "c_cls_step", "c_fs_ewm2", "c_fs_step", "c_fstop3_ewm2", "c_standing_ewm2", "c_opp_step",
                  "f3_fs_mean", "f3_fs_top3", "f3_fs_std", "f3_r"]
# 騎手・調教師の通算件数（LayerA は 2020-01 からの累積）。時間とともに単調に増えるため、時代を完全に見分ける（AUC=1.0）
B0_TIME_COUNTS = [f"{p}_prior_{s}" for p in ("jockey", "trainer") for s in ("races", "wins", "top3")]
# 件数系（時間とともに増える）
COUNT_COLS = ["d_career_days", "d_nprior", "e_n_l3f", "b_n_sd", "b_n_venue", "b_n_going", "b_n_dir", "b_n_course", "b_n_class", "f3_n_pre",
              "sr_n", "sr_var_n", "f2_n_reg0", "f2_n_reg1", "f2_n_reg2", "pc_n_reg0", "pc_n_reg1", "pc_n_reg2", "f1_sw", "f1_neff"]

LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100, "bagging_fraction": 0.8, "bagging_freq": 1,
    "feature_fraction": 0.7, "lambda_l2": 5.0, "verbosity": -1, "deterministic": True, "force_row_wise": True,
}
N_ROUNDS = 300
TARGETS = {"win": "is_win", "top2": "is_top2", "top3": "is_top3"}


def b0_columns(include_speed: bool = True) -> list[str]:
    cols = B0_CATEGORICAL + B0_RACE_NUMERIC + B0_PRIOR + B0_TRACK_BIAS + B0_SIRE
    return cols + (B0_SPEED if include_speed else [])


def load_table(path: str | Path) -> pd.DataFrame:
    """学習用テーブルを読み、評価に必要な列（ラベル・年・性・年齢・発走時）を足した pandas を返す。"""
    df = pl.read_parquet(path).to_pandas()
    df = df[df["finish_position"] > 0].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype(int)
    df["sex"] = df["sex_age"].astype(str).str[:1]
    df["age"] = pd.to_numeric(df["sex_age"].astype(str).str[1:], errors="coerce").astype("float32")
    df["start_hour"] = pd.to_numeric(df["start_time"].astype(str).str.extract(r"^(\d{1,2})")[0], errors="coerce").astype("float32")
    for name, col in TARGETS.items():
        n = {"win": 1, "top2": 2, "top3": 3}[name]
        df[col] = (df["finish_position"] <= n).astype("int8")
    df["has_speed_index"] = df["has_speed_index"].astype("int8")
    for c in B0_CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("(なし)").astype("category")
    return df


# ── 年内標準化（as-of）: 直前 window_days 日のレースだけで平均・標準偏差を作る
def asof_standardize(df: pd.DataFrame, cols: Sequence[str], window_days: int = 365, min_n: int = 2000) -> pd.DataFrame:
    """各行の日付より前 window_days 日の行で z 化した列を返す（同日以降は使わない）。窓の行数が min_n 未満なら欠損。"""
    out = pd.DataFrame(index=df.index)
    base = pl.from_pandas(df[["date"] + list(cols)].reset_index(names="_i")).with_columns(pl.col("date").cast(pl.Date))
    days = base.select("date").unique().sort("date")
    for c in cols:
        x = pl.col(c).cast(pl.Float64)
        daily = (base.group_by("date").agg(x.sum().alias("s"), (x ** 2).sum().alias("q"), x.is_not_null().sum().alias("n"))
                 .sort("date").with_columns(pl.col("s").cum_sum().alias("S"), pl.col("q").cum_sum().alias("Q"), pl.col("n").cum_sum().alias("N")))
        cum = daily.select("date", "S", "Q", "N")
        d1 = days.join_asof(cum, on="date", strategy="backward", allow_exact_matches=False).rename({"S": "S1", "Q": "Q1", "N": "N1"})
        d0 = days.with_columns(pl.col("date").dt.offset_by(f"-{window_days}d").alias("d0")).sort("d0").join_asof(
            cum.rename({"date": "d0"}), on="d0", strategy="backward", allow_exact_matches=False).sort("date").select("date", "S", "Q", "N")
        w = d1.join(d0.rename({"S": "S0", "Q": "Q0", "N": "N0"}), on="date").with_columns(
            (pl.col("N1").fill_null(0) - pl.col("N0").fill_null(0)).alias("n"),
            (pl.col("S1").fill_null(0) - pl.col("S0").fill_null(0)).alias("s"),
            (pl.col("Q1").fill_null(0) - pl.col("Q0").fill_null(0)).alias("q"))
        w = w.with_columns((pl.col("s") / pl.col("n")).alias("mu")).with_columns(
            ((pl.col("q") / pl.col("n") - pl.col("mu") ** 2).clip(lower_bound=1e-12)).sqrt().alias("sd")).select(
            "date", "mu", "sd", "n")
        j = base.select("_i", "date", x.alias("x")).join(w, on="date", how="left").sort("_i")
        z = pl.when(pl.col("n") >= min_n).then((pl.col("x") - pl.col("mu")) / pl.col("sd")).otherwise(None)
        out[c] = j.select(z.cast(pl.Float32).alias("z"))["z"].to_numpy()
    return out


# ── 評価
def race_nll(race_ids: np.ndarray, score: np.ndarray, label: np.ndarray) -> float:
    """レース内 softmax の勝ち馬の負の対数尤度の平均（勝ち馬がちょうど 1 頭のレースだけ）。"""
    d = pd.DataFrame({"r": race_ids, "s": score, "y": label})
    d["m"] = d.groupby("r")["s"].transform("max")
    d["e"] = np.exp(d["s"] - d["m"])
    d["z"] = d.groupby("r")["e"].transform("sum")
    d["lp"] = (d["s"] - d["m"]) - np.log(d["z"])
    w = d[d["y"] == 1]
    cnt = w.groupby("r").size()
    w = w[w["r"].isin(cnt[cnt == 1].index)]
    return float(-w["lp"].mean())


def metrics(race_ids: np.ndarray, raw: np.ndarray, y: np.ndarray, with_race_nll: bool = True) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score

    p = 1.0 / (1.0 + np.exp(-raw))
    m = {"logloss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))), "auc": float(roc_auc_score(y, raw))}
    if with_race_nll:
        m["race_nll"] = race_nll(race_ids, raw, y)
    return m


@dataclass
class Spec:
    """1 つの実験の定義。"""
    name: str
    feats: list[str]
    min_train_year: int | None = None          # この年より前の行を学習から外す
    weights: dict[int, float] | None = None    # 年 → 学習時の重み
    replace: dict[str, pd.Series] | None = None  # 差し替える特徴量（年内標準化など）
    note: str = ""

    def key(self) -> str:
        h = hashlib.md5(json.dumps([self.feats, self.min_train_year, self.weights, sorted((self.replace or {}).keys())], default=str).encode()).hexdigest()[:10]
        return f"{self.name}|{h}"


def build_x(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """実験の特徴量行列。差し替え（年内標準化など）があれば、その列だけ置き換える。"""
    X = df[spec.feats].copy()
    for k, v in (spec.replace or {}).items():
        if k in X.columns:
            X[k] = np.asarray(v)
    return X


def evaluate(df: pd.DataFrame, folds: Iterable, spec: Spec, seeds: Sequence[int], threads: int, targets: Sequence[str] = ("win",),
             with_race_nll: bool = True) -> list[dict]:
    """spec を fold × target × seed で評価する。1 つの fold・target では Dataset（ビン分割）を seed 間で使い回す。"""
    import lightgbm as lgb

    X = build_x(df, spec)
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    years = df["year"].to_numpy()
    race_ids = df["race_id"].to_numpy()
    rows = []
    for fold in folds:
        tr = fold.train_index
        if spec.min_train_year is not None:
            tr = tr[years[tr] >= spec.min_train_year]
            if len(tr) == 0:
                continue  # この fold は学習できない（例: 2022 の fold は 2020〜2021 年だけで学習する）
        w = np.array([spec.weights.get(int(y), 1.0) for y in years[tr]], dtype="float64") if spec.weights else None
        Xtr, Xva = X.iloc[tr], X.iloc[fold.valid_index]
        for target in targets:
            y = df[TARGETS[target]].to_numpy()
            ds = lgb.Dataset(Xtr, label=y[tr], weight=w, categorical_feature=cats or "auto", free_raw_data=False)
            for seed in seeds:
                booster = lgb.train(dict(LGB_PARAMS, seed=seed, num_threads=threads), ds, num_boost_round=N_ROUNDS)
                raw = booster.predict(Xva, raw_score=True)
                m = metrics(race_ids[fold.valid_index], raw, y[fold.valid_index], with_race_nll and target == "win")
                rows.append({"exp": spec.name, "key": spec.key(), "fold": fold.fold_id, "seed": seed, "target": target,
                             "n_train": int(len(tr)), "n_valid": int(len(fold.valid_index)), **m})
    return rows


# ── 集計と採否
def summarize(results: pd.DataFrame, target: str = "win", metric: str = "logloss") -> pd.DataFrame:
    """実験ごとに fold 別の平均（seed 平均）と全体平均・seed 間のばらつきを出す。"""
    r = results[results["target"] == target]
    per = r.groupby(["exp", "fold"])[metric].mean().unstack("fold")
    out = per.copy()
    out["mean"] = per.mean(axis=1)
    seed_sd = r.groupby(["exp", "seed"])[metric].mean().groupby("exp").std()
    out["seed_sd"] = seed_sd.reindex(out.index)
    return out


def compare_to_base(summary: pd.DataFrame, base: str, min_gain: float = 0.0001) -> pd.DataFrame:
    """base との差（負なら改善）。共通の fold だけで比べる。採否の基準:
    共通の全 fold で悪化せず、平均が改善し、改善幅が max(min_gain, 2×seed 間のばらつき) 以上。"""
    folds = [c for c in summary.columns if str(c).startswith("valid_")]
    rows = []
    for exp, row in summary.iterrows():
        if exp == base:
            continue
        common = [f for f in folds if pd.notna(row[f]) and pd.notna(summary.loc[base, f])]
        if not common:
            continue
        d = {f: float(row[f] - summary.loc[base, f]) for f in common}
        mean = float(np.mean(list(d.values())))
        sd = float(np.nanmax([row.get("seed_sd", np.nan), summary.loc[base, "seed_sd"]]))
        thr = max(min_gain, 2 * (0.0 if np.isnan(sd) else sd))
        rows.append({"exp": exp, **{f"Δ{f[-4:]}": d[f] for f in common}, "Δ平均": mean, "共通fold数": len(common),
                     "全foldで改善": all(v < 0 for v in d.values()), "採用基準を満たす": all(v < 0 for v in d.values()) and mean <= -thr})
    return pd.DataFrame(rows).set_index("exp").sort_values("Δ平均")


def drop_correlated(df: pd.DataFrame, feats: list[str], order: list[str], threshold: float = 0.9, sample: int = 60000, seed: int = 0) -> tuple[list[str], dict]:
    """重要度順（order）に特徴量を見て、すでに残した特徴量と |Spearman| >= threshold のものを落とす。落とした理由を返す。"""
    num = [f for f in order if f in feats and str(df[f].dtype) != "category"]
    cat = [f for f in feats if f not in num]
    s = df[num].sample(min(sample, len(df)), random_state=seed).rank(pct=True)
    corr = s.corr().abs()
    kept, dropped = [], {}
    for f in num:
        hit = next((k for k in kept if corr.loc[f, k] >= threshold), None)
        if hit is None:
            kept.append(f)
        else:
            dropped[f] = {"代表": hit, "相関": round(float(corr.loc[f, hit]), 3)}
    return kept + cat, dropped


def adversarial_auc(df: pd.DataFrame, feats: list[str], mask_a: np.ndarray, mask_b: np.ndarray, threads: int, seed: int = 0) -> tuple[float, list[tuple[str, float]]]:
    """a と b を見分けられるか（AUC が高いほど分布が違う）。レース単位で 7:3 に分け、同じレースが両方に入らないようにする。"""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    idx = np.where(mask_a | mask_b)[0]
    y = mask_b[idx].astype(int)
    races = df["race_id"].to_numpy()[idx]
    rng = np.random.default_rng(seed)
    uniq = np.unique(races)
    test_races = set(rng.choice(uniq, size=int(len(uniq) * 0.3), replace=False))
    te = np.array([r in test_races for r in races])
    X = df[feats].iloc[idx]
    cats = [c for c in feats if str(X[c].dtype) == "category"]
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 100, "feature_fraction": 0.7,
              "verbosity": -1, "num_threads": threads, "seed": seed}
    booster = lgb.train(params, lgb.Dataset(X[~te], label=y[~te], categorical_feature=cats or "auto"), num_boost_round=150)
    auc = float(roc_auc_score(y[te], booster.predict(X[te])))
    imp = sorted(zip(feats, booster.feature_importance("gain")), key=lambda t: -t[1])[:8]
    return auc, [(f, float(g)) for f, g in imp]

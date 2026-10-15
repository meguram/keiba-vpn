"""tp-04 前段モデル（位置取り・ペース・上がり・時計の予測）の部品。

各段は時系列の OOF（学習に使っていない年の予測）で出す。
- expanding: 年 Y の予測は、Y の開始 10 日前（purge 7 日 + embargo 3 日）より前のデータだけで学習したモデルで出す（本番と同じ向き）。
- loyo: 年 Y の予測は、Y 以外の年で学習したモデルで出す（学習行は増えるが、将来のデータで学習する軽い先読みを含む。上限の目安）。
入力はすべて as-of の特徴量。目的変数（そのレースの結果）は学習のラベルにだけ使う。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

GAP_DAYS = 10  # purge 7 日 + embargo 3 日
REG_PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100, "bagging_fraction": 0.8,
              "bagging_freq": 1, "feature_fraction": 0.7, "lambda_l2": 5.0, "verbosity": -1, "deterministic": True, "force_row_wise": True}
N_ROUNDS = 300
FRONT_THR = 0.25  # 予測位置（0=先頭,1=最後尾）がこれ未満なら「先行」とみなす


# ── ラベル（結果由来。学習専用）
def stage_labels(layer_a_path: str | Path, on_time: pl.DataFrame | None = None) -> pd.DataFrame:
    """(race_id, horse_id) ごとの前段のラベル。tp-02 と同じ修復（通過順・上がりの不正値の補完）を通した走行データから作る。
    pos_early/pos_late は 0=先頭〜1=最後尾、l3f は上がり 3F 秒、pace_bal は前半 3F − 後半 3F、rpci は G ファミリーと同じ定義。"""
    from src.pipeline.features.horse_history.common import apply_repair, build_repair_map, load_runs

    runs = load_runs()
    j = apply_repair(runs, build_repair_map(layer_a_path), on_time)
    out = j.select("race_id", "horse_id", "pos_early", "pos_late", pl.col("l3f"), "time_sec", "n_fin", "finish_position",
                   "pace_bal", "rpci", "dist_band", "date", (pl.col("surface") == "障").alias("is_hurdle")).to_pandas()
    out["horse_id"] = out["horse_id"].astype(str)
    out["race_id"] = out["race_id"].astype(str)
    return out


def pace_class_labels(labels: pd.DataFrame, pace_q: dict, rpci_q: dict) -> pd.DataFrame:
    """レース単位のペース 3 区分。閾値は tp-02 が FIT_END 以前だけで決めて fit_params.json に保存した距離帯別の (下側, 上側) 分位をそのまま使う
    （F2・G ファミリーの区分 f2_a_reg0〜2・pc_a_reg0〜2 と同じ区分にそろえるため）。値が下側以下 → 0、上側以上 → 2、それ以外 → 1。
    pace_bal（前半 3F − 後半 3F）は 0=後傾・2=前傾、rpci は 0=前傾・2=後傾。障害は対象外（NA）。"""
    r = labels[~labels["is_hurdle"]].groupby("race_id").agg(pace_bal=("pace_bal", "first"), rpci=("rpci", "first"),
                                                            dist_band=("dist_band", "first")).reset_index()
    for col, q in (("pace_bal", pace_q), ("rpci", rpci_q)):
        v = r[col].to_numpy(dtype=float)
        lo = r["dist_band"].map(lambda b: q[str(int(b))][0]).to_numpy(dtype=float)
        hi = r["dist_band"].map(lambda b: q[str(int(b))][1]).to_numpy(dtype=float)
        cls = np.where(v <= lo, 0.0, np.where(v >= hi, 2.0, 1.0))
        cls[np.isnan(v)] = np.nan
        r[f"{col}_cls"] = cls
    return r[["race_id", "pace_bal", "rpci", "pace_bal_cls", "rpci_cls"]]


def build_targets(df: pd.DataFrame, labels: pd.DataFrame, race_cls: pd.DataFrame) -> pd.DataFrame:
    """df（学習テーブル）の各行に揃えた前段の目的変数。上がり・時計は『そのレース内の平均との差』を基本にする。"""
    key = df[["race_id", "horse_id"]].astype({"race_id": str, "horse_id": str})
    lab = labels.drop_duplicates(["race_id", "horse_id"]).drop(columns=["date", "dist_band", "pace_bal", "rpci"])
    m = key.merge(lab, on=["race_id", "horse_id"], how="left")
    assert len(m) == len(df)
    m.index = df.index
    t = pd.DataFrame(index=df.index)
    t["t_pos_early"] = m["pos_early"]
    t["t_pos_late"] = m["pos_late"]
    t["is_hurdle"] = m["is_hurdle"].fillna(False).astype(bool)
    l3f = m["l3f"].where(~t["is_hurdle"])
    g = df["race_id"]
    cnt = l3f.groupby(g).transform("count")
    t["t_l3f_dev"] = (l3f - l3f.groupby(g).transform("mean")).where(cnt >= 5)
    t["t_l3f_raw"] = l3f
    ts = m["time_sec"].where(m["time_sec"] > 0)
    t["t_time_dev"] = (ts - ts.groupby(g).transform("mean")).where(ts.groupby(g).transform("count") >= 5)
    t["t_rank_norm"] = ((m["finish_position"] - 1) / (m["n_fin"] - 1)).clip(0, 1)
    rc = race_cls.set_index("race_id")
    for c in ("pace_bal", "rpci", "pace_bal_cls", "rpci_cls"):
        t[f"r_{c}"] = df["race_id"].astype(str).map(rc[c]).to_numpy()
    return t


# ── as-of の集計
def asof_group_mean(df: pd.DataFrame, group: str, value: str, k: float = 20.0, name: str | None = None) -> pd.DataFrame:
    """group ごとの value の平均を、その行の日付より前の行だけで作る（経験ベイズで全体平均へ縮約）。
    返す列: ``{name}_mean``（縮約後の平均）と ``{name}_n``（それまでの件数）。同日の行は使わない。"""
    assert isinstance(df.index, pd.RangeIndex), "RangeIndex が必要です"
    name = name or f"{group}_{value}"
    base = pl.from_pandas(df[["date", group, value]].reset_index(names="_i")).with_columns(pl.col("date").cast(pl.Date))
    v = pl.col(value).cast(pl.Float64)
    daily = (base.group_by([group, "date"]).agg(v.sum().alias("s"), v.is_not_null().sum().alias("n"))
             .sort(["date"]).with_columns(pl.col("s").cum_sum().over(group).alias("S"), pl.col("n").cum_sum().over(group).alias("N"))
             .select(group, "date", "S", "N").sort("date"))
    j = base.sort("date").join_asof(daily, on="date", by=group, strategy="backward", allow_exact_matches=False).sort("_i")
    # 縮約の目標（全体平均）も、その日より前の全行だけで作る（未来を混ぜない）。最初の日は欠損
    gd = (base.group_by("date").agg(v.sum().alias("gs"), v.is_not_null().sum().alias("gn")).sort("date")
          .with_columns(pl.col("gs").cum_sum().alias("GS"), pl.col("gn").cum_sum().alias("GN")).select("date", "GS", "GN"))
    g = base.sort("date").join_asof(gd, on="date", strategy="backward", allow_exact_matches=False).sort("_i")
    gm = (g["GS"] / g["GN"]).to_numpy().astype(float)
    n = j["N"].fill_null(0).to_numpy().astype(float)
    s = j["S"].fill_null(0.0).to_numpy().astype(float)
    return pd.DataFrame({f"{name}_mean": ((s + k * gm) / (n + k)).astype("float32"), f"{name}_n": n.astype("float32")}, index=df.index)


# ── 時系列 OOF
def year_splits(df: pd.DataFrame, valid_years: Sequence[int], mode: str) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(年, 学習の位置, 検証の位置)。expanding は年 Y の開始 GAP_DAYS 日前より前、loyo は Y 以外（Y の前後 GAP_DAYS 日は除く）。"""
    years = df["year"].to_numpy()
    dates = df["date"].to_numpy().astype("datetime64[D]")
    out = []
    gap = np.timedelta64(GAP_DAYS, "D")
    for y in valid_years:
        start = np.datetime64(f"{y}-01-01")
        end = np.datetime64(f"{y + 1}-01-01")
        va = np.where(years == y)[0]
        if mode == "expanding":
            tr = np.where(dates < start - gap)[0]
        elif mode == "loyo":
            tr = np.where((years != y) & ~((dates >= start - gap) & (dates < end + gap)))[0]
        else:
            raise ValueError(mode)
        if len(tr) and len(va):
            out.append((int(y), tr, va))
    return out


def oof_regression(df: pd.DataFrame, X: pd.DataFrame, target: np.ndarray, valid_years: Sequence[int], mode: str, threads: int,
                   seed: int = 0, variants: dict[str, dict[str, np.ndarray]] | None = None, n_rounds: int = N_ROUNDS,
                   params: dict | None = None) -> dict[str, np.ndarray]:
    """回帰の OOF 予測。ラベルが欠損の行は学習から外す。
    variants: 名前 → {列名: 全行の配列}。予測時にその列を差し替えた予測を名前ごとに返す（例: ペースを前傾/平均/後傾と仮定）。省略時は {'base': 予測}。"""
    import lightgbm as lgb

    variants = variants or {"base": {}}
    pred = {k: np.full(len(df), np.nan) for k in variants}
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    p = dict(REG_PARAMS, **(params or {}))
    for y, tr, va in year_splits(df, valid_years, mode):
        tr = tr[~np.isnan(target[tr])]
        if len(tr) < 500:
            continue
        booster = lgb.train(dict(p, seed=seed, num_threads=threads), lgb.Dataset(X.iloc[tr], label=target[tr], categorical_feature=cats or "auto"),
                            num_boost_round=n_rounds)
        for name, over in variants.items():
            Xv = X.iloc[va]
            if over:
                Xv = Xv.copy()
                for c, v in over.items():
                    Xv[c] = np.asarray(v)[va]
            pred[name][va] = booster.predict(Xv)
    return pred


def oof_multiclass(Xr: pd.DataFrame, y: np.ndarray, year: np.ndarray, date_: np.ndarray, valid_years: Sequence[int], mode: str, threads: int,
                   n_class: int = 3, seed: int = 0, n_rounds: int = 150) -> np.ndarray:
    """レース単位の多クラス分類の OOF 確率（n×n_class）。学習は小さいので浅く・正則化を強める。"""
    import lightgbm as lgb

    d = pd.DataFrame({"year": year, "date": date_})
    params = {"objective": "multiclass", "num_class": n_class, "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 40, "bagging_fraction": 0.8,
              "bagging_freq": 1, "feature_fraction": 0.7, "lambda_l2": 10.0, "verbosity": -1, "deterministic": True, "force_row_wise": True,
              "seed": seed, "num_threads": threads}
    P = np.full((len(Xr), n_class), np.nan)
    cats = [c for c in Xr.columns if str(Xr[c].dtype) == "category"]
    for _, tr, va in year_splits(d, valid_years, mode):
        tr = tr[~np.isnan(y[tr])]
        if len(tr) < 500:
            continue
        booster = lgb.train(params, lgb.Dataset(Xr.iloc[tr], label=y[tr].astype(int), categorical_feature=cats or "auto"), num_boost_round=n_rounds)
        P[va] = booster.predict(Xr.iloc[va])
    return P


# ── P2: 隊列の集計（P1 の予測だけから作る。自馬の結果は使わない）
def p2_aggregates(race_id: np.ndarray, pos_pred: np.ndarray, thr: float = FRONT_THR, margin: float = 0.05, scale: float = 0.05) -> pd.DataFrame:
    """馬ごとの s_pos_pred / s_pos_pred_rank / s_nfront_pred / s_front_gap / s_lone_front_prob。
    s_nfront_pred は自馬を除く『予測位置が thr 未満の馬』の数、s_front_gap は自馬 − 他馬の最前（負なら自馬が最前）、
    s_lone_front_prob は自馬が他の全馬より margin 以上前にいる確率の目安 Π σ((他馬 − 自馬 − margin) / scale)。"""
    d = pd.DataFrame({"r": race_id, "p": pos_pred})
    out = pd.DataFrame(index=np.arange(len(d)), columns=["s_pos_pred", "s_pos_pred_rank", "s_nfront_pred", "s_front_gap", "s_lone_front_prob"], dtype="float64")
    out["s_pos_pred"] = pos_pred
    ok = ~np.isnan(pos_pred)
    d["rank"] = d.groupby("r")["p"].rank(method="min")
    out["s_pos_pred_rank"] = d["rank"].to_numpy()
    for _, idx in d[ok].groupby("r").indices.items():
        ii = np.flatnonzero(ok)[idx]
        p = pos_pred[ii]
        n = len(p)
        if n < 2:
            continue
        front = (p < thr).astype(float)
        out.loc[ii, "s_nfront_pred"] = front.sum() - front
        srt = np.sort(p)
        others_min = np.where(p == srt[0], srt[1], srt[0])
        out.loc[ii, "s_front_gap"] = p - others_min
        z = (p[None, :] - p[:, None] - margin) / scale          # [i, j] = (p_j − p_i − margin)/scale
        sig = 1.0 / (1.0 + np.exp(-z))
        np.fill_diagonal(sig, 1.0)
        out.loc[ii, "s_lone_front_prob"] = sig.prod(axis=1)
    return out.astype("float32")


def race_level(df: pd.DataFrame, race_cols: Sequence[str], agg_cols: Sequence[str], extra: pd.DataFrame | None = None,
               extra_aggs: Sequence[str] = ()) -> pd.DataFrame:
    """レース単位の表（1 レース 1 行）。race_cols は先頭行の値、agg_cols は出走馬の as-of 値の平均・最大・最小・標準偏差。
    extra（馬ごとの P1/P2 由来の列）があれば、その extra_aggs を同様に集計する。"""
    g = df.groupby("race_id", sort=False)
    first = g[list(race_cols) + ["date", "year"]].first()
    parts = [first]
    src = df[list(agg_cols)]
    if extra is not None and len(extra_aggs):
        src = pd.concat([src, extra[list(extra_aggs)]], axis=1)
    a = src.groupby(df["race_id"], sort=False).agg(["mean", "max", "min", "std"])
    a.columns = [f"{c}__{s}" for c, s in a.columns]
    parts.append(a)
    out = pd.concat(parts, axis=1)
    out["n_runners"] = g.size()
    return out


# ── 評価
def within_race_spearman(race: np.ndarray, a: np.ndarray, b: np.ndarray, min_n: int = 5) -> float:
    """レース内の Spearman 相関の平均（欠損のある行は除く）。"""
    d = pd.DataFrame({"r": race, "a": a, "b": b}).dropna()
    d["ra"] = d.groupby("r")["a"].rank()
    d["rb"] = d.groupby("r")["b"].rank()
    d["x"] = (d["ra"] - d.groupby("r")["ra"].transform("mean"))
    d["y"] = (d["rb"] - d.groupby("r")["rb"].transform("mean"))
    g = d.assign(xy=d["x"] * d["y"], xx=d["x"] ** 2, yy=d["y"] ** 2).groupby("r").agg(xy=("xy", "sum"), xx=("xx", "sum"), yy=("yy", "sum"), n=("x", "size"))
    g = g[(g["n"] >= min_n) & (g["xx"] > 0) & (g["yy"] > 0)]
    return float((g["xy"] / np.sqrt(g["xx"] * g["yy"])).mean())


def reg_report(df: pd.DataFrame, target: np.ndarray, pred: np.ndarray, years: Sequence[int]) -> pd.DataFrame:
    """年ごとの MAE・相関・レース内 Spearman。MAE の基準は『その年より前の平均』を返す予測（定数）。"""
    rows = []
    y_arr = df["year"].to_numpy()
    for y in years:
        m = (y_arr == y) & ~np.isnan(target) & ~np.isnan(pred)
        if m.sum() < 100:
            continue
        prior = target[(y_arr < y) & ~np.isnan(target)]
        base = float(prior.mean()) if len(prior) else float(np.nanmean(target[m]))
        rows.append({"年": y, "n": int(m.sum()), "MAE": float(np.abs(target[m] - pred[m]).mean()),
                     "定数予測のMAE": float(np.abs(target[m] - base).mean()), "相関": float(np.corrcoef(target[m], pred[m])[0, 1]),
                     "レース内Spearman": within_race_spearman(df["race_id"].to_numpy()[m], target[m], pred[m])})
    return pd.DataFrame(rows).set_index("年")


def multiclass_logloss(P: np.ndarray, y: np.ndarray) -> float:
    ok = ~np.isnan(P).any(axis=1) & ~np.isnan(y)
    p = np.clip(P[ok][np.arange(ok.sum()), y[ok].astype(int)], 1e-12, 1.0)
    return float(-np.log(p).mean())

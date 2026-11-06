"""位置取り予想の学習モデル（§2.6 の位置取り予想を区分 A にする。2026-11-06、再学習の上流）。

目的変数: そのレースの初角の位置（0 = 先頭 〜 1 = 最後尾。通過順の最初の値と頭数から）。障害と、コーナーの無い直線（通過順が 1 つ）は除く。
特徴量: レース前に分かる量だけ（その日より前の走・騎手のその日より前の騎乗から作る as-of）。
  馬の過去の初角位置（重み付きの集計 = 今の位置取り予想と同じ重み・直近 1/3 走・ばらつき・同じ芝ダ×距離帯）、最終コーナーの位置、
  枠（馬番 ÷ 頭数）・頭数・コース・距離の変化・間隔・年齢・性別・斤量、騎手の位置取りの傾向、出走メンバーの中での相対位置。
評価: 年ごとに広げる walk-forward（年 Y の予測は Y の開始 10 日前より前で学習。tp-04 の P1 と同じ）。検証は 2022〜2024 年。
  2025 年（テスト）と 2026 年は評価に使わない。合格線（retrain_order.html §5.1）: 初角位置の MAE ≤ 0.2016（評価の前に固定）。
入力: data/page_reference/tables/<年>/race_result_flat.parquet（2016 年以降）。出力: data/local/modeling/pos_model/。

    python -m src.scripts.data.pos_model build   # 特徴量の表
    python -m src.scripts.data.pos_model eval    # walk-forward の評価（MAE・相関・基準線との比較）
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
TABLES = ROOT / "data/page_reference/tables"
OUT = Path(os.environ.get("POS_MODEL_OUT") or ROOT / "data/local/modeling/pos_model")
YEARS = list(range(2016, 2026))            # 2026 年は使わない
VALID = (2022, 2023, 2024)                 # 2025 年はテスト（ここでは使わない）
GAP_DAYS = 10
PASS_LINE = 0.2016                         # §5.1 の合格線（評価の前に固定。動かさない）
WEIGHTS = (0.35, 0.25, 0.20, 0.12, 0.08)   # position_forecast と同じ（直近ほど重い）
THREADS = int(os.environ.get("POS_THREADS", "6"))


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _first_corner(po: pd.Series) -> pd.Series:
    s = po.astype(str)
    ok = s.str.contains("-", regex=False)
    return pd.to_numeric(s.str.split("-").str[0].where(ok), errors="coerce")


def _last_corner(po: pd.Series) -> pd.Series:
    s = po.astype(str)
    ok = s.str.contains("-", regex=False)
    return pd.to_numeric(s.str.split("-").str[-1].where(ok), errors="coerce")


def load_runs() -> pd.DataFrame:
    cols = ["race_id", "date", "venue", "surface", "distance", "field_size", "horse_number", "horse_id", "sex_age",
            "jockey_weight", "jockey_id", "trainer_id", "weight", "weight_change", "passing_order", "race_class", "grade", "track_condition"]
    df = pd.concat([pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=cols) for y in YEARS], ignore_index=True)
    ped = pd.concat([pd.read_parquet(TABLES / str(y) / "race_shutuba_flat.parquet", columns=["race_id", "horse_id", "sire", "dam_sire"]) for y in YEARS])
    ped["race_id"], ped["horse_id"] = ped["race_id"].astype(str), ped["horse_id"].astype(str)
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    df = df.merge(ped.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    df = df[df["surface"].isin(["芝", "ダート"])].dropna(subset=["horse_id", "horse_number", "field_size"])
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["horse_id"], df["race_id"] = df["horse_id"].astype(str), df["race_id"].astype(str)
    df["jockey_id"] = df["jockey_id"].astype(str)
    n = df["field_size"].astype(float)
    fc, lc = _first_corner(df["passing_order"]), _last_corner(df["passing_order"])
    df["pos_first"] = ((fc - 1) / (n - 1)).where((n >= 4) & fc.between(1, n))
    df["pos_last"] = ((lc - 1) / (n - 1)).where((n >= 4) & lc.between(1, n))
    sc = pd.to_numeric(df["passing_order"].astype(str).str.split("-").str[1], errors="coerce")
    df["pos_second"] = ((sc - 1) / (n - 1)).where((n >= 4) & sc.between(1, n))
    df["trainer_id"] = df["trainer_id"].astype(str)
    df["sire"], df["dam_sire"] = df["sire"].fillna("?").astype(str), df["dam_sire"].fillna("?").astype(str)
    df["dist_band"] = pd.cut(df["distance"], [0, 1300, 1700, 2100, 9999], labels=False).astype(float)
    return df.sort_values(["date", "race_id", "horse_number"]).reset_index(drop=True)


def horse_history(df: pd.DataFrame) -> pd.DataFrame:
    """馬ごとに、その日より前の走だけから作る量（同じ日の別のレースは使わない）。"""
    d = df.sort_values(["horse_id", "date", "race_id"])
    g = d.groupby("horse_id", sort=False)
    out = pd.DataFrame(index=d.index)
    prev = [g["pos_first"].shift(k) for k in range(1, 6)]
    num = sum(w * p.fillna(0) for w, p in zip(WEIGHTS, prev))
    den = sum(w * p.notna() for w, p in zip(WEIGHTS, prev))
    out["h_pf_w5"] = (num / den).where(den > 0)                     # 今の位置取り予想と同じ重み付き（走の欠けは詰めない近似）
    out["h_pf_1"] = prev[0]
    out["h_pf_3"] = pd.concat(prev[:3], axis=1).mean(axis=1)
    out["h_pf_sd5"] = pd.concat(prev, axis=1).std(axis=1)
    out["h_pf_min5"] = pd.concat(prev, axis=1).min(axis=1)
    out["h_pl_3"] = pd.concat([g["pos_last"].shift(k) for k in range(1, 4)], axis=1).mean(axis=1)
    out["h_p2_3"] = pd.concat([g["pos_second"].shift(k) for k in range(1, 4)], axis=1).mean(axis=1)
    out["h_runs"] = g.cumcount()
    out["h_days"] = (d["date"] - g["date"].shift(1)).dt.days
    out["h_dist_chg"] = d["distance"] - g["distance"].shift(1)
    out["h_surf_chg"] = (d["surface"] != g["surface"].shift(1)).astype(float).where(out["h_runs"] > 0)
    # 同じ芝ダ×距離帯での過去の初角位置（as-of の平均）
    key = d["horse_id"] + "|" + d["surface"] + "|" + d["dist_band"].astype(str)
    gk = d.groupby(key, sort=False)["pos_first"]
    s = gk.transform(lambda x: x.shift(1).expanding().mean())
    out["h_pf_same"] = s
    return out.loc[df.index]


def asof_mean(df: pd.DataFrame, key: str, value: str = "pos_first", k: float = 30.0) -> np.ndarray:
    """key（騎手・調教師・父など）ごとの value の傾向。その日より前の走の平均を、全体の平均へ k 走ぶん縮める。同じ日は使わない。"""
    m = df[value].mean()
    day = df.groupby([key, "date"])[value].agg(["sum", "count"]).reset_index().sort_values([key, "date"])
    gj = day.groupby(key)
    day["cs"] = gj["sum"].cumsum() - day["sum"]
    day["cn"] = gj["count"].cumsum() - day["count"]
    day["v"] = (day["cs"] + k * m) / (day["cn"] + k)
    return df[[key, "date"]].merge(day[[key, "date", "v"]], on=[key, "date"], how="left")["v"].to_numpy()


def jockey_history(df: pd.DataFrame, k: float = 30.0) -> np.ndarray:
    return asof_mean(df, "jockey_id", k=k)


def build() -> pd.DataFrame:
    t0 = time.time()
    df = load_runs()
    _log(f"走 {len(df):,}（{df['year'].min()}〜{df['year'].max()}）")
    h = horse_history(df)
    df = pd.concat([df, h], axis=1)
    df["j_pf"] = jockey_history(df)
    df["t_pf"] = asof_mean(df, "trainer_id")
    df["s_pf"] = asof_mean(df, "sire", k=50.0)
    df["ds_pf"] = asof_mean(df, "dam_sire", k=50.0)
    df["debut"] = (df["h_runs"] == 0).astype(int)
    df["wt"] = pd.to_numeric(df["weight"], errors="coerce")
    df["wt_chg"] = pd.to_numeric(df["weight_change"], errors="coerce")
    n = df["field_size"].astype(float)
    df["gate_rel"] = (df["horse_number"].astype(float) - 1) / (n - 1)
    sa = df["sex_age"].astype(str)
    df["sex"] = sa.str[0].map({"牡": 0, "牝": 1, "セ": 2})
    df["age"] = pd.to_numeric(sa.str[1:], errors="coerce")
    df["is_turf"] = (df["surface"] == "芝").astype(int)
    df["course"] = (df["venue"].astype(str) + df["surface"] + df["distance"].astype(str)).astype("category")
    df["venue_c"] = df["venue"].astype("category")
    df["cls"] = df["grade"].astype(str).str.replace(r"\s", "", regex=True).astype("category")
    df["going"] = df["track_condition"].astype("category")
    # 出走メンバーの中での相対（自分の過去の位置取りが前から何番目か・自分より前に行きそうな馬の数・先行勢の多さ）
    base = df["h_pf_w5"].fillna(df["j_pf"])
    df["r_rank"] = base.groupby(df["race_id"]).rank(pct=True)
    df["r_ahead"] = df.groupby("race_id")["h_pf_w5"].transform(lambda x: x.lt(0.2).sum()) - base.lt(0.2).astype(int)
    df["r_mean"] = df.groupby("race_id")["h_pf_w5"].transform("mean")
    df["r_gap"] = base - df["r_mean"]
    df["r_known"] = df.groupby("race_id")["h_pf_w5"].transform("count") / n
    df["r_strong_front"] = df.groupby("race_id")["h_pf_min5"].transform(lambda x: x.lt(0.1).sum()) - df["h_pf_min5"].lt(0.1).astype(int)
    df["r_rank_min"] = df["h_pf_min5"].fillna(df["j_pf"]).groupby(df["race_id"]).rank(pct=True)
    prior = df["h_pf_w5"].fillna(0.5 * df["j_pf"] + 0.25 * df["s_pf"] + 0.25 * df["t_pf"])
    df["r_rank_prior"] = prior.groupby(df["race_id"]).rank(pct=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["passing_order", "sex_age", "weight", "weight_change"]).to_parquet(OUT / "features.parquet", index=False)
    _log(f"特徴量 {OUT / 'features.parquet'}（{time.time() - t0:.0f} 秒）")
    return df


FEATURES = ["h_pf_w5", "h_pf_1", "h_pf_3", "h_pf_sd5", "h_pf_min5", "h_pl_3", "h_p2_3", "h_runs", "h_days", "h_dist_chg", "h_surf_chg", "h_pf_same",
            "j_pf", "t_pf", "s_pf", "ds_pf", "debut", "wt", "wt_chg", "gate_rel", "field_size", "distance", "dist_band", "is_turf", "sex", "age", "jockey_weight",
            "course", "venue_c", "cls", "going", "r_rank", "r_ahead", "r_mean", "r_gap", "r_known", "r_strong_front", "r_rank_min", "r_rank_prior"]
PARAMS = {"objective": "l1", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1}
ROUNDS = 600


def evaluate() -> dict:
    import lightgbm as lgb

    df = pd.read_parquet(OUT / "features.parquet")
    for c in ("course", "venue_c", "cls", "going"):
        df[c] = df[c].astype("category")
    y = df["pos_first"].to_numpy()
    res: dict = {"pass_line": PASS_LINE, "years": {}, "features": FEATURES, "params": PARAMS, "rounds": ROUNDS}
    allp, ally, allb, allc = [], [], [], []
    for yv in VALID:
        start = pd.Timestamp(f"{yv}-01-01")
        tr = (df["date"] < start - pd.Timedelta(days=GAP_DAYS)) & ~np.isnan(y)
        va = (df["year"] == yv) & ~np.isnan(y)
        b = lgb.train(dict(PARAMS, num_threads=THREADS, seed=0), lgb.Dataset(df.loc[tr, FEATURES], label=y[tr]), num_boost_round=ROUNDS)
        p = np.clip(b.predict(df.loc[va, FEATURES]), 0, 1)
        yy = y[va]
        agg = df.loc[va, "h_pf_w5"].to_numpy()                            # 今の位置取り予想（集計）に当たる値
        const = np.full(len(yy), np.nanmean(y[tr]))
        has = ~np.isnan(agg)
        r = {"n": int(va.sum()), "mae_model": float(np.mean(np.abs(p - yy))), "mae_const": float(np.mean(np.abs(const - yy))),
             "mae_model_on_known": float(np.mean(np.abs(p[has] - yy[has]))), "mae_agg_on_known": float(np.mean(np.abs(agg[has] - yy[has]))),
             "corr_model": float(np.corrcoef(p, yy)[0, 1]), "known_share": float(has.mean())}
        res["years"][str(yv)] = r
        _log(f"{yv}: MAE モデル {r['mae_model']:.4f} / 定数 {r['mae_const']:.4f}（既知の馬で モデル {r['mae_model_on_known']:.4f} vs 集計 {r['mae_agg_on_known']:.4f}）相関 {r['corr_model']:.3f}")
        allp.append(p); ally.append(yy); allc.append(const)
        if yv == VALID[-1]:
            imp = dict(zip(FEATURES, b.feature_importance("gain").round(0).tolist()))
            res["importance_gain"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    p, yy, c = np.concatenate(allp), np.concatenate(ally), np.concatenate(allc)
    res["overall"] = {"mae_model": float(np.mean(np.abs(p - yy))), "mae_const": float(np.mean(np.abs(c - yy))), "corr_model": float(np.corrcoef(p, yy)[0, 1])}
    res["passed"] = res["overall"]["mae_model"] <= PASS_LINE
    (OUT / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"2022〜2024: MAE {res['overall']['mae_model']:.4f}（定数 {res['overall']['mae_const']:.4f}、合格線 {PASS_LINE}）→ {'合格' if res['passed'] else '未達'}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "eval", "all"])
    a = ap.parse_args()
    if a.cmd in ("build", "all"):
        build()
    if a.cmd in ("eval", "all"):
        evaluate()


if __name__ == "__main__":
    main()

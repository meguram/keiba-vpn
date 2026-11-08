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
    python -m src.scripts.data.pos_model oof     # 下流用の予測（年ごとに、その年より前だけで学習したモデル。2018〜2025 年）
    python -m src.scripts.data.pos_model eval2   # 2 段構え（1 段目の予測から出走メンバーの中での関係を作って 2 段目で学習）
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


def oikiri_features() -> pd.DataFrame:
    """最終追い切りの時計（4F・3F・ラスト 1F）・坂路かどうか・評価（A〜D）・レースまでの日数。レース前に公開される情報。
    表の列が一部ずれて入っている（コースが rider 列に入る行がある）ので、坂路の判定は course と rider の両方を見る。"""
    num = re.compile(r"(\d+\.\d)(?=\()")
    rows = []
    for y in YEARS:
        f = TABLES / str(y) / "race_oikiri_flat.parquet"
        if not f.exists():
            continue
        o = pd.read_parquet(f, columns=["race_id", "horse_id", "date", "course", "rider", "lap_times", "evaluation", "training_date"])
        rows.append(o)
    o = pd.concat(rows, ignore_index=True).drop_duplicates(["race_id", "horse_id"])
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    vals = o["lap_times"].fillna("").astype(str).map(lambda s: [float(x) for x in num.findall(s)])
    o["oi_1f"] = vals.map(lambda v: v[-1] if len(v) >= 1 else np.nan)
    o["oi_3f"] = vals.map(lambda v: v[-2] if len(v) >= 2 else np.nan)
    o["oi_4f"] = vals.map(lambda v: v[-3] if len(v) >= 3 else np.nan)
    lab = o["course"].fillna("").astype(str) + o["rider"].fillna("").astype(str)
    o["oi_hill"] = lab.str.contains("坂").astype(float)
    o["oi_eval"] = o["evaluation"].map({"A": 3, "B": 2, "C": 1, "D": 0})
    td = pd.to_datetime(o["training_date"].astype(str).str[:10], format="%Y/%m/%d", errors="coerce")
    o["oi_days"] = (pd.to_datetime(o["date"]) - td).dt.days
    return o[["race_id", "horse_id", "oi_1f", "oi_3f", "oi_4f", "oi_hill", "oi_eval", "oi_days"]]


def race_front_pace(df: pd.DataFrame, k: float = 20.0) -> pd.Series:
    """レースの前半 3 区間（200m×3）の合計と、同じコースのその日より前のレースの平均との差（秒。負 = 前半が速い）。
    そのレース自身の値は特徴量にしない（過去走の分だけを履歴として使う）。"""
    r = pd.concat([pd.read_parquet(TABLES / str(y) / "race_result_race.parquet", columns=["race_id", "lap_times"]) for y in YEARS])
    r["race_id"] = r["race_id"].astype(str)

    def f3(v):
        try:
            a = list(v)
            return float(sum(a[:3])) if len(a) >= 3 else np.nan
        except TypeError:
            return np.nan

    r["f3"] = r["lap_times"].map(f3)
    rk = df.drop_duplicates("race_id")[["race_id", "date", "venue", "surface", "distance"]].merge(r[["race_id", "f3"]], on="race_id", how="left")
    rk["ckey"] = rk["venue"].astype(str) + rk["surface"] + rk["distance"].astype(str)
    ok = rk.dropna(subset=["f3"]).reset_index(drop=True)
    ok["cm"] = asof_mean(ok, "ckey", "f3", k=k)
    ok["f3_dev"] = ok["f3"] - ok["cm"]
    return df[["race_id"]].merge(ok[["race_id", "f3_dev"]], on="race_id", how="left")["f3_dev"].to_numpy()


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
    out["h_pf_2"] = prev[1]
    if "f3_dev" in d.columns:                          # 過去走の前半のペース（そのレース自身の値は使わない）と、その走の位置取りの組
        for k in (1, 2, 3):
            out[f"h_f3d_{k}"] = g["f3_dev"].shift(k)
        es = [(1 - g["pos_first"].shift(k)) - 0.1 * g["f3_dev"].shift(k) for k in (1, 2, 3)]   # 速い流れで前に行けた走ほど高い（係数は粗い目安）
        out["h_es_3"] = pd.concat(es, axis=1).mean(axis=1)
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
    df["f3_dev"] = race_front_pace(df)
    df = df.merge(oikiri_features(), on=["race_id", "horse_id"], how="left")
    h = horse_history(df)
    df = pd.concat([df, h], axis=1)
    df["j_pf"] = jockey_history(df)
    df["t_pf"] = asof_mean(df, "trainer_id")
    df["s_pf"] = asof_mean(df, "sire", k=50.0)
    df["ds_pf"] = asof_mean(df, "dam_sire", k=50.0)
    df["debut"] = (df["h_runs"] == 0).astype(int)
    # 騎手の乗り替わり（前走と同じ騎手か・今回の騎手がこの馬のこれまでの騎手より前に行く傾向か）
    hs = df.sort_values(["horse_id", "date", "race_id"])
    gh = hs.groupby("horse_id", sort=False)
    df.loc[hs.index, "h_jk_same"] = (hs["jockey_id"] == gh["jockey_id"].shift(1)).astype(float).where(gh.cumcount() > 0).to_numpy()
    prev_j = pd.concat([gh["j_pf"].shift(k) for k in (1, 2, 3)], axis=1).mean(axis=1)
    df.loc[hs.index, "j_pf_diff"] = (hs["j_pf"] - prev_j).to_numpy()
    # コースごとの枠の有利不利（内・中・外の枠の馬がそのコースでそれまでに取った位置 − コース全体）
    ck = df["venue"].astype(str) + df["surface"] + df["distance"].astype(str)
    gb = pd.cut((df["horse_number"].astype(float) - 1) / (df["field_size"].astype(float) - 1), [-0.01, 0.33, 0.67, 1.01], labels=["i", "m", "o"]).astype(str)
    df["_ck"], df["_ckg"] = ck, ck + "|" + gb
    df["cg_pf"] = asof_mean(df, "_ckg", k=50.0) - asof_mean(df, "_ck", k=50.0)
    df = df.drop(columns=["_ck", "_ckg"])
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
    # 追い切りの時計を、坂路かどうか×年で標準化し、出走メンバーの中で比べる（速いほど小さい）
    for c in ("oi_1f", "oi_3f", "oi_4f"):
        v = df[c].where(df[c].between(df[c].quantile(0.001), df[c].quantile(0.999)))
        grp = df["oi_hill"].astype(str) + "|" + df["year"].astype(str)
        df[f"{c}_z"] = (v - v.groupby(grp).transform("mean")) / v.groupby(grp).transform("std")
        df[f"r_{c}_rank"] = df[f"{c}_z"].groupby(df["race_id"]).rank(pct=True)
    df["r_oi_eval_rank"] = df["oi_eval"].groupby(df["race_id"]).rank(pct=True, ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["passing_order", "sex_age", "weight", "weight_change", "f3_dev"]).to_parquet(OUT / "features.parquet", index=False)
    _log(f"特徴量 {OUT / 'features.parquet'}（{time.time() - t0:.0f} 秒）")
    return df


FEATURES = ["h_pf_w5", "h_pf_1", "h_pf_3", "h_pf_sd5", "h_pf_min5", "h_pl_3", "h_p2_3", "h_runs", "h_days", "h_dist_chg", "h_surf_chg", "h_pf_same",
            "j_pf", "t_pf", "s_pf", "ds_pf", "debut", "wt", "wt_chg", "gate_rel", "field_size", "distance", "dist_band", "is_turf", "sex", "age", "jockey_weight",
            "course", "venue_c", "cls", "going", "r_rank", "r_ahead", "r_mean", "r_gap", "r_known", "r_strong_front", "r_rank_min", "r_rank_prior",
            "h_pf_2", "h_f3d_1", "h_f3d_2", "h_f3d_3", "h_es_3", "h_jk_same", "j_pf_diff", "cg_pf",
            "oi_1f", "oi_3f", "oi_4f", "oi_hill", "oi_eval", "oi_days",
            "oi_1f_z", "oi_3f_z", "oi_4f_z", "r_oi_1f_rank", "r_oi_3f_rank", "r_oi_4f_rank", "r_oi_eval_rank"]
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


STACK_FROM = 2018           # 1 段目の OOF を作る最初の年（2016〜2017 年だけで学習）
SLOW = dict(PARAMS, learning_rate=0.02, num_leaves=127, min_data_in_leaf=300)
SLOW_ROUNDS = 2000


def _fit_predict(X_tr, y_tr, X_va, params=SLOW, rounds=SLOW_ROUNDS, seeds=(0,)):
    import lightgbm as lgb

    ps = [lgb.train(dict(params, num_threads=THREADS, seed=sd), lgb.Dataset(X_tr, label=y_tr), num_boost_round=rounds).predict(X_va) for sd in seeds]
    return np.clip(np.mean(ps, axis=0), 0, 1)


RANK_BLEND = float(os.environ.get("POS_RANK_BLEND", "0.1"))


def _rank_predict(d_tr: pd.DataFrame, X_tr: pd.DataFrame, y_tr: np.ndarray, d_va: pd.DataFrame, X_va: pd.DataFrame) -> np.ndarray:
    """LambdaRank（前ほど関連度が高い）でレース内の並びを学び、出力をレース内の位置（0〜1）に直す。"""
    import lightgbm as lgb

    order = np.argsort(d_tr["race_id"].to_numpy(), kind="stable")
    rid = d_tr["race_id"].to_numpy()[order]
    grp = pd.Series(rid).groupby(rid, sort=False).size().to_numpy()
    rel = np.clip((1 - y_tr[order]) * 30, 0, 30).astype(int)
    b = lgb.train(dict(objective="lambdarank", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                       bagging_freq=1, verbose=-1, num_threads=THREADS, seed=0, label_gain=list(range(31)), lambdarank_truncation_level=30),
                  lgb.Dataset(X_tr.iloc[order], label=rel, group=grp), num_boost_round=600)
    v = pd.DataFrame({"race_id": d_va["race_id"].to_numpy(), "s": b.predict(X_va)})
    n = v.groupby("race_id")["s"].transform("count")
    return ((v.groupby("race_id")["s"].rank(ascending=False) - 1) / (n - 1).clip(lower=1)).to_numpy()


def _stage2_cols(df: pd.DataFrame) -> pd.DataFrame:
    """1 段目の予測 p1 から、出走メンバーの中での関係を作る（自分より前に行きそうな馬の数・すぐ前/後ろとの差など）。"""
    g = df.groupby("race_id")["p1"]
    out = pd.DataFrame(index=df.index)
    out["p1"] = df["p1"]
    out["p1_rank"] = g.rank(pct=True)
    out["p1_ahead"] = g.rank(method="min") - 1
    out["p1_mean"], out["p1_std"] = g.transform("mean"), g.transform("std")
    srt = df[["race_id", "p1"]].sort_values(["race_id", "p1"])
    gs = srt.groupby("race_id")["p1"]
    out.loc[srt.index, "p1_gap_prev"] = (srt["p1"] - gs.shift(1)).to_numpy()
    out.loc[srt.index, "p1_gap_next"] = (gs.shift(-1) - srt["p1"]).to_numpy()
    out["p1_front_cnt"] = g.transform(lambda x: (x < 0.15).sum())
    return out


STAGE1_SEEDS = tuple(range(int(os.environ.get("POS_STAGE1_SEEDS", "3"))))
STAGE2_SEEDS = tuple(range(int(os.environ.get("POS_STAGE2_SEEDS", "5"))))


def evaluate_stack(seeds=STAGE2_SEEDS) -> dict:
    """2 段構え。1 段目は年ごとに広げる OOF（年 Y は Y の開始 10 日前より前で学習）を STACK_FROM〜2024 年で作り、
    2 段目は検証の年 Y より前の年の OOF で学習して Y を予測する。検証は 2022〜2024 年（2025 年は使わない）。"""
    df = pd.read_parquet(OUT / "features.parquet")
    for c in ("course", "venue_c", "cls", "going"):
        df[c] = df[c].astype("category")
    y = df["pos_first"].to_numpy()
    df["p1"] = np.nan
    for yv in range(STACK_FROM, VALID[-1] + 1):
        tr = (df["date"] < pd.Timestamp(f"{yv}-01-01") - pd.Timedelta(days=GAP_DAYS)) & ~np.isnan(y)
        va = df["year"] == yv
        df.loc[va, "p1"] = _fit_predict(df.loc[tr, FEATURES], y[tr], df.loc[va, FEATURES], seeds=STAGE1_SEEDS)
        _log(f"1 段目 OOF {yv}")
    S = _stage2_cols(df)
    cols2 = FEATURES + list(S.columns)
    X2 = pd.concat([df[FEATURES], S], axis=1)
    res: dict = {"pass_line": PASS_LINE, "years": {}, "stack_from": STACK_FROM, "seeds": list(seeds)}
    allp, ally, allp1 = [], [], []
    for yv in VALID:
        tr = (df["year"] >= STACK_FROM) & (df["date"] < pd.Timestamp(f"{yv}-01-01") - pd.Timedelta(days=GAP_DAYS)) & ~np.isnan(y)
        va = (df["year"] == yv) & ~np.isnan(y)
        p = _fit_predict(X2.loc[tr, cols2], y[tr], X2.loc[va, cols2], seeds=seeds)
        if RANK_BLEND > 0:                      # レース内の順位を直接学ぶ LambdaRank を RANK_BLEND だけ混ぜる（系統 3 の比較で 1 割が最良）
            p = (1 - RANK_BLEND) * p + RANK_BLEND * _rank_predict(df.loc[tr], X2.loc[tr, cols2], y[tr], df.loc[va], X2.loc[va, cols2])
        yy, p1 = y[va], df.loc[va, "p1"].to_numpy()
        res["years"][str(yv)] = {"mae_stack": float(np.mean(np.abs(p - yy))), "mae_stage1": float(np.mean(np.abs(p1 - yy)))}
        _log(f"{yv}: 2 段構え {res['years'][str(yv)]['mae_stack']:.4f}（1 段目 {res['years'][str(yv)]['mae_stage1']:.4f}）")
        allp.append(p); ally.append(yy); allp1.append(p1)
    p, yy, p1 = np.concatenate(allp), np.concatenate(ally), np.concatenate(allp1)
    res["overall"] = {"mae_stack": float(np.mean(np.abs(p - yy))), "mae_stage1": float(np.mean(np.abs(p1 - yy)))}
    res["passed"] = res["overall"]["mae_stack"] <= PASS_LINE
    (OUT / "eval_stack.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(OUT / "eval_stack_preds.npz", y=yy, p=p, p1=p1)
    _log(f"2022〜2024: 2 段構え {res['overall']['mae_stack']:.4f}（1 段目 {res['overall']['mae_stage1']:.4f}、合格線 {PASS_LINE}）→ {'合格' if res['passed'] else '未達'}")
    return res


def make_oof(last_year: int = 2025) -> pd.DataFrame:
    """下流（追走難度・前段 S3・ラップ予想）用の予測。年 Y の行は Y の開始 10 日前より前だけで学習したモデルで予測する（先読みなし）。
    1 段目は STACK_FROM〜last_year、2 段目は STACK_FROM+1〜last_year（STACK_FROM の年は 1 段目の値）。2016〜2017 年は予測なし。"""
    df = pd.read_parquet(OUT / "features.parquet")
    for c in ("course", "venue_c", "cls", "going"):
        df[c] = df[c].astype("category")
    y = df["pos_first"].to_numpy()
    df["p1"] = np.nan
    for yv in range(STACK_FROM, last_year + 1):
        tr = (df["date"] < pd.Timestamp(f"{yv}-01-01") - pd.Timedelta(days=GAP_DAYS)) & ~np.isnan(y)
        va = df["year"] == yv
        df.loc[va, "p1"] = _fit_predict(df.loc[tr, FEATURES], y[tr], df.loc[va, FEATURES], seeds=STAGE1_SEEDS)
        _log(f"1 段目 {yv}")
    S = _stage2_cols(df)
    cols2 = FEATURES + list(S.columns)
    X2 = pd.concat([df[FEATURES], S], axis=1)
    df["pos_pred"] = np.where(df["year"] == STACK_FROM, df["p1"], np.nan)
    for yv in range(STACK_FROM + 1, last_year + 1):
        tr = (df["year"] >= STACK_FROM) & (df["date"] < pd.Timestamp(f"{yv}-01-01") - pd.Timedelta(days=GAP_DAYS)) & ~np.isnan(y)
        va = df["year"] == yv
        p = _fit_predict(X2.loc[tr, cols2], y[tr], X2.loc[va, cols2], seeds=STAGE2_SEEDS)
        if RANK_BLEND > 0:
            p = (1 - RANK_BLEND) * p + RANK_BLEND * _rank_predict(df.loc[tr], X2.loc[tr, cols2], y[tr], df.loc[va], X2.loc[va, cols2])
        df.loc[va, "pos_pred"] = p
        _log(f"2 段目 {yv}")
    out = df.loc[df["pos_pred"].notna(), ["race_id", "horse_id", "horse_number", "date", "year", "pos_pred", "p1", "pos_first"]]
    out.to_parquet(OUT / "pos_oof.parquet", index=False)
    _log(f"予測 {OUT / 'pos_oof.parquet'}（{len(out):,} 行）")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "eval", "eval2", "oof", "all"])
    a = ap.parse_args()
    if a.cmd in ("build", "all"):
        build()
    if a.cmd in ("eval", "all"):
        evaluate()
    if a.cmd in ("eval2", "all"):
        evaluate_stack()
    if a.cmd == "oof":
        make_oof()


if __name__ == "__main__":
    main()

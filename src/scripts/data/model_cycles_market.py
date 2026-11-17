"""第 2・3 段（2026-11-16、ユーザーの依頼「1 番人気を ◎ にしない・市場が見落としている馬を当てる」）: 市場とのずれを学習するモデルの道具。

    python -m src.scripts.data.model_cycles_market market                 # 市場の勝率 q（確定の単勝オッズ）の表を作る（2016〜2025 年）
    python -m src.scripts.data.model_cycles_market member NAME --spec '{...}' [--seeds 1]
        # xgb の二値（勝ち・2 着以内・3 着以内）を、市場の確率の logit を offset（base_margin）にして学ぶ。OOF は valid 2022〜2024
    python -m src.scripts.data.model_cycles_market report NAME [--ref r04,prod]  # ◎ の選び方ごとの成績・1 番人気の割合・ずれの分位・1 番人気 ◎ の連対率

- 入力の特徴量は model_cycles の土台（直した表・漏れの除去・ずれの 38 列の除外）と同じで、オッズ・人気は入れない（offset にだけ使う）。
  再学習のランブックの「市場オッズ・人気は親モデルの特徴量にしない」の例外（ユーザーの指示、offset のみ）。check_no_market_features で確かめる。
- spec: model_cycles.apply_spec の変更 ＋ offset（"none"・"raw"・"fl"）・win_weight_alpha（第 3 段: 勝った馬の重み ∝ オッズ^α）・params。
- 市場の勝率 q: 確定の単勝オッズの逆数をレース内で正規化（払戻率を除く）。fl は本命-大穴の偏りの補正 q ∝ (1/odds)^β（β は 2016〜2021 年で当てる）。
  連対・複勝の offset は q から割引なしの Harville で作る。
- 出力: CYC/market/oof/NAME.parquet（race_id・horse_id・year・p_win・p_top2・p_top3・q・r = log(p_win/q)）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import model_cycles as mc

KEY = ["race_id", "horse_id"]
MK = mc.CYC / "market"
FORBIDDEN_FEATURES = ("odds", "popularity", "win_odds", "place_odds", "人気", "tansho", "fukusho", "market", "o_pop", "mkt_", "mk_", "_pop", "chalk", "ninki", "fav")
# 列名の部分一致の検査。人気から作った列を別の名前で入れたものは捕まえられないので、入力の列は model_cycles の土台（候補 v6e16 と同じ列から外すだけ）に限る
MEMBER_KEYS = {"drop", "drop_prefix", "blank_before", "blank", "clip_max", "copy", "zero_if_present", "blank_newcomer", "offset", "win_weight_alpha", "params",
               "offset_noise_sd"}
BASE_DROP_N = 38                                # 土台（ずれの 38 列の除外）。spec の drop の書き忘れを止める
ODDS_SUM_OK = (1.0, 1.6)                         # Σ(1/単勝オッズ) がこの外のレースはオッズの異常として市場の表から外す（指数の値が入った既知の不具合など）
BETA_YEARS = range(2016, 2022)                 # 本命-大穴の偏りの補正の β は、検証年（2022〜）より前の年だけで当てる


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def check_no_market_features(cols) -> None:
    """入力の特徴量にオッズ・人気が入っていないこと（offset にだけ使う決まり）。"""
    bad = [c for c in cols if any(k in str(c).lower() for k in FORBIDDEN_FEATURES)]
    if bad:
        raise SystemExit(f"入力にオッズ・人気の列がある（offset 以外で使わない）: {bad[:10]}")


def norm_race(race: np.ndarray, p: np.ndarray, total: float = 1.0) -> np.ndarray:
    s = pd.Series(p).groupby(race).transform("sum").to_numpy()
    return total * p / s


def fit_beta(race: np.ndarray, inv: np.ndarray, win: np.ndarray) -> float:
    """q ∝ inv^β の β を、勝ち馬の対数尤度の最大で当てる。"""
    from scipy.optimize import minimize_scalar

    li = np.log(inv)

    def nll(b):
        z = b * li
        zm = pd.Series(z).groupby(race).transform("max").to_numpy()
        e = np.exp(z - zm)
        s = pd.Series(e).groupby(race).transform("sum").to_numpy()
        return -float(((z - zm - np.log(s)) * win).sum())

    b = float(minimize_scalar(nll, bounds=(0.6, 1.6), method="bounded").x)
    if not 0.62 < b < 1.58:
        log(f"警告: β={b:.3f} が探索の端にある")
    return b


def q_from_odds(race: np.ndarray, odds: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """単勝オッズから市場の勝率（払戻率を除いてレース内で合計 1）。オッズの無い馬・0 以下は NaN。beta≠1 は本命-大穴の偏りの補正。"""
    o = np.asarray(odds, float)
    inv = np.where(np.isfinite(o) & (o > 0), 1.0 / np.where(o > 0, o, 1.0), np.nan) ** beta
    s = pd.Series(inv).groupby(np.asarray(race)).transform("sum").to_numpy()
    return inv / s


PRE_ODDS = mc.ROOT / "data/local/research/persona1/pre_odds.parquet"


def load_pre_odds(phase: str) -> pd.DataFrame:
    """手元の発走前オッズの表（週次の pre_odds_table が作る。2026 年の一部）から、ある時点（t60・t15・t10・t05・t02）の市場の勝率。
    列: race_id・horse_number・win_odds・q。GCS は読まない。"""
    d = pd.read_parquet(PRE_ODDS)
    d = d[d["phase"] == phase].copy()
    d["race_id"] = d["race_id"].astype(str)
    d = d.sort_values("snapshot_at").drop_duplicates(["race_id", "horse_number"], keep="last")
    d["q"] = q_from_odds(d["race_id"].to_numpy(), d["win_odds"].to_numpy())
    return d[["race_id", "horse_number", "win_odds", "q"]]


def drift_table(pre: pd.DataFrame, final: pd.DataFrame) -> dict:
    """発走前の市場の勝率と確定の市場の勝率のずれ（同じレース・同じ馬番）。logit の差の大きさ、1 番人気の一致、順位の相関。"""
    from scipy.stats import spearmanr

    m = pre.merge(final, on=["race_id", "horse_number"], suffixes=("_pre", "_fin")).dropna(subset=["q_pre", "q_fin"])
    for c in ("q_pre", "q_fin"):                            # 突き合わせた馬だけで正規化し直す（取消・欠けの馬の分の系統的なずれを除く）
        m[c] = m[c] / m.groupby("race_id")[c].transform("sum")
    dl = logit(m["q_pre"].to_numpy()) - logit(m["q_fin"].to_numpy())
    top_pre = m.loc[m.groupby("race_id")["q_pre"].idxmax(), ["race_id", "horse_number"]]
    top_fin = m.loc[m.groupby("race_id")["q_fin"].idxmax(), ["race_id", "horse_number"]]
    agree = top_pre.merge(top_fin, on="race_id", suffixes=("_p", "_f"))
    rho = [spearmanr(g["q_pre"], g["q_fin"]).correlation for _, g in m.groupby("race_id") if len(g) >= 3]
    return {"races": int(m["race_id"].nunique()), "horses": int(len(m)), "logit_diff_sd": float(np.std(dl)), "logit_diff_mae": float(np.mean(np.abs(dl))),
            "logit_diff_q": [float(x) for x in np.quantile(dl, [0.05, 0.25, 0.5, 0.75, 0.95])],
            "fav_agree": float((agree["horse_number_p"] == agree["horse_number_f"]).mean()), "spearman_median": float(np.nanmedian(rho))}


def cmd_drift(_a) -> None:
    """2026 年の発走前オッズ（T-60・T-15・T-10・T-5・T-2）の市場の勝率と、確定オッズの市場の勝率のずれ。推論で offset に使うときの効きの落ち方の材料。"""
    r = pd.read_parquet(mc.TABLES / "2026/race_result_flat.parquet", columns=["race_id", "horse_number", "odds"])
    r["race_id"] = r["race_id"].astype(str)
    r["odds"] = pd.to_numeric(r["odds"], errors="coerce")
    r = r.drop_duplicates(["race_id", "horse_number"])
    r["q"] = q_from_odds(r["race_id"].to_numpy(), r["odds"].to_numpy())
    out = {ph: drift_table(load_pre_odds(ph), r[["race_id", "horse_number", "q"]]) for ph in ("t60", "t15", "t10", "t05", "t02")}
    MK.mkdir(parents=True, exist_ok=True)
    (MK / "drift_2026.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for ph, d in out.items():
        log(f"{ph}: {d['races']}R logit 差の SD {d['logit_diff_sd']:.3f} 1 番人気の一致 {d['fav_agree']:.3f} 順位相関の中央値 {d['spearman_median']:.3f}")


def cmd_market(_a) -> None:
    rows = []
    for y in range(2016, 2026):
        r = pd.read_parquet(mc.TABLES / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_id", "horse_number", "odds", "popularity", "finish_position"])
        rows.append(r)
    r = pd.concat(rows, ignore_index=True)
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    r = r.drop_duplicates(KEY)
    for c in ("odds", "popularity", "finish_position"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    ok = r["odds"] > 0
    r["inv"] = np.where(ok, 1.0 / r["odds"].where(ok, np.nan), np.nan)
    ssum = r.groupby("race_id")["inv"].transform("sum")
    bad_races = r.loc[(ssum < ODDS_SUM_OK[0]) | (ssum > ODDS_SUM_OK[1]), "race_id"].unique()
    r = r[~r["race_id"].isin(bad_races)].copy()
    ok = r["odds"] > 0
    log(f"Σ(1/オッズ) が {ODDS_SUM_OK} の外のレース {len(bad_races)} を外した")
    r["q_raw"] = r["inv"] / r.groupby("race_id")["inv"].transform("sum")
    yr = r["race_id"].str[:4].astype(int).to_numpy()
    fit = ok.to_numpy() & np.isin(yr, list(BETA_YEARS)) & r["finish_position"].notna().to_numpy()
    sub = r[fit]
    beta = fit_beta(sub["race_id"].to_numpy(), sub["inv"].to_numpy(), (sub["finish_position"] == 1).to_numpy().astype(float))
    r["q_fl"] = r["inv"] ** beta
    r["q_fl"] = r["q_fl"] / r.groupby("race_id")["q_fl"].transform("sum")
    MK.mkdir(parents=True, exist_ok=True)
    r[KEY + ["horse_number", "odds", "popularity", "finish_position", "q_raw", "q_fl"]].to_parquet(MK / "market.parquet", index=False)
    (MK / "market.json").write_text(json.dumps({"beta": beta, "beta_years": [min(BETA_YEARS), max(BETA_YEARS)], "rows": len(r), "bad_odds_races": len(bad_races),
                                                "odds_missing": float((~ok).mean())}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"market: {len(r)} 行、β={beta:.4f}（{min(BETA_YEARS)}〜{max(BETA_YEARS)} 年で当てた）、オッズ欠け {float((~ok).mean()):.4f}")


def load_market(meta: pd.DataFrame, kind: str) -> np.ndarray:
    """meta の行の順に、市場の勝率 q（kind: raw・fl）。オッズの無い馬は、そのレースの頭数で割った値の半分（小さい値）にしてから正規化し直す。"""
    m = pd.read_parquet(MK / "market.parquet")
    j = meta[KEY].merge(m[KEY + [f"q_{kind}"]], on=KEY, how="left")
    q = j[f"q_{kind}"].to_numpy(float)
    race = meta["race_id"].to_numpy()
    n = pd.Series(np.ones(len(q))).groupby(race).transform("sum").to_numpy()
    q = np.where(np.isfinite(q) & (q > 0), q, 0.5 / n)
    return norm_race(race, np.clip(q, 1e-5, 1.0))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_pred(X_tr, y_tr, X_va, params: dict, seed: int, margin_tr=None, margin_va=None, weight=None) -> np.ndarray:
    import xgboost as xgb

    p = dict(params)
    m = xgb.XGBClassifier(n_estimators=p.get("n_estimators", 400), learning_rate=p.get("learning_rate", 0.05), max_depth=p.get("max_depth", 6),
                          subsample=p.get("subsample", 0.8), colsample_bytree=p.get("colsample_bytree", 0.7), reg_lambda=p.get("reg_lambda", 5.0),
                          min_child_weight=p.get("min_child_weight", 20), tree_method="hist", device=p.get("device", "cuda"),
                          enable_categorical=True, random_state=seed, n_jobs=4, verbosity=0)
    m.fit(X_tr, y_tr, sample_weight=weight, base_margin=margin_tr)
    return m.predict_proba(X_va, base_margin=margin_va)[:, 1]


def cmd_member(a) -> None:
    from src.pipeline.models import zoo

    spec = json.loads(a.spec) if a.spec else {}
    bad = set(spec) - MEMBER_KEYS
    if bad:
        raise SystemExit(f"この道具が対応しない spec: {sorted(bad)}（halflife_years・min_train_year・seed_offset などは未対応）")
    if spec.get("offset", "none") not in ("none", "raw", "fl"):
        raise SystemExit(f"offset は none・raw・fl のどれか: {spec.get('offset')}")
    out = MK / "oof" / f"{a.name}.parquet"
    if out.exists() and not a.force:
        log(f"{a.name} 既存")
        return
    X, meta, info = mc.load_cache()
    n0 = X.shape[1]
    X = mc.apply_spec(X, meta, {k: v for k, v in spec.items() if k not in ("offset", "win_weight_alpha", "params", "offset_noise_sd")})
    if n0 - X.shape[1] < BASE_DROP_N:
        raise SystemExit(f"入力の列が {n0} → {X.shape[1]}（土台の 38 列の除外が spec に無い）")
    check_no_market_features(X.columns)
    params = {**info["params"], **spec.get("params", {})}
    race = meta["race_id"].to_numpy()
    year = meta["year"].to_numpy()
    fin = meta["finish_position"].to_numpy().astype(float)
    off = spec.get("offset", "none")
    q = load_market(meta, "raw" if off == "none" else off)
    q2, q3 = zoo.harville_topk(race, q, 1.0, 1.0)
    q2, q3 = norm_race(race, q2, 2.0), norm_race(race, q3, 3.0)
    margins = {"win": logit(q), "top2": logit(np.clip(q2, 1e-5, 0.999)), "top3": logit(np.clip(q3, 1e-5, 0.999))} if off != "none" else None
    noise = float(spec.get("offset_noise_sd", 0.0))     # 学習の行の offset に、発走前と確定のずれと同じ大きさの雑音を足す（推論は T-60 のオッズを使うため）
    rng = np.random.default_rng(12345)
    ys = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    odds = pd.read_parquet(MK / "market.parquet")[KEY + ["odds"]]
    odds_row = meta[KEY].merge(odds, on=KEY, how="left")["odds"].to_numpy(float)
    alpha = float(spec.get("win_weight_alpha", 0.0))
    pred = {t: np.full(len(meta), np.nan) for t in ys}
    t0 = time.time()
    vys = tuple(int(y) for y in a.valid_years.split(",")) if getattr(a, "valid_years", "") else (2022, 2023, 2024)
    if off == "fl" and min(vys) <= max(BETA_YEARS):
        raise SystemExit(f"offset fl の β は {min(BETA_YEARS)}〜{max(BETA_YEARS)} 年で当てているので、検証年 {vys} には使えない（raw にする）")
    for vy in vys:
        if not (year == vy).any():
            raise SystemExit(f"検証年 {vy} の行が無い")
    for tr, va in mc.folds_from_meta(meta, valid_years=vys):
        for t, y in ys.items():
            w = None
            if alpha:                                   # 第 3 段: 当たった馬（勝ち・連対・複勝の各目的で正例）の重み ∝ オッズ^α（オッズの無い馬は 1）
                o = np.where(np.isfinite(odds_row[tr]) & (odds_row[tr] > 0), odds_row[tr], 1.0)
                w = np.where(y[tr] == 1, o ** alpha, 1.0)
                w = w / w.mean()
            acc = np.zeros(len(va))
            for s in range(a.seeds):
                mtr = None if margins is None else margins[t][tr] + (rng.normal(0.0, noise, len(tr)) if noise else 0.0)
                acc += fit_pred(X.iloc[tr], y[tr], X.iloc[va], params, s, mtr, None if margins is None else margins[t][va], w)
            pred[t][va] = acc / a.seeds
        log(f"{a.name} fold {int(year[va][0])} 済み（{time.time() - t0:.0f}s）")
    ok = np.isfinite(pred["win"])
    o = meta.loc[ok, KEY + ["year"]].copy().reset_index(drop=True)
    rr = race[ok]
    o["p_win"] = norm_race(rr, pred["win"][ok], 1.0)
    o["p_top2"] = norm_race(rr, pred["top2"][ok], 2.0)
    o["p_top3"] = norm_race(rr, pred["top3"][ok], 3.0)
    o["q"] = q[ok]
    o["r"] = np.log(o["p_win"] / o["q"])
    out.parent.mkdir(parents=True, exist_ok=True)
    o.to_parquet(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({"name": a.name, "spec": spec, "seeds": a.seeds, "cols": int(X.shape[1]), "seconds": round(time.time() - t0),
                                                    "valid_years": list(vys)},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{a.name}: cols={X.shape[1]} {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- 2026 年（推論の経路の入力）での確かめ
def cmd_fitfull(a) -> None:
    """2016〜2025 年の全行で学習して保存する（2026 年の推論の入力に当てるため）。spec は member と同じ。出力 MK/full/NAME/。"""
    import xgboost as xgb

    from src.pipeline.models import zoo

    spec = json.loads(a.spec) if a.spec else {}
    bad = set(spec) - MEMBER_KEYS
    if bad:
        raise SystemExit(f"この道具が対応しない spec: {sorted(bad)}")
    if spec.get("offset", "none") not in ("none", "raw"):
        raise SystemExit("全期間の版の offset は none・raw だけ（2026 年の発走前オッズの q は raw で作るため）")
    out0 = MK / "full" / a.name
    if (out0 / "meta.json").exists():
        raise SystemExit(f"{out0} は作成済み（上書きしない。名前を分ける）")
    if a.fit_end == 2024 and any(p_.joinpath("test2025.parquet").exists() for p_ in (MK / "full").glob("*")):
        log("注意: ほかの名前で 2025 年 test が作成済み（同じ版を 2 度見ないこと）")
    X, meta, info = mc.load_cache()
    n0 = X.shape[1]
    X = mc.apply_spec(X, meta, {k: v for k, v in spec.items() if k not in ("offset", "win_weight_alpha", "params", "offset_noise_sd")})
    if n0 - X.shape[1] < BASE_DROP_N:
        raise SystemExit(f"入力の列が {n0} → {X.shape[1]}（土台の 38 列の除外が spec に無い）")
    check_no_market_features(X.columns)
    params = {**info["params"], **spec.get("params", {})}
    race = meta["race_id"].to_numpy()
    fin = meta["finish_position"].to_numpy().astype(float)
    if a.fit_end == 2024:                               # 2025 年 test を 1 回だけ予測する版（2025-01-01 の 10 日前まで学習。最終の束と同じ切り方）
        dates = pd.to_datetime(meta["date"])
        tr = np.flatnonzero(np.isfinite(fin) & (dates < pd.Timestamp("2025-01-01") - pd.Timedelta(days=10)).to_numpy())
        if (MK / "full" / a.name / "test2025.parquet").exists():
            raise SystemExit("2025 年 test は作成済み（1 回きり）")
    else:
        tr = np.flatnonzero(np.isfinite(fin) & (meta["year"].to_numpy() <= 2025))
    off = spec.get("offset", "none")
    q = load_market(meta, "raw" if off == "none" else off)
    q2, q3 = zoo.harville_topk(race, q, 1.0, 1.0)
    q2, q3 = norm_race(race, q2, 2.0), norm_race(race, q3, 3.0)
    margins = {"win": logit(q), "top2": logit(np.clip(q2, 1e-5, 0.999)), "top3": logit(np.clip(q3, 1e-5, 0.999))}
    ys = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    noise = float(spec.get("offset_noise_sd", 0.0))
    rng = np.random.default_rng(12345)
    out = out0
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for t, y in ys.items():
        for s in range(a.seeds):
            p = dict(params)
            m = xgb.XGBClassifier(n_estimators=p.get("n_estimators", 400), learning_rate=p.get("learning_rate", 0.05), max_depth=p.get("max_depth", 6),
                                  subsample=p.get("subsample", 0.8), colsample_bytree=p.get("colsample_bytree", 0.7), reg_lambda=p.get("reg_lambda", 5.0),
                                  min_child_weight=p.get("min_child_weight", 20), tree_method="hist", device=p.get("device", "cuda"),
                                  enable_categorical=True, random_state=s, n_jobs=4, verbosity=0)
            mtr = None if off == "none" else margins[t][tr] + (rng.normal(0.0, noise, len(tr)) if noise else 0.0)
            m.fit(X.iloc[tr], y[tr], base_margin=mtr)
            m.save_model(str(out / f"{t}_s{s}.json"))
        log(f"fitfull {a.name} {t} 済み（{time.time() - t0:.0f}s）")
    cats = {c: [str(v) for v in X[c].cat.categories] for c in X.columns if str(X[c].dtype) == "category"}
    (out / "meta.json").write_text(json.dumps({"name": a.name, "spec": spec, "seeds": a.seeds, "feature_cols": list(X.columns), "cat_levels": cats,
                                               "train_rows": int(len(tr)), "fit_end": a.fit_end}, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.fit_end == 2024:
        va = np.flatnonzero(np.isfinite(fin) & (meta["year"].to_numpy() == 2025))
        P = predict_full(a.name, X.iloc[va].reset_index(drop=True), race[va], q[va])
        o = meta.iloc[va][KEY + ["year", "finish_position"]].reset_index(drop=True)
        o[["p_win", "p_top2", "p_top3"]] = P.to_numpy()
        o["q"] = q[va]
        o["r"] = np.log(o["p_win"] / o["q"])
        o.to_parquet(out / "test2025.parquet", index=False)
        log(f"2025 年 test: {len(o)} 行 → {out / 'test2025.parquet'}")


def predict_full(name: str, X: pd.DataFrame, race: np.ndarray, q: np.ndarray) -> pd.DataFrame:
    """保存した全期間のモデルを、入力 X（学習と同じ列・型）と市場の勝率 q（offset）で当てる。"""
    import xgboost as xgb

    from src.pipeline.models import zoo

    d = MK / "full" / name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    off = meta["spec"].get("offset", "none")
    q2, q3 = zoo.harville_topk(race, q, 1.0, 1.0)
    q2, q3 = norm_race(race, q2, 2.0), norm_race(race, q3, 3.0)
    margins = {"win": logit(q), "top2": logit(np.clip(q2, 1e-5, 0.999)), "top3": logit(np.clip(q3, 1e-5, 0.999))}
    out = {}
    for t, tot in (("win", 1.0), ("top2", 2.0), ("top3", 3.0)):
        acc = np.zeros(len(X))
        for s in range(meta["seeds"]):
            m = xgb.XGBClassifier(enable_categorical=True)
            m.load_model(str(d / f"{t}_s{s}.json"))
            acc += m.predict_proba(X[meta["feature_cols"]], base_margin=None if off == "none" else margins[t])[:, 1]
        out[f"p_{t}"] = norm_race(race, acc / meta["seeds"], tot)
    return pd.DataFrame(out)


def cmd_apply2026(a) -> None:
    """2026 年の推論の入力（stg の作り直しの書き出し）に、全期間のモデルを当てる。offset は確定オッズと、発走前のオッズ（t60・t15・t05）で替える。
    発走前のオッズがそろうレース（手元の pre_odds.parquet）だけで、◎ の選び方ごとの成績（単勝は確定オッズ × 100 円で近似）と、◎ の重なりを比べる。"""
    X0, _meta, _ = mc.load_cache()
    xi = mc.load_inf_x()
    xi = xi.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    fm = json.loads((MK / "full" / a.name / "meta.json").read_text(encoding="utf-8"))
    nr = mc.newcomer_races_2026()
    meta_i = pd.DataFrame({"race_class": np.where(xi["race_id"].isin(nr), "新馬", "")})   # 学習と同じ書き換え（推論の行は 2026 年なので blank_before は当たらない）
    X = mc.apply_spec(mc.cast_like(xi, X0), meta_i, {k: v for k, v in fm["spec"].items() if k not in ("offset", "win_weight_alpha", "params", "offset_noise_sd")})
    del X0
    r = pd.read_parquet(mc.TABLES / "2026/race_result_flat.parquet", columns=["race_id", "horse_id", "horse_number", "odds", "popularity", "finish_position"])
    for c in ("race_id", "horse_id"):
        r[c] = r[c].astype(str)
    for c in ("odds", "popularity", "finish_position", "horse_number"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.drop_duplicates(KEY)
    base = xi[KEY].merge(r, on=KEY, how="left")
    # 受け渡し用: 推論の入力のある全レースに、確定オッズの offset で当てた予測（走った馬＝確定オッズのある馬だけ。取消の馬は外し、レースは残す）
    run_ok = (base["odds"] > 0)
    sa = (run_ok & (run_ok.groupby(base["race_id"]).transform("sum") >= 3)).to_numpy()
    X, base = X.loc[run_ok.to_numpy()].reset_index(drop=True), base.loc[run_ok.to_numpy()].reset_index(drop=True)
    sa = sa[run_ok.to_numpy()]
    Xa, ba = X.loc[sa].reset_index(drop=True), base.loc[sa].reset_index(drop=True)
    ra = ba["race_id"].to_numpy()
    qa = q_from_odds(ra, ba["odds"].to_numpy())
    Pa = predict_full(a.name, Xa, ra, np.clip(qa, 1e-5, 1.0))
    hand = [ba[KEY + ["finish_position"]].assign(year=2026, p_win=Pa["p_win"].to_numpy(), p_top2=Pa["p_top2"].to_numpy(), p_top3=Pa["p_top3"].to_numpy(),
                                                   q=qa, r=np.log(Pa["p_win"].to_numpy() / qa), odds_phase="final")]
    del Xa
    # 比べるレース: 4 つの時点（t60・t15・t05 と確定）すべてで、走った馬の発走前オッズがそろうレース（時点ごとに欠けを埋めない）
    rid_ok = set(base["race_id"])
    cover = {}
    for ph in ("t60", "t15", "t05"):
        po = load_pre_odds(ph)
        j = base[["race_id", "horse_number"]].merge(po, on=["race_id", "horse_number"], how="left")
        has = j.groupby("race_id")["q"].apply(lambda s_: bool(s_.notna().all()))
        cover[ph] = int(has.sum())
        rid_ok &= set(has[has].index)
    sel = base["race_id"].isin(rid_ok).to_numpy()
    X, base = X.loc[sel].reset_index(drop=True), base.loc[sel].reset_index(drop=True)
    race = base["race_id"].to_numpy()
    q_fin = q_from_odds(race, base["odds"].to_numpy())
    res = {"races": int(len(set(race))), "races_with_phase": cover, "phases": {}}
    picks = {}
    for ph in ("final", "t60", "t15", "t05"):
        if ph == "final":
            q = q_fin
        else:
            po = load_pre_odds(ph)
            q = base[["race_id", "horse_number"]].merge(po, on=["race_id", "horse_number"], how="left")["q"].to_numpy()
            q = norm_race(race, q)                              # 比べるレースは全馬のオッズがそろう（取消の馬を除いて正規化し直す）
        P = predict_full(a.name, X, race, np.clip(q, 1e-5, 1.0))
        if ph != "final":                                       # 発走前のオッズの offset の版（そのオッズの q と、その q で測ったずれ）
            hand.append(base[KEY + ["finish_position"]].assign(year=2026, p_win=P["p_win"].to_numpy(), p_top2=P["p_top2"].to_numpy(), p_top3=P["p_top3"].to_numpy(),
                                                               q=q, r=np.log(P["p_win"].to_numpy() / q), odds_phase=ph))
        d = base[KEY + ["horse_number", "popularity", "finish_position", "odds"]].copy()
        d[["p_win", "p_top2", "p_top3"]] = P.to_numpy()
        d["q_used"] = q
        d["r"] = np.log(d["p_win"].clip(1e-9) / d["q_used"].clip(1e-9))   # 選ぶときのずれは、その時点で見えている q で
        if ph != "final":                                       # 人気（1 番人気を除く規則に使う）も、その時点の q の順位で作り直す（確定の人気は発走前に見えない）
            d["popularity"] = d.groupby("race_id")["q_used"].rank(ascending=False, method="first")
        d["ret_tan"] = np.where(d["finish_position"] == 1, d["odds"] * 100, 0.0)
        d["ret_fuku"] = np.nan
        nwin = d.groupby("race_id")["finish_position"].transform(lambda s_: (s_ == 1).sum())
        d = d[nwin == 1]
        rr = {}
        for ru in ("p", "nofav", "r_p05", "r_p05_nofav", "gate:0.1"):
            s_ = pick(d, ru)
            picks[(ph, ru)] = set(zip(s_["race_id"], s_["horse_id"]))
            sm = summarize(s_.assign(ret_fuku=0.0))
            sm.pop("roi_fuku")
            rr[ru] = sm
        res["phases"][ph] = rr
    for ph in ("t60", "t15", "t05"):
        for ru in ("p", "r_p05", "gate:0.1"):
            a_, b_ = picks[(ph, ru)], picks[("final", ru)]
            res["phases"][ph][ru]["same_pick_as_final"] = len(a_ & b_) / max(len(b_), 1)
    out = MK / "eval2026"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(hand, ignore_index=True).to_parquet(out / f"{a.name}_pred2026.parquet", index=False)
    res["pred2026"] = {"path": str(out / f"{a.name}_pred2026.parquet"), "rows_by_phase": {h["odds_phase"].iloc[0]: int(len(h)) for h in hand},
                       "races_final": int(hand[0]["race_id"].nunique()), "days": int(xi.attrs.get("days", 0))}
    (out / f"{a.name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for ph, rr in res["phases"].items():
        for ru in ("p", "r_p05", "gate:0.1"):
            s_ = rr[ru]
            log(f"{ph} {ru}: {s_['races']}R ◎1番人気 {s_['fav_share']:.3f} 勝 {s_['win']:.3f} 単 {s_['roi_tan']:.3f}" + (f" 確定と同じ ◎ {s_['same_pick_as_final']:.3f}" if "same_pick_as_final" in s_ else ""))


def cmd_handoff(a) -> None:
    """bet-optimizer への受け渡し: 2019〜2024 年の OOF（2020 年は追い切りを欠けにそろえた版）・2025 年 test・2026 年（確定・発走前のオッズの offset）を 1 つの表に。
    列: race_id・horse_id・year・finish_position・p_win・p_top2・p_top3・q（offset に使った市場の勝率）・r（log(p_win/q)）・odds_phase（final・t60・t15・t05）・part。"""
    m = pd.read_parquet(mc.CYC / "meta.parquet", columns=KEY + ["finish_position"])
    m["race_id"], m["horse_id"] = m["race_id"].astype(str), m["horse_id"].astype(str)
    parts = []
    if len({a.oof_main, a.oof_1921, a.oof_2020}) != 3:
        raise SystemExit("3 本の OOF は別の名前にする")
    want = {a.oof_main: {2022, 2023, 2024}, a.oof_1921: {2019, 2021}, a.oof_2020: {2020}}
    specs = {}
    for name, part in ((a.oof_main, "oof"), (a.oof_1921, "oof"), (a.oof_2020, "oof")):
        o = pd.read_parquet(MK / "oof" / f"{name}.parquet")
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        o = o[o["year"].isin(want[name])]
        if set(o["year"].unique()) != want[name]:
            raise SystemExit(f"{name} の年 {sorted(o['year'].unique())} が期待 {sorted(want[name])} と違う")
        sp = json.loads((MK / "oof" / f"{name}.json").read_text(encoding="utf-8"))["spec"]
        specs[name] = {k: v for k, v in sp.items() if k != "blank_before"}
        o = o.merge(m, on=KEY, how="left")
        parts.append(o.assign(odds_phase="final", part=part, source=name))
    for nm in (a.full2024, a.full2025):
        sp = json.loads((MK / "full" / nm / "meta.json").read_text(encoding="utf-8"))["spec"]
        specs[nm] = {k: v for k, v in sp.items() if k != "blank_before"}
    if len({json.dumps(v, sort_keys=True) for v in specs.values()}) != 1:
        raise SystemExit(f"5 つの版の spec（blank_before 以外）がそろわない: {specs}")
    t = pd.read_parquet(MK / "full" / a.full2024 / "test2025.parquet")
    parts.append(t.assign(odds_phase="final", part="test2025", source=a.full2024))
    p26 = pd.read_parquet(MK / "eval2026" / f"{a.full2025}_pred2026.parquet")
    parts.append(p26.assign(part="infer2026", source=a.full2025))
    cols = ["race_id", "horse_id", "year", "finish_position", "p_win", "p_top2", "p_top3", "q", "r", "odds_phase", "part", "source"]
    h = pd.concat([x[cols] for x in parts], ignore_index=True)
    dup = h.duplicated(["race_id", "horse_id", "odds_phase"]).sum()
    if dup:
        raise SystemExit(f"同じ馬・時点の行が重複 {dup}")
    out = MK / "handoff"
    out.mkdir(parents=True, exist_ok=True)
    h.to_parquet(out / f"{a.tag}.parquet", index=False)
    summ = h.groupby(["part", "year", "odds_phase"]).agg(rows=("horse_id", "size"), races=("race_id", "nunique")).reset_index()
    (out / f"{a.tag}.json").write_text(json.dumps({"rows": len(h), "by": summ.to_dict("records"), "note": a.note}, ensure_ascii=False, indent=1, default=int), encoding="utf-8")
    log(f"handoff {a.tag}: {len(h)} 行\n{summ.to_string(index=False)}")


# ---------------------------------------------------------------- 評価
def read_any(name: str) -> pd.DataFrame:
    p = MK / "oof" / f"{name}.parquet"
    d = pd.read_parquet(p) if p.exists() else mc.read_pred(name)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    return d


def frame(name: str) -> pd.DataFrame:
    """予測と市場・着順・払戻（単勝・複勝）をそろえた馬ごとの表。ずれ r は、どのモデルも同じ市場の q_raw（確定オッズ）で測る（尺度をそろえる）。
    走った馬（確定オッズあり）が予測に全部そろうレースだけ。オッズの無い馬が走ったレースは外す。"""
    from src.scripts.data import rank_bakeoff_eval as rbe

    d = read_any(name).drop(columns=[c for c in ("q", "r") if c in read_any(name).columns])
    m = pd.read_parquet(MK / "market.parquet")
    runners = m[m["finish_position"].notna() & (m["finish_position"] > 0)]
    no_odds = set(runners.loc[~(runners["odds"] > 0), "race_id"])
    n_run = runners[runners["odds"] > 0].groupby("race_id").size()
    d = d.merge(m[KEY + ["horse_number", "odds", "popularity", "finish_position", "q_raw"]], on=KEY, how="inner")
    d = d[(d["odds"] > 0) & ~d["race_id"].isin(no_odds)].copy()       # 走った馬（取消は確定オッズが無い）
    d = d[d.groupby("race_id")["horse_id"].transform("count") == d["race_id"].map(n_run)]   # 走った馬が全部そろうレースだけ
    d["p_win"] = d["p_win"] / d.groupby("race_id")["p_win"].transform("sum")
    d["q"] = d["q_raw"] / d.groupby("race_id")["q_raw"].transform("sum")
    d["r"] = np.log(d["p_win"].clip(1e-9) / d["q"].clip(1e-9))
    if "p_top2" not in d.columns:
        d["p_top2"] = np.nan
    pay = rbe.load_payoff()
    hn = d["horse_number"].astype(int).astype(str)
    d["ret_tan"] = [pay.get((r, "tansho"), {}).get(h, 0) for r, h in zip(d["race_id"], hn)]
    d["ret_fuku"] = [pay.get((r, "fukusho"), {}).get(h, 0) for r, h in zip(d["race_id"], hn)]
    has = np.array([((r, "tansho") in pay) and ((r, "fukusho") in pay) for r in d["race_id"]])
    d = d[has]
    nwin = d.groupby("race_id")["finish_position"].transform(lambda s: (s == 1).sum())
    return d[(nwin == 1) & (d.groupby("race_id")["horse_id"].transform("count") >= 3)]


def pick(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    """◎ を 1 レース 1 頭選ぶ。rule: p（p_win の 1 位）・r（ずれの 1 位）・r_p05（p_win ≥ 0.05 の中でずれの 1 位）・
    nofav（1 番人気を除いた p_win の 1 位）・gate:τ（p_win の 1 位が 1 番人気なら、ずれ r ≥ τ のときだけ採り、そうでなければ 1 番人気を除いた 1 位）。"""
    d = d.copy()
    if rule == "p":
        key = d["p_win"]
    elif rule == "r":
        key = d["r"]
    elif rule == "r_p05":                                   # p_win ≥ 0.05 の馬がいないレースは p_win の 1 位
        has = d.groupby("race_id")["p_win"].transform("max") >= 0.05
        key = d["r"].where(d["p_win"] >= 0.05, -np.inf).where(has, d["p_win"])
    elif rule == "nofav":
        key = d["p_win"].where(d["popularity"] != 1, -np.inf)
    elif rule == "r_p05_nofav":                             # 1 番人気は ◎ にしない（例外なし）＋ p ≥ 0.05 の中でずれの 1 位
        ok = (d["p_win"] >= 0.05) & (d["popularity"] != 1)
        has = ok.groupby(d["race_id"]).transform("any")
        key = d["r"].where(ok, -np.inf).where(has, d["p_win"].where(d["popularity"] != 1, -np.inf))
    elif rule.startswith("gate:"):
        tau = float(rule.split(":")[1])
        top = d.loc[d.groupby("race_id")["p_win"].idxmax()]
        allow = set(top.loc[(top["popularity"] != 1) | (top["r"] >= tau), "race_id"])
        key = d["p_win"].where(d["race_id"].isin(allow) | (d["popularity"] != 1), -np.inf)
    else:
        raise ValueError(rule)
    d["_k"] = key
    return d.loc[d.groupby("race_id")["_k"].idxmax()]


def summarize(sel: pd.DataFrame) -> dict:
    n = len(sel)
    return {"races": n, "fav_share": float((sel["popularity"] == 1).mean()), "win": float((sel["finish_position"] == 1).mean()),
            "place": float((sel["finish_position"] <= 3).mean()), "rentai": float((sel["finish_position"] <= 2).mean()),
            "roi_tan": float(sel["ret_tan"].sum() / (100 * n)), "roi_fuku": float(sel["ret_fuku"].sum() / (100 * n)),
            "mean_pop": float(sel["popularity"].mean())}


def boot_roi(a: pd.DataFrame, b: pd.DataFrame, col: str, n_boot: int = 1000, seed: int = 0) -> list[float]:
    """同じレースでの ◎ の回収率の差（a − b）の 95% 区間。"""
    m = a[["race_id", col]].merge(b[["race_id", col]], on="race_id", suffixes=("_a", "_b"))
    x = (m[f"{col}_a"] - m[f"{col}_b"]).to_numpy() / 100.0
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return [float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def cmd_report(a) -> None:
    rules = ["p", "nofav", "r", "r_p05", "r_p05_nofav", "gate:0", "gate:0.1", "gate:0.2", "gate:0.3"]
    SELECT_YEAR, CHECK_YEARS = 2022, (2023, 2024)         # 選び方・閾値は 2022 年で決め、2023〜2024 年で確かめる（選択の偏りを避ける）
    names = [a.name] + [x for x in a.ref.split(",") if x]
    F = {n: frame(n) for n in names}
    races = set.intersection(*[set(f["race_id"]) for f in F.values()])
    res = {"races": len(races), "models": {}}
    base_sel = None
    for n in names:
        d = F[n][F[n]["race_id"].isin(races)]
        r_ = {"rules": {}}
        for ru in rules:
            sel = pick(d, ru)
            s = summarize(sel)
            if n == names[0] and ru == "p":
                base_sel = sel
            if base_sel is not None and not (n == names[0] and ru == "p"):
                s["roi_tan_vs_first_p"] = boot_roi(sel, base_sel, "ret_tan")
                s["roi_fuku_vs_first_p"] = boot_roi(sel, base_sel, "ret_fuku")
            r_["rules"][ru] = s
        # ずれの分位（全馬・10 分位）× 勝率・回収率（その馬に単勝・複勝 100 円）
        dd = d.copy()
        dd["dec"] = pd.qcut(dd["r"].rank(method="first"), 10, labels=False)
        r_["r_deciles"] = [{"decile": int(k), "n": int(len(g)), "r_mean": float(g["r"].mean()), "win": float((g["finish_position"] == 1).mean()),
                            "roi_tan": float(g["ret_tan"].sum() / (100 * len(g))), "roi_fuku": float(g["ret_fuku"].sum() / (100 * len(g)))} for k, g in dd.groupby("dec")]
        # 予測 1 位が 1 番人気のレース: 確信度（p1・p_top2・r）の 5 分位ごとの ◎ の連対率
        top = pick(d, "p")
        fav = top[top["popularity"] == 1]
        conf = {}
        for c in ("p_win", "p_top2", "r"):
            if fav[c].notna().sum() < 50:
                continue
            qd = pd.qcut(fav[c].rank(method="first"), 5, labels=False)
            conf[c] = [{"quintile": int(k), "n": int(len(g)), f"{c}_min": float(g[c].min()), f"{c}_max": float(g[c].max()),
                        "rentai": float((g["finish_position"] <= 2).mean()), "win": float((g["finish_position"] == 1).mean())} for k, g in fav.groupby(qd)]
        tails = {}
        for c in ("p_win", "p_top2"):                      # 確信度の上位 20%・10%・5%（ほぼすべてが勝ち負けの帯があるか）。境目は検証の 3 年をまとめて決めた目安
            if fav[c].notna().sum() < 50:
                continue
            for qq in (0.8, 0.9, 0.95):
                cut = float(fav[c].quantile(qq))
                g = fav[fav[c] >= cut]
                tails[f"{c}_top{int(round((1 - qq) * 100))}"] = {"cut": cut, "n": int(len(g)), "rentai": float((g["finish_position"] <= 2).mean()),
                                                                 "win": float((g["finish_position"] == 1).mean())}
        r_["fav_top_races"] = {"n": int(len(fav)), "share": float(len(fav) / len(top)), "rentai": float((fav["finish_position"] <= 2).mean()), "by_conf": conf,
                               "tails": tails}
        # 選び方は 2022 年の単勝の回収率で決め、2023〜2024 年で確かめる
        by_year = {}
        for ru in rules:
            sel = pick(d, ru)
            sy = sel["race_id"].str[:4].astype(int)
            by_year[ru] = {"select": summarize(sel[sy == SELECT_YEAR]), "check": summarize(sel[sy.isin(CHECK_YEARS)])}
        cand_r = [ru for ru in rules if ru != "r" and by_year[ru]["select"]["races"] > 0]
        best = max(cand_r, key=lambda ru: by_year[ru]["select"]["roi_tan"]) if cand_r else None
        r_["select_check"] = {"select_year": SELECT_YEAR, "check_years": list(CHECK_YEARS), "chosen": best, "by_rule": by_year}
        res["models"][n] = r_
    d_ = MK / "eval"
    d_.mkdir(parents=True, exist_ok=True)
    (d_ / f"{a.name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for n in names:
        for ru in ("p", "nofav", "r_p05", "gate:0.1"):
            s = res["models"][n]["rules"][ru]
            log(f"{n} {ru}: ◎1番人気 {s['fav_share']:.3f} 勝 {s['win']:.3f} 複 {s['place']:.3f} 単 {s['roi_tan']:.3f} 複回 {s['roi_fuku']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("market")
    sub.add_parser("drift")
    m = sub.add_parser("member")
    m.add_argument("name")
    m.add_argument("--spec", default="")
    m.add_argument("--seeds", type=int, default=1)
    m.add_argument("--force", action="store_true")
    m.add_argument("--valid-years", default="", help="検証の年（既定 2022,2023,2024）。例 2019,2021")
    e = sub.add_parser("report")
    e.add_argument("name")
    e.add_argument("--ref", default="ens5_p01s3,prod_v6e")
    f = sub.add_parser("fitfull", help="2016〜2025 年の全行で学習して保存（2026 年に当てる用）")
    f.add_argument("name")
    f.add_argument("--spec", default="")
    f.add_argument("--seeds", type=int, default=3)
    f.add_argument("--fit-end", type=int, choices=[2024, 2025], default=2025, help="2024: 2025 年 test を 1 回だけ予測する版")
    g = sub.add_parser("apply2026", help="全期間のモデルを 2026 年の推論の入力に当て、確定・発走前（t60・t15・t05）のオッズの offset で比べる")
    g.add_argument("name")
    h = sub.add_parser("handoff", help="bet-optimizer への受け渡しの表を作る")
    h.add_argument("tag")
    h.add_argument("--oof-main", required=True, help="2022〜2024 年の OOF（oof/NAME）")
    h.add_argument("--oof-1921", required=True, help="2019・2021 年の OOF")
    h.add_argument("--oof-2020", required=True, help="2020 年の OOF（追い切りを欠けにそろえた版）")
    h.add_argument("--full2024", required=True, help="2016〜2024 で学習した版（full/NAME/test2025.parquet）")
    h.add_argument("--full2025", required=True, help="2016〜2025 で学習した版（eval2026/NAME_pred2026.parquet）")
    h.add_argument("--note", default="")
    a = ap.parse_args()
    {"market": cmd_market, "member": cmd_member, "report": cmd_report, "drift": cmd_drift, "fitfull": cmd_fitfull, "apply2026": cmd_apply2026, "handoff": cmd_handoff}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

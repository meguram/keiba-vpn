"""めぐサイクルの土台: 2016〜2024 年の走ごとの表（as-of の列・確定オッズ・T-60 の再現オッズ・払戻）と、市場の列を使わない自前の確率。

- 元の表は全レース版の frame（<COURSE_OUT>/v3/frame.parquet）。2025 年の行が入っているので、読むときに year<=2024 で絞り、assert する。
- 判定のオッズは T-60 の再現（bet_cycles.load_drift と同じ作り: 2026 年の発走前オッズと確定オッズの列だけを読み、着順・払戻は読まない）。
  2026 年のレースを 1 つ丸ごと借り、確定の人気の順位ごとのずれ d を当てて「確定 × exp(−d)」を作る。1/オッズの合計は元のレースに合わせる。5 通り。
- 自前の確率は、src/research/graded/catalog.py の ALLOWED_COLUMNS（市場・その走の結果の列を含まないことをテストで検査済み）から、
  めぐ指数（mg_rank_*）と日付の文字列を除いた列で、年ごとに前へ進める学習（y 年は y−1 年までで学習）の LightGBM。勝ち・3 着内の 2 本。
  ハイパーパラメータは固定（PARAMS）。2022〜24 年を見て選び直さない。
- 2022・23 年の着順は、2023・24 年の予測の学習の教師に使う（前へ進める学習の通常の作り）。成績はここでは数えない。台帳に reproduce を申告する。

費用 0 円（手元の表だけ。GCS を読まない）。重いので nice -n 19 で流す。
  python -m src.research.megu_cycles.mc_data build
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

ROOT = Path(__file__).resolve().parents[3]
COURSE_OUT = Path(os.environ.get("COURSE_OUT") or "/home/jovyan/work/course_out")
FRAME = COURSE_OUT / "v3" / "frame.parquet"
OUT = Path(os.environ.get("MEGU_CYCLES_OUT") or "/home/jovyan/work/megu_cycles_out")
TABLE = OUT / "runners.parquet"
GOAL = ROOT / "config/megu_cycles/goal.json"
YEAR_END = 2024
N_PRE = 5
GRADED_CLS = (6, 7, 8)
EXCLUDE_FEATURES = {"mg_rank_p1", "mg_rank_max5", "dt", "p_dt", "ok_eval", "jockey_name"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=6, seed=20261123,
              max_cat_to_onehot=8, cat_smooth=50, min_data_per_group=200)
N_ROUNDS = 500
KEEP = ["race_id", "horse_id", "date", "year", "month", "venue", "surf", "dist", "grade", "cls", "track", "field", "field_decl", "hn", "fin",
        "y1", "y3", "ret_tan", "ret_fuku", "pay_ok", "jockey_id", "trainer_id", "age", "sex", "starts_prev", "interval", "p_fin", "blinker",
        "p_blinker"]


def feature_columns() -> list[str]:
    from src.research.graded.catalog import ALLOWED_COLUMNS
    return sorted(c for c in ALLOWED_COLUMNS if c not in EXCLUDE_FEATURES)


def load_frame() -> pd.DataFrame:
    """frame を 2024 年までの行だけ読む（2025 年の行は開かない）。"""
    cols = sorted(set(KEEP) | set(feature_columns()))
    d = pd.read_parquet(FRAME, columns=cols, filters=[("year", "<=", YEAR_END)])
    yr = pd.to_numeric(d["year"]).to_numpy()
    assert yr.max() <= YEAR_END, "2025 年以降の行が入っている"
    return d


def attach_odds(d: pd.DataFrame) -> pd.DataFrame:
    from src.research.course.v2 import attach_final_odds
    d = attach_final_odds(d, range(2016, YEAR_END + 1))
    d = d[(d["odds_final"] > 0)].copy()
    inv = 1.0 / d["odds_final"].to_numpy(float)
    d["q_final"] = inv / pd.Series(inv).groupby(d["race_id"].to_numpy()).transform("sum").to_numpy()
    d["graded"] = pd.to_numeric(d["cls"], errors="coerce").isin(GRADED_CLS)
    return d


def simulate_pre(d: pd.DataFrame, n: int = N_PRE) -> np.ndarray:
    """T-60 の再現オッズ（n 通り）。bet_cycles.simulate_pre_period と同じ作り（レースを丸ごと借り、確定の人気の順位のずれを当てる）。"""
    from src.scripts.data.bet_cycles import load_drift
    pool = load_drift("t60")["races"]
    rid = d["race_id"].to_numpy()
    odds = d["odds_final"].to_numpy(float)
    pop = np.nan_to_num(d["pop_final"].to_numpy(float), nan=99.0)
    out = np.empty((n, len(d)))
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    ends = np.r_[starts[1:], len(d)]
    for k in range(n):
        rng = np.random.default_rng(1000 + k)
        for s, e in zip(starts, ends):
            src = pool[int(rng.integers(len(pool)))]
            rk = np.clip(np.where(pop[s:e] < 99, pop[s:e], len(src)).astype(int) - 1, 0, len(src) - 1)
            o = odds[s:e] * np.exp(-src[rk])
            o = o * (np.sum(1.0 / o) / np.sum(1.0 / odds[s:e]))
            out[k, s:e] = o
    return out


def walk_forward(d: pd.DataFrame, target: str, years=range(2017, YEAR_END + 1)) -> np.ndarray:
    import lightgbm as lgb
    feats = feature_columns()
    X = d[feats].copy()
    for c in feats:
        if X[c].dtype == object or str(X[c].dtype) in ("string", "category", "bool", "boolean"):
            X[c] = X[c].astype("string").astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    y = pd.to_numeric(d[target], errors="coerce").fillna(0).to_numpy(int)
    yr = d["year"].to_numpy(int)
    p = np.full(len(d), np.nan)
    for t in years:
        tr, te = yr < t, yr == t
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), N_ROUNDS)
        p[te] = m.predict(X[te])
        print(f"  {target} {t}: 学習 {tr.sum()} 行 → 予測 {te.sum()} 行", flush=True)
    return p


def cmd_build() -> None:
    from src.ops import data_ledger as DL
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with DL.guard(tool="src/research/megu_cycles/mc_data.py:build", family="megu_cycles", version="build-v1", period=("2022-01-01", "2024-12-31"),
                  purpose="reproduce", reads=("着順", "払戻"), frozen_paths=[GOAL],
                  reason="年ごとに前へ進める学習の教師（2022・23 年の着順）と、表の払戻の列を読むだけ。成績は数えない（数えるのは確かめの道具で別に申告）") as tok:
        d = load_frame()
        tok.check_dates(d.loc[d["year"] >= 2022, "date"].astype(str))
        d = d.sort_values(["race_id", "hn"]).reset_index(drop=True)
        d = attach_odds(d).reset_index(drop=True)
        print("frame", len(d), "行", round(time.time() - t0), "秒", flush=True)
        pre = simulate_pre(d)
        for k in range(N_PRE):
            d[f"odds_pre{k}"] = pre[k]
        print("T-60 の再現", round(time.time() - t0), "秒", flush=True)
        for tg, nm in (("y1", "pm_win"), ("y3", "pm_top3")):
            d[nm] = walk_forward(d, tg)
        keep = [c for c in KEEP if c in d.columns] + ["odds_final", "pop_final", "q_final", "graded"] + [f"odds_pre{k}" for k in range(N_PRE)] + ["pm_win", "pm_top3"]
        d[keep].to_parquet(TABLE, index=False)
        tok.finish(result_path=TABLE)
    (OUT / "build.json").write_text(json.dumps({"rows": len(d), "features": feature_columns(), "params": PARAMS, "rounds": N_ROUNDS,
                                                 "sec": round(time.time() - t0)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("build", len(d), "行", round(time.time() - t0), "秒", flush=True)


def load_table(years: tuple[int, int] = (2016, YEAR_END)) -> pd.DataFrame:
    d = pd.read_parquet(TABLE, filters=[("year", ">=", years[0]), ("year", "<=", years[1])])
    assert d["year"].max() <= YEAR_END
    return d



# ---------------------------------------------------------------- 豊富な特徴量の自前の確率（サイクル 4〜）
X_RICH = Path(os.environ.get("MEGU_CYCLES_XRICH") or "/home/jovyan/work/retrain_out/cycles16_fix3/X_xgb.parquet")
META_RICH = X_RICH.parent / "meta.parquet"
RICH_EXCLUDE = {"tl_f_untimed_later"}          # 名前に「後」の値が入りうる列は使わない
# 記録の取れ方が年で変わる列を外す（結果を使わない決まり。サイクル 5 の後に固定）:
# 2016〜21 のどこかの年の欠けでない率が、2022〜24 の平均より 0.15 以上低い列（後の年ほど記録がそろう＝生き残りの偏りの恐れ）と、2020 年からしか無い列。
# 該当: tl_*・wc_*（調教の記録。2016〜19 は 10〜66%）、rb_nk_*（2016〜19 は 0〜21%）、fr_class_level（50〜77%）、comment_*（2020〜）。
# きっかけはサイクル 4 の市場の残差との相関（tl_has_training 2017〜19 で 0.06〜0.11）。cycle_04.md・cycle_06.md
# 加えて、全期間（2020〜2025）の集計の表から作る列 rb_*（めぐ指数 v3a の基準の表。config/megu_v3a/meta.json の ref_years）を外す
# v5（サイクル 11）: mtl_*（上流の補助モデルの出力。読む aux_B.parquet は 11/08 の旧版で、今のコードと同じ作りか確かめられない。
#   2017〜19 の中だけの族の外しで、2018・19 の重みの山 0.41 を 2017 年の水準 0.24〜0.26 に戻す族）と、
#   年の水準が段で変わる列（has_comment・fd_n_a0 は 2020 年から、jockey_weight・fd_jw_* は 2023 年の斤量の規則の変更から。最頻値の割合で決めた）を外す
RICH_EXCLUDE_RE = r"^tl_|^wc_|^rb_|^fr_class_level$|^comment_|^mtl_|^has_comment$|^fd_n_a0$|^jockey_weight$|^fd_jw_"
XGB_PARAMS = dict(objective="binary:logistic", tree_method="hist", device="cuda", max_depth=8, learning_rate=0.03, subsample=0.7,
                  colsample_bytree=0.3, min_child_weight=100, reg_lambda=20.0, eval_metric="logloss", seed=20261123)
XGB_ROUNDS = 800
RICH = OUT / "rich_pred_v5.parquet"          # v1 は調教の列のリークあり。v2 は外す列を 2020〜21 の結果を見て決めた版（サイクル 5）。v3 は結果を使わない決まりの版（rb_mg_* が残る）。v4 は rb_* も外す。v5 は mtl_* と水準の段の列も外す


def cmd_build_rich() -> None:
    """model-cycles の特徴量の表（1,231 列。市場・その走の結果の列を名前で除いた版。2016〜2025 年）で、年ごとに前へ進める XGBoost。
    2025 年の行は読み込んだ直後に落とす（予測にも学習にも使わない）。パラメータは固定（2022〜24 を見て選ばない）。"""
    import re

    import pyarrow.parquet as pq
    import xgboost as xgb
    from src.ops import data_ledger as DL
    t0 = time.time()
    tok = DL.guard(tool="src/research/megu_cycles/mc_data.py:build_rich", family="megu_cycles", version="build-rich-v5", period=("2022-01-01", "2024-12-31"),
                   purpose="reproduce", reads=("着順",), frozen_paths=[GOAL],
                   reason="豊富な特徴量の前へ進める学習の教師（2022・23 年の着順）を読むだけ。成績は数えない")
    # 着順を含む meta は year<=2024 の行だけ読む。X は年の列が無いので、年の列だけ先に読んで行を絞る（X の 2025 年の行は特徴量だけで、結果を含まない）
    yr_all = pd.to_numeric(pq.read_table(META_RICH, columns=["year"]).column("year").to_pandas()).to_numpy()
    keep = yr_all <= YEAR_END
    meta = pd.read_parquet(META_RICH, filters=[("year", "<=", YEAR_END)])
    assert len(meta) == keep.sum()
    names = [c for c in pq.ParquetFile(X_RICH).schema_arrow.names
             if c not in RICH_EXCLUDE and not re.search(RICH_EXCLUDE_RE, c)
             and not re.search(r"odds|pop|ninki|tansho|fukusho|payoff|market|ret_", c, re.I)]
    X = pd.read_parquet(X_RICH, columns=names)
    assert len(X) == len(yr_all)
    X = X[keep].reset_index(drop=True)
    assert pd.to_numeric(meta["year"]).max() <= YEAR_END
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].cat.codes.where(X[c].notna(), np.nan)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.astype(np.float32)
    fin = pd.to_numeric(meta["finish_position"], errors="coerce").to_numpy()
    yr = pd.to_numeric(meta["year"]).to_numpy(int)
    out = meta[["race_id", "horse_id", "year"]].copy()
    for tg, nm, y in (("win", "pr_win", fin == 1), ("top3", "pr_top3", fin <= 3)):
        p = np.full(len(X), np.nan)
        for t in range(2017, YEAR_END + 1):
            tr, te = yr < t, yr == t
            m = xgb.train(XGB_PARAMS, xgb.DMatrix(X[tr], label=y[tr].astype(int)), XGB_ROUNDS)
            p[te] = m.predict(xgb.DMatrix(X[te]))
            print(f"  rich {tg} {t}: 学習 {tr.sum()} → 予測 {te.sum()}（{round(time.time() - t0)} 秒）", flush=True)
        out[nm] = p
    out["race_id"], out["horse_id"] = out["race_id"].astype(str), out["horse_id"].astype(str)
    out.to_parquet(RICH, index=False)
    (OUT / "build_rich.json").write_text(json.dumps({"rows": len(out), "n_features": len(names), "params": XGB_PARAMS, "rounds": XGB_ROUNDS},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    tok.finish(result_path=RICH)
    print("build_rich", len(out), round(time.time() - t0), "秒", flush=True)


# ---------------------------------------------------------------- T-60 の市場を offset にした学習（サイクル 9〜）
OFFSET = OUT / "offset_pred_v4.parquet"      # サイクル 9 は v4 の列（RICH_VERSIONS["v4"]）で作った。作り直すときは版を変える


# 確率の版ごとの、外す列の決まり（再現のため残す。サイクルごとの版は cycles.CYCLE_RICH）
RICH_VERSIONS = {"v1": r"^$", "v2": r"^tl_|^wc_|^rb_nk_", "v3": r"^tl_|^wc_|^rb_nk_|^fr_class_level$|^comment_",
                 "v4": r"^tl_|^wc_|^rb_|^fr_class_level$|^comment_", "v5": RICH_EXCLUDE_RE}


def _rich_matrix(version: str = "v5"):
    """豊富な特徴量の表（版 version の外す列の決まりを当てた版）と meta を、year<=2024 の行だけ返す。"""
    import re

    import pyarrow.parquet as pq
    yr_all = pd.to_numeric(pq.read_table(META_RICH, columns=["year"]).column("year").to_pandas()).to_numpy()
    keep = yr_all <= YEAR_END
    meta = pd.read_parquet(META_RICH, filters=[("year", "<=", YEAR_END)])
    assert len(meta) == keep.sum() and pd.to_numeric(meta["year"]).max() <= YEAR_END
    names = [c for c in pq.ParquetFile(X_RICH).schema_arrow.names
             if c not in RICH_EXCLUDE and not re.search(RICH_VERSIONS[version], c)
             and not re.search(r"odds|pop|ninki|tansho|fukusho|payoff|market|ret_", c, re.I)]
    X = pd.read_parquet(X_RICH, columns=names)
    assert len(X) == len(yr_all)
    assert (meta["race_id"].astype(str).to_numpy() == pd.read_parquet(META_RICH, columns=["race_id"])["race_id"].astype(str).to_numpy()[keep]).all()
    X = X[keep].reset_index(drop=True)
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].cat.codes.where(X[c].notna(), np.nan)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype(np.float32), meta, names


def cmd_build_offset() -> None:
    """勝ちの XGBoost を、T-60 の再現の市場の logit を base_margin（offset）にして学ぶ（市場の知らない分だけを学ぶ。Benter の第 2 段を特徴量ごと）。
    学習の offset は版 0、予測は版 k ごとに版 k の市場を offset にする。年ごとに前へ進める（y 年は y−1 年まで）。"""
    import xgboost as xgb
    from src.ops import data_ledger as DL
    t0 = time.time()
    tok = DL.guard(tool="src/research/megu_cycles/mc_data.py:build_offset", family="megu_cycles", version="build-offset-v1", period=("2022-01-01", "2024-12-31"),
                   purpose="reproduce", reads=("着順",), frozen_paths=[GOAL],
                   reason="T-60 の市場を offset にした前へ進める学習の教師（2022・23 年の着順）を読むだけ。成績は数えない")
    X, meta, names = _rich_matrix("v4")
    tab = pd.read_parquet(TABLE, columns=["race_id", "horse_id", "year", "y1"] + [f"odds_pre{k}" for k in range(N_PRE)])
    key = meta[["race_id", "horse_id"]].astype(str).reset_index().merge(tab.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"])
    X = X.loc[key["index"].to_numpy()].reset_index(drop=True)
    rid = key["race_id"].to_numpy()
    logit = {}
    for k in range(N_PRE):
        inv = 1.0 / key[f"odds_pre{k}"].to_numpy(float)
        q = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
        q = np.clip(q, 1e-4, 1 - 1e-4)
        logit[k] = np.log(q / (1 - q))
    y = key["y1"].fillna(0).to_numpy(int)
    yr = key["year"].to_numpy(int)
    out = key[["race_id", "horse_id", "year"]].copy()
    P = np.full((N_PRE, len(key)), np.nan)
    params = dict(XGB_PARAMS, max_depth=6, colsample_bytree=0.3, min_child_weight=200)
    for t in range(2017, YEAR_END + 1):
        tr, te = yr < t, yr == t
        dtr = xgb.DMatrix(X[tr], label=y[tr], base_margin=logit[0][tr])
        m = xgb.train(params, dtr, 400)
        for k in range(N_PRE):
            P[k, te] = m.predict(xgb.DMatrix(X[te], base_margin=logit[k][te]))
        print(f"  offset {t}: 学習 {tr.sum()} → 予測 {te.sum()}（{round(time.time() - t0)} 秒）", flush=True)
    for k in range(N_PRE):
        out[f"po{k}"] = P[k]
    out.to_parquet(OFFSET, index=False)
    tok.finish(result_path=OFFSET)
    print("build_offset", len(out), round(time.time() - t0), "秒", flush=True)


# ---------------------------------------------------------------- v5 の学習の平均（サイクル 13〜）
RICH_BAG = OUT / "rich_pred_v5bag.parquet"
BAG_SETS = [dict(max_depth=8, colsample_bytree=0.3, subsample=0.7, seed=1), dict(max_depth=6, colsample_bytree=0.5, subsample=0.8, seed=2),
            dict(max_depth=10, colsample_bytree=0.2, subsample=0.6, min_child_weight=300, seed=3)]


def cmd_build_bag() -> None:
    """v5 と同じ列で、設定の違う XGBoost 3 本を年ごとに前へ進めて学び、勝ちの確率を平均する（設定は固定。2022〜24 を見て選ばない）。"""
    import xgboost as xgb
    from src.ops import data_ledger as DL
    t0 = time.time()
    tok = DL.guard(tool="src/research/megu_cycles/mc_data.py:build_bag", family="megu_cycles", version="build-bag-v5", period=("2022-01-01", "2024-12-31"),
                   purpose="reproduce", reads=("着順",), frozen_paths=[GOAL],
                   reason="v5 の列の学習の平均の教師（2022・23 年の着順）を読むだけ。成績は数えない")
    X, meta, names = _rich_matrix()
    fin = pd.to_numeric(meta["finish_position"], errors="coerce").to_numpy()
    yr = pd.to_numeric(meta["year"]).to_numpy(int)
    y = (fin == 1).astype(int)
    P = np.zeros(len(X))
    for i, bs in enumerate(BAG_SETS):
        params = dict(XGB_PARAMS, **bs)
        for t in range(2017, YEAR_END + 1):
            tr, te = yr < t, yr == t
            m = xgb.train(params, xgb.DMatrix(X[tr], label=y[tr]), XGB_ROUNDS)
            P[te] += m.predict(xgb.DMatrix(X[te])) / len(BAG_SETS)
            print(f"  bag {i} {t}（{round(time.time() - t0)} 秒）", flush=True)
    out = meta[["race_id", "horse_id", "year"]].copy()
    out["race_id"], out["horse_id"] = out["race_id"].astype(str), out["horse_id"].astype(str)
    out["pr_win"] = np.where(yr >= 2017, P, np.nan)
    out["pr_top3"] = np.nan
    out.to_parquet(RICH_BAG, index=False)
    tok.finish(result_path=RICH_BAG)
    print("build_bag", len(out), round(time.time() - t0), "秒", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "build_rich", "build_offset", "build_bag"])
    a = ap.parse_args()
    {"build": cmd_build, "build_rich": cmd_build_rich, "build_offset": cmd_build_offset, "build_bag": cmd_build_bag}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())

"""出遅れ率（実ラベル＝戦績ページの備考『出遅』）の特徴量を、V4 全体（tp31 の V4 フレーム 1,062 列）に対して評価する。

python -m src.scripts.data.slowreal_eval labels     # horse_race_remarks（GCS）→ 1 走 1 行のフラグ表 slowreal/remarks_flags.parquet（備考の生テキストは持ち出さない）
python -m src.scripts.data.slowreal_eval coverage   # ラベルの被覆（年・月・場・クラス別）と発生率、被覆の偏り（取れた馬は 2023-10 以降も走った馬）の検査 → coverage.json
python -m src.scripts.data.slowreal_eval build      # 群ごとの特徴量 + 出遅れ確率 p_slow の OOF を slowreal/groups/*.parquet に保存
python -m src.scripts.data.slowreal_eval leakcheck  # 実データの先読み検査（cut 日以降のラベルを消して作り直しても、それより前の特徴量が完全一致）
python -m src.scripts.data.slowreal_eval stage1     # 第 1 段: valid 2024-01〜06（全員のラベルがある期間）、seed 1 本。V4 に対する増分
python -m src.scripts.data.slowreal_eval stage2     # 第 2 段: 通過群を valid 2024-07〜12、新 seed 2 本で確認 + 同時追加 + 欠損の扱いを変えた感度分析
python -m src.scripts.data.slowreal_eval report     # /tmp/slowreal_report.{json,md}

設計（被覆の偏りへの対処）: ラベルがあるのは直近 3 年に走った馬だけ。2020〜2022 年の走は『その後も走った馬』にしかラベルが無く、ラベルの有無が将来を漏らす。
そこで、ラベルの無い走の特徴量は全て欠損にしたうえで、学習はラベルのある走（取れた馬）だけ、検証はラベルが全員にある期間（2024-01 以降）だけにする。
ラベルのある走は『その後も走り続けた馬』に偏り、勝率が高い（2022 年: ラベルあり 13.5%、なし 3.8%）。学習行の勝つ割合を年ごとに全体（1 / 出走頭数）へそろえる重みを付ける（腕 A）。
重み無し（A0）と、ラベルの無い走も学習に使う腕 B は感度分析。
基準の V4 も同じ学習行で学習し直して比べる（対応のある比較）。test 2025・2026 は使わない。市場オッズ・人気は使わない。
環境変数: SLOWREAL_THREADS（既定 6）, SLOWREAL_LOAD_LIMIT（既定 16）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp35_eval as E  # noqa: E402  (環境変数 TP_BASE_DIR などを設定する)
from src.scripts.data import tp27_eval as t  # noqa: E402

ROOT = E.ROOT
V = E.V
OUT = Path(os.environ.get("SLOWREAL_OUT") or V / "slowreal")
CHAPTER = ROOT / "docs/html/modeling/chapters/slow_start_real_label.html"
KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("SLOWREAL_THREADS", 6))
LOAD_LIMIT = float(os.environ.get("SLOWREAL_LOAD_LIMIT", 16))
E.THREADS, E.LOAD_LIMIT = THREADS, LOAD_LIMIT
LABEL = "slow_rem"
WINDOWS = {"s1": ("2024-01-01", "2024-06-30"), "s2": ("2024-07-01", "2024-12-31")}      # 全員のラベルがあるのは 2024-01 から（coverage.json）
GAP_DAYS = 10
RAW = ["SS_horse", "SS_horse_ctx", "SS_gate", "SS_jockey", "SS_trainer", "SS_sire", "SS_misc", "SS_damage", "SS_rel"]
EXC = ["SX_horse", "SX_horse_ctx", "SX_gate", "SX_jockey", "SX_trainer", "SX_sire", "SX_misc", "SX_rel"]
GROUPS = RAW + EXC + ["SS_pred"]
COMBOS = {"ALL_raw": RAW, "ALL_exc": EXC, "ALL_both": RAW + EXC, "ALL_both_pred": RAW + EXC + ["SS_pred"]}
UNITS = GROUPS + list(COMBOS)
STAGE1_PASS = (-0.00012, 0.0001)        # Δwin がこれより小さく、CI 上限がこれより小さい群を第 2 段へ


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jsave(p: Path, d) -> None:
    E._jsave(p, d)


def _jload(p: Path):
    return E._jload(p)


# ───────────────────────── labels
def cmd_labels(a) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from src.pipeline.features import slowstart_real as SR
    from src.scraper.storage import HybridStorage

    OUT.mkdir(parents=True, exist_ok=True)
    st = HybridStorage()
    keys = st.list_keys("horse_race_remarks")
    _log(f"horse_race_remarks: {len(keys)} 頭")

    def one(h):
        try:
            d = st.load("horse_race_remarks", h) or {}
        except Exception:
            d = {}
        return SR.parse_remarks(h, d.get("race_history", []))

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [x for part in ex.map(one, keys) for x in part]
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "remarks_flags.parquet", index=False)
    _log(f"馬 {df['horse_id'].nunique()} 頭 / {len(df)} 走 / 備考あり {df['has_remarks'].mean():.3f} / 出遅 {df[LABEL].mean():.4f} / ｽﾀｰﾄ指数あり {pd.to_numeric(df['start_index'], errors='coerce').notna().mean():.3f}")


def _frame_and_labels():
    from src.pipeline.features import factory_slowstart as fs
    from src.pipeline.features import slowstart_real as SR
    from src.scripts.data import factory_slow_eval as F

    df, base, meta, folds, lp = t.setup()
    fr = df[KEY].merge(F._frame(), on=KEY, how="left")
    assert len(fr) == len(df) and (fr["race_id"].to_numpy() == df["race_id"].to_numpy()).all()
    fr.index = df.index
    fr = fs.prepare(fr)
    rem = pd.read_parquet(OUT / "remarks_flags.parquet")
    fr[LABEL] = SR.row_labels(fr, rem)
    return df, meta, folds, lp, fr, rem


# ───────────────────────── coverage
def cmd_coverage(a) -> None:
    df, meta, folds, lp, fr, rem = _frame_and_labels()
    fr["known"] = fr[LABEL].notna()
    fr["year"] = fr["date"].dt.year
    fr["ym"] = fr["date"].dt.strftime("%Y-%m")
    fr["win"] = (fr["finish_position"].astype(float) == 1).astype(float)
    fr["grade"] = df["grade"].astype(str).to_numpy() if "grade" in df.columns else "NA"
    rep: dict = {"rows": int(len(fr)), "horses_with_remarks": int(rem["horse_id"].nunique()), "remarks_rows": int(len(rem))}
    rep["known_share_by_year"] = {int(y): round(float(v), 4) for y, v in fr.groupby("year")["known"].mean().items()}
    ym = fr.groupby("ym")["known"].mean()
    rep["known_share_by_month_2023_2024"] = {k: round(float(v), 4) for k, v in ym.items() if "2023-01" <= k <= "2024-12"}
    full = [k for k, v in ym.items() if k >= "2023-06" and v >= 0.995]
    rep["first_month_full_coverage"] = full[0] if full else None
    rep["known_share_by_venue"] = {str(k): round(float(v), 4) for k, v in fr[fr["year"] == 2024].groupby("venue_code")["known"].mean().items()}
    rep["known_share_by_grade_2024"] = {str(k): round(float(v), 4) for k, v in fr[fr["year"] == 2024].groupby("grade")["known"].mean().items()}
    k = fr[fr["known"]]
    rep["rate_by_year"] = {int(y): round(float(v), 4) for y, v in k.groupby("year")[LABEL].mean().items()}
    rep["rate_by_venue"] = {str(c): round(float(v), 4) for c, v in k.groupby("venue_code")[LABEL].mean().items()}
    rep["rate_by_surface"] = {str(c): round(float(v), 4) for c, v in k.groupby("surface")[LABEL].mean().items()}
    rep["rate_by_grade"] = {str(c): round(float(v), 4) for c, v in k.groupby("grade")[LABEL].mean().items()}
    rep["rate_by_gate3"] = {str(c): round(float(v), 4) for c, v in k.groupby("gate3")[LABEL].mean().items()}
    # 被覆の偏り: ラベルがある走とない走で、勝率・馬の通算走数がどれだけ違うか（年別）。ラベルの有無が将来情報になっている証拠
    runs = fr.sort_values(["horse_id", "date"]).groupby("horse_id").cumcount()
    fr["career_runs"] = runs.reindex(fr.index)
    rep["win_rate_known_vs_unknown_by_year"] = {int(y): {"known": round(float(g[g["known"]]["win"].mean()), 4), "unknown": round(float(g[~g["known"]]["win"].mean()), 4) if (~g["known"]).any() else None,
                                                         "n_unknown": int((~g["known"]).sum())} for y, g in fr.groupby("year")}
    rep["career_runs_known_vs_unknown_2022"] = {"known": round(float(fr[(fr["year"] == 2022) & fr["known"]]["career_runs"].mean()), 2),
                                                "unknown": round(float(fr[(fr["year"] == 2022) & ~fr["known"]]["career_runs"].mean()), 2)}
    # 戦績に載っているはずの走（ラベルのある馬の走）で、その走が戦績に無かった割合
    h = set(rem["horse_id"])
    has_h = fr["horse_id"].isin(h)
    rep["covered_horse_runs_missing_in_history"] = round(float((has_h & ~fr["known"]).sum() / max(has_h.sum(), 1)), 5)
    rep["unreadable_label_note"] = "備考が空の走は『出遅れていない』（0）。『出遅』は目立った出遅れだけに付く（代理ラベル 9% に対し 2〜3%）"
    # 出遅れの影響（ラベル確認）: 出遅れた走の着順（相対）が悪いか
    k = k[k["finish_position"].astype(float) > 0].copy()
    k["fin_pct"] = (k["finish_position"].astype(float) - 1) / (k["field_size"].astype(float) - 1).clip(lower=1)
    hm = k.groupby("horse_id")["fin_pct"].transform("mean")
    rep["finish_pct_slow_minus_not"] = {"raw": round(float(k[k[LABEL] > 0]["fin_pct"].mean() - k[k[LABEL] == 0]["fin_pct"].mean()), 4),
                                        "within_horse": round(float((k[k[LABEL] > 0]["fin_pct"] - hm[k[LABEL] > 0]).mean()), 4), "n_slow": int((k[LABEL] > 0).sum())}
    kk = k.sort_values(["horse_id", "date"])
    kk["i"] = kk.groupby("horse_id").cumcount()
    kk = kk[kk.groupby("horse_id")["i"].transform("size") >= 12]
    a_ = kk[kk["i"] % 2 == 0].groupby("horse_id")[LABEL].mean()
    b_ = kk[kk["i"] % 2 == 1].groupby("horse_id")[LABEL].mean()
    j = pd.concat([a_, b_], axis=1, keys=["a", "b"]).dropna()
    rep["split_half_corr"] = {"horses": int(len(j)), "corr": round(float(j["a"].corr(j["b"])), 3), "spearman": round(float(j["a"].corr(j["b"], method="spearman")), 3)}
    _jsave(OUT / "coverage.json", rep)
    _log(json.dumps({k_: rep[k_] for k_ in ("rows", "horses_with_remarks", "known_share_by_year", "first_month_full_coverage", "rate_by_year", "covered_horse_runs_missing_in_history", "split_half_corr", "finish_pct_slow_minus_not")}, ensure_ascii=False))


# ───────────────────────── build
def cmd_build(a) -> None:
    import lightgbm as lgb

    from src.pipeline.features import slowstart_real as SR
    from src.scripts.data import factory_slow_eval as F

    df, meta, folds, lp, fr, rem = _frame_and_labels()
    gdir = OUT / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    events = SR.event_table(fr, rem)
    _log(f"走 {len(fr):,}（ラベルあり {fr[LABEL].notna().mean():.3f}）/ ラベル履歴 {len(events):,}")
    t0 = time.time()
    groups = SR.build(fr, events, F._parents(), p0=SR.P0)
    # 出遅れ確率 p_slow の OOF: 年ごとの expanding。学習はラベルのある走だけ。特徴量は上の群 + 文脈。
    parts = [groups[g].reset_index(drop=True) for g in SR.GROUP_ORDER if g in groups]
    num = fr[["horse_number", "field_size", "distance", "bracket_number", "weight", "age", "dband", "lband", "fband"]].reset_index(drop=True)
    X = pd.concat(parts + [num, fr[["surface", "venue_code", "sex", "gate3"]].astype("category").reset_index(drop=True)], axis=1)
    y = fr[LABEL].to_numpy(dtype=float)
    year = fr["date"].dt.year.to_numpy()
    date = fr["date"].to_numpy()
    p = np.full(len(fr), np.nan)
    for Y in range(2021, 2025):
        tr = np.flatnonzero((date < np.datetime64(f"{Y}-01-01") - np.timedelta64(10, "D")) & np.isfinite(y))
        te = np.flatnonzero(year == Y)
        m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                               reg_lambda=5.0, n_jobs=THREADS, verbose=-1, random_state=0)
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
        _log(f"p_slow {Y}: 学習 {len(tr):,} 走、平均 {np.nanmean(p[te]):.4f}")
    unknown = np.isnan(y)
    p[unknown] = np.nan
    race = fr["race_id"].to_numpy()
    s = pd.Series(p, index=fr.index)
    pred = pd.DataFrame({"ss_p": p}, index=fr.index)
    pred["ss_p_rank"] = s.groupby(race).rank(ascending=False, method="average").to_numpy()
    pred["ss_p_racesum"] = s.groupby(race).transform("sum").to_numpy()
    pred["ss_p_gap"] = (s - s.groupby(race).transform("mean")).to_numpy()
    dmg = groups["SS_damage"]["damage"].to_numpy(dtype=float)
    pred["ss_p_x_damage"] = p * dmg
    groups["SS_pred"] = pred
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(p) & np.isfinite(y)
    auc = {int(Y): round(float(roc_auc_score(y[ok & (year == Y)], p[ok & (year == Y)])), 4) for Y in range(2021, 2025) if (ok & (year == Y)).sum() > 100 and len(set(y[ok & (year == Y)])) > 1}
    _log(f"p_slow の OOF AUC: {auc}")
    meta_out = {"rows": int(len(fr)), "label_known_share": float(fr[LABEL].notna().mean()), "p_slow_auc": auc, "groups": {}}
    for name, g in groups.items():
        g = g.copy()
        nn = g.notna().mean()
        keep = [c for c in g.columns if nn[c] > 0.02 and g[c].nunique(dropna=True) > 1]
        out = pd.concat([fr[KEY].reset_index(drop=True), g[keep].reset_index(drop=True)], axis=1)
        out.to_parquet(gdir / f"{name}.parquet", index=False)
        meta_out["groups"][name] = {"cols": len(keep), "non_null_share": float(nn[keep].mean()) if keep else 0.0}
        _log(f"{name}: {len(keep)} 列 非欠損 {meta_out['groups'][name]['non_null_share']:.2f}")
    fr[KEY + [LABEL]].to_parquet(OUT / "row_labels.parquet", index=False)
    _jsave(OUT / "build_meta.json", meta_out)
    _log(f"完了 ({time.time() - t0:.0f}s)")


def cmd_leakcheck(a) -> None:
    """実データでの先読み検査: cut 日以降のラベルを全て欠損にして特徴量を作り直し、cut 日より前の走の特徴量が全群で 1 ビットも変わらないことを確かめる。
    cut 日の走（当日）の特徴量も当日のラベルに依存しない。月の途中（2024-03-15）と月初（2024-04-01）の 2 通り。"""
    from src.pipeline.features import slowstart_real as SR
    from src.scripts.data import factory_slow_eval as F

    df, meta, folds, lp, fr, rem = _frame_and_labels()
    parents = F._parents()
    full = SR.build(fr, SR.event_table(fr, rem), parents, p0=SR.P0)
    rep = {}
    for cut in ("2024-03-15", "2024-04-01"):
        c = pd.Timestamp(cut)
        fr2 = fr.copy()
        fr2.loc[fr2["date"] >= c, SR.LABEL] = np.nan
        rem2 = rem.copy()
        rd = pd.to_datetime(rem2["date"].astype(str).str.replace("/", "-"), errors="coerce")
        rem2 = rem2[~(rd >= c)]
        g2 = SR.build(fr2, SR.event_table(fr2, rem2), parents, p0=SR.P0)
        past = (fr["date"] < c).to_numpy() & fr[SR.LABEL].notna().to_numpy()
        bad = {k: int((~np.isclose(full[k][past].to_numpy(dtype=float), g2[k][past].to_numpy(dtype=float), rtol=0, atol=0, equal_nan=True)).sum()) for k in full}
        rep[cut] = {"rows_compared": int(past.sum()), "cells_different": bad, "exact": all(v == 0 for v in bad.values())}
        _log(f"{cut}: 比較 {int(past.sum()):,} 走、違う値 {sum(bad.values())} セル、完全一致={rep[cut]['exact']}")
    _jsave(OUT / "leakcheck.json", rep)


def cmd_diagnose(a) -> None:
    """p_slow が何を表すか（軽い集計。学習はしない）: 十分位別の実際の出遅れ率と、V4 が見落としている勝ち残差、V4 の列との相関。valid 2024-01〜06 の第 1 段の基準 OOF を使う。"""
    from scipy.stats import spearmanr

    df, base, meta, _f, lp = t.setup()
    p = _load_unit("SS_pred", df)
    X0 = E._frame(df)
    rl = pd.read_parquet(OUT / "row_labels.parquet")
    rl["race_id"], rl["horse_id"] = rl["race_id"].astype(str), rl["horse_id"].astype(str)
    y = df[KEY].merge(rl, on=KEY, how="left")[LABEL].to_numpy()
    d = df[KEY + ["date", "finish_position"]].copy()
    d["p"], d["y"] = p["ss_p"].to_numpy(), y
    d["date"] = pd.to_datetime(d["date"])
    oof = pd.read_parquet(OUT / "stage1/base.parquet")
    v = d.merge(oof[KEY + ["p_win", "p_top3"]], on=KEY, how="inner").dropna(subset=["p", "y"])
    v["dec"] = pd.qcut(v["p"], 10, labels=False, duplicates="drop")
    v["win"] = (v["finish_position"].astype(float) == 1).astype(float)
    v["top3"] = (v["finish_position"].astype(float) <= 3).astype(float)
    g = v.groupby("dec").agg(n=("p", "size"), p_mean=("p", "mean"), slow_rate=("y", "mean"), win=("win", "mean"), win_v4=("p_win", "mean"), top3=("top3", "mean"), top3_v4=("p_top3", "mean")).reset_index()
    g["win_resid"], g["top3_resid"] = g["win"] - g["win_v4"], g["top3"] - g["top3_v4"]
    rep = {"window": WINDOWS["s1"], "deciles": g.round(4).to_dict("records")}
    idx = np.flatnonzero(np.isfinite(p["ss_p"].to_numpy()))
    num = X0.iloc[idx].select_dtypes(include=[np.number])
    pv = p["ss_p"].to_numpy()[idx]
    sub = np.random.default_rng(0).choice(len(idx), size=min(60000, len(idx)), replace=False)
    cors = {}
    for c in num.columns:
        x = num[c].to_numpy(dtype=float)[sub]
        ok = np.isfinite(x) & np.isfinite(pv[sub])
        if ok.sum() > 1000 and np.nanstd(x[ok]) > 0:
            cors[c] = float(spearmanr(x[ok], pv[sub][ok])[0])
    top = sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:20]
    rep["top_corr_with_v4_columns"] = [{"col": k, "spearman": round(v_, 3)} for k, v_ in top]
    _jsave(OUT / "diagnose.json", rep)
    for r in rep["deciles"]:
        _log(f"十分位 {r['dec']}: n {r['n']} 予測 {r['p_mean']:.3f} 実際 {r['slow_rate']:.3f} 勝率 {r['win']:.4f} (V4 {r['win_v4']:.4f}) 複勝率 {r['top3']:.4f} (V4 {r['top3_v4']:.4f})")
    _log("V4 の列との相関 上位: " + ", ".join(f"{r['col']} {r['spearman']:+.2f}" for r in rep["top_corr_with_v4_columns"][:10]))


# ───────────────────────── stages
def _load_unit(name: str, df: pd.DataFrame) -> pd.DataFrame:
    names = COMBOS.get(name, [name])
    parts = []
    for n in names:
        T = pd.read_parquet(OUT / "groups" / f"{n}.parquet")
        T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
        m = df[KEY].merge(T, on=KEY, how="left")
        assert len(m) == len(df)
        m = m.drop(columns=KEY)
        m.index = df.index
        parts.append(m)
    return pd.concat(parts, axis=1)


def _folds(df: pd.DataFrame, known: np.ndarray | None, window: str):
    from src.pipeline.models.time_split import Fold

    vs, ve = (pd.Timestamp(x) for x in WINDOWS[window])
    dates = pd.to_datetime(df["date"])
    tr_mask = (dates < vs - pd.Timedelta(days=GAP_DAYS)) & (dates >= pd.Timestamp("2020-01-01"))
    if known is not None:
        tr_mask &= known
    va_mask = (dates >= vs) & (dates <= ve)
    return [Fold(fold_id=window, valid_year=str(vs.year), train_index=df.index[tr_mask].to_numpy(), valid_index=df.index[va_mask].to_numpy(),
                 train_end=vs - pd.Timedelta(days=GAP_DAYS), valid_start=vs)]


def _prior_weights(df: pd.DataFrame, known: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """学習行の重み: 年ごとに『勝つ割合』を全体（レース数 / 行数）へそろえる（ラベルのある走は勝つ馬に偏っている）。"""
    year = pd.to_datetime(df["date"]).dt.year.to_numpy()
    win = (df["finish_position"].astype(float).to_numpy() == 1)
    w = np.ones(len(df))
    for y in np.unique(year):
        yi = year == y
        pop = float(win[yi].sum() / yi.sum())
        ki = yi & known
        if ki.sum() < 100:
            continue
        kr = float(win[ki].mean())
        w[ki & win] = pop / max(kr, 1e-9)
        w[ki & ~win] = (1 - pop) / max(1 - kr, 1e-9)
    return w[idx]


def _run(name, X, meta, folds, lp, seeds, weight=None):
    from src.pipeline.models import zoo_runner

    E._wait()
    t0 = time.time()
    params = dict(lp)
    if weight is not None:
        params["weight"] = weight
    res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, folds, seeds, THREADS, params=params)
    lv = {m: float(res[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")}
    _log(f"{name}: win {lv['logloss_win']:.5f} top2 {lv['logloss_top2']:.5f} top3 {lv['logloss_top3']:.5f} cols={X.shape[1]} 学習 {len(folds[0].train_index):,} 検証 {len(folds[0].valid_index):,} ({time.time() - t0:.0f}s)")
    return lv, oof


def _stage(label: str, window: str, seeds: list[int], units: list[str], arm: str = "A") -> None:
    df, base, meta, _folds_unused, lp = t.setup()
    rl = pd.read_parquet(OUT / "row_labels.parquet")
    rl["race_id"], rl["horse_id"] = rl["race_id"].astype(str), rl["horse_id"].astype(str)
    known = df[KEY].merge(rl, on=KEY, how="left")[LABEL].notna().to_numpy()
    folds = _folds(df, known if arm in ("A", "A0") else None, window)
    weight = _prior_weights(df, known, folds[0].train_index) if arm == "A" else None
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = E._frame(df)
    d = OUT / label
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = _jload(resp)
    res.setdefault("setup", {"window": WINDOWS[window], "arm": arm, "seeds": seeds, "train_rows": int(len(folds[0].train_index)), "valid_rows": int(len(folds[0].valid_index)),
                             "valid_known_share": float(known[folds[0].valid_index].mean()),
                             "train_win_rate": float((df["finish_position"].astype(float).to_numpy()[folds[0].train_index] == 1).mean()),
                             "train_win_rate_weighted": float(np.average((df["finish_position"].astype(float).to_numpy()[folds[0].train_index] == 1), weights=weight)) if weight is not None else None})
    bp = d / "base.parquet"
    if bp.exists() and "base" in res:
        oof0 = pd.read_parquet(bp)
    else:
        lv, oof0 = _run("V4", X0, meta, folds, lp, seeds, weight)
        oof0.to_parquet(bp, index=False)
        res["base"] = {"levels": lv, "cols": int(X0.shape[1])}
        _jsave(resp, res)
    steps = res.setdefault("groups", {})
    for u in units:
        if u in steps:
            continue
        if u == "PASSED_SINGLES":
            names = [x for x in units if x in GROUPS]
            G = pd.concat([_load_unit(x, df) for x in names], axis=1)
        elif u == "SS_all_shuffled":                      # 対照: 全群を年内でランダムに入れ替えた列（効果が 0 近くになるはず）
            G = _load_unit("ALL_both_pred", df)
            rng = np.random.default_rng(0)
            yr = pd.to_datetime(df["date"]).dt.year.to_numpy()
            perm = np.arange(len(df))
            for y in np.unique(yr):
                ix = np.flatnonzero(yr == y)
                perm[ix] = rng.permutation(ix)
            G = G.iloc[perm].set_axis(G.index)
        else:
            G = _load_unit(u, df)
        dup = set(G.columns) & set(X0.columns)
        assert not dup, f"列名が V4 と重複: {sorted(dup)[:5]}"
        X = pd.concat([X0, G], axis=1)
        lv, oof = _run(f"V4+{u}", X, meta, folds, lp, seeds, weight)
        c = t._cmp(oof0, oof, fin_map, u)
        steps[u] = {"cols": int(G.shape[1]), "levels": lv, "vs_V4": c}
        _jsave(resp, res)
        _log(f"  {u}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")


def cmd_stage1(a) -> None:
    _stage("stage1", "s1", [0], UNITS + ["SS_all_shuffled"])


def _passed() -> list[str]:
    r = _jload(OUT / "stage1/results.json")
    return [g for g, s in r.get("groups", {}).items() if g in UNITS and s["vs_V4"]["Δwin"] < STAGE1_PASS[0] and s["vs_V4"]["win_hi"] < STAGE1_PASS[1]]


def cmd_stage2(a) -> None:
    passed = _passed()
    _log(f"第 1 段を通過: {passed}")
    _jsave(OUT / "stage1/passed.json", passed)
    singles = [g for g in passed if g in GROUPS]
    units = passed + (["PASSED_SINGLES"] if len(singles) > 1 else [])
    if units:
        _stage("stage2", "s2", [2, 3], units)
    # 感度分析: 学習の重みを無しにする（A0）、ラベルの無い走も学習に使う（B。特徴量は欠損のまま）。第 1 段で最も効いた p_slow の群で。
    _stage("stage2_armA0", "s2", [2, 3], ["SS_pred"], arm="A0")
    _stage("stage2_armB", "s2", [2, 3], ["SS_pred"], arm="B")


def cmd_report(a) -> None:
    rep = {"coverage": _jload(OUT / "coverage.json"), "build": _jload(OUT / "build_meta.json"), "stage1": _jload(OUT / "stage1/results.json"),
           "stage2": _jload(OUT / "stage2/results.json"), "stage2_armA0": _jload(OUT / "stage2_armA0/results.json"), "stage2_armB": _jload(OUT / "stage2_armB/results.json")}
    L = ["# 出遅れ率（実ラベル）の V4 に対する増分\n"]
    for k in ("stage1", "stage2", "stage2_armA0", "stage2_armB"):
        r = rep[k]
        if not r:
            continue
        L += [f"## {k}（valid {r['setup']['window']}、学習 {r['setup']['train_rows']:,} 走、検証 {r['setup']['valid_rows']:,} 走、seed {r['setup']['seeds']}）",
              f"V4 基準: win {r['base']['levels']['logloss_win']:.5f}\n", "| 群 | 列 | Δwin | CI 上限 | Δtop2 | Δtop3 | 採否 |", "|---|---|---|---|---|---|---|"]
        for g, s in r["groups"].items():
            c = s["vs_V4"]
            L.append(f"| {g} | {s['cols']} | {c['Δwin']:+.5f} | {c['win_hi']:+.5f} | {c['Δtop2']:+.5f} | {c['Δtop3']:+.5f} | {'採用' if c['採用'] else '不採用'} |")
        L.append("")
    Path("/tmp/slowreal_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    Path("/tmp/slowreal_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def _f(x, nd=5):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


def _unit_note(u: str) -> str:
    return {"SS_horse": "馬自身の出遅れ率（全期間・直近 5/10 走・指数平滑・前走・走数）", "SS_horse_ctx": "馬 × 面/距離帯/場/馬番位置の出遅れ率", "SS_gate": "場・面・距離帯・馬番位置、枠番、頭数 × 馬番位置のセル別の率",
            "SS_jockey": "騎手（全体・面別・馬番位置別）", "SS_trainer": "調教師（全体・面別）", "SS_sire": "父・母父（ID）、父 × 面", "SS_misc": "性齢、間隔、頭数、斤量、コースのセル",
            "SS_damage": "出遅れた走で着順（相対）がどれだけ悪くなるか（馬ごと）", "SS_rel": "同じレースの出走馬の中での差・順位", "SS_pred": "出遅れ確率の OOF 予測 p_slow と、レース内の順位・合計・差、p_slow × 損害",
            "ALL_raw": "SS の 9 群すべて", "ALL_exc": "SX の 8 群すべて", "ALL_both": "SS + SX", "ALL_both_pred": "SS + SX + p_slow", "SS_all_shuffled": "対照: ALL_both_pred を年内でランダムに入れ替えた列",
            "PASSED_SINGLES": "第 1 段を通過した単独群の同時追加"}.get(u, "SX_* は SS_* と同じ群を『時代の出遅れ率を引いた超過』で作ったもの" if u.startswith("SX_") else "")


def cmd_chapter(a) -> None:
    import html as H

    cov, bm, lk, dg = (_jload(OUT / n) for n in ("coverage.json", "build_meta.json", "leakcheck.json", "diagnose.json"))
    r1, r2, rA0, rB = (_jload(OUT / n / "results.json") for n in ("stage1", "stage2", "stage2_armA0", "stage2_armB"))
    s1, s2 = r1.get("groups", {}), r2.get("groups", {})

    def row(u):
        a_, b_ = s1.get(u), s2.get(u)
        c1, c2 = (a_ or {}).get("vs_V4", {}), (b_ or {}).get("vs_V4", {})
        adopted = ("採用" if c2.get("採用") else "不採用") if b_ else "第 1 段で不通過"
        return (f"<tr><td>{u}</td><td>{H.escape(_unit_note(u))}</td><td class=num>{(a_ or b_)['cols']}</td><td class=num>{_f(c1.get('Δwin'))}（{_f(c1.get('win_hi'))}）</td>"
                f"<td class=num>{_f(c2.get('Δwin')) if b_ else '-'}（{_f(c2.get('win_hi')) if b_ else '-'}）</td><td class=num>{_f(c2.get('Δtop2')) if b_ else _f(c1.get('Δtop2'))}</td>"
                f"<td class=num>{_f(c2.get('Δtop3')) if b_ else _f(c1.get('Δtop3'))}</td><td>{adopted}</td></tr>")

    units = [u for u in UNITS + ["SS_all_shuffled", "PASSED_SINGLES"] if u in s1 or u in s2]
    st_rows = "".join(row(u) for u in units)
    ky = cov["known_share_by_year"]
    wr = cov["win_rate_known_vs_unknown_by_year"]
    yr_rows = "".join(f"<tr><td>{y}</td><td class=num>{ky[y]:.3f}</td><td class=num>{wr[y]['known']:.3f}</td><td class=num>{'-' if wr[y]['unknown'] is None else format(wr[y]['unknown'], '.3f')}</td><td class=num>{wr[y]['n_unknown']:,}</td><td class=num>{cov['rate_by_year'].get(y, float('nan')):.3f}</td></tr>" for y in sorted(ky))
    dec_rows = "".join(f"<tr><td>{r['dec'] + 1}</td><td class=num>{r['p_mean']:.3f}</td><td class=num>{r['slow_rate']:.3f}</td><td class=num>{r['win']:.4f} / {r['win_v4']:.4f}</td><td class=num>{r['top3']:.4f} / {r['top3_v4']:.4f}</td></tr>" for r in dg.get("deciles", []))
    cor_rows = "".join(f"<tr><td>{r['col']}</td><td class=num>{r['spearman']:+.2f}</td></tr>" for r in dg.get("top_corr_with_v4_columns", [])[:8])

    def arm_row(name, r):
        if not r.get("groups"):
            return ""
        u, s = next(iter(r["groups"].items()))
        c = s["vs_V4"]
        return (f"<tr><td>{name}</td><td class=num>{r['setup']['train_rows']:,}</td><td class=num>{r['base']['levels']['logloss_win']:.5f}</td><td>{u}</td><td class=num>{_f(c['Δwin'])}（{_f(c['win_hi'])}）</td>"
                f"<td class=num>{_f(c['Δtop2'])}</td><td class=num>{_f(c['Δtop3'])}（{_f(c.get('top3_hi'))}）</td></tr>")

    arm_rows = arm_row("A（重み付き・ラベルのある走だけで学習）", {**r2, "groups": {"SS_pred": s2["SS_pred"]}} if "SS_pred" in s2 else {}) + arm_row("A0（重み無し）", rA0) + arm_row("B（ラベルの無い走も学習に使う。特徴量は欠損）", rB)
    lk_ok = all(v.get("exact") for v in lk.values()) if lk else False
    best1 = min((u for u in s1 if u in UNITS), key=lambda u: s1[u]["vs_V4"]["Δwin"]) if s1 else "-"
    adopted = [u for u, v in s2.items() if v.get("vs_V4", {}).get("採用")]
    summary = (f"第 2 段（valid 2024-07〜12・新 seed）で採用基準を満たした群はありませんでした（試した {len([u for u in s1 if u != 'SS_all_shuffled'])} 通り）。第 1 段（2024-01〜06）で最も良かった p_slow（出遅れ確率の予測、5 列）は"
               f"勝率 {_f(s1.get('SS_pred', {}).get('vs_V4', {}).get('Δwin'))}（CI 上限 {_f(s1.get('SS_pred', {}).get('vs_V4', {}).get('win_hi'))}）と効いたが、第 2 段では "
               f"{_f(s2.get('SS_pred', {}).get('vs_V4', {}).get('Δwin'))}（上限 {_f(s2.get('SS_pred', {}).get('vs_V4', {}).get('win_hi'))}）に縮み、再現しなかった。V4 には何も足しません。" if not adopted else f"採用: {adopted}")
    head = open(ROOT / "docs/html/modeling/chapters/transport_and_stay.html", encoding="utf-8").read().split("<body>")[0]
    head = head.replace("輸送と滞在（tp-33）", "出遅れ率（実ラベル版）")
    body = f"""<body><a href="../../index.html" class="muted">← ドキュメント一覧</a><h1>出遅れ率（実ラベル版）</h1>
<p class="muted">netkeiba の戦績ページの備考にある『出遅』を実ラベルにして、馬・騎手・調教師・枠・血統の出遅れ率と、出遅れ確率の予測（p_slow）を作り、V4 全体に足して効くかを測った。代理ラベル版（SmartRC のテン 1F から作ったもの、勝率 −0.00003）の置き換え。</p>
<div class="note"><b>要点.</b> {summary}
 ラベルの付く走は『直近 3 年に走った馬』だけで、2020〜2023 年はその後も走った馬に偏る（ラベルのある走の勝率はない走の 2.4〜5.7 倍）。ラベルの有無そのものが将来の情報になるため、学習はラベルのある走だけ、検証は全員にラベルのある 2024 年に限った。</div>
<h2>1. ラベルの中身と被覆</h2>
<ul><li><b>ラベル</b>: 備考に『出遅』を含む走を 1、含まない走を 0（備考が空の走は 0）。ラベル全体 {cov['remarks_rows']:,} 走・{cov['horses_with_remarks']:,} 頭。備考は目立った出遅れだけに付く（スタート指数の下位とは別物）。ラベルのある走での率は 17.7〜23.3%（年別、下表右端）。</li>
<li><b>馬ごとの個体差が強い</b>: 走数 12 以上の {cov['split_half_corr']['horses']:,} 頭で、偶数番目と奇数番目の走の出遅れ率の相関は {cov['split_half_corr']['corr']:.2f}（Spearman {cov['split_half_corr']['spearman']:.2f}）。代理ラベルでは 0.20 だった。</li>
<li><b>出遅れた走の着順</b>: レース内の相対順位が +{cov['finish_pct_slow_minus_not']['raw']:.3f}（同じ馬の他の走との差では +{cov['finish_pct_slow_minus_not']['within_horse']:.3f}）悪い。出遅れた走 {cov['finish_pct_slow_minus_not']['n_slow']:,} 走。</li></ul>
<h3>被覆の偏り（ラベルの有無が将来の情報になっている）</h3>
<table><tr><th>年</th><th>ラベルのある走の割合</th><th>勝率（ラベルあり）</th><th>勝率（ラベルなし）</th><th>ラベルなしの走数</th><th>出遅れ率（ラベルあり）</th></tr>{yr_rows}</table>
<p class="muted">2024 年以降は全員にラベルがある（全馬が『直近 3 年に走った馬』）。2023 年以前のラベルのない走は、その後走らなかった（引退・地方転出など）馬の走で、勝率が低い。勝率の比は 2020 年 2.4 倍、2021 年 2.5 倍、2022 年 3.6 倍、2023 年 5.7 倍。全員にラベルが揃う最初の月は {cov['first_month_full_coverage']}。</p>
<h2>2. 評価の設計</h2>
<ul><li><b>検証</b>: ラベルが全員にある 2024 年だけ。第 1 段 = 2024-01〜06（seed 1 本）、第 2 段 = 2024-07〜12（新 seed 2 本）。学習は検証の 10 日前まで。基準は V4（tp31 の 1,062 列）を同じ学習行で学び直したもの（対応のある比較）。test 2025・2026 は使わない。</li>
<li><b>学習行</b>: 腕 A（主）= ラベルのある走だけ。ラベルのある走は勝つ馬に偏る（学習行の勝率 {r1['setup']['train_win_rate']:.3f} → 全体の {r1['setup']['train_win_rate_weighted']:.3f}）ので、年ごとに勝つ割合を全体にそろえる重みを付ける。腕 A0 = 重み無し、腕 B = ラベルのない走も学習に使う（特徴量は欠損）。</li>
<li><b>ラベルのない走の特徴量は全て欠損</b>、馬以外の集団の件数列は落とした（取れた馬が時代で違うため、件数の水準が年を当てる手がかりになる）。</li>
<li><b>時代差</b>: 率は年とともに少し動く。生の率 SS_* と、各走のラベルから『その月より前 3 か月の全体の率』を引いた超過 SX_* の両方を作った。</li>
<li><b>採用基準</b>: 勝率 LogLoss の差の 95% CI（レース単位の paired bootstrap）上限 &lt; 0、効果 0.0002 以上、連対・複勝の悪化 0.0002 以内。第 1 段は Δ &lt; −0.00012 かつ CI 上限 &lt; +0.0001 で通過。</li></ul>
<h2>3. 特徴量</h2>
<p>キーは horse_id・jockey_id・trainer_id・種牡馬 ID。すべて『その走の日付より厳密に前』のラベルだけから作る（同日の他のレースも使わない）。群と列数は 4 節の表。全体の率は縮約の出発点（as-of）で、馬ごとの率は 5 件分、集団は 30〜200 件分を全体へ縮約。p_slow は年ごとの expanding（前年末までのラベルのある走で学習した LightGBM）の OOF で、検証の AUC は {', '.join(f'{y} 年 {v:.2f}' for y, v in bm.get('p_slow_auc', {}).items())}。</p>
<h2>4. 先読み検査</h2>
<ul><li>単体テスト（tests/pipeline/test_slowstart_real.py）: cut 日以降のラベルを全部書き換えても、cut 日以前の全群の特徴量が 1 ビットも変わらない（月初・月の途中の 3 通り。月の全体の率が当月の将来のラベルを使わないことも含む）、同日の他レースのラベルが漏れない、ラベル不明の走は全て欠損。</li>
<li>実データ（2024-03-15 と 2024-04-01 で切って作り直し、それ以前の {min((v['rows_compared'] for v in lk.values()), default=0):,}〜{max((v['rows_compared'] for v in lk.values()), default=0):,} 走を全群で比較）: 違う値のセル 0、完全一致 = {'はい' if lk_ok else 'いいえ'}。</li></ul>
<h2>5. 結果（V4 全体に対する勝率 LogLoss の差。かっこ内は 95% CI 上限）</h2>
<p class="muted">第 1 段の基準 {r1['base']['levels']['logloss_win']:.5f}（学習 {r1['setup']['train_rows']:,} 走・検証 {r1['setup']['valid_rows']:,} 走）、第 2 段の基準 {r2['base']['levels']['logloss_win']:.5f}（学習 {r2['setup']['train_rows']:,} 走・検証 {r2['setup']['valid_rows']:,} 走）。試した {len([u for u in s1 if u != 'SS_all_shuffled'])} 通りと対照 1 通り。</p>
<table><tr><th>群</th><th>内容</th><th>列数</th><th>第 1 段</th><th>第 2 段</th><th>Δtop2</th><th>Δtop3</th><th>採否</th></tr>{st_rows}</table>
<p class="muted">対照（ランダムに入れ替えた列）の勝率は {_f(s1.get('SS_all_shuffled', {}).get('vs_V4', {}).get('Δwin'))}、つまり雑音の水準は 0 付近。第 1 段で 22 通りのうち最良を選ぶと、その分だけ良く見える（選択の偏り）。</p>
<h3>感度分析（第 2 段の窓、p_slow の群）</h3>
<table><tr><th>腕</th><th>学習行</th><th>V4 基準</th><th>群</th><th>Δwin（上限）</th><th>Δtop2</th><th>Δtop3（上限）</th></tr>{arm_rows}</table>
<h2>6. p_slow は何を表すか</h2>
<p>2024-01〜06 の検証行を p_slow の十分位に分けた（下表）。実際の出遅れ率は 2.2% から 55.9% まで開く。V4 は出遅れやすい馬の勝率・複勝率を実際より高く、出遅れにくい馬を低く見積もっている（右 2 列は 実際 / V4 の予測）。この残差を学習し直した効果は、第 1 段の窓では勝率 −0.00073 だったが、第 2 段の窓では CI が 0 をまたいだ（5 節）。</p>
<table><tr><th>p_slow の十分位</th><th>p_slow 平均</th><th>実際の出遅れ率</th><th>勝率 実 / V4</th><th>複勝率 実 / V4</th></tr>{dec_rows}</table>
<p class="muted">p_slow と V4 の列の相関（Spearman、上位）: 出走経験の少なさが最大。出遅れやすさは『経験が浅い』ことと強く結びつき、V4 はその大半を既に持つ。</p>
<table><tr><th>V4 の列</th><th>相関</th></tr>{cor_rows}</table>
<h2>7. 解釈と次の一手</h2>
<ul><li>出遅れは馬ごとの癖が強く（偶奇相関 0.60）、着順にも効く（着順 +0.083）。それでも V4 に足した増分は、第 2 段の窓で再現しなかった。V4 は出走経験・位置取り・前段（S3）などを通じて、出遅れやすさの大半を既に持っている。</li>
<li>採用なし → v5 の再学習の材料にはならない。ただし top3 の悪化は無く、第 2 段の p_slow は Δtop3 {_f(s2.get('SS_pred', {}).get('vs_V4', {}).get('Δtop3'))}（上限 {_f(s2.get('SS_pred', {}).get('vs_V4', {}).get('top3_hi'))}）。複勝側の補助として弱い手がかりはある。</li>
<li>今回の窓は小さく（検証 22,000〜25,000 走）、全員にラベルが揃う期間は 2024 年以降のみ。ラベルが 2025・2026 年も溜まる来年以降は、学習行をすべてラベルありにできるため、重みなしの評価が可能になる。<b>再評価の条件</b>: ラベルが全員に揃う年が増え、重みなしで学習・検証をやり直せるとき。</li>
<li>運用に載せる場合の設計メモ（採用時のみ）: 出走馬の戦績ページ（horse_race_remarks）を各開催日の結果確定後に取る必要があり（1 開催日約 500 頭、1 頭 1 リクエスト）、騎手・調教師・父・枠のセル別の件数と出遅れ数、馬ごとの履歴を状態として持つ。p_slow モデルは年ごとに学び直す。</li></ul>
</body></html>"""
    CHAPTER.write_text(head + body, encoding="utf-8")
    _log(f"章を書いた: {CHAPTER}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["labels", "coverage", "build", "leakcheck", "diagnose", "stage1", "stage2", "report", "chapter"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"labels": cmd_labels, "coverage": cmd_coverage, "build": cmd_build, "leakcheck": cmd_leakcheck, "diagnose": cmd_diagnose, "stage1": cmd_stage1, "stage2": cmd_stage2, "report": cmd_report, "chapter": cmd_chapter}[a.cmd](a)


if __name__ == "__main__":
    main()

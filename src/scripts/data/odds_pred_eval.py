"""想定オッズ予測モデル（馬券最適化の前段）。フェーズ1=単勝・複勝、フェーズ2=馬連・ワイド・馬単・3連複。

    python -m src.scripts.data.odds_pred_eval phase1      # 単複: 年次 walk-forward（各年より前の年だけで学習）の OOF と評価
    python -m src.scripts.data.odds_pred_eval phase2      # 複数頭券種: 想定オッズの構成と評価
    python -m src.scripts.data.odds_pred_eval report      # 決定ログ用の要約 JSON

方針:
- レース前に分かる情報だけ（市場オッズ・人気は特徴量にしない。pred_odds.assert_no_odds_features で検査）。T-X 時点オッズは使わない。
- 学習は検証年より前の年だけ（2022 は 2020〜21 …）。2026 年は選択・チューニングに使わず、最後に 1 回だけ報告する。
- 目的変数はオッズ表 betting/odds_win_place（race_odds 由来）。race_result_flat.odds は使わない。番兵値（win_odds<=0）は除く。
- 払戻を使う評価は、データ監査の除外レース（odds_audit/races_*.txt の 4 本）を外す。
出力: data/local/modeling/odds_pred/（ODDS_PRED_OUT で変更）。
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
sys.path.insert(0, str(ROOT))

OUT = Path(os.environ.get("ODDS_PRED_OUT", ROOT / "data/local/modeling/odds_pred"))
BET = ROOT / "data/local/modeling/betting"
AUD = ROOT / "data/local/meta/odds_audit"
PO_DIR = Path(os.environ.get("TP71_PO", "/home/jovyan/work/tp71_out/po"))
OOF = ROOT / "data/local/modeling/training_pipeline_v4/v6/eval_rev/oof_z:RB@e.parquet"
KEY = ["race_id", "horse_id"]
YEARS = [2022, 2023, 2024, 2025]
PAYBACK_WIN = 0.797
EXCLUDE_FILES = ["races_null_payoff_races.txt", "races_finish_fixed_wide.txt",
                 "races_finish_position_le0_with_finish_time.txt", "races_payoff_fix_differs.txt"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def excluded_races() -> set[str]:
    s: set[str] = set()
    for f in EXCLUDE_FILES:
        p = AUD / f
        if p.exists():
            s |= {x.strip() for x in p.read_text().split() if x.strip()}
    return s


def load_wp() -> pd.DataFrame:
    wp = pd.read_parquet(BET / "odds_win_place.parquet")
    wp["race_id"], wp["horse_id"] = wp["race_id"].astype(str), wp["horse_id"].astype(str)
    wp = wp[(wp["win_odds"] > 0) & (wp["place_odds_min"] > 0) & (wp["place_odds_max"] > 0)].copy()
    wp = wp.drop_duplicates(KEY, keep="last")
    wp["year"] = wp["race_id"].str[:4].astype(int)
    wp["place_mid"] = (wp["place_odds_min"] + wp["place_odds_max"]) / 2
    inv = 1.0 / wp["win_odds"]
    wp["p_mkt"] = inv / inv.groupby(wp["race_id"]).transform("sum")
    n = wp.groupby("race_id")["horse_id"].transform("size")
    wp = wp[n >= 4]
    return wp.reset_index(drop=True)


def _frame_with_targets():
    from src.pipeline.betting import pred_odds as po
    from src.scripts.data import tp27_eval as t
    from src.scripts.data import v6_eval as ev

    df, _base, _meta, _folds, _lp = t.setup()
    X = ev.build_frame(ev.config_of("z:RB@e"), df)
    X = X.select_dtypes(include=[np.number])
    # mtl_（補助ヘッド NN の出力）は 2025 年で分布がずれる（診断: 外すと 2025 年の想定単勝 RMSE 0.87→0.50）ので使わない
    drop_pre = tuple(p_ for p_ in os.environ.get("ODDS_DROP_PREFIXES", "mtl_").split(",") if p_)
    if drop_pre:
        X = X.drop(columns=[c for c in X.columns if c.startswith(drop_pre)])
    bad = [c for c in X.columns if po.ODDS_PATTERN.search(c)]
    if bad:
        _log(f"オッズ・人気に関わる名前の列を除外: {bad}")
        X = X.drop(columns=bad)
    return df, X.astype("float32")


def _train_predict(Xtr, ytr, Xva, yva, Xte, threads: int) -> tuple[np.ndarray, int]:
    import lightgbm as lgb

    m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.06, num_leaves=63, min_child_samples=60, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.4, reg_lambda=5.0, n_jobs=threads, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(60, verbose=False)])
    return m.predict(Xte), int(m.best_iteration_ or 0)


def phase1_fit(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    threads = int(os.environ.get("ODDS_THREADS", "4"))
    df, X = _frame_with_targets()
    keys = df[KEY].astype(str).reset_index(drop=True)
    year = pd.to_datetime(df["date"]).dt.year.to_numpy()
    wp = load_wp()
    T = keys.assign(_i=np.arange(len(keys))).merge(wp[KEY + ["win_odds", "place_mid", "p_mkt", "popularity"]], on=KEY, how="inner")
    idx = T["_i"].to_numpy()
    Xs = X.iloc[idx].reset_index(drop=True)
    yr = year[idx]
    y_share = np.log(T["p_mkt"].to_numpy())
    y_place = np.log(T["place_mid"].to_numpy())
    _log(f"行 {len(T)}、列 {Xs.shape[1]}、年別 {dict(zip(*np.unique(yr, return_counts=True)))}")
    cols = {"race_id": T["race_id"].to_numpy(), "horse_id": T["horse_id"].to_numpy(), "year": yr}
    out = pd.DataFrame(cols)
    out["share_raw"] = np.nan
    out["place_raw"] = np.nan
    info = {}
    for Y in YEARS:
        te = yr == Y
        tr_all = (yr < Y) & (yr >= 2020)
        last = Y - 1
        va = tr_all & (yr == last)
        tr = tr_all & (yr < last)
        if tr.sum() < 5000:
            tr = tr_all
            va = tr_all & (yr == last)
        for name, y in (("share", y_share), ("place", y_place)):
            p, it = _train_predict(Xs[tr], y[tr], Xs[va], y[va], Xs[te], threads)
            out.loc[te, f"{name}_raw"] = p
            info[f"{Y}_{name}"] = {"iters": it, "n_train": int(tr.sum())}
            _log(f"{Y} {name}: 木 {it} 本、学習 {int(tr.sum())} 行")
    out.to_parquet(OUT / "wp_raw_oof.parquet", index=False)
    (OUT / "wp_fit_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    _log("保存: wp_raw_oof.parquet")


# ───────────── 評価 ─────────────
def spearman_by_race(df: pd.DataFrame, a: str, b: str) -> float:
    r = df.groupby("race_id")[[a, b]].rank()
    out = []
    for _, g in r.groupby(df["race_id"]):
        if len(g) >= 4 and g[a].std() > 0 and g[b].std() > 0:
            out.append(np.corrcoef(g[a], g[b])[0, 1])
    return float(np.mean(out)) if out else float("nan")


def log_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    d = p - y
    sl = np.polyfit(p, y, 1)[0] if len(y) > 10 else float("nan")
    return {"rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.mean(np.abs(d))), "bias": float(np.mean(d)), "slope_actual_on_pred": float(sl), "n": int(len(y))}


def _norm_share(race_ids: np.ndarray, logv: np.ndarray) -> np.ndarray:
    v = np.exp(logv - logv.max())
    s = pd.Series(v).groupby(race_ids).transform("sum").to_numpy()
    return v / s


def _odds_from_share(race_ids: np.ndarray, share: np.ndarray, payback: float = PAYBACK_WIN) -> np.ndarray:
    return np.clip(payback / np.clip(share, 1e-6, None), 1.0, None)


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def build_predictions() -> pd.DataFrame:
    """OOF の生出力から、各方式の想定単勝オッズ・想定複勝（中央）オッズの列を作る。"""
    raw = pd.read_parquet(OUT / "wp_raw_oof.parquet")
    wp = load_wp()
    raw = raw[raw["year"] >= YEARS[0]]
    d = raw.merge(wp[KEY + ["horse_number", "win_odds", "place_odds_min", "place_odds_max", "place_mid", "p_mkt", "popularity"]], on=KEY, how="inner")
    d = d.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    rid = d["race_id"].to_numpy()
    d["ly"] = np.log(d["p_mkt"])
    d["lplace"] = np.log(d["place_mid"])
    # A: 生の予測（レース内で正規化して払戻率を乗せる）
    d["win_A"] = _odds_from_share(rid, _norm_share(rid, d["share_raw"].to_numpy()))
    d["place_A"] = np.clip(np.exp(d["place_raw"]), 1.05, None)
    # B: 縮みの補正（前の年までの OOF に ly ~ a + b·share_raw を当てはめ）。2022 は前年が無いので A と同じ
    d["win_B"] = d["win_A"]
    d["place_B"] = d["place_A"]
    for Y in YEARS[1:]:
        prev = (d["year"] >= 2022) & (d["year"] < Y)
        te = d["year"] == Y
        a, b = _fit_line(d.loc[prev, "share_raw"].to_numpy(), d.loc[prev, "ly"].to_numpy())
        d.loc[te, "win_B"] = _odds_from_share(rid[te], _norm_share(rid[te], a + b * d.loc[te, "share_raw"].to_numpy()))
        a2, b2 = _fit_line(d.loc[prev, "place_raw"].to_numpy(), d.loc[prev, "lplace"].to_numpy())
        d.loc[te, "place_B"] = np.clip(np.exp(a2 + b2 * d.loc[te, "place_raw"].to_numpy()), 1.05, None)
    # V8（既存の tp-71。各年より前だけで学習）
    v8 = []
    for Y in YEARS:
        f = PO_DIR / f"po_{Y}.parquet"
        if f.exists():
            x = pd.read_parquet(f)
            x["race_id"], x["horse_id"] = x["race_id"].astype(str), x["horse_id"].astype(str)
            v8.append(x[KEY + ["win_pred", "place_pred"]].rename(columns={"win_pred": "win_V8", "place_pred": "place_V8"}))
    if v8:
        d = d.merge(pd.concat(v8), on=KEY, how="left")
        rid = d["race_id"].to_numpy()
        ok = d["win_V8"].notna()
        d["win_V8"] = np.where(ok, _odds_from_share(rid, _norm_share(rid, -np.log(d["win_V8"].fillna(10.0).to_numpy()))), np.nan)
    # 自前の確率（OOF は 2022〜2024 のみ）からの機械的な変換
    o = pd.read_parquet(OOF)
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    d = d.merge(o[KEY + ["p_win", "p_top2", "p_top3"]], on=KEY, how="left")
    rid = d["race_id"].to_numpy()
    has = d["p_win"].notna().to_numpy()
    d["win_own"] = np.where(has, _odds_from_share(rid, np.where(has, d["p_win"].fillna(1e-3).to_numpy(), 1e-3) / pd.Series(np.where(has, d["p_win"].fillna(0).to_numpy(), 0)).groupby(rid).transform("sum").replace(0, 1).to_numpy()), np.nan)
    d["place_own"] = np.where(has, np.clip(0.80 / np.clip(d["p_top3"].fillna(0.3).to_numpy(), 0.02, None), 1.05, None), np.nan)
    # C: モデル(B)と自前確率の対数平均（2022〜24 のみ）
    d["win_C"] = np.where(has, _odds_from_share(rid, _norm_share(rid, 0.5 * np.log(1 / d["win_B"]) + 0.5 * np.log(1 / d["win_own"].fillna(10.0)))), np.nan)
    d["place_C"] = np.where(has, np.exp(0.5 * np.log(d["place_B"]) + 0.5 * np.log(d["place_own"].fillna(5.0))), np.nan)
    return d


def realized_tables() -> tuple[dict, dict]:
    p = pd.read_parquet(BET / "payoff.parquet")
    p["race_id"] = p["race_id"].astype(str)
    t = p[(p["bet_type"] == "tansho")]
    f = p[(p["bet_type"] == "fukusho")]
    tan = {(r, str(c)): v / 100.0 for r, c, v in zip(t["race_id"], t["combo"], t["payout"])}
    fuk = {(r, str(c)): v / 100.0 for r, c, v in zip(f["race_id"], f["combo"], f["payout"])}
    return tan, fuk


def ev_bands(ev: np.ndarray, ret: np.ndarray, race: np.ndarray, edges: list[float]) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ev >= lo) & (ev < hi)
        if m.sum() < 30:
            rows.append({"lo": lo, "hi": hi, "n": int(m.sum()), "roi": None})
            continue
        r = ret[m]
        # レース単位のブートストラップ（同一レースの複数馬を 1 まとまりに）
        g = pd.DataFrame({"r": r, "race": race[m]}).groupby("race")["r"].agg(["sum", "count"])
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(300):
            s = g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))
            bs.append(s["sum"].sum() / s["count"].sum())
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()), "mean_ev": float(ev[m].mean()), "roi": float(r.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]})
    return rows


def evaluate_phase1(a=None) -> None:
    d = build_predictions()
    d.to_parquet(OUT / "wp_oof_pred.parquet", index=False)
    exc = excluded_races()
    tan, fuk = realized_tables()
    res: dict = {"by_year": {}, "ev": {}}
    wins = ["win_A", "win_B", "win_V8", "win_own", "win_C"]
    places = ["place_A", "place_B", "place_V8", "place_own", "place_C"]
    for Y in YEARS:
        g = d[d["year"] == Y]
        r = {"n_rows": int(len(g)), "n_races": int(g["race_id"].nunique()), "win": {}, "place": {}}
        for c in wins:
            if c not in g or g[c].isna().all():
                continue
            m = g[c].notna()
            gg = g[m]
            lm = log_metrics(np.log(gg["win_odds"].to_numpy()), np.log(gg[c].to_numpy()))
            lm["spearman_race"] = spearman_by_race(gg.assign(_p=-gg[c], _a=-gg["win_odds"]), "_p", "_a")
            r["win"][c] = lm
        for c in places:
            if c not in g or g[c].isna().all():
                continue
            gg = g[g[c].notna()]
            r["place"][c] = log_metrics(np.log(gg["place_mid"].to_numpy()), np.log(gg[c].to_numpy()))
        res["by_year"][str(Y)] = r
        _log(f"{Y}: " + " ".join(f"{k}={v['rmse']:.3f}" for k, v in r["win"].items()))
    # 較正（暗黙確率の十分位 → 実際の市場確率の平均）
    cal = {}
    for c in ("win_A", "win_B", "win_V8"):
        g = d[d[c].notna() & (d["year"] >= 2023)].copy()
        if len(g) < 100:
            continue
        g["ip"] = 1.0 / g[c]
        g["q"] = pd.qcut(g["ip"], 10, duplicates="drop")
        t = g.groupby("q", observed=True).agg(pred=("ip", "mean"), actual=("p_mkt", "mean"), n=("ip", "size"))
        t["ratio"] = t["actual"] / (t["pred"] / PAYBACK_WIN)
        cal[c] = t.reset_index(drop=True).round(4).to_dict("records")
    res["calibration_decile_2023_2025"] = cal
    # EV の単調性（2022〜24: 自前確率 × 想定オッズ → 実現回収率）
    ev_edges = [0, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 100]
    g = d[(d["year"] <= 2024) & d["p_win"].notna() & ~d["race_id"].isin(exc)].copy()
    g["ret_win"] = [tan.get((r, str(h)), 0.0) for r, h in zip(g["race_id"], g["horse_number"])]
    g["ret_place"] = [fuk.get((r, str(h)), 0.0) for r, h in zip(g["race_id"], g["horse_number"])]
    for c in ("win_A", "win_B", "win_V8", "win_C"):
        gg = g[g[c].notna() & g["win_V8"].notna()]
        res["ev"][c] = ev_bands((gg["p_win"] * gg[c]).to_numpy(), gg["ret_win"].to_numpy(), gg["race_id"].to_numpy(), ev_edges)
    gg = g
    res["ev"]["win_actual_oracle"] = ev_bands((gg["p_win"] * gg["win_odds"]).to_numpy(), gg["ret_win"].to_numpy(), gg["race_id"].to_numpy(), ev_edges)
    for c in ("place_A", "place_B", "place_V8", "place_C"):
        gg = g[g[c].notna() & g["place_V8"].notna()]
        res["ev"][c] = ev_bands((gg["p_top3"] * gg[c]).to_numpy(), gg["ret_place"].to_numpy(), gg["race_id"].to_numpy(), ev_edges)
    res["ev"]["place_actual_oracle"] = ev_bands((g["p_top3"] * g["place_mid"]).to_numpy(), g["ret_place"].to_numpy(), g["race_id"].to_numpy(), ev_edges)
    (OUT / "phase1_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    _log("保存: phase1_results.json, wp_oof_pred.parquet")


# ───────────── フェーズ2: 複数頭券種 ─────────────
from itertools import combinations  # noqa: E402

PAIR_TYPES = ["umaren", "wide", "umatan"]          # 実オッズがある券種
ALL_TYPES = PAIR_TYPES + ["sanrenpuku", "sanrentan"]  # 3連複・3連単は払戻（的中組）のみ


def _marginals(w: np.ndarray, place_mid: np.ndarray):
    """単勝の暗黙確率と複勝オッズ（中央）から、勝率・連対率・複勝率の周辺確率を作る（連対率は勝率と複勝率の平均）。"""
    from src.pipeline.inference.exotic_probabilities import normalize_marginal

    w = w / w.sum()
    p3 = normalize_marginal(1.0 / np.clip(place_mid, 1.0, None), 3.0)
    p3 = np.maximum(p3, w)
    p2 = normalize_marginal(np.maximum((w + p3) / 2, w), 2.0)
    return w, p2, np.maximum(p3, p2)


def _dist(w, p2, p3):
    try:
        from src.pipeline.inference.exotic_probabilities import fit_blend_distribution
    except ImportError:          # ホストの exotic_probabilities.py が旧版のとき（MB60 を含む版を別名で置く）
        from src.pipeline.inference.exotic_probabilities_mb60 import fit_blend_distribution

    d, _ = fit_blend_distribution(w, p2, p3)
    return d


def _arrays(dist, hn: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """戻り値 {券種: (キー文字列の配列, 確率の配列)}。キーは馬番表記（小さい順 'a-b'、馬単 'a>b'、3連複 'a-b-c'）。"""
    n = len(hn)
    p2, p3 = dist._p2, dist._p3
    iu, ju = np.triu_indices(n, 1)
    a, b = np.minimum(hn[iu], hn[ju]), np.maximum(hn[iu], hn[ju])
    out = {"umaren": (np.char.add(np.char.add(a.astype(str), "-"), b.astype(str)), (p2 + p2.T)[iu, ju])}
    ii, jj = np.where(~np.eye(n, dtype=bool))
    out["umatan"] = (np.char.add(np.char.add(hn[ii].astype(str), ">"), hn[jj].astype(str)), p2[ii, jj])
    if n >= 3:
        T = p3 + p3.transpose(0, 2, 1) + p3.transpose(1, 0, 2) + p3.transpose(1, 2, 0) + p3.transpose(2, 0, 1) + p3.transpose(2, 1, 0)
        out["wide"] = (out["umaren"][0], T.sum(axis=2)[iu, ju])
        tri = np.array(list(combinations(range(n), 3)))
        srt = np.sort(hn[tri], axis=1)
        keys = np.char.add(np.char.add(np.char.add(srt[:, 0].astype(str), "-"), np.char.add(srt[:, 1].astype(str), "-")), srt[:, 2].astype(str))
        out["sanrenpuku"] = (keys, T[tri[:, 0], tri[:, 1], tri[:, 2]])
    return out


def _sanrentan_hit(dist, hn: np.ndarray, order: tuple[int, int, int]) -> float:
    pos = {int(h): k for k, h in enumerate(hn)}
    if not all(h in pos for h in order):
        return float("nan")
    return float(dist._p3[pos[order[0]], pos[order[1]], pos[order[2]]])


def phase2_build(a=None) -> None:
    """年ごとに、全組合せの確率（実オッズ由来の周辺 P_act / 予測の周辺 P_hat / 自前モデルの周辺 P_own）と実オッズ・払戻を表にする。"""
    var_w, var_p = os.environ.get("ODDS_WIN_VARIANT", "win_B"), os.environ.get("ODDS_PLACE_VARIANT", "place_B")
    d = pd.read_parquet(OUT / "wp_oof_pred.parquet")
    exc = excluded_races()
    pay = pd.read_parquet(BET / "payoff.parquet")
    pay["race_id"] = pay["race_id"].astype(str)
    amb = set(pay.loc[pay["ambiguous"].astype(bool), "race_id"])
    pay = pay[~pay["race_id"].isin(exc | amb)]
    hitmap = {t: {(r, str(c)): v / 100.0 for r, c, v in zip(g["race_id"], g["combo"], g["payout"])} for t, g in pay.groupby("bet_type")}
    hit3: dict[str, list] = {}
    for (r_, c_), v_ in hitmap.get("sanrentan", {}).items():
        hit3.setdefault(r_, []).append((c_, v_))
    pr = pd.read_parquet(BET / "odds_pair.parquet", columns=["race_id", "bet_type", "combo", "odds", "odds_min", "odds_max"])
    pr["race_id"] = pr["race_id"].astype(str)
    pr = pr[pr["bet_type"].isin(PAIR_TYPES)]
    pr["odds_act"] = np.where(pr["bet_type"] == "wide", (pr["odds_min"] + pr["odds_max"]) / 2, pr["odds"])
    pr = pr[(pr["odds_act"] > 0) & ~pr["race_id"].isin(exc)][["race_id", "bet_type", "combo", "odds_act"]]
    pr["combo"] = pr["combo"].astype(str)
    for Y in YEARS:
        g = d[(d["year"] == Y) & ~d["race_id"].isin(exc)]
        rows = {t: [] for t in ALL_TYPES}
        t0 = time.time()
        for k, (rid, x) in enumerate(g.groupby("race_id", sort=True)):
            if len(x) < 4:
                continue
            hn = x["horse_number"].to_numpy(dtype=int)
            w_act, p2a, p3a = _marginals(x["p_mkt"].to_numpy(float), x["place_mid"].to_numpy(float))
            w_hat0 = 1.0 / x[var_w].to_numpy(float)
            w_hat, p2h, p3h = _marginals(w_hat0, x[var_p].to_numpy(float))
            Da, Dh, Do = _dist(w_act, p2a, p3a), _dist(w_hat, p2h, p3h), None
            A, H, O = _arrays(Da, hn), _arrays(Dh, hn), None
            if x["p_win"].notna().all():
                w_o = x["p_win"].to_numpy(float)
                Do = _dist(w_o / w_o.sum(), x["p_top2"].to_numpy(float), x["p_top3"].to_numpy(float))
                O = _arrays(Do, hn)
            for t in ("umaren", "wide", "umatan", "sanrenpuku"):
                if t not in A:
                    continue
                keys = A[t][0]
                hm = hitmap.get(t, {})
                rows[t].append(pd.DataFrame({"race_id": rid, "combo": keys, "P_act": A[t][1], "P_hat": H[t][1],
                                             "P_own": O[t][1] if O else np.nan,
                                             "ret": [hm.get((rid, c), 0.0) for c in keys]}))
            # 3連単は的中組だけ（全組合せは大きすぎる）
            for c3, v3 in hit3.get(rid, []):
                order = tuple(int(z) for z in c3.split(">"))
                rows["sanrentan"].append(pd.DataFrame({"race_id": [rid], "combo": [c3], "P_act": [_sanrentan_hit(Da, hn, order)],
                                                       "P_hat": [_sanrentan_hit(Dh, hn, order)],
                                                       "P_own": [_sanrentan_hit(Do, hn, order) if Do is not None else np.nan], "ret": [v3]}))
            if k % 2000 == 0:
                _log(f"{Y}: {k} レース ({time.time() - t0:.0f}s)")
        for t in ALL_TYPES:
            if not rows[t]:
                continue
            df_t = pd.concat(rows[t], ignore_index=True)
            if t in PAIR_TYPES:
                o = pr[pr["bet_type"] == t][["race_id", "combo", "odds_act"]]
                df_t = df_t.merge(o, on=["race_id", "combo"], how="left")
            df_t["year"] = Y
            df_t.to_parquet(OUT / f"pair_{t}_{Y}.parquet", index=False)
            _log(f"{Y} {t}: {len(df_t)} 行")


def _poly_fit(x: np.ndarray, y: np.ndarray, deg: int = 2) -> np.ndarray:
    if len(x) > 1_500_000:
        sel = np.random.default_rng(0).choice(len(x), 1_500_000, replace=False)
        x, y = x[sel], y[sel]
    return np.polyfit(x, y, deg)


def _load_pair(t: str) -> pd.DataFrame:
    parts = [pd.read_parquet(OUT / f"pair_{t}_{Y}.parquet") for Y in YEARS if (OUT / f"pair_{t}_{Y}.parquet").exists()]
    d = pd.concat(parts, ignore_index=True)
    for c in ("P_act", "P_hat", "P_own"):
        d[c] = d[c].astype("float64")
    return d


ODDS_EDGES = [1, 3, 5, 10, 20, 50, 100, 300, 1000, 1e9]
EV_EDGES = [0, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 1e9]


def _bands_by_odds(odds: np.ndarray, ret: np.ndarray, race: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(ODDS_EDGES[:-1], ODDS_EDGES[1:]):
        m = (odds >= lo) & (odds < hi)
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()), "roi": float(ret[m].mean()) if m.sum() else None})
    return rows


def phase2_eval(a=None) -> None:
    res: dict = {}
    final_rows = []
    for t in ALL_TYPES:
        if not all((OUT / f"pair_{t}_{Y}.parquet").exists() for Y in YEARS[:1]):
            continue
        d = _load_pair(t)
        d = d[(d["P_act"] > 0) & (d["P_hat"] > 0)].copy()
        d["x_act"], d["x_hat"] = np.log(1 / d["P_act"]), np.log(1 / d["P_hat"])
        has_odds = t in PAIR_TYPES
        if has_odds:
            d = d[d["odds_act"] > 0].copy()
            d["ly"] = np.log(d["odds_act"])
            tr_mask = np.ones(len(d), bool)
        else:                                            # 3連複・3連単: 的中組の払戻だけが価格情報
            tr_mask = (d["ret"] > 0).to_numpy()
            d["ly"] = np.where(d["ret"] > 0, np.log(d["ret"].clip(lower=1e-9)), np.nan)
        out_t: dict = {"n_rows": int(len(d)), "by_year": {}, "method_note": "写像の係数は 1 年を除いて当てはめ（係数は 2〜3 個）"}
        d["odds_H0"] = np.nan
        d["odds_H1"] = np.nan
        d["odds_H2"] = np.nan
        for Y in YEARS:
            te = (d["year"] == Y).to_numpy()
            tr = tr_mask & ~te & d["ly"].notna().to_numpy()
            if te.sum() == 0:
                continue
            c = float(np.median(np.exp(d.loc[tr, "ly"]) * d.loc[tr, "P_act"])) if has_odds else float(np.median(d.loc[tr, "ret"] * d.loc[tr, "P_act"]))
            d.loc[te, "odds_H0"] = np.clip(c / d.loc[te, "P_hat"], 1.0, None)
            co1 = _poly_fit(d.loc[tr, "x_act"].to_numpy(), d.loc[tr, "ly"].to_numpy())
            d.loc[te, "odds_H1"] = np.clip(np.exp(np.polyval(co1, d.loc[te, "x_hat"].to_numpy())), 1.0, None)
            co2 = _poly_fit(d.loc[tr, "x_hat"].to_numpy(), d.loc[tr, "ly"].to_numpy())
            d.loc[te, "odds_H2"] = np.clip(np.exp(np.polyval(co2, d.loc[te, "x_hat"].to_numpy())), 1.0, None)
        ev_mask = tr_mask if not has_odds else np.ones(len(d), bool)
        for Y in YEARS:
            g = d[d["year"] == Y]
            ge = g[g["ly"].notna()]
            r = {"n": int(len(g)), "n_eval": int(len(ge))}
            for v in ("H0", "H1", "H2"):
                r[v] = log_metrics(ge["ly"].to_numpy(), np.log(ge[f"odds_{v}"].to_numpy()))
            out_t["by_year"][str(Y)] = r
        # 買い目全体の回収率（予測オッズの帯ごと。単純に全組合せを等額で買う）と、自前確率による期待値の帯
        for v in ("H0", "H1", "H2"):
            ok = d["odds_" + v].notna()
            out_t[f"roi_by_pred_odds_{v}"] = _bands_by_odds(d.loc[ok, f"odds_{v}"].to_numpy(), d.loc[ok, "ret"].to_numpy(), d.loc[ok, "race_id"].to_numpy())
            q = d[ok & d["P_own"].notna()]
            if t != "sanrentan":
                out_t[f"ev_bands_{v}_2022_2024"] = ev_bands((q["P_own"] * q[f"odds_{v}"]).to_numpy(), q["ret"].to_numpy(), q["race_id"].to_numpy(), EV_EDGES)
        if has_odds:
            q = d[d["P_own"].notna()]
            out_t["ev_bands_actual_oracle_2022_2024"] = ev_bands((q["P_own"] * q["odds_act"]).to_numpy(), q["ret"].to_numpy(), q["race_id"].to_numpy(), EV_EDGES)
            out_t["roi_by_actual_odds"] = _bands_by_odds(d["odds_act"].to_numpy(), d["ret"].to_numpy(), d["race_id"].to_numpy())
        res[t] = out_t
        _log(t + ": " + " ".join(f"{Y}:H1 rmse={out_t['by_year'][str(Y)]['H1']['rmse']:.3f}/H2 {out_t['by_year'][str(Y)]['H2']['rmse']:.3f}" for Y in YEARS if str(Y) in out_t["by_year"]))
        if t != "sanrentan":
            keep = d[["race_id", "combo", "year", "odds_H0", "odds_H1", "odds_H2"] + (["odds_act"] if has_odds else [])].copy()
            keep.insert(1, "bet_type", t)
            keep.to_parquet(OUT / f"odds_pred_{t}.parquet", index=False)
    (OUT / "phase2_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    _log("保存: phase2_results.json")


def diag_2025(a=None) -> None:
    """2025 年の想定単勝の誤差が 2022〜24 年より大きい原因を、特徴量の群ごとに外して切り分ける（速い設定の LightGBM）。"""
    import lightgbm as lgb

    df, X = _frame_with_targets()
    keys = df[KEY].astype(str).reset_index(drop=True)
    year = pd.to_datetime(df["date"]).dt.year.to_numpy()
    wp = load_wp()
    T = keys.assign(_i=np.arange(len(keys))).merge(wp[KEY + ["p_mkt"]], on=KEY, how="inner")
    idx = T["_i"].to_numpy()
    Xs = X.iloc[idx].reset_index(drop=True)
    yr = year[idx]
    y = np.log(T["p_mkt"].to_numpy())
    pre = pd.Series([c.split("_")[0] for c in Xs.columns]).value_counts()
    _log("列の接頭辞: " + ", ".join(f"{k}:{v}" for k, v in pre.head(25).items()))
    groups = {"none": [], "s_(前段)": [c for c in Xs.columns if c.startswith("s_")], "fe_(暦)": [c for c in Xs.columns if c.startswith("fe_")],
              "mtl_": [c for c in Xs.columns if c.startswith("mtl_")], "rb_": [c for c in Xs.columns if c.startswith("rb_")],
              "jockey/trainer_prior": [c for c in Xs.columns if c.startswith(("jockey_prior_", "trainer_prior_", "fc_jk_", "pair_hj_", "pair_ht_"))]}
    tr, va, te = (yr < 2024), (yr == 2024), (yr == 2025)
    res = {}
    for name, drop in groups.items():
        cols = [c for c in Xs.columns if c not in set(drop)]
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.1, num_leaves=63, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.4,
                              reg_lambda=5.0, n_jobs=int(os.environ.get("ODDS_THREADS", "4")), verbose=-1)
        m.fit(Xs.loc[tr, cols], y[tr], eval_set=[(Xs.loc[va, cols], y[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
        out = {}
        for nm, mask in (("2024", va), ("2025", te)):
            p = m.predict(Xs.loc[mask, cols])
            rid = T["race_id"].to_numpy()[mask]
            sh = _norm_share(rid, p)
            odds_p, odds_a = _odds_from_share(rid, sh), PAYBACK_WIN / np.exp(y[mask])
            out[nm] = log_metrics(np.log(odds_a), np.log(odds_p))
        res[name] = {"n_drop": len(drop), "iters": int(m.best_iteration_ or 0), **{k: {a_: round(b_, 3) for a_, b_ in v.items() if a_ in ("rmse", "bias", "slope_actual_on_pred")} for k, v in out.items()}}
        _log(f"{name}: drop={len(drop)} 2024 rmse={out['2024']['rmse']:.3f} 2025 rmse={out['2025']['rmse']:.3f} bias={out['2025']['bias']:.3f} slope={out['2025']['slope_actual_on_pred']:.2f}")
    (OUT / "diag_2025.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def export_config(a=None) -> None:
    """本番・最適化で使う係数を JSON に書く（全 OOF 年 2022〜2025 で当てはめ）。2026 年の入力に適用する用。"""
    d = pd.read_parquet(OUT / "wp_oof_pred.parquet")
    cfg: dict = {"payback_win": PAYBACK_WIN, "note": "log(実際の p_mkt) = a + b·share_raw、log(実際の複勝中央オッズ) = a + b·place_raw（2022〜2025 の OOF）。複数頭券種は log(オッズ) = poly2(log(1/P_hat))（P_hat は想定単複から MB60 で作る確率）。3連複・3連単は的中組の払戻から当てはめ"}
    a1, b1 = _fit_line(d["share_raw"].to_numpy(), np.log(d["p_mkt"].to_numpy()))
    a2, b2 = _fit_line(d["place_raw"].to_numpy(), np.log(d["place_mid"].to_numpy()))
    cfg["win_share"] = {"a": a1, "b": b1}
    cfg["place_mid"] = {"a": a2, "b": b2}
    cfg["poly2"] = {}
    for t in ALL_TYPES:
        if not all((OUT / f"pair_{t}_{Y}.parquet").exists() for Y in YEARS):
            continue
        x = _load_pair(t)
        x = x[(x["P_hat"] > 0) & (x["P_act"] > 0)]
        if t in PAIR_TYPES:
            x = x[x["odds_act"] > 0]
            y = np.log(x["odds_act"].to_numpy())
        else:
            x = x[x["ret"] > 0]
            y = np.log(x["ret"].to_numpy())
        co = _poly_fit(np.log(1 / x["P_hat"].to_numpy()), y)
        cfg["poly2"][t] = [float(v) for v in co]
    (OUT / "odds_pred_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    _log("保存: odds_pred_config.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "eval1", "build2", "eval2", "diag", "export"])
    a = ap.parse_args()
    {"fit": phase1_fit, "eval1": evaluate_phase1, "build2": phase2_build, "eval2": phase2_eval, "diag": diag_2025, "export": export_config}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

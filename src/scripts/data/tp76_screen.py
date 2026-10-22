"""tp-76: 学習前の特徴量スクリーニング。候補特徴量（A〜G）を as-of で作り、基準モデル（v6e の LOYO OOF 2022〜24 と test 2025）の
logit に少数の係数で足したときの LogLoss の増分（負 = 改善）を測る。モデルの学習（GBM/NN）はしない。市場オッズ・人気は使わない。

python -m src.scripts.data.tp76_screen prereg     # 候補・基準・シャドウ・束の設計を tp76_prereg.json に書く（評価の前にコミットする）
python -m src.scripts.data.tp76_screen build      # 特徴量を作る（<TP76_OUT>/features.parquet）＋リーク検査
python -m src.scripts.data.tp76_screen eval       # スクリーニング・シャドウ較正・基準の段階・束・合成スコア・ルール台帳の束
環境変数: TP76_MINING（tp72 の mining.parquet）, TP76_OUT, TP76_MODELS（data/local/modeling）, TP76_LEDGER（tp75_rule_ledger.csv）
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp76_asof as T

OUT = Path(os.environ.get("TP76_OUT", "data/local/modeling/tp76"))
MINING = Path(os.environ.get("TP76_MINING", "/home/jovyan/work/tp72_out/mining.parquet"))
MODELS = Path(os.environ.get("TP76_MODELS", "data/local/modeling"))
LEDGER = Path(os.environ.get("TP76_LEDGER", "docs/operations/tp75_rule_ledger.csv"))
PREREG = Path(__file__).with_name("tp76_prereg.json")

DISC = (2022, 2023, 2024)
TEST = 2025
TARGETS = ("win", "top2", "top3")
N_SHADOW_REP = 3

# 単位（候補）: 名前 -> (列, 族, 推論コスト 低/中/高, コストの理由)
UNITS: dict[str, tuple[list[str], str, str, str]] = {
    "A1": (["A_fit_front"], "A 同日バイアス", "中", "同日の自分より前のレース結果を、開催中にその都度取り込む必要"),
    "A2": (["A_fit_inner"], "A 同日バイアス", "中", "同上"),
    "A3": (["A_fit_front", "A_fit_inner"], "A 同日バイアス", "中", "同上"),
    "B1": (["B_m"], "B 潜在能力", "中", "馬ごとの状態（日次更新）と種牡馬の事前表が必要"),
    "B2": (["B_m", "B_sd"], "B 潜在能力", "中", "同上"),
    "B3": (["B_m_megu"], "B 潜在能力", "中", "めぐ指数の履歴が必要（RB と同じ日次状態）"),
    "C1": (["C_elo"], "C 対戦関係", "中", "全レースの時系列更新の状態（日次）が必要"),
    "C2": (["C_rel"], "C 対戦関係", "中", "同上＋出走馬の強さの平均"),
    "C3": (["C_rel", "C_unc"], "C 対戦関係", "中", "同上"),
    "D1": (["D_dist"], "D 条件別適性", "高", "馬×条件の集計と、種牡馬×条件の事前表の 2 種が必要"),
    "D2": (["D_venue"], "D 条件別適性", "高", "同上"),
    "D3": (["D_going"], "D 条件別適性", "高", "同上"),
    "D4": (["D_cush"], "D 条件別適性", "高", "同上＋クッション値"),
    "D5": (["D_all"], "D 条件別適性", "高", "同上"),
    "D5r": (["D_raw_all"], "D 条件別適性", "中", "馬×条件の集計のみ（種牡馬の事前表なし）"),
    "D6": (["D_ag_all"], "D 条件別適性", "高", "上がりの予測誤差を条件別に集計"),
    "E1": (["E_abs"], "E 上がりの形", "低", "過去走の上がりとコースの基準表"),
    "E2": (["E_day"], "E 上がりの形", "低", "同上＋同日の全レースの水準（過去走は確定済み）"),
    "E3": (["E_sustain", "E_accel"], "E 上がりの形", "低", "過去走の上がりと通過順"),
    "E4": (["E_unlucky_prev", "E_unlucky_m3"], "E 上がりの形", "低", "過去走の着順・上がり・通過順"),
    "E5": (["E_std5"], "E 上がりの形", "低", "過去 5 走の上がり順位"),
    "F1": (["F_n30", "F_n90", "F_since"], "F 負荷・疲労", "低", "過去走の日付だけ"),
    "F2": (["F_dist90", "F_burden90"], "F 負荷・疲労", "低", "過去走の距離・タイム"),
    "F3": (["F_tataki"], "F 負荷・疲労", "低", "過去走の間隔だけ"),
    "G1": (["G_n_front", "G_front_rank"], "G メンバー構成", "低", "出走表と各馬の普段の位置取り"),
    "G2": (["G_inner_front"], "G メンバー構成", "低", "同上"),
    "G3": (["G_fit"], "G メンバー構成", "低", "同上"),
}
ALL_COLS = sorted({c for v in UNITS.values() for c in v[0]})
HYPER = {
    "ridge_lambda_unit": 1.0, "ridge_lambda_bundle": 200.0, "ridge_lambda_rules": 400.0,
    "missing_indicator_if_rate_over": 0.02, "bootstrap_B": 500, "seed": 76,
    "corr_cluster_threshold": 0.85,
}


# ------------------------------------------------------------------ 事前登録
def prereg() -> None:
    doc = {
        "id": "tp76", "title": "学習前の特徴量スクリーニング（候補 A〜G）",
        "frozen_before_eval": True,
        "base_model": "v6e（LOYO OOF 2022〜2024、test 2025 = bundle_v6_2024 の予測）。市場オッズ・人気は特徴量にも係数にも使わない。2026 は選択に使わず記述のみ",
        "n_units": len(UNITS), "n_columns": len(ALL_COLS), "units": {k: {"cols": v[0], "family": v[1], "infer_cost": v[2], "cost_note": v[3]} for k, v in UNITS.items()},
        "optional_unit": "B3（めぐ指数への適用）はめぐ指数の表を読めたときだけ作る。読めない場合は 26 単位とし、結果にその旨を書く",
        "metric": "Δ3 = 3 目的（勝ち・2 着内・3 着内）の 1 頭あたり 2 値 LogLoss の増分の平均（負 = 改善）。基準は切片だけ再較正した v6e",
        "model": "offset = logit(v6e 確率)、切片 + 候補の標準化した列（欠損率が 2% 超の列は欠損フラグ付き）の係数をロジスティック回帰（L2）で当てはめる",
        "folds": "LOYO: 2022/2023/2024 のそれぞれを保留して他の 2 年で当てはめ、保留年で評価。2025 は 2022〜2024 で当てはめて 1 回だけ評価",
        "ci": "レース単位のクラスタ（比推定量の標準誤差）と、報告用にはレース単位ブートストラップ B=500",
        "criteria": {
            "S1": "Δ3 の 95% 区間上限 < 0 かつ Δ3 ≤ −0.0002",
            "S2": "LOYO の 3 年すべて Δ3 < 0 かつ プールの点推定 ≤ −0.0002",
            "S3": "LOYO の 3 年すべて Δ3 < 0 かつ プールの点推定 ≤ −0.0001",
            "S4": "経験ベイズ（候補の効果の分布に正規事前、モーメント法）で P(効果<0) ≥ 0.8、または局所 FDR ≤ 0.2（帰無はシャドウの z 分布）",
            "S5": "片側 90%（S5a）または 80%（S5b）で上限 < 0",
        },
        "shadow": {
            "design": "候補の列ごとに偽の特徴量を作る。P = 同じ日の中で行の値を並べ替える（分布と欠損の構造は同じで情報だけ壊す）、H = 同じレース内で馬を 1 つずらして別の馬に付ける。1 単位につき 3 本（P, H, P）",
            "n_shadow_units": len(UNITS) * N_SHADOW_REP,
            "judgement": "各基準の偽陽性率 = 偽の単位のうち基準を満たした割合。本物の最良 Δ3 が偽の最良 Δ3 を上回るかも併記（Boruta 型）",
        },
        "bundle": {
            "weak_signal": "個別の Δ3 が小さくても、LOYO の 3 年で符号が一致した列は『弱いシグナル』として束に入れる",
            "order": "符号の一貫年数（多い順）→ 冗長性（他の列との平均絶対相関、低い順）。効果の大きさでは並べない",
            "redundancy": f"|相関| > {HYPER['corr_cluster_threshold']} の列は同じクラスタに束ね、代表を 1 列選ぶ",
            "curve": "順に 1 列ずつ足した束の Δ3（LOYO）を曲線にし、同じ列数のシャドウ束（P, H, P の 3 本）と比べる。2025 は曲線の最小点の列数で 1 回だけ評価",
            "composite": "全代表列を L2 ロジスティックで当てはめた線形予測を合成スコア（勝ち用・3 着内用の 2 本）にする。係数は A 年で当てはめ、切片と傾きは B 年で較正し、C 年で評価（発見年の外側の年に当てる）",
        },
        "rules_bundle": {
            "source": "docs/operations/tp75_rule_ledger.csv（1,070 型、市場非依存）の expr を該当フラグにして L2 ロジスティックで束ねる。全ルール／採用済みのみ／未採用のみ の 3 通り＋日内並べ替えのシャドウ",
            "pending": "tp-75b（第 2 ラウンド、連言ルール）の結果が出たら追加で取り込む",
        },
        "hyper": HYPER, "params": T.PARAMS, "fit_end": str(T.FIT_END.date()),
        "notes": ["血統（種牡馬）は現状の不完全な 10 世代表（5 世代＋牡アンカー）に基づく。完全な表が揃ったら B・D を再実行する（印: rerun_after_complete_10gen）",
                  "少数パラメータの加法的な増分は GBM の非線形・交互作用を過小評価する。判定は『不採用』ではなく本試験の優先度"],
        "rerun_after_complete_10gen": ["B1", "B2", "B3", "D1", "D2", "D3", "D4", "D5", "D6"],
    }
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("prereg:", len(UNITS), "units,", len(ALL_COLS), "cols,", len(UNITS) * N_SHADOW_REP, "shadow units")


# ------------------------------------------------------------------ 特徴量の作成
def load_mining() -> pd.DataFrame:
    cols = ["race_id", "horse_id", "date", "venue_code", "surface", "distance", "track_condition", "field_size", "round", "finish_position", "horse_number",
            "bracket_number", "age", "sex", "run_no", "lay_days", "pos1_frac", "posL_frac", "fin_pct", "agari_rank", "last_3f", "time_sec", "cush_bin",
            "dist_band", "sire", "year"]
    return pd.read_parquet(MINING, columns=cols)


def load_megu(d: pd.DataFrame) -> pd.DataFrame | None:
    root = MODELS / "megu_recompute_20261019e"
    parts = []
    for y in sorted(d["date"].dt.year.unique()):
        f = root / str(y) / "megu_index_flat.parquet"
        if not f.exists():
            return None
        m = pd.read_parquet(f)
        col = "megu_index_v3a_feat" if "megu_index_v3a_feat" in m.columns else ("megu_index" if "megu_index" in m.columns else None)
        if col is None:
            return None
        m = m[["race_id", "horse_id", col]].rename(columns={col: "megu"})
        parts.append(m)
    m = pd.concat(parts, ignore_index=True).dropna()
    mm = d[["race_id", "horse_id"]].merge(m, on=["race_id", "horse_id"], how="left")
    mm["megu_z"] = (mm["megu"] - mm.groupby("race_id")["megu"].transform("mean")) / mm.groupby("race_id")["megu"].transform("std")
    return mm[["race_id", "horse_id", "megu_z"]]


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_mining()
    d0 = T.prepare(df)
    mg = None
    try:
        mg = load_megu(d0)
    except Exception as e:  # noqa: BLE001
        print("megu 読めず:", e)
    f = T.build_features(df, megu=mg)
    f.to_parquet(OUT / "features.parquet")
    meta = {"rows": len(f), "megu": mg is not None, "cols": [c for c in f.columns if c not in ("race_id", "horse_id", "date")]}
    (OUT / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print("features:", f.shape, "megu:", mg is not None)
    leak_check(df, f)


def max_diff(a: np.ndarray, b: np.ndarray) -> float:
    """欠損の位置が違えば inf、そろっていれば値の最大差。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (np.isnan(a) != np.isnan(b)).any():
        return float("inf")
    return float(np.nanmax(np.abs(a - b))) if a.size else 0.0


def leak_check(df: pd.DataFrame, f_full: pd.DataFrame, n_probe: int = 2) -> dict:
    """(1) 日付で切って作り直した特徴量が、全期間で作った同じ行の値と一致するか（切り詰め一致）。
    (2) ある日以降の結果（着順・位置取り・上がり・タイム）を壊しても、その日までの行の特徴量が変わらないか（未来の摂動）。
    (3) 同じ日の、あるレース以降の結果を壊しても、そのレースまでの特徴量が変わらないか（同日の先読み）。"""
    res = {"truncation": [], "future_perturb": [], "sameday_perturb": []}
    d = T.prepare(df)
    dates = np.sort(d["date"].unique())
    rng = np.random.default_rng(7)
    cols = [c for c in f_full.columns if c not in ("race_id", "horse_id", "date")]
    consts = T.fit_constants(d)
    key = f_full.set_index(["race_id", "horse_id"])
    outcome = ["finish_position", "pos1_frac", "posL_frac", "fin_pct", "agari_rank", "last_3f", "time_sec"]
    for cut in rng.choice(dates[dates > np.datetime64("2023-01-01")], n_probe, replace=False):
        cut = pd.Timestamp(cut)
        sub = df[pd.to_datetime(df["date"]) <= cut]
        g = T.build_features(sub, consts=consts).set_index(["race_id", "horse_id"])
        a = key.loc[g.index, cols]
        diff = max_diff(a.to_numpy(float), g[cols].to_numpy(float))
        res["truncation"].append({"cut": str(cut.date()), "rows": len(g), "max_abs_diff": diff})
        pert = df.copy()
        late = pd.to_datetime(pert["date"]) > cut
        shuf = rng.permutation(late.sum())
        for c in outcome:
            v = pert.loc[late, c].to_numpy()
            pert.loc[late, c] = v[shuf]
        h = T.build_features(pert, consts=consts).set_index(["race_id", "horse_id"])
        early = g.index
        diff2 = max_diff(h.loc[early, cols].to_numpy(float), g[cols].to_numpy(float))
        res["future_perturb"].append({"cut": str(cut.date()), "max_abs_diff": diff2})
    # 同日: ある日の途中のレース以降を壊す
    day = pd.Timestamp(rng.choice(dates[dates > np.datetime64("2023-06-01")]))
    dd = d[d["date"] == day]
    rid = sorted(dd["race_id"].unique())
    mid = rid[len(rid) // 2]
    pert = df.copy()
    sel = (pd.to_datetime(pert["date"]) > day) | ((pd.to_datetime(pert["date"]) == day) & (pert["race_id"] >= mid))
    shuf = rng.permutation(sel.sum())
    for c in outcome:
        v = pert.loc[sel, c].to_numpy()
        pert.loc[sel, c] = v[shuf]
    h = T.build_features(pert, consts=consts).set_index(["race_id", "horse_id"])
    base = f_full.set_index(["race_id", "horse_id"])
    keep = base.index[(base["date"] < day) | ((base["date"] == day) & (base.index.get_level_values(0) <= mid))]
    diff3 = max_diff(h.loc[keep, cols].to_numpy(float), base.loc[keep, cols].to_numpy(float))
    res["sameday_perturb"].append({"day": str(day.date()), "race_id": mid, "max_abs_diff": diff3})
    res["pass"] = all(x["max_abs_diff"] < 1e-9 for k in ("truncation", "future_perturb", "sameday_perturb") for x in res[k])
    (OUT / "leak_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("leak_check pass:", res["pass"], res)
    return res


# ------------------------------------------------------------------ ロジスティック
def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def fit_logit(X: np.ndarray, y: np.ndarray, off: np.ndarray, lam: float) -> np.ndarray:
    n, m = X.shape
    beta = np.zeros(m)
    pen = np.full(m, lam)
    pen[0] = 0.0
    for _ in range(40):
        eta = off + X @ beta
        pr = 1.0 / (1.0 + np.exp(-eta))
        g = X.T @ (pr - y) + pen * beta
        W = pr * (1 - pr)
        H = (X * W[:, None]).T @ X + np.diag(pen) + 1e-9 * np.eye(m)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


def loss_rows(X: np.ndarray, y: np.ndarray, off: np.ndarray, beta: np.ndarray) -> np.ndarray:
    eta = off + X @ beta
    return softplus(eta) - y * eta


class Data:
    """解析用の表（2022〜2025）。基準の logit、目的変数、レース番号を持つ。"""

    def __init__(self, f: pd.DataFrame, df: pd.DataFrame):
        pw = pd.read_parquet(MODELS / "training_pipeline_v4/tp27_v6e/ens/v6.parquet", columns=["race_id", "horse_id", "p_win", "p_top2", "p_top3"])
        t25 = pd.read_parquet(MODELS / "training_pipeline/final_eval_v6e/test2025_predictions.parquet", columns=["race_id", "horse_id", "p_win", "p_top2", "p_top3"])
        pr = pd.concat([pw, t25], ignore_index=True).drop_duplicates(["race_id", "horse_id"])
        base = df[["race_id", "horse_id", "finish_position", "year"]].merge(pr, on=["race_id", "horse_id"], how="inner")
        m = base.merge(f, on=["race_id", "horse_id"], how="left")
        m = m[(m["year"] >= 2022) & (m["year"] <= 2025) & (m["finish_position"] > 0)].sort_values(["date", "race_id"], kind="mergesort").reset_index(drop=True)
        self.m = m
        self.year = m["year"].to_numpy()
        self.date = m["date"].to_numpy()
        self.race = pd.factorize(m["race_id"])[0]
        fp = m["finish_position"].to_numpy()
        self.y = {"win": (fp == 1).astype(float), "top2": (fp <= 2).astype(float), "top3": (fp <= 3).astype(float)}
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        self.off = {"win": lg(m["p_win"].to_numpy()), "top2": lg(m["p_top2"].to_numpy()), "top3": lg(m["p_top3"].to_numpy())}
        self.disc = np.isin(self.year, DISC)
        self.test = self.year == TEST
        self.n = len(m)

    def col(self, name: str) -> np.ndarray:
        return self.m[name].to_numpy(float)


def design(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd[sd < 1e-12] = 1.0
    miss = np.isnan(Xtr).mean(axis=0) > HYPER["missing_indicator_if_rate_over"]
    def tf(X):
        Z = (X - mu) / sd
        parts = [np.ones((len(X), 1)), np.nan_to_num(Z, nan=0.0)]
        if miss.any():
            parts.append(np.isnan(X[:, miss]).astype(float))
        return np.hstack(parts)
    return tf(Xtr), tf(Xte)


def run_deltas(D: Data, X: np.ndarray, lam: float, use_test: bool = True) -> dict:
    """LOYO（2022/23/24 を保留）と 2025（2022〜24 で当てはめ）の、行ごとの損失差（候補 − 基準）。"""
    dl = {t: np.full(D.n, np.nan) for t in TARGETS}
    for yr in DISC:
        tr = D.disc & (D.year != yr)
        te = D.year == yr
        Atr, Ate = design(X[tr], X[te])
        for t in TARGETS:
            b0 = fit_logit(Atr[:, :1], D.y[t][tr], D.off[t][tr], 0.0)
            bc = fit_logit(Atr, D.y[t][tr], D.off[t][tr], lam)
            l0 = loss_rows(Ate[:, :1], D.y[t][te], D.off[t][te], b0)
            l1 = loss_rows(Ate, D.y[t][te], D.off[t][te], bc)
            dl[t][te] = l1 - l0
    if use_test:
        tr, te = D.disc, D.test
        Atr, Ate = design(X[tr], X[te])
        for t in TARGETS:
            b0 = fit_logit(Atr[:, :1], D.y[t][tr], D.off[t][tr], 0.0)
            bc = fit_logit(Atr, D.y[t][tr], D.off[t][tr], lam)
            dl[t][te] = loss_rows(Ate, D.y[t][te], D.off[t][te], bc) - loss_rows(Ate[:, :1], D.y[t][te], D.off[t][te], b0)
    return dl


def ratio_stats(delta: np.ndarray, race: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    r = race[mask]
    d = delta[mask]
    u, inv = np.unique(r, return_inverse=True)
    S = np.bincount(inv, weights=d)
    N = np.bincount(inv).astype(float)
    mean = S.sum() / N.sum()
    se = float(np.sqrt(np.sum((S - mean * N) ** 2)) / N.sum())
    return float(mean), se, len(u)


def summarize(D: Data, dl: dict, boot: bool = False, rng=None) -> dict:
    d3 = np.nanmean(np.vstack([dl[t] for t in TARGETS]), axis=0)
    out = {}
    m, se, nr = ratio_stats(d3, D.race, D.disc)
    out.update({"d3": m, "se3": se, "n_races": nr})
    for t in TARGETS:
        mt, st, _ = ratio_stats(dl[t], D.race, D.disc)
        out[f"d_{t}"], out[f"se_{t}"] = mt, st
    for yr in DISC:
        out[f"d3_{yr}"], out[f"se3_{yr}"], _ = ratio_stats(d3, D.race, D.year == yr)
    out["d3_2025"], out["se3_2025"], _ = ratio_stats(d3, D.race, D.test)
    for t in TARGETS:
        out[f"d_{t}_2025"], _, _ = ratio_stats(dl[t], D.race, D.test)
    out["signs_neg"] = int(sum(out[f"d3_{y}"] < 0 for y in DISC))
    if boot:
        rng = rng or np.random.default_rng(HYPER["seed"])
        m_ = D.disc
        r = D.race[m_]
        u, inv = np.unique(r, return_inverse=True)
        S = np.bincount(inv, weights=d3[m_])
        N = np.bincount(inv).astype(float)
        cnt = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=HYPER["bootstrap_B"])
        bs = (cnt @ S) / (cnt @ N)
        out["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


# ------------------------------------------------------------------ シャドウ
def make_perms(D: Data, rng) -> list[np.ndarray]:
    """P: 同じ日の中で並べ替え、H: 同じレース内で 1 つずらす、P（別の乱数）。"""
    n = D.n
    perms = []
    for typ in ("P", "H", "P"):
        perm = np.arange(n)
        if typ == "P":
            grp = pd.Series(np.arange(n)).groupby(D.date).indices
        else:
            grp = pd.Series(np.arange(n)).groupby(D.race).indices
        for idx in grp.values():
            idx = np.asarray(idx)
            if typ == "P":
                perm[idx] = rng.permutation(idx)
            else:
                perm[idx] = np.roll(idx, 1)
        perms.append(perm)
    return perms


# ------------------------------------------------------------------ 経験ベイズ・基準の段階
def eb_posterior(d: np.ndarray, se: np.ndarray, d_real: np.ndarray, se_real: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(np.mean(d_real))
    tau2 = max(float(np.var(d_real, ddof=1) - np.mean(se_real ** 2)), 1e-12)
    w = tau2 / (tau2 + se ** 2)
    pm = mu + w * (d - mu)
    pv = w * se ** 2
    from scipy.stats import norm

    return norm.cdf((0 - pm) / np.sqrt(pv)), mu, tau2


def lfdr_fn(z_real: np.ndarray, z_null: np.ndarray):
    from scipy.stats import gaussian_kde, norm

    m0, s0 = float(np.mean(z_null)), float(np.std(z_null, ddof=1))
    kde = gaussian_kde(z_real, bw_method=0.6)
    def f(z):
        return np.clip(norm.pdf(z, m0, s0) / np.maximum(kde(z), 1e-12), 0, 1)
    return f, m0, s0


def criteria(rows: list[dict], real_mask: np.ndarray) -> dict:
    d = np.array([r["d3"] for r in rows])
    se = np.array([r["se3"] for r in rows])
    neg = np.array([r["signs_neg"] for r in rows])
    z = d / se
    post, mu, tau2 = eb_posterior(d, se, d[real_mask], se[real_mask])
    lf, m0, s0 = lfdr_fn(z[real_mask], z[~real_mask])
    lfd = lf(z)
    C = {
        "S1": (d + 1.96 * se < 0) & (d <= -0.0002),
        "S2": (neg == 3) & (d <= -0.0002),
        "S3": (neg == 3) & (d <= -0.0001),
        "S4": (d < 0) & ((post >= 0.8) | (lfd <= 0.2)),
        "S5a": d + 1.2816 * se < 0,
        "S5b": d + 0.8416 * se < 0,
    }
    tab = {}
    for k, v in C.items():
        tab[k] = {"real_pass": int(v[real_mask].sum()), "n_real": int(real_mask.sum()), "shadow_pass": int(v[~real_mask].sum()),
                  "n_shadow": int((~real_mask).sum()), "fpr": float(v[~real_mask].mean())}
    return {"table": tab, "pass": {k: v.tolist() for k, v in C.items()}, "eb": {"mu": mu, "tau2": tau2, "null_mean": m0, "null_sd": s0},
            "post": post.tolist(), "lfdr": lfd.tolist()}


# ------------------------------------------------------------------ ルール台帳の束
def rule_flags(m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    led = pd.read_csv(LEDGER)
    base = pd.read_parquet(MINING)
    base = base.merge(m[["race_id", "horse_id"]], on=["race_id", "horse_id"], how="inner")
    base = base.sort_values(["date", "race_id"], kind="mergesort")
    keys = m[["race_id", "horse_id"]]
    base = keys.merge(base, on=["race_id", "horse_id"], how="left")
    flags, failed = {}, 0
    adopted = []
    for _, r in led.iterrows():
        try:
            v = base.eval(r["expr"], engine="python")
            v = np.asarray(v, dtype=bool)
        except Exception:  # noqa: BLE001
            failed += 1
            continue
        if v.sum() < 50 or v.sum() > len(v) - 50:
            continue
        flags[r["id"]] = v
        if any(str(r.get(c)) == "True" for c in ("ev_buy_adopted", "ev_kill_adopted", "place_buy_adopted", "place_kill_adopted") if c in led.columns):
            adopted.append(r["id"])
    F = pd.DataFrame(flags)
    return F, {"ledger_rows": len(led), "failed": failed, "used": F.shape[1], "adopted_used": len([a for a in adopted if a in F.columns]), "adopted_ids": adopted}


def ridge_offset(Xs, y, off, lam, x0=None):
    from scipy.optimize import minimize

    n, m = Xs.shape
    def fg(w):
        b, ww = w[0], w[1:]
        eta = off + b + Xs @ ww
        loss = np.sum(softplus(eta) - y * eta) + 0.5 * lam * np.sum(ww ** 2)
        pr = 1 / (1 + np.exp(-eta))
        g = Xs.T @ (pr - y) + lam * ww
        return loss, np.r_[np.sum(pr - y), g]
    r = minimize(fg, np.zeros(m + 1) if x0 is None else x0, jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return r.x


def bundle_deltas(D: Data, X, lam: float, sparse: bool = False) -> dict:
    """多数の列（または 0/1 フラグ）を一括で入れる L2 ロジスティック。LOYO と 2025。"""
    import scipy.sparse as sp

    dl = {t: np.full(D.n, np.nan) for t in TARGETS}
    def one(tr, te, fill):
        Xtr, Xte = X[tr], X[te]
        if sparse:
            Xtr_s, Xte_s = sp.csr_matrix(Xtr.astype(np.float32)), sp.csr_matrix(Xte.astype(np.float32))
        else:
            A, B = design(Xtr, Xte)
            Xtr_s, Xte_s = A[:, 1:], B[:, 1:]
        for t in TARGETS:
            w = ridge_offset(Xtr_s, D.y[t][tr], D.off[t][tr], lam)
            w0 = ridge_offset(Xtr_s[:, :0] if not sparse else sp.csr_matrix((Xtr.shape[0], 0)), D.y[t][tr], D.off[t][tr], 0.0)
            e1 = D.off[t][te] + w[0] + Xte_s @ w[1:]
            e0 = D.off[t][te] + w0[0]
            y = D.y[t][te]
            fill[t][te] = (softplus(e1) - y * e1) - (softplus(e0) - y * e0)
    for yr in DISC:
        one(D.disc & (D.year != yr), D.year == yr, dl)
    one(D.disc, D.test, dl)
    return dl


# ------------------------------------------------------------------ 本体
def evaluate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(HYPER["seed"])
    f = pd.read_parquet(OUT / "features.parquet")
    meta = json.loads((OUT / "build_meta.json").read_text(encoding="utf-8"))
    df = load_mining()
    D = Data(f, df)
    units = {k: v for k, v in UNITS.items() if all(c in D.m.columns for c in v[0])}
    skipped = [k for k in UNITS if k not in units]
    perms = make_perms(D, rng)
    lam_u = HYPER["ridge_lambda_unit"]
    res = {"n_rows": D.n, "n_races": int(D.race.max() + 1), "units_used": list(units), "units_skipped": skipped, "meta": meta}
    rows, labels, kinds = [], [], []
    detail = {}
    for k, (cols, fam, cost, note) in units.items():
        X = np.column_stack([D.col(c) for c in cols])
        dl = run_deltas(D, X, lam_u)
        s = summarize(D, dl, boot=True, rng=rng)
        s["missing_rate"] = float(np.isnan(X).any(axis=1).mean())
        base_logit = D.off["win"]
        Z = np.nan_to_num((X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-12))
        s["corr_base_logit"] = float(max(abs(np.corrcoef(Z[:, j], base_logit)[0, 1]) for j in range(Z.shape[1])))
        detail[k] = s
        rows.append(s); labels.append(k); kinds.append(True)
    for r, perm in enumerate(perms):
        for k, (cols, fam, cost, note) in units.items():
            X = np.column_stack([D.col(c) for c in cols])[perm]
            s = summarize(D, run_deltas(D, X, lam_u))
            rows.append(s); labels.append(f"{k}#s{r}"); kinds.append(False)
    real_mask = np.array(kinds)
    cr = criteria(rows, real_mask)
    res["units"] = {k: {**detail[k], "family": units[k][1], "cols": units[k][0], "infer_cost": units[k][2], "cost_note": units[k][3]} for k in units}
    for i, k in enumerate(units):
        res["units"][k]["pass"] = {c: bool(cr["pass"][c][i]) for c in cr["pass"]}
        res["units"][k]["post_neg"] = cr["post"][i]
        res["units"][k]["lfdr"] = cr["lfdr"][i]
    res["criteria"] = cr["table"]
    res["eb"] = cr["eb"]
    d_real = np.array([r["d3"] for r, kd in zip(rows, kinds) if kd])
    d_sh = np.array([r["d3"] for r, kd in zip(rows, kinds) if not kd])
    res["best_real_d3"] = float(d_real.min())
    res["best_shadow_d3"] = float(d_sh.min())
    res["shadow_d3_q"] = {"min": float(d_sh.min()), "p05": float(np.percentile(d_sh, 5)), "median": float(np.median(d_sh)), "p95": float(np.percentile(d_sh, 95))}
    res["shadow_units"] = {lab: {"d3": r["d3"], "se3": r["se3"]} for lab, r, kd in zip(labels, rows, kinds) if not kd}

    # --- 列ごと（束の材料）
    colsd = {}
    for c in [c for c in ALL_COLS if c in D.m.columns]:
        X = D.col(c)[:, None]
        s = summarize(D, run_deltas(D, X, lam_u))
        colsd[c] = {k_: s[k_] for k_ in ("d3", "se3", "signs_neg", "d3_2022", "d3_2023", "d3_2024", "d3_2025")}
    Z = pd.DataFrame({c: pd.Series(D.col(c)).fillna(pd.Series(D.col(c)).mean()) for c in colsd})
    C = Z.corr().abs()
    red = {c: float((C[c].sum() - 1) / max(len(C) - 1, 1)) for c in C.columns}
    thr = HYPER["corr_cluster_threshold"]
    parent = {c: c for c in C.columns}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a in C.columns:
        for b in C.columns:
            if a < b and C.loc[a, b] > thr:
                parent[find(a)] = find(b)
    clusters: dict[str, list[str]] = {}
    for c in C.columns:
        clusters.setdefault(find(c), []).append(c)
    reps = []
    for mem in clusters.values():
        mem.sort(key=lambda c: (-colsd[c]["signs_neg"], red[c]))
        reps.append(mem[0])
    reps.sort(key=lambda c: (-colsd[c]["signs_neg"], red[c]))
    res["columns"] = {c: {**colsd[c], "redundancy": red[c]} for c in colsd}
    res["clusters"] = [sorted(m) for m in clusters.values() if len(m) > 1]
    res["bundle_order"] = reps
    weak = [c for c in reps if colsd[c]["signs_neg"] == 3]
    res["weak_signal_cols"] = weak

    # --- 束の曲線（LOYO）と、シャドウ束
    lam_b = HYPER["ridge_lambda_bundle"]
    curve = []
    for kcount in range(1, len(reps) + 1):
        X = np.column_stack([D.col(c) for c in reps[:kcount]])
        s = summarize(D, bundle_deltas(D, X, lam_b))
        curve.append({"k": kcount, "col": reps[kcount - 1], "d3": s["d3"], "se3": s["se3"], "d3_2025": s["d3_2025"], "signs_neg": s["signs_neg"]})
    sh_curves = []
    for perm in perms:
        cc = []
        for kcount in range(1, len(reps) + 1):
            X = np.column_stack([D.col(c) for c in reps[:kcount]])[perm]
            s = summarize(D, bundle_deltas(D, X, lam_b))
            cc.append(s["d3"])
        sh_curves.append(cc)
    res["bundle_curve"] = curve
    res["bundle_shadow_curves"] = sh_curves
    kbest = int(np.argmin([c["d3"] for c in curve])) + 1
    res["bundle_k_best"] = kbest
    res["bundle_eval_2025_at_kbest"] = {"k": kbest, "d3_2025": curve[kbest - 1]["d3_2025"]}

    # --- 合成スコア（A 年で係数、B 年で較正、C 年で評価）
    Xall = np.column_stack([D.col(c) for c in reps])
    comp = {t: np.full(D.n, np.nan) for t in TARGETS}
    for ci in DISC:
        others = [y for y in DISC if y != ci]
        accum = {t: [] for t in TARGETS}
        for a, b in ((others[0], others[1]), (others[1], others[0])):
            tra, trb, te = D.year == a, D.year == b, D.year == ci
            A, B_ = design(Xall[tra], Xall[trb])
            _, Te = design(Xall[tra], Xall[te])
            for t in TARGETS:
                tt = "top3" if t == "top2" else t
                w = fit_logit(A, D.y[tt][tra], D.off[tt][tra], lam_b)
                cb = B_[:, 1:] @ w[1:]
                ct = Te[:, 1:] @ w[1:]
                Bm = np.column_stack([np.ones(trb.sum()), cb])
                b2 = fit_logit(Bm, D.y[t][trb], D.off[t][trb], 0.0)
                b0 = fit_logit(Bm[:, :1], D.y[t][trb], D.off[t][trb], 0.0)
                Tm = np.column_stack([np.ones(te.sum()), ct])
                y = D.y[t][te]
                l1 = loss_rows(Tm, y, D.off[t][te], b2)
                l0 = loss_rows(Tm[:, :1], y, D.off[t][te], b0)
                accum[t].append(l1 - l0)
        for t in TARGETS:
            comp[t][D.year == ci] = np.mean(accum[t], axis=0)
    # 2025: A = 2022〜2023, B = 2024
    tra = np.isin(D.year, (2022, 2023)); trb = D.year == 2024; te = D.test
    A, B_ = design(Xall[tra], Xall[trb]); _, Te = design(Xall[tra], Xall[te])
    for t in TARGETS:
        tt = "top3" if t == "top2" else t
        w = fit_logit(A, D.y[tt][tra], D.off[tt][tra], lam_b)
        Bm = np.column_stack([np.ones(trb.sum()), B_[:, 1:] @ w[1:]])
        b2 = fit_logit(Bm, D.y[t][trb], D.off[t][trb], 0.0); b0 = fit_logit(Bm[:, :1], D.y[t][trb], D.off[t][trb], 0.0)
        Tm = np.column_stack([np.ones(te.sum()), Te[:, 1:] @ w[1:]])
        comp[t][te] = loss_rows(Tm, D.y[t][te], D.off[t][te], b2) - loss_rows(Tm[:, :1], D.y[t][te], D.off[t][te], b0)
    res["composite_nested"] = summarize(D, comp, boot=True, rng=rng)
    # 合成スコアのシャドウ
    shc = []
    for perm in perms:
        Xs = Xall[perm]
        cs = {t: np.full(D.n, np.nan) for t in TARGETS}
        for ci in DISC:
            others = [y for y in DISC if y != ci]
            acc = {t: [] for t in TARGETS}
            for a, b in ((others[0], others[1]), (others[1], others[0])):
                tra, trb, te = D.year == a, D.year == b, D.year == ci
                A, B_ = design(Xs[tra], Xs[trb]); _, Te = design(Xs[tra], Xs[te])
                for t in TARGETS:
                    tt = "top3" if t == "top2" else t
                    w = fit_logit(A, D.y[tt][tra], D.off[tt][tra], lam_b)
                    Bm = np.column_stack([np.ones(trb.sum()), B_[:, 1:] @ w[1:]])
                    b2 = fit_logit(Bm, D.y[t][trb], D.off[t][trb], 0.0); b0 = fit_logit(Bm[:, :1], D.y[t][trb], D.off[t][trb], 0.0)
                    Tm = np.column_stack([np.ones(te.sum()), Te[:, 1:] @ w[1:]])
                    acc[t].append(loss_rows(Tm, D.y[t][te], D.off[t][te], b2) - loss_rows(Tm[:, :1], D.y[t][te], D.off[t][te], b0))
            for t in TARGETS:
                cs[t][D.year == ci] = np.mean(acc[t], axis=0)
        shc.append(summarize(D, {t: np.where(np.isnan(cs[t]), 0, cs[t]) for t in TARGETS})["d3"])
    res["composite_shadow_d3"] = shc

    # --- ルール台帳の束
    try:
        Fr, rmeta = rule_flags(D.m)
        res["rules_meta"] = {k: v for k, v in rmeta.items() if k != "adopted_ids"}
        Xr = Fr.to_numpy(np.float32)
        lam_r = HYPER["ridge_lambda_rules"]
        out_r = {}
        sets = {"all": list(range(Xr.shape[1])), "adopted": [i for i, c in enumerate(Fr.columns) if c in set(rmeta["adopted_ids"])],
                "non_adopted": [i for i, c in enumerate(Fr.columns) if c not in set(rmeta["adopted_ids"])]}
        for nm, idx in sets.items():
            if not idx:
                continue
            s = summarize(D, bundle_deltas(D, Xr[:, idx], lam_r, sparse=True), boot=True, rng=rng)
            s["n_flags"] = len(idx)
            out_r[nm] = s
        sh_r = []
        for perm in perms[:2]:
            sh_r.append(summarize(D, bundle_deltas(D, Xr[perm], lam_r, sparse=True))["d3"])
        out_r["shadow_all_d3"] = sh_r
        res["rules_bundle"] = out_r
    except Exception as e:  # noqa: BLE001
        res["rules_bundle"] = {"error": str(e)}

    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("評価完了:", OUT / "results.json")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    {"prereg": prereg, "build": build, "eval": evaluate}[cmd]()


if __name__ == "__main__":
    main()

"""穴馬予想モデル（2026-11-18、ユーザーの依頼・統括役の承認）: 4 番人気以下の馬の 3 着以内（主）・1 着（副）を、
較正した市場の確率を offset にした xgb で学ぶ。第 2 段（model_cycles_market）の作法を流用する。

    python -m src.scripts.data.longshot_model gate1 [--years 2019,2020,2021] [--variants B0,B1,B1perm,A1]
        # 関門 1: 決める段（訓練年の中の walk-forward）で、基準 3'（B0）と穴馬向けの列を足した版を比べる
    python -m src.scripts.data.longshot_model report1      # 関門 1 の集計（事前登録 config/longshot_model/prereg_gate1.json の基準）

- 計画と記録: docs/operations/longshot_model_20261118.md。
- 市場の確率は、すべての腕で「T-60 の再現」（確定オッズ × 2026 年の実測のずれ。bet_opt.sim_t60）から作る。
  訓練の offset は seed ごとに 1 回引いた再現、検証は 5 通りの再現の平均。再現は確定の情報をほぼ残すので、どの腕にも同じ楽観が入る。
- 較正（favourite-longshot bias の補正）は fold の訓練の行だけで当てる（calibrate_fit）。
- 2026 年は着順・払戻を読まない。T-60 のずれのために、2026 年の race_result_flat からオッズと人気の列だけを読む（drift_pool_2026 の assert）。
- オッズ・人気は特徴量に入れない（offset にだけ使う）。足す列（ls_*）は X_xgb と meta の既存の列から作り、check_no_market_features を通す。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CYCLES_OUT", "/home/jovyan/work/retrain_out/cycles16_fix3")

from src.scripts.data import model_cycles as mc  # noqa: E402
from src.scripts.data import model_cycles_market as mm  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("LONGSHOT_OUT") or "/home/jovyan/work/retrain_out/longshot")
PREREG = ROOT / "config/longshot_model/prereg_gate1.json"
KEY = ["race_id", "horse_id"]
# 第 2 段の採用版（s2_13_noise_s3）と同じ土台の spec（offset と雑音は除く。offset はこの道具で作る）
BASE_SPEC = {
    "blank_before": {"year": 2020, "prefix": ["wc_", "tl_"]},
    "drop": ["wc_cc_d1", "wc_cc_d2", "wc_cc_curv", "wc_cc_slope", "wc_cc_slope3", "wc_c1_d1", "wc_c1_d2", "wc_c1_curv", "wc_c1_slope",
             "wc_c1_slope3", "wc_ca_d1", "wc_ca_d2", "wc_ca_curv", "wc_ca_slope", "wc_ca_slope3", "wc_ccH_d1", "wc_cc_n", "wc_c1_n",
             "wc_ca_n", "wc_ccH_n"],
    "drop_prefix": ["rb_nk_", "track_bias_", "sire_x_track_bias", "dam_sire_x_track_bias"],
}
LOCAL_VENUES = ("01", "02", "03", "04", "10")      # 札幌・函館・福島・新潟・小倉
FIELD_EDGES = (8, 12, 15)                           # 頭数の帯: 〜8・9〜12・13〜15・16〜
NO_OUTCOME_COLS = ("race_id", "horse_number", "odds", "popularity")   # 2026 年から読む列（着順・払戻は読まない）
LS_RACE_COLS = ["ls_field_band", "ls_handicap", "ls_local", "ls_local_handicap", "ls_class_grp", "ls_front_ratio"]   # レースの中で同じ値
LS_HORSE_COLS = ["ls_unlucky", "ls_unlucky_flag", "ls_closer_x_front", "ls_front_x_front"]                           # 馬ごとの値
LS_COLS = LS_RACE_COLS[:5] + LS_HORSE_COLS[:2] + ["ls_front_ratio"] + LS_HORSE_COLS[2:]
LS_GROUPS = {"G1": LS_RACE_COLS[:5], "G2": LS_HORSE_COLS[:2], "G3": ["ls_front_ratio"] + LS_HORSE_COLS[2:]}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- 純粋関数（テスト対象）
def field_band(n: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(FIELD_EDGES), np.asarray(n, float), side="left")


def class_group(grade: pd.Series, race_class: pd.Series) -> np.ndarray:
    """0 新馬・1 未勝利・2 条件（1〜3 勝・500 万〜1600 万）・3 上級（OP・重賞）。
    X の grade は 2020 年から 1〜3 勝が空になり、meta の race_class は 2019 年前半まで条件戦のクラスが無い。両方を見て決める
    （2016〜2025 の全行がどれかに当たることを確かめた）。どちらにも当たらない行は 3。"""
    g = grade.astype(str).reset_index(drop=True)
    rc = race_class.astype(str).reset_index(drop=True)
    return np.select([g.str.contains("新馬") | rc.str.contains("新馬"),
                      g.str.contains("未勝利") | rc.str.contains("未勝利"),
                      g.str.contains("勝") | rc.str.contains("勝クラス")], [0, 1, 2], 3)


def longshot_features(X: pd.DataFrame, race_class: pd.Series) -> pd.DataFrame:
    """穴馬向けの列（ls_*）。どれもレース前に分かる X・meta の列だけから作る（オッズ・人気・着順は使わない）。
    G1 荒れる型: 頭数の帯・ハンデ・ローカル・ローカル × ハンデ・クラスの群。
    G2 前走の不利: 前走の着順の率（(着順−1)/(頭数−1)）− 前走の上がりの順位の率（lag1_l3f_rank。0 が最速）。正 = 上がりの割に着順が悪い。
                   フラグは、上がりが上位 1/4 なのに着順が下半分。
    G3 展開: 先行の頭数の率（f2_nfront / 脚質の分かる頭数 f2_nknown）、その馬の脚質（a_style。0 = 前・1 = 後ろ）との積。"""
    n = X["field_size"].to_numpy(float)
    wr = X["weight_rule"].astype(str).to_numpy()
    vc = X["venue_code"].astype(str).to_numpy()
    o = pd.DataFrame(index=X.index)
    o["ls_field_band"] = field_band(n).astype(float)
    o["ls_handicap"] = (wr == "ハンデ").astype(float)
    o["ls_local"] = np.isin(vc, LOCAL_VENUES).astype(float)
    o["ls_local_handicap"] = o["ls_handicap"] * o["ls_local"]
    o["ls_class_grp"] = class_group(X["grade"], race_class).astype(float)
    l3 = X["lag1_l3f_rank"].to_numpy(float)
    pf, pn = X["d_prev_fin"].to_numpy(float), X["d_prev_field"].to_numpy(float)
    fp = np.where(pn > 1, (pf - 1) / np.where(pn > 1, pn - 1, 1), np.nan)
    o["ls_unlucky"] = fp - l3
    o["ls_unlucky_flag"] = np.where(np.isfinite(l3) & np.isfinite(fp), ((l3 <= 0.25) & (fp >= 0.5)).astype(float), np.nan)
    nk = X["f2_nknown"].to_numpy(float)
    fr = np.where(nk > 0, X["f2_nfront"].to_numpy(float) / np.where(nk > 0, nk, 1), np.nan)
    st = X["a_style"].to_numpy(float)
    o["ls_front_ratio"] = fr
    o["ls_closer_x_front"] = st * fr
    o["ls_front_x_front"] = (1.0 - st) * fr
    return o[LS_COLS].astype("float32")


def _shuffle_within(groups: np.ndarray, rng) -> np.ndarray:
    """groups の中で並べ替えた位置（元の i 番目に入る値の元の位置）。"""
    codes = pd.factorize(groups)[0]
    order = np.lexsort((rng.random(len(codes)), codes))     # グループの中で乱数の順
    base = np.lexsort((np.arange(len(codes)), codes))        # グループの中で元の順
    src = np.empty(len(codes), dtype=int)
    src[base] = order
    return src


def permute_control(ls: pd.DataFrame, race: np.ndarray, cluster: np.ndarray, seed: int) -> pd.DataFrame:
    """負の対照: 馬ごとの列はレースの中で並べ替える。レースの中で同じ値の列は、同じ開催日 × 場の中で、レースどうしを並べ替える。
    どちらも、列の分布は保ったまま、馬・レースとの対応だけを崩す。"""
    rng = np.random.default_rng(seed)
    out = ls.copy()
    for c in LS_HORSE_COLS:
        out[c] = ls[c].to_numpy()[_shuffle_within(race, rng)]
    first = pd.DataFrame({"race": race, "cl": cluster}).drop_duplicates("race")
    src_race = first["race"].to_numpy()[_shuffle_within(first["cl"].to_numpy(), rng)]
    mp = dict(zip(first["race"].to_numpy(), src_race))          # レース → 値を借りるレース
    row_of = pd.Series(np.arange(len(race))).groupby(race).first()
    take = row_of.loc[pd.Series(race).map(mp).to_numpy()].to_numpy()
    for c in LS_RACE_COLS:
        out[c] = ls[c].to_numpy()[take]
    return out


def calib_design(lq: np.ndarray, lq3: np.ndarray | None, fband: np.ndarray, cgrp: np.ndarray) -> np.ndarray:
    """較正の説明変数: logit(q)・その 2 乗・（3 着以内なら logit(q3_harville)）・頭数の帯・クラスの群のダミーと、logit(q) との積。"""
    cols = [lq, lq ** 2]
    if lq3 is not None:
        cols.append(lq3)
    for k in range(1, len(FIELD_EDGES) + 1):
        d = (fband == k).astype(float)
        cols += [d, d * lq]
    for k in range(1, 4):
        d = (cgrp == k).astype(float)
        cols += [d, d * lq]
    return np.column_stack(cols)


def calibrate_fit(design: np.ndarray, y: np.ndarray):
    """目盛りをそろえたロジスティック回帰。収束しなければ止める（警告のまま先に進まない）。"""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(StandardScaler(), LogisticRegression(C=100.0, max_iter=2000))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        m.fit(design, y)
    return m


def calibrate_logit(model, design: np.ndarray) -> np.ndarray:
    return model.decision_function(design)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def norm_race_cap(race: np.ndarray, p: np.ndarray, total: float) -> np.ndarray:
    """レース内で合計 total に直す（3 着以内は 3）。1 を超えた分は 1 に切る。"""
    return np.clip(mm.norm_race(race, np.clip(p, 1e-7, None), total), 1e-6, 1 - 1e-6)


def binary_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def cluster_boot_mean(v: np.ndarray, cluster: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """行の値 v の平均と、開催日 × 場（cluster）で束ねたブートストラップの 95% 区間。"""
    d = pd.DataFrame({"c": cluster, "v": v}).groupby("c")["v"].agg(["sum", "count"])
    s, n = d["sum"].to_numpy(), d["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    r = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def cluster_key(race_id: pd.Series, date: pd.Series) -> np.ndarray:
    """開催日 × 場（race_id の 5〜6 桁目が場）。"""
    return date.astype(str).to_numpy() + "_" + race_id.astype(str).str[4:6].to_numpy()


def fill_outside(p_arm: np.ndarray, p_ref: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """穴馬だけで学んだ腕（A1）の、学んでいない馬（本命側）の値を基準の腕の値で埋める（レース内の正規化の分母を揃えるため）。"""
    return np.where(inside, p_arm, p_ref)


# ---------------------------------------------------------------- 2026 年のずれ（オッズと人気だけ）
def drift_pool_2026(phase: str = "t60") -> list[np.ndarray]:
    """bet_cycles.load_drift の races と同じもの（確定の人気の順に並べた、レースごとの d = log(確定 / 発走前)。5 頭以上）。
    2026 年の race_result_flat からは NO_OUTCOME_COLS だけを読む。"""
    from src.config.data_paths import TABLES_DIR
    from src.scripts.data import bet_cycles as BC

    pre = pd.read_parquet(BC.PRE_ODDS)
    pre["race_id"] = pre["race_id"].astype(str)
    pre = pre[(pre["phase"] == phase) & (pd.to_numeric(pre["win_odds"], errors="coerce") > 0)][["race_id", "horse_number", "win_odds"]]
    fl = pd.read_parquet(Path(TABLES_DIR) / "2026" / "race_result_flat.parquet", columns=list(NO_OUTCOME_COLS))
    assert set(fl.columns) == set(NO_OUTCOME_COLS), "2026 年は着順・払戻を読まない"
    fl["race_id"] = fl["race_id"].astype(str)
    for c in ("horse_number", "odds", "popularity"):
        fl[c] = pd.to_numeric(fl[c], errors="coerce")
    pre["horse_number"] = pd.to_numeric(pre["horse_number"], errors="coerce")
    m = pre.merge(fl, on=["race_id", "horse_number"])
    m = m[(m["odds"] > 0) & (m["popularity"] > 0)].copy()
    m["d"] = np.log(m["odds"].to_numpy(float) / pd.to_numeric(m["win_odds"]).to_numpy(float))
    return [g.sort_values("popularity")["d"].to_numpy() for _, g in m.groupby("race_id") if len(g) >= 5]


def reproduce_q(meta: pd.DataFrame, odds: np.ndarray, pop: np.ndarray, pool: list[np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """meta の行の順に、T-60 の再現の市場の勝率 q（レース内で合計 1）と再現の人気。オッズの無い馬は頭数で割った値の半分。"""
    from src.research.course import bet_opt

    f = pd.DataFrame({"race_id": meta["race_id"].to_numpy(), "hn": meta["horse_number"].to_numpy(), "odds_final": odds,
                      "pop_final": np.where(np.isfinite(pop), pop, 99), "fin": np.nan})
    s = bet_opt.sim_t60(f, pool, seed)
    j = meta[["race_id", "horse_number"]].rename(columns={"horse_number": "hn"}).merge(s, on=["race_id", "hn"], how="left")
    o = j["pre"].to_numpy(float)
    race = meta["race_id"].to_numpy()
    inv = np.where(np.isfinite(o) & (o > 0), 1.0 / np.where(o > 0, o, 1.0), np.nan)
    n = pd.Series(np.ones(len(o))).groupby(race).transform("sum").to_numpy()
    q = inv / pd.Series(inv).groupby(race).transform("sum").to_numpy()
    q = np.where(np.isfinite(q) & (q > 0), q, 0.5 / n)
    return mm.norm_race(race, np.clip(q, 1e-5, 1.0)), j["pre_pop"].to_numpy(float)


# ---------------------------------------------------------------- 関門 1
def _load(max_year: int):
    """X（第 2 段と同じ土台）・meta・確定オッズ・人気。max_year より後の年の行は落とす（使わない行の再現・Harville を省く）。"""
    X, meta, info = mc.load_cache()
    X = mc.apply_spec(X, meta, BASE_SPEC)
    mm.check_no_market_features(X.columns)
    keep = np.flatnonzero((meta["year"] <= max_year).to_numpy())
    X, meta = X.iloc[keep].reset_index(drop=True), meta.iloc[keep].reset_index(drop=True)
    mk = pd.read_parquet(mm.MK / "market.parquet")[KEY + ["odds", "popularity"]]
    j = meta[KEY].merge(mk, on=KEY, how="left")
    num = lambda c: pd.to_numeric(j[c], errors="coerce").astype("float64").to_numpy(na_value=np.nan)  # noqa: E731
    return X, meta, info, num("odds"), num("popularity")


def _margins(race, q, fband, cgrp):
    """q（再現の勝率）から、勝ち・3 着以内の生の確率と、較正の design。"""
    from src.pipeline.models import zoo

    _, q3 = zoo.harville_topk(race, q, 1.0, 1.0)
    q3 = norm_race_cap(race, q3, 3.0)
    lq, lq3 = mm.logit(q), mm.logit(q3)
    des = {"win": calib_design(lq, None, fband, cgrp), "top3": calib_design(lq, lq3, fband, cgrp)}
    return {"win": q, "top3": q3}, des


def _run_conf(pre: dict, variants: list[str], seeds: int, smoke: bool) -> dict:
    return {"variants": variants, "seeds": seeds, "n_eval": int(pre["n_repro_eval"]), "prereg_sha": _sha(PREREG), "smoke": smoke, "ls_cols": LS_COLS,
            "cycles_out": str(mc.CYC)}


def cmd_gate1(a) -> None:
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    years = tuple(int(y) for y in a.years.split(","))
    if max(years) >= 2022:
        raise SystemExit("関門 1 は訓練年の中（〜2021）だけ。2022〜24 は凍結の後に 1 回")
    variants = a.variants.split(",")
    if "A1" in variants and "B0" in variants and variants.index("A1") < variants.index("B0"):
        raise SystemExit("A1 は B0 の値で本命側を埋めるので、B0 を先に置く")
    seeds = int(pre["seeds"]) if not a.smoke else 1
    conf = _run_conf(pre, variants, seeds, a.smoke)
    X, meta, info, odds, pop = _load(max(years))
    out_dir = OUT
    if a.smoke:                                         # 動作の確かめ: レースの 3% と、木の数 20
        keep = meta["race_id"].isin(pd.Series(meta["race_id"].unique()).sample(frac=0.03, random_state=0))
        idx = np.flatnonzero(keep.to_numpy())
        X, meta, odds, pop = X.iloc[idx].reset_index(drop=True), meta.iloc[idx].reset_index(drop=True), odds[idx], pop[idx]
        pre = {**pre, "xgb_params": {"n_estimators": 20}}
        out_dir = OUT / "smoke"
    race = meta["race_id"].to_numpy()
    fin = meta["finish_position"].to_numpy(float)
    ys = {"win": (fin == 1).astype(int), "top3": (fin <= 3).astype(int)}
    ls = longshot_features(X, meta["race_class"])
    mm.check_no_market_features(ls.columns)
    fband = ls["ls_field_band"].to_numpy(int)
    cgrp = ls["ls_class_grp"].to_numpy(int)
    clus = cluster_key(meta["race_id"], meta["date"])
    params = {**info["params"], **pre.get("xgb_params", {})}
    pool = drift_pool_2026()
    log(f"ずれの元: 2026 年 {len(pool)} レース。行 {len(meta)}（〜{max(years)} 年）")
    n_eval = conf["n_eval"]
    a1_min = int(pre["a1_train_min_pop"])
    t0 = time.time()
    # 再現は全行で引く（訓練: seed s、検証: 100+k）。人気は再現の版のもの
    rq_tr = {s: reproduce_q(meta, odds, pop, pool, s) for s in range(seeds)}
    rq_ev = {k: reproduce_q(meta, odds, pop, pool, 100 + k) for k in range(n_eval)}
    log(f"再現 {seeds}+{n_eval} 通り（{time.time() - t0:.0f}s）")
    mg_tr = {s: _margins(race, rq_tr[s][0], fband, cgrp) for s in rq_tr}
    mg_ev = {k: _margins(race, rq_ev[k][0], fband, cgrp) for k in rq_ev}
    log(f"Harville・design（{time.time() - t0:.0f}s）")
    Xb1 = pd.concat([X, ls], axis=1)
    base_cols = list(X.columns)
    del X
    perm = {s: permute_control(ls, race, clus, 777 + s) for s in range(seeds)} if "B1perm" in variants else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for vy in years:
        (tr, va), = mc.folds_from_meta(meta, valid_years=(vy,))
        outp = out_dir / f"gate1_{vy}.parquet"
        confp = outp.with_suffix(".json")
        if outp.exists() and not a.force:
            old = json.loads(confp.read_text(encoding="utf-8")) if confp.exists() else {}
            if {k: old.get(k) for k in conf} != conf:
                raise SystemExit(f"{outp} は別の設定で作った（{old}）。--force で作り直す")
            log(f"{vy} 既存（同じ設定）")
            continue
        o = meta.loc[va, KEY + ["year", "date", "horse_number", "finish_position"]].reset_index(drop=True)
        o["pop_final"] = pop[va]
        for k in rq_ev:
            o[f"pre_pop_{k}"] = rq_ev[k][1][va]
        Xva = Xb1.iloc[va]
        for t, y in ys.items():
            tot = 1.0 if t == "win" else 3.0
            acc = {v: np.zeros((n_eval, len(va))) for v in variants}
            mkt_raw = np.zeros((n_eval, len(va)))
            mkt_cal = np.zeros((n_eval, len(va)))
            for s in range(seeds):
                _, des_tr = mg_tr[s]
                cal = calibrate_fit(des_tr[t][tr], y[tr])
                m_tr = calibrate_logit(cal, des_tr[t][tr])
                m_ev = [calibrate_logit(cal, mg_ev[k][1][t][va]) for k in range(n_eval)]
                for k in range(n_eval):
                    mkt_cal[k] += norm_race_cap(race[va], sigmoid(m_ev[k]), tot) / seeds
                    if s == 0:
                        mkt_raw[k] = mg_ev[k][0][t][va]
                raw_b0 = None
                for v in variants:
                    rows, mtr = tr, m_tr
                    if v == "B0":
                        cols, Xs_va = base_cols, Xva[base_cols]
                    elif v in ("B1", "A1"):
                        cols, Xs_va = list(Xb1.columns), Xva
                    elif v == "B1perm":
                        cols = None
                    else:
                        raise SystemExit(f"未知の版 {v}")
                    if v == "A1":                       # 穴馬だけで学ぶ: 訓練の再現（seed s）の人気で a1_min 番人気以下
                        sel = rq_tr[s][1][tr] >= a1_min
                        rows, mtr = tr[sel], m_tr[sel]
                    if v == "B1perm":
                        Xp = pd.concat([Xb1[base_cols], perm[s]], axis=1)
                        X_tr, Xs_va = Xp.iloc[rows], Xp.iloc[va]
                    else:
                        X_tr = Xb1.iloc[rows, [Xb1.columns.get_loc(c) for c in cols]]
                    p = _fit_multi(X_tr, y[rows], Xs_va, params, s, mtr, m_ev)
                    del X_tr
                    if v == "B0":
                        raw_b0 = p
                    for k in range(n_eval):
                        pk = p[k]
                        if v == "A1":                   # 本命側（再現の人気が a1_min より上）は B0（無ければ較正した市場）の値で埋める
                            ref = raw_b0[k] if raw_b0 is not None else sigmoid(m_ev[k])
                            pk = fill_outside(pk, ref, rq_ev[k][1][va] >= a1_min)
                        acc[v][k] += norm_race_cap(race[va], pk, tot) / seeds
                    log(f"{vy} {t} seed {s} {v} 済み（{time.time() - t0:.0f}s）")
            for k in range(n_eval):
                o[f"{t}_mraw_{k}"] = mkt_raw[k]
                o[f"{t}_mcal_{k}"] = mkt_cal[k]
                for v in variants:
                    o[f"{t}_{v}_{k}"] = acc[v][k]
        o.to_parquet(outp, index=False)
        confp.write_text(json.dumps(conf, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"{vy} 書いた {outp}")
    (out_dir / "gate1_run.json").write_text(json.dumps({**conf, "years": list(years), "cols_base": len(base_cols), "seconds": round(time.time() - t0)},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")


def _sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def _fit_multi(X_tr, y_tr, X_va, params: dict, seed: int, margin_tr, margins_va: list[np.ndarray]) -> list[np.ndarray]:
    """1 回学んで、検証の行を複数の offset（再現の版ごと）で予測する。木の出力は offset に足すだけなので、木の出力を 1 回だけ計算する。"""
    import xgboost as xgb

    p = dict(params)
    m = xgb.XGBClassifier(n_estimators=p.get("n_estimators", 400), learning_rate=p.get("learning_rate", 0.05), max_depth=p.get("max_depth", 6),
                          subsample=p.get("subsample", 0.8), colsample_bytree=p.get("colsample_bytree", 0.7), reg_lambda=p.get("reg_lambda", 5.0),
                          min_child_weight=p.get("min_child_weight", 20), tree_method="hist", device=p.get("device", "cuda"),
                          enable_categorical=True, random_state=seed, n_jobs=4, verbosity=0)
    m.fit(X_tr, y_tr, base_margin=margin_tr)
    tree = m.predict(X_va, output_margin=True, base_margin=np.zeros(len(X_va)))
    return [sigmoid(tree + mv) for mv in margins_va]


# ---------------------------------------------------------------- 関門 1 の集計
def gate1_summary(d: pd.DataFrame, variants: list[str], n_eval: int, min_pop: int = 4, ref: str = "B0") -> dict:
    """確定の人気 min_pop 以下（4 なら 4 番人気以下）の行での logloss（再現の版の平均）と、腕どうしの差（開催日 × 場で束ねた区間）。
    差は「腕 − 相手」（負 = 腕のほうが良い）。"""
    sub = d[d["pop_final"] >= min_pop].reset_index(drop=True)
    cl = cluster_key(sub["race_id"], sub["date"])
    out = {"rows": int(len(sub)), "races": int(sub["race_id"].nunique())}
    for t in ("win", "top3"):
        y = ((sub["finish_position"] == 1) if t == "win" else (sub["finish_position"] <= 3)).to_numpy(float)
        ll = {}
        for arm in ["mraw", "mcal"] + variants:
            ll[arm] = np.mean([binary_logloss(y, sub[f"{t}_{arm}_{k}"].to_numpy()) for k in range(n_eval)], axis=0)
        res = {"base_rate": float(y.mean()), "logloss": {k: float(v.mean()) for k, v in ll.items()}, "diff": {}}
        pairs = [("mcal", "mraw")] + [(v, "mcal") for v in variants] + [(v, ref) for v in variants if v != ref]
        for a_, b_ in pairs:
            m, lo, hi = cluster_boot_mean(ll[a_] - ll[b_], cl)
            yrs = {}
            for yy in sorted(sub["year"].unique()):
                w = (sub["year"] == yy).to_numpy()
                yrs[str(int(yy))] = float((ll[a_][w] - ll[b_][w]).mean())
            res["diff"][f"{a_}-{b_}"] = {"mean": m, "lo": lo, "hi": hi, "years": yrs}
        cal = {}                                         # 較正: 腕の予測の 10 分位ごとの予測の平均と実際の率
        for arm in ["mraw", "mcal"] + variants:
            p = np.mean([sub[f"{t}_{arm}_{k}"].to_numpy() for k in range(n_eval)], axis=0)
            b = pd.qcut(p, 10, labels=False, duplicates="drop")
            g = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").mean()
            cal[arm] = {"pred": [round(float(x), 5) for x in g["p"]], "obs": [round(float(x), 5) for x in g["y"]]}
        res["calibration"] = cal
        out[t] = res
    return out


def cmd_report1(a) -> None:
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    run = json.loads((OUT / "gate1_run.json").read_text(encoding="utf-8"))
    if run.get("prereg_sha") != _sha(PREREG):
        raise SystemExit("事前登録が実行の後に変わっている")
    d = pd.concat([pd.read_parquet(OUT / f"gate1_{y}.parquet") for y in run["years"]], ignore_index=True)
    res = {"prereg_sha": _sha(PREREG), "run": run}
    for mp in pre["min_pop_shift"]:
        res[f"pop{mp}"] = gate1_summary(d, run["variants"], run["n_eval"], mp)
    res["verdict"] = gate1_verdict(res[f"pop{pre['min_pop']}"], pre)
    (OUT / "gate1_summary.json").write_text(json.dumps(mm._no_nan(res), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(mm._no_nan({k: res[k] for k in ("verdict",)}), ensure_ascii=False, indent=1))


def gate1_verdict(s: dict, pre: dict) -> dict:
    """事前登録の基準: 主（3 着以内）で、腕 − B0 の logloss の差の 95% 区間の上端が 0 未満、3 年のうち min_years_neg 年以上で負、
    並べ替えの対照（B1perm − B0）が流れていて、区間の上端が 0 以上。兆しのある腕が 1 つでもあれば関門 2 へ。"""
    out = {}
    t = pre["primary_target"]
    need = int(pre.get("min_years_neg", 2))
    perm = s[t]["diff"].get("B1perm-B0")
    perm_ok = perm is not None and perm["hi"] >= 0
    for v in pre["candidates"]:
        dd = s[t]["diff"].get(f"{v}-B0")
        if dd is None:
            continue
        yrs = list(dd["years"].values())
        neg = sum(x < 0 for x in yrs)
        out[v] = {"hi_below_0": dd["hi"] < 0, "years_neg": neg, "years": len(yrs), "perm_control_ok": perm_ok,
                  "signal": bool(dd["hi"] < 0 and len(yrs) == 3 and neg >= need and perm_ok)}
    out["any_signal"] = any(x["signal"] for x in out.values() if isinstance(x, dict))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate1")
    g.add_argument("--years", default="2019,2020,2021")
    g.add_argument("--variants", default="B0,B1,B1perm,A1")
    g.add_argument("--force", action="store_true")
    g.add_argument("--smoke", action="store_true")
    sub.add_parser("report1")
    a = ap.parse_args()
    {"gate1": cmd_gate1, "report1": cmd_report1}[a.cmd](a)


if __name__ == "__main__":
    main()

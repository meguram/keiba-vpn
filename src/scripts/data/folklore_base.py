"""v6 に頼らないベース（実力）モデルを年ごとに抜く形（walk-forward）で自作し、『前日の想定オッズで期待値 1.2 以上を買う』形を 2023〜2025 年で評価する
（通説の検証の章の第 13 節）。

    python -m src.scripts.data.folklore_base fit    # → <FOLKLORE_OUT>/base_oof.parquet・base_fit.json（2020〜2025 年の全出走に p_win・p_top3）
    python -m src.scripts.data.folklore_base eval   # → <FOLKLORE_OUT>/base.json（章は folklore_verify report で作る）

設計（計画: 第 12 節の『前日の想定オッズで期待値 1.2 以上の単勝 108.6%±7.3』の評価が 2 年分で、v6 の OOF が 2022〜2024 年しか無かったため）:
- ベース: LightGBM（1 着・3 着内の二値）。評価年 Y の学習は 2016〜Y−1 年だけ。反復回数は Y−1 年を検証にした早期停止で決め、全期間で学び直す。
  特徴量は master の as-of 列から、当日のオッズ・人気・レース結果由来の列を除いたもの（EXCLUDE と先頭の一致）。追い切り（training_laps 由来の 2020 年以降の列）も除く。
- 2 段目: 年 Y の評価には、2022〜Y−1 年のベースの OOF だけで、log(ベースの確率) と log(想定オッズの確率) の重みを当てはめる（想定オッズ share_raw は 2022 年から）。
- 買い方は事前に固定: 期待値 = 確率 × 想定オッズが 1.2 以上（記事の閾値）。払戻は確定オッズ。1.0・1.1・1.5 は感度としての併記。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.scripts.data.folklore_verify import OUT  # noqa: E402

YEARS_FIT = (2020, 2021, 2022, 2023, 2024, 2025)
YEARS_EVAL = (2023, 2024, 2025)
# 当日のオッズ・人気・レース結果・同じ日のレース由来で、先読みになる列（完全一致）
EXCLUDE = {
    "finish_position", "time_sec", "last_3f", "odds", "popularity", "fin", "pop", "l3f", "tsec", "y1", "y2", "y3", "tdiff", "margin", "passing_order",
    "q", "p_mkt", "r3", "ret", "ret_cap", "jday_minpop_rank", "tr_pop_rank", "own_pop_rank", "top2q", "fav_odds", "has_pacemaker",
    "pos1", "pos4", "pos1r", "pos4r", "pace_cat", "lap_cat", "pdiff", "kick", "lp_diff", "lp_acc", "lp_cv", "lp_type", "lt_exp", "lt_exp_share",
    "bias_pos", "bias_hn", "n_before", "race_fast", "day_fast", "stime", "year", "dt", "weight", "weight_change", "race_no", "horse_number", "bracket_number",
    # 当該レースの結果から作った列（先読み）: ok = 着順がある、l3rank = 当該レースの上がり順位、nige_f = 当該レースで逃げたか、megu = 当該レースの指数、index_* = バロメーター（レース後の値）
    "ok", "l3rank", "nige_f", "megu", "megu_index_v3a", "index_total", "index_start", "index_chase", "index_closing",
    "index_total_rank", "index_start_rank", "index_chase_rank", "index_closing_rank",
    # 当日にならないと分からない測定値（前日判断の土台にしない）: 馬体重・馬場・クッション値・含水率・気象・パドック・同じ日の騎手の成績
    "wt", "wtc", "wt_rank", "track", "bad", "cush", "tmoist", "dmoist", "temp", "wind", "rain_cum", "tmax", "cush_lo", "cush_hi", "dm_lo", "dm_hi", "tm_lo", "tm_hi",
    "pd_rank", "jday_k", "jday_n", "jday_w_before",
    # 評価年全体・全年で決めた閾値や順位、全年の最頻値、全期間の値を過去の行へ写した列（先読み）と、取消のあとの実頭数（前日には分からない）
    "j_top", "t_top", "j_nige_top", "j_nige_70", "mg_sd_q25", "t_rank", "tozai", "f_dirt", "f_1400", "f_dirt_share", "f_1400_share", "field", "field_size",
    # 追い切り（training_laps）から作った派生列は、追い切りなしの版に入らないようにする（追い切りの上積みをきれいに分けるため）
    "t_ippai_rate", "ok_l1_q25", "ok_l1_q50", "ok_l1_wq25", "ok_pat", "ok_pat2",
}
EXCLUDE_PREFIX = ("ok_", "tc_pos", "tc_neg", "tc_next", "tc_sp", "tc_has", "p_ok_", "t_first", "t_last", "tp_", "tp2_", "e_", "lt_", "tr_", "tl_", "w1_")
CATS = ["vc", "surf", "sex", "track", "dband", "io", "shape", "weight_rule", "tozai", "line", "grade", "cl_cmb", "dir2", "direction", "ok_int", "ok_eval", "w1_int"]
# 追い切りの特徴量（training_laps 由来、2020 年以降。tl 版のベースにだけ入れる）。調教日は出走日より前だけ。
TRAINING_FEATURES = ["ok_l1", "ok_acc", "ok_n_f", "ok_days", "ok_pos", "ok_slope", "ok_wood", "ok_poly", "ok_int", "ok_eval", "w1_int", "w1_l1", "w1_days",
                     "tl_n", "tl_n14", "tl_hard", "tl_hard14", "tl_ippai", "tl_first_days", "tl_best_mid"]
# 注意: tl_has（追い切りが載るか）は、後日に馬のページをまとめて取得した結果の載り方（その後の出走歴と関係する）なので入れない。欠損そのものは残る。載り方の情報はその欠損に残るので、上積みは上限として読む


def feature_columns(d: pd.DataFrame, with_training: bool = False) -> list[str]:
    cols = []
    for c in d.columns:
        if c in EXCLUDE or c.startswith(EXCLUDE_PREFIX) or c in ("race_id", "horse_id", "jockey_id", "trainer_id"):
            continue
        if c in CATS or pd.api.types.is_numeric_dtype(d[c]) or d[c].dtype == bool:
            cols.append(c)
    if with_training:
        cols += [c for c in TRAINING_FEATURES if c in d.columns and c not in cols]
    return cols


def design(d: pd.DataFrame, cols: list[str], cats: dict | None = None):
    X = pd.DataFrame(index=d.index)
    cats = cats or {}
    for c in cols:
        if c in CATS and not pd.api.types.is_numeric_dtype(d[c]):
            vals = cats.setdefault(c, sorted(d[c].dropna().astype(str).unique().tolist()))
            X[c] = pd.Categorical(d[c].astype("string"), categories=vals)
        else:
            X[c] = d[c].astype(float) if d[c].dtype == bool else d[c]
    return X, cats


def fit(variant: str = "") -> None:
    """variant == "tl" なら追い切りの特徴量（TRAINING_FEATURES）を加えた版（base_oof_tl.parquet）。"""
    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score

    d = pd.read_pickle(OUT / "_prep.pkl")
    d = d.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    cols = feature_columns(d, with_training=(variant == "tl"))
    X, cats = design(d, cols)
    y1 = (d["fin"] == 1).to_numpy()
    y3 = (d["fin"] <= 3).to_numpy()
    yr = d["year"].to_numpy()
    print("特徴量", len(cols), flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=10.0, verbose=-1, n_jobs=16, seed=7)
    out = []
    info = {"n_features": len(cols), "features": cols, "fits": {}}
    imp = None
    for Y in YEARS_FIT:
        tr = yr < Y
        va = yr == Y - 1
        te = yr == Y
        res = {}
        pred = {}
        for name, y in (("win", y1), ("top3", y3)):
            inner = tr & ~va
            m0 = lgb.train(params, lgb.Dataset(X[inner], y[inner]), num_boost_round=1500, valid_sets=[lgb.Dataset(X[va], y[va])],
                           callbacks=[lgb.early_stopping(50, verbose=False)])
            n_it = max(int(m0.best_iteration * 1.1), 50)
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=n_it)
            pred[name] = m.predict(X[te])
            res[name] = {"iters": n_it, "logloss": float(log_loss(y[te], pred[name])), "auc": float(roc_auc_score(y[te], pred[name]))}
            if Y == YEARS_FIT[-1] and name == "top3":
                imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print(Y, res, flush=True)
        info["fits"][str(Y)] = res
        o = d.loc[te, ["race_id", "horse_id", "horse_number", "year"]].copy()
        pw = pd.Series(pred["win"], index=o.index)
        o["p_win_raw"] = pw
        o["p_win"] = pw / pw.groupby(o["race_id"]).transform("sum")
        o["p_top3"] = pred["top3"]
        out.append(o)
    sfx = "_tl" if variant == "tl" else ""
    pd.concat(out).to_parquet(OUT / f"base_oof{sfx}.parquet", index=False)
    info["importance_top"] = {k: float(v) for k, v in imp.head(25).items()}
    if "tl_has" in d.columns:
        info["tl_coverage"] = {str(y): float(v) for y, v in d.groupby("year")["tl_has"].mean().items() if y >= 2020}
    json.dump(info, open(OUT / f"base_fit{sfx}.json", "w"), ensure_ascii=False, indent=1)
    print(imp.head(15).to_string())


# ---------------------------------------------------------------- 評価
THRESHOLDS = (1.0, 1.1, 1.2, 1.5)
MAIN_TH = 1.2  # 事前に固定した買い方（記事の閾値。評価期間を見て選ばない）


def _merge_inputs(base_file: str = "base_oof.parquet", market: str = "share") -> pd.DataFrame:
    """market == "share": odds_pred の想定オッズ（share_raw・place_raw、2022 年から）。"est": SmartRC の想定人気から作った前日の市場（prerace.parquet の E、2021 年から）。"""
    wp_raw = ROOT / "data/local/modeling/odds_pred/wp_raw_oof.parquet"
    wp = ROOT / "data/local/modeling/odds_pred/wp_oof_pred.parquet"
    d = pd.read_parquet(OUT / "flags.parquet", columns=["race_id", "horse_id", "horse_number", "year", "fin", "odds", "pop", "field", "win", "plc", "place_odds_min"])
    d = d[(d["odds"] > 0) & (d["year"] >= (2021 if market == "est" else 2022))].copy()
    for df_ in (d,):
        df_["race_id"] = df_["race_id"].astype(str)
        df_["horse_id"] = df_["horse_id"].astype(str)
    b = pd.read_parquet(OUT / base_file)
    b["race_id"], b["horse_id"] = b["race_id"].astype(str), b["horse_id"].astype(str)
    if market == "est":
        r = pd.read_parquet(OUT / "prerace.parquet", columns=["race_id", "horse_id", "E_lq", "E_lplc"]).rename(columns={"E_lq": "share_raw", "E_lplc": "place_raw"})
    else:
        r = pd.read_parquet(wp_raw, columns=["race_id", "horse_id", "share_raw", "place_raw"])
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    v = pd.read_parquet(wp, columns=["race_id", "horse_id", "p_win", "p_top3"]).rename(columns={"p_win": "v6_win", "p_top3": "v6_top3"})
    v["race_id"], v["horse_id"] = v["race_id"].astype(str), v["horse_id"].astype(str)
    d = d.merge(b[["race_id", "horse_id", "p_win", "p_top3"]], on=["race_id", "horse_id"], how="inner")
    d = d.merge(r, on=["race_id", "horse_id"], how="inner").merge(v, on=["race_id", "horse_id"], how="left")
    d = d.dropna(subset=["p_win", "p_top3", "share_raw", "place_raw"]).sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    d = d[d.groupby("race_id")["p_win"].transform("size") == d.groupby("race_id")["field"].transform("first")].reset_index(drop=True)
    return d


def _prep_cols(d: pd.DataFrame, market: str = "share") -> pd.DataFrame:
    d = d.copy()
    sh = d["share_raw"].to_numpy(float)
    mx = pd.Series(sh).groupby(d["race_id"].to_numpy()).transform("max").to_numpy()
    lse = np.log(pd.Series(np.exp(sh - mx)).groupby(d["race_id"].to_numpy()).transform("sum").to_numpy()) + mx
    d["lsh"] = sh - lse  # 想定オッズの確率（レース内で合計 1）の対数
    d["lpb"] = np.log(d["p_win"].clip(1e-4))
    p3 = d["p_top3"].clip(1e-4, 1 - 1e-4)
    d["lp3b"] = np.log(p3 / (1 - p3))
    d["lpv"] = np.log(d["v6_win"].clip(1e-4))
    p3v = d["v6_top3"].clip(1e-4, 1 - 1e-4)
    d["lp3v"] = np.log(p3v / (1 - p3v))
    d["zero"] = 0.0
    cfg = json.load(open(ROOT / "data/local/modeling/odds_pred/odds_pred_config.json"))
    d["win_hat"] = cfg["payback_win"] / np.exp(d["lsh"])  # 想定の単勝オッズ = 払戻率 / 想定の勝つ確率
    d["plc_hat"] = np.exp(d["place_raw"]) if market == "est" else np.exp(cfg["place_mid"]["a"] + cfg["place_mid"]["b"] * d["place_raw"])
    return d


def _two_stage(d: pd.DataFrame, wcols: list[str], pcols: list[str], years=YEARS_EVAL, first: int = 2022) -> pd.DataFrame:
    """年 Y ごとに、first〜Y−1 年の（ベースの OOF と想定オッズの）重みを当てはめ、Y 年の単勝・複勝の確率を出す。"""
    from src.scripts.data.folklore_model import model_win, model_top3

    out = []
    coefs = {}
    for Y in years:
        tr, te = d[d["year"].between(first, Y - 1)], d[d["year"] == Y]
        if len(tr) == 0 or len(te) == 0:
            continue
        pw, bw, _ = model_win(tr, te, [], wcols, "zero", lam_fixed=1)
        p3, b3, _ = model_top3(tr, te, [], pcols, "zero", lam_fixed=1)
        o = te[["race_id", "year"]].copy()
        o["pw"], o["p3"] = pw, p3
        out.append(o)
        coefs[str(Y)] = {"win": dict(zip(wcols, map(float, bw))), "top3": dict(zip(pcols + ["切片"], map(float, b3)))}
    res = pd.concat(out)
    res.attrs["coefs"] = coefs
    return res


def _stats(d: pd.DataFrame, pred: pd.DataFrame, kind: str, final: bool = False) -> dict:
    """final=True なら期待値を確定オッズで測る（発走直前に買える場合の対照）。False は前日の想定オッズ。払戻はどちらも確定オッズ。"""
    from src.scripts.data.folklore_model import ev_curve

    x = d.loc[pred.index]  # pred の行名は d の行名（_two_stage が te の行名を保つ）
    res = {}
    for label, m in (("全体", np.ones(len(pred), bool)), *[(str(y), (pred["year"] == y).to_numpy()) for y in sorted(pred["year"].unique())]):
        if kind == "win":
            ev = (pred["pw"] * (x["odds"] if final else x["win_hat"])).to_numpy()
            ret, hit = x["win"].to_numpy(), (x["fin"] == 1).to_numpy()
        else:
            ev = (pred["p3"] * (x["place_odds_min"].fillna(0) if final else x["plc_hat"])).to_numpy()
            ret, hit = x["plc"].to_numpy(), (x["fin"] <= 3).to_numpy()
        rows = ev_curve(ev[m], ret[m], hit[m], x["race_id"].to_numpy()[m], x["pop"].to_numpy(float)[m])
        res[label] = {f"{r['th']:.2f}": r for r in rows if r["th"] in THRESHOLDS}
    return res


def evaluate(arg: str = "") -> None:
    """arg: "" = 既定、"tl" = 追い切りの特徴量を加えたベース、"est" = SmartRC の想定人気による前日の市場、"tl+est" = 両方。"""
    from src.scripts.data.folklore_model import _cmp, _ll_bin

    toks = set(arg.split("+")) if arg else set()
    market = "est" if "est" in toks else "share"
    sfx = ("_" + arg.replace("+", "_")) if arg else ""
    first = 2021 if market == "est" else 2022
    years = (2022, 2023, 2024, 2025) if market == "est" else YEARS_EVAL
    d = _prep_cols(_merge_inputs("base_oof_tl.parquet" if "tl" in toks else "base_oof.parquet", market), market)
    info = {"races": {str(y): int(d[d["year"] == y]["race_id"].nunique()) for y in sorted(d["year"].unique())}, "variants": {}, "ll": {}}
    variants = {"B": (["lpb", "lsh"], ["lp3b", "place_raw"]), "S": (["lsh"], ["place_raw"])}
    preds = {}
    for k, (wc, pc) in variants.items():
        preds[k] = _two_stage(d, wc, pc, years=years, first=first)
        info["variants"][k] = {"win": _stats(d, preds[k], "win"), "plc": _stats(d, preds[k], "plc")}
        info["variants"][k + "_final"] = {"win": _stats(d, preds[k], "win", True), "plc": _stats(d, preds[k], "plc", True)}
        info.setdefault("coefs", {})[k] = preds[k].attrs["coefs"]
        print(k, {y: v.get("1.20") and (v["1.20"]["n"], round(v["1.20"]["roi"], 3), round(v["1.20"]["se"], 3)) for y, v in info["variants"][k]["win"].items()}, flush=True)
    # v6 を使った版（2023〜2024 年）との比較
    d6 = d[d["v6_win"].notna() & d["v6_top3"].notna()].reset_index(drop=True)
    p6 = _two_stage(d6, ["lpv", "lsh"], ["lp3v", "place_raw"], years=(2023, 2024), first=first)
    info["variants"]["V6"] = {"win": _stats(d6, p6, "win"), "plc": _stats(d6, p6, "plc")}
    info["variants"]["V6_final"] = {"win": _stats(d6, p6, "win", True), "plc": _stats(d6, p6, "plc", True)}
    # ベース単体の対数損失（v6・市場との比較、2023〜2024 年と各年）
    for label, ys in (("2023-2024", (2023, 2024)), ("2025", (2025,))):
        x = d[d["year"].isin(ys)]
        yw, y3 = (x["fin"] == 1).to_numpy().astype(float), (x["fin"] <= 3).to_numpy().astype(float)
        ll = {"win_base": float(-(yw * np.log(np.clip(x["p_win"], 1e-9, 1))).sum() / x["race_id"].nunique()),
              "top3_base": float(_ll_bin(x["p_top3"].to_numpy(), y3)),
              "win_share": float(-(yw * np.log(np.clip(np.exp(x["lsh"]), 1e-9, 1))).sum() / x["race_id"].nunique())}
        if label != "2025":
            xv = x[x["v6_win"].notna()]
            ywv, y3v = (xv["fin"] == 1).to_numpy().astype(float), (xv["fin"] <= 3).to_numpy().astype(float)
            ll["win_v6"] = float(-(ywv * np.log(np.clip(xv["v6_win"], 1e-9, 1))).sum() / xv["race_id"].nunique())
            ll["top3_v6"] = float(_ll_bin(xv["v6_top3"].to_numpy(), y3v))
            ll["cmp_win"] = _cmp(xv["p_win"].to_numpy(), xv["v6_win"].to_numpy(), ywv, xv["race_id"].to_numpy(), "win")
            ll["cmp_top3"] = _cmp(xv["p_top3"].to_numpy(), xv["v6_top3"].to_numpy(), y3v, xv["race_id"].to_numpy(), "top3")
        info["ll"][label] = ll
    # 偽の確認: 各レース内でベースの確率を入れ替える（ベースの情報を壊しても 1.2 以上の回収率が出てしまわないか）
    rc = pd.factorize(d["race_id"])[0]
    plc = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        order = np.lexsort((rng.random(len(d)), rc))
        dp = d.copy()
        dp["lpb"] = d["lpb"].to_numpy()[order]
        dp["lp3b"] = d["lp3b"].to_numpy()[order]
        pr = _two_stage(dp, ["lpb", "lsh"], ["lp3b", "place_raw"], years=years, first=first)
        st = _stats(d, pr, "win")["全体"].get("1.20")
        plc.append(st)
    info["placebo"] = [x and {"n": x["n"], "roi": x["roi"], "se": x["se"]} for x in plc]
    from sklearn.metrics import roc_auc_score

    auc = {}
    for y in years:
        x = d[d["year"] == y]
        y3 = (x["fin"] <= 3).to_numpy()
        fq = (1 / x["odds"]) / (1 / x["odds"]).groupby(x["race_id"]).transform("sum")
        auc[str(y)] = {"base": float(roc_auc_score(y3, x["p_top3"])), "pre_share": float(roc_auc_score(y3, x["lsh"])), "final_odds": float(roc_auc_score(y3, fq))}
        if x["v6_top3"].notna().all():
            auc[str(y)]["v6"] = float(roc_auc_score(y3, x["v6_top3"]))
    info["auc"] = auc
    fit = json.load(open(OUT / ("base_fit_tl.json" if "tl" in toks else "base_fit.json")))
    info["fit"] = fit["fits"]
    info["n_features"] = fit["n_features"]
    info["importance_top"] = fit["importance_top"]
    info["arg"] = arg
    json.dump(info, open(OUT / f"base{sfx}.json", "w"), ensure_ascii=False, indent=1)
    for k in ("B", "B_final", "S", "S_final", "V6", "V6_final"):
        for kind in ("win", "plc"):
            for lab, v in info["variants"][k][kind].items():
                print(k, kind, lab, " | ".join(f"{t}:{r['n']} {r['roi']:.3f}±{r['se']:.3f}" for t, r in v.items()))
    print("placebo", info["placebo"])
    print("ll", json.dumps(info["ll"], ensure_ascii=False))


# ---------------------------------------------------------------- 章（第 13 節）
def _roi_cell(r) -> str:
    """回収率のセル。点数が少ない（100 点未満）か誤差が出せない（SE=0）ときは、参考値として点数を添える。"""
    if not r:
        return "<td class='num'>30 点未満</td>"
    if r["n"] < 100 or r["se"] == 0:
        return f"<td class='num'>参考値 {r['roi'] * 100:.0f}%<br><span class='muted'>{r['n']:,} 点（少なく誤差を出せない）</span></td>"
    return f"<td class='num'>{r['roi'] * 100:.1f}% ±{r['se'] * 100:.1f}<br><span class='muted'>{r['n']:,} 点</span></td>"


def base_section(w) -> None:
    f = OUT / "base.json"
    if not f.exists():
        return
    o = json.load(open(f))
    V = o["variants"]

    def c(v, kind, year, th="1.20"):
        return _roi_cell(V[v][kind].get(year, {}).get(th))

    yrs = ["2023", "2024", "2025", "全体"]
    w("<h2 id='base'>13. 年ごとに抜く形（walk-forward）のベースで、前日の想定オッズ判断を確かめ直す</h2>")
    w("<p>第 12 節の『前日の想定オッズで期待値 1.2 以上の単勝 108.6%』は評価が 2 年分で、ベースに使った本番系モデル v6 の OOF が 2022〜2024 年だけだった。"
      "そこで v6 に頼らず、<b>評価年より前の年だけで学習したベース</b>を自作して、2023〜2025 年の 3 年で評価し直した。</p>")
    w("<div class='note'><b>v6 の OOF の読み方.</b> v6 の OOF は、検証年より前だけで学習した expanding（purge・embargo あり）で、年ごとに抜く形そのもの。"
      "ただし (1) 入力の一部（EASE2・ラップ予想）は leave-one-year-out の予測で『軽い先読みがある』と v6 の章に書かれている、(2) 特徴量・構成の採用を 2022〜2024 年の OOF で選んでいる（選択バイアス）、"
      "(3) 想定オッズ（odds_pred）も同じ特徴量の枠から作られ、同じ先読みを含みうる。下の表のとおり、v6 の 3 着内の AUC は確定オッズの市場に近い。"
      "自作ベースとの差は、特徴量・学習の始まりの年・選択の違いで、先読みの影響だけを切り分けてはいない。<b>v6 を使った結果は、上限として読む。</b></div>")
    w("<div class='wrap'><table><thead><tr><th>3 着内の AUC</th><th>自作ベース（厳密な walk-forward）</th><th>v6 OOF（expanding・軽い先読みあり）</th><th>想定オッズ（前日）</th><th>確定オッズ（発走直前）</th></tr></thead><tbody>")
    for y, a in o["auc"].items():
        w(f"<tr><td>{y}</td><td class='num'>{a['base']:.3f}</td><td class='num'>{a['v6']:.3f}</td>" if "v6" in a else f"<tr><td>{y}</td><td class='num'>{a['base']:.3f}</td><td class='num'>—</td>")
        w(f"<td class='num'>{a['pre_share']:.3f}</td><td class='num'>{a['final_odds']:.3f}</td></tr>")
    w("</tbody></table></div>")
    w("<p>作り方:</p><ul>"
      f"<li><b>ベース</b>: LightGBM（1 着・3 着内の二値）。評価年 Y の学習は 2016〜Y−1 年だけ。反復回数は Y−1 年を検証にした早期停止で決めてから、全期間で学び直した。"
      f"特徴量 {o['n_features']} 列（master の as-of 列）。当日のオッズ・人気、レース結果由来の列（着順の有無・上がり順位・当該レースの指数・バロメーターなど）、"
      "当日にならないと分からない測定値（馬体重・馬場・クッション値・気象・パドック・同じ日の騎手の成績）、評価年全体・全年で決めた閾値や順位（j_top・t_rank・tozai など）、取消後の頭数、追い切り（training_laps 由来の 2020 年以降の列。追い切りなしの版からは、それから作った派生列も）は除いた。"
      "<b>前走の人気・オッズ（p_odds・p_pop・p_q）は入っている</b>（過去の情報。重要度の上位）。</li>"
      "<li><b>2 段目</b>: 年 Y の評価には、2022〜Y−1 年のベースの OOF だけで、log(ベースの確率) と log(想定オッズの確率) の重みを当てはめた（想定オッズ = odds_pred の share_raw・place_raw、2022 年から。同じ特徴量の枠から作られている）。</li>"
      "<li><b>買い方</b>: 事前に固定。期待値 = 確率 × 想定オッズが 1.2 以上（記事の閾値）。払戻は確定オッズ。対照として、期待値を確定オッズで測る版（発走直前に買える場合の上限）も出した。</li></ul>")
    w("<h3>13.1 期待値 1.2 以上を買ったときの回収率（年別）</h3><div class='wrap'><table><thead><tr><th>買い方</th>" + "".join(f"<th>{y}</th>" for y in yrs) + "</tr></thead><tbody>")
    rows = [("単勝 — 前日の想定オッズで判断（自作ベース）", "B", "win"), ("複勝 — 前日の想定オッズで判断（自作ベース）", "B", "plc"),
            ("単勝 — 確定オッズで判断（対照・自作ベース）", "B_final", "win"), ("複勝 — 確定オッズで判断（対照・自作ベース）", "B_final", "plc"),
            ("単勝 — 前日の想定オッズで判断（v6、2023〜2024 年だけ・上限として）", "V6", "win"), ("複勝 — 前日の想定オッズで判断（v6、2023〜2024 年だけ・上限として）", "V6", "plc"),
            ("単勝 — 確定オッズで判断（対照・v6、2023〜2024 年だけ）", "V6_final", "win"), ("複勝 — 確定オッズで判断（対照・v6、2023〜2024 年だけ）", "V6_final", "plc")]
    for lab, v, kind in rows:
        w(f"<tr><td>{lab}</td>" + "".join(c(v, kind, y) for y in yrs) + "</tr>")
    w("</tbody></table></div>")
    b = V["B"]["win"]["全体"].get("1.20")
    bp = V["B"]["plc"]["全体"].get("1.20")
    bf = V["B_final"]["win"]["全体"].get("1.20")
    v6 = V["V6"]["win"]["全体"].get("1.20")
    v6f = V["V6_final"]["win"]["全体"].get("1.20")
    yr_list = [(y, V["B"]["win"].get(y, {}).get("1.20")) for y in ("2023", "2024", "2025")]
    yr_txt = "・".join(f"{y} 年 {r['roi'] * 100:.0f}%（{r['n']:,} 点）" for y, r in yr_list if r)
    over = [y for y, r in yr_list if r and r["n"] >= 100 and r["roi"] - 2 * r["se"] >= 1]
    auc_all = o["auc"]
    auc_b13 = np.mean([v["base"] for v in auc_all.values()])
    auc_f13 = np.mean([v["final_odds"] for v in auc_all.values()])
    t10, t15 = V["B"]["win"]["全体"].get("1.10"), V["B"]["win"]["全体"].get("1.50")
    w("<div class='note'><b>結論.</b> "
      f"年ごとに抜いた自作ベースでは、前日の想定オッズで期待値 1.2 以上の単勝は {b['roi'] * 100:.0f}%±{b['se'] * 100:.0f}（{b['n']:,} 点）、複勝は {bp['roi'] * 100:.0f}%±{bp['se'] * 100:.0f}（{bp['n']:,} 点）で、"
      f"年別の単勝は {yr_txt}。100% を有意に超えた年は{'ある' if over else '無い'}（点数の少ない年は誤差が大きく、閾値を変えると 1.10 で {t10['roi'] * 100:.0f}%±{t10['se'] * 100:.0f}・1.50 で {t15['roi'] * 100:.0f}%±{t15['se'] * 100:.0f}（{t15['n']} 点）と揺れる）。"
      f"期待値を確定オッズで測る対照でも単勝 {bf['roi'] * 100:.0f}%±{bf['se'] * 100:.0f} で、自作ベースは市場（3 着内 AUC は自作 {auc_b13:.3f}・確定オッズ {auc_f13:.3f}）より弱く、人気以上の情報を持つ証拠は出なかった。"
      f"v6 の OOF を使うと同じ買い方が {v6['roi'] * 100:.0f}%±{v6['se'] * 100:.0f}（2023〜2024 年）になるが、同じ v6 でも期待値を確定オッズで測ると {v6f['roi'] * 100:.0f}%±{v6f['se'] * 100:.0f} で、市場に勝つ証拠にはならない。"
      f"v6 の {v6['roi'] * 100:.0f}% は『想定オッズより長い所を拾う』形で出ており、想定オッズが v6 と同じ特徴量の枠から作られていること、v6 の選択バイアス、EASE2・ラップ予想の軽い先読みの影響を切り分けられていない。"
      "第 12 節の前日判断の単勝も同じ上限として読む。また想定オッズ（前日）は確定オッズの予測にすぎず、『想定オッズより有利』と『確定オッズより有利』は別。前日オッズの記録が無い今の手元のデータでは、前日判断の本当の成績は測れない。</div>")
    w("<h3>13.2 閾値を変えた感度（全体、自作ベース・前日の想定オッズ）</h3><div class='wrap'><table><thead><tr><th>期待値 ≥</th><th>単勝</th><th>複勝</th></tr></thead><tbody>")
    for th in ("1.00", "1.10", "1.20", "1.50"):
        w(f"<tr><td>{th}</td>{c('B', 'win', '全体', th)}{c('B', 'plc', '全体', th)}</tr>")
    w("</tbody></table></div>")
    pl = [x for x in o["placebo"] if x]
    w(f"<p>偽の確認: 各レース内でベースの確率を入れ替えて 2 段目をやり直すと（5 回）、期待値 1.2 以上の買いが 30 点以上出た回は {len(pl)} 回（ベースの情報を壊すと、想定オッズだけが残って期待値が 1.2 に届かない）。"
      "これは『ベースが買い判断に効いている』ことの確認で、回収率で比べた偽の確認ではない。"
      "また 13.1 の『全体』は、自作ベースが 2023〜2025 年、v6 が 2023〜2024 年。想定オッズの較正（place_mid）は 2022〜2025 年の OOF 全体で当てはめた値で、厳密な walk-forward ではない（係数は a≈0・b≈1 で影響は小さい）。</p>")
    w("<details><summary>2 段目の重み（年別）と、ベースの重要な特徴量（2025 年の 3 着内モデル、gain）</summary><div class='wrap'><table><thead><tr><th>年</th><th>単勝: ベース / 想定オッズ</th><th>3 着内: ベース / 想定複勝 / 切片</th></tr></thead><tbody>")
    for y, cf in o["coefs"]["B"].items():
        w(f"<tr><td>{y}</td><td class='num'>{cf['win']['lpb']:.2f} / {cf['win']['lsh']:.2f}</td><td class='num'>{cf['top3']['lp3b']:.2f} / {cf['top3']['place_raw']:.2f} / {cf['top3']['切片']:.2f}</td></tr>")
    w("</tbody></table></div><div class='wrap'><table><thead><tr><th>特徴量</th><th>gain</th></tr></thead><tbody>")
    for k, v in list(o["importance_top"].items())[:15]:
        w(f"<tr><td class='id'>{k}</td><td class='num'>{v:,.0f}</td></tr>")
    w("</tbody></table></div></details>")
    _est_section(w)
    _diag_section(w)
    w("<p class='muted'>コード: <code>python -m src.scripts.data.folklore_base fit [tl]</code> → <code>eval [tl][+est]</code>、<code>python -m src.scripts.data.folklore_prerace build</code>（flags・odds_pred の OOF が必要）。</p>")


def _diag_section(w) -> None:
    """13.5: 『期待値 1.2 以上の買いが払戻率並みの約 80%』になる理由の切り分け。"""
    f = OUT / "diagnose.json"
    if not f.exists():
        return
    g = json.load(open(f))
    w("<h3>13.5 なぜ『期待値 1.2 以上の買い』の回収率が約 80%（払戻率並み）になるのか — 切り分け</h3>")
    w("<p>期待値 1.2 以上の買いの回収率が約 80%（単勝の払戻率並み。控除率は約 20%）になった。払戻率そのものは力の有無の基準ではなく、<b>同じオッズ帯の馬を無作為に買った回収率との差</b>で、選び方に力があるかを見る。力があれば、無作為に買うより高いはず。次の 3 点で調べた。</p><ol>"
      "<li><b>選び方の力</b> = 選んだ馬の回収率 − <b>同じオッズ帯の馬を無作為に買った回収率</b>（オッズを 25 分位に分けて揃える）。0 なら無作為と同じ。"
      f"（『無作為に買うと払戻率通り』は、オッズ帯で揃えないと正確でない。全頭を無作為に買うと単勝は {g['est'].get('random_all_roi', float('nan')) * 100:.0f}%（est 版の年）、穴馬ほど低い。）</li>"
      "<li><b>較正</b> = 予測の勝つ確率と実際の勝率、確定オッズの確率との比較。</li>"
      "<li><b>モデルと確定市場の食い違い</b>が大きい馬で、実際の勝率は予測側・市場側のどちらに近いか。</li></ol>")
    w("<div class='wrap'><table><thead><tr><th>前日の市場</th><th>判断のしかた</th><th>買い（点）</th><th>回収率</th><th>同じオッズ帯を無作為に買った回収率</th><th>選び方の力（差 ±SE）</th><th>予測の勝率 / 確定市場 / 実際</th><th>オッズの中央値</th><th>20 倍以上の割合</th></tr></thead><tbody>")
    for mk, nm in (("share", "odds_pred の想定オッズ（2023〜2025）"), ("est", "est_pop から作った想定（2022〜2025）")):
        for k in ("前日の想定オッズで判断", "確定オッズで判断"):
            r = g[mk][k]
            w(f"<tr><td>{nm}</td><td>{k}</td><td class='num'>{r['n']:,}</td><td class='num'>{r['roi'] * 100:.1f}%</td><td class='num'>{r['roi_odds_matched'] * 100:.1f}%</td>"
              f"<td class='num'>{r['skill'] * 100:+.1f}pt ±{r['skill_se'] * 100:.1f}</td><td class='num'>{r['pred_win_mean'] * 100:.1f}% / {r['market_final_q'] * 100:.1f}% / {r['actual_win'] * 100:.1f}%</td>"
              f"<td class='num'>{r['odds_median']:.0f} 倍</td><td class='num'>{r['odds_ge20'] * 100:.0f}%</td></tr>")
    w("</tbody></table></div>")
    e = g["est"]["gap_bins"]
    w("<p>モデルの確率と確定オッズの確率の食い違い（対数の比）を 5 分位に分けた（est 版、2022〜2025 年）。0 = モデルが市場より低く見た馬、4 = 高く見た馬。</p>"
      "<div class='wrap'><table><thead><tr><th>分位</th><th>モデルの確率</th><th>確定市場の確率</th><th>実際の勝率</th><th>単勝回収率</th></tr></thead><tbody>")
    for k, r in e.items():
        w(f"<tr><td>{k}</td><td class='num'>{r['pred'] * 100:.2f}%</td><td class='num'>{r['market'] * 100:.2f}%</td><td class='num'>{r['actual'] * 100:.2f}%</td><td class='num'>{r['roi'] * 100:.1f}%</td></tr>")
    w("</tbody></table></div>")
    def sig(sk, se):
        z = sk / se if se > 0 else 0.0
        return "無作為と区別できない" if abs(z) < 2 else ("無作為より有意に良い" if z > 0 else "無作為より有意に悪い")

    def judge(mk):
        r = g[mk]["前日の想定オッズで判断"]
        return f"{r['roi'] * 100:.1f}%（同じオッズ帯の無作為買い {r['roi_odds_matched'] * 100:.1f}%、差 {r['skill'] * 100:+.1f}pt ±{r['skill_se'] * 100:.1f} → {sig(r['skill'], r['skill_se'])}）"

    t = g["est"]["前日の想定オッズで判断"]
    top = e[max(e, key=int)]
    near = "市場" if abs(top["actual"] - top["market"]) < abs(top["actual"] - top["pred"]) else "モデル"
    pf = g["est"]["pre_vs_final"]["corr_log"], g["share"]["pre_vs_final"]["corr_log"]
    aucs = [json.load(open(OUT / f)) for f in ("base_est.json",) if (OUT / f).exists()]  # 2022〜2025 年。base.json（2023〜2025）と合わせると二重に数えるので使わない
    auc_b = np.mean([v["base"] for o_ in aucs for v in o_["auc"].values()]) if aucs else float("nan")
    auc_f = np.mean([v["final_odds"] for o_ in aucs for v in o_["auc"].values()]) if aucs else float("nan")
    w("<div class='note'><b>結論.</b> "
      f"(1) <b>選んだ馬の大半が穴馬</b>（est 版: オッズの中央値 {t['odds_median']:.0f} 倍、20 倍以上が {t['odds_ge20'] * 100:.0f}%）。穴馬は無作為に買っても約 {t['roi_odds_matched'] * 100:.0f}% しか返らない（人気薄ほど低い）。"
      f"前日の想定オッズで期待値 1.2 以上を買った回収率は、est 版 {judge('est')}、odds_pred 版 {judge('share')}。"
      f"「約 80%」は払戻率そのものではなく、穴馬を無作為に買った水準に近く、選び方の力は est 版で {sig(g['est']['前日の想定オッズで判断']['skill'], g['est']['前日の想定オッズで判断']['skill_se'])}、"
      f"odds_pred 版で {sig(g['share']['前日の想定オッズで判断']['skill'], g['share']['前日の想定オッズで判断']['skill_se'])}"
      + ("（どちらも力があるとは示せない）。" if sig(g['est']['前日の想定オッズで判断']['skill'], g['est']['前日の想定オッズで判断']['skill_se']) == sig(g['share']['前日の想定オッズで判断']['skill'], g['share']['前日の想定オッズで判断']['skill_se']) == "無作為と区別できない" else "。")
      +       f"(2) <b>予測が確定市場より弱い</b>。モデルと市場が最も食い違う馬（上の分位 {max(e, key=int)}）で、モデルは {top['pred'] * 100:.1f}%・市場は {top['market'] * 100:.1f}% と見たが、実際の勝率は {top['actual'] * 100:.1f}% で<b>{near}</b>の側に近い。"
      + ("食い違いは情報ではなくモデルの誤差で、期待値の高く見える馬ほど見込み違いになる（勝者の呪い）。" if near == "市場" else "食い違いの側に実際の勝率が寄っており、モデルに市場に無い情報がある可能性がある。")
      + f"自作ベースの 3 着内 AUC は平均 {auc_b:.3f}、確定オッズは {auc_f:.3f}（2022〜2025 年・est 版の行）。"
      + 
      f"(3) <b>前日の市場の誤差が上乗せされる</b>。前日の想定オッズは確定オッズの予測にすぎず（確定の確率との相関は est 版 {pf[0]:.2f}・odds_pred 版 {pf[1]:.2f}）、『想定オッズより有利』は『確定オッズより有利』と別。"
      "想定オッズが外れた馬を拾うと、市場との食い違いではなく想定オッズの誤差を買うことになる。"
      "(4) <b>券種</b>: 単勝は穴馬に偏って分散が大きく、期待値フィルタの誤差が大きい。複勝・ワイドなどでの同じ検証は未実施。"
      "<br><b>次にやること</b>: 予測を『確定市場に対する残差』として作り（市場を土台に置く第 10 節の形）、評価も『選び方の力（選んだ馬を除いたオッズ帯で揃えた回収率の差）』で行う。</div>")


def _est_section(w) -> None:
    """13.3: 前日の市場を SmartRC の想定人気（est_pop）で作る。13.4: 追い切りの特徴量をベースに加えた上積み。"""
    pr = OUT / "prerace.json"
    if not pr.exists():
        return
    p = json.load(open(pr))
    files = {k: json.load(open(OUT / f"base{sfx}.json")) for k, sfx in (("est", "_est"), ("tl_est", "_tl_est"), ("tl", "_tl")) if (OUT / f"base{sfx}.json").exists()}

    cell = _roi_cell

    w("<h3>13.3 前日の市場を SmartRC の想定人気（est_pop）で作る（2022〜2025 年の 4 年）</h3>")
    ev_ = p["est_pop_vs_final"]
    w(f"<p>SmartRC の runners にある <b>est_pop（想定人気）</b>は 2020〜2025 年の全行にあり、確定人気との順位相関は {ev_['spearman_rank']:.2f}、完全一致は {ev_['exact_share'] * 100:.0f}%（確定人気そのものではない）。"
      f"ただし<b>想定人気の頭数は確定の出走頭数と {ev_['n_est_eq_field'] * 100:.0f}% 一致</b>しており、順位は取消のあとの出走馬でつけ直されている（出走確定後の値）。取得時刻の記録は無く、前日の値とは断定できない。"
      "この節では<b>『前日の想定の近似』</b>として使う（SmartRC の過去のオッズ列 odds_tan は取得時刻が発走後で、前日のものではない）。"
      "順位だけでオッズの値は無いので、<b>est_pop の順位・順位/頭数・頭数から確定の単勝の確率と複勝オッズを予測するモデル</b>（評価年より前の年だけで学習）で想定オッズを作った。"
      "第 13.1 節の想定オッズ（odds_pred、2022 年から）の代わりに使うと、評価が 2022〜2025 年の 4 年になる。</p>")
    w("<div class='wrap'><table><thead><tr><th>年</th><th>est_pop の 3 着内 AUC</th><th>確定オッズの 3 着内 AUC</th><th>勝ちの対数損失: 確定オッズ</th><th>想定（est のみ）</th><th>想定（odds_pred）</th><th>確定の確率との相関（est のみ / odds_pred）</th></tr></thead><tbody>")
    for y, r in p["years"].items():
        sl = "—" if "win_ll_share" not in r else f"{r['win_ll_share']:.3f}"
        sc = "—" if "corr_lq_share" not in r else f"{r['corr_lq_share']:.3f}"
        w(f"<tr><td>{y}</td><td class='num'>{r['est_pop_top3_auc']:.3f}</td><td class='num'>{r['final_odds_top3_auc']:.3f}</td><td class='num'>{r['win_ll_final']:.3f}</td><td class='num'>{r['win_ll_E']:.3f}</td>"
          f"<td class='num'>{sl}</td><td class='num'>{r['corr_lq_E']:.3f} / {sc}</td></tr>")
    w("</tbody></table></div>")
    rc = p["resid_check"]
    sign = "正" if rc["coef_est_rel_given_final_odds"] > 0 else "負"
    w(f"<p>est_pop が結果由来の情報を含んでいないかの確認: 確定オッズの確率（対数、3 次まで）を条件にしたロジスティック回帰で、est_pop の順位/頭数が 3 着内を追加で説明するか（レースで束ねた標準誤差）。"
      f"係数 {rc['coef_est_rel_given_final_odds']:+.3f}（z={rc['z']:+.1f}、p={rc['p']:.3g}）で、符号は<b>{sign}</b>。"
      "est_pop が確定後の情報（着順など）から作られているなら、想定人気が上位の馬ほど 3 着内に来る＝係数は<b>負</b>になるはず。"
      + ("観測は正で、結果由来の混入とは逆の向き（同じ確定オッズなら、想定人気が低かった馬の方が走る＝発売の早い時期の市場に対して後から資金が集まった馬が走る、という向きと合う）。"
         if rc["coef_est_rel_given_final_odds"] > 0 else "観測は負で、結果由来の混入が疑われる。この節の結果は参考にとどめる。")
      + " ただし est_pop が確定より後に別の方法で作られた値である可能性までは排除できないので、符号は混入が無いことの証明ではない。</p>")
    dj = OUT / "drift.json"
    if dj.exists():
        dr = json.load(open(dj))
        nbin = [int(r["n"]) for r in dr["drift_bins"].values()]
        w("<h4>人気の動き（想定人気 → 確定人気）は、確定オッズに無い情報を持つか</h4><p>各馬について、想定人気の順位と確定人気の差を頭数で割った値（<b>d_rel</b> = (確定人気 − 想定人気) / 頭数。マイナスは確定までに人気が上がった馬）を、固定の幅"
          f"（{dr['drift_bin_edges'][1]}・{dr['drift_bin_edges'][2]}・{dr['drift_bin_edges'][3]}・{dr['drift_bin_edges'][4]}）で 5 つに分け、確定オッズで買ったときの成績を見た（2021〜2025 年、各区分 {min(nbin):,}〜{max(nbin):,} 頭。"
          "d_rel は約 3 分の 1 が 0 で、分位に分けると同値が年順に割られるので、幅で分けた）。</p><div class='wrap'><table><thead><tr><th>人気の動き</th><th>頭数</th><th>3 着内率</th><th>確定オッズの単勝確率（平均）</th><th>単勝回収率</th><th>複勝回収率</th></tr></thead><tbody>")
        for k, r in dr["drift_bins"].items():
            w(f"<tr><td>{k}</td><td class='num'>{int(r['n']):,}</td><td class='num'>{r['top3'] * 100:.1f}%</td><td class='num'>{r['p_final'] * 100:.1f}%</td><td class='num'>{r['roi_win'] * 100:.1f}%</td><td class='num'>{r['roi_plc'] * 100:.1f}%</td></tr>")
        w("</tbody></table></div>")
        m0, m1 = dr["ev"]["M0"]["全体"], dr["ev"]["M1"]["全体"]
        c = dr["cmp"]
        cp = dr["cmp_pop"]

        def f_(r):
            return "30 点未満" if not r else f"{r['roi'] * 100:.0f}%±{r['se'] * 100:.0f}（{r['n']:,} 点）"

        w("<p>確定オッズの市場（単勝は 1 着の条件付きロジット、複勝は 3 着内のロジスティック回帰。確定オッズは単勝・複勝とも 3 次まで）に、まず確定人気の順位/頭数（M0p）、次に想定人気（M1）を足し、"
          "2021 年以降の年だけで学習して 2023〜2025 年で比べた。<b>M1 − M0p が『想定人気（人気の動き）が、確定人気の順位に無い情報を持つか』</b>。"
          f"対数損失の差（負 = 良くなる）は 確定人気を足した効果（M0p − M0）が 1 着 {cp['win']['d']:+.4f}（z={cp['win']['d'] / cp['win']['se']:+.1f}）・3 着内 {cp['top3']['d']:+.4f}（z={cp['top3']['d'] / cp['top3']['se']:+.1f}）、"
          f"想定人気を足した効果（M1 − M0p）が 1 着 {c['win']['d']:+.4f}（z={c['win']['d'] / c['win']['se']:+.1f}）・3 着内 {c['top3']['d']:+.4f}（z={c['top3']['d'] / c['top3']['se']:+.1f}）。</p>")
        bins = list(dr["drift_bins"].items())
        best, worst = max(bins, key=lambda kv: kv[1]["roi_win"]), min(bins, key=lambda kv: kv[1]["roi_win"])
        m1v = m1["win"].get("1.00")
        zc = c["top3"]["d"] / c["top3"]["se"]
        cw = [v["win"].get("est_rel") for v in dr["years"]["M1"].values()]
        c3 = [v["top3"].get("est_rel") for v in dr["years"]["M1"].values()]
        same = lambda xs: all(x > 0 for x in xs) or all(x < 0 for x in xs)  # noqa: E731
        if same(cw) and same(c3):
            sign_txt = (f"想定人気（est_rel）の係数は 3 年とも同じ向き（1 着 {'・'.join(f'{x:+.2f}' for x in cw)}、3 着内 {'・'.join(f'{x:+.2f}' for x in c3)}）で、"
                        + ("正＝同じ確定人気なら、想定人気が低かった（確定までに人気が上がった）馬ほど走る向き。" if cw[0] > 0 else "負＝同じ確定人気なら、想定人気が高かった（確定までに人気が下がった）馬ほど走る向き。"))
        else:
            sign_txt = f"想定人気の係数は年で符号が変わり（1 着 {'・'.join(f'{x:+.2f}' for x in cw)}、3 着内 {'・'.join(f'{x:+.2f}' for x in c3)}）、向きは断定できない。"
        w(f"<div class='note'><b>読み.</b> 確定人気の順位/頭数を入れたうえで想定人気を足すと、3 着内の予測は{'統計的に良くなる' if zc <= -2 else '良くならない'}（z={zc:+.1f}）。"
          + sign_txt +
          f"区分では、単勝回収率が最も高いのは『{best[0]}』の {best[1]['roi_win'] * 100:.0f}%、最も低いのは『{worst[0]}』の {worst[1]['roi_win'] * 100:.0f}%。"
          "ただしこの差には、穴馬ほど回収率が低い傾向（区分ごとの確定の単勝確率が違う）が混ざっている。"
          f"期待値（確率 × 確定オッズ）で買うと、単勝 1.0 以上は M0 {f_(m0['win'].get('1.00'))}・M1 {f_(m1v)}。100% を超える点があっても誤差の範囲で、払戻率（約 80%）を明確に越える形にはなっていない。"
          "確定オッズを見て買う形なので、前日判断ではなく『締切直前の判断』の話。</div>")
    if "est" in files:
        _dg = json.load(open(OUT / "diagnose.json")) if (OUT / "diagnose.json").exists() else None
        _r = _dg["est"]["前日の想定オッズで判断"] if _dg else None
        w("<p class='muted'>下の回収率は約 80%（払戻率並み）。選んだ馬の大半が穴馬で、同じオッズ帯を無作為に買った回収率"
          + (f"（{_r['roi_odds_matched'] * 100:.0f}%）との差は {_r['skill'] * 100:+.1f}pt（±{_r['skill_se'] * 100:.1f}）" if _r else "との差")
          + "。切り分けは 13.5 節。</p>")

        def row(lab, o, v, kind):
            return f"<tr><td>{lab}</td>" + "".join(cell(o['variants'][v][kind].get(y, {}).get('1.20')) for y in ("2022", "2023", "2024", "2025", "全体")) + "</tr>"

        w("<h4>期待値 1.2 以上を買ったときの回収率（前日の市場 = est_pop から作った想定、自作ベース）</h4><div class='wrap'><table><thead><tr><th>買い方</th><th>2022</th><th>2023</th><th>2024</th><th>2025</th><th>全体</th></tr></thead><tbody>")
        e = files["est"]
        w(row("単勝 — 前日の想定で判断", e, "B", "win") + row("複勝 — 前日の想定で判断", e, "B", "plc") + row("単勝 — 確定オッズで判断（対照）", e, "B_final", "win") + row("複勝 — 確定オッズで判断（対照）", e, "B_final", "plc"))
        w("</tbody></table></div>")
    if "tl" in files or "tl_est" in files:
        tlc = json.load(open(OUT / "base_fit_tl.json")).get("tl_coverage", {}) if (OUT / "base_fit_tl.json").exists() else {}
        cov_txt = "・".join(f"{y} 年 {v * 100:.0f}%" for y, v in tlc.items())
        w("<h3>13.4 追い切りの特徴量をベースに加えた上積み</h3><p>ベースの特徴量に、training_laps 由来の追い切り（最終の強度・終い 1F・加速・場所、1 週前、直近 14 日の本数・強め以上の本数、坂路の自己ベスト、初めの追い切りの日）を加えた版。"
          f"追い切りが載る馬の割合（年別）: {cov_txt}。載らない馬は欠損のまま。</p>")
        w("<p class='muted'>欠損の並びは『載り方』（後日に馬のページをまとめて取得した結果。その後の出走歴と関係する）と同じなので、追い切りの載り方の情報も特徴量に入る。載るかどうかの列は外したが、欠損そのものは残るため、上積みは上限として読む。</p>")
        fb = json.load(open(OUT / "base_fit.json"))["fits"]
        ft = json.load(open(OUT / "base_fit_tl.json"))["fits"] if (OUT / "base_fit_tl.json").exists() else {}
        if ft:
            w("<div class='wrap'><table><thead><tr><th>評価年</th><th>3 着内 AUC（追い切りなし）</th><th>（あり）</th><th>3 着内の対数損失（なし）</th><th>（あり）</th><th>1 着の AUC（なし）</th><th>（あり）</th></tr></thead><tbody>")
            for y in ft:
                a0, a1 = fb[y], ft[y]
                w(f"<tr><td>{y}</td><td class='num'>{a0['top3']['auc']:.4f}</td><td class='num'>{a1['top3']['auc']:.4f}</td><td class='num'>{a0['top3']['logloss']:.4f}</td><td class='num'>{a1['top3']['logloss']:.4f}</td>"
                  f"<td class='num'>{a0['win']['auc']:.4f}</td><td class='num'>{a1['win']['auc']:.4f}</td></tr>")
            w("</tbody></table></div>")
        if "tl_est" in files and "est" in files:
            w("<div class='wrap'><table><thead><tr><th>期待値 1.2 以上（単勝・前日の想定）</th><th>2022</th><th>2023</th><th>2024</th><th>2025</th><th>全体</th></tr></thead><tbody>")
            for lab, o in (("追い切りなし", files["est"]), ("追い切りあり", files["tl_est"])):
                w(f"<tr><td>{lab}</td>" + "".join(cell(o["variants"]["B"]["win"].get(y, {}).get("1.20")) for y in ("2022", "2023", "2024", "2025", "全体")) + "</tr>")
            w("</tbody></table></div>")
            ya = [y for y in ft if y != "2020"]
            d_auc = np.mean([ft[y]["top3"]["auc"] - fb[y]["top3"]["auc"] for y in ya])
            d_ll = np.mean([ft[y]["top3"]["logloss"] - fb[y]["top3"]["logloss"] for y in ya])
            r0, r1 = files["est"]["variants"]["B"]["win"]["全体"].get("1.20"), files["tl_est"]["variants"]["B"]["win"]["全体"].get("1.20")
            auc_fin = np.mean([v["final_odds"] for v in files["est"]["auc"].values()])
            auc_none = np.mean([v["base"] for v in files["est"]["auc"].values()])  # 2022〜2025 年・est 対象の行（確定オッズの AUC と同じ行）
            auc_tl = np.mean([v["base"] for v in files["tl_est"]["auc"].values()])
            ct = json.load(open(OUT / "compare_tl.json")) if (OUT / "compare_tl.json").exists() else None
            if ct:
                diff_txt = (f"差 {ct['diff'] * 100:+.1f}pt。同じ候補から選ぶ買い（選ぶ馬の重なり {ct['overlap_share'] * 100:.0f}%）なので、レースで対応を取った復元抽出で誤差を出すと ±{ct['se'] * 100:.1f}pt"
                            f"（z={ct['diff'] / ct['se']:+.1f}。{'誤差の範囲' if abs(ct['diff']) < 2 * ct['se'] else ('誤差を超えるが z が 3 未満で境界に近い' if abs(ct['diff']) < 3 * ct['se'] else '誤差を超える')}。評価は 4 年で、買う馬は穴馬に偏る）")
            else:
                diff_txt = f"差 {(r1['roi'] - r0['roi']) * 100:+.1f}pt（対応のある誤差は未計算）"
            w(f"<div class='note'><b>結論.</b> 追い切りを加えると、3 着内の AUC は 2021〜2025 年の平均で {d_auc:+.4f}、対数損失は {d_ll:+.4f}。予測はわずかに良くなる（全頭の平均）。"
              f"単勝の期待値 1.2 以上の回収率は {r0['roi'] * 100:.1f}% → {r1['roi'] * 100:.1f}%（{diff_txt}）。"
              f"追い切りありでも {r1['roi'] * 100:.1f}% で、払戻率（約 80%）を明確には越えない（100% に届かない）。"
              f"確定オッズの市場に対する弱さ（2022〜2025 年・同じ行で 3 着内 AUC は確定オッズ {auc_fin:.3f}、自作ベースは追い切りなし {auc_none:.3f}・あり {auc_tl:.3f}）は、追い切りを足しても埋まらない。上積みは上限として読む。</div>")


def compare_tl() -> None:
    """est 市場で、追い切りなし/ありのベースが選ぶ『期待値 1.2 以上』の単勝の回収率の差を、レースで対応を取ったブートストラップで出す
    （同じ候補から選ぶので、選ぶ馬が重なる。SE を独立と見なすと差の誤差を大きく見積もる）。"""
    years, first = (2022, 2023, 2024, 2025), 2021
    d0 = _prep_cols(_merge_inputs("base_oof.parquet", "est"), "est")
    d1 = _prep_cols(_merge_inputs("base_oof_tl.parquet", "est"), "est")
    key = ["race_id", "horse_id"]
    k0 = d0[key].astype(str).agg("|".join, axis=1)
    k1 = d1[key].astype(str).agg("|".join, axis=1)
    common = set(k0) & set(k1)
    d0, d1 = d0[k0.isin(common)].reset_index(drop=True), d1[k1.isin(common)].reset_index(drop=True)
    assert (d0[key].to_numpy() == d1[key].to_numpy()).all()
    sel = {}
    for name, d in (("none", d0), ("tl", d1)):
        pr = _two_stage(d, ["lpb", "lsh"], ["lp3b", "place_raw"], years=years, first=first)
        x = d.loc[pr.index]
        sel[name] = (pr["pw"].to_numpy() * x["win_hat"].to_numpy() >= 1.2, x["win"].to_numpy() / 100, x["race_id"].to_numpy())
    assert (sel["none"][2] == sel["tl"][2]).all()  # 2 つの選び方が同じ行の並びであること
    rid = pd.factorize(sel["none"][2])[0]
    nr = int(rid.max()) + 1
    m0, r0, _ = sel["none"]
    m1, r1, _ = sel["tl"]
    S0, N0 = np.bincount(rid, r0 * m0, nr), np.bincount(rid, m0.astype(float), nr)
    S1, N1 = np.bincount(rid, r1 * m1, nr), np.bincount(rid, m1.astype(float), nr)
    diff = S1.sum() / N1.sum() - S0.sum() / N0.sum()
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(500):
        w_ = np.bincount(rng.integers(0, nr, nr), minlength=nr).astype(float)
        bs.append((w_ @ S1) / (w_ @ N1) - (w_ @ S0) / (w_ @ N0))
    out = {"n_none": int(m0.sum()), "n_tl": int(m1.sum()), "roi_none": float(S0.sum() / N0.sum()), "roi_tl": float(S1.sum() / N1.sum()), "diff": float(diff), "se": float(np.std(bs, ddof=1)),
           "overlap_share": float((m0 & m1).sum() / max(min(m0.sum(), m1.sum()), 1)), "rows": int(len(m0))}
    json.dump(out, open(OUT / "compare_tl.json", "w"), ensure_ascii=False, indent=1)
    print(out)


def diagnose() -> None:
    """『期待値 1.2 以上』で選んだ馬の回収率が払戻率並み（約 80%）になる理由の切り分け。
    - 選んだ馬の回収率 − 同じオッズ帯の馬を無作為に買った回収率（オッズ帯で揃えた基準）= 選び方の力（0 なら無作為と同じ）。
    - 予測の確率と実際の勝率（較正）、選んだ馬のオッズの分布。
    - 市場の想定（前日の想定オッズ）と確定オッズの食い違い（想定オッズの誤差）。"""
    from src.scripts.data.folklore_model import ev_curve  # noqa: F401

    res = {}
    for market, years, first in (("share", YEARS_EVAL, 2022), ("est", (2022, 2023, 2024, 2025), 2021)):
        d = _prep_cols(_merge_inputs("base_oof.parquet", market), market)
        inv = 1.0 / d["odds"]
        d["q_final"] = inv / inv.groupby(d["race_id"]).transform("sum")
        pr = _two_stage(d, ["lpb", "lsh"], ["lp3b", "place_raw"], years=years, first=first)
        x = d.loc[pr.index].copy()
        x["pw"] = pr["pw"].to_numpy()
        x["p_pre"] = np.exp(x["lsh"])
        x["hit"] = (x["fin"] == 1).astype(float)
        x["roi"] = x["win"] / 100
        # オッズ帯（対数の 25 分位）ごとの無作為買いの回収率。選んだ馬は基準から除く（選んだ馬が多いほど力が 0 に寄るのを避ける）
        x["ob"] = pd.qcut(np.log(x["odds"]), 25, labels=False, duplicates="drop")
        nb = int(x["ob"].max()) + 1
        rid = pd.factorize(x["race_id"])[0]
        nr = int(rid.max()) + 1
        cell = rid * nb + x["ob"].to_numpy().astype(int)
        roi_all = x["roi"].to_numpy()

        def mats(mask):
            S = np.bincount(cell[mask], roi_all[mask], nr * nb).reshape(nr, nb)
            N = np.bincount(cell[mask], None, nr * nb).reshape(nr, nb)
            return S, N

        S_all, N_all = mats(np.ones(len(x), bool))

        def skill_of(S_sel, N_sel, w):
            sa, na, ss, ns = w @ S_all, w @ N_all, w @ S_sel, w @ N_sel
            rest = na - ns
            ro = np.where(rest > 0, (sa - ss) / np.maximum(rest, 1), sa / np.maximum(na, 1))  # 選んだ馬を除いた帯ごとの回収率
            sel_roi = ss.sum() / ns.sum()
            base = float((ns * ro).sum() / ns.sum())
            return float(sel_roi), base

        rng = np.random.default_rng(0)
        out = {"random_all_roi": float(x["roi"].mean())}
        for name, ev, th in (("前日の想定オッズで判断", x["pw"] * x["win_hat"], 1.2), ("確定オッズで判断", x["pw"] * x["odds"], 1.2)):
            m = (ev >= th).to_numpy()
            z = x[m]
            S_sel, N_sel = mats(m)
            sel_roi, base = skill_of(S_sel, N_sel, np.ones(nr))
            # 標準誤差: レースを復元抽出した 300 回のやり直しで、選んだ馬の回収率と基準（帯ごとの回収率）の両方の誤差を含めて出す
            bs = []
            for _ in range(300):
                w = np.bincount(rng.integers(0, nr, nr), minlength=nr).astype(float)
                if (w @ N_sel).sum() > 0:
                    a_, b_ = skill_of(S_sel, N_sel, w)
                    bs.append(a_ - b_)
            se = float(np.std(bs, ddof=1))
            out[name] = {"n": int(m.sum()), "roi": sel_roi, "roi_odds_matched": base, "skill": sel_roi - base, "skill_se": se,
                         "pred_win_mean": float(z["pw"].mean()), "actual_win": float(z["hit"].mean()), "market_final_q": float(z["q_final"].mean()),
                         "odds_median": float(z["odds"].median()), "odds_ge20": float((z["odds"] >= 20).mean()), "odds_ge50": float((z["odds"] >= 50).mean()),
                         "win_hat_over_odds_median": float((z["win_hat"] / z["odds"]).median())}
        # 較正: 予測の勝つ確率の 10 分位ごとの、予測の平均と実際の勝率
        x["pb"] = pd.qcut(x["pw"].rank(method="first"), 10, labels=False)
        cal = x.groupby("pb").agg(pred=("pw", "mean"), actual=("hit", "mean"), market=("q_final", "mean"), n=("hit", "size"))
        out["calibration"] = {str(int(k)): {kk: float(vv) for kk, vv in r.items()} for k, r in cal.iterrows()}
        # 想定オッズの誤差: 想定の勝つ確率と確定の確率の食い違い（対数）
        e = np.log(x["p_pre"] / x["q_final"])
        out["pre_vs_final"] = {"log_ratio_mean": float(e.mean()), "log_ratio_sd": float(e.std()), "corr_log": float(np.corrcoef(np.log(x["p_pre"]), np.log(x["q_final"]))[0, 1])}
        # モデルの確率と確定オッズの確率の食い違い（対数）が大きい馬ほど、確定の市場の方が当たっているか
        x["gap"] = np.log(x["pw"] / x["q_final"])
        x["gb"] = pd.qcut(x["gap"].rank(method="first"), 5, labels=False)
        gb = x.groupby("gb").agg(pred=("pw", "mean"), market=("q_final", "mean"), actual=("hit", "mean"), roi=("roi", "mean"), n=("hit", "size"))
        out["gap_bins"] = {str(int(k)): {kk: float(vv) for kk, vv in r.items()} for k, r in gb.iterrows()}
        res[market] = out
        print(market)
        for k in ("前日の想定オッズで判断", "確定オッズで判断"):
            print(" ", k, json.dumps(out[k], ensure_ascii=False))
        print("  pre_vs_final", out["pre_vs_final"])
        print("  gap bins", {k: (round(v["pred"], 4), round(v["market"], 4), round(v["actual"], 4), round(v["roi"], 3)) for k, v in out["gap_bins"].items()})
    json.dump(res, open(OUT / "diagnose.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    arg = sys.argv[2] if len(sys.argv) > 2 else ""
    {"fit": lambda: fit(arg), "eval": lambda: evaluate(arg), "diagnose": diagnose, "compare_tl": compare_tl}[cmd]()

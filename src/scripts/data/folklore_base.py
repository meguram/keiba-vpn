"""v6 に頼らないベース（実力）モデルを年ごとに抜く形（walk-forward）で自作し、『前日の想定オッズで期待値 1.2 以上を買う』形を 2023〜2025 年で評価する
（通説の検証の章の第 13 節）。

    python -m src.scripts.data.folklore_base fit    # → <FOLKLORE_OUT>/base_oof.parquet・base_fit.json（2020〜2025 年の全出走に p_win・p_top3）
    python -m src.scripts.data.folklore_base eval   # → <FOLKLORE_OUT>/base.json（章は folklore_verify report で作る）

設計（計画: 第 12 節の『前日の想定オッズで期待値 1.2 以上の単勝 108.6%±7.3』の評価が 2 年分で、v6 の OOF が 2022〜2024 年しか無かったため）:
- ベース: LightGBM（1 着・3 着内の二値）。評価年 Y の学習は 2016〜Y−1 年だけ。反復回数は Y−1 年を検証にした早期停止で決め、全期間で学び直す。
  特徴量は master の as-of 列から、当日のオッズ・人気・レース結果由来の列を除いたもの（EXCLUDE と先頭の一致）。追い切りの詳細（2016〜2019 年だけ）も除く。
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
}
EXCLUDE_PREFIX = ("ok_", "tc_pos", "tc_neg", "tc_next", "tc_sp", "tc_has", "p_ok_", "t_first", "t_last", "tp_", "e_", "lt_", "tr_")
CATS = ["vc", "surf", "sex", "track", "dband", "io", "shape", "weight_rule", "tozai", "line", "grade", "cl_cmb", "dir2", "direction"]


def feature_columns(d: pd.DataFrame) -> list[str]:
    cols = []
    for c in d.columns:
        if c in EXCLUDE or c.startswith(EXCLUDE_PREFIX) or c in ("race_id", "horse_id", "jockey_id", "trainer_id"):
            continue
        if c in CATS or pd.api.types.is_numeric_dtype(d[c]) or d[c].dtype == bool:
            cols.append(c)
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


def fit() -> None:
    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score

    d = pd.read_pickle(OUT / "_prep.pkl")
    d = d.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    cols = feature_columns(d)
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
    pd.concat(out).to_parquet(OUT / "base_oof.parquet", index=False)
    info["importance_top"] = {k: float(v) for k, v in imp.head(25).items()}
    json.dump(info, open(OUT / "base_fit.json", "w"), ensure_ascii=False, indent=1)
    print(imp.head(15).to_string())


# ---------------------------------------------------------------- 評価
THRESHOLDS = (1.0, 1.1, 1.2, 1.5)
MAIN_TH = 1.2  # 事前に固定した買い方（記事の閾値。評価期間を見て選ばない）


def _merge_inputs() -> pd.DataFrame:
    wp_raw = ROOT / "data/local/modeling/odds_pred/wp_raw_oof.parquet"
    wp = ROOT / "data/local/modeling/odds_pred/wp_oof_pred.parquet"
    d = pd.read_parquet(OUT / "flags.parquet", columns=["race_id", "horse_id", "horse_number", "year", "fin", "odds", "pop", "field", "win", "plc", "place_odds_min"])
    d = d[(d["odds"] > 0) & (d["year"] >= 2022)].copy()
    for df_ in (d,):
        df_["race_id"] = df_["race_id"].astype(str)
        df_["horse_id"] = df_["horse_id"].astype(str)
    b = pd.read_parquet(OUT / "base_oof.parquet")
    b["race_id"], b["horse_id"] = b["race_id"].astype(str), b["horse_id"].astype(str)
    r = pd.read_parquet(wp_raw, columns=["race_id", "horse_id", "share_raw", "place_raw"])
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    v = pd.read_parquet(wp, columns=["race_id", "horse_id", "p_win", "p_top3"]).rename(columns={"p_win": "v6_win", "p_top3": "v6_top3"})
    v["race_id"], v["horse_id"] = v["race_id"].astype(str), v["horse_id"].astype(str)
    d = d.merge(b[["race_id", "horse_id", "p_win", "p_top3"]], on=["race_id", "horse_id"], how="inner")
    d = d.merge(r, on=["race_id", "horse_id"], how="inner").merge(v, on=["race_id", "horse_id"], how="left")
    d = d.dropna(subset=["p_win", "p_top3", "share_raw", "place_raw"]).sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    d = d[d.groupby("race_id")["p_win"].transform("size") == d.groupby("race_id")["field"].transform("first")].reset_index(drop=True)
    return d


def _prep_cols(d: pd.DataFrame) -> pd.DataFrame:
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
    d["plc_hat"] = np.exp(cfg["place_mid"]["a"] + cfg["place_mid"]["b"] * d["place_raw"])
    return d


def _two_stage(d: pd.DataFrame, wcols: list[str], pcols: list[str], years=YEARS_EVAL) -> pd.DataFrame:
    """年 Y ごとに、2022〜Y−1 年の（ベースの OOF と想定オッズの）重みを当てはめ、Y 年の単勝・複勝の確率を出す。"""
    from src.scripts.data.folklore_model import model_win, model_top3

    out = []
    coefs = {}
    for Y in years:
        tr, te = d[d["year"].between(2022, Y - 1)], d[d["year"] == Y]
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


def evaluate() -> None:
    from src.scripts.data.folklore_model import _cmp, _ll_bin

    d = _prep_cols(_merge_inputs())
    info = {"races": {str(y): int(d[d["year"] == y]["race_id"].nunique()) for y in sorted(d["year"].unique())}, "variants": {}, "ll": {}}
    variants = {"B": (["lpb", "lsh"], ["lp3b", "place_raw"]), "S": (["lsh"], ["place_raw"])}
    preds = {}
    for k, (wc, pc) in variants.items():
        preds[k] = _two_stage(d, wc, pc)
        info["variants"][k] = {"win": _stats(d, preds[k], "win"), "plc": _stats(d, preds[k], "plc")}
        info["variants"][k + "_final"] = {"win": _stats(d, preds[k], "win", True), "plc": _stats(d, preds[k], "plc", True)}
        info.setdefault("coefs", {})[k] = preds[k].attrs["coefs"]
        print(k, {y: v.get("1.20") and (v["1.20"]["n"], round(v["1.20"]["roi"], 3), round(v["1.20"]["se"], 3)) for y, v in info["variants"][k]["win"].items()}, flush=True)
    # v6 を使った版（2023〜2024 年）との比較
    d6 = d[d["v6_win"].notna() & d["v6_top3"].notna()].reset_index(drop=True)
    p6 = _two_stage(d6, ["lpv", "lsh"], ["lp3v", "place_raw"], years=(2023, 2024))
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
        pr = _two_stage(dp, ["lpb", "lsh"], ["lp3b", "place_raw"])
        st = _stats(d, pr, "win")["全体"].get("1.20")
        plc.append(st)
    info["placebo"] = [x and {"n": x["n"], "roi": x["roi"], "se": x["se"]} for x in plc]
    from sklearn.metrics import roc_auc_score

    auc = {}
    for y in (2023, 2024, 2025):
        x = d[d["year"] == y]
        y3 = (x["fin"] <= 3).to_numpy()
        fq = (1 / x["odds"]) / (1 / x["odds"]).groupby(x["race_id"]).transform("sum")
        auc[str(y)] = {"base": float(roc_auc_score(y3, x["p_top3"])), "pre_share": float(roc_auc_score(y3, x["lsh"])), "final_odds": float(roc_auc_score(y3, fq))}
        if x["v6_top3"].notna().all():
            auc[str(y)]["v6"] = float(roc_auc_score(y3, x["v6_top3"]))
    info["auc"] = auc
    fit = json.load(open(OUT / "base_fit.json"))
    info["fit"] = fit["fits"]
    info["n_features"] = fit["n_features"]
    info["importance_top"] = fit["importance_top"]
    json.dump(info, open(OUT / "base.json", "w"), ensure_ascii=False, indent=1)
    for k in ("B", "B_final", "S", "S_final", "V6", "V6_final"):
        for kind in ("win", "plc"):
            for lab, v in info["variants"][k][kind].items():
                print(k, kind, lab, " | ".join(f"{t}:{r['n']} {r['roi']:.3f}±{r['se']:.3f}" for t, r in v.items()))
    print("placebo", info["placebo"])
    print("ll", json.dumps(info["ll"], ensure_ascii=False))


# ---------------------------------------------------------------- 章（第 13 節）
def base_section(w) -> None:
    f = OUT / "base.json"
    if not f.exists():
        return
    o = json.load(open(f))
    V = o["variants"]

    def c(v, kind, year, th="1.20"):
        r = V[v][kind].get(year, {}).get(th)
        if not r:
            return "<td class='num'>30 点未満</td>"
        return f"<td class='num'>{r['roi'] * 100:.1f}% ±{r['se'] * 100:.1f}<br><span class='muted'>{r['n']:,} 点</span></td>"

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
      "当日にならないと分からない測定値（馬体重・馬場・クッション値・気象・パドック・同じ日の騎手の成績）、評価年全体・全年で決めた閾値や順位（j_top・t_rank・tozai など）、取消後の頭数、追い切りの詳細（2016〜2019 年だけ）は除いた。"
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
    b25 = V["B"]["win"]["2025"].get("1.20")
    t10, t15 = V["B"]["win"]["全体"].get("1.10"), V["B"]["win"]["全体"].get("1.50")
    w("<div class='note'><b>結論.</b> "
      f"年ごとに抜いた自作ベースでは、前日の想定オッズで期待値 1.2 以上の単勝は {b['roi'] * 100:.0f}%±{b['se'] * 100:.0f}（{b['n']:,} 点）、複勝は {bp['roi'] * 100:.0f}%±{bp['se'] * 100:.0f}（{bp['n']:,} 点）で、"
      f"3 年とも 100% を下回る（単勝の 2025 年は {b25['n']} 点と少なく {b25['roi'] * 100:.0f}%±{b25['se'] * 100:.0f}。閾値を変えると 1.10 で {t10['roi'] * 100:.0f}%±{t10['se'] * 100:.0f}・1.50 で {t15['roi'] * 100:.0f}%±{t15['se'] * 100:.0f}（{t15['n']} 点）と揺れ、単勝は誤差が大きい）。"
      f"期待値を確定オッズで測る対照でも単勝 {bf['roi'] * 100:.0f}%±{bf['se'] * 100:.0f} で、自作ベースは市場（AUC は自作 0.79・確定オッズ 0.82）より弱く、人気以上の情報を持つ証拠は出なかった。"
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
    w("<p class='muted'>コード: <code>python -m src.scripts.data.folklore_base fit</code> → <code>eval</code>（flags・odds_pred の OOF が必要）。</p>")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    {"fit": fit, "eval": evaluate}[cmd]()

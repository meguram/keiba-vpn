"""前日の市場を、SmartRC の想定人気（est_pop）から作る（通説の検証の章の第 13.3 節）。

    python -m src.scripts.data.folklore_prerace build   # → <FOLKLORE_OUT>/prerace.parquet・prerace.json

背景: 前日オッズの記録が手元に無く、第 13 節は odds_pred の想定オッズ（2022 年から）で代用していた。
SmartRC の runners にある est_pop（想定人気）は 2020〜2025 年の全行にあり、確定人気との完全一致は 34%（確定人気そのものではない）。
ただし順位は取消のあとの出走馬でつけ直されていて（想定人気の頭数 = 確定の出走頭数が 99%）、取得時刻の記録も無いので、前日の値とは断定できない。
下の確認は、結果由来の混入が疑われる向きか（疑われない向きか）を見るもので、混入が無いことの証明ではない。順位だけでオッズの値は無いので、
『est_pop の順位 × 頭数 から、確定の単勝・複勝オッズの確率を予測する』モデルで前日の想定オッズを作る。

- 目的: log(確定の単勝の確率 q)（レース内で合計 1）、log(確定の複勝オッズの下限)。
- 説明: est_pop の順位・順位/頭数・頭数（est だけの版 E）。2022 年以降は、さらに odds_pred の想定（share_raw・place_raw）を足した版 ES。
- 学習は評価年より前の年だけ（walk-forward）。評価年 Y = 2021〜2025（E）、2023〜2025（ES）。
- 確認: 想定の確率が確定の確率にどれだけ近いか、est_pop が確定オッズに無い情報（レース結果）を持っていないか
  （確定オッズの確率を条件にしたとき、est_pop の順位が 3 着内を追加で説明しないこと）。
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

WP_RAW = ROOT / "data/local/modeling/odds_pred/wp_raw_oof.parquet"


def load_est_pop(years=range(2020, 2026)) -> pd.DataFrame:
    rows = []
    for y in years:
        s = pd.read_parquet(ROOT / f"data/page_reference/tables/{y}/smartrc_race_flat.parquet", columns=["race_id", "uno", "est_pop"])
        s["race_id"] = s["race_id"].astype(str)
        s["horse_number"] = pd.to_numeric(s["uno"], errors="coerce")
        s["est_pop"] = pd.to_numeric(s["est_pop"], errors="coerce")
        rows.append(s[["race_id", "horse_number", "est_pop"]])
    return pd.concat(rows).drop_duplicates(["race_id", "horse_number"])


def build() -> None:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    d = pd.read_parquet(OUT / "flags.parquet", columns=["race_id", "horse_id", "horse_number", "year", "fin", "odds", "pop", "field", "place_odds_min"])
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    d = d[(d["odds"] > 0) & (d["year"] >= 2020)].merge(load_est_pop(), on=["race_id", "horse_number"], how="left")
    r = pd.read_parquet(WP_RAW, columns=["race_id", "horse_id", "share_raw", "place_raw"])
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    d = d.merge(r, on=["race_id", "horse_id"], how="left")
    d = d.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    ok = d.groupby("race_id")["est_pop"].transform(lambda s: s.notna().all() and len(s) >= 5)
    d = d[ok].reset_index(drop=True)
    inv = 1.0 / d["odds"]
    d["q"] = inv / inv.groupby(d["race_id"]).transform("sum")
    d["lq"] = np.log(d["q"].clip(1e-5))
    d["lplc"] = np.log(d["place_odds_min"].where(d["place_odds_min"] > 0).fillna(d["odds"] / 3).clip(1.0))
    d["n_est"] = d.groupby("race_id")["est_pop"].transform("max")  # 想定人気の頭数。確定の出走頭数 field とほぼ一致する（99%）（順位が取消後につけ直されている）ので、実質は同じ情報
    d["est_rel"] = d["est_pop"] / d["n_est"]
    sh = d["share_raw"].to_numpy(float)
    mx = pd.Series(sh).groupby(d["race_id"].to_numpy()).transform("max").to_numpy()
    d["lsh"] = sh - (np.log(pd.Series(np.exp(sh - mx)).groupby(d["race_id"].to_numpy()).transform("sum").to_numpy()) + mx)
    feats = {"E": ["est_pop", "est_rel", "n_est"], "ES": ["est_pop", "est_rel", "n_est", "lsh", "place_raw"]}
    params = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_child_samples=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=10.0, verbose=-1, n_jobs=8, seed=3)
    out = d[["race_id", "horse_id", "horse_number", "year"]].copy()
    info = {"years": {}}
    for ver, cols in feats.items():
        for tgt, col in (("lq", f"{ver}_lq"), ("lplc", f"{ver}_lplc")):
            out[col] = np.nan
        y0 = 2020 if ver == "E" else 2022
        for Y in range(y0 + 1, 2026):
            tr = (d["year"] >= y0) & (d["year"] < Y)
            te = d["year"] == Y
            if ver == "ES":
                tr &= d["lsh"].notna()
                te &= d["lsh"].notna()
            for tgt, col in (("lq", f"{ver}_lq"), ("lplc", f"{ver}_lplc")):
                m = lgb.train(params, lgb.Dataset(d.loc[tr, cols], d.loc[tr, tgt]), num_boost_round=250)
                out.loc[te, col] = m.predict(d.loc[te, cols])
    # レース内で合計 1 に正規化した想定の確率
    for ver in feats:
        e = np.exp(out[f"{ver}_lq"])
        out[f"{ver}_q"] = e / e.groupby(out["race_id"]).transform("sum")
    out.to_parquet(OUT / "prerace.parquet", index=False)
    # 確認 1: 想定の確率と確定の確率
    for Y in range(2021, 2026):
        x = d[d["year"] == Y]
        o = out.loc[x.index]
        y3 = (x["fin"] <= 3).to_numpy()
        yw = (x["fin"] == 1).to_numpy().astype(float)
        nr = x["race_id"].nunique()
        row = {"est_pop_top3_auc": float(roc_auc_score(y3, -x["est_pop"])), "final_odds_top3_auc": float(roc_auc_score(y3, x["q"])),
               "win_ll_final": float(-(yw * np.log(x["q"].clip(1e-9))).sum() / nr), "win_ll_E": float(-(yw * np.log(o["E_q"].clip(1e-9))).sum() / nr),
               "corr_lq_E": float(np.corrcoef(x["lq"], o["E_lq"])[0, 1]), "exact_est_eq_pop": float((x["est_pop"] == x["pop"]).mean())}
        if Y >= 2023:
            k = x["lsh"].notna().to_numpy() & o["ES_lq"].notna().to_numpy()  # odds_pred の想定が欠けた行は除いて比べる（欠けた行を含むと相関が NaN になっていた）
            nrk = x.loc[k, "race_id"].nunique()
            row["win_ll_ES"] = float(-(yw[k] * np.log(o.loc[k, "ES_q"].clip(1e-9))).sum() / nrk)
            row["win_ll_share"] = float(-(yw[k] * np.log(np.exp(x.loc[k, "lsh"]).clip(1e-9))).sum() / nrk)
            row["corr_lq_ES"] = float(np.corrcoef(x.loc[k, "lq"], o.loc[k, "ES_lq"])[0, 1])
            row["corr_lq_share"] = float(np.corrcoef(x.loc[k, "lq"], x.loc[k, "lsh"])[0, 1])
        info["years"][str(Y)] = row
    # 確認 2: 確定オッズの確率を条件にしても est_pop が 3 着内を説明するか（結果由来の混入の検査。説明するなら est_pop は確定後の情報を含む）
    from scipy.stats import norm
    import statsmodels.api as sm

    x = d[d["year"] >= 2021].copy()
    # 確定オッズの確率は 3 次まで入れる（本命・穴馬の偏りで曲がるぶんが est_rel に乗らないように）
    X = sm.add_constant(np.column_stack([x["lq"], x["lq"] ** 2, x["lq"] ** 3, x["est_rel"]]))
    fit = sm.Logit((x["fin"] <= 3).astype(float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(x["race_id"])[0]})
    from scipy.stats import spearmanr

    info["est_pop_vs_final"] = {"spearman_rank": float(spearmanr(x["est_pop"], x["pop"])[0]), "exact_share": float((x["est_pop"] == x["pop"]).mean()),
                                "n_est_eq_field": float((x["n_est"] == x["field"]).mean())}
    info["resid_check"] = {"coef_est_rel_given_final_odds": float(fit.params[4]), "z": float(fit.tvalues[4]), "p": float(2 * norm.sf(abs(fit.tvalues[4]))), "terms": "lq, lq^2, lq^3, est_rel"}
    json.dump(info, open(OUT / "prerace.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False, indent=1))


def drift() -> None:
    """人気の動き（想定人気 est_pop → 確定人気 pop）を加えたとき、確定オッズで期待値を測って買う成績が変わるか（2023〜2025 年）。
    est_pop の順位が確定オッズに無い情報を持つ（確定オッズの確率を条件にしても 3 着内を説明する）ことの検証と、その使い道の評価。
    学習は評価年より前の 2021 年以降だけ（walk-forward）。買い方は事前に固定（期待値 1.0・1.1・1.2 以上、払戻は確定オッズ）。"""
    from src.scripts.data.folklore_model import _cmp, ev_curve, model_top3, model_win

    d = pd.read_parquet(OUT / "flags.parquet", columns=["race_id", "horse_id", "horse_number", "year", "fin", "odds", "pop", "field", "win", "plc", "place_odds_min", "place_odds_max"])
    d["race_id"] = d["race_id"].astype(str)
    d = d[(d["odds"] > 0) & (d["year"] >= 2021)].merge(load_est_pop(), on=["race_id", "horse_number"], how="left")
    d = d.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    d = d[d.groupby("race_id")["est_pop"].transform(lambda s: s.notna().all() and len(s) >= 5)].reset_index(drop=True)
    inv = 1.0 / d["odds"]
    d["q"] = inv / inv.groupby(d["race_id"]).transform("sum")
    d["lq"] = np.log(d["q"].clip(1e-5))
    d["lq2"], d["lq3"] = d["lq"] ** 2, d["lq"] ** 3
    d["lplc"] = np.log(d["place_odds_min"].where(d["place_odds_min"] > 0).fillna(d["odds"] / 3).clip(1.0))
    d["lplc2"] = d["lplc"] ** 2
    d["n_est"] = d.groupby("race_id")["est_pop"].transform("max")  # 想定人気の頭数（確定の出走頭数とほぼ一致、99%）
    d["est_rel"] = d["est_pop"] / d["n_est"]
    d["pop_rel"] = d["pop"] / d["n_est"]  # 確定人気の順位 / 頭数
    d["d_rel"] = (d["pop"] - d["est_pop"]) / d["n_est"]  # ＋ = 確定の方が人気薄（人気が下がった）、− = 確定の方が人気（人気が上がった）
    d["zero"] = 0.0
    out = {"years": {}, "ev": {}}
    # 確定オッズは単勝・複勝とも 3 次まで。M0p は確定人気の順位/頭数を足し、M1 は更に想定人気を足す。M1 − M0p が『想定人気（人気の動き）が確定人気の順位に無い情報を持つか』
    d["lplc3"] = d["lplc"] ** 3
    wb, pb = ["lq", "lq2", "lq3"], ["lq", "lq2", "lq3", "lplc", "lplc2", "lplc3"]
    variants = {"M0": (wb, pb), "M0p": (wb + ["pop_rel"], pb + ["pop_rel"]), "M1": (wb + ["pop_rel", "est_rel"], pb + ["pop_rel", "est_rel"])}
    preds = {k: [] for k in variants}
    for Y in (2023, 2024, 2025):
        tr, te = d[d["year"].between(2021, Y - 1)], d[d["year"] == Y]
        for k, (wc, pc) in variants.items():
            pw, bw, _ = model_win(tr, te, [], wc, "zero", lam_fixed=1)
            p3, b3, _ = model_top3(tr, te, [], pc, "zero", lam_fixed=1)
            o = te[["race_id", "year"]].copy()
            o["pw"], o["p3"] = pw, p3
            preds[k].append(o)
            out["years"].setdefault(k, {})[str(Y)] = {"win": dict(zip(wc, map(float, bw))), "top3": dict(zip(pc + ["切片"], map(float, b3)))}
    for k in variants:
        pr = pd.concat(preds[k])
        x = d.loc[pr.index]
        res = {}
        for lab, m in (("全体", np.ones(len(pr), bool)), *[(str(y), (pr["year"] == y).to_numpy()) for y in (2023, 2024, 2025)]):
            rw = ev_curve((pr["pw"] * x["odds"]).to_numpy()[m], x["win"].to_numpy()[m], (x["fin"] == 1).to_numpy()[m], x["race_id"].to_numpy()[m], x["pop"].to_numpy(float)[m])
            rp = ev_curve((pr["p3"] * x["place_odds_min"].fillna(0)).to_numpy()[m], x["plc"].to_numpy()[m], (x["fin"] <= 3).to_numpy()[m], x["race_id"].to_numpy()[m], x["pop"].to_numpy(float)[m])
            res[lab] = {"win": {f"{r['th']:.2f}": r for r in rw}, "plc": {f"{r['th']:.2f}": r for r in rp}}
        out["ev"][k] = res
        preds[k] = pr
    te = d[d["year"] >= 2023]
    pm = {k: preds[k].loc[te.index] for k in variants}  # 行名で対応させる（並び順に頼らない）
    yw, y3 = (te["fin"] == 1).to_numpy().astype(float), (te["fin"] <= 3).to_numpy().astype(float)
    rr = te["race_id"].to_numpy()
    out["cmp"] = {"win": _cmp(pm["M1"]["pw"].to_numpy(), pm["M0p"]["pw"].to_numpy(), yw, rr, "win"), "top3": _cmp(pm["M1"]["p3"].to_numpy(), pm["M0p"]["p3"].to_numpy(), y3, rr, "top3")}
    out["cmp_pop"] = {"win": _cmp(pm["M0p"]["pw"].to_numpy(), pm["M0"]["pw"].to_numpy(), yw, rr, "win"), "top3": _cmp(pm["M0p"]["p3"].to_numpy(), pm["M0"]["p3"].to_numpy(), y3, rr, "top3")}
    # 人気の動きの 5 区分（固定の幅）ごとの 3 着内率・単勝回収率（確定オッズ、2021〜2025 年）
    # d_rel は約 34% が 0（想定人気と確定人気が同じ）。分位に分けると同値が年順で割られるので、固定の幅で分ける
    edges = [-9, -0.15, -0.04, 0.04, 0.15, 9]
    d["drift_q"] = pd.cut(d["d_rel"], edges, labels=["上昇（人気が上がった）", "やや上昇", "不変", "やや下降", "下降（人気が下がった）"], include_lowest=True)
    g = d.groupby("drift_q", observed=True).agg(n=("fin", "size"), top3=("fin", lambda s: (s <= 3).mean()), p_final=("q", "mean"), roi_win=("win", lambda s: s.mean() / 100), roi_plc=("plc", lambda s: s.mean() / 100))
    out["drift_bins"] = {str(k): {kk: float(vv) for kk, vv in r.items()} for k, r in g.iterrows()}
    out["drift_bin_edges"] = edges
    json.dump(out, open(OUT / "drift.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out["cmp"], ensure_ascii=False))
    print(g.round(4).to_string())
    for k in variants:
        for kind in ("win", "plc"):
            print(k, kind, {lab: " | ".join(f"{t}:{r['n']} {r['roi']:.3f}±{r['se']:.3f}" for t, r in v[kind].items() if t in ("1.00", "1.10", "1.20")) for lab, v in out["ev"][k].items() if lab == "全体"})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build, "drift": drift}[cmd]()

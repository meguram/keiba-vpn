"""市場の残差を目的にした target encoding を、市場 ＋ 血統（Benter）の入力にする版（統括役の判断で、血統だけの版と分けた別の版。開発の行・時間の順・記述）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.market_resid [--null 1]

- 残差: 過去のレースの A/E（Σ勝ち ÷ Σq）の対数。q は T-60 の再現（種を固定した 1 回）か、--final で確定オッズ。予測するレースのオッズは集計に入れない（as-of）。
- TE: 父 → 父の父 → 全体の階層で、芝ダごとに、q の和を擬似件数 20 にして上の段の比へ縮める（自分の行・同じ日を除く）。列は上の段との比の対数。父系の 4 代上（大系統）と母父も並べる。
- 評価: Benter（α log p_血統 + β log q_T60 + Σγ z）を、年 t を 2019〜t−1 年で当てて t 年に当てる（t = 2020〜24・開発の行）。
  基準は z なしの Benter。採点はレースの対数損失の差（勝ち馬の log p）と、行ごとの二値の差。区間は開催日のブートストラップ 1,000 回。
- 帰無（--null）: 系統の札（父・父の父・大系統・母父）を、同じ生年の馬の中で入れ替える。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data, pre_odds
from src.research.pedigree_lab.data import asof
from src.research.pedigree_lab.hyp_tests import clogit
from src.research.pedigree_lab.lineage import line_ids

M = 200.0
M_AE = 20.0            # A/E の縮めの擬似件数（勝ちの見込みの和で 20 勝ぶん）
PLACEBO = "--placebo" in sys.argv      # 偽薬: 目的を確定の q にする（2 号）
FINAL = "--final" in sys.argv          # q を確定オッズで作る版（1 号の直し: T-60 の再現だけだと揺れの分を血統が取り戻すだけの恐れ）


def main() -> None:
    null_seed = int(sys.argv[sys.argv.index("--null") + 1]) if "--null" in sys.argv else None
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "market_resid" + ("_final" if FINAL else "") + ("_placebo" if PLACEBO else "") + ("_null" if null_seed is not None else ""),
                                   "市場の残差を目的にした TE を Benter の入力にする版（尤度だけ・記述・血統だけの版とは別）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "win", "odds", "horse_number", "n_run", "surf", "sire", "ss", "ds"])
    X = X[(X["date"] <= pd.Timestamp(data.DEV_END)) & (X["odds"] > 0)]
    X = X[X.groupby("race_id")["horse_id"].transform("size") == X["n_run"]].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    X = X.merge(line_ids()[["horse_id", "L4"]], on="horse_id", how="left")
    if null_seed is not None:
        rng = np.random.default_rng(null_seed)
        H = X.drop_duplicates("horse_id")[["horse_id", "sire", "ss", "L4", "ds"]].reset_index(drop=True)
        H["by"] = H["horse_id"].str[:4]
        perm = np.arange(len(H))
        for ii in H.groupby("by").indices.values():
            perm[ii] = rng.permutation(ii)
        for c in ("sire", "ss", "L4", "ds"):                 # 札は組のまま入れ替える（父・父の父・大系統の整合を保つ）
            X[c] = X["horse_id"].map(dict(zip(H["horse_id"], H[c].to_numpy()[perm])))     # 母父も同じ生年の中で組のまま（2 号）
    rid = X["race_id"].to_numpy()
    rk = X.groupby("race_id")["odds"].rank(method="first").to_numpy()
    inv = 1 / X["odds"].to_numpy(float)
    q_fin = inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
    sims = pre_odds.simulate(rid, X["odds"].to_numpy(float), rk, pre_odds.load_drift("t60"), n=5)
    q_sims = [(1 / o) / pd.Series(1 / o).groupby(rid).transform("sum").to_numpy() for o in sims]
    # Benter の市場の入力: 確定の版は確定の q、T-60 の版は再現の 1 回目（判定に使うのは 1 回のオッズ）
    q60 = q_fin if FINAL else q_sims[0]
    # 残差の側の q: 確定の q か、T-60 の再現の 5 回の平均（揺れを減らす。2 号）。さらに q^β を和 1 に直して較正する（β は前の年までの市場だけの当てはめ。2 号）
    q_raw = q_fin if FINAL else np.mean(q_sims, axis=0)
    yrX = X["date"].dt.year.to_numpy()
    q_res = np.empty(len(X))
    lqr = np.log(np.clip(q_raw, 1e-6, 1))
    yw = X["win"].to_numpy(float)
    betas = {}
    for y in range(2016, 2025):
        if y == 2016:                                  # 2016 年は前の年が無いので較正しない（β = 1。同じ年の勝ちを使わない。critic）
            b = 1.0
        else:
            tr = yrX < y
            b = clogit(lqr[tr][:, None], yw[tr], pd.factorize(rid[tr])[0])[0]
        betas[y] = round(float(b), 4)
        m = yrX == y
        e = np.exp(b * lqr[m] - pd.Series(b * lqr[m]).groupby(rid[m]).transform("max").to_numpy())
        q_res[m] = e / pd.Series(e).groupby(rid[m]).transform("sum").to_numpy()
    # 偽薬（2 号）: 目的を勝ちではなく確定の q にして、再現の作りの癖だけを測る（T-60 の版で意味を持つ）
    if PLACEBO and FINAL:
        raise SystemExit("--placebo は T-60 の版だけ（確定の版では目的と分母がどちらも確定の q になる。critic）")
    num = q_fin if PLACEBO else X["win"].to_numpy(float)
    den = q_raw if PLACEBO else q_res              # 偽薬の分母は較正前の 5 回の平均（β の分の系統的なずれを入れない。critic）
    # A/E の対数（Σ目的 ÷ Σq）を、q の和を擬似件数にして上の段の比へ縮める（1 号の直し: 確率の差だと本命の行が値を決める）
    E = X.assign(_w=num, _q=den, _k="1")
    cols = []

    def ratio(keys, parent):
        s_, _ = asof(E, keys, ["_w", "_q"])
        return (s_[:, 0] + M_AE * parent) / (s_[:, 1] + M_AE)
    one = np.ones(len(X))
    r_base = ratio(["_k", "surf"], one)
    r_ss = ratio(["ss", "surf"], r_base)
    r_si = ratio(["sire", "surf"], r_ss)
    r_l4 = ratio(["L4", "surf"], r_base)
    r_ds = ratio(["ds", "surf"], r_base)
    for nm, v in (("mr_sire", np.log(r_si / r_ss)), ("mr_ss", np.log(r_ss / r_base)), ("mr_l4", np.log(r_l4 / r_base)), ("mr_bms", np.log(r_ds / r_base))):
        X[nm] = v
        cols.append(nm)
    WF = pd.read_parquet(data.OUT / "oof_v4_pure.parquet")[["race_id", "horse_id", "p_win"]]
    S = X[X["split"].isin([0, 1])].merge(WF, on=["race_id", "horse_id"], how="inner")
    S = S[S.groupby("race_id")["horse_id"].transform("size") == S["n_run"]].sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    # 新しい父の群は、帰無の札の入れ替えの前の本物の父で、2017 年以降の全行の初出から決める（timeband・compare_null と同じ。critic）
    F0 = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "sire"])
    F0 = F0[(F0["date"].dt.year >= 2017) & (F0["date"] <= pd.Timestamp(data.DEV_END))]
    first_run = F0.groupby("sire")["date"].min()
    S = S.merge(F0[["race_id", "horse_id", "sire"]].rename(columns={"sire": "sire_true"}), on=["race_id", "horse_id"], how="left")
    S = S.merge(X[["race_id", "horse_id"]].assign(q60=q60), on=["race_id", "horse_id"], how="left")
    rid = S["race_id"].to_numpy()
    yr = S["date"].dt.year.to_numpy()
    y = S["win"].to_numpy(float)
    lp = np.log(np.clip(S["p_win"].to_numpy(), 1e-6, 1))
    lq = np.log(np.clip(S["q60"].to_numpy(), 1e-6, 1))

    def zc(v):
        c = v - pd.Series(v).groupby(rid).transform("mean").to_numpy()
        sd = c[yr < 2020].std()
        return c / (sd if sd > 0 else 1.0)
    Z = np.c_[tuple(zc(S[c].to_numpy()) for c in cols)]
    out = {}
    pr = {"base": np.full(len(S), np.nan), "mr": np.full(len(S), np.nan)}
    coefs = {}
    for t in range(2020, 2025):
        tr, te = (yr >= 2019) & (yr < t), yr == t
        fr = pd.factorize(rid[tr])[0]
        b0 = clogit(np.c_[lp[tr], lq[tr]], y[tr], fr)
        b1 = clogit(np.c_[lp[tr], lq[tr], Z[tr]], y[tr], fr)
        coefs[t] = {"base": np.round(b0, 4).tolist(), "mr": np.round(b1, 4).tolist()}
        for nm, M_, b in (("base", np.c_[lp, lq], b0), ("mr", np.c_[lp, lq, Z], b1)):
            sc = M_[te] @ b
            e = np.exp(sc - pd.Series(sc).groupby(rid[te]).transform("max").to_numpy())
            pr[nm][te] = e / pd.Series(e).groupby(rid[te]).transform("sum").to_numpy()
    te = yr >= 2020
    unseen = (S["sire_true"].map(first_run) >= pd.to_datetime(S["date"].dt.year.astype(str) + "-01-01")).to_numpy()
    has_sire = S["sire_true"].fillna("").ne("").to_numpy()           # 採点の行は父が空でない行（rulebased.score と同じ。critic）

    def bll(yv, pv):
        pv = np.clip(pv, 1e-6, 1 - 1e-6)
        return -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))
    d_row = bll(y, pr["base"]) - bll(y, pr["mr"])
    d_win = np.where(y == 1, np.log(np.clip(pr["mr"], 1e-9, 1)) - np.log(np.clip(pr["base"], 1e-9, 1)), 0.0)
    day = S["date"].dt.strftime("%Y%m%d").to_numpy()
    du, dinv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(3)
    Wb = np.stack([np.bincount(rng.integers(0, len(du), len(du)), minlength=len(du)) for _ in range(1000)]).astype(float)
    out_p = S.loc[te, ["race_id", "horse_id"]].copy()
    out_p["p_base"], out_p["p_mr"] = pr["base"][te], pr["mr"][te]
    tag0 = f"{'_final' if FINAL else ''}{'_placebo' if PLACEBO else ''}{'' if null_seed is None else f'_null{null_seed}'}"
    out_p.to_parquet(data.OUT / f"market_resid_wf{tag0}.parquet", index=False)      # 版 − 帰無の比べのため（critic）
    for nm, mk in {"全体": te & has_sire, "学習の期間に父の産駒が 1 走も無い": te & unseen & has_sire, "それ以外": te & ~unseen & has_sire}.items():
        r = {"rows": int(mk.sum())}
        for k, v, den in ((("row_binary", d_row, None), ("race_ll", d_win, y)) if nm == "全体" else (("row_binary", d_row, None),)):
            dd = np.bincount(dinv[mk], weights=v[mk], minlength=len(du))
            nn = np.bincount(dinv[mk], weights=(np.ones(len(v)) if den is None else den)[mk], minlength=len(du))
            bs = (Wb @ dd) / (Wb @ nn)
            r[k] = [round(float(dd.sum() / nn.sum()), 6), np.percentile(bs, [2.5, 97.5]).round(6).tolist()]
        out[nm] = r
        print(nm, r, flush=True)
    res = {"cols": cols, "coefs": coefs, "score": out, "null_seed": null_seed, "betas": betas,
           "sd_raw": {c: round(float(S[c].std()), 6) for c in cols}}
    path = data.OUT / f"market_resid{'_final' if FINAL else ''}{'_placebo' if PLACEBO else ''}{'' if null_seed is None else f'_null{null_seed}'}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(coefs, ensure_ascii=False), flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()

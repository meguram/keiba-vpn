"""同じレースの 2 頭が「一緒に来やすいか」を、脚質・展開予想・走法軸の近さで測る（走法軸の説明力の調査、2026-11-18）。

python -m src.scripts.data.pair_cohesion run [--years 2016-2024] [--boot 300]

- 期間は 2016〜2024 年だけ（2025・2026 年は読まない）。
- 型はどれもレースの前に分かる値で作る（結果の位置取りは使わない）:
  - 脚質: 位置取りモデルの特徴量 h_pl_3（前 3 走の 4 角の位置の平均、0=先頭〜1=最後方）と h_pf_w5（前 5 走の初角の位置の加重平均）。
    data/local/modeling/pos_model/features.parquet（src/scripts/data/pos_model.py。shift(1) 以降の前走だけ）。
    群は h_pl_3 で 逃げ（≤0.15）・先行（≤0.35）・差し（≤0.65）・追込（それ以外）。position_forecast.py の初角の境界と同じ値。
  - 展開予想: ラップ予想の確率 lp_p_slow・lp_p_mid・lp_p_high（retrain_out/tp55_2016/groups/LAP.parquet。年を外した予測で、
    記録に「軽い先読み」の注記がある）。レースの型 = 最大の確率。向く側 = スローなら逃げ・先行、ハイなら差し・追込。
  - 走法軸: data/local/running_profile/axis_scores10.parquet（各レースの直前の時点の 10 点。build_axis_scores10.py）。
    2023 年までほぼ空なので、2024 年の TS・ギアチェンジ・TS持続力の 3 軸だけで測る（位置系の軸は 2024 年で 11% しか無い）。
- 見込み（独立を仮定）: 市場の単勝オッズの確率から、Harville（Plackett–Luce）で「2 頭とも 3 着以内」の確率を出す。Harville はそれ自体に偏り（人気馬の過小・人気薄の過大）があるので、
  比はその確率の水準（10 分位）や、2 頭の人気帯の組ごとにそろえてから出す（層別の比）。
  確率は 1〜3 着の並びを総当たりした正確な値（harville_exact）。
- 区間はレースで束ねたブートストラップ（Poisson の重み）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path("/home/jovyan/work/retrain_out/pair_cohesion")
POS_FEAT = ROOT / "data/local/modeling/pos_model/features.parquet"
LAP = Path("/home/jovyan/work/retrain_out/tp55_2016/groups/LAP.parquet")
AXIS = ROOT / "data/local/running_profile/axis_scores10.parquet"
STYLE_EDGES = (0.15, 0.35, 0.65)
STYLE_NAMES = ("逃げ", "先行", "差し", "追込")
POP_EDGES = (1, 3, 6, 9)          # 人気帯: 1・2〜3・4〜6・7〜9・10〜


def style_of(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), np.searchsorted(np.asarray(STYLE_EDGES), x, side="left"), -1)


def popband(p: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(POP_EDGES), p, side="left")


def load(years: list[int]) -> pd.DataFrame:
    from src.config.data_paths import TABLES_DIR

    cols = ["race_id", "date", "surface", "distance", "track_condition", "horse_number", "horse_id", "finish_position", "odds", "popularity"]
    f = pd.concat([pd.read_parquet(Path(TABLES_DIR) / str(y) / "race_result_flat.parquet", columns=cols) for y in years], ignore_index=True)
    f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    for c in ("odds", "popularity", "finish_position", "horse_number", "distance"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f[(f["odds"] > 0) & (f["finish_position"] > 0)]                 # 走った馬（取消・除外・中止は除く）
    f = f[f["surface"].astype(str).isin(["芝", "ダート"])]               # 障害は外す（位置取りの特徴量の対象外）
    miss = f["popularity"].isna()                                       # 人気が欠けた馬は、レース内の単勝オッズの順位で補う
    if miss.any():
        f.loc[miss, "popularity"] = f[miss].groupby("race_id")["odds"].rank(method="min")
    f["track_condition"] = f["track_condition"].fillna("不明")
    pf = pd.read_parquet(POS_FEAT, columns=["race_id", "horse_id", "h_pl_3", "h_pf_w5"])
    pf["race_id"], pf["horse_id"] = pf["race_id"].astype(str), pf["horse_id"].astype(str)
    f = f.merge(pf.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    lp = pd.read_parquet(LAP, columns=["race_id", "lp_p_slow", "lp_p_mid", "lp_p_high"]).drop_duplicates("race_id")
    lp["race_id"] = lp["race_id"].astype(str)
    f = f.merge(lp, on="race_id", how="left")
    ax = pd.read_parquet(AXIS, columns=["race_id", "horse_id", "ts", "gear", "sustain"])
    ax["race_id"], ax["horse_id"] = ax["race_id"].astype(str), ax["horse_id"].astype(str)
    f = f.merge(ax.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    return f


def harville_exact(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """勝率 q（合計 1）の Harville（Plackett–Luce）で、各馬の 3 着以内の確率と、2 頭とも 3 着以内の確率の行列（正確な値）。
    1〜3 着の並び（a, b, c）の確率 q_a · q_b/(1−q_a) · q_c/(1−q_a−q_b) を総当たりして足す。"""
    n = len(q)
    qa = q[:, None, None]
    qb = q[None, :, None]
    qc = q[None, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        T = qa * qb / (1 - qa) * qc / (1 - qa - qb)
    idx = np.arange(n)
    T[idx, idx, :] = 0
    T[idx, :, idx] = 0
    T[:, idx, idx] = 0
    T = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
    Mo = T.sum(2) + T.sum(1) + T.sum(0)          # i が j より前で、2 頭とも 3 着以内
    pp = Mo + Mo.T
    p3 = T.sum((1, 2)) + T.sum((0, 2)) + T.sum((0, 1))
    return p3, pp


def build_pairs(f: pd.DataFrame) -> pd.DataFrame:
    out = []
    for rid, g in f.groupby("race_id", sort=True):
        n = len(g)
        if n < 5:
            continue
        inv = 1.0 / g["odds"].to_numpy(float)
        q = inv / inv.sum()
        p3, pp = harville_exact(q)
        fin = g["finish_position"].to_numpy(float)
        hn = g["horse_number"].to_numpy(int)
        top3 = (fin <= 3).astype(float)
        pl = g["h_pl_3"].to_numpy(float)
        pfw = g["h_pf_w5"].to_numpy(float)
        st = style_of(pl)
        pop = popband(g["popularity"].to_numpy(float))
        pace = np.nan
        lp = g[["lp_p_slow", "lp_p_mid", "lp_p_high"]].iloc[0].to_numpy(float)
        if np.isfinite(lp).all():
            pace = int(np.argmax(lp))                                    # 0 スロー・1 ミドル・2 ハイ
        fav = np.where(pace == 0, np.isin(st, (0, 1)), np.where(pace == 2, np.isin(st, (2, 3)), False)) if np.isfinite(pace) else np.zeros(n, bool)
        fav = np.where(st >= 0, fav, False)                              # 脚質の分からない馬は、向く側に数えない（組ごと外す: 下の known）
        axv = g[["ts", "gear", "sustain"]].to_numpy(float)
        i, j = np.triu_indices(n, 1)
        ok_ax = np.isfinite(axv[i]).all(1) & np.isfinite(axv[j]).all(1)
        d_ax = np.where(ok_ax, np.sqrt(((axv[i] - axv[j]) ** 2).sum(1)), np.nan)
        out.append(pd.DataFrame({
            "race_id": rid, "year": int(rid[:4]), "n": n, "surface": str(g["surface"].iloc[0]), "distance": float(g["distance"].iloc[0]),
            "going": str(g["track_condition"].iloc[0]), "pace": pace,
            "y": top3[i] * top3[j], "e": pp[i, j], "res_i": top3[i] - p3[i], "res_j": top3[j] - p3[j],
            "fdiff": np.abs(fin[i] - fin[j]) / n,
            "p3_i": p3[i], "p3_j": p3[j], "t3_i": top3[i], "t3_j": top3[j], "po_i": pop[i], "po_j": pop[j],
            "hn_i": hn[i], "hn_j": hn[j],
            "d_pl": np.abs(pl[i] - pl[j]), "d_pf": np.abs(pfw[i] - pfw[j]),
            "st_i": st[i], "st_j": st[j], "pb_i": np.minimum(pop[i], pop[j]), "pb_j": np.maximum(pop[i], pop[j]),
            "fav_n": fav[i].astype(int) + fav[j].astype(int), "d_ax": d_ax}))
    d = pd.concat(out, ignore_index=True)
    d["e_dec"] = pd.qcut(d["e"].rank(method="first"), 10, labels=False)
    d["pbpair"] = d["pb_i"] * 10 + d["pb_j"]
    known = (d["st_i"] >= 0) & (d["st_j"] >= 0)
    d["style_rel"] = np.where(~known, "不明", np.where(d["st_i"] == d["st_j"], "同じ", np.where(np.abs(d["st_i"] - d["st_j"]) == 1, "隣", "離れた")))
    return d


def calibrate(d: pd.DataFrame) -> pd.DataFrame:
    """1 頭ずつの市場のずれ（3 着以内の実際 ÷ Harville の見込み）を、ペースの予想 × 脚質 × 人気帯ごとに、その年を外した年で作る。
    組の見込み e を e × c_i × c_j に直し（e_raw に元の値）、その確率の 10 分位を作り直す。脚質ごとに市場が 1 頭単位でずれる分を、
    組で一緒に来る分と混ぜないため（レビューの指摘、2026-11-18）。"""
    pace = d["pace"].fillna(-1).astype(int)
    hs = []
    for side in ("i", "j"):
        hs.append(pd.DataFrame({"race_id": d["race_id"], "year": d["year"], "pace": pace, "st": d[f"st_{side}"], "po": d[f"po_{side}"],
                                "p3": d[f"p3_{side}"], "t3": d[f"t3_{side}"], "hn": d[f"hn_{side}"]}))
    h = pd.concat(hs, ignore_index=True).drop_duplicates(["race_id", "hn"])     # 1 頭は複数の組に出るので、馬ごとに 1 行にする
    key = ["pace", "st", "po"]
    tot = h.groupby(key + ["year"]).agg(t=("t3", "sum"), e=("p3", "sum")).reset_index()
    allk = tot.groupby(key)[["t", "e"]].sum()
    tot = tot.join(allk, on=key, rsuffix="_all")
    tot["c"] = ((tot["t_all"] - tot["t"]) / (tot["e_all"] - tot["e"]).where(lambda x: x > 0)).fillna(1.0).clip(0.5, 2.0)
    cm = tot.set_index(key + ["year"])["c"]
    for side in ("i", "j"):
        idx = pd.MultiIndex.from_arrays([pace, d[f"st_{side}"], d[f"po_{side}"], d["year"]])
        d[f"c_{side}"] = cm.reindex(idx).fillna(1.0).to_numpy()
    d["e_raw"] = d["e"]
    d["e"] = d["e"] * d["c_i"] * d["c_j"]
    d["e_dec_raw"] = d["e_dec"]
    d["e_dec"] = pd.qcut(d["e"].rank(method="first"), 10, labels=False)
    return d


def _race_strata(d: pd.DataFrame, strata: str):
    rc = d["race_id"].astype("category").cat.codes.to_numpy()
    sk = d[strata].astype(int).to_numpy()
    return rc, sk, rc.max() + 1, sk.max() + 1


def _sums(rc, sk, nr, ns, y, e, mask):
    Y = np.zeros((nr, ns))
    E = np.zeros((nr, ns))
    np.add.at(Y, (rc[mask], sk[mask]), y[mask])
    np.add.at(E, (rc[mask], sk[mask]), e[mask])
    return Y, E


def lift_table(d: pd.DataFrame, by: str, strata: str, n_boot: int, seed: int = 1) -> pd.DataFrame:
    """層（strata）ごとの全体の比（Σy/Σe）で見込みを補正した、群（by）ごとの比 = Σy / Σ(e × 層の比)。
    区間はレースで束ねたブートストラップ（Poisson の重み）。層の比も回ごとに作り直す（diff_ci と同じ作り）。"""
    d = d.reset_index(drop=True)
    rc, sk, nr, ns = _race_strata(d, strata)
    y, e = d["y"].to_numpy(float), d["e"].to_numpy(float)
    Y, E = _sums(rc, sk, nr, ns, y, e, np.ones(len(d), bool))
    ls0 = Y.sum(0) / np.maximum(E.sum(0), 1e-12)
    rng = np.random.default_rng(seed)
    W = rng.poisson(1.0, size=(n_boot, nr)).astype(np.float32)
    ls = (W @ Y) / np.maximum(W @ E, 1e-12)
    ea = e * ls0[sk]
    rows = []
    g_all = d[by].astype(str).to_numpy()
    for lv in sorted(d[by].dropna().unique(), key=lambda v: str(v)):
        m = g_all == str(lv)
        if m.sum() == 0:
            continue
        Yg, Eg = _sums(rc, sk, nr, ns, y, e, m)
        bs = (W @ Yg).sum(1) / np.maximum(((W @ Eg) * ls).sum(1), 1e-12)
        ri, rj = d["res_i"].to_numpy()[m], d["res_j"].to_numpy()[m]
        corr = float(np.corrcoef(ri, rj)[0, 1]) if m.sum() > 10 else float("nan")
        den = (Eg.sum(0) * ls0).sum()
        rows.append({"群": lv, "組の数": int(m.sum()), "両方3着内（実）": float(y[m].mean()), "見込み（層をそろえた）": float(ea[m].mean()),
                     "比": float(Yg.sum() / den) if den > 0 else float("nan"), "比_lo": float(np.quantile(bs, 0.025)), "比_hi": float(np.quantile(bs, 0.975)),
                     "着順の差（頭数で割った平均）": float(d["fdiff"].to_numpy()[m].mean()), "市場を除いた相関": corr})
    return pd.DataFrame(rows)


def diff_ci(d: pd.DataFrame, by: str, a, b, strata: str, n_boot: int, seed: int = 1) -> dict:
    """群 a の比 − 群 b の比（点の値は重みなし）と、同じ重みのレースのブートストラップの 95% 区間（層の比も回ごとに作り直す）。"""
    d = d.reset_index(drop=True)
    rc, sk, nr, ns = _race_strata(d, strata)
    y, e = d["y"].to_numpy(float), d["e"].to_numpy(float)
    g = d[by].astype(str).to_numpy()
    Y, E = _sums(rc, sk, nr, ns, y, e, np.ones(len(d), bool))
    Ya, Ea = _sums(rc, sk, nr, ns, y, e, g == str(a))
    Yb, Eb = _sums(rc, sk, nr, ns, y, e, g == str(b))
    ls0 = Y.sum(0) / np.maximum(E.sum(0), 1e-12)
    pt = Ya.sum() / (Ea.sum(0) * ls0).sum() - Yb.sum() / (Eb.sum(0) * ls0).sum()
    rng = np.random.default_rng(seed)
    W = rng.poisson(1.0, size=(n_boot, nr)).astype(np.float32)
    ls = (W @ Y) / np.maximum(W @ E, 1e-12)
    ra = (W @ Ya).sum(1) / np.maximum(((W @ Ea) * ls).sum(1), 1e-12)
    rb = (W @ Yb).sum(1) / np.maximum(((W @ Eb) * ls).sum(1), 1e-12)
    dd = ra - rb
    return {"a": str(a), "b": str(b), "diff": float(pt), "lo": float(np.quantile(dd, 0.025)), "hi": float(np.quantile(dd, 0.975))}


def run(a) -> None:
    t0 = time.time()
    ys = [int(x) for x in str(a.years).split("-")]
    y0, y1 = ys[0], ys[-1]
    years = list(range(y0, y1 + 1))
    if max(years) > 2024:
        raise SystemExit("2025・2026 年は読まない")
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"pairs_{y0}_{y1}_exact_v3.parquet"
    if cache.exists():
        d = pd.read_parquet(cache)
    else:
        f = load(years)
        d = build_pairs(f)
        d.to_parquet(cache, index=False)
    print("組", len(d), "レース", d["race_id"].nunique(), f"{time.time() - t0:.0f}s", flush=True)
    d = calibrate(d)
    d["d_pl_q"] = pd.qcut(d["d_pl"].rank(method="first"), 5, labels=False).where(d["d_pl"].notna())
    d["d_pf_q"] = pd.qcut(d["d_pf"].rank(method="first"), 5, labels=False).where(d["d_pf"].notna())
    d["yband"] = pd.cut(d["year"], [2015, 2018, 2021, 2024], labels=["2016-18", "2019-21", "2022-24"]).astype(str)
    d["fband"] = pd.cut(d["n"], [0, 12, 15, 99], labels=["~12頭", "13-15頭", "16頭~"]).astype(str)
    d["dband"] = pd.cut(d["distance"], [0, 1400, 1800, 2200, 9999], right=False, labels=["~1300m", "1400-1700m", "1800-2100m", "2200m~"]).astype(str)
    d["paceL"] = d["pace"].map({0: "スロー予想", 1: "ミドル予想", 2: "ハイ予想"}).fillna("不明")
    d["goingL"] = np.where(d["going"] == "良", "良", np.where(d["going"].isin(["不明", "nan", ""]), "不明", "稍重以上"))
    res = {"n_pairs": int(len(d)), "n_races": int(d["race_id"].nunique()), "years": years, "expected": "Harville の正確な値（市場の単勝オッズ）", "n_boot": a.boot, "tables": {}}
    T = res["tables"]
    known = d[d["style_rel"] != "不明"]
    for strata in ("e_dec", "pbpair"):
        T[f"脚質の関係|{strata}"] = lift_table(known, "style_rel", strata, a.boot).to_dict("records")
        T[f"4角の位置の差の5分位|{strata}"] = lift_table(d[d["d_pl_q"].notna()], "d_pl_q", strata, a.boot).to_dict("records")
        T[f"初角の位置の差の5分位|{strata}"] = lift_table(d[d["d_pf_q"].notna()], "d_pf_q", strata, a.boot).to_dict("records")
        T[f"展開予想の向く側の頭数|{strata}"] = lift_table(known[known["paceL"] != "不明"], "fav_n", strata, a.boot).to_dict("records")
    # 対照: 人気が近い組（同じ人気帯）と、無作為の組（全組の比は層をそろえると 1。脚質の不明を含む全組）
    d["pb_rel"] = np.where(d["pb_i"] == d["pb_j"], "同じ人気帯", np.where(d["pb_j"] - d["pb_i"] == 1, "隣の人気帯", "離れた人気帯"))
    T["人気帯の関係（対照）|e_dec"] = lift_table(d, "pb_rel", "e_dec", a.boot).to_dict("records")
    rng_ = np.random.default_rng(7)
    d["rand"] = rng_.integers(0, 5, len(d))
    T["無作為の5群（対照）|e_dec"] = lift_table(d, "rand", "e_dec", a.boot).to_dict("records")
    # 上乗せ: 脚質の関係の中で、4 角の位置の差の 5 分位（脚質の群の境目の効きを除く）と、人気帯の関係をそろえた層での脚質の関係
    known = d[d["style_rel"] != "不明"]
    rows = []
    for lv, g in known.groupby("pb_rel"):
        for r in lift_table(g, "style_rel", "e_dec", a.boot).to_dict("records"):
            rows.append({"人気帯の関係": lv, **r})
    T["脚質の関係×人気帯の関係|e_dec"] = rows
    T["脚質の関係|較正なし（前の版）"] = lift_table(known.assign(e=known["e_raw"], e_dec=known["e_dec_raw"]), "style_rel", "e_dec", a.boot).to_dict("records")
    T["展開予想の向く側の頭数|較正なし（前の版）"] = lift_table(known[known["paceL"] != "不明"].assign(e=lambda x: x["e_raw"], e_dec=lambda x: x["e_dec_raw"]), "fav_n", "e_dec", a.boot).to_dict("records")
    sh = known[known["paceL"].isin(["スロー予想", "ハイ予想"])]
    T["展開予想の向く側の頭数（スロー・ハイ予想のレースだけ）|較正なし（前の版）"] = lift_table(sh.assign(e=sh["e_raw"], e_dec=sh["e_dec_raw"]), "fav_n", "e_dec", a.boot).to_dict("records")
    T["展開予想の向く側の頭数（スロー・ハイ予想のレースだけ）|e_dec"] = lift_table(sh, "fav_n", "e_dec", a.boot).to_dict("records")
    res["diffs"] = {"脚質 同じ−離れた": diff_ci(known, "style_rel", "同じ", "離れた", "e_dec", a.boot),
                    "脚質 隣−離れた": diff_ci(known, "style_rel", "隣", "離れた", "e_dec", a.boot),
                    "4角の差 最も近い−最も遠い": diff_ci(d[d["d_pl_q"].notna()], "d_pl_q", 0.0, 4.0, "e_dec", a.boot),
                    "展開 2頭−0頭（スロー・ハイ）": diff_ci(sh, "fav_n", 2, 0, "e_dec", a.boot),
                    "展開 2頭−0頭（スロー・ハイ・較正なし）": diff_ci(sh.assign(e=sh["e_raw"], e_dec=sh["e_dec_raw"]), "fav_n", 2, 0, "e_dec", a.boot)}
    res["calibration"] = "1 頭ずつの市場のずれ（ペースの予想 × 脚質 × 人気帯、その年を外した年）を見込みに掛けた版を主にする"
    T["全体（層をそろえた比は 1 になる確かめ）"] = lift_table(d.assign(all_="全組"), "all_", "e_dec", a.boot).to_dict("records")
    # 条件別・年ごと（脚質が同じ組と、4 角の位置の差が最小の 5 分位）
    for cond in ("yband", "fband", "surface", "dband", "paceL", "goingL"):
        rows = []
        for lv, g in known[known[cond] != "不明"].groupby(cond):
            t = lift_table(g, "style_rel", "e_dec", a.boot)
            for r in t.to_dict("records"):
                rows.append({"条件": lv, **r})
        T[f"条件別の脚質の関係|{cond}"] = rows
    # 走法軸（2024 年・TS 系 3 軸）: 脚質の関係ごとに、走法軸の距離の 5 分位で比べる（脚質を超える上乗せ）
    ax = known[known["d_ax"].notna()].copy()
    if len(ax):
        ax["d_ax_q"] = pd.qcut(ax["d_ax"].rank(method="first"), 5, labels=False)
        T["走法軸の距離の5分位（2024・TS系3軸）|e_dec"] = lift_table(ax, "d_ax_q", "e_dec", a.boot).to_dict("records")
        T["走法軸の距離の5分位（2024・TS系3軸）|pbpair"] = lift_table(ax, "d_ax_q", "pbpair", a.boot).to_dict("records")
        rows = []
        for lv, g in ax.groupby("style_rel"):
            for r in lift_table(g, "d_ax_q", "e_dec", a.boot).to_dict("records"):
                rows.append({"脚質の関係": lv, **r})
        T["走法軸の距離の5分位×脚質の関係（2024）"] = rows
        res["axis_note"] = {"n_pairs": int(len(ax)), "n_races": int(ax["race_id"].nunique())}
        ax["d_pl_q24"] = pd.qcut(ax["d_pl"].rank(method="first"), 5, labels=False)
        T["同じ組（2024）での比べ: 脚質の関係"] = lift_table(ax, "style_rel", "e_dec", a.boot).to_dict("records")
        T["同じ組（2024）での比べ: 4角の位置の差の5分位"] = lift_table(ax, "d_pl_q24", "e_dec", a.boot).to_dict("records")
        res["diffs"]["2024 走法軸 最も近い−最も遠い"] = diff_ci(ax, "d_ax_q", 0, 4, "e_dec", a.boot)
        res["diffs"]["2024 4角の差 最も近い−最も遠い"] = diff_ci(ax, "d_pl_q24", 0, 4, "e_dec", a.boot)
        res["axis_rank_corr_with_pl"] = float(ax[["d_ax", "d_pl"]].rank().corr().iloc[0, 1])
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("書いた", OUT / "result.json", f"{time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--years", default="2016-2024")
    r.add_argument("--boot", type=int, default=300)
    r.set_defaults(func=run)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

"""去勢明け N 戦目（1, 2, 3, 4-5, 6+）の成績分析（tp-32 の GELD は初戦のみ）。

python -m src.scripts.data.geld_analysis run [--out DIR]   # 集計して DIR/results.json と図を出す
python -m src.scripts.data.geld_analysis plot [--out DIR]  # 図
python -m src.scripts.data.geld_analysis chapter [--out DIR]  # results.json から docs の章を出す

定義: 馬の戦績（2020 年〜、障害を除く）で、直前の出走が牡で今回がセの最初の出走を『去勢明け 1 戦目』とし、以降の出走を 2, 3, … 戦目と数える。
最初に見えた出走が既にセの馬（去勢時点不明）は N を決められないので、N 別の集計から除き、『セ馬一般』の比較対象にだけ使う。
市場オッズは回収率（結果の評価）にだけ使う。特徴量にはしない。2026 年は記述統計のみ。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("GELD_ROOT") or Path(__file__).resolve().parents[3])
TABLES = ROOT / "data/page_reference/tables"
FRAME = ROOT / "data/local/modeling/betting/tp36/frame_all.parquet"
CLASS_LEVEL = {"新馬": 0.0, "未勝利": 0.0, "1勝": 1.0, "2勝": 2.0, "3勝": 3.0, "OP": 4.0, "L": 4.5, "G3": 5.0, "G2": 6.0, "G1": 7.0}
NGROUPS = ["1", "2", "3", "4-5", "6+"]
B = 1000
SEED = 20261022


def n_group(n: float) -> str | None:
    if not np.isfinite(n):
        return None
    n = int(n)
    return "1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4-5" if n <= 5 else "6+"


def dist_band(d: float) -> str:
    return "S" if d < 1400 else "M1" if d < 1700 else "M2" if d < 2000 else "L1" if d < 2500 else "L2"


# ───────────────────────── データ
def build_runs() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT))
    from src.pipeline.features import factory_self as fs

    r = fs.load_runs(TABLES, list(range(2020, 2027)))
    r = r[r["surface"].isin(["芝", "ダート"])].copy()
    s = r["sex_age"].astype(str).str.extract(r"^(\D+)(\d+)")
    r["sex"], r["age"] = s[0], pd.to_numeric(s[1], errors="coerce")
    r = r.sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)
    g = r.groupby("horse_id", sort=False)
    r["idx"] = g.cumcount()
    r["psex"] = g["sex"].shift()
    r["gap"] = (r["date"] - g["date"].shift()).dt.days
    fin = pd.to_numeric(r["finish_position"], errors="coerce")
    n = pd.to_numeric(r["field_size"], errors="coerce").clip(lower=2)
    r["rank"] = fin
    r["fp"] = (fin - 1.0) / (n - 1.0)
    r["win"], r["top2"], r["top3"] = (fin == 1).astype(float), (fin <= 2).astype(float), (fin <= 3).astype(float)
    r["cls"] = r["grade"].map(CLASS_LEVEL)
    r["dband"] = r["distance"].astype(float).map(dist_band)
    r["year"] = r["date"].dt.year
    r["rest"] = (r["gap"] >= 84).astype(float)

    # 去勢明け N 戦目
    geld_first_idx = r[(r["sex"] == "セ") & (r["psex"] == "牡")].groupby("horse_id")["idx"].min()
    r["gidx"] = r["horse_id"].map(geld_first_idx)
    r["n_post"] = np.where(r["gidx"].notna() & (r["sex"] == "セ"), r["idx"] - r["gidx"] + 1, np.nan)
    r["ng"] = r["n_post"].map(n_group)
    r["kind"] = np.where(r["sex"] == "牝", "female", np.where(r["sex"] == "牡", "male", np.where(r["ng"].notna(), "geld_known", "geld_unknown")))

    # 去勢前の成績・休養日数（去勢した馬ごと）
    pre = r[r["gidx"].notna() & (r["idx"] < r["gidx"])]
    last3 = pre.sort_values(["horse_id", "idx"]).groupby("horse_id").tail(3)
    pre3 = last3.groupby("horse_id").agg(pre_fp=("fp", "mean"), pre_win=("win", "mean"), pre_top3=("top3", "mean"), pre_n=("fp", "size"))
    lay = r[r["n_post"] == 1].set_index("horse_id")["gap"].rename("layoff")
    r = r.join(pre3, on="horse_id").join(lay, on="horse_id")

    # 予測・オッズ・払戻（2022 年〜）
    f = pd.read_parquet(FRAME, columns=["race_id", "horse_id", "p_win", "p_top2", "p_top3", "ret_win", "ret_place", "popularity", "win_odds"])
    f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    f = f.drop_duplicates(["race_id", "horse_id"])
    r = r.merge(f, on=["race_id", "horse_id"], how="left")
    return r


# ───────────────────────── 期待値（同条件の平均）
def cell_expect(r: pd.DataFrame, pool: pd.Series, cols=("win", "top2", "top3", "fp")) -> pd.DataFrame:
    """pool の行から、同条件セル（芝ダ×距離帯×クラス×年齢群×休み明け）の平均を作り、全行に当てる。小さいセルは粗いセルへ。"""
    d = r.copy()
    d["agrp"] = d["age"].clip(upper=6)
    levels = [["surface", "dband", "cls", "agrp", "rest"], ["surface", "dband", "cls", "agrp"], ["surface", "dband", "cls"], ["surface", "dband"]]
    out = pd.DataFrame(index=d.index, columns=[f"exp_{c}" for c in cols], dtype=float)
    done = pd.Series(False, index=d.index)
    P = d[pool]
    for keys in levels:
        agg = P.groupby(keys, dropna=False)[list(cols)].agg(["mean", "size"])
        m = agg.xs("mean", axis=1, level=1)
        n = agg.xs("size", axis=1, level=1).iloc[:, 0]
        ok = n[n >= 200].index
        if len(ok) == 0:
            continue
        mm = m.loc[ok].rename(columns={c: f"exp_{c}" for c in cols}).reset_index()
        j = d[keys].merge(mm, on=keys, how="left")
        hit = j[f"exp_{cols[0]}"].notna().to_numpy() & ~done.to_numpy()
        for c in cols:
            out.loc[hit, f"exp_{c}"] = j[f"exp_{c}"].to_numpy()[hit]
        done |= pd.Series(hit, index=d.index)
    return out


# ───────────────────────── ブートストラップ（レース単位）
def _boot_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float32)


def stats(df: pd.DataFrame, cols: dict[str, str], cluster: str = "race_id", seed: int = SEED) -> dict:
    """cols: 出力名 → df の列。各列の（欠損を除く）平均と、cluster 単位のブートストラップ 95% 区間。"""
    if len(df) == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    key = df[cluster].astype(str).to_numpy()
    uniq, inv = np.unique(key, return_inverse=True)
    W = _boot_weights(len(uniq), rng)
    res: dict = {"n": int(len(df)), "clusters": int(len(uniq))}
    for name, c in cols.items():
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        ok = np.isfinite(v)
        S = np.bincount(inv, weights=np.where(ok, v, 0.0), minlength=len(uniq))
        C = np.bincount(inv, weights=ok.astype(float), minlength=len(uniq))
        if C.sum() == 0:
            res[name] = None
            continue
        point = S.sum() / C.sum()
        num, den = W @ S, W @ C
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = num / den
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        res[name] = {"v": float(point), "lo": float(lo), "hi": float(hi), "n": int(C.sum())}
    return res


def add_diffs(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c in ("win", "top2", "top3"):
        d[f"d_{c}"] = d[c] - d[f"exp_{c}"]          # 同条件平均との差
    d["d_fp"] = d["fp"] - d["exp_fp"]
    d["m_win"] = d["win"] - d["p_win"]               # 現行モデルの予測との差（予測より走る=正）
    d["m_top2"] = d["top2"] - d["p_top2"]
    d["m_top3"] = d["top3"] - d["p_top3"]
    d["roi_win"] = d["ret_win"] / 100.0
    d["roi_place"] = d["ret_place"] / 100.0
    return d


BASE_COLS = {"win": "win", "top2": "top2", "top3": "top3", "fp": "fp", "rank": "rank"}
ROI_COLS = {"roi_win": "roi_win", "roi_place": "roi_place"}
DIFF_COLS = {"d_win_vs_cell": "d_win", "d_top3_vs_cell": "d_top3", "d_fp_vs_cell": "d_fp"}
MODEL_COLS = {"m_win": "m_win", "m_top2": "m_top2", "m_top3": "m_top3", "p_win": "p_win", "p_top3": "p_top3"}


def summarize(df: pd.DataFrame, with_model: bool = True) -> dict:
    cols = {**BASE_COLS, **DIFF_COLS}
    out = stats(df, cols)
    if with_model:
        sub = df[df["p_win"].notna()]
        out["model"] = stats(sub, {**{k: v for k, v in BASE_COLS.items() if k in ("win", "top3")}, **MODEL_COLS, **ROI_COLS})
    return out


# ───────────────────────── 本体
def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    r = build_runs()
    r["agrp"] = r["age"].clip(upper=6)
    R: dict = {}
    R["counts"] = {
        "runs_total": int(len(r)), "horses_total": int(r["horse_id"].nunique()),
        "geld_events_known": int(r[r["n_post"] == 1]["horse_id"].nunique()),
        "horses_first_seen_geld_unknown": int(r[r["kind"] == "geld_unknown"]["horse_id"].nunique()),
        "runs_by_N": {g: int((r["ng"] == g).sum()) for g in NGROUPS},
        "sex_back_to_male": int(((r["psex"] == "セ") & (r["sex"] == "牡")).sum()),
    }
    # 同条件期待値: 比較プールは『去勢明け 1〜5 戦目以外の牡 + セ』（自分自身を含めない）
    focus = r["ng"].isin(NGROUPS) & (r["n_post"] <= 5)
    pool_male = (r["kind"] == "male") & (r["year"] <= 2025)
    pool_geld = (r["kind"] == "geld_unknown") | ((r["kind"] == "geld_known") & (r["n_post"] >= 7))
    pool_geld = pool_geld & (r["year"] <= 2025)
    E_m = cell_expect(r, pool_male)
    E_g = cell_expect(r, pool_geld)
    rm = add_diffs(pd.concat([r, E_m], axis=1))
    rg = add_diffs(pd.concat([r, E_g], axis=1))

    main = (r["year"] <= 2025) & (r["kind"] == "geld_known")
    res_n: dict = {}
    for g in NGROUPS + ["2+"]:
        sel = main & (r["ng"] == g) if g != "2+" else main & (r["n_post"] >= 2)
        d_m, d_g = rm[sel], rg[sel]
        row = summarize(d_g)                                   # 比較: 同条件のセ一般
        row["vs_male_cell"] = stats(d_m, DIFF_COLS)
        row["roi_2022_25"] = stats(d_g[d_g["roi_win"].notna()], ROI_COLS)
        res_n[g] = row
    R["by_N"] = res_n
    # 比較プールそのもの（基準線）
    for nm, sel in {"male_pool": pool_male, "geld_pool": pool_geld}.items():
        R[nm] = summarize(rg[sel & (r["year"] >= 2022)] if nm == "geld_pool" else rm[sel & (r["year"] >= 2022)])

    # 2026（記述のみ）
    sel26 = (r["year"] == 2026) & (r["kind"] == "geld_known")
    R["y2026"] = {g: stats(rg[sel26 & (r["ng"] == g)], {**BASE_COLS, **DIFF_COLS}) for g in NGROUPS}

    # 年別（再現性）: N=1 / N=2 / N>=3
    yr: dict = {}
    for lab, mask in {"N=1": r["n_post"] == 1, "N=2": r["n_post"] == 2, "N>=3": r["n_post"] >= 3}.items():
        yr[lab] = {str(y): stats(rg[main & mask & (r["year"] == y)], {"win": "win", "top3": "top3", "d_top3_vs_cell": "d_top3", "d_win_vs_cell": "d_win"}) for y in range(2020, 2026)}
    R["by_year"] = yr
    ym: dict = {}
    for lab, mask in {"N=1": r["n_post"] == 1, "N=2": r["n_post"] == 2, "N>=3": r["n_post"] >= 3}.items():
        ym[lab] = {str(y): stats(rg[(r["kind"] == "geld_known") & mask & (r["year"] == y) & rg["p_win"].notna()], {"m_win": "m_win", "m_top3": "m_top3"}) for y in range(2022, 2027)}
    R["by_year_model"] = ym

    # (a) 同じ馬の去勢前との対応あり比較: 去勢明け N 戦目の『同条件平均との差』 − 去勢前直近 3 走の『同条件平均との差』
    pre = rm[rm["gidx"].notna() & (rm["idx"] < rm["gidx"]) & (rm["year"] <= 2025)]
    pre3 = pre.sort_values(["horse_id", "idx"]).groupby("horse_id").tail(3)
    pre3 = pre3.groupby("horse_id").agg(pre_d_win=("d_win", "mean"), pre_d_top3=("d_top3", "mean"), pre_d_fp=("d_fp", "mean"), pre_cnt=("d_fp", "size"))
    pre3 = pre3[pre3["pre_cnt"] >= 3]
    post = rm[main].join(pre3, on="horse_id")
    post = post[post["pre_d_fp"].notna()]
    paired: dict = {}
    for g in NGROUPS + ["2+"]:
        s = post[post["ng"] == g] if g != "2+" else post[post["n_post"] >= 2]
        s = s.assign(dd_win=s["d_win"] - s["pre_d_win"], dd_top3=s["d_top3"] - s["pre_d_top3"], dd_fp=s["d_fp"] - s["pre_d_fp"])
        paired[g] = stats(s, {"post_d_top3": "d_top3", "pre_d_top3": "pre_d_top3", "diff_top3": "dd_top3", "post_d_fp": "d_fp", "pre_d_fp": "pre_d_fp", "diff_fp": "dd_fp", "diff_win": "dd_win"}, cluster="horse_id")
    R["paired_pre"] = paired

    # 2 戦目は 1 戦目より伸びるか（同じ馬の 1 戦目と 2 戦目）
    a = rg[main & (r["n_post"] == 1)][["horse_id", "d_win", "d_top3", "d_fp"]].rename(columns=lambda c: c if c == "horse_id" else c + "_1")
    b = rg[main & (r["n_post"] == 2)][["horse_id", "d_win", "d_top3", "d_fp", "m_top3", "m_win", "race_id"]].rename(columns=lambda c: c if c in ("horse_id", "race_id") else c + "_2")
    ab = a.merge(b, on="horse_id")
    ab["diff_top3"], ab["diff_fp"], ab["diff_win"] = ab["d_top3_2"] - ab["d_top3_1"], ab["d_fp_2"] - ab["d_fp_1"], ab["d_win_2"] - ab["d_win_1"]
    R["second_vs_first"] = stats(ab, {"n1_top3": "d_top3_1", "n2_top3": "d_top3_2", "diff_top3": "diff_top3", "n1_fp": "d_fp_1", "n2_fp": "d_fp_2", "diff_fp": "diff_fp", "diff_win": "diff_win"}, cluster="horse_id")
    # 1 戦目の出来が 2 戦目に持ち越されるか（1 戦目の fp 差 → 2 戦目の fp 差）
    ok = ab[["d_fp_1", "d_fp_2"]].dropna()
    corr = float(np.corrcoef(ok["d_fp_1"], ok["d_fp_2"])[0, 1]) if len(ok) > 30 else None
    rng = np.random.default_rng(SEED)
    cs = []
    for _ in range(B):
        i = rng.integers(0, len(ok), len(ok))
        cs.append(np.corrcoef(ok["d_fp_1"].to_numpy()[i], ok["d_fp_2"].to_numpy()[i])[0, 1])
    R["carryover_corr_fp"] = {"n": int(len(ok)), "r": corr, "lo": float(np.percentile(cs, 2.5)), "hi": float(np.percentile(cs, 97.5))}
    # ベースライン: 牡馬の連続する 2 走の同条件差の相関（比較）
    mm = rm[(rm["kind"] == "male") & (rm["year"] <= 2025)].sort_values(["horse_id", "idx"])
    nxt = mm.groupby("horse_id")["d_fp"].shift(-1)
    okm = pd.DataFrame({"a": mm["d_fp"], "b": nxt}).dropna().sample(30000, random_state=1)
    R["carryover_corr_fp_male_baseline"] = float(np.corrcoef(okm["a"], okm["b"])[0, 1])

    # 層別（去勢明け 1 戦目 と 2 戦目以降）
    strata: dict = {}
    base = rg[main].copy()
    base["layoff_b"] = pd.cut(base["layoff"], [0, 60, 120, 240, 10000], labels=["~60日", "61~120日", "121~240日", "241日~"]).astype("string")
    q = base["pre_fp"].quantile([1 / 3, 2 / 3]).to_numpy()
    base["pre_b"] = np.where(base["pre_fp"].isna(), None, np.where(base["pre_fp"] <= q[0], "去勢前:好調(上位1/3)", np.where(base["pre_fp"] <= q[1], "去勢前:中位", "去勢前:不振(下位1/3)")))
    base["pop_b"] = pd.cut(base["popularity"], [0, 3, 9, 100], labels=["1~3番人気", "4~9番人気", "10番人気~"]).astype("string")
    base["age_b"] = np.where(base["age"] >= 6, "6歳以上", base["age"].astype("Int64").astype("string") + "歳")
    base["grp"] = np.where(base["n_post"] == 1, "N=1", np.where(base["n_post"] == 2, "N=2", "N>=3"))
    dims = {"surface": "芝ダ", "dband": "距離帯", "age_b": "年齢", "layoff_b": "去勢で休んだ日数", "pre_b": "去勢前の成績", "pop_b": "人気帯"}
    cols = {"win": "win", "top3": "top3", "d_win": "d_win", "d_top3": "d_top3"}
    for dk, lab in dims.items():
        tab = {}
        for val, sub in base.groupby(dk, dropna=True):
            row = {}
            for gname, ss in sub.groupby("grp"):
                if len(ss) < 40:
                    continue
                st = stats(ss, cols)
                st["roi"] = stats(ss[ss["roi_win"].notna()], ROI_COLS)
                st["model"] = stats(ss[ss["p_win"].notna()], {"m_win": "m_win", "m_top3": "m_top3"})
                row[gname] = st
            if row:
                tab[str(val)] = row
        strata[dk] = {"label": lab, "table": tab}
    R["strata"] = strata

    # 簡易の増分評価: 現行モデルの確率に『去勢後 N 戦目』の補正を足すと対数損失が下がるか（2022-24 で当てはめ、2025 で 1 回評価。2026 は記述のみ）
    R["increment"] = increment(rg)

    (out / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o)), encoding="utf-8")
    print("saved", out / "results.json")


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_offset_logit(X: np.ndarray, y: np.ndarray, off: np.ndarray, l2: float = 1.0, it: int = 50) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(it):
        z = off + X @ w
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def increment(rg: pd.DataFrame) -> dict:
    d = rg[rg["p_win"].notna() & rg["p_top3"].notna() & (rg["kind"] != "female")].copy()
    d = d[d["kind"].isin(["male", "geld_known", "geld_unknown"])]
    names = ["N1", "N2", "N3", "N45", "N6p"]
    for nm, g in zip(names, NGROUPS):
        d[nm] = (d["ng"] == g).astype(float)
    # 前走までの去勢後の出来（2 戦目以降の補正）
    out: dict = {}
    for tgt, pcol in (("win", "p_win"), ("top3", "p_top3")):
        out[tgt] = {}
        for vname, feats in (("N1-N6+_5群", names), ("N2のみ", ["N2"]), ("N2-N5(2戦目以降の序盤)", ["N2", "N3", "N45"])):
            tr = d[d["year"].between(2022, 2024)]
            w = _fit_offset_logit(tr[feats].to_numpy(float), tr[tgt].to_numpy(float), _logit(tr[pcol].to_numpy(float)))
            res = {"coef": dict(zip(feats, [float(x) for x in w]))}
            for lab, years in (("test2025", [2025]), ("y2026_descriptive", [2026])):
                te = d[d["year"].isin(years)].copy()
                off = _logit(te[pcol].to_numpy(float))
                z1 = off + te[feats].to_numpy(float) @ w
                y = te[tgt].to_numpy(float)
                ll0 = -(y * np.log(1 / (1 + np.exp(-off))) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-off))))
                ll1 = -(y * np.log(1 / (1 + np.exp(-z1))) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-z1))))
                te["dll"] = ll0 - ll1                      # 正 = 補正で良くなった
                te["geld"] = te["kind"] == "geld_known"
                res[lab] = {"all": stats(te, {"dll": "dll"}), "geld_known_rows": stats(te[te["geld"]], {"dll": "dll"})}
            out[tgt][vname] = res
    return out


def plot(R: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = [n for n in ("Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "Yu Gothic", "Meiryo", "DejaVu Sans") if n in have][:1] or ["DejaVu Sans"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    labs = NGROUPS
    for a, (key, title) in zip(ax, [("d_top3_vs_cell", "複勝率: 同条件のセ馬一般との差"), ("d_win_vs_cell", "勝率: 同条件のセ馬一般との差"), ("m_top3", "複勝率: 現行モデルの予測との差（2022-25）")]):
        vs, lo, hi = [], [], []
        for g in labs:
            row = R["by_N"][g]
            s = row["model"][key] if key.startswith("m_") else row[key]
            vs.append(s["v"] * 100), lo.append(s["v"] * 100 - s["lo"] * 100), hi.append(s["hi"] * 100 - s["v"] * 100)
        a.errorbar(range(len(labs)), vs, yerr=[lo, hi], fmt="o-", capsize=4)
        a.axhline(0, color="gray", lw=0.8)
        a.set_xticks(range(len(labs)))
        a.set_xticklabels([f"{g}戦目" for g in labs])
        a.set_title(title, fontsize=10)
        a.set_ylabel("ポイント")
    fig.tight_layout()
    fig.savefig(out / "geld_by_n.png", dpi=130)


# ───────────────────────── 章の HTML
def _v(s, d=1, signed=False, mult=100):
    if not s:
        return "–"
    f = "{:+." + str(d) + "f}" if signed else "{:." + str(d) + "f}"
    return f.format(s["v"] * mult)


def _ci(s, d=1, signed=False, mult=100):
    if not s:
        return "–"
    f = "{:+." + str(d) + "f}" if signed else "{:." + str(d) + "f}"
    return f"{f.format(s['v'] * mult)} <span class='muted'>[{f.format(s['lo'] * mult)}, {f.format(s['hi'] * mult)}]</span>"


def chapter(out: Path, dest: Path) -> None:
    import base64
    import re

    R = json.loads((out / "results.json").read_text(encoding="utf-8"))
    css = re.search(r"<style>.*?</style>", (ROOT / "docs/html/modeling/chapters/equipment_and_change.html").read_text(encoding="utf-8"), re.S).group(0)
    png = base64.b64encode((out / "geld_by_n.png").read_bytes()).decode()
    C, BN = R["counts"], R["by_N"]
    lab = lambda g: "2 戦目以降（合計）" if g == "2+" else f"去勢明け {g} 戦目"

    t_main = "".join(
        f"<tr><td>{lab(g)}</td><td class='num'>{BN[g]['n']:,}</td><td class='num'>{_v(BN[g]['win'])}</td><td class='num'>{_v(BN[g]['top2'])}</td><td class='num'>{_v(BN[g]['top3'])}</td>"
        f"<td class='num'>{BN[g]['rank']['v']:.2f}</td><td class='num'>{_ci(BN[g]['d_win_vs_cell'], 1, True)}</td><td class='num'>{_ci(BN[g]['d_top3_vs_cell'], 1, True)}</td>"
        f"<td class='num'>{_ci(BN[g]['vs_male_cell']['d_top3_vs_cell'], 1, True)}</td></tr>" for g in NGROUPS + ["2+"])
    t_model = "".join(
        f"<tr><td>{lab(g)}</td><td class='num'>{(BN[g].get('model') or {}).get('n', 0):,}</td><td class='num'>{_ci(BN[g]['model'].get('m_win'), 1, True)}</td><td class='num'>{_ci(BN[g]['model'].get('m_top3'), 1, True)}</td>"
        f"<td class='num'>{_ci(BN[g]['model'].get('roi_win'), 0)}</td><td class='num'>{_ci(BN[g]['model'].get('roi_place'), 0)}</td></tr>" for g in NGROUPS + ["2+"])
    mp, gp = R["male_pool"]["model"], R["geld_pool"]["model"]
    t_base = (f"<tr><td>牡馬一般（同期間 2022–25）</td><td class='num'>{R['male_pool']['n']:,}</td><td class='num'>{_ci(mp.get('m_win'), 1, True)}</td><td class='num'>{_ci(mp.get('m_top3'), 1, True)}</td><td class='num'>{_ci(mp.get('roi_win'), 0)}</td><td class='num'>{_ci(mp.get('roi_place'), 0)}</td></tr>"
              f"<tr><td>セ馬一般（去勢時点不明＋去勢明け 7 戦目以降）</td><td class='num'>{R['geld_pool']['n']:,}</td><td class='num'>{_ci(gp.get('m_win'), 1, True)}</td><td class='num'>{_ci(gp.get('m_top3'), 1, True)}</td><td class='num'>{_ci(gp.get('roi_win'), 0)}</td><td class='num'>{_ci(gp.get('roi_place'), 0)}</td></tr>")
    t_pair = "".join(
        f"<tr><td>{lab(g)}</td><td class='num'>{R['paired_pre'][g]['n']:,}</td><td class='num'>{_ci(R['paired_pre'][g]['pre_d_top3'], 1, True)}</td><td class='num'>{_ci(R['paired_pre'][g]['post_d_top3'], 1, True)}</td>"
        f"<td class='num'>{_ci(R['paired_pre'][g]['diff_top3'], 1, True)}</td><td class='num'>{_ci(R['paired_pre'][g]['diff_fp'], 1, True)}</td></tr>" for g in NGROUPS + ["2+"])
    sv = R["second_vs_first"]
    t_year = ""
    for labn in ("N=1", "N=2", "N>=3"):
        cells = "".join(f"<td class='num'>{_v(R['by_year'][labn][str(y)].get('d_top3_vs_cell'), 1, True) if R['by_year'][labn][str(y)].get('n') else '–'}<br><span class='muted'>n={R['by_year'][labn][str(y)].get('n', 0)}</span></td>" for y in range(2020, 2026))
        cm = "".join(f"<td class='num'>{_v((R['by_year_model'][labn].get(str(y)) or {}).get('m_top3'), 1, True) if (R['by_year_model'][labn].get(str(y)) or {}).get('n') else '–'}<br><span class='muted'>n={(R['by_year_model'][labn].get(str(y)) or {}).get('n', 0)}</span></td>" for y in range(2022, 2027))
        t_year += f"<tr><td>{labn}</td>{cells}<td class='num'>{_v(R['y2026'][{'N=1': '1', 'N=2': '2', 'N>=3': '3'}[labn]].get('d_top3_vs_cell'), 1, True) if labn != 'N>=3' else '–'}</td></tr>"
        t_year += f"<tr class='muted'><td>{labn}（現行モデルとの差）</td><td></td><td></td>{cm}</tr>"
    t_strata = ""
    for dk, d in R["strata"].items():
        t_strata += f"<tr><th colspan='10'>{d['label']}</th></tr>"
        for val, row in d["table"].items():
            cells = ""
            for g in ("N=1", "N=2", "N>=3"):
                sx = row.get(g)
                if not sx:
                    cells += "<td colspan='3' class='muted'>n&lt;40</td>"
                    continue
                mm = sx["model"]
                cells += f"<td class='num'>{sx['n']}</td><td class='num'>{_v(sx['top3'])}</td><td class='num'>{_v(mm.get('m_top3'), 1, True) if mm.get('n') else '–'}</td>"
            t_strata += f"<tr><td>{val}</td>{cells}</tr>"
    t_inc = ""
    for tgt, nm in (("win", "勝率"), ("top3", "複勝率")):
        for vn, r in R["increment"][tgt].items():
            a, b = r["test2025"], r["y2026_descriptive"]
            e4 = lambda s: _ci(s, 2, True, 1e4)
            t_inc += f"<tr><td>{nm}</td><td>{vn}</td><td class='num'>{e4(a['all']['dll'])}</td><td class='num'>{e4(a['geld_known_rows']['dll'])}</td><td class='num'>{e4(b['all']['dll'])}</td></tr>"
    carry, base = R["carryover_corr_fp"], R["carryover_corr_fp_male_baseline"]

    body = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>去勢明け 2 戦目以降の成績（tp-32 GELD の拡張）— keiba-vpn</title>{css}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>去勢明け 2 戦目以降の成績（tp-32 GELD の拡張）</h1>
<p class="muted">tp-32 の GELD は『去勢明け初戦』だけを扱った（特徴量としては不採用）。ここでは去勢明けの 1・2・3・4–5・6+ 戦目ごとに、成績・回収率・現行モデルの予測とのずれ・去勢前との対応あり比較・層別を測り、特徴量化の見込みを簡易に評価する。再現: <code>python -m src.scripts.data.geld_analysis run</code> → <code>chapter</code>。</p>
<div class="note"><b>要点.</b>
<ul>
<li><b>初戦は明確に弱く、2 戦目から戻る。</b> 同条件のセ馬一般との複勝率の差は 1 戦目 {_v(BN['1']['d_top3_vs_cell'], 1, True)} → 2 戦目 {_v(BN['2']['d_top3_vs_cell'], 1, True)} → 3 戦目 {_v(BN['3']['d_top3_vs_cell'], 1, True)} → 4–5 戦目 {_v(BN['4-5']['d_top3_vs_cell'], 1, True)} → 6+ 戦目 {_v(BN['6+']['d_top3_vs_cell'], 1, True)}（ポイント）。同じ馬の 1 戦目→2 戦目でも複勝率の差が {_ci(sv['diff_top3'], 1, True)} 改善する。</li>
<li><b>現行モデルは初戦は少し過大評価、2 戦目は過小評価。</b> モデルの複勝率との差（2022–25）は 1 戦目 {_v(BN['1']['model']['m_top3'], 1, True)}・2 戦目 {_v(BN['2']['model']['m_top3'], 1, True)}。2 戦目の過小評価は 2023–2026 の各年で +5 前後（2026 年は記述のみ）で向きが揃うが、年ごとの区間は 0 をまたぐ。</li>
<li><b>回収率に優位はない。</b> 去勢明け 1 戦目の単勝回収率は {_ci(BN['1']['model']['roi_win'], 0)}%・複勝 {_ci(BN['1']['model']['roi_place'], 0)}%（基準の牡馬一般 {_v(mp.get('roi_win'), 0)}%・{_v(mp.get('roi_place'), 0)}%より低い、買わない側の根拠）。2 戦目以降の合計は単勝 {_ci(BN['2+']['model']['roi_win'], 0)}%・複勝 {_ci(BN['2+']['model']['roi_place'], 0)}% で 100% を下回る（3 戦目の単勝 {_v(BN['3']['model']['roi_win'], 0)}% のように点推定が 100% を超える戦目もあるが、区間が [{BN['3']['model']['roi_win']['lo']*100:.0f}, {BN['3']['model']['roi_win']['hi']*100:.0f}] と広く有意でない）。</li>
<li><b>特徴量化の増分は小さい。</b> 現行モデルの確率に『去勢後 N 戦目』の補正を足しても、2022–24 で当てはめ test 2025 で評価した対数損失は ±0.0002 以内で区間が 0 をまたぐ（採用基準に届かない）。既に <code>gd_n_since</code> 等の候補列があり、新規列を作る価値は薄い。</li>
</ul></div>

<h2>1. 定義とデータ</h2>
<ul>
<li>対象: 2020–2026 年の芝・ダート（障害を除く）{C['runs_total']:,} 走・{C['horses_total']:,} 頭。</li>
<li><b>去勢明け 1 戦目</b> = 直前の出走が牡で今回がセの最初の出走（{C['geld_events_known']:,} 頭）。以降を 2, 3, … 戦目。性別の表記は牡・牝・セの 3 種だけで揺れはなく、セ→牡の戻りは {C['sex_back_to_male']} 件。</li>
<li><b>最初に見えた出走が既にセの馬（{C['horses_first_seen_geld_unknown']:,} 頭）</b>は去勢の時点が分からない（2020 年より前に去勢）ので、N 別の集計から除いて『セ馬一般』の比較対象にだけ使う。</li>
<li>N 別の走数: {', '.join(f'{g} 戦目 {C["runs_by_N"][g]:,}' for g in NGROUPS)}（2020–26 年の合計）。この章の N 別の表は 2020–2025 年（回収率とモデルとの差は予測のある 2022–2025 年）。2026 年は記述統計のみ。</li>
<li>『同条件』= 芝ダ × 距離帯（1400/1700/2000/2500 で 5 区分）× クラス × 年齢（6+ 区分）× 休み明け（間隔 84 日以上）の平均（200 走未満のセルは粗いセルへ）。比較プールは 2025 年までの牡馬一般 / セ馬一般（自分自身と去勢明け 1–5 戦目を含めない）。</li>
<li>現行モデルの勝率・連対率・複勝率は <code>betting/tp36/frame_all.parquet</code>（2022 年〜、ホールドアウトの予測）。市場オッズは回収率の計算にだけ使い、特徴量にはしない。回収率は『その馬の単勝（複勝）を 100 円ずつ買った場合』で、区間はレース単位のブートストラップ 1,000 回の 95%。</li>
</ul>

<h2>2. 去勢明け N 戦目の成績</h2>
<img alt="去勢明け N 戦目の成績" style="max-width:100%" src="data:image/png;base64,{png}" />
<table><tr><th>区分</th><th>走数</th><th>勝率%</th><th>連対率%</th><th>複勝率%</th><th>平均着順</th><th>勝率 − 同条件セ馬一般（pt）</th><th>複勝率 − 同条件セ馬一般（pt）</th><th>複勝率 − 同条件牡馬（pt）</th></tr>{t_main}</table>
<p class="muted">平均着順・勝率・連対率・複勝率は 2020–2025 年。差はポイントで、[ ] はレース単位ブートストラップの 95% 区間。</p>

<h3>現行モデルの予測との差と回収率（2022–2025 年）</h3>
<table><tr><th>区分</th><th>走数</th><th>勝率 − 予測（pt）</th><th>複勝率 − 予測（pt）</th><th>単勝回収率%</th><th>複勝回収率%</th></tr>{t_model}{t_base}</table>
<p class="muted">正 = 予測より走る（モデルが過小評価）。回収率は 100 円ずつ全頭買った場合。</p>

<h2>3. 去勢前の同じ馬との対応あり比較</h2>
<p>去勢前の直近 3 走（3 走以上ある馬）の『同条件牡馬との差』と、去勢明け N 戦目の同じ差を、馬ごとに対にして比べた（区間は馬単位）。去勢は不振のあとに行われやすいので、去勢前の水準は低く出る（平均への回帰を含む）。</p>
<table><tr><th>区分</th><th>馬数</th><th>去勢前 3 走：複勝率 − 同条件牡馬（pt）</th><th>去勢明け N 戦目：同</th><th>差（後 − 前）</th><th>着順の相対位置（fp）の差（pt、負 = 良化）</th></tr>{t_pair}</table>
<p><b>2 戦目は 1 戦目より伸びるか。</b> 同じ馬の 1 戦目と 2 戦目（{sv['n']:,} 頭）で、同条件セ馬一般との複勝率の差は 1 戦目 {_ci(sv['n1_top3'], 1, True)} → 2 戦目 {_ci(sv['n2_top3'], 1, True)}、改善幅 {_ci(sv['diff_top3'], 1, True)}、着順の相対位置の差は {_ci(sv['diff_fp'], 1, True)}（負 = 良化）。1 戦目の出来が 2 戦目に持ち越される相関は {carry['r']:.2f}（区間 {carry['lo']:.2f}〜{carry['hi']:.2f}）で、牡馬の連続 2 走の相関 {base:.2f} と同程度 — <b>特別な持ち越しはなく、初戦の反動（休養明けの重さ）が 2 戦目で抜ける</b>と読むのが自然。</p>

<h2>4. 年別の再現性と 2026 年</h2>
<table><tr><th>区分</th><th>2020</th><th>2021</th><th>2022</th><th>2023</th><th>2024</th><th>2025</th><th>2026（記述のみ）</th></tr>{t_year}</table>
<p class="muted">上段 = 複勝率 − 同条件セ馬一般（pt、2020–25）、下段（薄い行）= 複勝率 − 現行モデル（pt、2022–2026 の各年。2022 → 2026 の順で右詰め）。2026 年は選択に使っていない。</p>
<ul>
<li>1 戦目の弱さは全年で負（2025 年は −3.2 と縮小）。2 戦目は年とともに −4〜−5（2020–22）→ +1〜+3（2024–25）と上がる傾向があるが、区間は広い。</li>
<li>現行モデルとの差は、1 戦目が 2022–24 年で負・2025 年に正へ（−4.2/−5.3/−2.2/+2.0）、2 戦目が 2023–2026 年の 4 年とも約 +5（区間下限は 0 付近）。3 戦目以降は 0 付近で、この年別にも傾向は見えない。</li>
</ul>

<h2>5. 層別（去勢明け 1 戦目／2 戦目／3 戦目以降）</h2>
<table><tr><th rowspan='2'>区分</th><th colspan='3'>1 戦目</th><th colspan='3'>2 戦目</th><th colspan='3'>3 戦目以降</th></tr><tr><th>走数</th><th>複勝率%</th><th>対モデル pt</th><th>走数</th><th>複勝率%</th><th>対モデル pt</th><th>走数</th><th>複勝率%</th><th>対モデル pt</th></tr>{t_strata}</table>
<p class="muted">対モデル = 複勝率 − 現行モデルの予測（2022–25）。人気帯は 2022 年以降のみ。区間は省略（本文の主要な数値だけ区間つき）。層別は検定を多数行っているので、1 つ 1 つは仮説として読む。</p>
<ul>
<li>1 戦目の弱さは、芝ダ・距離・年齢・休養の長さ・去勢前の成績のどの層でも負で、<b>去勢前が不振（下位 1/3）の 1 戦目は複勝率 6.7%（同条件より −12 pt）</b>と特に低い。逆に去勢前が好調だった馬の 1 戦目は同条件並み（+2.6）。</li>
<li>2 戦目の過小評価（対モデル +5 pt 前後）は、ダート・短距離（S: +7.9）・3 歳・休養 61–120 日 / 241 日〜・4–9 番人気で大きい。6 歳以上と芝では見えない（芝 +1.1、6 歳以上 +0.6）。</li>
<li>去勢前が好調の馬は 2 戦目に複勝率 26.1%（同条件より +6.7 pt）。去勢が『成功した』馬は 2 戦目から結果が出る。</li>
</ul>

<h2>6. 特徴量化の見込み（簡易）</h2>
<p>現行モデルの確率（logit）をオフセットにして、去勢明け N 戦目のダミーを足すロジスティック補正を、2022–2024 年で当てはめ、2025 年（test、1 回）で対数損失の改善（正 = 良化、×10⁻⁴）を見た。2026 年は記述のみ。</p>
<table><tr><th>目的</th><th>補正の形</th><th>test 2025：全走</th><th>test 2025：去勢明けの走のみ</th><th>2026（記述）：全走</th></tr>{t_inc}</table>
<ul>
<li>全走の改善は 10⁻⁴ のオーダーで区間は 0 をまたぎ、tp-32 の採用基準（Δ勝率の CI 上限 &lt; 0）に届かない。去勢明けの走は全体の約 2% しかなく、2 戦目の過小評価（+5 pt）が本物でも、全体の損失への寄与は小さい。</li>
<li>2 戦目だけの補正（N2 のみ）が複勝で最も良い（+0.78×10⁻⁴、区間 [−0.21, +1.95]）が、採用の根拠にはならない。1 戦目の補正（負）を入れると 2025 年は悪化する（2025 年は 1 戦目が +2.0 とモデルを上回ったため）。</li>
<li>見込み: <b>V5/V6 への新規特徴量の増分は ≤ 0.0001 と見積もる。独立の特徴量群としては作らない。</b>代わりに、(a) 馬券・表示側では『去勢明け 1 戦目』を減点要因として扱う（回収率が低い）、(b) 『去勢明け 2 戦目』は単独では買い材料にならないが、モデルの確率が過小になりやすい層（短距離ダート・3 歳・4–9 番人気）の監視対象にする。</li>
</ul>

<h2>7. 結論と注意</h2>
<ul>
<li>去勢明けは 1 戦目が最も弱く、2 戦目から戻り、3–5 戦目は同条件のセ馬一般を少し上回り、6 戦目以降は一般のセ馬と同じ。『2 戦目以降に特有の持ち越し』は見えず、初戦の反動が抜けるだけと読める。</li>
<li>現行モデルは 1 戦目をやや過大評価、2 戦目をやや過小評価（約 +5 pt、年をまたいで同じ向きだが区間は広い）。3 戦目以降のずれは小さい。</li>
<li>回収率は 1 戦目が低い（単勝 {_v(BN['1']['model']['roi_win'], 0)}%・複勝 {_v(BN['1']['model']['roi_place'], 0)}%、区間の上限も 100% 未満）。2 戦目以降は合計で 100% 未満で、区間の下限が 100% を超える層はない（層別で単勝回収率の高い層は走数 100 前後でばらつきが大きい）。</li>
<li>注意: 去勢は不振のあとに行われるため、去勢前との比較には平均への回帰が含まれる。去勢の時点が分かるのは 2020 年以降に牡として走った馬だけ（旧来のセ馬は『時点不明』）。層別は多重比較。</li>
</ul>
<p class="muted">関連: <a href="equipment_and_change.html">馬具と『初めて・替わり』系の状況特徴（tp-32）</a> / <a href="training_pipeline_decisions.html">決定ログ</a></p>
</body></html>
"""
    dest.write_text(body, encoding="utf-8")
    print("saved", dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "chapter", "plot"])
    ap.add_argument("--out", default=os.environ.get("GELD_OUT", str(ROOT / "data/local/modeling/geld")))
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "plot":
        plot(json.loads((out / "results.json").read_text(encoding="utf-8")), out)
    elif a.cmd == "chapter":
        chapter(out, ROOT / "docs/html/modeling/chapters/geld_after_races.html")
    else:
        run(out)


if __name__ == "__main__":
    main()

"""tp-53: 想定位置取り・追走難度・想定上がりの連動を、結果から読める関係と比べる分析。

python -m src.scripts.data.tp53_analysis   # ホストで実行（tp53_dataset の出力を読む）。出力 data/local/modeling/tp53/analysis.json と config 案
A. 2 つのペース定義（位置取り側 PF / 追走側 TD）の一致と、実測ペースとの関係
B. 結果から読み取れる関係（位置取り × 上がり、ペース別）と、予測値どうしの関係
C. 個別の予測精度（位置取り・上がり）と、連動の誤差
D. 整合層（PF の位置取り + PF ペース + 自己の過去上がり偏差 → 上がり偏差）を walk-forward で検証し、最終の係数を作る
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/local/modeling/tp53"
CONFIG_OUT = D / "pace_coherence.json"


def _wcorr(h: pd.DataFrame, a: str, b: str) -> tuple[float, int]:
    x = h[[a, b, "race_id"]].dropna()
    A = x[a] - x.groupby("race_id")[a].transform("mean")
    B = x[b] - x.groupby("race_id")[b].transform("mean")
    return round(float(np.corrcoef(A, B)[0, 1]), 3), int(len(x))


def _slope(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    return round(float(np.polyfit(x[m], y[m], 1)[0]), 3) if m.sum() > 500 else float("nan")


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    h = pd.read_parquet(D / "horse.parquet")
    r = pd.read_parquet(D / "race.parquet")
    r["date"] = pd.to_datetime(r["date"])
    g = r.groupby(["venue", "surface", "distance", "going_group"])["act_t3f"]
    r["act_z"] = (r["act_t3f"] - g.transform("median")) / g.transform("std")
    r.loc[g.transform("count") < 30, "act_z"] = np.nan
    r["act_pace"] = pd.cut(r["act_z"], [-99, -0.5, 0.5, 99], labels=["ハイ", "ミドル", "スロー"]).astype(str)
    r["pf_lab"] = r["pf_level"].map({"high": "ハイ", "mid": "ミドル", "slow": "スロー", "unknown": "不明"})
    r["dist_band"] = pd.cut(r["distance"], [0, 1300, 1700, 2100, 2600, 9999], labels=["~1200", "1400-1600", "1800-2000", "2200-2500", "2600~"]).astype(str)
    h["date"] = pd.to_datetime(h["date"])
    h = h.merge(r[["race_id", "act_z", "act_pace", "going_group", "td_pace_type", "pf_lab", "td_l3f_base", "dist_band", "pf_front", "pf_stalker",
                   "pf_pressure", "pf_known", "td_pace_index"]], on="race_id", how="left")
    h["surf"] = (h["surface"] == "芝").astype(float)
    h = h.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    # 自己の過去の上がり偏差（直近 5 走の平均。レース日より前だけ: 同じ馬の 1 つ前までを shift）
    h["prior_l3f_dev"] = h.groupby("horse_id")["act_l3f_dev"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    h["prior_early"] = h.groupby("horse_id")["act_early_norm"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    for c in ["pf_early", "td_early", "td_l3f", "act_l3f", "act_early_norm", "prior_l3f_dev", "td_stamina", "td_ease"]:
        h[c + "_d"] = h[c] - h.groupby("race_id")[c].transform("mean")
    return h, r


def part_a(r: pd.DataFrame) -> dict:
    out: dict = {}
    m = r[(r.pf_lab != "不明") & r.td_pace_type.notna() & r.pf_lab.notna()]
    out["pf_x_td"] = pd.crosstab(r["pf_lab"], r["td_pace_type"]).to_dict()
    out["pf_td_agree"] = round(float((m.pf_lab == m.td_pace_type).mean()), 3)
    out["td_mid_share"] = round(float((r.td_pace_type == "ミドル").mean()), 3)
    a = r[r.act_pace.isin(["ハイ", "ミドル", "スロー"])]
    for nm, col in (("PF", "pf_lab"), ("TD", "td_pace_type")):
        x = a[a[col].notna() & (a[col] != "不明")]
        out[f"{nm}_acc"] = round(float((x[col] == x["act_pace"]).mean()), 3)
        out[f"{nm}_x_actual"] = pd.crosstab(x[col], x["act_pace"]).to_dict()
    cor = {}
    for c in ["pf_pressure", "pf_front", "pf_stalker", "td_pace_index", "td_pressure", "td_front", "td_stalker", "td_first3f"]:
        x = r[[c, "act_z"]].dropna()
        cor[c] = round(float(np.corrcoef(x[c], x["act_z"])[0, 1]), 3)
    out["corr_actual_z"] = cor
    by_front = {}
    for k, gr in r.groupby(r.pf_front.fillna(-1).clip(upper=3)):
        by_front[int(k)] = {"races": int(len(gr)), "mean_act_z": round(float(gr.act_z.mean()), 3), "slow_share": round(float((gr.pf_level == "slow").mean()), 3)}
    out["pf_front_vs_actual_z"] = by_front
    r = r.copy()
    r["td_n"] = r[["td_n_nige", "td_n_senko", "td_n_sashi", "td_n_oikomi"]].sum(axis=1)
    out["td_sashi_ge75"] = round(float(((r.td_n_sashi / r.td_n) >= 0.75).mean()), 3)
    out["td_no_front_no_stalker"] = round(float(((r.td_n_nige == 0) & (r.td_n_senko == 0)).mean()), 3)
    out["mean_front_plus_stalker"] = {"PF": round(float((r.pf_front + r.pf_stalker).mean()), 2), "TD": round(float((r.td_n_nige + r.td_n_senko).mean()), 2)}
    x = r[["pf_front", "pf_stalker", "td_n_nige", "td_n_senko"]].dropna()
    out["corr_front_stalker_counts"] = round(float(np.corrcoef(x.pf_front + x.pf_stalker, x.td_n_nige + x.td_n_senko)[0, 1]), 3)
    return out


def part_b_c(h: pd.DataFrame, r: pd.DataFrame) -> dict:
    out: dict = {}
    acc = {
        "td_early~act_early": _wcorr(h, "td_early", "act_early_norm"),
        "pf_early~act_early": _wcorr(h, "pf_early", "act_early_norm"),
        "td_early~pf_early": _wcorr(h, "td_early", "pf_early"),
        "td_l3f~act_l3f": _wcorr(h, "td_l3f", "act_l3f"),
        "td_stamina~act_l3f": _wcorr(h, "td_stamina", "act_l3f"),
        "td_ease~act_l3f": _wcorr(h, "td_ease", "act_l3f"),
        "td_stamina~act_early": _wcorr(h, "td_stamina", "act_early_norm"),
        "td_ease~act_early": _wcorr(h, "td_ease", "act_early_norm"),
        "prior_l3f_dev~act_l3f": _wcorr(h, "prior_l3f_dev", "act_l3f"),
    }
    link = {
        "td_l3f~td_early": _wcorr(h, "td_l3f", "td_early"),
        "td_l3f~pf_early": _wcorr(h, "td_l3f", "pf_early"),
        "td_l3f~td_stamina": _wcorr(h, "td_l3f", "td_stamina"),
        "td_early~td_stamina": _wcorr(h, "td_early", "td_stamina"),
        "td_early~td_ease": _wcorr(h, "td_early", "td_ease"),
        "act_l3f~act_early": _wcorr(h, "act_l3f", "act_early_norm"),
    }
    out["accuracy"] = acc
    out["linkage"] = link
    sl = {}
    h = h.copy()
    h["act_l3f_dd"] = h["act_l3f_d"]
    for k, gr in h.groupby("act_pace"):
        if k in ("ハイ", "ミドル", "スロー"):
            sl[f"実測(ペース={k})"] = _slope(gr["act_early_norm_d"], gr["act_l3f_dd"])
            sl[f"TD予測(実測ペース={k})"] = _slope(gr["td_early_d"], gr["td_l3f_d"])
    for k, gr in h.groupby("td_pace_type"):
        sl[f"TD予測(TDペース型={k})"] = _slope(gr["td_early_d"], gr["td_l3f_d"])
    sl["実測 全体"] = _slope(h["act_early_norm_d"], h["act_l3f_dd"])
    sl["TD予測 全体"] = _slope(h["td_early_d"], h["td_l3f_d"])
    sl["TD想定上がり ~ PF位置取り"] = _slope(h["pf_early_d"], h["td_l3f_d"])
    out["slopes"] = sl
    out["sd"] = {"act_l3f_dev": round(float(h.act_l3f_dd.std()), 3), "td_l3f_dev": round(float(h.td_l3f_d.std()), 3)}
    rr = r.merge(h.groupby("race_id")["act_l3f"].median().rename("act_med_l3f"), on="race_id").dropna(subset=["td_l3f_base", "act_med_l3f"])
    out["baseline"] = {
        "bias": round(float((rr.td_l3f_base - rr.act_med_l3f).mean()), 2), "sd": round(float((rr.td_l3f_base - rr.act_med_l3f).std()), 2),
        "corr": round(float(np.corrcoef(rr.td_l3f_base, rr.act_med_l3f)[0, 1]), 3),
        "bias_by_surface": {s: round(float((g.td_l3f_base - g.act_med_l3f).mean()), 2) for s, g in rr.groupby("surface")},
    }
    out["td_style_share"] = h.td_style.value_counts(normalize=True).round(3).to_dict()
    out["pf_style_share"] = h.pf_style_raw.value_counts(normalize=True).round(3).to_dict()
    out["td_l3f_n0_share"] = round(float((h.td_l3f_n.fillna(0) == 0).mean()), 3)
    return out


from src.pipeline.inference import pace_coherence as pc   # noqa: E402

TABLES = ROOT / "data/page_reference/tables"
L3F_FEATS = ["e", "e_z", "e_dist", "e_surf", "prior"]


def _pace_X(r: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=r.index)
    x["pf_front"] = r["pf_front"].clip(upper=5)
    x["pf_stalker"] = r["pf_stalker"].clip(upper=8)
    x["pf_pressure"] = r["pf_pressure"]
    x["surf"] = (r["surface"] == "芝").astype(float)
    x["dist_k"] = (r["distance"] - 1600) / 1000.0
    x["wet"] = (r["going_group"] == "道悪").astype(float)
    return x.fillna(0.0)


def cohort_tables(h: pd.DataFrame, r: pd.DataFrame) -> dict:
    """同条件（場×面×距離×馬場）の上がり中央値（馬単位）と、前半 3F の中央値・標準偏差（レース単位）。件数が少ない条件は入れない。"""
    k = ["venue", "surface", "distance", "going_group"]
    x = h.dropna(subset=["act_l3f"])
    g = x.groupby(k)["act_l3f"].agg(["median", "count"])
    cohort_l3f = {pc.cohort_key(a, b, c, "良" if d == "良" else "x"): float(v) for (a, b, c, d), v in g[g["count"] >= 200]["median"].items()}
    g2 = x.groupby(["surface", "distance", "going_group"])["act_l3f"].agg(["median", "count"])
    fallback = {f"{a}|{pc._dist(b)}|{c}": float(v) for (a, b, c), v in g2[g2["count"] >= 200]["median"].items()}
    y = r.dropna(subset=["act_t3f"])
    gt = y.groupby(k)["act_t3f"].agg(["median", "std", "count"])
    cohort_t3f = {pc.cohort_key(a, b, c, "良" if d == "良" else "x"): {"median": float(m), "std": float(sd)} for (a, b, c, d), (m, sd, n) in gt.iterrows() if n >= 30 and sd > 0}
    return {"cohort_l3f": cohort_l3f, "cohort_l3f_fallback": fallback, "cohort_t3f": cohort_t3f}


def _gkey(venue, surface, distance, going_group) -> str:
    return pc.cohort_key(venue, surface, distance, "良" if going_group == "良" else "x")


def z_from_tables(r: pd.DataFrame, tab: dict) -> pd.Series:
    def f(row):
        t = tab["cohort_t3f"].get(_gkey(row["venue"], row["surface"], row["distance"], row["going_group"]))
        if not t or pd.isna(row["act_t3f"]):
            return np.nan
        return (row["act_t3f"] - t["median"]) / t["std"]
    return r.apply(f, axis=1)


def load_runs() -> pd.DataFrame:
    """各馬の各レースについて、馬柱の過去走（対象日より前）を解析した一覧。キー (race_id, horse_number) → runs。"""
    parts = []
    for y in range(2020, 2026):
        p = pd.read_parquet(TABLES / str(y) / "race_shutuba_past_flat.parquet", columns=["race_id", "date", "horse_number", "past_races"])
        parts.append(p)
    p = pd.concat(parts, ignore_index=True)
    p = p[p.horse_number.notna()].copy()
    p["horse_number"] = p["horse_number"].astype(int)

    def parse(js, date):
        try:
            lst = json.loads(js) if isinstance(js, str) else []
        except Exception:
            return []
        rd = str(date).replace("-", "")
        runs = []
        for q in lst:
            r = pc.parse_past_run(str((q or {}).get("raw") or ""))
            if r and r["date"] < rd:
                runs.append(r)
        runs.sort(key=lambda r: r["date"], reverse=True)
        return runs[:5]

    p["runs"] = [parse(a, b) for a, b in zip(p["past_races"], p["date"])]
    return p[["race_id", "horse_number", "runs"]]


def prior_from_runs(runs_df: pd.DataFrame, tab: dict) -> pd.DataFrame:
    cfg = {"cohort_l3f": tab["cohort_l3f"], "cohort_l3f_fallback": tab["cohort_l3f_fallback"]}
    vals = []
    for runs in runs_df["runs"]:
        devs = []
        for r in runs:
            med = pc.cohort_l3f_median(cfg, r["venue"], r["surface"], r["distance"], r["going"])
            if med is not None:
                devs.append(r["l3f"] - med)
        vals.append(sum(devs) / len(devs) if devs else np.nan)
    out = runs_df[["race_id", "horse_number"]].copy()
    out["prior_c"] = vals
    return out


def _thresholds(rtr: pd.DataFrame, zhat_tr: pd.Series) -> tuple[float, float]:
    """予測 z の閾値を、学習データの実測クラスの比率に合わせた分位で決める（ハイ=速い側の比率、スロー=遅い側の比率）。"""
    a = pd.cut(rtr["act_z_f" if "act_z_f" in rtr else "act_z_y"], [-99, -0.5, 0.5, 99], labels=["ハイ", "ミドル", "スロー"]).astype(str)
    q_fast = float((a == "ハイ").mean())
    q_slow = float((a == "スロー").mean())
    tf, ts = np.quantile(zhat_tr, [q_fast, 1.0 - q_slow])
    return float(tf), float(ts)


def _l3f_X(d: pd.DataFrame, z: pd.Series, kind: str = "full") -> pd.DataFrame:
    e = d["pf_early_d"]
    x = pd.DataFrame(index=d.index)
    zc = z.clip(-2, 2)
    if kind in ("full", "coupling"):
        x["e"] = e
        x["e_z"] = e * zc
        x["e_dist"] = e * (d["distance"] - 1600) / 1000.0
        x["e_surf"] = e * d["surf"]
    if kind in ("full", "prior"):
        x["prior"] = d["prior_d"]
    return x


def part_d(h: pd.DataFrame, r: pd.DataFrame) -> tuple[dict, dict]:
    out: dict = {"walk_forward": {}, "ablation": {}, "baseline": {}}
    runs = load_runs()
    h = h.merge(runs, on=["race_id", "horse_number"], how="left")
    h["runs"] = h["runs"].map(lambda x: x if isinstance(x, list) else [])
    base_cols = ["race_id", "horse_number"]
    for y in (2023, 2024, 2025):
        tab = cohort_tables(h[h.year < y], r[r.year < y])
        d = h.copy()
        pr = prior_from_runs(d[base_cols + ["runs"]], tab)
        d = d.merge(pr, on=base_cols, how="left")
        d["prior_d"] = (d["prior_c"] - d.groupby("race_id")["prior_c"].transform("mean")).fillna(0.0)
        rr = r.copy()
        rr["act_z_y"] = z_from_tables(rr, tab)
        # ペース推定（PF の先行勢の数 → z）。z は学習年より前の表で標準化
        pm_tr = rr[(rr.year < y)].dropna(subset=["act_z_y", "pf_front", "pf_stalker", "pf_pressure"])
        pm = Ridge(alpha=5.0).fit(_pace_X(pm_tr), pm_tr["act_z_y"])
        rr["zhat"] = pm.predict(_pace_X(rr.fillna({"pf_front": 0, "pf_stalker": 0, "pf_pressure": 0})))
        d["zhat"] = d["race_id"].map(rr.set_index("race_id")["zhat"])
        te_r = rr[(rr.year == y)].dropna(subset=["act_z_y", "pf_front"])
        cls_act = lambda s: pd.cut(s, [-99, -0.5, 0.5, 99], labels=["ハイ", "ミドル", "スロー"]).astype(str)   # noqa: E731
        act_cls = cls_act(te_r["act_z_y"])
        tf, ts = _thresholds(pm_tr, rr.loc[pm_tr.index, "zhat"])
        cls_new = lambda s: pd.cut(s, [-99, tf, ts, 99], labels=["ハイ", "ミドル", "スロー"]).astype(str)   # noqa: E731
        res = {"pace_corr_new": round(float(np.corrcoef(te_r["zhat"], te_r["act_z_y"])[0, 1]), 3),
               "pace_acc_new": round(float((cls_new(te_r["zhat"]) == act_cls).mean()), 3),
               "pace_thresholds": [round(tf, 3), round(ts, 3)],
               "act_class_share": {k: round(float((act_cls == k).mean()), 3) for k in ("ハイ", "ミドル", "スロー")},
               "pace_acc_always_mid": round(float((act_cls == "ミドル").mean()), 3),
               "pace_corr_PF_pressure": round(float(np.corrcoef(te_r["pf_pressure"], te_r["act_z_y"])[0, 1]), 3),
               "pace_acc_PF": round(float((te_r["pf_lab"].replace("不明", np.nan).dropna() == act_cls[te_r["pf_lab"].replace("不明", np.nan).notna()]).mean()), 3),
               "pace_acc_TD": round(float((te_r["td_pace_type"].dropna() == act_cls[te_r["td_pace_type"].notna()]).mean()), 3)}
        cols = ["pf_early_d", "act_l3f_d", "zhat", "td_l3f_d"]
        tr = d[d.year < y].dropna(subset=cols)
        te = d[d.year == y].dropna(subset=cols).copy()
        abl = {}
        for kind, nm in (("full", "位置×ペース＋自己の過去上がり"), ("coupling", "位置×ペースのみ"), ("prior", "自己の過去上がりのみ")):
            m = Ridge(alpha=3.0).fit(_l3f_X(tr, tr["zhat"], kind), tr["act_l3f_d"])
            p = m.predict(_l3f_X(te, te["zhat"], kind))
            te["p_" + kind] = p - pd.Series(p, index=te.index).groupby(te["race_id"]).transform("mean")
            abl[nm] = {"corr": round(float(np.corrcoef(te["p_" + kind], te["act_l3f_d"])[0, 1]), 3), "mae": round(float((te["p_" + kind] - te["act_l3f_d"]).abs().mean()), 4)}
        mfull = Ridge(alpha=3.0).fit(_l3f_X(tr, tr["zhat"], "full"), tr["act_l3f_d"])
        cfg_fold = {"l3f": {"features": L3F_FEATS, "coef": [float(c) for c in mfull.coef_], "intercept": float(mfull.intercept_)}}
        slopes_c = np.array([pc.l3f_slope(cfg_fold, z, dist, sf) for z, dist, sf in zip(te["zhat"], te["distance"], te["surface"])])
        pcl = cfg_fold["l3f"]["intercept"] + slopes_c * te["pf_early_d"].values + float(mfull.coef_[L3F_FEATS.index("prior")]) * te["prior_d"].values
        te["p_clamp"] = pcl - pd.Series(pcl, index=te.index).groupby(te["race_id"]).transform("mean")
        abl["位置×ペース＋自己の過去上がり（傾きの上限つき・採用）"] = {"corr": round(float(np.corrcoef(te["p_clamp"], te["act_l3f_d"])[0, 1]), 3), "mae": round(float((te["p_clamp"] - te["act_l3f_d"]).abs().mean()), 4)}
        b = np.polyfit(te["td_l3f_d"], te["act_l3f_d"], 1)
        abl["TD既存（そのまま）"] = {"corr": round(float(np.corrcoef(te["td_l3f_d"], te["act_l3f_d"])[0, 1]), 3), "mae": round(float((te["td_l3f_d"] - te["act_l3f_d"]).abs().mean()), 4)}
        abl["TD既存（最良の線形再較正）"] = {"corr": abl["TD既存（そのまま）"]["corr"], "mae": round(float((np.polyval(b, te["td_l3f_d"]) - te["act_l3f_d"]).abs().mean()), 4)}
        abl["全員 0（偏差なし）"] = {"corr": 0.0, "mae": round(float(te["act_l3f_d"].abs().mean()), 4)}
        res["ablation"] = abl
        res["n"] = int(len(te))
        sl = {}
        for k, gr in te.groupby("act_pace"):
            if k in ("ハイ", "ミドル", "スロー"):
                sl[k] = {"実測": _slope(gr["act_early_norm_d"], gr["act_l3f_d"]), "TD": _slope(gr["td_early_d"], gr["td_l3f_d"]), "新": _slope(gr["pf_early_d"], gr["p_clamp"])}
        res["slopes_by_actual_pace"] = sl
        res["slope_all"] = {"実測": _slope(te["act_early_norm_d"], te["act_l3f_d"]), "TD": _slope(te["td_early_d"], te["td_l3f_d"]), "新": _slope(te["pf_early_d"], te["p_clamp"])}
        res["corr_new_vs_td_early"] = round(float(np.corrcoef(te["p_clamp"], te["td_early_d"].fillna(0))[0, 1]), 3)
        out["walk_forward"][str(y)] = res
        # 基準（レース全体の上がりの水準）
        med = d.groupby("race_id")["act_l3f"].median().rename("act_med")
        q = r[r.year == y].merge(med, on="race_id").dropna(subset=["td_l3f_base", "act_med"])
        q = q.assign(cohort=q.apply(lambda x: tab["cohort_l3f"].get(_gkey(x["venue"], x["surface"], x["distance"], x["going_group"]),
                                                                       tab["cohort_l3f_fallback"].get(f"{x['surface']}|{pc._dist(x['distance'])}|{x['going_group']}")), axis=1)).dropna(subset=["cohort"])
        out["baseline"][str(y)] = {"n": int(len(q)), "MAE_td": round(float((q.td_l3f_base - q.act_med).abs().mean()), 3), "MAE_cohort": round(float((q.cohort - q.act_med).abs().mean()), 3),
                                   "bias_td": round(float((q.td_l3f_base - q.act_med).mean()), 3), "bias_cohort": round(float((q.cohort - q.act_med).mean()), 3),
                                   "corr_td": round(float(np.corrcoef(q.td_l3f_base, q.act_med)[0, 1]), 3), "corr_cohort": round(float(np.corrcoef(q.cohort, q.act_med)[0, 1]), 3)}
    # 最終（2020-2025 全部）
    tab = cohort_tables(h, r)
    d = h.copy()
    d = d.merge(prior_from_runs(d[base_cols + ["runs"]], tab), on=base_cols, how="left")
    d["prior_d"] = (d["prior_c"] - d.groupby("race_id")["prior_c"].transform("mean")).fillna(0.0)
    rr = r.copy()
    rr["act_z_f"] = z_from_tables(rr, tab)
    pm_tr = rr.dropna(subset=["act_z_f", "pf_front", "pf_stalker", "pf_pressure"])
    pm = Ridge(alpha=5.0).fit(_pace_X(pm_tr), pm_tr["act_z_f"])
    rr["zhat"] = pm.predict(_pace_X(rr.fillna({"pf_front": 0, "pf_stalker": 0, "pf_pressure": 0})))
    d["zhat"] = d["race_id"].map(rr.set_index("race_id")["zhat"])
    ok = d.dropna(subset=["pf_early_d", "act_l3f_d", "zhat"])
    mdl = Ridge(alpha=3.0).fit(_l3f_X(ok, ok["zhat"], "full"), ok["act_l3f_d"])
    FINAL_TF, FINAL_TS = _thresholds(pm_tr.assign(act_z_f=pm_tr["act_z_f"]), rr.loc[pm_tr.index, "zhat"])
    cfg = {
        "version": 1, "fit_years": [2020, 2025], "n_races": int(len(r)), "n_horses": int(len(ok)),
        "pace": {"features": list(_pace_X(rr).columns), "coef": [float(c) for c in pm.coef_], "intercept": float(pm.intercept_),
                 "thresholds": {"fast": FINAL_TF, "slow": FINAL_TS}, "note": "予測 z の閾値は、学習データの実測クラス比率に合わせた分位。z = 実際の前半3Fの同条件（場×面×距離×馬場）中央値からの標準化。小さいほど速い（ハイ）。特徴量は pf_front, pf_stalker（位置取り側の逃げ・先行頭数）, pf_pressure, 芝, 距離(km-1.6), 道悪"},
        "l3f": {"features": L3F_FEATS, "coef": [float(c) for c in mdl.coef_], "intercept": float(mdl.intercept_), "slope_max": pc.SLOPE_MAX,
                "note": "上がり偏差(秒)=係数×[e, e×z, e×(距離km-1.6), e×芝, prior]。e=位置取り(0=先頭,1=最後方)のレース内平均との差、prior=自己の過去5走の(上がり-同条件中央値)のレース内平均との差"},
        **tab,
    }
    out["final_coef"] = {"pace": dict(zip(cfg["pace"]["features"], cfg["pace"]["coef"])), "l3f": dict(zip(L3F_FEATS, cfg["l3f"]["coef"])), "l3f_intercept": cfg["l3f"]["intercept"]}
    out["cohort_sizes"] = {"cohort_l3f": len(tab["cohort_l3f"]), "fallback": len(tab["cohort_l3f_fallback"]), "cohort_t3f": len(tab["cohort_t3f"])}
    return out, cfg


def main() -> None:
    h, r = prepare()
    r["year"] = pd.to_datetime(r["date"]).dt.year
    h["year"] = pd.to_datetime(h["date"]).dt.year
    res = {"A": part_a(r), "BC": part_b_c(h, r)}
    d, cfg = part_d(h, r)
    res["D"] = d
    (D / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    CONFIG_OUT.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"walk_forward": d["walk_forward"], "baseline": d["baseline"], "final": d["final_coef"], "sizes": d["cohort_sizes"]}, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()

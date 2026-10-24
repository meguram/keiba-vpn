"""tp-60 血統ベースのレース質分析: 血統マップのグループ g × レースの質 q ごとの好走率・人気以上の好走を測り、レース前に出せる質の確率 P(q̂) で未知のレースに適用する。

python -m src.scripts.data.tp60_quality table     # レース質ラベルの共有表（実測側 + 事前の予測確率）→ data/local/modeling/shared_cache/race_quality/race_quality.parquet
python -m src.scripts.data.tp60_quality clusters  # 血統マップ（構成埋め込み A）のクラスタ g（K=16, 40）→ <TP60_OUT>/embed/cluster_K*.parquet と説明
python -m src.scripts.data.tp60_quality gq        # g × q の効果・翌年再現・FDR、オラクル上限と実運用の下限、馬券側の補正、V5 への群（PQ_*）→ roi/gq.json, groups/PQ_*.parquet

レース質 q（実測）:
  joint   tp-61 の展開タイプ 9 通り（ペース slow/mid/high × 隊列 compact/normal/stretched）
  lap     tp-61 のラップ型 6 通り（lap_type_rule）
  l3f     上がり水準（レース上がり 3F の同条件中央値からの差の 3 分位: fast/mid/slow。分位点は 2020〜2021 年で固定）
  going   馬場（良 / 道悪）
  wtime   勝ち時計の基準比（同条件の中央値との差の 3 分位。馬場の速さ）  [実測のみ]
  upset   荒れ度（勝ち馬の人気: 1-2 / 3-5 / 6 以上）                          [実測のみ]
事前側: 展開タイプ・ラップ型・上がり水準は、レース前（前日〜T-60）に分かる量（想定位置取りの分布・コースの履歴・ハナ主張馬など = tp-61 の pre.parquet）から
  年ごとの拡張窓（y 年の確率は y 年より前だけで学習）で出した確率。馬場は発表値をそのまま使う（T-60 時点でほぼ確定）。
人気以上の好走: r_win = 勝ち − 市場の勝率（単勝オッズを正規化）、r_t3 = 3 着内 − 市場の 3 着内率（複勝オッズを 3 になるよう正規化）。市場は評価の基準にだけ使い、特徴量にはしない。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp60_lib as L
from src.scripts.data.tp60_roi import _save

ROOT = Path(__file__).resolve().parents[3]
SHARED = ROOT / "data/local/modeling/shared_cache/race_quality"
TP61 = ROOT / "data/local/modeling/tp61"
TABLES = ROOT / "data/page_reference/tables"
CUSHION = ROOT / "data/page_reference/cushion"
EMB = L.OUTD / "embed"
GROUPS = L.OUTD / "groups"
PACE = ["slow", "mid", "high"]
SHAPE = ["compact", "normal", "stretched"]
JOINT = [f"{p}|{s}" for p in PACE for s in SHAPE]
LAP_NAMES = ["slow_kick", "slow_spurt", "even_kick", "even_flat", "high_breather", "high_collapse"]
KS = [16, 40]
GROUPINGS = [("K16", 16), ("K40", 40), ("line", 8), ("sire30", 30)]          # 血統マップのクラスタ 2 通り + 父の系統 + 産駒数上位 30 の父（解釈しやすい対照）
LINES = ["SS系", "ロベルト系", "MrP系", "ND系", "ナスルーラ系", "ネイティヴダンサー系", "ヘイルトゥリーズン系", "その他"]
EXCLUDE = {"race_id", "date", "year", "venue", "surface", "going_group", "grade", "race_class", "band", "pace_cls", "shape_cls", "bal", "t3f_dev", "race_l3f_dev", "c4_gap_per_horse", "c4_lead_group_frac",
           "lap_type", "joint", "q_joint", "q_lap", "q_l3f"}


def _log(msg):
    L.log(msg)


# ---------------------------------------------------------------- 共有表
def _winner_stats() -> pd.DataFrame:
    parts = []
    for y in range(2020, 2027):
        f = TABLES / str(y) / "race_result_flat.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["race_id", "date", "venue_code", "surface", "distance", "track_condition", "finish_position", "time_sec", "popularity", "last_3f"])
        d["race_id"] = d["race_id"].astype(str)
        d["finish_position"] = pd.to_numeric(d["finish_position"], errors="coerce")
        d["popularity"] = pd.to_numeric(d["popularity"], errors="coerce")
        d["time_sec"] = pd.to_numeric(d["time_sec"], errors="coerce")
        parts.append(d)
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["race_id", "finish_position", "popularity"], keep="first")
    w = d[d["finish_position"] == 1].groupby("race_id").agg(win_time=("time_sec", "min"), winner_pop=("popularity", "min"), venue_code=("venue_code", "first"), date=("date", "first"))
    t3 = d[d["finish_position"].between(1, 3)].groupby("race_id")["popularity"].sum().rename("top3_pop_sum")
    return w.join(t3).reset_index()


def _cushion_table() -> pd.DataFrame:
    rows = []
    for f in sorted(CUSHION.glob("cushion_*.json")):
        try:
            rows += json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
    c = pd.DataFrame(rows)
    if c.empty:
        return c
    c["date"] = pd.to_datetime(c["date"])
    return c.drop_duplicates(["date", "venue_code"])[["date", "venue_code", "cushion_value", "turf_moisture_goal", "dirt_moisture_goal"]]


def cmd_table(a) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    rc = pd.read_parquet(TP61 / "race_cls.parquet")
    pre = pd.read_parquet(TP61 / "pre.parquet")
    rc["race_id"], pre["race_id"] = rc["race_id"].astype(str), pre["race_id"].astype(str)
    w = _winner_stats()
    rq = rc[["race_id", "date", "year", "venue", "surface", "distance", "band", "track_condition", "going_group", "field_size", "grade", "race_class", "t3f", "race_l3f", "bal", "t3f_dev", "race_l3f_dev",
             "c4_gap_per_horse", "c4_lead_group_frac", "pace_cls", "shape_cls"]].merge(w[["race_id", "win_time", "winner_pop", "top3_pop_sum", "venue_code"]], on="race_id", how="left")
    rq["date"] = pd.to_datetime(rq["date"])
    lt = pre[["race_id", "lap_type"]].copy()
    rq = rq.merge(lt, on="race_id", how="left")
    rq["lap_name"] = rq["lap_type"].where(rq["lap_type"].isin(LAP_NAMES))
    rq["q_joint"] = np.where(rq["pace_cls"].notna() & rq["shape_cls"].notna(), rq["pace_cls"].astype(str) + "|" + rq["shape_cls"].astype(str), None)
    rq["q_lap"] = rq["lap_name"]
    tr = rq["year"] <= 2021
    q1, q2 = rq.loc[tr, "race_l3f_dev"].quantile([1 / 3, 2 / 3])
    rq["q_l3f"] = np.where(rq["race_l3f_dev"].isna(), None, np.where(rq["race_l3f_dev"] <= q1, "fast", np.where(rq["race_l3f_dev"] <= q2, "mid", "slow")))
    rq["q_going"] = rq["going_group"].astype(str)
    # 勝ち時計の基準比（同じ場×面×距離×馬場の中央値との差。馬場の速さの実測）
    key = rq["venue"].astype(str) + "|" + rq["surface"].astype(str) + "|" + rq["distance"].astype(str) + "|" + rq["going_group"].astype(str)
    med = rq.groupby(key)["win_time"].transform("median")
    rq["win_time_dev"] = rq["win_time"] - med
    t1, t2 = rq.loc[tr, "win_time_dev"].quantile([1 / 3, 2 / 3])
    rq["q_wtime"] = np.where(rq["win_time_dev"].isna(), None, np.where(rq["win_time_dev"] <= t1, "fast", np.where(rq["win_time_dev"] <= t2, "mid", "slow")))
    rq["q_upset"] = np.where(rq["winner_pop"].isna(), None, np.where(rq["winner_pop"] <= 2, "fav", np.where(rq["winner_pop"] <= 5, "mid", "long")))
    # 含水率・クッション値（発表値。レース前に分かる）
    cu = _cushion_table()
    if not cu.empty:
        cu["venue_code"] = cu["venue_code"].astype(str).str.zfill(2)
        rq["venue_code"] = rq["venue_code"].astype(str).str.zfill(2)
        rq = rq.merge(cu, on=["date", "venue_code"], how="left")
        rq["moisture"] = np.where(rq["surface"].astype(str) == "芝", rq["turf_moisture_goal"], rq["dirt_moisture_goal"])
    # クラス水準
    from src.scripts.data.tp59_data import class_of

    rq["cls"] = class_of(rq["race_class"], rq["grade"]).to_numpy()
    # ---- 事前の予測確率（年ごとの拡張窓）
    feats = [c for c in pre.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(pre[c])]
    X = pre.set_index("race_id")[feats]
    lab = rq.set_index("race_id")
    y_joint = lab["q_joint"].map({j: i for i, j in enumerate(JOINT)})
    y_lap = lab["lap_name"].map({n: i for i, n in enumerate(LAP_NAMES)})
    y_l3f = lab["q_l3f"].map({"fast": 0, "mid": 1, "slow": 2})
    probs = {}
    for name, y, classes in (("joint", y_joint, len(JOINT)), ("lap", y_lap, 6), ("l3f", y_l3f, 3)):
        P = np.full((len(lab), classes), np.nan)
        idx = lab.index
        for yr in range(2021, 2027):
            trm = (lab["year"] < yr) & y.notna()
            tem = lab["year"] == yr
            if trm.sum() < 500 or tem.sum() == 0:
                continue
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_depth=4, l2_regularization=2.0, random_state=0)
            clf.fit(X.loc[idx[trm]], y[trm].astype(int))
            pp = clf.predict_proba(X.loc[idx[tem]])
            cols = clf.classes_
            full = np.zeros((tem.sum(), classes))
            full[:, cols] = pp
            P[np.flatnonzero(tem.to_numpy())] = full
            _log(f"予測 {name} {yr}: 学習 {int(trm.sum())} レース、検証 {int(tem.sum())}")
        probs[name] = P
    for i, j in enumerate(JOINT):
        rq[f"p_joint_{j}"] = probs["joint"][:, i]
    for i, n in enumerate(LAP_NAMES):
        rq[f"p_lap_{n}"] = probs["lap"][:, i]
    for i, n in enumerate(["fast", "mid", "slow"]):
        rq[f"p_l3f_{n}"] = probs["l3f"][:, i]
    SHARED.mkdir(parents=True, exist_ok=True)
    rq.to_parquet(SHARED / "race_quality.parquet", index=False)
    meta = {"rows": int(len(rq)), "years": sorted(rq["year"].unique().tolist()),
            "labels_measured": ["pace_cls", "shape_cls", "q_joint(9)", "lap_type/q_lap(6)", "race_l3f_dev", "q_l3f", "q_going", "win_time_dev", "q_wtime", "q_upset", "winner_pop", "top3_pop_sum", "moisture", "cushion_value", "cls"],
            "labels_pre_race": ["p_joint_<pace>|<shape>(9)", "p_lap_<name>(6)", "p_l3f_<fast|mid|slow>(3)", "going_group(発表値)", "moisture/cushion_value(発表値)", "cls/venue/surface/distance/band/field_size"],
            "l3f_thresholds": [float(q1), float(q2)], "wtime_thresholds": [float(t1), float(t2)], "source": "tp61 race_cls/pre.parquet + tables race_result_flat + cushion",
            "note": "p_* は y 年より前だけで学習した HistGradientBoosting（拡張窓）。2020 年は NaN。実測側の列は結果から作るので、特徴量には使わない（評価・g×q の効果表用）"}
    (SHARED / "README.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    acc = {}
    for name, y, classes in (("joint", y_joint, len(JOINT)), ("lap", y_lap, 6), ("l3f", y_l3f, 3)):
        ok = probs[name][:, 0] == probs[name][:, 0]
        yy = y.to_numpy()
        m_ = ok & ~np.isnan(yy)
        ll = -np.log(np.clip(probs[name][m_, yy[m_].astype(int)], 1e-9, 1)).mean()
        prior = np.bincount(yy[m_].astype(int), minlength=classes) / m_.sum()
        acc[name] = {"logloss": float(ll), "prior_logloss": float(-np.log(np.clip(prior[yy[m_].astype(int)], 1e-9, 1)).mean()), "n": int(m_.sum())}
    _save("quality_pre_eval.json", acc)
    _log(f"共有表 {rq.shape} → {SHARED}、事前予測の対数損失 {acc}")


# ---------------------------------------------------------------- クラスタ（血統マップ）
def cmd_clusters(a) -> None:
    from sklearn.cluster import KMeans

    from src.scripts.data import tp60_embfeat as EF

    m = L.load_master()
    E, horses = EF.load_emb("A_t2_T10_g100")
    U = EF.unit(E)
    first = m.groupby("horse_id")["year"].min()
    fit = np.array([first.get(h, 9999) <= 2021 for h in horses]) & (np.linalg.norm(E, axis=1) > 1e-9)
    names = json.loads((L.PED / "anc_names.json").read_text(encoding="utf-8"))
    hs = pd.read_parquet(L.PED / "horse_static.parquet").drop_duplicates("horse_id").set_index("horse_id").reindex(horses)
    info = {}
    for K in KS:
        km = KMeans(n_clusters=K, random_state=0, n_init=4).fit(U[fit])
        lab = np.full(len(horses), -1)
        ok = np.linalg.norm(E, axis=1) > 1e-9
        lab[ok] = km.predict(U[ok])
        pd.DataFrame({"horse_id": horses, "g": lab}).to_parquet(EMB / f"cluster_K{K}.parquet", index=False)
        ent = []
        for c in range(K):
            mk = lab == c
            sub = hs[mk]
            top = sub["sire_id"].value_counts().head(3)
            ent.append({"g": c, "horses": int(mk.sum()), "top_sires": [{"id": i, "name": (names.get(i, i))[:14], "n": int(n)} for i, n in top.items()],
                        "sire_line": sub["sire_line"].value_counts(normalize=True).head(3).round(2).to_dict(), "dsire_line": sub["dsire_line"].value_counts(normalize=True).head(2).round(2).to_dict(),
                        "mean_xF": float(sub["x_F"].mean())})
        info[f"K{K}"] = ent
        _log(f"クラスタ K={K} 完了")
    top = m[m["year"] <= 2021]["sire_id"].value_counts().head(30)
    info["sire30"] = [{"g": i, "id": sid, "name": (names.get(sid, sid))[:14], "n_2020_21": int(n)} for i, (sid, n) in enumerate(top.items())]
    info["line"] = [{"g": i, "name": n} for i, n in enumerate(LINES)]
    _save("clusters_info.json", info)


# ---------------------------------------------------------------- g × q
def _load_base() -> pd.DataFrame:
    m = L.load_master()
    rq = pd.read_parquet(SHARED / "race_quality.parquet")
    rq["race_id"] = rq["race_id"].astype(str)
    keep = ["race_id", "q_joint", "q_lap", "q_l3f", "q_going", "q_wtime", "q_upset"] + [c for c in rq.columns if c.startswith(("p_joint_", "p_lap_", "p_l3f_"))]
    m = m.merge(rq[keep], on="race_id", how="left")
    inv = 1.0 / m["place_eff"].replace(0, np.nan)
    m["q3"] = 3.0 * inv / inv.groupby(m["race_id"]).transform("sum")
    m["r_t3"] = m["top3"] - m["q3"]
    m["r_win"] = m["win"] - m["q_win"]
    m["r_t3_v4"] = m["top3"] - m["p_top3"]
    m["r_win_v4"] = m["win"] - m["p_win"]
    return m


def _add_groups(m: pd.DataFrame, name: str, size: int) -> pd.DataFrame:
    col = f"g{name}"
    if name.startswith("K"):
        cl = pd.read_parquet(EMB / f"cluster_{name}.parquet")
        cl["horse_id"] = cl["horse_id"].astype(str)
        m = m.drop(columns=[col], errors="ignore").merge(cl.rename(columns={"g": col}), on="horse_id", how="left")
        m.loc[m[col] == -1, col] = np.nan
    elif name == "line":
        m[col] = m["sire_line"].map({n: i for i, n in enumerate(LINES)}).where(m["sire_line"].notna())
        m.loc[m["sire_line"].notna() & m[col].isna(), col] = len(LINES) - 1
    else:
        top = m[m["year"] <= 2021]["sire_id"].value_counts().head(size).index
        m[col] = m["sire_id"].map({s_: i for i, s_ in enumerate(top)})
    return m


QSYS = {"joint": ("q_joint", [f"{p}|{s}" for p in PACE for s in SHAPE], "p_joint_"), "lap": ("q_lap", LAP_NAMES, "p_lap_"), "l3f": ("q_l3f", ["fast", "mid", "slow"], "p_l3f_"),
        "going": ("q_going", ["良", "道悪"], None), "wtime": ("q_wtime", ["fast", "mid", "slow"], None), "upset": ("q_upset", ["fav", "mid", "long"], None)}
PRED_SYS = ["joint", "lap", "l3f", "going"]
KSHR = 300.0


def effect_table(d: pd.DataFrame, gcol: str, qcol: str, metric: str, k: float = KSHR) -> tuple[pd.DataFrame, pd.Series]:
    """g × q の交互作用 = 平均(g,q) − 平均(g)（市場の残差は 1 レース内で和が 0 なので q の主効果は ~0）。経験ベイズで 0 へ縮約。z は縮約前。"""
    dd = d.dropna(subset=[gcol, qcol, metric])
    gm = dd.groupby(gcol)[metric].mean()
    sd = dd[metric].std()
    c = dd.groupby([gcol, qcol])[metric].agg(["mean", "size"]).reset_index()
    c["raw"] = c["mean"] - c[gcol].map(gm)
    c["z"] = c["raw"] / (sd / np.sqrt(c["size"]))
    c["eff"] = c["raw"] * c["size"] / (c["size"] + k)
    return c, gm


def uplift(d: pd.DataFrame, tab: pd.DataFrame, gcol: str, qcol: str, classes: list[str], prefix: str | None, K: int, known_pre: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """オラクル（実際の q）と事前の確率で重み付けした期待上乗せ。known_pre（馬場など発表値）はオラクル = 事前。"""
    E = np.zeros((K + 1, len(classes)))
    cidx = {c: i for i, c in enumerate(classes)}
    for gv, qv, ef in zip(tab[gcol].to_numpy(), tab[qcol].to_numpy(), tab["eff"].to_numpy()):
        if qv in cidx:
            E[int(gv), cidx[qv]] = ef
    g = d[gcol].to_numpy(dtype=float)
    qi = d[qcol].map(cidx).to_numpy(dtype=float)
    valid = ~np.isnan(g) & ~np.isnan(qi)
    orc = np.full(len(d), np.nan)
    orc[valid] = E[g[valid].astype(int), qi[valid].astype(int)]
    pred = np.full(len(d), np.nan)
    if known_pre:
        pred = orc.copy()
    elif prefix:
        P = np.column_stack([d[f"{prefix}{c}"].to_numpy(dtype=float) for c in classes])
        okp = ~np.isnan(g) & ~np.isnan(P[:, 0])
        pred[okp] = (P[okp] * E[g[okp].astype(int)]).sum(1)
    return orc, pred


def cmd_gq(a) -> None:
    from src.scripts.data.tp60_deep import _ols_cluster

    m = _load_base()
    m = m[m["year"] >= 2020].reset_index(drop=True)
    out: dict = {"K": {}, "notes": {"k_shrink": KSHR}}
    keep_cols: dict[str, np.ndarray] = {}
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    for gname, K in GROUPINGS:
        m = _add_groups(m, gname, K)
        gcol = f"g{gname}"
        res_K: dict = {"qsys": {}}
        for qs, (qcol, classes, prefix) in QSYS.items():
            R: dict = {"years": {}}
            for metric in ("r_t3", "r_win"):
                for y in years:
                    prior = m[m["year"] < y]
                    cur = m[m["year"] == y]
                    tab, gm = effect_table(prior, gcol, qcol, metric)
                    tab_c, _ = effect_table(cur, gcol, qcol, metric, k=0.0)
                    j = tab.merge(tab_c[[gcol, qcol, "raw", "size"]], on=[gcol, qcol], suffixes=("", "_y"))
                    j = j[(j["size"] >= 150) & (j["size_y"] >= 40)]
                    ent = {"cells": int(len(j))}
                    if len(j) >= 20:
                        ent["rho_prior_vs_year"] = float(j["eff"].corr(j["raw_y"], method="spearman"))
                        sig = j[np.abs(j["z"]) >= 2.0]
                        ent["n_sig_prior"] = int(len(sig))
                        ent["sign_agree_sig"] = float((np.sign(sig["eff"]) == np.sign(sig["raw_y"])).mean()) if len(sig) else None
                        # 前の累積の上位 10% の（g,q）セルが翌年にどれだけ上乗せしたか（セル平均の加重）
                        thr = j["eff"].quantile(0.9)
                        top = j[j["eff"] >= thr]
                        ent["top10_next_year_raw"] = float((top["raw_y"] * top["size_y"]).sum() / top["size_y"].sum())
                        ent["bottom10_next_year_raw"] = float((j[j["eff"] <= j["eff"].quantile(0.1)]["raw_y"] * j[j["eff"] <= j["eff"].quantile(0.1)]["size_y"]).sum() / j[j["eff"] <= j["eff"].quantile(0.1)]["size_y"].sum())
                    R["years"][f"{metric}:{y}"] = ent
            # 年別に作った上乗せ（オラクル / 事前）の説明力と、V4・市場の残差との関係
            cur_cols = []
            for y in years:
                prior = m[m["year"] < y]
                cur = m[m["year"] == y]
                if len(prior) < 20000:
                    continue
                for metric, tag in (("r_t3", "t3"), ("r_win", "win")):
                    tab, _ = effect_table(prior, gcol, qcol, metric)
                    orc, pred = uplift(cur, tab, gcol, qcol, classes, prefix, K, known_pre=(qs == 'going'))
                    cur_cols.append((y, tag, cur.index.to_numpy(), orc, pred))
            for tag in ("t3", "win"):
                rows = {}
                for y in years[1:]:
                    sub = [x for x in cur_cols if x[0] == y and x[1] == tag]
                    if not sub:
                        continue
                    _, _, idx, orc, pred = sub[0]
                    d_ = m.loc[idx]
                    for nm, up in (("oracle", orc), ("pred", pred)):
                        ok = np.isfinite(up) & np.isfinite(d_[f"r_{tag}"].to_numpy())
                        if ok.sum() < 3000 or np.nanstd(up[ok]) == 0:
                            continue
                        X = np.column_stack([np.ones(ok.sum()), up[ok]])
                        b, se = _ols_cluster(X, d_.loc[ok, f"r_{tag}"].to_numpy(), d_.loc[ok, "race_id"].to_numpy())
                        rows.setdefault(nm, {})[str(y)] = {"slope": float(b[1]), "t": float(b[1] / se[1]) if se[1] > 0 else None, "corr": float(np.corrcoef(up[ok], d_.loc[ok, f"r_{tag}"].to_numpy())[0, 1]), "n": int(ok.sum()),
                                                           "up_sd": float(np.std(up[ok]))}
                        # V4 の残差に対しても（2022 年以降）
                        okv = ok & d_[f"r_{tag}_v4"].notna().to_numpy()
                        if okv.sum() > 3000:
                            Xv = np.column_stack([np.ones(okv.sum()), up[okv]])
                            bv, sev = _ols_cluster(Xv, d_.loc[okv, f"r_{tag}_v4"].to_numpy(), d_.loc[okv, "race_id"].to_numpy())
                            rows[nm][str(y)]["slope_v4"], rows[nm][str(y)]["t_v4"] = float(bv[1]), float(bv[1] / sev[1]) if sev[1] > 0 else None
                R[f"explain_{tag}"] = rows
            # 列として保存（事前の確率版。PRED_SYS のみ）
            if qs in PRED_SYS:
                for tag in ("t3", "win"):
                    col = np.full(len(m), np.nan)
                    for y, tg, idx, orc, pred in cur_cols:
                        if tg == tag:
                            col[idx] = pred
                    keep_cols[f"pq60_{gname}_{qs}_up_{tag}"] = col
            # 全期間の最大のセル（解釈用）: 2020〜2025 の合算
            tab_all, gm_all = effect_table(m[m["year"] <= 2025], gcol, qcol, "r_t3")
            tab_h1, _ = effect_table(m[m["year"] <= 2022], gcol, qcol, "r_t3", k=0.0)
            tab_h2, _ = effect_table(m[(m["year"] >= 2023) & (m["year"] <= 2025)], gcol, qcol, "r_t3", k=0.0)
            tab_26, _ = effect_table(m[m["year"] == 2026], gcol, qcol, "r_t3", k=0.0)
            tab_all = (tab_all.merge(tab_h1[[gcol, qcol, "raw"]].rename(columns={"raw": "raw_h1"}), on=[gcol, qcol], how="left")
                       .merge(tab_h2[[gcol, qcol, "raw"]].rename(columns={"raw": "raw_h2"}), on=[gcol, qcol], how="left")
                       .merge(tab_26[[gcol, qcol, "raw", "size"]].rename(columns={"raw": "raw_26", "size": "n_26"}), on=[gcol, qcol], how="left"))
            tab_all = tab_all[tab_all["size"] >= 400].sort_values("z")
            def rowdict(r):
                return {"g": int(r[gcol]), "q": r[qcol], "n": int(r["size"]), "raw_pt": float(100 * r["raw"]), "z": float(r["z"]), "h1": None if pd.isna(r["raw_h1"]) else float(100 * r["raw_h1"]), "h2": None if pd.isna(r["raw_h2"]) else float(100 * r["raw_h2"]),
                        "y26": None if pd.isna(r["raw_26"]) else float(100 * r["raw_26"]), "n26": None if pd.isna(r["n_26"]) else int(r["n_26"])}
            R["table"] = [rowdict(r) for _, r in tab_all.sort_values([gcol, qcol]).iterrows()]
            R["top_pos"] = [rowdict(r) for _, r in tab_all.tail(8).iloc[::-1].iterrows()]
            R["top_neg"] = [rowdict(r) for _, r in tab_all.head(8).iterrows()]
            R["n_cells_pooled"] = int(len(tab_all))
            R["n_abs_z_gt3"] = int((tab_all["z"].abs() > 3).sum())
            # BH（q=10%）
            from scipy.stats import norm
            pv = 2 * (1 - norm.cdf(tab_all["z"].abs().to_numpy()))
            order = np.argsort(pv)
            ranks = np.arange(1, len(pv) + 1)
            passed = (pv[order] <= 0.10 * ranks / len(pv))
            R["bh_significant_cells"] = int(passed.nonzero()[0].max() + 1) if passed.any() else 0
            res_K["qsys"][qs] = R
        out["K"][gname] = res_K
        _log(f"{gname} 完了")
    _save("gq.json", out)
    # ---- 馬券側（EV の補正・フィルタ）と V5 用の群
    pq = pd.DataFrame({"race_id": m["race_id"], "horse_id": m["horse_id"]})
    for c, v in keep_cols.items():
        pq[c] = v.astype("float32")
    GROUPS.mkdir(parents=True, exist_ok=True)
    for gname, K in GROUPINGS:
        cols = [c for c in pq.columns if c.startswith(f"pq60_{gname}_")]
        pq[["race_id", "horse_id"] + cols].to_parquet(GROUPS / f"PQ_{gname}.parquet", index=False)
    _log(f"群 PQ_K16 / PQ_K40 を書き出し（{[c for c in pq.columns][:6]} ...）")
    bet = _bet_side(m, pq)
    _save("gq_bet.json", bet)


def _bet_side(m: pd.DataFrame, pq: pd.DataFrame) -> dict:
    """V4 の EV（p_win × 確定オッズ）≥ τ の買い目に、事前の上乗せ（pred）が正／負かで絞ったときの回収率の差。"""
    d = pd.concat([m, pq.drop(columns=["race_id", "horse_id"])], axis=1)
    d = d[(d["year"] >= 2022) & d["p_win"].notna()].reset_index(drop=True)
    d["ev_win"] = d["p_win"] * d["win_odds"]
    d["ev_place"] = d["p_top3"] * d["place_eff"]
    out: dict = {"rows": []}
    for gname, K in GROUPINGS:
        for qs in PRED_SYS:
            col_w, col_t = f"pq60_{gname}_{qs}_up_win", f"pq60_{gname}_{qs}_up_t3"
            if col_w not in d.columns:
                continue
            for bet, evc, retc, ucol in (("win", "ev_win", "ret_win", col_w), ("place", "ev_place", "ret_place", col_t)):
                for tau in ((1.0, 1.2) if bet == "win" else (1.0, 1.1)):
                    base = (d[evc] >= tau).to_numpy() & d[ucol].notna().to_numpy()
                    for lab, ys in (("2022-2024", [2022, 2023, 2024]), ("2025", [2025]), ("2026", [2026])):
                        yy = d["year"].isin(ys).to_numpy()
                        elig = base & yy
                        pos = elig & (d[ucol] > 0).to_numpy()
                        r = L.boot_diff(d, pos, elig, retc, B=600)
                        out["rows"].append({"K": gname, "qs": qs, "bet": bet, "tau": tau, "period": lab, **{k: r.get(k) for k in ("sel_roi", "all_roi", "diff", "diff_lo", "diff_hi", "p_le0", "n_sel")}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("table", "clusters", "gq"):
        sub.add_parser(n)
    a = ap.parse_args()
    {"table": cmd_table, "clusters": cmd_clusters, "gq": cmd_gq}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

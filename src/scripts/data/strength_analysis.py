"""新しい親モデル（v6e16）の得意・苦手を条件別に測る。ホストで実行（読み取り＋出力のみ）。

    python -m src.scripts.data.strength_analysis tables --out DIR   # 条件別の精度表（勝率 LogLoss・市場・旧 V4 との差、的中率、ECE）
    python -m src.scripts.data.strength_analysis shap   --out DIR   # 特徴量群ごとの寄与（2025 年・学習に使わない年）を条件別に

予測は『その年を学習に使っていない』もの（2022〜2024: leave-one-year-out、2025: 2016〜2024 学習の test）。2026 は含めない。
市場の確率（確定オッズ由来 q_win）は物差しとしてだけ使い、特徴量にはしない。母数・区間はレース単位（レースを再標本化）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["race_id", "horse_id"]
BASE = Path("/home/jovyan/work/retrain_out/base_2016")
FRAME = BASE / "final_layer/frame_all.parquet"
OLD_FRAME = Path("data/local/modeling/betting/tp36/frame_all.parquet")
TRAIN_TABLE = BASE / "features/train_table.parquet"
BOOT = 400
EPS = 1e-9


def _ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load() -> pd.DataFrame:
    f = pd.read_parquet(FRAME)
    f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    ok = f.groupby("race_id")["win"].transform("sum") == 1
    f = f[ok].copy()
    old = pd.read_parquet(OLD_FRAME, columns=KEY + ["p_win"]).rename(columns={"p_win": "p_v4"})
    old["race_id"], old["horse_id"] = old["race_id"].astype(str), old["horse_id"].astype(str)
    f = f.merge(old, on=KEY, how="left")
    cols = ["race_id", "horse_id", "venue", "surface", "distance", "race_class", "track_condition", "field_size", "sex_age", "a_style", "horse_prior_races", "date"]
    t = pd.read_parquet(TRAIN_TABLE, columns=cols)
    t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
    t = t.rename(columns={"date": "date_t"})
    d = f.merge(t, on=KEY, how="left")
    d["month"] = pd.to_datetime(d["date"]).dt.month
    d["season"] = d["month"].map(lambda m: "春(3-5)" if 3 <= m <= 5 else "夏(6-8)" if 6 <= m <= 8 else "秋(9-11)" if 9 <= m <= 11 else "冬(12-2)")
    d["dist_band"] = pd.cut(d["distance"], [0, 1200, 1600, 2000, 2400, 9999], labels=["〜1200", "1300〜1600", "1700〜2000", "2100〜2400", "2500〜"]).astype(str)
    cls = d["race_class"].astype(str)
    d["class_b"] = np.select([cls.str.contains("新馬"), cls.str.contains("未勝利"), cls.str.contains("１勝"), cls.str.contains("２勝"), cls.str.contains("３勝"), cls.str.contains("オープン")],
                             ["新馬", "未勝利", "1勝", "2勝", "3勝", "オープン以上"], "その他(障害等)")
    d["field_b"] = pd.cut(d["field_size"], [0, 12, 15, 99], labels=["〜12頭", "13〜15頭", "16〜18頭"]).astype(str)
    d["age"] = pd.to_numeric(d["sex_age"].astype(str).str[1:], errors="coerce")
    d["age_b"] = pd.cut(d["age"], [0, 2, 3, 4, 99], labels=["2歳", "3歳", "4歳", "5歳以上"]).astype(str)
    d["sex"] = d["sex_age"].astype(str).str[0].map({"牡": "牡", "牝": "牝", "セ": "セン"})
    d["pop_b"] = pd.cut(d["popularity"], [0, 1, 3, 6, 9, 99], labels=["1人気", "2-3人気", "4-6人気", "7-9人気", "10人気〜"]).astype(str)
    d["hist_b"] = pd.cut(d["horse_prior_races"], [-1, 0, 2, 9, 999], labels=["履歴0走", "1-2走", "3-9走", "10走〜"]).astype(str)
    q = d["a_style"].quantile([1 / 3, 2 / 3]).to_list()
    d["style_b"] = np.where(d["a_style"].isna(), "脚質欠損", np.where(d["a_style"] <= q[0], "a_style 低", np.where(d["a_style"] <= q[1], "a_style 中", "a_style 高")))
    d["track_condition"] = d["track_condition"].astype(str).map({"稍": "稍重", "不": "不良"}).fillna(d["track_condition"].astype(str))
    d["course"] = d["venue"].astype(str) + "・" + d["surface"].astype(str) + "・" + d["dist_band"]
    d["l_model"] = _ll(d["win"].to_numpy(float), d["p_win"].to_numpy(float))
    d["l_mkt"] = _ll(d["win"].to_numpy(float), d["q_win"].to_numpy(float))
    d["l_v4"] = _ll(d["win"].to_numpy(float), d["p_v4"].to_numpy(float))
    return d


def race_table(d: pd.DataFrame) -> pd.DataFrame:
    """1 レース 1 行（レース単位の的中率用）。"""
    g = d.sort_values(["race_id", "p_win"], ascending=[True, False]).groupby("race_id", sort=True)
    top = g.head(1).set_index("race_id")
    mk = d.sort_values(["race_id", "q_win"], ascending=[True, False]).groupby("race_id", sort=True).head(1).set_index("race_id")
    return pd.DataFrame({"hit_model": top["win"].to_numpy(), "hit_mkt": mk.loc[top.index, "win"].to_numpy(), "p_top": top["p_win"].to_numpy()}, index=top.index)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if len(p) < 50:
        return float("nan")
    q = pd.qcut(p, min(bins, max(2, len(p) // 50)), duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] * (g["y"] - g["p"]).abs()).sum() / g["n"].sum())


def boot_ci(per_race_sum: np.ndarray, per_race_cnt: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(per_race_sum)
    if n < 20:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, (BOOT, n))
    m = per_race_sum[idx].sum(1) / np.maximum(per_race_cnt[idx].sum(1), 1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def summarize(sub: pd.DataFrame, races: pd.DataFrame, race_level: bool, rng: np.random.Generator, ref: dict | None = None) -> dict:
    n_r = int(sub["race_id"].nunique())
    cnt = sub.groupby("race_id").size()
    d_m = (sub["l_model"] - sub["l_mkt"]).groupby(sub["race_id"]).sum().reindex(cnt.index)
    d_4 = (sub["l_model"] - sub["l_v4"]).groupby(sub["race_id"]).sum().reindex(cnt.index)
    lo, hi = boot_ci(d_m.to_numpy(), cnt.to_numpy(), rng)
    lo4, hi4 = boot_ci(d_4.fillna(0).to_numpy(), cnt.to_numpy(), rng)
    out = {"n_races": n_r, "n_horses": int(len(sub)), "ll_model": float(sub["l_model"].mean()), "ll_mkt": float(sub["l_mkt"].mean()), "d_mkt": float(d_m.sum() / cnt.sum()),
           "d_mkt_lo": lo, "d_mkt_hi": hi, "d_v4": float(d_4.sum() / cnt.sum()), "d_v4_lo": lo4, "d_v4_hi": hi4,
           "ece": ece(sub["win"].to_numpy(float), sub["p_win"].to_numpy(float)), "win_rate": float(sub["win"].mean())}
    if race_level:
        r = races.loc[cnt.index]
        out.update({"hit_model": float(r["hit_model"].mean()), "hit_mkt": float(r["hit_mkt"].mean())})
    yrs = {}
    for y, g in sub.groupby("year"):
        yrs[int(y)] = float((g["l_model"] - g["l_mkt"]).mean())
    out["d_mkt_by_year"] = yrs
    vals = [v for v in yrs.values() if v == v]
    out["n_years"] = len(vals)
    if ref is None:
        out["verdict"] = "基準"
        return out
    # 市場（確定オッズ）はモデルの特徴量に無い情報を持つので、全体でも差は正になる。『全体の差より小さい/大きい』で判定する。
    rel = {y: v - ref["by_year"][y] for y, v in yrs.items() if v == v and y in ref["by_year"]}
    out["rel_mkt"] = out["d_mkt"] - ref["d_mkt"]
    out["rel_lo"], out["rel_hi"] = out["d_mkt_lo"] - ref["d_mkt"], out["d_mkt_hi"] - ref["d_mkt"]
    out["rel_by_year"] = rel
    ng, nb = sum(v < 0 for v in rel.values()), sum(v > 0 for v in rel.values())
    out["same_sign_years"] = int(max(ng, nb))
    need = max(3, len(rel) - 1)
    if out["rel_hi"] != out["rel_hi"]:
        out["verdict"] = "母数不足"
    else:
        out["verdict"] = "得意" if out["rel_hi"] < 0 and ng >= need else "苦手" if out["rel_lo"] > 0 and nb >= need else "有意でない"
    return out


RACE_DIMS = ["course", "venue", "surface", "dist_band", "track_condition", "class_b", "field_b", "season"]
HORSE_DIMS = ["age_b", "sex", "pop_b", "hist_b", "style_b"]


def tables(out: Path) -> None:
    rng = np.random.default_rng(0)
    d = load()
    d = d[d["surface"].isin(["芝", "ダート", "障"])].copy()
    races = race_table(d)
    rows = []
    allr = summarize(d, races, True, rng)
    rows.append({"dim": "全体", "value": "全体", **allr})
    ref = {"d_mkt": allr["d_mkt"], "by_year": allr["d_mkt_by_year"]}
    for dim in RACE_DIMS + HORSE_DIMS:
        for v, sub in d.groupby(dim):
            if v in ("nan", "None") or len(sub) == 0:
                continue
            rows.append({"dim": dim, "value": str(v), **summarize(sub, races, dim in RACE_DIMS, rng, ref)})
    # 平地だけの重ね合わせ（場×芝ダ×距離は course に含む）。クラス×芝ダ
    flat = d[d["surface"] != "障"]
    for (c, s), sub in flat.groupby(["class_b", "surface"]):
        rows.append({"dim": "class×surface", "value": f"{c}・{s}", **summarize(sub, races, True, rng, ref)})
    res = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "strength_tables.csv", index=False)
    for c in ("d_mkt_by_year", "rel_by_year"):
        res[c] = res[c].map(lambda x: json.dumps(x) if isinstance(x, dict) else x)
    (out / "strength_tables.json").write_text(res.to_json(orient="records", force_ascii=False, indent=1), encoding="utf-8")
    print(allr)
    print(res[res["n_races"] >= 100].sort_values("d_mkt")[["dim", "value", "n_races", "ll_model", "d_mkt", "rel_mkt", "rel_lo", "rel_hi", "d_v4", "hit_model", "same_sign_years", "verdict"]].to_string())


def _setup_matrices():
    """parent_final_2016 と同じ環境設定で、学習側の特徴量行列を作る（fit_final_v6.main の前半）。"""
    from src.scripts.data import parent_2016 as p

    p.prepare()
    for k, v in {"TP_BASE_DIR": BASE, "TP_WORK": BASE, "TP25C_OUT": BASE / "tp25c", "TP25E_OUT": BASE / "tp25e", "TP27_FAC_FIELD": BASE / "factory/field",
                 "V6_OUT": BASE / "v6", "V5_TP38_OUT": BASE / "tp38", "TP27_OUT": BASE / "tp27_v6e"}.items():
        os.environ[k] = str(v)
    os.environ.setdefault("V5_GROUPS", "CY,BEST,RB")
    os.environ.pop("V6_STAGE", None)
    os.environ["V6_TAG"] = "e16"
    from src.scripts.data import rebuild_groups_2016 as rg

    rg.redirect()
    from src.scripts.data import fit_final_v4 as f4
    from src.scripts.data import fit_final_v5 as f5
    from src.scripts.data import fit_final_v6 as f6
    from src.scripts.data import tp27_eval as t27
    from src.scripts.data import v5_common

    f4.V4 = BASE
    f6.OUT, f6.EVAL = BASE / "final_v6e16", BASE / "final_eval_v6e16"
    f6.TP27_V6 = f5.TP27_V5 = BASE / "tp27_v6e"
    t27.OUT = BASE / "tp27_v6e"
    os.environ["TP27_OUT"] = str(t27.OUT)
    v5_common.V = BASE
    f6.patch()
    df, meta, mats, A, _ = f4.build_matrices()
    names = [m[0] for m in f4.MEMBERS]
    X_all = pd.concat([mats[n][2] for n in names], axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]
    return meta, X_all


def family_of(col: str, fam: dict[str, list[str]]) -> str:
    for k, v in fam.items():
        if col in v:
            return k
    for pre, g in (("jockey_", "騎手"), ("trainer_", "調教師"), ("horse_prior", "馬の通算"), ("speed_", "スピード指数"), ("sire", "血統"), ("dam_sire", "血統"), ("ped", "血統"),
                   ("track_bias", "馬場バイアス"), ("mtl_", "多目的NN前段(mtl)"), ("oikiri", "調教(追切評価)"), ("tl_", "調教(追切)"), ("wc_", "調教(坂路/W)"), ("EASE", "追走難度"), ("lp_", "ラップ予想"), ("pp_", "位置取り予想"), ("s_", "前段S3"),
                   ("rb_", "めぐ指数(RB)"), ("meg", "めぐ指数(RB)")):
        if col.startswith(pre):
            return g
    return "レース条件・その他"


KEYF = ["s_time_gap", "s_rank_pred", "s_time_pred", "tl_n_work_28", "tl_n_hard_14", "tl_w1_exists", "oikiri_eval_rank", "mtl_t_rank_norm_gap", "mtl_t_pos_late_gap", "fc_n_jockeys", "race_class"]


def shap(out: Path) -> None:
    import lightgbm as lgb

    meta, X = _setup_matrices()
    fam = json.loads((BASE / "features/feature_families.json").read_text(encoding="utf-8"))
    bdir = BASE / "final_v6e16/bundle_v6_2024"
    models = {s.name: lgb.Booster(model_file=str(s / "lgb_win.txt")) for s in sorted((bdir / "lgb_extras").glob("seed*"))}
    names = list(next(iter(models.values())).feature_name())
    va = np.flatnonzero(meta["year"].to_numpy() == 2025)
    Xv = X.iloc[va][names]
    imp = pd.DataFrame({"feature": names, "gain": np.mean([m.feature_importance("gain") for m in models.values()], 0)})
    imp["family"] = [family_of(c, fam) for c in names]
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    out.mkdir(parents=True, exist_ok=True)
    imp.sort_values("gain", ascending=False).to_csv(out / "feature_gain.csv", index=False)
    m0 = models["seed0"]
    contrib = m0.predict(Xv, pred_contrib=True)[:, :-1]
    c = pd.DataFrame(contrib, columns=names)
    c["family_dummy"] = 0
    fams = sorted(set(imp["family"]))
    fc = pd.DataFrame({f: c[[n for n, ff in zip(names, imp["family"]) if ff == f]].sum(1).to_numpy() for f in fams})
    fc["race_id"] = meta["race_id"].to_numpy()[va]
    fc["horse_id"] = meta["horse_id"].to_numpy()[va]
    fc["missing_rate"] = Xv.isna().mean(1).to_numpy()
    fc["na_by_family"] = 0.0
    for f in fams:
        cols = [n for n, ff in zip(names, imp["family"]) if ff == f]
        fc[f"na_{f}"] = Xv[cols].isna().mean(1).to_numpy()
    for k in KEYF:
        if k in Xv.columns:
            fc[f"x_{k}"] = Xv[k].to_numpy()
    fc.drop(columns=["na_by_family"]).to_parquet(out / "family_contrib_2025.parquet", index=False)
    print(imp.sort_values("gain", ascending=False).head(30).to_string())
    print(imp.groupby("family")["gain_share"].sum().sort_values(ascending=False))


def evidence(out: Path) -> None:
    """2025 年（bundle_v6_2024 の test）で、レース単位の『市場との差』が何と結びつくかを調べる。"""
    from scipy.stats import spearmanr

    d = load()
    d = d[(d["year"] == 2025) & (d["surface"] != "障")].copy()
    fc = pd.read_parquet(out / "family_contrib_2025.parquet")
    fc["race_id"], fc["horse_id"] = fc["race_id"].astype(str), fc["horse_id"].astype(str)
    fams = [c for c in fc.columns if c not in ("race_id", "horse_id", "missing_rate") and not c.startswith(("na_", "x_"))]
    d = d.merge(fc, on=KEY, how="inner")
    tot = d[fams].abs().sum(1).replace(0, np.nan)
    for f in fams:
        d[f"sh_{f}"] = d[f].abs() / tot
    d["gap"] = d["l_model"] - d["l_mkt"]
    d["no_tl"] = (d["x_tl_n_work_28"].fillna(0) <= 0).astype(float)
    d["hist0"] = (d["horse_prior_races"].fillna(0) == 0).astype(float)
    d["na_sgap"] = d["x_s_time_gap"].isna().astype(float)
    g = d.groupby("race_id")
    r = pd.DataFrame({"gap": g["gap"].mean(), "ll_mkt": g["l_mkt"].mean(), "field": g.size(), "fav_q": g["q_win"].max(),
                      "no_tl": g["no_tl"].mean(), "hist0": g["hist0"].mean(), "na_sgap": g["na_sgap"].mean(), "missing_rate": g["missing_rate"].mean(),
                      "sgap_sd": g["x_s_time_gap"].std(), "rankpred_sd": g["x_s_rank_pred"].std(), "p_top": g["p_win"].max(), "year": g["year"].first()})
    for f in fams:
        r[f"sh_{f}"] = g[f"sh_{f}"].mean()
    rows = []
    for c in r.columns:
        if c in ("gap", "year"):
            continue
        ok = r[[c, "gap"]].dropna()
        if len(ok) > 50 and ok[c].nunique() > 1:
            rho, pv = spearmanr(ok[c], ok["gap"])
            rows.append({"race_feature": c, "rho_vs_gap": float(rho), "p": float(pv), "n_races": int(len(ok))})
    ev = pd.DataFrame(rows).sort_values("rho_vs_gap")
    ev.to_csv(out / "evidence_race_corr_2025.csv", index=False)
    print(ev.round(4).to_string())
    # 得意/苦手の条件ごとの素顔（2025 年）
    conds = {"得意:未勝利": d["class_b"] == "未勝利", "苦手:1勝": d["class_b"] == "1勝", "得意:牝": d["sex"] == "牝", "苦手:牡": d["sex"] == "牡",
             "得意:10人気〜": d["pop_b"] == "10人気〜", "苦手:2-3人気": d["pop_b"] == "2-3人気", "得意:脚質高": d["style_b"] == "a_style 高", "全体": d["race_id"].notna()}
    prof = []
    for k, m in conds.items():
        s_ = d[m]
        row = {"cond": k, "n_horses": int(len(s_)), "n_races": int(s_["race_id"].nunique()), "gap": float(s_["gap"].mean()), "ll_mkt": float(s_["l_mkt"].mean()),
               "no_tl": float(s_["no_tl"].mean()), "hist0": float(s_["hist0"].mean()), "na_sgap": float(s_["na_sgap"].mean()), "missing_rate": float(s_["missing_rate"].mean())}
        for f in fams:
            row[f"sh_{f}"] = float(s_[f"sh_{f}"].mean())
        prof.append(row)
    pf = pd.DataFrame(prof)
    pf.to_csv(out / "evidence_condition_profile_2025.csv", index=False)
    print(pf.round(4).T.to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tables", "shap", "evidence"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"tables": tables, "shap": shap, "evidence": evidence}[a.cmd](Path(a.out))


if __name__ == "__main__":
    main()

"""tp-61: 前日〜T-60 に分かる量から『展開タイプ（ペース × 隊列の凝縮度）』を予測できるか。

python -m src.scripts.data.tp61_pre build   # レース単位の事前特徴（想定位置取りの分布・コースの履歴・条件）
python -m src.scripts.data.tp61_pre eval    # 学習 2020〜2023 → 検証 2024・2025 の分類精度（ベースライン比較）
入力: tp53（想定位置取り pf_*）, tp61（実測の展開 race_cls）, tp55（ラップ予想の確率、2022〜2025）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/local/modeling/tp61"
TP53 = ROOT / "data/local/modeling/tp53"
TP55 = Path("/home/jovyan/work/tp55_out")
PACE = ["slow", "mid", "high"]
SHAPE = ["compact", "normal", "stretched"]
JOINT = [f"{p}|{s}" for p in PACE for s in SHAPE]

RACE_FEATS = ["n_known", "known_frac", "e1", "e2", "e3", "e4", "e5", "d12", "d15", "std5", "n_le15", "n_le25", "n_le35", "n_le50",
              "near_lead10", "mean_e", "std_e", "q25", "q50", "q75", "pf_front", "pf_stalker", "pf_pressure",
              "field_size", "distance", "turf", "heavy", "venue_code", "grade_code",
              "crs_gap", "crs_bal", "crs_t3", "crs_lead", "crs_n"]


def race_pf_features(h: pd.DataFrame) -> pd.DataFrame:
    """h: race_id, pf_early（想定位置取り: 0=先頭〜1=後方、欠損あり）。レースごとの分布の特徴。"""
    rows = []
    for rid, g in h.groupby("race_id", sort=False):
        e = np.sort(g["pf_early"].dropna().to_numpy(float))
        n = len(e)
        rec = {"race_id": rid, "n_known": n}
        if n >= 4:
            pad = np.pad(e[:5], (0, max(0, 5 - len(e[:5]))), constant_values=np.nan)
            rec.update({f"e{i + 1}": pad[i] for i in range(5)})
            rec["d12"] = e[1] - e[0]
            rec["d15"] = e[min(4, n - 1)] - e[0]
            rec["std5"] = float(np.std(e[:5]))
            rec["n_le15"] = int((e <= 0.15).sum())
            rec["n_le25"] = int((e <= 0.25).sum())
            rec["n_le35"] = int((e <= 0.35).sum())
            rec["n_le50"] = int((e <= 0.50).sum())
            rec["near_lead10"] = int((e <= e[0] + 0.10).sum())
            rec["mean_e"], rec["std_e"] = float(e.mean()), float(e.std())
            rec["q25"], rec["q50"], rec["q75"] = [float(np.percentile(e, q)) for q in (25, 50, 75)]
        rows.append(rec)
    return pd.DataFrame(rows)


def course_asof(r: pd.DataFrame, cols: list[str], shrink: float = 20.0) -> pd.DataFrame:
    """コース（場×面×距離）ごとに、日付が厳密に前のレースだけの平均（全体平均へ縮約）。"""
    r = r.sort_values(["date", "race_id"]).reset_index(drop=True)
    r["ck"] = r["venue"].astype(str) + "|" + r["surface"].astype(str) + "|" + r["distance"].astype(int).astype(str)
    out = {}
    for c in cols:
        v = pd.to_numeric(r[c], errors="coerce")
        ok = v.notna().astype(float)
        vv = v.fillna(0.0)
        # 日ごとに合計 → 日付順に累積して 1 日ずらす（同じ日の結果は入れない）
        daily = pd.DataFrame({"ck": r["ck"], "date": r["date"], "s": vv, "n": ok}).groupby(["ck", "date"], sort=True)[["s", "n"]].sum().reset_index()
        daily["cs"] = daily.groupby("ck")["s"].cumsum() - daily["s"]
        daily["cn"] = daily.groupby("ck")["n"].cumsum() - daily["n"]
        r = r.merge(daily[["ck", "date", "cs", "cn"]].rename(columns={"cs": f"_cs_{c}", "cn": f"_cn_{c}"}), on=["ck", "date"], how="left")
        # 全体（日ごとの累積）
        gd = pd.DataFrame({"date": r["date"], "s": vv, "n": ok}).groupby("date")[["s", "n"]].sum().reset_index().sort_values("date")
        gd["gs"] = gd["s"].cumsum() - gd["s"]
        gd["gn"] = gd["n"].cumsum() - gd["n"]
        r = r.merge(gd[["date", "gs", "gn"]].rename(columns={"gs": f"_gs_{c}", "gn": f"_gn_{c}"}), on="date", how="left")
        gm = (r[f"_gs_{c}"] / r[f"_gn_{c}"].replace(0, np.nan)).fillna(v.mean())
        n = r[f"_cn_{c}"].fillna(0.0)
        out[c] = ((r[f"_cs_{c}"].fillna(0.0) + shrink * gm) / (n + shrink)).to_numpy()
        out[c + "_n"] = n.to_numpy()
    o = pd.DataFrame(out)
    o["race_id"] = r["race_id"].to_numpy()
    return o


W5 = (0.35, 0.25, 0.20, 0.12, 0.08)


def pf_from_results(years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """指定年の出走馬の想定位置取りを、結果表の過去走（直前 5 走の最初のコーナーの位置の重み付き平均）から作る。
    race_shutuba_past の表が無い年（2026）用。枠順の補正は入れない。2020〜2025 年の tp53 の値との一致は eval_pf で確認する。"""
    h = pd.read_parquet(D / "horse.parquet", columns=["race_id", "date", "horse_id", "horse_number", "early_norm", "field_size", "year"])
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    g = h.groupby("horse_id", sort=False)["early_norm"]
    num = np.zeros(len(h))
    den = np.zeros(len(h))
    n = np.zeros(len(h))
    for k, w in enumerate(W5, 1):
        v = g.shift(k).to_numpy()
        ok = ~np.isnan(v)
        num += np.where(ok, v, 0.0) * w
        den += ok * w
        n += ok
    h["pf_early"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    h["pf_style"] = pd.cut(h["pf_early"], [-1, 0.15, 0.35, 0.65, 2], labels=["front", "stalker", "mid", "closer"]).astype(object)
    h = h[h["year"].isin(years)]
    rows = []
    for rid, gg in h.groupby("race_id", sort=False):
        field = int(gg["field_size"].iloc[0])
        known = gg["pf_early"].notna().sum()
        front = int((gg["pf_style"] == "front").sum())
        stalker = int((gg["pf_style"] == "stalker").sum())
        pressure = (front + stalker) / field if field else 0.0
        if known < max(4, field * 0.5):
            level = "unknown"
        elif front >= 3 or (front >= 2 and pressure >= 0.4):
            level = "high"
        elif front == 0 or (front == 1 and pressure < 0.35):
            level = "slow"
        else:
            level = "mid"
        rows.append({"race_id": rid, "pf_level": level, "pf_front": front, "pf_stalker": stalker, "pf_pressure": pressure})
    return h[["race_id", "horse_number", "pf_early", "pf_style"]], pd.DataFrame(rows)


def _pf_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """想定位置取り（as-of）。2020〜2025 は tp53 の出力、2026 は結果表の過去走から作る。"""
    hp = pd.read_parquet(TP53 / "horse.parquet", columns=["race_id", "horse_number", "pf_early", "pf_style"])
    rp = pd.read_parquet(TP53 / "race.parquet")[["race_id", "pf_level", "pf_front", "pf_stalker", "pf_pressure"]]
    h6, r6 = pf_from_results([2026])
    return pd.concat([hp, h6], ignore_index=True), pd.concat([rp, r6], ignore_index=True)


def eval_pf() -> int:
    """結果表から作る想定位置取りと tp53（race_shutuba_past 由来、枠順補正あり）の一致（2025 年）。"""
    a, ra = pf_from_results([2025])
    b = pd.read_parquet(TP53 / "horse.parquet", columns=["race_id", "horse_number", "pf_early", "year"])
    b = b[b["year"] == 2025]
    m = a.merge(b, on=["race_id", "horse_number"], suffixes=("_res", "_tp53")).dropna(subset=["pf_early_res", "pf_early_tp53"])
    print("pf 一致（2025）: n", len(m), "相関", round(float(m["pf_early_res"].corr(m["pf_early_tp53"])), 3), "平均絶対差", round(float((m["pf_early_res"] - m["pf_early_tp53"]).abs().mean()), 3))
    r = ra.merge(pd.read_parquet(TP53 / "race.parquet")[["race_id", "pf_level"]], on="race_id", suffixes=("_res", "_tp53"))
    print("pf_level 一致率", round(float((r["pf_level_res"] == r["pf_level_tp53"]).mean()), 3))
    return 0


def cmd_build() -> int:
    rc = pd.read_parquet(D / "race_cls.parquet")
    hp, rp = _pf_all()
    f = race_pf_features(hp)
    cas = course_asof(rc.rename(columns={"c4_gap_per_horse": "gp", "c4_lead_group_frac": "lf"}), ["gp", "bal", "t3f_dev", "lf"])
    cas = cas.rename(columns={"gp": "crs_gap", "bal": "crs_bal", "t3f_dev": "crs_t3", "lf": "crs_lead", "gp_n": "crs_n"}).drop(columns=["bal_n", "t3f_dev_n", "lf_n"])
    base = rc[["race_id", "date", "year", "venue", "surface", "distance", "going_group", "field_size", "grade", "race_class", "band", "pace_cls", "shape_cls", "bal", "t3f_dev",
               "race_l3f_dev", "c4_gap_per_horse", "c4_lead_group_frac"]].copy()
    d = base.merge(f, on="race_id", how="inner").merge(rp, on="race_id", how="left").merge(cas, on="race_id", how="left")
    hn = D / "hana.parquet"
    if hn.exists():
        d = d.merge(pd.read_parquet(hn), on="race_id", how="left")
    tx = D / "taxon.parquet"
    if tx.exists():
        d = d.merge(pd.read_parquet(tx)[["race_id", "lap_type"]], on="race_id", how="left")
    d["turf"] = (d["surface"] == "芝").astype(float)
    d["heavy"] = (d["going_group"] == "道悪").astype(float)
    vc = {v: i for i, v in enumerate(sorted(d["venue"].dropna().unique()))}
    d["venue_code"] = d["venue"].map(vc)
    gr = {v: i for i, v in enumerate(sorted(d["grade"].fillna("").astype(str).unique()))}
    d["grade_code"] = d["grade"].fillna("").astype(str).map(gr)
    d["joint"] = d["pace_cls"].astype(str) + "|" + d["shape_cls"].astype(str)
    d.to_parquet(D / "pre.parquet", index=False)
    (D / "pre_meta.json").write_text(json.dumps({"venue_code": vc, "grade_code": gr}, ensure_ascii=False), encoding="utf-8")
    print(d.shape, d["joint"].value_counts().to_dict())
    return 0


def load_tp55() -> pd.DataFrame:
    fs = []
    for y in (2022, 2023, 2024, 2025):
        p = TP55 / f"main_A_full_{y}_pred.parquet"
        if p.exists():
            fs.append(pd.read_parquet(p)[["race_id", "p0", "p1", "p2"]])
    return pd.concat(fs) if fs else pd.DataFrame(columns=["race_id", "p0", "p1", "p2"])


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum(1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


HANA = ["hn_claim", "hn_claim_strong", "hn_claim_vstrong", "hn_l1", "hn_l2", "hn_l3", "hn_sum3", "hn_pairs", "hn_outer", "hn_inner", "hn_minrel",
        "hn_front5_n", "hn_firstcurve", "hn_start_geom", "hn_uncertain"]
LGB_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                  n_jobs=3, verbose=-1, random_state=0)
TAXON = ["slow_kick", "slow_spurt", "even_kick", "even_flat", "high_breather", "high_collapse"]


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """最大確率のクラスについての ECE（確信度の区間ごとの平均確率と的中率の差の加重平均）。"""
    conf, pred = p.max(1), p.argmax(1)
    ok = (pred == y).astype(float)
    e, n = 0.0, len(y)
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf >= lo) & (conf < lo + 1 / bins)
        if m.any():
            e += m.sum() / n * abs(conf[m].mean() - ok[m].mean())
    return float(e)


def _fit_lgb(X, y, k, rounds=300):
    import lightgbm as lgb

    return lgb.LGBMClassifier(objective="multiclass", num_class=k, n_estimators=rounds, **LGB_PARAMS).fit(X, y)


def cmd_eval() -> int:
    from sklearn.linear_model import LogisticRegression

    d = pd.read_parquet(D / "pre.parquet")
    lap = load_tp55()
    d = d.merge(lap, on="race_id", how="left")
    d = d[d["pace_cls"].isin(PACE) & d["shape_cls"].isin(SHAPE)].copy()
    tr = d[d["year"].isin([2020, 2021, 2022, 2023])]
    va = d[d["year"].isin([2024, 2025])]
    feats = [c for c in RACE_FEATS if c in d.columns]
    hana = [c for c in HANA if c in d.columns]
    res: dict = {}
    targets = {"pace": (PACE, "pace_cls"), "shape": (SHAPE, "shape_cls"), "joint": (JOINT, "joint")}
    if "lap_type" in d.columns:
        targets["taxon"] = (TAXON, "lap_type")
    preds = {}
    for name, (classes, col) in targets.items():
        sub_tr, sub_va = tr[tr[col].isin(classes)], va[va[col].isin(classes)]
        ytr = sub_tr[col].map({c: i for i, c in enumerate(classes)}).to_numpy()
        yva = sub_va[col].map({c: i for i, c in enumerate(classes)}).to_numpy()
        prior = np.bincount(ytr, minlength=len(classes)) / len(ytr)
        res[f"{name}/prior"] = _ll(yva, np.tile(prior, (len(yva), 1)))
        lv = pd.get_dummies(sub_tr["pf_level"].fillna("unknown")).reindex(columns=["slow", "mid", "high", "unknown"], fill_value=0).astype(float)
        lvv = pd.get_dummies(sub_va["pf_level"].fillna("unknown")).reindex(columns=["slow", "mid", "high", "unknown"], fill_value=0).astype(float)
        pb = LogisticRegression(max_iter=500).fit(lv, ytr).predict_proba(lvv)
        res[f"{name}/pf_level_only"] = _ll(yva, pb)
        res[f"{name}/pf_level_only_acc"] = float((pb.argmax(1) == yva).mean())
        for tag, fs in (("base", feats), ("+hana", feats + hana)):
            m = _fit_lgb(sub_tr[fs], ytr, len(classes))
            p = m.predict_proba(sub_va[fs])
            res[f"{name}/lgbm[{tag}]"] = _ll(yva, p)
            res[f"{name}/lgbm[{tag}]_acc"] = float((p.argmax(1) == yva).mean())
            res[f"{name}/lgbm[{tag}]_ece"] = _ece(yva, p)
            preds[(name, tag)] = (sub_va["race_id"].to_numpy(), p)
            if name == "pace":
                ok = sub_va["p0"].notna().to_numpy()
                res["pace/tp55_lap"] = _ll(yva[ok], sub_va[["p0", "p1", "p2"]].to_numpy()[ok])
                res["pace/tp55_lap_acc"] = float((sub_va[["p0", "p1", "p2"]].to_numpy()[ok].argmax(1) == yva[ok]).mean())
                res[f"pace/lgbm[{tag}]_same_rows_as_tp55"] = _ll(yva[ok], p[ok])
                tr2 = sub_tr[sub_tr["p0"].notna()]
                y2 = tr2[col].map({c: i for i, c in enumerate(classes)}).to_numpy()
                f2 = fs + ["p0", "p1", "p2"]
                p3 = _fit_lgb(tr2[f2], y2, 3).predict_proba(sub_va[f2])
                res[f"pace/lgbm[{tag}]+tp55 (train22-23)"] = _ll(yva, p3)
    # 独立の積（pace × shape）と joint の比較
    ids_p, pp = preds[("pace", "base")]
    ids_s, ps = preds[("shape", "base")]
    mm = pd.DataFrame({"race_id": ids_p}).merge(pd.DataFrame({"race_id": ids_s}), on="race_id")
    pace_i = pd.Series(range(len(ids_p)), index=ids_p)[mm["race_id"]].to_numpy()
    shape_i = pd.Series(range(len(ids_s)), index=ids_s)[mm["race_id"]].to_numpy()
    prod = (pp[pace_i][:, :, None] * ps[shape_i][:, None, :]).reshape(len(mm), 9)
    vv = va.set_index("race_id").loc[mm["race_id"]]
    yj = vv["joint"].map({c: i for i, c in enumerate(JOINT)}).to_numpy()
    res["joint/product_of_pace_and_shape"] = _ll(yj, prod)
    res["joint/product_acc"] = float((prod.argmax(1) == yj).mean())
    for k, v in res.items():
        print(f"{k:44s} {v:.5f}")
    (D / "pre_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval", "evalpf"])
    a = ap.parse_args()
    return {"build": cmd_build, "eval": cmd_eval, "evalpf": eval_pf}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())

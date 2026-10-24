"""tp-77 追加4（探索）: 馬体重と 面×距離 の関係の分析と、特徴量化の可否（v6e の OOF に対する増分）。学習には使わない。

python -m src.scripts.data.ptrain_weight_explore   # PTRAIN_OUT/weight_explore.json と要約
すべてレース開催日より前の自分の過去走だけから作る（as-of）。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain"))
TABLES = Path(os.environ.get("PTRAIN_TABLES", ROOT / "data/page_reference/tables"))
OOF = Path(os.environ.get("PTRAIN_OOF", ROOT / "data/local/modeling/training_pipeline_v4/v6/eval/oof_z:RB@e.parquet"))
KEY = ["race_id", "horse_id"]
BANDS = [0, 1200, 1500, 1900, 2400, 99999]


def load() -> pd.DataFrame:
    cols = ["race_id", "date", "horse_id", "distance", "surface", "field_size", "finish_position", "weight", "weight_change", "sex_age"]
    r = pd.concat([pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=cols) for y in range(2019, 2026)], ignore_index=True)
    r = r[r["surface"].isin(["芝", "ダート"])].copy()
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    r["date"] = pd.to_datetime(r["date"])
    for c in ("distance", "field_size", "finish_position", "weight", "weight_change"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.dropna(subset=["weight", "distance", "finish_position", "field_size"])
    r = r[r["weight"].between(300, 650)]
    r["sex"], r["age"] = r["sex_age"].str.slice(0, 1), pd.to_numeric(r["sex_age"].str.slice(1), errors="coerce")
    r["dband"] = pd.cut(r["distance"], BANDS, labels=False)
    r["res"] = (r["field_size"] - r["finish_position"]) / (r["field_size"] - 1).replace(0, np.nan) - 0.5
    return r.sort_values(["horse_id", "date"]).reset_index(drop=True)


def asof_feats(r: pd.DataFrame) -> pd.DataFrame:
    """馬ごとに、より前の走だけから作る体重の特徴量。"""
    n = len(r)
    names = ["w_rel_self", "w_rel_self_z", "w_rel_best", "w_rel_dist", "w_rel_surf", "w_dslope_pred", "w_n"]
    out = np.full((n, len(names)), np.nan, dtype="float32")
    for _, idx in r.groupby("horse_id", sort=False).indices.items():
        w = r["weight"].to_numpy(float)[idx]; d = r["date"].to_numpy("datetime64[D]")[idx]
        dist = r["distance"].to_numpy(float)[idx]; sf = (r["surface"].to_numpy()[idx] == "芝"); res = r["res"].to_numpy(float)[idx]
        for k in range(1, len(idx)):
            m = d[:k] < d[k]
            if m.sum() < 1:
                continue
            pw, pd_, ps, pr = w[:k][m], dist[:k][m], sf[:k][m], res[:k][m]
            row = out[idx[k]]
            row[6] = len(pw)
            mu = pw.mean(); row[0] = w[k] - mu
            if len(pw) >= 3 and pw.std() > 0:
                row[1] = (w[k] - mu) / max(pw.std(), 2.0)
            good = pr >= 0.2
            if good.sum() >= 1:
                row[2] = w[k] - pw[good].mean()
            near = np.abs(pd_ - dist[k]) <= 200
            if near.sum() >= 1:
                row[3] = w[k] - pw[near].mean()
            same = ps == sf[k]
            if same.sum() >= 1:
                row[4] = w[k] - pw[same].mean()
            if len(pw) >= 4 and np.ptp(pd_) >= 400:
                b = np.polyfit(pd_, pw, 1); row[5] = w[k] - np.polyval(b, dist[k])
    return pd.DataFrame(out, columns=names, index=r.index)


def main() -> None:
    r = load()
    F = asof_feats(r)
    r = pd.concat([r, F], axis=1)
    g = r.groupby(["race_id"])["weight"]
    r["w_rel_field"] = r["weight"] - g.transform("mean")
    pri = r[r["date"] <= "2021-12-21"].groupby(["sex", "age", "surface", "dband"])["weight"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "_m", "std": "_s"})
    r = r.merge(pri, on=["sex", "age", "surface", "dband"], how="left")
    r["w_rel_sas"] = (r["weight"] - r["_m"]) / r["_s"].clip(lower=8)
    wcols = ["w_rel_self", "w_rel_self_z", "w_rel_best", "w_rel_dist", "w_rel_surf", "w_dslope_pred", "w_rel_field", "w_rel_sas", "w_n"]
    ex = r[KEY + wcols].rename(columns={c: f"P77_W_{c[2:]}" for c in wcols})
    ex.astype({c: "float32" for c in ex.columns if c.startswith("P77_W_")}).to_parquet(OUT / "weight_rows.parquet", index=False)
    if os.environ.get("PTRAIN_WEIGHT_EXPORT_ONLY"):
        return
    rep: dict = {}
    # A. 体重の水準: 面×距離帯×性別
    rep["weight_level"] = r.groupby(["surface", "dband", "sex"])["weight"].agg(["mean", "std", "size"]).round(1).reset_index().to_dict("records")
    # B. 成績との関係（面×距離帯ごとの、相対体重の五分位別の平均成績 res）
    feats = ["w_rel_self", "w_rel_self_z", "w_rel_best", "w_rel_dist", "w_rel_surf", "w_dslope_pred", "w_rel_field", "w_rel_sas", "weight_change"]
    tab = {}
    for (sf, db), x in r[r["date"].dt.year >= 2020].groupby(["surface", "dband"]):
        for f in feats:
            y = x.dropna(subset=[f])
            if len(y) < 3000:
                continue
            q = pd.qcut(y[f], 5, labels=False, duplicates="drop")
            m = y.groupby(q)["res"].mean()
            tab[f"{sf}|{int(db)}|{f}"] = {"top_minus_bottom": round(float(m.iloc[-1] - m.iloc[0]), 4), "spearman": round(float(y[[f, "res"]].corr(method="spearman").iloc[0, 1]), 4), "n": int(len(y))}
    rep["by_surface_dist"] = tab
    # C. v6e の OOF に対する増分（残差への当てはめ。年をまたぐ: 2022→2023→2024 は前の年で学習）
    from sklearn.ensemble import HistGradientBoostingClassifier as H

    oof = pd.read_parquet(OOF)
    oof["race_id"], oof["horse_id"] = oof["race_id"].astype(str), oof["horse_id"].astype(str)
    pc = [c for c in oof.columns if c.startswith("p_")]
    d = oof.merge(r, on=KEY, how="inner")
    d["year"] = d["date"].dt.year
    d["surf"] = (d["surface"] == "芝").astype(int)
    X = ["weight", "weight_change", "distance", "surf"] + feats[:-1]
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    inc = {}
    for tg, col in (("win", "p_win"), ("top3", "p_top3")):
        if col not in d.columns:
            continue
        y = (d["finish_position"] <= (1 if tg == "win" else 3)).astype(int)
        rows = []
        for vy in (2023, 2024):                      # 学習は valid より前の OOF 年だけ
            tr, te = d["year"] < vy, d["year"] == vy
            if tr.sum() < 20000:
                continue
            off_tr, off_te = lg(d.loc[tr, col]), lg(d.loc[te, col])
            m = H(max_iter=120, learning_rate=0.05, max_depth=3, l2_regularization=10.0, random_state=0)
            # 残差モデル: 2 値分類の予測を v6e の logit に足す（簡易: v6e の logit を特徴量に含める）
            Xt = np.column_stack([off_tr, d.loc[tr, X]]); Xe = np.column_stack([off_te, d.loc[te, X]])
            m.fit(Xt, y[tr]); p = np.clip(m.predict_proba(Xe)[:, 1], 1e-6, 1 - 1e-6)
            m0 = H(max_iter=120, learning_rate=0.05, max_depth=3, l2_regularization=10.0, random_state=0).fit(off_tr.to_numpy().reshape(-1, 1), y[tr])
            p0 = np.clip(m0.predict_proba(off_te.to_numpy().reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
            ll = lambda p_: float(-(y[te] * np.log(p_) + (1 - y[te]) * np.log(1 - p_)).mean())
            rows.append({"valid": vy, "ll_base": round(ll(p0), 5), "ll_with_weight": round(ll(p), 5), "delta": round(ll(p) - ll(p0), 5)})
        inc[tg] = rows
    rep["increment_over_v6e"] = inc
    (OUT / "weight_explore.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    top = sorted(tab.items(), key=lambda kv: -abs(kv[1]["top_minus_bottom"]))[:12]
    print("INCREMENT", json.dumps(inc, ensure_ascii=False))
    for k, v in top:
        print("TOP", k, v)
    print("COVER", {f: round(float(r[f].notna().mean()), 3) for f in feats})


if __name__ == "__main__":
    main()

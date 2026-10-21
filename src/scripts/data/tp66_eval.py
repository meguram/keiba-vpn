"""tp-66 めぐ指数の集約方式の比較（ホストで実行）。

python -m src.scripts.data.tp66_eval build      # 特徴量（as-of）を作る → <OUT>/feat.parquet
python -m src.scripts.data.tp66_eval score      # 指標ごとの予測力（レース内の順位・印の lift・V5 に対する追加の説明力）→ <OUT>/score_table.json
python -m src.scripts.data.tp66_eval rules      # マーキング規則の比較（選んだ指標）→ <OUT>/rules.json

期間: 選択 2022〜2023、確認 2024、最終 2025。2026 年はテストセット②（最後に選んだ規則だけを 1 回確認）。
モデルの確率は 2022〜2024 年が V5（LightGBM 単体 d_CY_BEST）の out-of-fold、2025〜2026 年は tp59 master の p_*（最終モデルの予測）。
市場の値（人気・複勝払戻）は『人気以上の好走』と回収率の評価にだけ使い、指標には使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.inference import class_benchmark as cb
from src.scripts.data import class_benchmark_build as cbb
from src.scripts.data import tp66_lib as L

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
OUT = Path(os.environ.get("TP66_OUT", "/home/jovyan/work/tp66_out"))
MASTER = Path(os.environ.get("TP59_OUT", "/home/jovyan/work/tp59_out")) / "master.parquet"
OOF = ROOT / "data/local/modeling/training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet"
VALUE = cbb.VALUE
YEARS = list(range(2020, 2027))
PERIODS = {"sel": (2022, 2023), "conf": (2024, 2024), "final": (2025, 2025), "t2": (2026, 2026)}
NON_SCORE = {"line", "cond_n"}


def load_runs(years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        f = TABLES / str(y) / "megu_index_flat.parquet"
        if not f.exists():
            continue
        m = pd.read_parquet(f, columns=["race_id", "horse_id", VALUE, "megu_v3a_status", "megu_v3a_low_confidence"])
        r = pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet",
                            columns=["race_id", "horse_id", "race_class", "grade", "race_name", "surface", "distance", "date", "finish_position", "field_size", "horse_number"])
        frames.append(m.merge(r, on=["race_id", "horse_id"], how="inner"))
    df = pd.concat(frames, ignore_index=True)
    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    df["idx"] = df[VALUE]
    df["low"] = df["megu_v3a_low_confidence"].fillna(False).astype(bool)
    df["valid"] = (df["megu_v3a_status"] == "ok") & df["idx"].notna()
    df["ok"] = df["valid"] & ~df["low"]
    df["year"] = df["race_id"].str[:4].astype(int)
    df["fin"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"].astype(str).str.replace("/", "-"))
    df = df.drop_duplicates(["race_id", "horse_id"])
    win = df[df["valid"] & (df["fin"] == 1)].groupby("race_id")["idx"].max().rename("win_idx")
    df = df.merge(win, left_on="race_id", right_index=True, how="left")
    df["idx_fl"] = np.where(df["win_idx"].notna(), np.maximum(df["idx"], df["win_idx"] - 40.0), df["idx"])
    return cbb.add_context(df)


def race_lines(df: pd.DataFrame, basis: str = "second") -> dict[str, float | None]:
    """評価年ごとに、その年より前の年だけで作った突破ライン表を使い、レースごとの突破ライン（p25）を引く。"""
    lines: dict[str, float | None] = {}
    races = df.drop_duplicates("race_id")
    for y, g in races.groupby("year"):
        base = cbb.basis_rows(df[df["year"] < y], basis)
        doc = {"cells": cbb.build_cells(base)} if len(base) else {"cells": {}}
        for r in g.itertuples():
            hit = cb.lookup(r.race_class, r.grade, r.race_name, r.surface, r.distance, doc)
            lines[r.race_id] = hit["line"] if hit else None
    return lines


def build(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_runs(YEARS)
    lines = race_lines(df)
    print("runs", len(df), "races", df.race_id.nunique(), "lines", sum(v is not None for v in lines.values()), flush=True)
    feats = L.compute_features(df, lines, min_year=2022)
    feats["v3b"] = L.v3b_feature(df).where(df["year"] >= 2022)
    ln = df["race_id"].map(lines)
    feats["v3b_margin"] = feats["v3b"] - ln
    out = pd.concat([df[["race_id", "horse_id", "year", "date", "idx", "idx_fl", "ok", "tier", "age", "surf", "band", "fin", "field_size", "horse_number"]], feats], axis=1)
    out["has_hist"] = feats["last1"].notna() if "last1" in feats else False
    out.to_parquet(OUT / "feat.parquet")
    print("saved", out.shape, "hist share (2022+):", float(out.loc[out.year >= 2022, "has_hist"].mean()))
    return 0


def attach_outcomes(feat: pd.DataFrame) -> pd.DataFrame:
    """feat に結果・市場・モデルの確率を付ける。モデルは 2022〜2024 が V5 の OOF、2025〜2026 が master の p_*。"""
    m = pd.read_parquet(MASTER, columns=["race_id", "horse_id", "win", "top3", "win_odds", "popularity", "ret_win", "ret_place", "p_win", "p_top2", "p_top3", "q_win", "field_size", "horse_number"])
    m["race_id"] = m["race_id"].astype(str)
    m["horse_id"] = m["horse_id"].astype(str)
    m = m.rename(columns={"field_size": "n_run", "horse_number": "hn"})
    df = feat.merge(m, on=["race_id", "horse_id"], how="inner")
    o = pd.read_parquet(OOF)
    o["race_id"] = o["race_id"].astype(str)
    o["horse_id"] = o["horse_id"].astype(str)
    o = o.rename(columns={"p_win": "v5_win", "p_top2": "v5_top2", "p_top3": "v5_top3"})[["race_id", "horse_id", "v5_win", "v5_top2", "v5_top3"]]
    df = df.merge(o, on=["race_id", "horse_id"], how="left")
    use_oof = df["year"].between(2022, 2024)
    for k in ("win", "top2", "top3"):
        df[f"m_{k}"] = np.where(use_oof, df[f"v5_{k}"], df[f"p_{k}"])
    n = df.groupby("race_id")["horse_id"].transform("count")
    df["n_in_race"] = n
    df["exp_t3"] = np.minimum(3.0 / n, 1.0)
    df["exp_w"] = 1.0 / n
    df["place_pay"] = pd.to_numeric(df["ret_place"], errors="coerce").fillna(0.0) / 100.0
    df["win_pay"] = pd.to_numeric(df["ret_win"], errors="coerce").fillna(0.0) / 100.0
    df["longshot"] = pd.to_numeric(df["popularity"], errors="coerce") >= 6
    return df


def period_mask(df: pd.DataFrame, p: str) -> pd.Series:
    lo, hi = PERIODS[p]
    return df["year"].between(lo, hi)


def L_FEATURE_NAMES(df: pd.DataFrame) -> list[str]:
    base = ["last1", "mean2", "max2", "med2", "mean3", "max3", "med3", "mean5", "max5", "med5", "mean8", "max8", "med8", "hub5", "ew5", "ew5_fl", "mean5_fl", "ewhuber5", "ewhuber5_fl", "v3b", "top2of8", "top3of8",
            "trend", "cond_mean", "cond_max", "cond_mean3", "cond_max3", "fitw5", "max5_margin", "mean5_margin", "ew5_margin", "ew5_fl_margin", "ewhuber5_fl_margin", "v3b_margin", "n_above5", "frac_above5",
            "cond_max_margin", "cond_mean_margin", "last1_margin", "top2of8_margin"]
    return [c for c in base if c in df.columns]


def mark_stats(g: pd.DataFrame, flag: pd.Series) -> dict:
    s = g[flag]
    if not len(s):
        return {"n": 0}
    return {"n": int(len(s)), "races": int(s.race_id.nunique()), "top3": round(float(s.top3.mean()), 4), "win": round(float(s.win.mean()), 4),
            "lift_top3": round(float(s.top3.sum() / s.exp_t3.sum()), 3), "lift_win": round(float(s.win.sum() / s.exp_w.sum()), 3),
            "place_roi": round(float(s.place_pay.mean()), 3), "win_roi": round(float(s.win_pay.mean()), 3),
            "longshot_share_of_top3": round(float((s.top3 * s.longshot).sum() / max(s.top3.sum(), 1)), 3)}


def rank_auc(g: pd.DataFrame, col: str, target: str) -> float | None:
    from sklearn.metrics import roc_auc_score

    x = g[g[col].notna()]
    if x[target].nunique() < 2 or len(x) < 200:
        return None
    pct = x.groupby("race_id")[col].rank(pct=True)
    return round(float(roc_auc_score(x[target], pct)), 4)


def v5_added(df: pd.DataFrame, col: str, target: str, pcol: str) -> dict:
    """V5 の確率のロジットに、レース内の z（欠損は 0 + 欠損フラグ）を足すと LogLoss がどれだけ下がるか。
    2022〜2024 は年を 1 つ外す学習（LOYO）。2025 は 2022〜2024 で学習して当てる。差は 追加あり − 基準（負が良い）。"""
    d = df[df[pcol].notna() & df["year"].between(2022, 2025)].copy()
    d["z"] = L.within_race_z(d, col)
    d["zmiss"] = d["z"].isna().astype(float)
    d["z0"] = d["z"].fillna(0.0)
    d["lp"] = L.logit(d[pcol].to_numpy())
    res = {}
    for label, years, trainer in (("loyo_2022_24", [2022, 2023, 2024], "loyo"), ("final_2025", [2025], "fwd")):
        ll_a, ll_b, rids, coefs = [], [], [], []
        for y in years:
            tr = d[d["year"].between(2022, 2024) & (d["year"] != y)]
            te = d[d["year"] == y]
            wa = L.fit_logit(tr[["lp"]].to_numpy(), tr[target].to_numpy(dtype=float))
            wb = L.fit_logit(tr[["lp", "z0", "zmiss"]].to_numpy(), tr[target].to_numpy(dtype=float))
            pa = L.predict_logit(wa, te[["lp"]].to_numpy())
            pb = L.predict_logit(wb, te[["lp", "z0", "zmiss"]].to_numpy())
            yv = te[target].to_numpy(dtype=float)
            e = 1e-6
            la = -(yv * np.log(np.clip(pa, e, 1 - e)) + (1 - yv) * np.log(np.clip(1 - pa, e, 1)))
            lb = -(yv * np.log(np.clip(pb, e, 1 - e)) + (1 - yv) * np.log(np.clip(1 - pb, e, 1)))
            ll_a.append(la); ll_b.append(lb); rids.append(te["race_id"].to_numpy()); coefs.append(float(wb[2]))
        la, lb, rid = np.concatenate(ll_a), np.concatenate(ll_b), np.concatenate(rids)
        mean, lo, hi = L.race_bootstrap_delta(rid, lb, la, n_boot=500)
        res[label] = {"delta_ll": round(mean, 5), "ci": [round(lo, 5), round(hi, 5)], "z_coef": round(float(np.mean(coefs)), 4)}
    return res


def score(a) -> int:
    feat = pd.read_parquet(OUT / "feat.parquet")
    df = attach_outcomes(feat)
    df = df[df["n_in_race"] >= 6]
    df["ref_v5_top3"] = df["m_top3"]
    df["ref_market"] = -pd.to_numeric(df["popularity"], errors="coerce")        # 人気が高いほど大きい（参考。指標ではない）
    cols = [c for c in L_FEATURE_NAMES(df)] + ["ref_v5_top3", "ref_market"]
    table = {}
    for c in cols:
        row = {"cover": {}, "auc_top3": {}, "top1": {}, "top2": {}, "top3": {}}
        for p in PERIODS:
            g = df[period_mask(df, p)]
            if not len(g):
                continue
            row["cover"][p] = round(float(g[c].notna().mean()), 3)
            row["auc_top3"][p] = rank_auc(g, c, "top3")
            for k in (1, 2, 3):
                flag = L.mark_top_k(g, c, k)
                row[f"top{k}"][p] = mark_stats(g, flag)
        if not c.startswith("ref_"):
            row["v5_top3"] = v5_added(df, c, "top3", "m_top3")
            row["v5_win"] = v5_added(df, c, "win", "m_win")
        table[c] = row
        print(c, row["auc_top3"].get("sel"), row.get("v5_top3", {}).get("loyo_2022_24"), flush=True)
    (OUT / "score_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score", "rules"])
    a = ap.parse_args(argv)
    if a.cmd == "build":
        return build(a)
    if a.cmd == "score":
        return score(a)
    from src.scripts.data import tp66_rules
    return tp66_rules.main()


if __name__ == "__main__":
    sys.exit(main())

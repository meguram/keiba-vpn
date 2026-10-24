"""セリ取引価格のパイロット（馬プロフィール表の抜き出し）の集計と、基準モデルの OOF に対する『価格が残差を説明するか』の簡易検定（ホストで実行）。

python -m src.scripts.data.acq_sale_pilot_report --profiles /home/jovyan/acq/profile_pilot.jsonl /home/jovyan/acq/profile_pilot2.jsonl --out /home/jovyan/acq/sale_pilot_report.json

- 被覆: プロフィール表に『セリ取引価格』の値がある馬の割合（世代別）、価格の分布、セールの種類、売却時の馬齢
- 残差: 基準 d_CY_BEST の OOF（valid 2022〜24）の p_win / p_top3 に、log 価格・有無を足したロジスティック回帰が、
  馬ごとの 5 分割 CV で LogLoss をどれだけ下げるか（全走・初出走だけ）。標本が小さいので、点推定と区間（馬ブートストラップ）を併記する。
市場の値（オッズ・人気）は使わない。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KEIBA_ROOT") or Path(__file__).resolve().parents[3])
OOF = ROOT / "data/local/modeling/training_pipeline_v4/v6/eval/oof_d_CY_BEST.parquet"


def load_profiles(paths: list[str]) -> pd.DataFrame:
    from src.scraper.profile_extra import parse_profile_rows

    rows = {}
    for p in paths:
        if not Path(p).exists():
            continue
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if not d.get("rows"):
                continue
            o = parse_profile_rows(d["rows"])
            rows[d["horse_id"]] = {"horse_id": d["horse_id"], **{k: v for k, v in o.items() if k != "relatives"}, "n_rel": len(o["relatives"])}
    df = pd.DataFrame(list(rows.values()))
    df["birth_year"] = df["horse_id"].str[:4].astype(int)
    df["sale_age"] = df["sale_year"] - df["birth_year"]
    return df


def coverage(df: pd.DataFrame) -> dict:
    has = df["sale_price_man"].notna()
    by = df.assign(has=has).groupby("birth_year").agg(n=("horse_id", "size"), price=("has", "sum"), owner_id=("owner_id", lambda s: int((s != "").sum())))
    return {
        "horses": int(len(df)), "with_price": int(has.sum()), "rate": round(float(has.mean()), 4),
        "owner_id_rate": round(float((df["owner_id"] != "").mean()), 4),
        "by_birth_year": {str(k): {kk: int(vv) for kk, vv in v.items()} for k, v in by.iterrows()},
        "price_man_quantiles": {str(q): float(df.loc[has, "sale_price_man"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if has.any() else {},
        "sale_name_top": df.loc[has, "sale_name"].value_counts().head(8).to_dict(),
        "sale_age": df.loc[has, "sale_age"].value_counts().sort_index().to_dict(),
    }


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def residual_test(df: pd.DataFrame, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression

    oof = pd.read_parquet(OOF)
    oof["race_id"], oof["horse_id"] = oof["race_id"].astype(str), oof["horse_id"].astype(str)
    runs = pd.read_parquet(Path(os.environ.get("ACQ_DIR", "/home/jovyan/work/acq")) / "runs_all.parquet", columns=["horse_id", "race_id", "date", "finish_position"])
    runs["race_id"] = runs["race_id"].astype(str)
    m = oof.merge(runs, on=["race_id", "horse_id"], how="inner").merge(df, on="horse_id", how="inner")
    m = m[m["finish_position"] > 0].copy()
    m["y_win"], m["y_top3"] = (m["finish_position"] == 1).astype(int), (m["finish_position"] <= 3).astype(int)
    m["debut"] = (m.groupby("horse_id")["date"].rank(method="first") == 1).astype(int)
    # 売却時の馬齢が 2 歳以下のものだけ使う（競走後の売却を除く）。それ以外は価格なし扱い
    ok = m["sale_price_man"].notna() & (m["sale_age"] <= 2)
    m["has"] = ok.astype(float)
    m["lp"] = np.where(ok, np.log(m["sale_price_man"].clip(lower=1)), 0.0)
    m["lp_c"] = np.where(ok, m["lp"] - m.loc[ok, "lp"].mean(), 0.0)
    horses = m["horse_id"].unique()
    rng = np.random.default_rng(seed)
    fold = dict(zip(horses, rng.integers(0, 5, len(horses))))
    m["fold"] = m["horse_id"].map(fold)
    out = {"runs": int(len(m)), "horses": int(len(horses)), "price_runs": int(ok.sum())}
    for label, sub in (("all", m), ("debut", m[m["debut"] == 1])):
        for tgt, pc in (("y_win", "p_win"), ("y_top3", "p_top3")):
            gain = np.zeros(len(sub))
            for k in range(5):
                tr, te = sub["fold"] != k, sub["fold"] == k
                X = lambda d: np.c_[np.log(d[pc] / (1 - d[pc])), d["has"], d["lp_c"]]
                base = lambda d: np.c_[np.log(d[pc] / (1 - d[pc]))]
                lr0 = LogisticRegression(C=1e6, max_iter=500).fit(base(sub[tr]), sub.loc[tr, tgt])
                lr1 = LogisticRegression(C=1e6, max_iter=500).fit(X(sub[tr]), sub.loc[tr, tgt])
                gain[te.to_numpy()] = logloss(sub.loc[te, tgt].to_numpy(), lr0.predict_proba(base(sub[te]))[:, 1]) - logloss(sub.loc[te, tgt].to_numpy(), lr1.predict_proba(X(sub[te]))[:, 1])
            hs = sub["horse_id"].to_numpy()
            g = pd.Series(gain).groupby(hs).sum()
            n = len(sub)
            boot = [g.sample(len(g), replace=True, random_state=int(s)).sum() / n for s in range(300)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            full = LogisticRegression(C=1e6, max_iter=500).fit(np.c_[np.log(sub[pc] / (1 - sub[pc])), sub["has"], sub["lp_c"]], sub[tgt])
            out[f"{label}_{tgt}"] = {"n": int(n), "ll_gain_per_run": round(float(gain.mean()), 6), "ci95": [round(float(lo), 6), round(float(hi), 6)],
                                      "coef_has": round(float(full.coef_[0][1]), 4), "coef_logprice": round(float(full.coef_[0][2]), 4)}
        # 単純な集計: 価格あり/なしの実績 − 予測の差
        for tgt, pc in (("y_win", "p_win"), ("y_top3", "p_top3")):
            for flag, g in sub.groupby(sub["has"] > 0):
                out.setdefault(f"{label}_resid_{tgt}", {})["priced" if flag else "none"] = {"n": int(len(g)), "actual": round(float(g[tgt].mean()), 4), "pred": round(float(g[pc].mean()), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--out", default="/home/jovyan/acq/sale_pilot_report.json")
    a = ap.parse_args()
    df = load_profiles(a.profiles)
    res = {"coverage": coverage(df)}
    try:
        res["residual"] = residual_test(df)
    except Exception as e:  # noqa: BLE001
        res["residual_error"] = repr(e)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

"""ドリフト診断（一回きりの分析）: モデルと市場の勝率 LogLoss の差が 2026 年に開いた原因を切り分ける。

  python -m src.scripts.data.drift_diagnose --out /home/jovyan/work/drift_out/diagnosis.json [--dump-dir data/local/inference/drift_features]

1. 月別（2022-01〜）: 件数・モデル/市場の勝率 LL・差（95% 区間）・ECE・市場の Σ(1/odds)・本命(1番人気)勝率・◎勝率
2. 期間の要約: 基準期間(2022-2025)の同季節 vs 2026 Jan-May vs 2026 Jun-Oct、z 値
3. 層別の差と、構成の変化(mix)/層内の変化(within)への分解
4. 較正: 分位ごとの予測と実現率（モデル/市場）、較正の補正でどれだけ差が縮むか（較正 vs 順位付けの切り分け）
5. 市場側: オーバーラウンド・人気別勝率・市場の LL/ECE の年推移
6. 評価の人工物: 被覆（結果があるのに予測がない日・レース）、履歴の浅い馬の比率、race_class 欠損
7. （dump があれば）特徴量の分布: 2025 vs 2026 の PSI/KS/欠損率、敵対的検証、推論経路の誤差

指標は報告・警報のみ。モデルや閾値の選択には使わない。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import drift_metrics as M
from src.scripts.data.drift_data import add_strata, load_cfg, load_eval_frame

STRATA = ["hist", "age_c", "surface", "dist_band", "f_size", "track_condition", "venue", "grade"]


def period_of(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(df["year"] < 2026, "base", np.where(df["month"] <= 5, "26_JanMay", "26_JunOct")), index=df.index)


def monthly(df: pd.DataFrame) -> list[dict]:
    out = []
    for (y, m), g in df.groupby(["year", "month"]):
        ci = M.gap_with_ci(g, n_boot=200)
        top = g.loc[g.groupby("race_id")["p_win"].idxmax()]
        fav = g.loc[g.groupby("race_id")["p_mkt"].idxmax()]
        out.append({
            "ym": f"{y}-{m:02d}", "src": str(g["src"].mode().iat[0]), "races": ci["races"], "rows": ci["rows"],
            "gap": ci["gap"], "gap_lo": ci["lo"], "gap_hi": ci["hi"],
            "ll_model": float(M.bin_ll(g["y_win"], g["p_win"]).mean()), "ll_market": float(M.bin_ll(g["y_win"], g["p_mkt"]).mean()),
            "ll_top2": float(M.bin_ll(g["y_top2"], g["p_top2"]).mean()), "ll_top3": float(M.bin_ll(g["y_top3"], g["p_top3"]).mean()),
            "ece_model": M.ece(g["p_win"], g["y_win"]), "ece_market": M.ece(g["p_mkt"], g["y_win"]),
            "top_pick_win": float(top["y_win"].mean()), "favorite_win": float(fav["y_win"].mean()),
        })
    return out


def period_summary(df: pd.DataFrame) -> dict:
    d = df.assign(per=period_of(df))
    out: dict = {}
    base_all = d[d["per"] == "base"]
    for name, season in (("JanMay", range(1, 6)), ("JunOct", range(6, 11))):
        b = base_all[base_all["month"].isin(list(season))]
        c = d[(d["per"] == f"26_{name}")]
        cb, cc = M.gap_with_ci(b), M.gap_with_ci(c)
        z = (cc["gap"] - cb["gap"]) / float(np.hypot(cb["se"], cc["se"]))
        by_year = {int(y): M.gap_with_ci(g)["gap"] for y, g in b.groupby("year")}
        out[name] = {"base": cb, "cur": cc, "diff": cc["gap"] - cb["gap"], "z": z, "base_by_year": by_year}
    return out


def strata_table(df: pd.DataFrame, season: tuple[int, int]) -> dict:
    d = add_strata(df)
    d = d[d["month"].between(*season)]
    base, cur = d[d["year"] < 2026], d[d["year"] == 2026]
    out = {}
    for col in STRATA:
        if col not in d.columns:
            continue
        out[col] = M.decompose_gap(base, cur, col)
    return out


def calibration(df: pd.DataFrame) -> dict:
    d = df.assign(per=period_of(df))
    out = {}
    for per, g in d.groupby("per"):
        rec = {"rows": int(len(g))}
        for who, col in (("model", "p_win"), ("market", "p_mkt")):
            rec[who] = {"ece": M.ece(g[col], g["y_win"]), "bins": M.calib_bins(g[col], g["y_win"]).round(5).to_dict("records"),
                        "slope": M.calib_slope(g[col], g["y_win"])[1], "intercept": M.calib_slope(g[col], g["y_win"])[0]}
        # 較正の補正（期間内で a, b を当てはめ）後のモデルの差
        a, b = M.calib_slope(g["p_win"], g["y_win"])
        z = np.log(np.clip(g["p_win"], 1e-6, 1 - 1e-6) / (1 - np.clip(g["p_win"], 1e-6, 1 - 1e-6)))
        p_re = 1 / (1 + np.exp(-(a + b * z)))
        ll_m, ll_k, ll_r = (float(M.bin_ll(g["y_win"], x).mean()) for x in (g["p_win"], g["p_mkt"], p_re))
        rec["gap_raw"], rec["gap_after_recalibration"] = ll_m - ll_k, ll_r - ll_k
        rec["calibration_share_of_gap"] = (ll_m - ll_r) / (ll_m - ll_k) if ll_m != ll_k else float("nan")
        out[per] = rec
    return out


def market_side(df: pd.DataFrame, tables: Path) -> dict:
    from src.scripts.data.drift_data import market_probs

    rows = []
    for y in sorted(df["year"].unique()):
        _, ok = market_probs(tables, int(y))
        g = df[df["year"] == y]
        fav = g.loc[g.groupby("race_id")["p_mkt"].idxmax()]
        pop = g.assign(pop=pd.to_numeric(g["popularity"], errors="coerce").clip(1, 6)).groupby("pop")["y_win"].mean().round(4).to_dict()
        pe = g.assign(pop=pd.to_numeric(g["popularity"], errors="coerce").clip(1, 6)).groupby("pop")["p_mkt"].mean().round(4).to_dict()
        rows.append({"year": int(y), "overround_mean": float(ok["overround"].mean()), "favorite_win": float(fav["y_win"].mean()),
                     "ll_market": float(M.bin_ll(g["y_win"], g["p_mkt"]).mean()), "ece_market": M.ece(g["p_mkt"], g["y_win"]),
                     "win_rate_by_popularity": {str(k): v for k, v in pop.items()}, "mean_p_by_popularity": {str(k): v for k, v in pe.items()}})
    d = df.assign(per=period_of(df))
    per = {}
    for p, g in d.groupby("per"):
        fav = g.loc[g.groupby("race_id")["p_mkt"].idxmax()]
        per[p] = {"ll_market": float(M.bin_ll(g["y_win"], g["p_mkt"]).mean()), "favorite_win": float(fav["y_win"].mean()),
                  "ll_model": float(M.bin_ll(g["y_win"], g["p_win"]).mean())}
    return {"by_year": rows, "by_period": per}


def artifacts(root: Path, df: pd.DataFrame) -> dict:
    from src.scripts.data.drift_data import load_results_year

    tables = root / "data/page_reference/tables"
    out: dict = {}
    res = load_results_year(tables, 2026)
    n_by_race = res.groupby("race_id").size()
    races_all = set(n_by_race[n_by_race >= 5].index)
    have = set(df[df["year"] == 2026]["race_id"])
    miss = sorted(races_all - have)
    by_day = res[res["race_id"].isin(miss)].groupby(res["date"].dt.strftime("%Y%m%d"))["race_id"].nunique().to_dict()
    out["coverage_2026"] = {"races_with_results": len(races_all), "races_evaluated": len(have & races_all), "missing_races": len(miss),
                            "missing_by_day": by_day}
    d = add_strata(df).assign(per=period_of(df))
    out["share_short_history"] = {p: {"zero_prior": float((g["n_prior"] == 0).mean()), "age2": float((g["age"] == 2).mean()),
                                      "newhorse_race": float((g["grade"].astype(str) == "新馬").mean())} for p, g in d.groupby("per")}
    out["race_class_missing"] = {p: float((g["cls"].isin(["None", "nan", ""])).mean()) for p, g in d.groupby("per")}
    r = df[df["year"] == 2026]
    out["rows_by_month_2026"] = {int(m): int(len(g)) for m, g in r.groupby("month")}
    return out


def coverage_diag(df: pd.DataFrame) -> dict:
    """入力データの供給(調教ページ)の被覆を月ごとに数え、同じ月の中で『調教の行がある馬』と『無い馬』の差を比べる（2025-2026）。"""
    from src.scripts.data import drift_coverage as C

    d = df[df["year"] >= 2025].copy()
    recs = C.default_loader()(sorted(d["horse_id"].astype(str).unique().tolist()))
    blocks = C.block_dates(recs)
    present = {h: bool(r) for h, r in recs.items()}
    tab = C.census(d, blocks=blocks, records_present=present)
    dstr = d["date"].dt.strftime("%Y-%m-%d")
    d["has_block"] = [x in blocks.get(str(h), set()) for h, x in zip(d["horse_id"], dstr)]
    d["gap_row"] = M.bin_ll(d["y_win"], d["p_win"]) - M.bin_ll(d["y_win"], d["p_mkt"])
    d["lm"] = np.log(np.clip(d["p_win"], 1e-6, 1 - 1e-6) / (1 - np.clip(d["p_win"], 1e-6, 1 - 1e-6)))
    d["season"] = np.where(d["month"] <= 5, "Jan-May", "Jun-Oct")
    d = d[d["month"] <= 10]
    by = (d.groupby(["year", "season", "has_block"]).agg(rows=("gap_row", "size"), gap=("gap_row", "mean"), logit_sd=("lm", "std")).reset_index())
    return {"monthly": tab.round(4).to_dict("records"), "gap_by_block": by.round(5).to_dict("records")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-year", type=int, default=2022)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    cfg = load_cfg()
    years = list(range(a.from_year, 2027))
    df = load_eval_frame(root, years, cfg)
    res = {"periods": {"train": cfg["train"], "test1": cfg["test1"], "test2": cfg["test2"]}, "rows": int(len(df)),
           "monthly": monthly(df), "period_summary": period_summary(df),
           "strata_JunOct": strata_table(df, (6, 10)), "strata_JanMay": strata_table(df, (1, 5)),
           "calibration": calibration(df), "market": market_side(df, root / "data/page_reference/tables"), "artifacts": artifacts(root, df), "coverage": coverage_diag(df)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ps = res["period_summary"]
    print(json.dumps({k: {"base": round(v["base"]["gap"], 4), "cur": round(v["cur"]["gap"], 4), "z": round(v["z"], 2)} for k, v in ps.items()}))


if __name__ == "__main__":
    main()

"""tp-61: 『確実にハナを取りたい馬』の数と強さがペースを決めるか（仮説の実測検証）と、事前の説明変数。

python -m src.scripts.data.tp61_hana build     # 馬ごとの as-of ハナ指標（直近5走の1角先頭率など）→ レース集計
python -m src.scripts.data.tp61_hana analyze   # ハナ主張馬の頭数別のペース・前半3F、要因を揃えた比較
入力: tp61 の horse / race_cls、tp55 の races_feat（テン1F・先行頭数）、コース形状（jra_courses.json）。
先読みなし: 各馬の指標は『その走より前の走だけ』（日付が厳密に前）。
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
RF = Path("/home/jovyan/work/tp55_out/races_feat.parquet")
REF = (2020, 2021, 2022, 2023)
HANA_FEATS = ["hn_claim", "hn_claim_strong", "hn_claim_vstrong", "hn_l1", "hn_l2", "hn_l3", "hn_sum3", "hn_pairs", "hn_outer", "hn_inner", "hn_minrel",
              "hn_front5_n", "hn_firstcurve", "hn_start_geom", "hn_ten_min", "hn_ten_gap12", "hn_uncertain"]


def _roll5(v: np.ndarray, grp_start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """馬ごとに並んだ値 v（NaN 可）の『直前 5 走』の合計と件数。grp_start: 各行の、その馬の最初の行の位置。"""
    ok = (~np.isnan(v)).astype(float)
    x = np.where(np.isnan(v), 0.0, v)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cn = np.concatenate([[0.0], np.cumsum(ok)])
    i = np.arange(len(v))
    lo = np.maximum(i - 5, grp_start)               # 直前 5 走 = [i-5, i-1]、馬の先頭より前には遡らない
    return cs[i] - cs[lo], cn[i] - cn[lo]


def horse_asof(h: pd.DataFrame) -> pd.DataFrame:
    """h: race_id, date, horse_id, horse_number, early_norm, pass_first。各走について『その走より前の直近 5 走』の指標（ベクトル化）。"""
    d = h[["race_id", "date", "horse_id", "horse_number", "early_norm", "pass_first", "field_size"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    led = (d["pass_first"] == 1).astype(float).where(d["pass_first"].notna()).to_numpy()
    f15 = (d["early_norm"] <= 0.15).astype(float).where(d["early_norm"].notna()).to_numpy()
    start = d.groupby("horse_id", sort=False).cumcount().to_numpy()
    grp_start = np.arange(len(d)) - start
    s_led, n5 = _roll5(led, grp_start)
    s_f, n_f = _roll5(f15, grp_start)
    d["n5"] = n5
    d["n_led5"] = s_led
    d["lead5"] = np.where(n5 > 0, s_led / np.where(n5 > 0, n5, 1), np.nan)
    d["front5"] = np.where(n_f > 0, s_f / np.where(n_f > 0, n_f, 1), np.nan)
    d["last_led"] = pd.Series(led).groupby(d["horse_id"].to_numpy()).shift(1).to_numpy()
    d["min_e5"] = np.nan
    return d[["race_id", "horse_id", "horse_number", "field_size", "lead5", "n_led5", "n5", "front5", "min_e5", "last_led"]]


def race_hana(a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in a.groupby("race_id", sort=False):
        n = int(g["field_size"].iloc[0])
        lead = g["lead5"].to_numpy(float)
        n5 = g["n5"].to_numpy(float)
        nl = g["n_led5"].to_numpy(float)
        no = g["horse_number"].to_numpy(float)
        ok = (n5 >= 3) & ~np.isnan(lead)
        claim = ok & (lead >= 0.4)
        strong = ok & (lead >= 0.6)
        vstrong = ok & (nl >= 4)
        order = np.sort(np.where(ok, lead, -1.0))[::-1]
        rel = (no - 1) / max(n - 1, 1)
        cn = no[claim]
        pairs = 0
        if len(cn) >= 2:
            cn_s = np.sort(cn)
            pairs = int((np.diff(cn_s) <= 2).sum())
        rows.append({"race_id": rid, "hn_claim": int(claim.sum()), "hn_claim_strong": int(strong.sum()), "hn_claim_vstrong": int(vstrong.sum()),
                     "hn_l1": max(order[0], 0) if len(order) > 0 else 0.0, "hn_l2": max(order[1], 0) if len(order) > 1 else 0.0,
                     "hn_l3": max(order[2], 0) if len(order) > 2 else 0.0, "hn_sum3": float(np.clip(order[:3], 0, None).sum()),
                     "hn_pairs": pairs, "hn_outer": int((claim & (rel >= 0.65)).sum()), "hn_inner": int((claim & (rel <= 0.35)).sum()),
                     "hn_minrel": float(rel[claim].min()) if claim.any() else np.nan,
                     "hn_front5_n": float(np.nansum(g["front5"].to_numpy(float))),
                     "hn_uncertain": int((n5 < 3).sum())})
    return pd.DataFrame(rows)


def cmd_build() -> int:
    from src.pipeline.features import tp55_lapfeat as lf

    h = pd.read_parquet(D / "horse.parquet", columns=["race_id", "date", "horse_id", "horse_number", "early_norm", "pass_first", "field_size"])
    a = horse_asof(h)
    r = race_hana(a)
    rc = pd.read_parquet(D / "race_cls.parquet")[["race_id", "venue", "surface", "distance"]]
    geo = lf.load_course_geo(ROOT / "data/meta/course_master/jra_courses.json")
    fc, sg = [], []
    for rid, s, d in zip(rc["race_id"].astype(str), rc["surface"], rc["distance"]):
        gx = geo.get((rid[4:6], s, int(float(d))))
        fc.append(gx["geo_first_curve_m"] if gx else np.nan)
        sg.append(gx["geo_start_geom"] if gx else np.nan)
    rc["hn_firstcurve"], rc["hn_start_geom"] = fc, sg
    tf = pd.read_parquet(RF, columns=["race_id", "ten_min", "ten_gap12"]).rename(columns={"ten_min": "hn_ten_min", "ten_gap12": "hn_ten_gap12"})
    out = r.merge(rc[["race_id", "hn_firstcurve", "hn_start_geom"]], on="race_id", how="left").merge(tf, on="race_id", how="left")
    out.to_parquet(D / "hana.parquet", index=False)
    print(out.shape)
    print(out[["hn_claim", "hn_claim_strong", "hn_claim_vstrong", "hn_l1", "hn_sum3", "hn_firstcurve", "hn_ten_min"]].describe().round(2).to_string())
    return 0


def _ols_boot(df: pd.DataFrame, y: str, xs: list[str], term: list[str], n: int = 200, seed: int = 0) -> dict:
    X = df[xs].to_numpy(float)
    X = np.column_stack([np.ones(len(X)), X])
    yv = df[y].to_numpy(float)
    ok = ~np.isnan(X).any(1) & ~np.isnan(yv)
    X, yv = X[ok], yv[ok]
    rng = np.random.default_rng(seed)
    cols = [xs.index(t) + 1 for t in term]
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(X), len(X))
        bs.append(np.linalg.lstsq(X[i], yv[i], rcond=None)[0][cols])
    bs = np.array(bs)
    return {t: (float(beta[c]), float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))) for j, (t, c) in enumerate(zip(term, cols))}


def cmd_analyze() -> int:
    pd.set_option("display.width", 220)
    r = pd.read_parquet(D / "race_cls.parquet")
    hn = pd.read_parquet(D / "hana.parquet")
    d = r.merge(hn, on="race_id", how="inner")
    d = d[d["year"] <= 2025]                     # 2026 年はテストセット②
    d = d[d["bal"].notna() & d["pace_cls"].notna() & (d["hn_uncertain"] <= d["field_size"] * 0.5)].copy()
    d["claim_cat"] = d["hn_claim_strong"].clip(upper=3)           # 0,1,2,3+（直近 5 走の 1 角先頭率 0.6 以上、3 走以上の履歴あり）
    d["claim_cat2"] = d["hn_claim"].clip(upper=3)                 # 0.4 以上
    print("全体", len(d))
    for key in ("claim_cat", "claim_cat2"):
        t = d.groupby(key).agg(n=("race_id", "size"), t3f_dev=("t3f_dev", "mean"), bal=("bal", "mean"), p_high=("pace_cls", lambda s: (s == "high").mean()),
                              p_slow=("pace_cls", lambda s: (s == "slow").mean()), gap=("c4_gap_per_horse", "mean"), l3dev=("race_l3f_dev", "mean"))
        print(key); print(t.round(3).to_string())
    print("面×距離帯 別（強い主張馬の頭数）")
    t = d.groupby(["surface", "band", "claim_cat"]).agg(n=("race_id", "size"), t3f_dev=("t3f_dev", "mean"), bal=("bal", "mean"), p_high=("pace_cls", lambda s: (s == "high").mean()),
                                                        p_slow=("pace_cls", lambda s: (s == "slow").mean()))
    print(t.round(3).to_string())
    # 要因を揃えた回帰（前半3Fの速さ t3f_dev、バランス bal を、ハナ主張馬の頭数ダミー + 条件）
    d["turf"] = (d["surface"] == "芝").astype(float)
    d["heavy"] = (d["going_group"] == "道悪").astype(float)
    d["dist_k"] = (d["distance"] - 1600) / 1000
    d["fs"] = (d["field_size"] - 14) / 4
    d["fc"] = ((d["hn_firstcurve"] - 300) / 300).fillna(0)
    d["sg"] = d["hn_start_geom"].fillna(0)
    for k in (1, 2, 3):
        d[f"c{k}"] = (d["claim_cat"] == k).astype(float)
        d[f"d{k}"] = (d["claim_cat2"] == k).astype(float)
    for yv in ("t3f_dev", "bal"):
        for dummies, nm in ((["c1", "c2", "c3"], "強い(0.6以上)"), (["d1", "d2", "d3"], "主張(0.4以上)")):
            xs = dummies + ["turf", "heavy", "dist_k", "fs", "fc", "sg"] + [f"y{y}" for y in range(2021, 2026)]
            for y in range(2021, 2026):
                d[f"y{y}"] = (d["year"] == y).astype(float)
            res = _ols_boot(d, yv, xs, dummies)
            print(f"[{yv}] {nm}: " + ", ".join(f"{t}={v[0]:+.3f}[{v[1]:+.3f},{v[2]:+.3f}]" for t, v in res.items()))
    # 先行争いの激しさ: 実測の 1 角通過での先頭付近の混み具合は horse から（act）
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "analyze"])
    a = ap.parse_args()
    return {"build": cmd_build, "analyze": cmd_analyze}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())

"""TS の尺度を winz（0.5%/99.5% 打ち切り。走数が少ないコースは帯の z で戻す）にしたとき、実データで現行からどう変わるかのサンプル。

horse_ts.py は変えない。ts_scale_compare.py の winz の計算（mode_bounds・row_params）を、全走（本番の course_ts と同じ範囲）に当てて
現行（cur）と並べる。ホストで cwd=リポジトリ、`TS_CACHE=/tmp/ts_scale_h.pkl nice -n 19 /opt/conda/bin/python -I <このファイル>` で実行
（ts_scale_compare.py と同じ場所に置く。GCS は読まない）。出力: /tmp/ts_winz_samples.json と標準出力（表）。

  1) 異常走のあるコース（京都芝1600・小倉芝1200・中山ダ2400）の、異常走そのものと同じレースの他の馬の前後
  2) 変化の小さい典型レース（芝短・ダ・芝中・ダ中）の全頭の x・ts（現行・winz）・レース内順位
  3) 全体: |Δts| の割合・コース別の変化・打ち切り端の走
  4) 馬別: 平均 TS・最大 TS の変化
  5) run_axes.py:228 の LT 経由の持続力軸の変化と 10 点換算
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("ts_scale_compare", Path(__file__).with_name("ts_scale_compare.py"))
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

from src.research.race import axis_score10 as s10  # noqa: E402
from src.research.race import run_axes as ra  # noqa: E402

COURSE = ["venue", "surface", "distance"]
OUT = Path(os.environ.get("OUT", "/tmp/ts_winz_samples.json"))
FAMOUS = ["ドウデュース", "リバティアイランド", "フォーエバーヤング", "レモンポップ", "ジャンタルマンタル", "ダノンデサイル", "ソールオリエンス",
          "ナミュール", "ジャスティンミラノ", "ベラジオオペラ", "チェルヴィニア", "クロワデュノール", "ミュージアムマイル", "マスカレードボール"]


def pr(df: pd.DataFrame, title: str, fmt: str = "{:.2f}") -> None:
    print(f"\n## {title}")
    print(df.to_string(index=False, float_format=lambda v: fmt.format(v)))


def build(h: pd.DataFrame) -> pd.DataFrame:
    fit = np.ones(len(h), bool)
    x = h["x"].to_numpy(float)
    mu0, sd0 = sc.row_params(h, h["x"], fit, sc.spread_sd)
    ts0 = 50 - 10 * (x - mu0) / sd0
    assert np.nanmax(np.abs(ts0 - h["ts"].to_numpy(float))) < 1e-6, "現行の再現が合わない"
    lo, hi = sc.mode_bounds(h, "winz", fit)
    xc = np.minimum(np.maximum(x, np.nan_to_num(lo, nan=-np.inf)), np.nan_to_num(hi, nan=np.inf))
    mu1, sd1 = sc.row_params(h, pd.Series(xc, index=h.index), fit, sc.spread_sd)
    ts1 = 50 - 10 * (xc - mu1) / sd1
    d = h.copy()
    d["xc"], d["lo"], d["hi"] = xc, lo, hi
    d["mu0"], d["sd0"], d["mu1"], d["sd1"] = mu0, sd0, mu1, sd1
    d["ts0"], d["ts1"] = ts0, ts1
    d["dts"] = ts1 - ts0
    d["cside"] = np.where(x < np.nan_to_num(lo, nan=-np.inf), "fast", np.where(x > np.nan_to_num(hi, nan=np.inf), "slow", ""))
    d["course"] = d["venue"] + d["surface"] + d["distance"].astype(int).astype(str)
    return d


def sample_abnormal(d: pd.DataFrame, out: dict) -> None:
    out["abnormal"] = {}
    for v, s, di in sc.FOCUS:
        c = d[(d.venue == v) & (d.surface == s) & (d.distance == di)]
        name = f"{v}{s}{di}"
        n = len(c)
        sdv = (float(c.ts0.std()), float(c.ts1.std()))
        print(f"\n### {name}: {n} 走  mu_s {c.mu0.iloc[0]:.3f}→{c.mu1.iloc[0]:.3f}  sd_s {c.sd0.iloc[0]:.3f}→{c.sd1.iloc[0]:.3f}  ts の SD {sdv[0]:.2f}→{sdv[1]:.2f}"
              f"  打ち切り x∈[{c.lo.iloc[0]:.2f},{c.hi.iloc[0]:.2f}]")
        rows = []
        pick = list(c.nlargest(2, "x").index) + list(c.nsmallest(1, "x").index)
        for i in pick:
            r = d.loc[i]
            rows.append(r)
            race = d[d.race_id == r.race_id].sort_values("finish_order")
            for j in [race.index[0], race.index[1], race.index[len(race) // 2]]:
                if j != i and j not in [q.name for q in rows]:
                    rows.append(d.loc[j])
        t = pd.DataFrame(rows)[["race_id", "date", "horse_name", "finish_order", "nent", "peak", "x", "xc", "ts0", "ts1", "dts"]]
        t["date"] = t["date"].astype(str).str[:10]
        pr(t.drop_duplicates(["race_id", "horse_name"]), name, "{:.2f}")
        out["abnormal"][name] = {"n": n, "mu": [float(c.mu0.iloc[0]), float(c.mu1.iloc[0])], "sd": [float(c.sd0.iloc[0]), float(c.sd1.iloc[0])], "ts_sd": sdv,
                                 "rows": json.loads(t.drop_duplicates(["race_id", "horse_name"]).to_json(orient="records", force_ascii=False))}


def sample_typical(d: pd.DataFrame, out: dict) -> None:
    out["typical"] = {}
    for v, s, di in [("阪神", "芝", 1200), ("東京", "ダート", 1400), ("東京", "芝", 2000), ("阪神", "ダート", 1800)]:
        c = d[(d.venue == v) & (d.surface == s) & (d.distance == di) & (d.nent >= 14) & (d.nent <= 18) & (d.track_condition == "良") & (d.date >= "2026-01-01")]
        rid = c.sort_values("date").race_id.iloc[-1]
        r = d[d.race_id == rid].sort_values("finish_order")
        t = pd.DataFrame({"finish": r.finish_order, "horse_name": r.horse_name, "x": r.x, "ts現行": r.ts0, "tswinz": r.ts1, "差": r.dts,
                          "順位現行": r.ts0.rank(ascending=False, method="min"), "順位winz": r.ts1.rank(ascending=False, method="min")})
        same = bool((t["順位現行"] == t["順位winz"]).all())
        print(f"\n### {v}{s}{di} {rid} {str(r.date.iloc[0])[:10]}  mu_s {r.mu0.iloc[0]:.3f}→{r.mu1.iloc[0]:.3f}  sd_s {r.sd0.iloc[0]:.3f}→{r.sd1.iloc[0]:.3f}  順位同一={same}")
        pr(t, f"{v}{s}{di} {rid}", "{:.2f}")
        out["typical"][f"{v}{s}{di}"] = {"race_id": rid, "same_rank": same, "rows": json.loads(t.to_json(orient="records", force_ascii=False))}


def overall(d: pd.DataFrame, out: dict) -> None:
    a = d["dts"].abs()
    o = {"n": int(len(d)), "ge0.5": float((a >= 0.5).mean()), "ge1": float((a >= 1).mean()), "ge2": float((a >= 2).mean()),
         "up": float((d.dts > 0.05).mean()), "down": float((d.dts < -0.05).mean()), "mean": float(d.dts.mean()), "mean_abs": float(a.mean())}
    print("\n## 全体", json.dumps(o, ensure_ascii=False))
    g = d.groupby("course")
    c = pd.DataFrame({"runs": g.size(), "mean": g.dts.mean(), "mean_abs": g.dts.apply(lambda s: s.abs().mean()),
                      "max_up": g.dts.max(), "max_down": g.dts.min(), "sd_cur": g.ts0.std(), "sd_winz": g.ts1.std(),
                      "sd_s": g.sd0.first(), "sd_s_winz": g.sd1.first()}).reset_index()
    c["max_abs"] = np.where(c.max_up >= -c.max_down, c.max_up, c.max_down)
    top = c[c.runs >= 100].sort_values("mean_abs", ascending=False).head(10)
    pr(top[["course", "runs", "mean", "mean_abs", "max_abs", "sd_cur", "sd_winz"]], "コース別（100 走以上、平均 |Δ| の大きい順 10）")
    pr(c[c.runs < 100].sort_values("mean_abs", ascending=False).head(5)[["course", "runs", "mean", "mean_abs", "max_abs"]], "コース別（100 走未満の上位 5）")
    e = {}
    for side in ("fast", "slow"):
        s = d[d.cside == side]
        e[side] = {"n": int(len(s)), "share": float(len(s) / len(d)), "mean_dts": float(s.dts.mean()), "median_dts": float(s.dts.median()),
                   "max_abs_dts": float(s.dts.abs().max()), "mean_x_shift": float((s.xc - s.x).mean())}
    print("端の走", json.dumps(e, ensure_ascii=False))
    nonc = d[d.cside == ""]
    print("端でない走の Δ（コース尺度の変化だけ）: 平均 %.3f  |Δ|>=0.5 %.3f  >=1 %.3f" % (nonc.dts.mean(), (nonc.dts.abs() >= 0.5).mean(), (nonc.dts.abs() >= 1).mean()))
    out["overall"] = {**o, "edge": e, "nonedge_mean": float(nonc.dts.mean()), "nonedge_ge0.5": float((nonc.dts.abs() >= 0.5).mean()), "nonedge_ge1": float((nonc.dts.abs() >= 1).mean()),
                      "top_courses": json.loads(top.to_json(orient="records", force_ascii=False))}


def per_horse(d: pd.DataFrame, out: dict) -> None:
    g = d.groupby("horse_id")
    p = pd.DataFrame({"name": g.horse_name.last(), "runs": g.size(), "mean0": g.ts0.mean(), "mean1": g.ts1.mean(), "max0": g.ts0.max(), "max1": g.ts1.max()})
    p["dmean"], p["dmax"] = p.mean1 - p.mean0, p.max1 - p.max0
    res = {}
    for lab, q in (("全馬", p), ("3走以上", p[p.runs >= 3])):
        r = {"n": int(len(q))}
        for k in ("dmean", "dmax"):
            a = q[k].abs()
            r[k] = {"q": {str(z): float(q[k].quantile(z)) for z in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}, "ge0.5": float((a >= 0.5).mean()), "ge1": float((a >= 1).mean()), "ge2": float((a >= 2).mean()),
                    "up": float((q[k] > 0.05).mean()), "down": float((q[k] < -0.05).mean())}
        res[lab] = r
        print(f"\n## 馬別 {lab} {r['n']} 頭: " + json.dumps({k: r[k] for k in ("dmean", "dmax")}, ensure_ascii=False))
    q3 = p[p.runs >= 3]
    cols = ["name", "runs", "mean0", "mean1", "dmean", "max0", "max1", "dmax"]
    pr(q3.reindex(q3.dmean.abs().sort_values(ascending=False).index).head(5)[cols], "平均 TS の変化の大きい馬 5（3 走以上）")
    pr(q3.reindex(q3.dmax.abs().sort_values(ascending=False).index).head(5)[cols], "最大 TS の変化の大きい馬 5（3 走以上）")
    fam = p[p.name.isin(FAMOUS)].sort_values("max0", ascending=False)
    pr(fam[cols], "有名馬（データにいるもの）")
    out["per_horse"] = {"dist": res, "famous": json.loads(fam[cols].to_json(orient="records", force_ascii=False)),
                        "top_mean": json.loads(q3.reindex(q3.dmean.abs().sort_values(ascending=False).index).head(5)[cols].to_json(orient="records", force_ascii=False)),
                        "top_max": json.loads(q3.reindex(q3.dmax.abs().sort_values(ascending=False).index).head(5)[cols].to_json(orient="records", force_ascii=False))}
    return p


def peak_points(d: pd.DataFrame, ref: dict) -> dict:
    """馬ごとの『直近 10 走の上位 2 走の平均』（最新時点）を TS の 10 点換算（現行の節点のまま）で比べる。節点は現行の分布で作ったもの。"""
    t = d.sort_values(["horse_id", "date", "race_id"]).groupby("horse_id").tail(10)
    knots = ref["ts"]["knots"]
    res = {}
    for c in ("ts0", "ts1"):
        r = t.groupby("horse_id")[c].rank(ascending=False, method="first")
        top = t[c].where(r <= 2).groupby(t.horse_id).sum()
        n = t.groupby("horse_id").size().clip(upper=2)
        res[c] = (top + (2 - n) * s10.PEAK_PRIOR) / 2
    pk = pd.DataFrame(res)
    pk["p0"] = [s10._round_pt(s10._ts_points(v, knots)) for v in pk.ts0]
    pk["p1"] = [s10._round_pt(s10._ts_points(v, knots)) for v in pk.ts1]
    dp = (pk.p1 - pk.p0)
    return {"n": int(len(pk)), "peak_dts_mean": float((pk.ts1 - pk.ts0).mean()), "peak_dts_p5": float((pk.ts1 - pk.ts0).quantile(0.05)), "peak_dts_p95": float((pk.ts1 - pk.ts0).quantile(0.95)),
            "pt_changed": float((dp != 0).mean()), "pt_ge1": float((dp.abs() >= 1).mean()), "pt_up": float((dp > 0).mean()), "pt_down": float((dp < 0).mean())}


def spill(d: pd.DataFrame, out: dict, ref: dict) -> None:
    LP = np.array([lt[-5:] for lt in d["lt"]], float)
    off = d["offset"].to_numpy(float)[:, None]
    LT0 = 50 - 10 * ((LP - off) - d["mu0"].to_numpy()[:, None]) / d["sd0"].to_numpy()[:, None]
    # winz では x だけでなく mu・sd も変わる。ラップは打ち切らない（horse_ts.py の案は peak（x）だけを打ち切る）。LT は mu1・sd1 で
    LT1 = 50 - 10 * ((LP - off) - d["mu1"].to_numpy()[:, None]) / d["sd1"].to_numpy()[:, None]
    dist = d["distance"].to_numpy(float)
    ts0, ts1 = d["ts0"].to_numpy(), d["ts1"].to_numpy()
    s0 = np.array([ra._sustain_one(LT0[i], ts0[i], dist[i], LP[i]) for i in range(len(d))])
    s1 = np.array([ra._sustain_one(LT1[i], ts1[i], dist[i], LP[i]) for i in range(len(d))])
    t = d[["race_id", "horse_id", "venue", "surface", "distance", "course", "horse_name"]].copy()
    t["raw0"], t["raw1"] = s0, s1
    ax0 = ra.course_dev(t.assign(sustain_raw=s0), "sustain_raw")
    ax1 = ra.course_dev(t.assign(sustain_raw=s1), "sustain_raw")
    a = ref["axes"]["sustain"]
    k = ref["k"]
    t["dax"] = ax1 - ax0
    t["dpt"] = k * a["sign"] * t["dax"] / a["sd"]    # 1 走の値のまま 10 点換算した変化（馬のプロファイルは複数走の平均なので、これは上限）
    dlt = LT1 - LT0
    t["dlt_mean"] = dlt.mean(axis=1)
    t["dlt_maxabs"] = np.abs(dlt).max(axis=1)
    t["dts"] = d["dts"].values
    o = {"raw_changed": float((s0 != s1).mean()), "dlt_mean_mean": float(t.dlt_mean.mean()), "dlt_ge1": float((t.dlt_maxabs >= 1).mean()), "dlt_ge2": float((t.dlt_maxabs >= 2).mean()),
         "run_pt_ge1": float((t.dpt.abs() >= 1).mean()), "run_pt_ge0.5": float((t.dpt.abs() >= 0.5).mean()), "run_pt_up": float((t.dpt > 0.05).mean()), "run_pt_down": float((t.dpt < -0.05).mean()),
         "ref_k": k, "ref_sd": a["sd"], "ref_mu": a["mu"]}
    # 馬ごと（全走の平均）の変化 = プロファイル（走の平均）の近似
    hp = t.groupby("horse_id").agg(name=("horse_name", "last"), runs=("dpt", "size"), dpt=("dpt", "mean"))
    hp3 = hp[hp.runs >= 3]
    o["horse_pt_ge1"] = float((hp3.dpt.abs() >= 1).mean())
    o["horse_pt_ge0.5"] = float((hp3.dpt.abs() >= 0.5).mean())
    o["horse_n3"] = int(len(hp3))
    g = t.groupby("course")
    c = pd.DataFrame({"runs": g.size(), "dax_mean": g.dax.mean(), "dpt_mean": g.dpt.mean(), "run_pt_ge1": g.dpt.apply(lambda s: (s.abs() >= 1).mean()), "dlt_mean": g.dlt_mean.mean()}).reset_index()
    top = c[c.runs >= 100].sort_values("run_pt_ge1", ascending=False).head(8)
    print("\n## 持続力軸への波及", json.dumps(o, ensure_ascii=False))
    pr(top, "コース別（100 走以上、1 点以上動く走の割合の大きい順 8）", "{:.3f}")
    o["top_courses"] = json.loads(top.to_json(orient="records", force_ascii=False))
    o["ts_points"] = peak_points(d, ref)
    print("TS の 10 点（節点は現行のまま）", json.dumps(o["ts_points"], ensure_ascii=False))
    out["spill"] = o


def main() -> None:
    h = pd.read_pickle(os.environ.get("TS_CACHE", "/tmp/ts_scale_h.pkl")).reset_index(drop=True)
    h["date"] = pd.to_datetime(h["date"])
    ref = json.loads((REPO / "data/local/running_profile/reference/reference.json").read_text(encoding="utf-8"))["score10"]
    d = build(h)
    print(f"{len(d)} 走  {d.date.min().date()}〜{d.date.max().date()}  打ち切り走: fast {(d.cside == 'fast').sum()} / slow {(d.cside == 'slow').sum()}")
    out: dict = {"meta": {"n": int(len(d)), "from": str(d.date.min().date()), "to": str(d.date.max().date()), "fit": "全走（本番の course_ts と同じ）"}}
    sample_abnormal(d, out)
    sample_typical(d, out)
    overall(d, out)
    per_horse(d, out)
    spill(d, out, ref)
    OUT.write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()

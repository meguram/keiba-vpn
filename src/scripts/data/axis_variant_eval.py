"""走法軸の候補（持続力の短距離限定・fgain）を、現行の軸と同じ手順で比べる（読むだけ・JSON を書く）。

候補:
  sustain_short … 現行の持続力を 1400m 以下の走だけ残し、他は欠測にしたもの（as-of も短距離の過去走だけで作る）
  fg3           … 3C 末 → ゴールの累積タイム順位の上がり（r3 − 最終順位。cr に最終直線の分を足したもの）
  fg4           … 4C 末 → ゴールの累積タイム順位の上がり（r4 − 最終順位。直線の末脚だけ）
  どちらも run_axes と同じ流儀（course_dev → 距離帯の SD → 50 + 10z）。
入力（ホストのローカルだけ。GCS は読まない）: axis_rank_eval.py と同じ（run_axes.parquet・horse_lap・race_result_flat・コース台帳）
評価（axis_rank_eval.py の部品をそのまま使う）: 各馬の「その走より前の直近 10 走の平均（2 走以上）」を as-of 値にし、レース内で平均を引く。
  ・傾き = 1SD あたりの人気補正後の勝率・複勝率（pt）。レース単位ブートストラップ 300 回の 95% 区間つき。距離帯（≤1400 / 1600 / 1800+）別。
  ・増分 = 他の軸（TS など）に回帰して残った分の傾き。選択期間で係数を決め、全期間に当てる。
  ・持続力は、現行 vs 短距離限定を同じ走どうしで比べる（差の区間は同じレースの引き直し）。
  ・再現性 = 馬ごとの最初の 4 走・6 走の奇偶分割 → Spearman-Brown で 1 走換算。
期間: 選択 = 2024-10〜2025-06、確認 = 2025-07〜12（1 回だけ見る）、参考 = 2026（選択に使わない）。
使い方（ホストで）: cd <repo> && nice -n 19 /opt/conda/bin/python -I src/scripts/data/axis_variant_eval.py OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from src.research.race import run_axes as ra  # noqa: E402
from src.scripts.data import axis_rank_eval as ae  # noqa: E402

SHORT_MAX = 1400
EVAL_AXES = ["ts", "sustain", "sustain_short", "cr", "makuri", "fg3", "fg4"]
BANDS = {"全距離": (0, 9999), "≤1400m": (0, 1400), "1500〜1700m": (1401, 1799), "1800m〜": (1800, 9999)}
CONTROLS = {"sustain": ["ts"], "sustain_short": ["ts"], "fg3": ["ts", "cr", "makuri", "sustain"], "fg4": ["ts", "cr", "makuri", "sustain"],
            "cr": ["ts"], "makuri": ["ts"]}


def make_sustain_short(sustain: pd.Series, distance: pd.Series, short_max: int = SHORT_MAX) -> pd.Series:
    """短距離（short_max 以下）の走だけ値を残し、他は欠測。"""
    return sustain.where(distance.astype(float) <= short_max)


def residualize(y: np.ndarray, X: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """y を X（定数項つき）で回帰した残差。係数は fit の走だけで決める。X・y のどれかが欠測の走は NaN。"""
    A = np.column_stack([np.ones(len(y)), X])
    good = np.isfinite(A).all(1) & np.isfinite(y)
    use = good & fit
    beta, *_ = np.linalg.lstsq(A[use], y[use], rcond=None)
    out = np.full(len(y), np.nan)
    out[good] = y[good] - A[good] @ beta
    return out


def build_variant_runs(years: list[int]) -> pd.DataFrame:
    """現行の run_axes（2024 年以降）に、持続力の短距離限定と fgain の列を足す。fgain は horse_lap から作る。"""
    base = ae.prep_axes(pd.read_parquet(ae.OLD_PARQUET))
    t = ra.with_ts(ra.read_runs(ae.LAPS), years)
    last = np.array([len(c) - 1 for c in t["cum"]])
    r3 = ra._cum_rank_rel(t, t["i3"].to_numpy(int))
    r4 = ra._cum_rank_rel(t, t["i4"].to_numpy(int))
    rf = ra._cum_rank_rel(t, last)
    band = ra.band_of(t["distance"].to_numpy(float))
    for name, raw in (("fg3", r3 - rf), ("fg4", r4 - rf)):
        t[name + "_raw"] = raw
        t[name] = 50 + 10 * ra._band_std(ra.course_dev(t, name + "_raw"), band)
    keys = ["race_id", "horse_id"]
    f = t[keys + ["fg3", "fg4"]].assign(race_id=t["race_id"].astype(str), horse_id=t["horse_id"].astype(str))
    f = f.drop_duplicates(keys, keep="last")
    base = base.merge(f, on=keys, how="left")
    base["sustain_short"] = make_sustain_short(base["sustain"], base["distance"])
    return base.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)


def band_masks(d: pd.DataFrame) -> dict[str, np.ndarray]:
    dist = d["distance"].to_numpy(float)
    return {k: (dist >= lo) & (dist <= hi) for k, (lo, hi) in BANDS.items()}


def rel_asof(d: pd.DataFrame, a: str) -> pd.Series:
    v = d[f"{a}_asof_old"]
    ok = v.notna()
    return ae.race_rel(d[ok], v[ok]).reindex(d.index)


def evaluate(d: pd.DataFrame, periods: dict) -> dict:
    X = {a: rel_asof(d, a) for a in EVAL_AXES}
    bands = band_masks(d)
    y = {"勝率": d["resid_w"].to_numpy(), "複勝率": d["resid_p"].to_numpy()}
    rid = d["race_id"].to_numpy()
    sel = ae.in_period(d, periods["選択"])
    out: dict = {"own": {}, "pair_sustain": {}, "incremental": {}, "corr_sel": {}}
    for a in EVAL_AXES:
        for bn, bm in bands.items():
            for pn, p in periods.items():
                pm = bm & ae.in_period(d, p)
                for tn, yy in y.items():
                    out["own"][f"{a}|{bn}|{pn}|{tn}"] = ae.slope_one(X[a].to_numpy()[pm], yy[pm], rid[pm])
    so, sn = d["sustain_asof_old"], d["sustain_short_asof_old"]
    cm = so.notna() & sn.notna()      # 両方の as-of が取れる走だけで、レース内の平均を引き直す（引く相手の馬が両案で同じになる）
    xo_c = ae.race_rel(d[cm], so[cm]).reindex(d.index).to_numpy()
    xn_c = ae.race_rel(d[cm], sn[cm]).reindex(d.index).to_numpy()
    for bn, bm in bands.items():
        for pn, p in periods.items():
            pm = bm & ae.in_period(d, p)
            for tn, yy in y.items():
                out["pair_sustain"][f"{bn}|{pn}|{tn}"] = ae.slope_pair(xo_c[pm], xn_c[pm], yy[pm], rid[pm])
    for a, ctrl in CONTROLS.items():
        Z = np.column_stack([X[c].to_numpy() for c in ctrl])
        res = pd.Series(residualize(X[a].to_numpy(), Z, sel & np.isfinite(X[a].to_numpy())), index=d.index)
        ok = res.notna()
        res = ae.race_rel(d[ok], res[ok]).reindex(d.index)      # 全軸が取れる走だけでレース内の平均を引き直す（slope_one は定数項なし）
        for bn, bm in bands.items():
            for pn, p in periods.items():
                pm = bm & ae.in_period(d, p)
                for tn, yy in y.items():
                    out["incremental"][f"{a}(控除 {'+'.join(ctrl)})|{bn}|{pn}|{tn}"] = ae.slope_one(res.to_numpy()[pm], yy[pm], rid[pm])
    sub = pd.DataFrame({a: X[a] for a in EVAL_AXES})[sel]
    out["corr_sel"] = {"corr": {a: {b: round(float(sub[a].corr(sub[b])), 3) for b in EVAL_AXES} for a in EVAL_AXES},
                       "n": {a: {b: int(sub[[a, b]].dropna().shape[0]) for b in EVAL_AXES} for a in EVAL_AXES}}   # 2 軸ずつ、両方が取れる走
    return out


def print_tables(res: dict) -> None:
    for title, key in (("傾き（own）", "own"), ("増分（他の軸を除いた残り）", "incremental")):
        print(f"\n### {title}: 1SD あたり人気補正後の pt。* は区間が 0 を跨がない")
        for k, r in res[key].items():
            if r is not None:
                print(f"{k} | {ae.fmt(r)} | n={r['n']}")
    print("\n### 持続力: 現行 vs 短距離限定（同じ走。差 = 短距離限定 − 現行）")
    for k, r in res["pair_sustain"].items():
        if r is not None:
            print(f"{k} | 現行 {ae.fmt(r['old'])} | 短距離限定 {ae.fmt(r['new'])} | 差 {ae.fmt(r['diff'])} | n={r['new']['n']}")


def knot_variants(cur: list[list[float]]) -> dict[str, list[list[float]]]:
    """cur = 現行の節点。no55 = 5.5 の節点を外して 5→10 を直線で結ぶ。lin = 3→10 を 1 本の直線で結ぶ。"""
    kv = {y: x for x, y in cur}
    return {"cur": cur,
            "no55": [[x, y] for x, y in cur if y != 5.5],
            "lin": [[kv[3.0], 3.0], [kv[10.0], 10.0]]}


def peak_asof(d: pd.DataFrame) -> pd.Series:
    """各走の「その走より前」のピーク TS（直近 PEAK_LOOKBACK 走の上位 PEAK_TOP 走の平均。本番の peak_ts_as_of と同じ式）。前の走が無ければ NaN。
    d は 2024 年以降の走だけ（prep_axes）なので、2023 年以前のピークは入らない近似。"""
    from src.research.race import axis_score10 as a10

    v = d["ts"].to_numpy(float)       # d は馬・日付で整列済み（結果で絞る前の全走）
    out = np.full(len(d), np.nan)
    for _h, idx in d.groupby("horse_id").indices.items():
        for j in range(1, len(idx)):
            prev = v[idx[:j]]
            prev = prev[np.isfinite(prev)][-a10.PEAK_LOOKBACK:]       # 本番と同じ順（TS の無い走を除いてから直近）
            if len(prev):
                out[idx[j]] = a10._peak(prev)
    return pd.Series(out, index=d.index)


def knot_eval(d: pd.DataFrame, knots_cur: list[list[float]]) -> dict:
    """as-of のピーク TS（本番と同じ量）を、節点の違いで 0.5 刻みの点にし、点ごとの人気補正後の勝率・複勝率（pt）と走数を比べる。
    点ごとの平均から、直線あてはめの R²（走数で重み付け）・隣り合う点で下がった数・最大の点の走数の割合を出す。"""
    from src.research.race import axis_score10 as a10

    out: dict = {}
    for name, kn in knot_variants(knots_cur).items():
        ts = d["peak_asof"]
        pts = ts.map(lambda v: a10._round_pt(a10._ts_points(float(v), kn)) if np.isfinite(v) else np.nan)
        res: dict = {"knots": kn}
        for pn, p in ae.PERIODS.items():
            m = ae.in_period(d, p) & pts.notna().to_numpy()
            g = d[m].assign(pt=pts[m]).groupby("pt").agg(n=("resid_w", "size"), w=("resid_w", "mean"), p=("resid_p", "mean"))
            g = g[g["n"] >= 100]
            row = {"by_pt": {str(k): [int(r.n), round(float(r.w), 2), round(float(r.p), 2)] for k, r in g.iterrows()}}
            for tag in ("w", "p"):
                x, y, wt = g.index.to_numpy(float), g[tag].to_numpy(), g["n"].to_numpy(float)
                A = np.column_stack([np.ones(len(x)), x])
                beta = np.linalg.solve(A.T @ (A * wt[:, None]), A.T @ (wt * y))
                ss_res = float((wt * (y - A @ beta) ** 2).sum())
                ss_tot = float((wt * (y - np.average(y, weights=wt)) ** 2).sum())
                row[f"r2_{tag}"] = round(1 - ss_res / ss_tot, 3)
                row[f"slope_{tag}"] = round(float(beta[1]), 3)
                row[f"drops_{tag}"] = int((np.diff(y) < 0).sum())
            row["max_share"] = round(float(g["n"].max() / g["n"].sum()), 3)
            res[pn] = row
        out[name] = res
    return out


def main() -> None:
    out_path = Path(sys.argv[1])
    years = ra.default_years()
    runs = build_variant_runs(years)
    print("走数", len(runs), "| 欠測率(%)", {a: round(float(runs[a].isna().mean() * 100), 1) for a in EVAL_AXES}, flush=True)
    ae.AXES = EVAL_AXES
    ae.SUBSET_AXES = []
    runs["peak_asof"] = peak_asof(runs)
    d = ae.build_eval_frame(runs.copy(), runs.copy())
    d = d.merge(runs[["race_id", "horse_id", "peak_asof"]], on=["race_id", "horse_id"], how="left")
    print("評価用", len(d), "走", flush=True)
    if "--knots" in sys.argv:
        ref = json.load(open(REPO / "data/local/running_profile/reference/reference.json", encoding="utf-8"))
        kr = knot_eval(d, ref["score10"]["ts"]["knots"])
        for name, v in kr.items():
            print("\n###", name, v["knots"])
            for pn in ae.PERIODS:
                r = v[pn]
                print(pn, {k: r[k] for k in r if k != "by_pt"})
                print("  点|走数|勝率|複勝率:", r["by_pt"])
        out_path.write_text(json.dumps(kr, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    res = evaluate(d, ae.PERIODS)
    print_tables(res)
    rel = {a: {"first4": ae.reliability(runs, a, 4), "first6": ae.reliability(runs, a, 6)} for a in EVAL_AXES}
    print("\n### 再現性（1 走換算 rho1）")
    for a, v in rel.items():
        print(a, v)
    print("\n### 相関（選択期間・as-of のレース内偏差）")
    print(json.dumps(res["corr_sel"], ensure_ascii=False))
    out_path.write_text(json.dumps({"periods": ae.PERIODS, "slopes": res, "reliability": rel}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()

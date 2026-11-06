"""tp-84 事前調査: 実際のペース（前半5F・RPCI のずれ）の帯ごとに、どんなタイプの馬の好走度が傾くか。年をまたいで再現するかを見る。ホストで実行。

python -m src.scripts.data.tp84_pace_aptitude [--out /tmp/tp84_out]
好走度のずれ = その走の指数 a（めぐ指数）－ 過去 K 走の期待（as-of）／ 着順の位置 fp －過去 K 走の期待。馬の実力の差を引いたあとの『その展開で上振れ・下振れしたか』。
タイプ = 出走馬の中での相対（レース内の順位の三分位）: 脚質＝過去3走の最初のコーナー位置、末脚＝過去3走の上がり順位。過去走だけで作る。
ペースの帯 = 実際の前半5F／RPCI のずれ（par 基準は 2021 年以前）の、面×距離帯ごとの五分位。2026 年は使わない。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import factory_self as fs
from src.pipeline.features import tp39_growth as tg
from src.pipeline.features import tp42_racelevel as rl
from src.scripts.data import tp39_eval as te
from src.scripts.data import tp42_eval as te42
from src.scripts.data import tp84_pace_eval as ev

YEARS = list(range(2020, 2026))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/tp84_out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = fs.load_runs(te.TABLES, YEARS)
    megu = tg.load_megu(te42.MEGU_ROOT, YEARS)
    d = rl.prepare(runs, megu)
    K = 5
    for col, name in (("a", "exp_a"), ("fp", "exp_fp")):
        d[name] = rl._ewm(rl.lags_m(d, col, K, 0))
    d["n_past"] = np.isfinite(rl.lags_m(d, "a", K, 0)).sum(1)
    d["r_a"] = d["a"] - d["exp_a"]
    d["r_fp"] = -(d["fp"] - d["exp_fp"])                      # 大きいほど期待より好走
    d["style"] = np.nanmean(fs.lags(d, "pos_first", 3), axis=1)
    d["kick"] = np.nanmean(fs.lags(d, "l3f_rank", 3), axis=1)
    from src.research.race import running_profile as rp

    ax = pd.read_parquet(rp.AXES_PATH)
    ax = ax[~ax["abn"].fillna(False).astype(bool)]
    ax["race_id"], ax["horse_id"] = ax["race_id"].astype(str), ax["horse_id"].astype(str)
    axc = [c for c in ("ts", "sustain", "gear", "cr", "endure", "front", "makuri") if c in ax.columns]
    d = d.merge(ax[["race_id", "horse_id"] + axc].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    d = d.sort_values(["horse_id", "dnum", "race_id"], kind="stable").reset_index(drop=True)
    for c in axc:                                   # 過去10走の値のある走の加重平均（1走ずらし）
        d["pf_" + c] = rl._ewm(fs.lags(d, c, 10))
    types = {"style": "脚質(前/中/後)", "kick": "上がり順位(切れ/中/鈍)"}
    types.update({"pf_" + c: "走法軸 " + c for c in axc})
    d["style"] = np.nanmean(fs.lags(d, "pos_first", 3), axis=1)
    d["kick"] = np.nanmean(fs.lags(d, "l3f_rank", 3), axis=1)
    for c in types:
        pct = d.groupby("race_id")[c].rank(pct=True)
        d["t_" + c] = pd.cut(pct, [0, 1 / 3, 2 / 3, 1.0001], labels=["低", "中", "高"], include_lowest=True)
    d["year"] = d["date"].dt.year

    rt = pd.read_parquet(ev.FEAT)
    rt["year"] = rt["date"].str[:4].astype(int)
    rt = ev.race_table(rt[rt["surface"].isin(["芝", "ダート"]) & (rt["year"] <= 2025)].copy())
    fit = rt[pd.to_datetime(rt["date"]) <= ev.PAR_END]
    for c in ("F5", "RPCI"):
        rt["d_" + c] = rt[c].to_numpy() - ev.par_scalar(fit, rt, c)
    rt["band"] = rt["surface"].astype(str) + np.where(rt["distance"] <= 1400, "短", "長")
    for c in ("d_F5", "d_RPCI"):
        rt["q_" + c] = rt.groupby("band")[c].transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    d = d.merge(rt[["race_id", "band", "d_F5", "d_RPCI", "q_d_F5", "q_d_RPCI"]], on="race_id", how="inner")
    d = d[(d["n_past"] >= 2) & d["r_a"].notna()].copy()
    print(f"対象 {len(d):,} 走 / {d['race_id'].nunique():,} レース", flush=True)
    pd.set_option("display.width", 200)
    H1, H2 = "2020-22/24", "2023-25/25"
    d["half"] = np.where(d["year"] <= 2022, H1, H2)
    isax = lambda c: c.startswith("pf_")           # 走法軸は実測ラップが 2024 年からなので 2024 と 2025 で分ける
    rows = []
    for pace_col in ("q_d_F5", "q_d_RPCI"):
        for c in types:
            for tgt in ("r_a", "r_fp"):
                dd = d[d["year"] >= 2024].assign(half=lambda x: np.where(x["year"] == 2024, H1, H2)) if isax(c) else d
                g = dd[dd["t_" + c].notna()].groupby(["t_" + c, pace_col, "half"], observed=True)[tgt].agg(["mean", "count", "std"]).reset_index()
                g.columns = ["tercile", "q", "half", "mean", "n", "sd"]
                g["pace"], g["type"], g["tgt"] = pace_col, c, tgt
                rows.append(g)
    res = pd.concat(rows)
    res.to_csv(out / "aptitude_cells.csv", index=False)
    print("\n== 『高 − 低』の三分位差（好走度のずれの平均差）。ペース帯 q0〜q4 ごと。左=前期（2020-22。走法軸は2024） / 右=後期（2023-25。走法軸は2025）。目安: r_fp は 0.01 で約 1 着順/10頭 ==")
    print("   q0=最も速い前半 / 前傾（F5・RPCI の小さい側）… q4=最も遅い前半 / 後傾")
    for pace_col in ("q_d_F5", "q_d_RPCI"):
        for c in types:
            for tgt in ("r_a", "r_fp"):
                r = res[(res.pace == pace_col) & (res.type == c) & (res.tgt == tgt)]
                m = r.pivot_table(index=["half", "q"], columns="tercile", values="mean")
                nn = r.pivot_table(index=["half", "q"], columns="tercile", values="n")
                sd = r.pivot_table(index=["half", "q"], columns="tercile", values="sd")
                se = np.sqrt(sd["高"] ** 2 / nn["高"] + sd["低"] ** 2 / nn["低"])
                diff = (m["高"] - m["低"])
                f = "{:+.3f}" if tgt == "r_fp" else "{:+.2f}"
                line = " | ".join(f"q{int(q)} " + "/".join(f.format(diff[(h, q)]) for h in (H1, H2)) for q in sorted({k[1] for k in diff.index}))
                print(f"[{pace_col}] {c:12s} {tgt}: {line}  (SE≈{se.mean():.3f})", flush=True)
    # 傾き（q を 0〜4 の数として回帰した係数）を半期ごとに: 同符号なら再現
    print("\n== ペース帯に対する『高−低』差の傾き（q が 1 段進むごとの変化）==")
    for pace_col in ("q_d_F5", "q_d_RPCI"):
        for c in types:
            for tgt in ("r_a", "r_fp"):
                out_s = []
                for h in (H1, H2):
                    r = res[(res.pace == pace_col) & (res.type == c) & (res.tgt == tgt) & (res.half == h)]
                    m = r.pivot_table(index="q", columns="tercile", values="mean")
                    dd = (m["高"] - m["低"]).dropna()
                    out_s.append(np.polyfit(dd.index.to_numpy(float), dd.to_numpy(), 1)[0])
                same = "同符号" if out_s[0] * out_s[1] > 0 else "逆"
                print(f"[{pace_col}] {c:12s} {tgt}: {out_s[0]:+.4f} / {out_s[1]:+.4f} {same}")


if __name__ == "__main__":
    main()

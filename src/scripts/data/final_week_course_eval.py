"""最終追い切りのパターン別の成績を、コース（芝/ダート × 距離帯）別に集計し、コースを分ける価値があるかを前向きに確かめる。

python -m src.scripts.data.final_week_course_eval [--out DIR] [--min-races 20]

パターンの定義・集計の方法は final_week_pattern_eval と同じ（レース単位の期待値との比 lift、調教師の基準を引いた rel）。
ここでは (コース帯, パターン) をセルとして、(1) セルごとの lift、(2) 前の期間のセルが後の期間に続くか、
(3) 『コース別のセル』が『パターン全体（コース込み）』より後の期間をよく当てるか、(4) 向き B（3 着以内の構成比）をコース帯ごとに出す。
主 = 2020〜2025。2026 年・市場オッズ・人気は使わない。障害は含めない（芝・ダートのみ）。
節 0・1・4 は 2025 を含む記述だけの表で、セルの採否には使わない（採否の確認は節 2・3 の 2020〜24 → 2025 だけ）。2 つの前向きの分け方は 2023〜24 が重なり独立ではない。
距離帯: 芝 ~1200（新潟直線 1000 を含む）/ 1300〜1600 / 1700〜2000 / 2100〜2400 / 2500~、ダート ~1200 / 1300〜1600 / 1700〜1800 / 1900~。ラベルは実在する距離（芝 1300・1700・2100 は無い）に合わせてある。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import final_week_pattern_eval as P
from src.scripts.data.final_week_check import md

SPLITS = (("2020〜22 → 2023〜24", (2020, 2022), (2023, 2024)), ("2020〜24 → 2025（手を付けていない年）", (2020, 2024), (2025, 2025)))
SHRINK_K = (50, 200, 1000)


def course_band(surface: pd.Series, distance: pd.Series) -> pd.Series:
    """芝 / ダート × 距離帯。距離は 100m 刻み。"""
    turf = pd.cut(distance, [0, 1200, 1600, 2000, 2400, 99999], labels=["~1200", "1400-1600", "1800-2000", "2200-2400", "2500~"])
    dirt = pd.cut(distance, [0, 1200, 1600, 1800, 99999], labels=["~1200", "1300-1600", "1700-1800", "1900~"])
    out = pd.Series(np.where(surface == "芝", "芝" + turf.astype(str), "ダ" + dirt.astype(str)), index=surface.index)
    return out.where(distance.notna() & (distance > 0) & surface.isin(["芝", "ダート"]))


def with_course(d: pd.DataFrame) -> pd.DataFrame:
    c = d.copy()
    c["course"] = course_band(c["surface"].astype(str), pd.to_numeric(c["distance"], errors="coerce"))
    c = c[c["course"].notna()].copy()
    c["pat_only"] = c["pat"]
    c["pat"] = c["course"] + "/" + c["pat_only"].where(c["pat_only"].notna(), "")
    c.loc[c["pat_only"].isna(), "pat"] = np.nan
    return c


def cell_table(c: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """(コース帯/パターン) セルの lift と、調教師の基準を引いた rel。"""
    S = P.pooled(c, y0, y1, ci=False)
    R = P.pattern_rel(P.cells(c, y0, y1))
    out = S[["races", "horse_win", "race_win", "lift_win", "race_top3", "lift_top3", "se_win", "se_top3"]].join(R[["rel_win", "rel_top3", "se_win", "se_top3"]].add_prefix("adj_"))
    out["z_top3"] = (out["lift_top3"] - 1) / out["se_top3"]
    out["z_win"] = (out["lift_win"] - 1) / out["se_win"]
    return out


def wcorr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    if len(x) < 5:
        return float("nan")
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    c = np.average((x - mx) * (y - my), weights=w)
    return float(c / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)))


def predictive(c: pd.DataFrame, sel: tuple[int, int], test: tuple[int, int], nmin: int, boot: int = 400) -> pd.DataFrame:
    """前の期間の情報で、後の期間のセルの lift を当てられるか。
    A: パターン全体（コースを分けない）の前の期間の lift / B: セル自身の前の期間の lift / C: B をパターン全体 A へ縮める（k レース分の重み）。
    相関に加え、後の期間のレース数で重み付けした二乗誤差 MSE と、A に対する MSE の差（負 = A より良い）とそのセル・ブートストラップの 95% 区間を出す。
    相関はパターン間の差（栗東坂路 A3 ≒ 2、美浦坂路 X ≒ 0.5 など）で大半が決まるので、コース別の上乗せは MSE の差で見る。"""
    E, L = cell_table(c, *sel), cell_table(c, *test)
    PE = P.pooled(c.assign(pat=c["pat_only"]), *sel, ci=False)
    rng = np.random.default_rng(0)
    rows = []
    for k_, nm in (("top3", "複勝率"), ("win", "勝率")):
        m = E.join(L, lsuffix="_E", rsuffix="_L", how="inner")
        m = m[(m["races_E"] >= nmin) & (m["races_L"] >= nmin)].copy()
        m["A"] = pd.Series([i.split("/", 1)[1] for i in m.index], index=m.index).map(PE[f"lift_{k_}"])
        m = m[np.isfinite(m["A"]) & np.isfinite(m[f"lift_{k_}_E"]) & np.isfinite(m[f"lift_{k_}_L"])]
        y, w = m[f"lift_{k_}_L"].to_numpy(), m["races_L"].to_numpy().astype(float)
        preds = {"A パターン全体（コースを分けない）": m["A"].to_numpy(), "B コース別のセル自身": m[f"lift_{k_}_E"].to_numpy()}
        for k in SHRINK_K:
            wt = (m["races_E"] / (m["races_E"] + k)).to_numpy()
            preds[f"C B を A へ縮める（k={k}）"] = m["A"].to_numpy() + wt * (m[f"lift_{k_}_E"].to_numpy() - m["A"].to_numpy())
        eA = (preds["A パターン全体（コースを分けない）"] - y) ** 2
        idx = rng.integers(0, len(m), size=(boot, len(m)))
        for name, pr in preds.items():
            e = (pr - y) ** 2
            d = e - eA
            row = {"指標": nm, "セル数": int(len(m)), "予測の作り方": name, "相関": round(wcorr(pr, y, w), 3), "MSE": round(float(np.average(e, weights=w)), 4)}
            if name[0] != "A":
                bs = [np.average(d[i], weights=w[i]) for i in idx]
                row.update({"MSE−Aの MSE": round(float(np.average(d, weights=w)), 4), "95%区間": f"[{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]"})
            rows.append(row)
    return pd.DataFrame(rows)


def residual_corr(c: pd.DataFrame, sel: tuple[int, int], test: tuple[int, int], nmin: int) -> pd.DataFrame:
    """コース固有の効果だけを取り出して続くかを見る。セルの lift からパターン全体（同じ期間）の lift を引いた残差 r を、前後の期間で比べる。
    相関はノイズで小さく出る（減衰）ので、共分散 cov(r_前, r_後)（= コース固有の効果の分散 τ² の推定）と、
    残差の分散 − 標準誤差² の平均（これも τ² の推定）を併記する。重みは min(前のレース数, 後のレース数)。"""
    pat_only = c.assign(pat=c["pat_only"])
    E, L = cell_table(c, *sel), cell_table(c, *test)
    PE, PL = P.pooled(pat_only, *sel, ci=False), P.pooled(pat_only, *test, ci=False)
    m = E.join(L, lsuffix="_E", rsuffix="_L", how="inner")
    m = m[(m["races_E"] >= nmin) & (m["races_L"] >= nmin)].copy()
    po = pd.Series([i.split("/", 1)[1] for i in m.index], index=m.index)
    w = np.minimum(m["races_E"], m["races_L"]).to_numpy().astype(float)
    rows = []
    for k, nm in (("top3", "複勝率"), ("win", "勝率")):
        re_ = (m[f"lift_{k}_E"] - po.map(PE[f"lift_{k}"])).to_numpy()
        rl_ = (m[f"lift_{k}_L"] - po.map(PL[f"lift_{k}"])).to_numpy()
        ok = np.isfinite(re_) & np.isfinite(rl_) & (w > 0)
        a, b, ww = re_[ok], rl_[ok], w[ok]
        ma, mb = np.average(a, weights=ww), np.average(b, weights=ww)
        cov = float(np.average((a - ma) * (b - mb), weights=ww))
        tau_cov = max(cov, 0.0) ** 0.5
        va = float(np.average((a - ma) ** 2, weights=ww) - np.average(m[f"se_{k}_E"].to_numpy()[ok] ** 2, weights=ww))
        vb = float(np.average((b - mb) ** 2, weights=ww) - np.average(m[f"se_{k}_L"].to_numpy()[ok] ** 2, weights=ww))
        rows.append({"指標": nm, "セル数": int(ok.sum()), "相関": round(wcorr(a, b, ww), 3), "共分散": round(cov, 4), "τ（共分散から）": round(tau_cov, 3),
                     "前の期間: 分散−se²": round(va, 4), "後の期間: 分散−se²": round(vb, 4),
                     "前の期間の差のsd": round(float(np.sqrt(np.average((a - ma) ** 2, weights=ww))), 3),
                     "後の期間の差のsd": round(float(np.sqrt(np.average((b - mb) ** 2, weights=ww))), 3)})
    return pd.DataFrame(rows)


def reproduce(c: pd.DataFrame, sel: tuple[int, int], test: tuple[int, int], nmin: int) -> pd.DataFrame:
    """前の期間で z が上位・下位 20% のセルが、後の期間でどうなったか（lift の平均。1 = 平均並み。前後ともレース数で重み付け）。
    『セルの z』で選んだ場合と、『パターン全体（コースを分けない）の z』で選んだ場合を並べる。後者より前者が良く分かれなければ、コースを分ける根拠にならない。"""
    E, L = cell_table(c, *sel), cell_table(c, *test)
    PE = P.pooled(c.assign(pat=c["pat_only"]), *sel, ci=False)
    m = E.join(L, lsuffix="_1", rsuffix="_2", how="inner")
    m = m[(m["races_1"] >= nmin) & (m["races_2"] >= nmin)].copy()
    po = pd.Series([i.split("/", 1)[1] for i in m.index], index=m.index)
    rows = []
    for k, nm in (("top3", "複勝率"), ("win", "勝率")):
        PEz = (PE[f"lift_{k}"] - 1) / PE[f"se_{k}"]
        for sname, z in (("セルの z", m[f"z_{k}_1"]), ("パターン全体の z", po.map(PEz))):
            mm = m[np.isfinite(z) & np.isfinite(m[f"lift_{k}_2"])]
            order = z[mm.index].rank(pct=True)
            for lab, pick in (("上位20%", order >= 0.8), ("下位20%", order <= 0.2)):
                x = mm[pick]
                if not len(x):
                    continue
                rows.append({"指標": nm, "選び方": sname, "前の期間での位置": lab, "セル数": int(len(x)),
                             "前の期間のlift": round(float(np.average(x[f"lift_{k}_1"], weights=x["races_1"])), 3),
                             "後の期間のlift": round(float(np.average(x[f"lift_{k}_2"], weights=x["races_2"])), 3)})
    return pd.DataFrame(rows)


def direction_b_by_course(c: pd.DataFrame, y0: int, y1: int, nmin_t3: int = 30) -> pd.DataFrame:
    """コース帯ごとに、パターンの構成比（全出走 / 3 着以内）と比。"""
    out = []
    for cb, sub in c[c["year"].between(y0, y1) & c["pat_only"].notna()].groupby("course"):
        allsh = sub["pat_only"].value_counts()
        t3 = sub[sub["t3"] == 1]["pat_only"].value_counts()
        b = pd.DataFrame({"出走数": allsh, "3着以内の頭数": t3}).fillna(0)
        b["全出走の構成比"], b["3着以内の構成比"] = b["出走数"] / b["出走数"].sum(), b["3着以内の頭数"] / max(b["3着以内の頭数"].sum(), 1)
        b["比"] = b["3着以内の構成比"] / b["全出走の構成比"]
        p = b["全出走の構成比"].clip(1e-9, 1 - 1e-9)
        n3 = b["3着以内の頭数"].sum()
        b["z"] = (b["3着以内の頭数"] - n3 * p) / np.sqrt(n3 * p * (1 - p))
        b.insert(0, "course", cb)
        out.append(b[b["3着以内の頭数"] >= nmin_t3].rename_axis("pat").reset_index())
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jovyan/work/retrain_out/base_2016/article_check/final_week/course_eval")
    ap.add_argument("--min-races", type=int, default=20)
    ap.add_argument("--laps", default=str(P.LAPS))
    ap.add_argument("--table", default=str(P.TABLE))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = with_course(P.load(Path(a.laps), Path(a.table)))
    print("# 最終追い切りパターン × コース（芝/ダート × 距離帯）の成績\n")

    print("## 0. コース帯ごとの規模（2020〜2025、パターンが付いた出走）\n")
    sz = c[c["year"].between(*P.MAIN) & c["pat_only"].notna()].groupby("course").agg(出走=("win", "size"), レース=("race_id", "nunique"), パターン数=("pat_only", "nunique"))
    print(md(sz.reset_index()))

    print("\n## 1. セル（コース帯/パターン）ごとの lift（2020〜2025、レース単位。1 = 無作為並み）\n")
    T = cell_table(c, *P.MAIN)
    T.to_csv(out / "cells_course_pattern_2020_2025.csv", encoding="utf-8-sig")
    big = T[T["races"] >= 100]
    print(f"セル数 {len(T):,}、うち 100 レース以上 {len(big):,}。多数のセルを並べるので、上位・下位は偶然でも出る。判断は 2・3 の前向きの確認で行う。\n")
    cols = ["races", "race_win", "lift_win", "z_win", "race_top3", "lift_top3", "z_top3", "adj_rel_win", "adj_rel_top3"]
    print("### 1-1. 複勝率の lift が高いセル 上位15（100 レース以上）\n")
    print(md(P.fmt(big.sort_values("z_top3", ascending=False)[cols].head(15).rename_axis("セル").reset_index(), nd=2)))
    print("\n### 1-2. 複勝率の lift が低いセル 下位15（100 レース以上）\n")
    print(md(P.fmt(big.sort_values("z_top3")[cols].head(15).rename_axis("セル").reset_index(), nd=2)))

    print("\n## 2. 前の期間で上位・下位だったセルは、後の期間で続くか\n")
    for lab, sel, test in SPLITS:
        print(f"### {lab}（両期間とも {a.min_races} レース以上のセル。後の期間の lift は後の期間のレース数で重み付け）\n")
        r = reproduce(c, sel, test, a.min_races)
        r.to_csv(out / f"reproduce_{sel[0]}_{sel[1]}_{test[0]}_{test[1]}.csv", index=False, encoding="utf-8-sig")
        print(md(r))
        print()

    print("## 3. コースを分けると、後の期間の予想が良くなるか（セルの lift の予測。相関が高いほど当たる）\n")
    for lab, sel, test in SPLITS:
        print(f"### {lab}\n")
        q = predictive(c, sel, test, a.min_races)
        q.to_csv(out / f"predictive_{sel[0]}_{sel[1]}_{test[0]}_{test[1]}.csv", index=False, encoding="utf-8-sig")
        print(md(q))
        print()
    print("### コース固有の差だけが続くか（セルの lift − パターン全体の lift を、前後の期間で相関）\n")
    for lab, sel, test in SPLITS:
        print(f"**{lab}**\n")
        rc = residual_corr(c, sel, test, a.min_races)
        rc.to_csv(out / f"residual_{sel[0]}_{sel[1]}_{test[0]}_{test[1]}.csv", index=False, encoding="utf-8-sig")
        print(md(rc))
        print()
    print("MSE−A が負で 95% 区間が 0 を跨がなければ、コースを分ける意味がある。B（縮めない）が A より悪いなら、コースを分けても偶然のばらつきを増やすだけ。残差の τ は『コース固有の効果』の大きさ（lift の単位）。\n")

    print("## 4. 向き B: コース帯ごとの、3 着以内の馬のパターン構成（2020〜2025、3 着以内 30 頭以上のパターン）\n")
    b = direction_b_by_course(c, *P.MAIN)
    b.to_csv(out / "B_by_course_2020_2025.csv", index=False, encoding="utf-8-sig")
    s = ["pat", "出走数", "3着以内の頭数", "全出走の構成比", "3着以内の構成比", "比", "z"]
    for cb, g in b.groupby("course"):
        g = g.sort_values("z", ascending=False)
        print(f"### {cb}\n")
        print(md(P.fmt(pd.concat([g.head(3), g.tail(3)])[s].drop_duplicates(), pct=("全出走の構成比", "3着以内の構成比"), nd=2)))
        print()


if __name__ == "__main__":
    main()

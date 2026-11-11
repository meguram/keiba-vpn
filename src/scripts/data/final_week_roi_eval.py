"""最終追い切りパターン別の単勝回収率・複勝回収率（結果表の払戻のみ。市場オッズ・人気は使わない）。

python -m src.scripts.data.final_week_roi_eval [--out DIR] [--min-races 100]

回収率 = 該当馬を 1 頭ずつ同額（100 円）で全頭買ったときの払戻 / 賭け金（馬単位）。レース均等（1 レースの賭け金を一定にして該当馬に等分）も併記する。
区間はレース単位のブートストラップ。期間: 主 = 2020〜2025、再現性は 2020〜22 → 2023〜24 と 2020〜24 → 2025（2025 は選択に使わない参考）。
前の期間で回収率の高かったセルを選び、後の期間で回収率が続くかを見る（偶然の上位は後の期間で落ちる）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import final_week_course_eval as CE
from src.scripts.data import final_week_pattern_eval as P
from src.scripts.data import tp35_race_level as R
from src.scripts.data.final_week_check import md

ROI = ["roi_horse_w", "roi_horse_p", "roi_race_w", "roi_race_p"]
PERIODS = {"2020〜2025": (2020, 2025), "前半(2020〜22)": (2020, 2022), "後半(2023〜24)": (2023, 2024), "2025": (2025, 2025)}


def add_returns(d: pd.DataFrame) -> pd.DataFrame:
    """各行に 100 円あたりの単勝・複勝の払戻（/100）を付ける。払戻が読めない行は NaN。"""
    from src.scripts.data.tp28_eval import _load_returns

    assert d["horse_number"].notna().all() and not d.duplicated(["race_id", "horse_number"]).any(), "払戻の結合キー（race_id, horse_number）に欠損か重複がある"
    r = _load_returns(d[["race_id", "horse_number", "year"]])
    out = d.copy()
    out["win_ret"], out["place_ret"] = r["win_ret"].to_numpy(), r["place_ret"].to_numpy()
    assert out["win_ret"].notna().mean() > 0.99, "払戻が読めた割合が低すぎる（払戻の表を確認）"
    return out


def roi_table(d: pd.DataFrame, y0: int, y1: int, ci: bool = False) -> pd.DataFrame:
    """パターン（d["pat"]）ごとの回収率。払戻の読めない行は除く。"""
    sub = d[d["year"].between(y0, y1) & d["pat"].notna() & d["win_ret"].notna() & d["place_ret"].notna()]
    rt = R.race_table(sub, sub["pat"])
    S = R.summarize(rt)[["races", "horses", "horse_win", "horse_top3"] + ROI]
    if ci:
        S = R.with_ci(rt, S, metrics=("roi_horse_w", "roi_horse_p"))
    return S


def baseline(d: pd.DataFrame, y0: int, y1: int) -> dict:
    """比べる基準: 払戻の読める出走を 1 頭ずつ買った回収率。全出走と、パターンが付いた出走（調教の記録がある馬だけ）の 2 通り。"""
    sub = d[d["year"].between(y0, y1) & d["win_ret"].notna() & d["place_ret"].notna()]
    tg = sub[sub["pat"].notna()]
    return {"単勝": float(sub["win_ret"].mean()), "複勝": float(sub["place_ret"].mean()), "頭数": int(len(sub)),
            "単勝(パターン付き)": float(tg["win_ret"].mean()), "複勝(パターン付き)": float(tg["place_ret"].mean()), "頭数(パターン付き)": int(len(tg))}


def persistence(d: pd.DataFrame, sel: tuple[int, int], test: tuple[int, int], nmin: int) -> pd.DataFrame:
    """sel の期間で回収率が高い／低い上位・下位 20% のセルを選び、test の期間でまとめて買った回収率（馬単位）。"""
    a, b = roi_table(d, *sel), roi_table(d, *test)
    m = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
    m = m[(m["races_1"] >= nmin) & (m["races_2"] >= nmin)]
    rows = []
    for k, nm in (("w", "単勝"), ("p", "複勝")):
        col1, col2 = f"roi_horse_{k}_1", f"roi_horse_{k}_2"
        w = m["horses_2"]
        r = float(np.corrcoef(m[col1], m[col2])[0, 1]) if len(m) > 2 else np.nan
        order = m[col1].rank(pct=True)
        allroi = float((m[col2] * w).sum() / w.sum())
        for lab, pick in (("選択期間で上位20%", order >= 0.8), ("選択期間で下位20%", order <= 0.2)):
            x = m[pick]
            rows.append({"賭け式": nm, "対象": lab, "セル数": int(len(x)), "選択期間の回収率": round(float((x[col1] * x["horses_1"]).sum() / x["horses_1"].sum()), 3),
                         "検証期間の回収率": round(float((x[col2] * x["horses_2"]).sum() / x["horses_2"].sum()), 3), "検証期間・全セル": round(allroi, 3), "相関(全セル)": round(r, 3)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jovyan/work/retrain_out/base_2016/article_check/final_week/roi_eval")
    ap.add_argument("--min-races", type=int, default=100)
    ap.add_argument("--laps", default=str(P.LAPS))
    ap.add_argument("--table", default=str(P.TABLE))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = add_returns(P.load(Path(a.laps), Path(a.table)))
    print("# 最終追い切りパターン別の単勝・複勝回収率\n")
    print("回収率 1.00 = 100%。払戻は結果表の確定払戻のみ（市場オッズ・人気は使わない）。馬単位 = 該当馬を 1 頭ずつ同額で全頭買う。レース均等 = 1 レースの賭け金を一定にして該当馬に等分。\n")

    print("## 0. 基準（払戻の読める出走を 1 頭ずつ買った回収率。パターンの比較にはパターン付きの列を使う）\n")
    rows = [{"期間": k, **baseline(d, *v)} for k, v in PERIODS.items()]
    print(md(pd.DataFrame(rows).round(3)))
    miss = d[d["year"].between(*PERIODS["2020〜2025"]) & d["pat"].notna()]
    print(f"\n払戻が読めない出走（パターン付き・2020〜2025）: {int(miss['win_ret'].isna().sum()):,} / {len(miss):,}（回収率の集計から除く）\n")

    print("## 1. パターン別の回収率（2020〜2025）\n")
    T = roi_table(d, *PERIODS["2020〜2025"], ci=True)
    for k, v in list(PERIODS.items())[1:]:
        t = roi_table(d, *v)
        T = T.join(t[["races", "roi_horse_w", "roi_horse_p"]].add_suffix(f"_{k}"))
    T.to_csv(out / "pattern_roi.csv", encoding="utf-8-sig")
    cols = ["races", "horses", "roi_horse_w", "roi_horse_w_lo", "roi_horse_w_hi", "roi_horse_p", "roi_horse_p_lo", "roi_horse_p_hi", "roi_race_w", "roi_race_p"]
    print("### 1-1. 全パターン（レース数順）\n")
    print(md(P.fmt(T.sort_values("races", ascending=False)[cols].reset_index().rename(columns={"_k": "パターン", "index": "パターン"}), nd=3)))
    per = [c for c in T.columns if "(" in c or c.endswith("_2025")]
    print("\n### 1-2. 期間別（単勝・複勝の回収率。前半 2020〜22 / 後半 2023〜24 / 2025）\n")
    keep = ["races"] + [c for c in per if c.startswith("roi_")]
    print(md(P.fmt(T[T["races"] >= a.min_races].sort_values("roi_horse_p", ascending=False)[keep].reset_index().rename(columns={"_k": "パターン", "index": "パターン"}), nd=3)))
    print("\n（100 レース以上のパターンを、2020〜2025 全体の複勝回収率の高い順に並べた。並べ替えに 2025 年を含む。見るための並びで、選択ではない）\n")

    print("## 2. 前の期間で回収率が高かったパターンは、後の期間でも高いか\n")
    for nm, sel, test in (("2020〜22 → 2023〜24", (2020, 2022), (2023, 2024)), ("2020〜24 → 2025（手を付けていない年）", (2020, 2024), (2025, 2025))):
        r = persistence(d, sel, test, nmin=30)
        r.to_csv(out / f"persistence_pattern_{sel[0]}_{sel[1]}_{test[0]}.csv", index=False, encoding="utf-8-sig")
        print(f"### {nm}（両期間とも 30 レース以上のパターン）\n")
        print(md(r))
        print()

    print("## 3. コース（芝/ダート × 距離帯）× パターン別の回収率\n")
    c = CE.with_course(d)
    C = roi_table(c, *PERIODS["2020〜2025"])
    C.to_csv(out / "cell_roi_2020_2025.csv", encoding="utf-8-sig")
    big = C[C["races"] >= a.min_races]
    print(f"セル数 {len(C)}、うち {a.min_races} レース以上 {len(big)}。多数のセルを並べるので、上位は偶然でも出る。判断は 3-2 の再現性で行う。\n")
    sh = ["races", "horses", "horse_win", "horse_top3", "roi_horse_w", "roi_horse_p", "roi_race_w", "roi_race_p"]
    print("### 3-1. 複勝回収率の高いセル 上位15 / 低いセル 下位15\n")
    srt = big.sort_values("roi_horse_p", ascending=False)
    print(md(P.fmt(srt[sh].head(15).reset_index().rename(columns={"_k": "セル", "index": "セル"}), nd=3)))
    print()
    print(md(P.fmt(srt[sh].tail(15).reset_index().rename(columns={"_k": "セル", "index": "セル"}), nd=3)))
    print("\n### 3-1b. 単勝回収率の高いセル 上位15\n")
    print(md(P.fmt(big.sort_values("roi_horse_w", ascending=False)[sh].head(15).reset_index().rename(columns={"_k": "セル", "index": "セル"}), nd=3)))
    for nm, sel, test in (("2020〜22 → 2023〜24", (2020, 2022), (2023, 2024)), ("2020〜24 → 2025（手を付けていない年）", (2020, 2024), (2025, 2025))):
        r = persistence(c, sel, test, nmin=30)
        r.to_csv(out / f"persistence_cell_{sel[0]}_{sel[1]}_{test[0]}.csv", index=False, encoding="utf-8-sig")
        print(f"\n### 3-2. {nm}（両期間とも 30 レース以上のセル）\n")
        print(md(r))


if __name__ == "__main__":
    main()

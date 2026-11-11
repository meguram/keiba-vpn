"""調教師 × 最終追い切りパターンごとの成績表（着順内訳・勝率・単勝回収率・複勝率・複勝回収率）。ターゲットエンコーディングの元データ。

python -m src.scripts.data.trainer_pattern_table [--out DIR] [--y0 2020] [--y1 2025]

出力:
  trainer_pattern_summary_{y0}_{y1}.csv   期間まるごとの集計（見る・調べる用。特徴量には使わない: 未来の結果を含む）
  trainer_pattern_asof.parquet            レースごとに『その日より前の』調教師 × パターンの累計（特徴量用。同日のレースは含めない）

集計の単位は馬（その調教師のそのパターンで出走した馬 1 頭 = 1 行）。回収率 = 払戻 / 賭け金（100 円ずつ買ったとき。1.00 = 100%）。
払戻が結果表から読めない出走は、率・回収率の両方から除く（分母を揃える）。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import final_week_pattern_eval as P
from src.scripts.data import final_week_roi_eval as RO
from src.scripts.data.final_week_check import md

KEY = ["trainer_id", "pat"]
SUMS = ["n", "c1", "c2", "c3", "win_ret_sum", "place_ret_sum"]


def _indicators(d: pd.DataFrame) -> pd.DataFrame:
    """パターンが付き、払戻が読める行に、集計用の 0/1 と払戻を付ける（読めない行は全て 0）。"""
    x = d[d["pat"].notna() & ~d["trainer_id"].isin(["nan", "None", ""])].copy()   # 調教師が不明の行は、1 人の調教師として累計しない
    ok = (x["win_ret"].notna() & x["place_ret"].notna()).to_numpy()
    fp = x["finish_position"].to_numpy()
    x["n"] = ok.astype("int64")
    for k in (1, 2, 3):
        x[f"c{k}"] = (ok & (fp == k)).astype("int64")
    x["win_ret_sum"] = np.where(ok, x["win_ret"].fillna(0.0), 0.0)
    x["place_ret_sum"] = np.where(ok, x["place_ret"].fillna(0.0), 0.0)
    return x


def derive(g: pd.DataFrame, pre: str = "") -> pd.DataFrame:
    """累計（n, c1, c2, c3, 払戻の合計）から、4着以下の頭数と率・回収率を求める。n=0 の行は率が NaN。"""
    n = g["n"].astype("float64").where(g["n"] > 0)
    out = pd.DataFrame(index=g.index)
    out[f"{pre}c4p"] = g["n"] - g["c1"] - g["c2"] - g["c3"]
    out[f"{pre}win_rate"] = g["c1"] / n
    out[f"{pre}win_roi"] = g["win_ret_sum"] / n
    out[f"{pre}top3_rate"] = (g["c1"] + g["c2"] + g["c3"]) / n
    out[f"{pre}place_roi"] = g["place_ret_sum"] / n
    return out


def summary(d: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """調教師 × パターンごとの期間集計。着順内訳は『1着-2着-3着-4着以下』の頭数。"""
    x = _indicators(d[d["year"].between(y0, y1)])
    S = x.groupby(KEY)[SUMS].sum()
    S = S[S["n"] > 0]
    S = S.join(derive(S))
    names = d.sort_values("race_date").drop_duplicates("trainer_id", keep="last").set_index("trainer_id")["trainer_name"]
    S = S.reset_index()
    S["trainer_name"] = S["trainer_id"].map(names)
    S["着順内訳"] = S["c1"].astype(str) + "-" + S["c2"].astype(str) + "-" + S["c3"].astype(str) + "-" + S["c4p"].astype(str)
    S = S.rename(columns={"trainer_id": "調教師ID", "trainer_name": "調教師", "pat": "調教パターン", "n": "出走数", "c1": "1着", "c2": "2着", "c3": "3着", "c4p": "4着以下",
                          "win_rate": "勝率", "win_roi": "単勝回収率", "top3_rate": "複勝率", "place_roi": "複勝回収率"})
    cols = ["調教師ID", "調教師", "調教パターン", "出走数", "着順内訳", "1着", "2着", "3着", "4着以下", "勝率", "単勝回収率", "複勝率", "複勝回収率"]
    return S[cols].sort_values(["調教師ID", "出走数"], ascending=[True, False]).reset_index(drop=True)


def asof_table(d: pd.DataFrame) -> pd.DataFrame:
    """各出走（パターンが付いたもの）に、その調教師 × パターンの『出走日より前の』累計を付ける。同じ日の結果は含めない。
    列: trpat_n, trpat_c1, trpat_c2, trpat_c3, trpat_c4p, trpat_win_rate, trpat_win_roi, trpat_top3_rate, trpat_place_roi（と払戻の合計 2 列）。"""
    x = _indicators(d)
    day = x.groupby(KEY + ["race_date"], sort=True)[SUMS].sum().reset_index()
    cum = day.groupby(KEY)[SUMS].cumsum()
    prior = day[KEY + ["race_date"]].join(cum - day[SUMS])   # 累計から同日分を引く = 厳密に前の日までの合計
    prior = prior.join(derive(prior[SUMS]))
    prior = prior.rename(columns={c: f"trpat_{c}" for c in prior.columns if c not in KEY + ["race_date"]})
    keep = [c for c in ("race_id", "horse_id", "horse_number", "race_date") if c in x.columns] + KEY
    out = x[keep].merge(prior, on=KEY + ["race_date"], how="left", validate="many_to_one")
    assert len(out) == len(x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jovyan/work/retrain_out/base_2016/article_check/final_week/trainer_pattern")
    ap.add_argument("--y0", type=int, default=2020)
    ap.add_argument("--y1", type=int, default=2025)
    ap.add_argument("--laps", default=str(P.LAPS))
    ap.add_argument("--table", default=str(P.TABLE))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = RO.add_returns(P.load(Path(a.laps), Path(a.table)))

    S = summary(d, a.y0, a.y1)
    S.to_csv(out / f"trainer_pattern_summary_{a.y0}_{a.y1}.csv", index=False, encoding="utf-8-sig")
    A = asof_table(d)
    A.to_parquet(out / "trainer_pattern_asof.parquet", index=False)

    print(f"# 調教師 × 調教パターンの成績表（{a.y0}〜{a.y1}）\n")
    print(f"- 組（調教師 × パターン）: {len(S):,}（調教師 {S['調教師ID'].nunique():,} 人、パターン {S['調教パターン'].nunique()} 種）")
    for k in (5, 10, 30, 100):
        print(f"- 出走 {k} 頭以上の組: {int((S['出走数'] >= k).sum()):,}（出走の {S.loc[S['出走数'] >= k, '出走数'].sum() / S['出走数'].sum():.1%} をカバー）")
    tot = S["出走数"].sum()
    print(f"- 全体: 出走 {int(tot):,} 頭、単勝回収率 {(S['単勝回収率'] * S['出走数']).sum() / tot:.3f}、複勝回収率 {(S['複勝回収率'] * S['出走数']).sum() / tot:.3f}")
    ex = S.sort_values("出走数", ascending=False).head(15)
    print("\n## 出走数の多い組（先頭 15）\n")
    print(md(P.fmt(ex, nd=3)))

    pr = A.dropna(subset=["trpat_n"])
    print("\n## 日付つきの累計（特徴量用）\n")
    print(f"- 行数 {len(A):,}（パターンが付いた出走。{d['year'].min()}〜{d['year'].max()}）")
    print(f"- 直前までの出走が 1 頭以上ある行: {(pr['trpat_n'] > 0).mean():.1%}、10 頭以上: {(pr['trpat_n'] >= 10).mean():.1%}、30 頭以上: {(pr['trpat_n'] >= 30).mean():.1%}")
    by = pr.assign(year=pr["race_date"].str[:4]).groupby("year")["trpat_n"].agg(中央値="median", ゼロの割合=lambda s: (s == 0).mean()).round(3)
    print(md(by.reset_index()))
    print("\n注: 2016〜2019 年は調教の記録の被覆が低く（10〜66%）、累計も薄い。2020 年以降を使う場合も、各年の頭の累計は 2016 年からの積み上げ。")


if __name__ == "__main__":
    main()

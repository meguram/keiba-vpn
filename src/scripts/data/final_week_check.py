"""最終追い切り（レース週の平日のベストタイム）の分類が正しいかの点検。結果は標準出力（markdown）。

python -m src.scripts.data.final_week_check [--laps PATH] [--seed 7]
点検: ①時計の整合 ②ラップ種別を別実装で再計算して突き合わせ ③資料の例 6 本 ④選択を総当たりで照合 ⑤分布 ⑥サンプル表示
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import final_week_workout as W
from src.pipeline.features import tp35_lap_groups as L

ROOT = Path(__file__).resolve().parents[3]
LAPS = ROOT / "data/local/modeling/training_pipeline_v4/tp35/training_laps_fresh.parquet"   # 再取得版（2021 年の欠けが少ない）
WD = "月火水木金土日"


def cls_by_hand(l1: float, l2: float) -> str:
    """資料の定義を if 文で素直に書いた別実装（l1 = 終い 1F、l2 = その一つ前の 1F。小数 1 位）。"""
    if l1 != l1 or l2 != l2:
        return "NA"
    a, b = round(l1, 1), round(l2, 1)
    accel = a <= b
    if a < 12.0 and accel:
        return "A3"
    if b < 12.0 and not accel:
        return "B3"
    a12, b12 = 12.0 <= a < 13.0, 12.0 <= b < 13.0
    if a12 and b12:
        return "A2" if accel else "B2"
    if a12 and accel:
        return "A1"
    if b12 and not accel:
        return "B1"
    return "X"


def md(df: pd.DataFrame) -> str:
    if not len(df):
        return "(なし)"
    cell = lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v).replace("|", "/")
    head = "| " + " | ".join(str(c) for c in df.columns) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", default=str(LAPS))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    raw = pd.read_parquet(a.laps)
    raw = raw[(raw["race_date"] >= "2020-01-01") & (raw["race_date"] < "2026-01-01")]
    tp = L.prepare_laps(raw)
    ok = tp[tp["kind"].isin(["slope", "wood", "poly"])]
    print("# 最終追い切り（レース週の平日のベストタイム）の分類点検\n")

    print("## ① 時計の整合（累積時計 t1<t2<t3<t4 か、終い 1F の値の範囲）\n")
    rows = []
    for k, cols in (("slope", ["t1", "t2", "t3", "t4"]), ("wood", ["t1", "t3", "t4"]), ("poly", ["t1", "t3", "t4"])):
        s = ok[ok["kind"] == k].dropna(subset=cols)
        mono = np.ones(len(s), bool)
        for x, y in zip(cols[:-1], cols[1:]):
            mono &= (s[x] < s[y]).to_numpy()
        rows.append({"種別": k, "行数": len(s), "累積が単調増加": f"{mono.mean():.4f}", "l1 が 10〜16": f"{s['l1'].between(10, 16).mean():.4f}",
                     "l2 が 10〜20": f"{s['l2'].between(10, 20).mean():.4f}", "l1 中央値": s["l1"].median(), "l2 中央値": s["l2"].median()})
    print(md(pd.DataFrame(rows)))
    sl = ok[ok["kind"] == "slope"].dropna(subset=["slope_l2", "t2"])
    print(f"\n坂路の l2（= t2−t1）と、元の表の slope_l2 の差が 0.05 以内の割合: {((sl['l2'] - sl['slope_l2']).abs() <= 0.05).mean():.4f}（{len(sl):,} 本）")
    print(f"終い 1F の l1 と元の表の l1 の一致割合: {((ok['l1'] - raw.loc[ok.index, 'l1']).abs() <= 0.051).mean():.4f}\n")

    print("## ② ラップ種別を別実装（if 文）で再計算して突き合わせ\n")
    hand = np.array([cls_by_hand(x, y) if k in ("slope", "wood", "poly") else "NA" for x, y, k in zip(tp["l1"], tp["l2"], tp["kind"])])
    diff = hand != tp["cls"].to_numpy()
    print(f"全 {len(tp):,} 本のうち不一致: {int(diff.sum())} 本\n")
    if diff.any():
        print(md(tp.loc[diff, ["l1", "l2", "cls"]].assign(hand=hand[diff]).head(10)))

    print("## ③ 資料の例 6 本（最後の 2 つが l2・l1）\n")
    ex = [("A1", "15.0-14.0-13.1-12.5"), ("B1", "15.0-14.0-12.8-13.1"), ("A2", "15.0-14.0-12.5-12.1"), ("B2", "15.0-14.0-12.2-12.6"),
          ("A3", "15.0-14.0-12.5-11.9"), ("B3", "15.0-14.0-11.8-12.7")]
    r = []
    for want, lap in ex:
        v = [float(x) for x in lap.split("-")]
        got = L.classify_lap(np.array([v[-1]]), np.array([v[-2]]))[0]
        r.append({"資料の種別": want, "例のラップ": lap, "l2(後ろから2つ目)": v[-2], "l1(終い)": v[-1], "判定": got, "一致": "○" if got == want else "×"})
    print(md(pd.DataFrame(r)))

    print("\n## ④ ベストタイムの選択を総当たりで照合（無作為 3,000 組）\n")
    bw = W.pick_best_weekday(tp)
    rng = np.random.default_rng(a.seed)
    keys = bw[["horse_id", "race_date"]].drop_duplicates().sample(3000, random_state=a.seed)
    cand = tp.merge(keys, on=["horse_id", "race_date"])
    bad = 0
    for (h, d), g in cand.groupby(["horse_id", "race_date"]):
        rd = pd.Timestamp(d)
        ws = rd - pd.Timedelta(days=7 if rd.weekday() == 0 else rd.weekday())
        c = [(float(x.t4), int(x.days_before)) for x in g.itertuples() if x.cls != "NA" and x.t4 == x.t4 and ws <= pd.Timestamp(x.train_date) <= ws + pd.Timedelta(days=4)]
        exp = min(c) if c else None
        got = bw[(bw["horse_id"] == h) & (bw["race_date"] == d)]
        if exp is None or len(got) != 1 or (float(got["BW_t4"].iloc[0]), int(got["BW_days"].iloc[0])) != exp:
            bad += 1
    print(f"照合した 3,000 組のうち、別実装の最小（t4, days_before）と違う: {bad} 組\n")
    print("（参考）選ばれた本の調教曜日は月〜金のみか: ", sorted(set(pd.to_datetime(tp.merge(bw, left_on=["horse_id", "race_date", "t4", "days_before"],
          right_on=["horse_id", "race_date", "BW_t4", "BW_days"])["train_date"]).dt.weekday)), "（0=月 … 4=金）\n")

    print("## ⑤ 分布\n")
    yr = pd.to_datetime(bw["race_date"]).dt.year
    print("年別の件数（出走ごと。調教が読めた分）:", yr.value_counts().sort_index().to_dict(), "\n")
    print("選ばれた本の調教場所:\n")
    print(md(bw["BW_place"].value_counts().rename_axis("場所").reset_index(name="本数")))
    print("\n調教場所 × ラップ種別（行ごとの割合）:\n")
    ct = pd.crosstab(bw["BW_place"], bw["BW_cls"], normalize="index").round(3)
    ct["本数"] = bw["BW_place"].value_counts()
    print(md(ct.reset_index().sort_values("本数", ascending=False)))
    print("\n坂路: 調教場所 × ラップ区分の本数（BW_pat）:\n")
    sl_b = bw[bw["BW_kind"] == "slope"]
    ct2 = pd.crosstab(sl_b["BW_place"], sl_b["BW_pat"])
    print(md(ct2.reset_index()))
    print("\nウッド・ポリ: 調教場所 × 4F 時計の区分の本数（BW_pat）:\n")
    wd_b = bw[bw["BW_kind"] != "slope"]
    ct3 = pd.crosstab(wd_b["BW_place"], wd_b["BW_pat"]).reindex(columns=W.WOOD_LABELS, fill_value=0)
    print(md(ct3.reset_index()))
    chk = W.wood_band(wd_b["BW_t4"].round(1))
    lo = wd_b["BW_t4"].round(1)
    edges_ok = all(((lo >= a) & (lo < b)).eq(chk.eq(lab)).all() for a, b, lab in zip([0.0] + W.WOOD_EDGES, W.WOOD_EDGES + [999.0], W.WOOD_LABELS))
    print(f"\n区分が [下限, 上限) の半開区間と全件一致か（別の書き方で照合）: {edges_ok}\n")
    print("\n選ばれた本のレースまでの日数（BW_days）:", bw["BW_days"].value_counts().sort_index().to_dict(), "\n")
    print(f"ウッド・ポリ（4F 時計の区分で扱う）の割合: {(bw['BW_kind'] != 'slope').mean():.3f}\n")

    print("## ⑥ サンプル（その週の平日の追い切りをすべて並べ、★が選ばれた本）\n")
    samp = []
    for p in ["栗東坂路", "美浦坂路"]:
        for c in ["A1", "A2", "A3", "B1", "B2", "B3", "X"]:
            x = bw[(bw["BW_place"] == p) & (bw["BW_pat"] == c) & (bw["BW_n"] >= 2)]
            if len(x):
                samp.append(x.sample(1, random_state=a.seed))
    for p in ["Cウッド(栗東)", "美浦ウッド(南)", "美浦ウッド(北)", "美浦ポリ"]:
        for c in ["W50.0-51.9", "W52.0-53.9", "W54.0-55.9", "W56.0-57.9"]:
            x = bw[(bw["BW_place"] == p) & (bw["BW_pat"] == c) & (bw["BW_n"] >= 2)]
            if len(x):
                samp.append(x.sample(1, random_state=a.seed))
    for g in pd.concat(samp).drop_duplicates(["horse_id", "race_date"]).itertuples():
        rd = pd.Timestamp(g.race_date)
        ws = rd - pd.Timedelta(days=7 if rd.weekday() == 0 else rd.weekday())
        w = tp[(tp["horse_id"] == g.horse_id) & (tp["race_date"] == g.race_date)].sort_values("train_date")
        w = w[pd.to_datetime(w["train_date"]).between(ws - pd.Timedelta(days=7), rd)]
        t = pd.DataFrame({"": np.where((w["train_date"] == w["train_date"]) & (w["t4"] == g.BW_t4) & (w["days_before"] == g.BW_days) & (w["cls"] != "NA"), "★", ""),
                          "調教日": w["train_date"].astype(str).str[5:] + "(" + pd.to_datetime(w["train_date"]).dt.weekday.map(lambda x: WD[x]) + ")",
                          "レースまで": w["days_before"], "場所": W.place_of(w["course"]), "1F": w["t1"], "2F": w["t2"], "3F": w["t3"], "4F": w["t4"],
                          "終い1F(l1)": w["l1"], "その前1F(l2)": w["l2"], "ラップ種別(参考)": w["cls"]})
        print(f"**馬 {g.horse_id} / レース日 {g.race_date}({WD[rd.weekday()]}) / 採用: {g.BW_place} {g.BW_pat}**（累積時計 1F〜4F。ウッド・ポリは2Fなし。区分は4F時計）\n")
        print(md(t))
        print()


if __name__ == "__main__":
    main()

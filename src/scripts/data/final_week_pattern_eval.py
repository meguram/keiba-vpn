"""最終追い切り（レース週の平日のベストタイム）のパターン別の成績を、2 つの向きで集計する。

python -m src.scripts.data.final_week_pattern_eval [--out DIR] [--min-races 30]

パターン = 調教場所 × 区分（坂路はラップ A1〜X、ウッド・ポリは 4F 時計の 2 秒刻み）。
向き A（パターン → 成績）: 調教師 × パターンごとの勝率・複勝率（レース単位。無作為に選んだ場合の期待値との比 = lift）。
向き B（3 着以内 → パターン）: 3 着以内の馬のうち、どの調教師のどのパターンが多いか（調教師の普段の使い方と比べた比）。
期間: 主 = 2020〜2025（調教の記録の被覆が揃う）。2016〜2019 は被覆が年 10〜66% で生き残りの偏りがあるため、パターン全体の表だけ別に出す。
2026 年・市場オッズ・人気は使わない。成績は結果表の着順のみ（払戻は扱わない）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import final_week_workout as W
from src.pipeline.features import tp35_lap_groups as L
from src.scripts.data import tp35_race_level as R
from src.scripts.data.final_week_check import md

ROOT = Path(__file__).resolve().parents[3]
LAPS = ROOT / "data/local/modeling/training_pipeline_v4/tp35/training_laps_fresh.parquet"
TABLE = Path("/home/jovyan/work/retrain_out/base_2016/features/train_table.parquet")
MAIN = (2020, 2025)
HALVES = ((2020, 2022), (2023, 2024))   # 再現性の前半・後半。2025 は選ばずに最後に 1 回だけ見る
HOLDOUT = 2025


def load(laps: Path, table: Path) -> pd.DataFrame:
    t = pd.read_parquet(table, columns=["race_id", "date", "surface", "field_size", "horse_id", "trainer_id", "trainer_name", "finish_position"])
    t = t[t["surface"].astype(str).isin(["芝", "ダート"]) & (t["finish_position"] > 0) & (t["date"].astype(str) < "2026-01-01")].copy()
    t["race_id"], t["horse_id"], t["trainer_id"] = t["race_id"].astype(str), t["horse_id"].astype(str), t["trainer_id"].astype(str)
    t["trainer_name"] = t["trainer_name"].astype(str)
    t["race_date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
    t["year"] = t["race_date"].str[:4].astype(int)
    raw = pd.read_parquet(laps)
    raw = raw[(raw["race_date"] >= "2016-01-01") & (raw["race_date"] < "2026-01-01")].copy()
    raw["horse_id"] = raw["horse_id"].astype(str)
    bw = W.pick_best_weekday(L.prepare_laps(raw))
    bw["horse_id"] = bw["horse_id"].astype(str)
    d = t.merge(bw[["horse_id", "race_date", "BW_place", "BW_pat"]], on=["horse_id", "race_date"], how="left")
    assert len(d) == len(t)
    assert d["BW_pat"].notna().mean() > 0.3, "調教パターンの結合率が低すぎる（キーの型を確認）"
    d["pat"] = d["BW_place"] + "|" + d["BW_pat"]
    d["win"], d["t3"] = (d["finish_position"] == 1).astype("int8"), (d["finish_position"] <= 3).astype("int8")
    d["t2"] = (d["finish_position"] <= 2).astype("int8")
    d["fs"] = d.groupby("race_id")["win"].transform("size")
    for c in ("win_ret", "place_ret", "win_odds", "p_win"):
        d[c] = np.nan
    return d.reset_index(drop=True)


def stats(sub: pd.DataFrame, key: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """key ごとの レース単位の勝率・複勝率と期待値・比、馬単位の率、lift の標準誤差。"""
    rt = R.race_table(sub, key)
    S = R.summarize(rt)
    S = S.drop(columns=[c for c in S.columns if c.startswith("roi_") or c in ("n_fav", "n_mod")])
    g = rt.assign(vw=rt["exp_win"] * (1 - rt["exp_win"]), v3=rt["exp_top3"] * (1 - rt["exp_top3"])).groupby(level=0)
    S["se_win"] = np.sqrt(g["vw"].sum()) / g["exp_win"].sum()
    S["se_top3"] = np.sqrt(g["v3"].sum()) / g["exp_top3"].sum()
    return S, rt


def pooled(d: pd.DataFrame, y0: int, y1: int, ci: bool = True) -> pd.DataFrame:
    sub = d[d["year"].between(y0, y1) & d["pat"].notna()]
    S, rt = stats(sub, sub["pat"])
    if ci:
        S = R.with_ci(rt, S, metrics=("race_win", "race_top3"))
    return S


def cells(d: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    sub = d[d["year"].between(y0, y1) & d["pat"].notna()]
    key = sub["trainer_id"] + "|" + sub["pat"]
    S, _ = stats(sub, key)
    S["trainer_id"], S["pat"] = [i.split("|", 1)[0] for i in S.index], [i.split("|", 1)[1] for i in S.index]
    T, _ = stats(sub, sub["trainer_id"])
    S = S.join(T[["races", "lift_win", "lift_top3", "se_win", "se_top3"]].add_prefix("tr_"), on="trainer_id")
    for k in ("win", "top3"):
        S[f"rel_{k}"] = (S[f"lift_{k}"] / S[f"tr_lift_{k}"]).replace([np.inf, -np.inf], np.nan)
        S[f"z_{k}"] = (S[f"lift_{k}"] - S[f"tr_lift_{k}"]) / np.sqrt(S[f"se_{k}"] ** 2 + S[f"tr_se_{k}"] ** 2)
    return S


def pattern_rel(C: pd.DataFrame) -> pd.DataFrame:
    """パターンごとに、調教師の基準（その調教師の全パターンの比）を引いた効果 rel = 実際 / 調教師の基準で期待される数。"""
    rows = {}
    for k in ("win", "top3"):
        e = C["races"] * C[f"exp_{k}"]
        obs = C["races"] * C[f"race_{k}"]
        base = e * C[f"tr_lift_{k}"]
        g = pd.DataFrame({"obs": obs, "base": base, "var": (C[f"se_{k}"] * e) ** 2, "pat": C["pat"]}).groupby("pat").sum()
        rows[f"rel_{k}"] = g["obs"] / g["base"]
        rows[f"se_{k}"] = np.sqrt(g["var"]) / g["base"]
    out = pd.DataFrame(rows)
    out.insert(0, "races", C.groupby("pat")["races"].sum())
    return out


def reproducibility(d: pd.DataFrame, sel: tuple[int, int], test: tuple[int, int], nmin: int = 20) -> pd.DataFrame:
    """sel の期間で出したパターン固有の効果（rel）が test の期間で続くか。"""
    a, b = cells(d, *sel), cells(d, *test)
    m = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
    m = m[(m["races_1"] >= nmin) & (m["races_2"] >= nmin)]
    rows = []
    for k, nm in (("win", "勝率"), ("top3", "複勝率")):
        mm = m[np.isfinite(m[f"rel_{k}_1"]) & np.isfinite(m[f"rel_{k}_2"])]
        r = float(np.corrcoef(mm[f"rel_{k}_1"], mm[f"rel_{k}_2"])[0, 1])
        order = mm[f"z_{k}_1"].rank(pct=True)
        for lab, pick in (("選択期間で上位20%", order >= 0.8), ("選択期間で下位20%", order <= 0.2)):
            x = mm[pick]
            rows.append({"指標": nm, "対象": lab, "セル数": int(len(x)), "選択期間のrel(平均)": round(float(x[f"rel_{k}_1"].mean()), 3),
                         "検証期間のrel(平均)": round(float(x[f"rel_{k}_2"].mean()), 3), "相関(全セル)": round(r, 3)})
    return pd.DataFrame(rows)


def direction_b(d: pd.DataFrame, y0: int, y1: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = d[d["year"].between(y0, y1) & d["pat"].notna()].copy()
    # B1: 全体のパターン構成（馬単位）
    allsh = sub["pat"].value_counts(normalize=True).rename("全出走の構成比")
    t3 = sub[sub["t3"] == 1]
    t3sh = t3["pat"].value_counts(normalize=True).rename("3着以内の構成比")
    b1 = pd.concat([allsh, t3sh], axis=1).fillna(0)
    b1["比"] = b1["3着以内の構成比"] / b1["全出走の構成比"]
    b1["出走数"] = sub["pat"].value_counts()
    b1["3着以内の頭数"] = t3["pat"].value_counts()
    b1 = b1.fillna(0)
    # B2: 調教師 × パターンのセル
    tr_n = sub.groupby("trainer_id").size().rename("tr_runs")
    tr_t3 = t3.groupby("trainer_id").size().rename("tr_t3")
    c_all = sub.groupby(["trainer_id", "pat"]).size().rename("runs")
    c_t3 = t3.groupby(["trainer_id", "pat"]).size().rename("t3_horses")
    b2 = pd.concat([c_all, c_t3], axis=1).fillna(0).reset_index()
    b2 = b2.join(tr_n, on="trainer_id").join(tr_t3, on="trainer_id")
    b2["全3着以内に占める割合"] = b2["t3_horses"] / len(t3)
    b2["調教師の使用率"] = b2["runs"] / b2["tr_runs"]
    b2["調教師の3着以内に占める割合"] = b2["t3_horses"] / b2["tr_t3"]
    b2["比"] = b2["調教師の3着以内に占める割合"] / b2["調教師の使用率"]
    p = b2["調教師の使用率"].clip(1e-9, 1 - 1e-9)
    b2["z"] = (b2["t3_horses"] - b2["tr_t3"] * p) / np.sqrt(b2["tr_t3"] * p * (1 - p))
    nm = sub.drop_duplicates("trainer_id").set_index("trainer_id")["trainer_name"]
    b2.insert(1, "調教師", b2["trainer_id"].map(nm))
    # B3: 調教師ごとに 3 着以内で最も多いパターン
    b3 = b2[b2["tr_t3"] >= 60].sort_values(["trainer_id", "t3_horses"], ascending=[True, False]).groupby("trainer_id").head(1)
    return b1.sort_values("3着以内の頭数", ascending=False), b2, b3.sort_values("tr_t3", ascending=False)


def fmt(df: pd.DataFrame, pct=(), nd=3) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c in pct:
            df[c] = (df[c] * 100).round(1).astype(str) + "%"
        elif df[c].dtype.kind == "f":
            df[c] = df[c].round(nd)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jovyan/work/retrain_out/base_2016/article_check/final_week/pattern_eval")
    ap.add_argument("--min-races", type=int, default=30)
    ap.add_argument("--laps", default=str(LAPS))
    ap.add_argument("--table", default=str(TABLE))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(Path(a.laps), Path(a.table))
    print("# 最終追い切りパターン別の成績（両方向）\n")

    print("## 0. 調教の記録が読めた出走の割合（芝・ダート、パターンが付いた割合）\n")
    cov = d.groupby("year").agg(出走=("win", "size"), パターン付き=("pat", "count"))
    cov["被覆"] = (cov["パターン付き"] / cov["出走"]).round(3)
    print(md(cov.reset_index()))
    print("\n2016〜2019 は被覆が低く、生き残った馬に偏る。主の集計は 2020〜2025。\n")

    print("## 1. パターン全体（全調教師合算）の勝率・複勝率 — 向き A の土台\n")
    cols = ["races", "horse_win", "race_win", "exp_win", "lift_win", "race_win_lo", "race_win_hi", "horse_top3", "race_top3", "exp_top3", "lift_top3", "race_top3_lo", "race_top3_hi"]
    P = pooled(d, *MAIN)
    P.to_csv(out / "pattern_pooled_2020_2025.csv", encoding="utf-8-sig")
    print("### 1-1. 2020〜2025（レース単位。比 = 実際 / 無作為に選んだ場合の期待値）\n")
    print(md(fmt(P[cols].reset_index().rename(columns={"_k": "パターン", "index": "パターン"}))))
    P0 = pooled(d, 2016, 2019, ci=False)
    P0.to_csv(out / "pattern_pooled_2016_2019.csv", encoding="utf-8-sig")
    print("\n### 1-2. 2016〜2019（参考。被覆が低く偏る）と 2020〜2025 の比の比べ\n")
    cmp_ = P[["races", "lift_win", "lift_top3"]].join(P0[["races", "lift_win", "lift_top3"]], rsuffix="_16_19", how="outer")
    print(md(fmt(cmp_.reset_index().rename(columns={"_k": "パターン", "index": "パターン"}))))

    print("\n## 2. 向き A: 調教師 × パターンで勝率・複勝率はどう変わるか（2020〜2025）\n")
    C = cells(d, *MAIN)
    nm = d.drop_duplicates("trainer_id").set_index("trainer_id")["trainer_name"]
    C.insert(0, "調教師", C["trainer_id"].map(nm))
    C.to_csv(out / "cells_A_2020_2025.csv", encoding="utf-8-sig")
    big = C[C["races"] >= a.min_races]
    print(f"セル数 {len(C):,}、うち {a.min_races} レース以上 {len(big):,}。rel = そのセルの比 / その調教師の全パターンの比（パターン固有の効果）。z は差の目安（正規近似。調教師の基準にそのセル自身が含まれるため rel は 1 に寄る側にずれる。分散は帰無の値で近似）。\n多数のセルを並べるので、上位・下位は偶然でも出る。判断は 2-3・2-4 の再現性で行う。\n母数は調教パターンが付いた馬だけ（期待値は出走全頭の頭数 fs から超幾何で計算）。\n")
    nz = int((big["z_top3"].abs() >= 3).sum()), int((big["z_win"].abs() >= 3).sum())
    print(f"|z|≥3 のセル数: 複勝率 {nz[0]}、勝率 {nz[1]}（正規近似での偶然の目安 {len(big) * 0.0027:.1f} 件。小さいセルでは右裾が重く、実際はこれより多い）\n")
    show = ["調教師", "pat", "races", "horse_win", "race_win", "lift_win", "race_top3", "lift_top3", "tr_lift_top3", "rel_top3", "z_top3"]
    print("### 2-0. パターン別: 調教師の基準を引いた効果 rel（1 より大 = 同じ調教師の他のパターンより良い）。前半・後半の別も\n")
    pr = pattern_rel(C)
    h1, h2 = pattern_rel(cells(d, *HALVES[0])), pattern_rel(cells(d, *HALVES[1]))
    pr = pr.join(h1[["rel_win", "rel_top3"]].add_suffix("_前半(20-22)")).join(h2[["rel_win", "rel_top3"]].add_suffix("_後半(23-24)"))
    pr.to_csv(out / "pattern_rel_2020_2025.csv", encoding="utf-8-sig")
    print(md(fmt(pr.reset_index().rename(columns={"pat": "パターン"}), nd=2)))
    print()
    print("### 2-1. 複勝率の rel が高いセル 上位15\n")
    print(md(fmt(big.sort_values("z_top3", ascending=False)[show].head(15))))
    print("\n### 2-2. 複勝率の rel が低いセル 下位15\n")
    print(md(fmt(big.sort_values("z_top3")[show].head(15))))
    print("\n### 2-3. 前半（2020〜22）の結果が後半（2023〜24）に続くか（各半期 20 レース以上のセル）\n")
    rep = reproducibility(d, *HALVES)
    rep.to_csv(out / "reproducibility.csv", index=False, encoding="utf-8-sig")
    print(md(rep))
    print(f"\n### 2-4. 2020〜2024 で選んだセルが、手を付けていない {HOLDOUT} 年で続くか（各期間 20 レース以上のセル）\n")
    rep2 = reproducibility(d, (MAIN[0], HOLDOUT - 1), (HOLDOUT, HOLDOUT), nmin=10)
    rep2.to_csv(out / "reproducibility_holdout.csv", index=False, encoding="utf-8-sig")
    print(md(rep2))

    print("\n## 3. 向き B: 3 着以内の馬は、どの調教師のどのパターンが多いか（2020〜2025、馬単位）\n")
    b1, b2, b3 = direction_b(d, *MAIN)
    b1.to_csv(out / "B1_pattern_share.csv", encoding="utf-8-sig")
    b2.to_csv(out / "B2_cells.csv", index=False, encoding="utf-8-sig")
    b3.to_csv(out / "B3_trainer_top_pattern.csv", index=False, encoding="utf-8-sig")
    print("### 3-1. パターン別: 3 着以内の構成比 と 全出走の構成比（比 > 1 なら 3 着以内に偏る）\n")
    print(md(fmt(b1.reset_index().rename(columns={"index": "パターン"}), pct=("全出走の構成比", "3着以内の構成比"))))
    print("\n### 3-2. 3 着以内の頭数が多い 調教師 × パターン 上位20（使用率 = その調教師が普段そのパターンを使う割合）\n")
    s2 = ["調教師", "pat", "t3_horses", "runs", "全3着以内に占める割合", "調教師の使用率", "調教師の3着以内に占める割合", "比", "z"]
    print(md(fmt(b2.sort_values("t3_horses", ascending=False)[s2].head(20), pct=("全3着以内に占める割合", "調教師の使用率", "調教師の3着以内に占める割合"))))
    print("\n### 3-3. 使用率に比べて 3 着以内に偏るセル（3 着以内 10 頭以上）上位15 / 下位15\n")
    e = b2[b2["t3_horses"] >= 10].sort_values("z", ascending=False)
    print(md(fmt(e[s2].head(15), pct=("全3着以内に占める割合", "調教師の使用率", "調教師の3着以内に占める割合"))))
    print()
    print(md(fmt(e[s2].tail(15), pct=("全3着以内に占める割合", "調教師の使用率", "調教師の3着以内に占める割合"))))
    print("\n### 3-4. 調教師ごとに、3 着以内で最も多いパターン（3 着以内 60 頭以上の調教師、頭数順 上位20）\n")
    print(md(fmt(b3[s2 + ["tr_t3"]].head(20), pct=("全3着以内に占める割合", "調教師の使用率", "調教師の3着以内に占める割合"))))


if __name__ == "__main__":
    main()

"""新しい種牡馬への備えの確かめ（ユーザーの指摘・11/26。開発の行・記述・2025・26 は開けない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.newsire_eval

- 行: 2019〜24 年の開発の行（split 0・1）。予測は 3 つを並べる:
  - 時間の順（walk-forward・oof_v4_pure: 年 t を t より前の年で学ぶ）
  - 開催の単位の無作為の OOF（kf_v4_pure_meeting）
  - 2 代母の単位の OOF（kf_v4_pure_dd）
- 帯: その走りの時点での父の産駒の as-of の走数（sire_all_n = log1p(走数)）の 0・1〜20・21〜100・101〜500・500 超と、初年度産駒か
  （父の産駒の生年の最小 = この馬の生年、かつ最小 ≥ 2014）、時間の順の学習の期間（その年より前）に父の産駒が 1 走も無いか。
- 物差し: 勝ち = 帯の中の勝ち馬の「log p − log(1/頭数)」の平均（レースの対数損失の、その帯の勝ち馬の分）。
  3 着内 = 帯の行の二値の対数損失の、一様（min(3,n)/n）からの改善。較正 = Σ実際 ÷ Σ予測。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "newsire_eval",
                                   "新しい種牡馬の帯ごとの精度（時間の順と無作為の OOF の比べ・尤度だけ）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "sire", "sire_all_n", "win", "top3", "n_run"])
    X = X[(X["date"].dt.year >= 2017) & (X["date"].dt.year <= 2024)].reset_index(drop=True)
    first_run = X.groupby("sire")["date"].min()
    X["Y"] = X["date"].dt.year
    by = pd.to_numeric(X["horse_id"].str[:4], errors="coerce")
    H = X.drop_duplicates("horse_id").assign(by=by)
    min_by = H.groupby("sire")["by"].min()
    X["first_crop"] = (by == X["sire"].map(min_by)) & (X["sire"].map(min_by) >= 2014)
    X["unseen_wf"] = X["sire"].map(first_run) >= pd.to_datetime(X["Y"].astype(str) + "-01-01")
    n = np.expm1(X["sire_all_n"].to_numpy())
    X["band"] = pd.cut(n, [-0.5, 0.5, 20.5, 100.5, 500.5, np.inf], labels=["0 走", "1〜20 走", "21〜100 走", "101〜500 走", "500 走超"]).astype(str)
    WF = pd.read_parquet(data.OUT / "oof_v4_pure.parquet")[["race_id", "horse_id", "p_win", "p_top3"]].rename(columns={"p_win": "wf_win", "p_top3": "wf_top3"})
    KM = pd.read_parquet(data.OUT / "kf_v4_pure_meeting.parquet")[["race_id", "horse_id", "p_win", "p_top3"]].rename(columns={"p_win": "km_win", "p_top3": "km_top3"})
    KD = pd.read_parquet(data.OUT / "kf_v4_pure_dd.parquet")[["race_id", "horse_id", "p_win", "p_top3"]].rename(columns={"p_win": "kd_win", "p_top3": "kd_top3"})
    S = X[X["split"].isin([0, 1]) & (X["Y"] >= 2019)].merge(WF, on=["race_id", "horse_id"], how="inner").merge(KM, on=["race_id", "horse_id"], how="inner").merge(KD, on=["race_id", "horse_id"], how="inner")
    full = S.groupby("race_id")["horse_id"].transform("size") == S["n_run"]
    S = S[full].reset_index(drop=True)
    has_sire = S["sire"].fillna("").ne("").to_numpy()              # 父が空の行は帯に入れない（as-of の件数が 0 になるため。critic）
    late = (S["Y"] >= 2022).to_numpy()                             # 時間の順の学習の量が足りる後の年だけの比べ（critic）
    u1 = 1 / S["n_run"].to_numpy(float)
    u3 = np.minimum(3, S["n_run"].to_numpy(float)) / S["n_run"].to_numpy(float)
    w, t3 = S["win"].to_numpy(float), S["top3"].to_numpy(float)

    def bll(y, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def block(mk):
        out = {"rows": int(mk.sum()), "wins": int(w[mk].sum())}
        for tag in ("wf", "km", "kd"):
            pw, p3 = S[f"{tag}_win"].to_numpy(), S[f"{tag}_top3"].to_numpy()
            m1 = mk & (w == 1)
            # 主の物差しは行ごとの二値の対数損失（勝ち）の改善。帯の勝ち馬だけの物差しは、帯に確率を多く配るほど良く出るので参考（critic）
            out[tag] = {"win_bll_gain": round(float(np.mean(bll(w[mk], u1[mk]) - bll(w[mk], pw[mk]))), 5),
                        "win_gain": round(float(np.mean(np.log(np.clip(pw[m1], 1e-9, 1)) - np.log(u1[m1]))), 4) if m1.sum() else None,
                        "top3_gain": round(float(np.mean(bll(t3[mk], u3[mk]) - bll(t3[mk], p3[mk]))), 4),
                        "win_ratio": round(float(w[mk].sum() / pw[mk].sum()), 3), "top3_ratio": round(float(t3[mk].sum() / p3[mk].sum()), 3)}
        return out
    res = {"rows": int(len(S)), "years": "2019〜2024（開発の行）", "bands": {}, "first_crop": {}, "unseen_wf": {}}
    for b in ["0 走", "1〜20 走", "21〜100 走", "101〜500 走", "500 走超"]:
        res["bands"][b] = block((S["band"] == b).to_numpy() & has_sire)
    fc, uw = S["first_crop"].to_numpy(), S["unseen_wf"].to_numpy()
    res["first_crop"]["初年度産駒"] = block(fc & has_sire)
    res["first_crop"]["それ以外"] = block(~fc & has_sire)
    res["unseen_wf"]["学習の期間に父の産駒が 1 走も無い"] = block(uw & has_sire)
    res["unseen_wf"]["それ以外"] = block(~uw & has_sire)
    res["unseen_wf_2022_24"] = {"学習の期間に父の産駒が 1 走も無い": block(uw & has_sire & late), "それ以外": block(~uw & has_sire & late)}
    res["all"] = block(np.ones(len(S), bool))
    res["all_2022_24"] = block(late)
    path = data.OUT / "newsire_eval.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False), flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()

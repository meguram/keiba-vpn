"""レースごとの本賞金（1 着の本賞金）の表を、手元の SmartRC 由来の馬ごとの戦績から作る（0 円・GCS を読まない）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.prize_table

- 元: data/local/modeling/training_pipeline_v4/tp32/career_eq.parquet（馬 × 戦績・prize と subplize は百円の単位と推定）と、
  data/page_reference/tables/{2020..2024}/smartrc_race_race.parquet の fullresults（全頭）。中央（jyocd 01〜10）だけ。
- JRA の本賞金は 2 着 40%・3 着 25%・4 着 15%・5 着 10%（1 着に対して）。1〜5 着の行の prize を比で割り、1 着の本賞金を推す。
  同じレースで推した値の中央値を使い、ずれ（比が合わないレース）の割合も出す。付加賞（subplize）は入れない。
- 出力: OUT/prize_race.parquet（race_id・p1〔万円〕・n_src・ok・pz1〜pz5〔着順ごとの実額・万円〕）。
  各走の本賞金は、実額があればそれ、無ければ p1 × 比（着順）。2〜5 着は 10 万円の単位で丸めるので、比は 10% まで許す。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data

ROOT = Path("/home/jovyan/work/keiba-vpn")
RATIO = {1: 1.0, 2: 0.4, 3: 0.25, 4: 0.15, 5: 0.1}


def from_career() -> pd.DataFrame:
    C = pd.read_parquet(ROOT / "data/local/modeling/training_pipeline_v4/tp32/career_eq.parquet",
                        columns=["race_id", "jyocd", "arrival", "prize"])
    return C


def from_smartrc() -> pd.DataFrame:
    rows = []
    for y in range(2020, 2025):
        f = ROOT / f"data/page_reference/tables/{y}/smartrc_race_race.parquet"
        if not f.exists():
            continue
        R = pd.read_parquet(f, columns=["race_id", "fullresults"])
        for rid, fr in zip(R["race_id"], R["fullresults"]):
            try:
                d = json.loads(fr) if isinstance(fr, str) else (fr or {})
            except Exception:
                continue
            for recs in d.values():
                for r in recs or []:
                    rc = str(r.get("rcode") or "")
                    if len(rc) != 16:
                        continue
                    # fullresults は出走馬ごとの全戦績で、各記録に race_id が無い。rcode（yyyymmdd 場 回 日 R）から 12 桁に直す（critic）
                    rows.append((rc[:4] + rc[8:16], r.get("jyocd"), r.get("arrival"), r.get("prize")))
    return pd.DataFrame(rows, columns=["race_id", "jyocd", "arrival", "prize"])


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "prize_table", "本賞金の表を作る（着順と賞金の対応だけ・予測は見ない）"))
    S = pd.concat([from_career(), from_smartrc()], ignore_index=True)
    S = S[S["race_id"].astype(str).str.fullmatch(r"\d{12}")]
    S = S[S["jyocd"].astype(str).str.zfill(2).isin([f"{i:02d}" for i in range(1, 11)])]
    S["arrival"] = pd.to_numeric(S["arrival"], errors="coerce")
    S["prize"] = pd.to_numeric(S["prize"], errors="coerce")
    S = S[S["arrival"].between(1, 5) & (S["prize"] > 0)].drop_duplicates(["race_id", "arrival", "prize"])
    S["p1"] = S["prize"] / S["arrival"].map(RATIO) / 100.0          # 百円 → 万円
    g = S.groupby("race_id")["p1"].agg(["median", "min", "max", "size"]).reset_index()
    g["ok"] = (g["max"] / g["min"] < 1.10)                         # 2〜5 着は 10 万円の単位で丸めるので、10% までのずれは許す
    # 同着の実額は分けた後の額で、比で割ると低く出るので、推した値の最大を 1 着の額にする（critic）
    T = g.rename(columns={"max": "p1", "size": "n_src"})[["race_id", "p1", "n_src", "ok"]]
    # 着順ごとの実額（ある分）。各走の本賞金は、実額があればそれ、無ければ p1 × 比
    P = S.groupby(["race_id", "arrival"])["prize"].median().div(100.0).unstack()
    P.columns = [f"pz{int(c)}" for c in P.columns]
    T = T.merge(P.reset_index(), on="race_id", how="left")
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "date"]).drop_duplicates("race_id")
    X = X[X["date"] <= pd.Timestamp(data.DEV_END)]
    X["y"] = X["date"].dt.year
    M = X.merge(T, on="race_id", how="left")
    print("レースの数", len(X), "本賞金の付く割合（年ごと）", M.groupby("y")["p1"].apply(lambda s: round(float(s.notna().mean()), 3)).to_dict(), flush=True)
    print("比が合うレースの割合", round(float(T["ok"].mean()), 4), "p1 の分布（万円）", T["p1"].describe().round(0).to_dict(), flush=True)
    path = data.OUT / "prize_race.parquet"
    T.to_parquet(path, index=False)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()

"""記述: どこで血統が効くか（開発の行・第 4 版 pure の 2 代母の単位の OOF・勝ちのレース内の対数損失の一様からの改善）。

クラス・レースの年齢条件・芝ダ・距離帯・頭数・父の産駒の数（as-of）で分ける。尤度だけ・回収率は見ない・買い方に使わない。
    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.where_gain
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "where_gain", "どこで血統が効くかの記述（尤度だけ）"))
    O = pd.read_parquet(data.OUT / "kf_v4_pure_dd.parquet")
    O = O[O["part"] == "dev"][["race_id", "horse_id", "p_win"]]
    F = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "win", "n_run", "cls", "rage", "surf", "dband", "sire_all_n"])
    X = O.merge(F, on=["race_id", "horse_id"])
    W = X[X["win"] == 1].copy()
    W["g"] = np.log(W["p_win"]) - np.log(1 / W["n_run"])
    W["field"] = pd.cut(W["n_run"], [0, 10, 14, 18], labels=["〜10 頭", "11〜14 頭", "15〜18 頭"]).astype(str)
    # 父の産駒の数は馬の属性で、勝ち馬の行で分けると結果で条件付けた集計になるので出さない（2 体の R1・critic 2）
    lab = {"cls": {"NEW": "新馬", "MDN": "未勝利", "C1": "1 勝", "C2": "2 勝", "C3": "3 勝", "OP": "OP・重賞"},
           "rage": {"A2": "2 歳戦", "A3": "3 歳戦", "A3U": "3 歳以上", "A4U": "4 歳以上"}, "surf": {"T": "芝", "D": "ダート"},
           "dband": {"S": "〜1300", "M": "1400-1600", "I": "1700-2000", "L": "2100-2400", "E": "2500〜"}}
    out = {"all": {"races": len(W), "gain": round(float(W["g"].mean()), 4)}}
    for col in ("cls", "rage", "surf", "dband", "field"):
        g = W.groupby(col)["g"].agg(["mean", "size"])
        out[col] = {lab.get(col, {}).get(k, k): {"gain": round(float(v["mean"]), 4), "races": int(v["size"])} for k, v in g.iterrows()}
    path = data.OUT / "where_gain.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()

"""tp-62: 騎手・調教師の条件（tp-59 の as-of 判定）を馬ごとの旗にして保存する。

python -m src.scripts.data.tp62_overlay build     # ホストで実行。<TP62_OUT>/overlay.parquet（race_id, horse_number, res2, ex2）
res2: 騎手・調教師の両方が『当年より前の勝ち数 > 市場（単勝オッズ由来）の見込み』（縮約つき）。
ex2:  騎手・調教師の両方が『当年より前の超過払戻（単勝オッズ帯の平均払戻を引いた分）が正』。
当年より前の成績だけで作る（as-of）。2022 年以降の各年に、その年の旗を付ける。回収率由来の統計なので馬券側の判断にだけ使い、特徴量にはしない。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

OUT = Path(os.environ.get("TP62_OUT", "/home/jovyan/work/tp62_out"))


def build() -> None:
    from src.scripts.data import tp59_profile as P
    from src.scripts.data import tp59_rules as R

    d = R.add_asof_conditions(P.load())
    d = d[d["year"] >= 2022].copy()
    d["res2"] = ((d["j_res_pos"] == 1) & (d["t_res_pos"] == 1)).astype("int8")
    d["ex2"] = ((d["j_ex_pos"] == 1) & (d["t_ex_pos"] == 1)).astype("int8")
    o = d[["race_id", "horse_number", "res2", "ex2"]].copy()
    o["horse_number"] = o["horse_number"].astype("int16")
    OUT.mkdir(parents=True, exist_ok=True)
    o.to_parquet(OUT / "overlay.parquet", index=False)
    print(o.shape, o[["res2", "ex2"]].mean().round(3).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.parse_args()
    build()

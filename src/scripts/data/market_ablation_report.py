"""市場（人気）に間接的につながる特徴量を外した学習の比較（調査・分析）。基準 z:RB@e との精度差と、モデルの順位が市場（確定単勝オッズ）の順位とどれだけ一致するか。

python -m src.scripts.data.market_ablation_report   # market_ablation.md
外した特徴量: MKT_NK（netkeiba 指数の履歴）、MKT_JK（騎手・調教師の成績など）、MKT_ALL（両方）。一致の指標: レース内のスピアマン相関、完全に同じ順位のレースの割合、モデル1位=1番人気の割合。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import odds_band_ev as ob
from src.scripts.data import v6_eval as ev

OUT = Path(os.environ.get("PTRAIN_OUT", "/home/jovyan/work/ptrain_out"))
NAMES = {"基準": "z:RB@e", "netkeiba指数を外す": "z:RB+-MKT_NK@e", "騎手・調教師を外す": "z:RB+-MKT_JK@e", "両方を外す": "z:RB+-MKT_ALL@e"}
KEY = ["race_id", "horse_id"]


def main() -> None:
    d0, _ = ev._paths("")
    lv = json.loads((d0 / "levels.json").read_text(encoding="utf-8"))
    mk = ob.load([2022, 2023, 2024])
    raw = pd.concat([pd.read_parquet(ob.TABLES / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_number", "horse_id"]) for y in (2022, 2023, 2024)])
    raw["race_id"], raw["horse_id"] = raw["race_id"].astype(str), raw["horse_id"].astype(str)
    mk = mk.merge(raw, on=["race_id", "horse_number"])[["race_id", "horse_id", "odds", "finish_position"]]
    base = lv["z:RB@e"]["levels"]
    L = ["| 構成 | 勝率 LogLoss | 連対 | 複勝 | 基準との平均差 | 市場順位とのスピアマン | 完全に同じ順位 | モデル1位=1番人気 |", "|---|---|---|---|---|---|---|---|"]
    for label, n in NAMES.items():
        f = d0 / ev._oof_file(n)
        if not f.exists() or n not in lv:
            L.append(f"| {label} | 未完了 | | | | | | |")
            continue
        o = pd.read_parquet(f)
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        d = o.merge(mk, on=KEY)
        d["a"] = d.groupby("race_id")["p_win"].rank(ascending=False, method="first")
        d["b"] = d.groupby("race_id")["odds"].rank(method="first")
        sp = float(d.groupby("race_id").apply(lambda g: g["a"].corr(g["b"], method="spearman")).mean())
        same = float(d.groupby("race_id").apply(lambda g: (g["a"] == g["b"]).all()).mean())
        top1 = float(d.groupby("race_id").apply(lambda g: g.loc[g["a"] == 1, "b"].iloc[0]).eq(1).mean())
        l = lv[n]["levels"]
        dd = [l[k] - base[k] for k in ("logloss_win", "logloss_top2", "logloss_top3")]
        L.append(f"| {label} | {l['logloss_win']:.5f} | {l['logloss_top2']:.5f} | {l['logloss_top3']:.5f} | {np.mean(dd):+.5f} | {sp:.3f} | {same:.4f} | {top1:.3f} |")
    (OUT / "market_ablation.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

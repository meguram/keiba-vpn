"""父・母父 × 条件（芝ダ・距離帯・馬場状態・競馬場）の産駒成績を、そのレースより過去のデータだけで集計する特徴量。

目的: 経験の浅い馬（新馬など）でも、血統から「この条件が合うか」「どんな脚質・末脚の傾向か」を推定できるようにする。
各特徴量は、同じキーの過去の行（日付が厳密に前）の平均を、as-of の全体平均へ経験ベイズで縮約したもの。件数（_n）も付ける。
- 値: 勝率（is_win）、複勝率（is_top3）、脚質の傾向（pos_early: 0=先頭〜1=最後尾）、上がりの速さ（l3f_dev: レース内平均との差の秒。小さいほど速い）
- キー: 父/母父 × (芝ダ × 距離帯)、(芝ダ × 馬場状態)、(芝ダ × 競馬場)
当日の馬場傾向（先行有利・差し有利）との組み合わせは、木モデル／NN が既存の track_bias_* 列と合わせて学習する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.models.stage_models import asof_group_mean

CONDS = {
    "sd": lambda d: d["surface"].astype(str) + "|" + d["dist_band"].astype(str),
    "go": lambda d: d["surface"].astype(str) + "|" + np.where(d["track_condition"].astype(str).isin(["良"]), "good", "heavy"),
    "vn": lambda d: d["surface"].astype(str) + "|" + d["venue_code"].astype(str),
}
VALUES = {"win": "is_win", "t3": "is_top3", "pe": "pos_early", "l3": "l3f_dev"}


def dist_band(distance: pd.Series) -> pd.Series:
    return pd.cut(distance.astype(float), [0, 1300, 1500, 1900, 2400, 99999], labels=False).astype("float").fillna(-1).astype(int)


def build(df: pd.DataFrame, labels: pd.DataFrame, k: float = 30.0) -> pd.DataFrame:
    """df: date, race_id, horse_id, sire, dam_sire, surface, distance, track_condition, venue_code, finish_position（RangeIndex）。
    labels: stage_labels の race_id, horse_id, pos_early, l3f（レース内平均との差は ここで作る）。"""
    assert isinstance(df.index, pd.RangeIndex)
    d = df[["date", "race_id", "horse_id", "sire", "dam_sire", "surface", "track_condition", "venue_code"]].copy()
    d["dist_band"] = dist_band(df["distance"])
    fin = df["finish_position"].astype(float)
    d["is_win"] = (fin == 1).astype(float)
    d["is_top3"] = (fin <= 3).astype(float)
    lab = labels.drop_duplicates(["race_id", "horse_id"])[["race_id", "horse_id", "pos_early", "l3f"]]
    d = d.merge(lab, on=["race_id", "horse_id"], how="left")
    d["l3f_dev"] = d["l3f"] - d.groupby("race_id")["l3f"].transform("mean")
    out = {}
    for role in ("sire", "dam_sire"):
        for cname, fn in CONDS.items():
            key = d[role].astype(str) + "|" + fn(d)
            tmp = pd.DataFrame({"date": d["date"].to_numpy(), "k": key.to_numpy()})
            for vname, vcol in VALUES.items():
                if role == "dam_sire" and vname in ("pe", "l3"):
                    continue                       # 母父は勝率・複勝率のみ（脚質・末脚は父で十分）
                tmp["v"] = d[vcol].to_numpy()
                r = asof_group_mean(tmp, "k", "v", k=k, name="x")
                out[f"ped_{role}_{cname}_{vname}"] = r["x_mean"].to_numpy()
                if vname == "win":
                    out[f"ped_{role}_{cname}_n"] = r["x_n"].to_numpy()
    return pd.DataFrame(out, index=df.index).astype("float32")

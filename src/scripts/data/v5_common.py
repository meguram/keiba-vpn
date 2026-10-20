"""v5 の特徴量セット（V4 の列 + tl_ 再取得版 + 採用された追加群）を、tp-27 / fit_final_v4 の組み立てに差し込む共通部品。

環境変数 V5_GROUPS（例 "CY" / "CY,BEST" / "CY,BEST,RLC"）で追加する群を指定。tl_ は常に再取得版（tp35/cand_TL_A_v2.parquet）にする。
NN には追加列を重要度上位に絞って渡す設計（tp-27）なので、新しい群は木・線形のメンバー（追加列を全部渡す側）にだけ足す。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

KEY = ["race_id", "horse_id"]
V = Path(__file__).resolve().parents[3] / "data/local/modeling/training_pipeline_v4"
V5_TP27 = V / "tp27_v5"


def groups() -> list[str]:
    return [g for g in os.environ.get("V5_GROUPS", "").split(",") if g]


def fresh_tlx(df: pd.DataFrame, old_tlx: pd.DataFrame) -> pd.DataFrame:
    """旧 TLX と同じ列名（tl_ + 列名）で、中身を再取得版の TL_A（tp35/cand_TL_A_v2）にする。"""
    new = pd.read_parquet(V / "tp35/cand_TL_A_v2.parquet").drop_duplicates(KEY)
    new["race_id"], new["horse_id"] = new["race_id"].astype(str), new["horse_id"].astype(str)
    m = df[KEY].merge(new, on=KEY, how="left")
    m.index = df.index
    out = old_tlx.copy()
    for c in out.columns:
        base = c[3:]
        if base in m.columns:
            out[c] = m[base].to_numpy()
    return out


def group_frame(df: pd.DataFrame) -> pd.DataFrame:
    from src.scripts.data import v5_select_eval as sel

    gs = groups()
    if not gs:
        return pd.DataFrame(index=df.index)
    G = pd.concat([sel.load_group(g, df) for g in gs], axis=1)
    return G.loc[:, ~G.columns.duplicated()]


def patch() -> None:
    """tp25e_eval.load_v3（TLX を再取得版に）と tp27_eval._extra_for_ensemble（木・線形の追加列に V5_GROUPS を足す）を差し替える。"""
    from src.scripts.data import tp25e_eval as e
    from src.scripts.data import tp27_eval as t27

    orig_load = e.load_v3
    orig_extra = t27._extra_for_ensemble

    def load_v3():
        df, meta, folds, members, TLX = orig_load()
        return df, meta, folds, members, fresh_tlx(df, TLX)

    def extra(df, variant="core", top=None):
        E = orig_extra(df, variant, top)
        if top:                                   # NN 用（重要度上位に絞る）には足さない
            return E
        G = group_frame(df)
        dup = [c for c in G.columns if c in E.columns]
        return pd.concat([E, G.drop(columns=dup)], axis=1)

    e.load_v3 = load_v3
    t27._extra_for_ensemble = extra

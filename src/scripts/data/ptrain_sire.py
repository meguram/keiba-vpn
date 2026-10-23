"""tp-77 追加3: 種牡馬・父系統ごとの『勝利距離』の統計量（血統予想家向け。すべてレース開催日より前の産駒の結果だけから作る）。

python -m src.scripts.data.ptrain_sire build   # PTRAIN_OUT/sire_rows.parquet（race_id, horse_id キー）
キー: 父・母父・父の系統（○○系、sub_group）・父の大系統（main_group）・母父の系統。面（芝/ダート）ごとに分けて集計。
列（接頭辞 P77_S_<キー>_）: wd=勝利距離の平均（縮約）, wsd=幅, wn=勝利数, gap=今回の距離−平均, z=gap/幅, t3d=3着内の距離の平均（縮約）, t3gap。
市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features.horse_history.common import FIT_END

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain"))
MINING = Path(os.environ.get("PTRAIN_MINING", "/home/jovyan/work/tp72_out/mining.parquet"))
LINEAGE = Path(os.environ.get("PTRAIN_LINEAGE", ROOT / "data/page_reference/pedigree_race_index/stallion_lineage.parquet"))
KEY = ["race_id", "horse_id"]
K_WIN, K_T3 = 5.0, 10.0
KEYS = {"sire": "sire", "dsire": "dam_sire", "sline": "sire_sub", "smain": "sire_main", "dline": "dam_sire_sub"}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_lines(df: pd.DataFrame, lin: pd.DataFrame) -> pd.DataFrame:
    sub = dict(zip(lin["stallion_id"].astype(str), lin["sub_group_label"]))
    main = dict(zip(lin["stallion_id"].astype(str), lin["main_group_name"]))
    df = df.copy()
    df["sire_sub"] = df["sire"].astype(str).map(sub).fillna("不明")
    df["sire_main"] = df["sire"].astype(str).map(main).fillna("不明")
    df["dam_sire_sub"] = df["dam_sire"].astype(str).map(sub).fillna("不明")
    return df


def asof_dist_stats(df: pd.DataFrame, col: str, pre: str, prior: pd.DataFrame) -> pd.DataFrame:
    """df: race_id, horse_id, date, distance, surface, fp と col（キー）。col×面の、より前の開催日までの産駒の結果から作る。"""
    d = df.assign(
        win=(df["fp"] == 1).astype(float), t3=(df["fp"] <= 3).astype(float),
        wd=np.where(df["fp"] == 1, df["distance"], 0.0), wd2=np.where(df["fp"] == 1, df["distance"] ** 2, 0.0),
        td=np.where(df["fp"] <= 3, df["distance"], 0.0),
    )
    agg = d.groupby([col, "surface", "date"], sort=True)[["win", "t3", "wd", "wd2", "td"]].sum().reset_index()
    for c in ("win", "t3", "wd", "wd2", "td"):
        agg[c] = agg.groupby([col, "surface"])[c].cumsum() - agg[c]
    m = df[KEY + [col, "surface", "date", "distance"]].merge(agg, on=[col, "surface", "date"], how="left").merge(prior, on="surface", how="left")
    wn, t3 = m["win"].fillna(0), m["t3"].fillna(0)
    mean = (m["wd"].fillna(0) + K_WIN * m["mu_w"]) / (wn + K_WIN)
    var = np.maximum((m["wd2"].fillna(0) + K_WIN * (m["sd_w"] ** 2 + m["mu_w"] ** 2)) / (wn + K_WIN) - mean**2, 1.0)
    t3d = (m["td"].fillna(0) + K_T3 * m["mu_t"]) / (t3 + K_T3)
    o = pd.DataFrame({f"{pre}_wd": mean, f"{pre}_wsd": np.sqrt(var), f"{pre}_wn": wn})
    o[f"{pre}_gap"] = m["distance"] - mean
    o[f"{pre}_z"] = o[f"{pre}_gap"] / o[f"{pre}_wsd"]
    o[f"{pre}_t3d"] = t3d
    o[f"{pre}_t3gap"] = m["distance"] - t3d
    o.index = df.index
    return o.astype("float32")


def make_prior(df: pd.DataFrame) -> pd.DataFrame:
    f = df[df["date"] <= pd.Timestamp(FIT_END)]
    w, t = f[f["fp"] == 1], f[f["fp"] <= 3]
    return pd.concat([w.groupby("surface")["distance"].agg(mu_w="mean", sd_w="std"), t.groupby("surface")["distance"].mean().rename("mu_t")], axis=1).reset_index()


def build_frame(df: pd.DataFrame, lin: pd.DataFrame) -> pd.DataFrame:
    df = add_lines(df, lin).reset_index(drop=True)
    prior = make_prior(df)
    parts = [df[KEY]]
    for pre, col in KEYS.items():
        parts.append(asof_dist_stats(df, col, f"P77_S_{pre}", prior))
    return pd.concat(parts, axis=1)


def cmd_build(a) -> None:
    m = pd.read_parquet(MINING, columns=KEY + ["date", "distance", "surface", "finish_position", "sire", "dam_sire"])
    m["race_id"], m["horse_id"] = m["race_id"].astype(str), m["horse_id"].astype(str)
    m["date"] = pd.to_datetime(m["date"])
    m["fp"] = pd.to_numeric(m["finish_position"], errors="coerce")
    m["distance"] = pd.to_numeric(m["distance"], errors="coerce")
    m = m.dropna(subset=["fp", "distance"]).drop(columns=["finish_position"])
    m = m[m["surface"].isin(["芝", "ダート"])]
    m["sire"], m["dam_sire"] = m["sire"].astype(str), m["dam_sire"].astype(str)
    out = build_frame(m, pd.read_parquet(LINEAGE))
    out.to_parquet(OUT / "sire_rows.parquet", index=False)
    cols = [c for c in out.columns if c not in KEY]
    (OUT / "sire_rows.meta.json").write_text(json.dumps({"rows": len(out), "cols": cols, "coverage": {c: round(float(out[c].notna().mean()), 3) for c in cols}}, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"sire_rows: {out.shape}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build").set_defaults(fn=cmd_build)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

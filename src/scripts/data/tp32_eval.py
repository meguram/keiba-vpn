"""tp-32: 馬具と『初めて・替わり』系の状況特徴の評価（V4 全体に対する増分。tp-31 の評価基盤を使い回す）。

python -m src.scripts.data.tp32_eval groups   # 12 群を作って tp32/groups/*.parquet に保存（先に tp32_extract_equipment が必要）
python -m src.scripts.data.tp32_eval stage1   # 第 1 段: valid 2022+2023、seed 1 本。V4 全体に対する増分
python -m src.scripts.data.tp32_eval stage2   # 第 2 段: 第 1 段を通った群を valid 2024、新 seed 2 本で確認（通過群の同時追加も）
test 2025・2026 年は使わない。市場オッズ・人気は使わない。tp-31 の v4_frame.parquet（V4 の特徴量表）を symlink で共有する。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V = ROOT / "data/local/modeling/training_pipeline_v4"
OUT = V / "tp32"
os.environ.setdefault("TP31_THREADS", os.environ.get("TP32_THREADS", "6"))
os.environ.setdefault("TP31_LOAD_LIMIT", os.environ.get("TP32_LOAD_LIMIT", "16"))

from src.pipeline.features import factory_self as fs  # noqa: E402
from src.pipeline.features import tp32_equipment as te  # noqa: E402
from src.scripts.data import tp31_eval as T31  # noqa: E402
from src.scripts.data import tp27_eval as t  # noqa: E402

KEY = ["race_id", "horse_id"]
TABLES = ROOT / "data/page_reference/tables"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs():
    runs = fs.load_runs(TABLES, list(range(2020, 2027)))
    c = pd.read_parquet(OUT / "career_eq.parquet", columns=["horse_id", "race_id", "rdate", "blinker", "rank", "entry"])
    c = c[c["race_id"].astype(str).str.len() == 12]
    r = pd.read_parquet(OUT / "runners_eq.parquet", columns=["horse_id", "race_id", "date", "blinker", "tozai"])
    # 出走表の申告値（取得時点）と確定戦績の値が一致していれば戦績を使う。戦績にまだ無い（直近の）レースだけ出走表の値で補う
    miss = r.merge(c[["horse_id", "race_id"]], on=["horse_id", "race_id"], how="left", indicator=True)
    miss = miss[miss["_merge"] == "left_only"]
    extra = pd.DataFrame({"horse_id": miss["horse_id"], "race_id": miss["race_id"], "rdate": miss["date"].astype(str), "blinker": miss["blinker"], "rank": np.nan, "entry": np.nan})
    E = pd.concat([c, extra], ignore_index=True)
    toz = r.dropna(subset=["tozai"]).groupby("horse_id")["tozai"].agg(lambda s: s.astype(str).mode().iloc[0])
    return runs, E, toz


def home_from_tozai(runs: pd.DataFrame, toz: pd.Series) -> pd.Series:
    """tozai のコード → 東/西。そのコードの馬が西の開催場（中京・京都・阪神・小倉）で走る割合の大きいほうを西とする。"""
    v = runs["venue_code"].astype(str).str.zfill(2)
    w = v.isin(te.WEST).astype(float).groupby(runs["horse_id"].astype(str).map(toz)).mean()
    codes = w.dropna().sort_values()
    if len(codes) < 2:
        return pd.Series(dtype=object)
    mp = {codes.index[0]: "E", codes.index[-1]: "W"}
    _log(f"tozai コードの対応: {mp}（西の開催場で走る割合 {codes.round(3).to_dict()}）")
    return toz.map(mp)


def cmd_groups(a) -> None:
    df, base, meta, folds, lp = t.setup()
    runs, E, toz = load_inputs()
    home = home_from_tozai(runs, toz)
    parts = te.build_groups(runs, E=E, home=home)
    gdir = OUT / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    for g, G in parts.items():
        G["race_id"], G["horse_id"] = G["race_id"].astype(str), G["horse_id"].astype(str)
        G = G.drop_duplicates(KEY)
        T = df[KEY].astype(str).merge(G, on=KEY, how="left")
        T.to_parquet(gdir / f"{g}.parquet", index=False)
        cols = [c for c in T.columns if c not in KEY]
        _log(f"{g}: {len(cols)} 列、欠損率 {float(T[cols].isna().mean().mean()):.3f}")


def _setup_tp31():
    OUT.mkdir(parents=True, exist_ok=True)
    T31.OUT = OUT
    link = OUT / "v4_frame.parquet"
    if not link.exists():
        link.symlink_to(V / "tp31" / "v4_frame.parquet")


def cmd_stage1(a) -> None:
    _setup_tp31()
    T31.GROUP_NAMES = te.GROUPS
    T31.cmd_stage1(a)


def cmd_stage2(a) -> None:
    _setup_tp31()
    T31.cmd_stage2(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["groups", "stage1", "stage2"])
    a = ap.parse_args()
    {"groups": cmd_groups, "stage1": cmd_stage1, "stage2": cmd_stage2}[a.cmd](a)


if __name__ == "__main__":
    main()

"""最終層（確率の較正・期待値の較正・印・馬券）用の frame_all を、新しい親モデル（v6e16・2016〜2025 学習）の確率で作り直す。ホストで実行。

    python -m src.scripts.data.final_layer_frame --out /home/jovyan/work/retrain_out/base_2016/final_layer/frame_all.parquet

確率 p_win/p_top2/p_top3 の出どころ（どちらも『その年を学習に使っていない』予測）:
- 2022〜2024 年: tp27_v6e/ens/v6.parquet（leave-one-year-out のアンサンブル OOF）
- 2025 年: final_eval_v6e16/test2025_predictions.parquet（2016〜2024 で学習した最終モデルの test 予測）
オッズ・払戻・人気・着順の列は旧 frame_all（tp36）のものをそのまま使う（確定値で、親モデルに依らない）。
2026 年はこの時点では作らない（bundle_v6_2025 で別途。評価専用で、選択に使わない）。p_blend は旧モデル由来なので落とす。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY = ["race_id", "horse_id"]
PROBS = ["p_win", "p_top2", "p_top3"]
BASE = Path("/home/jovyan/work/retrain_out/base_2016")
OLD_FRAME = Path("data/local/modeling/betting/tp36/frame_all.parquet")


def new_probabilities(base: Path = BASE) -> pd.DataFrame:
    a = pd.read_parquet(base / "tp27_v6e/ens/v6.parquet")
    a = a[a["year"].between(2022, 2024)][KEY + PROBS]
    b = pd.read_parquet(base / "final_eval_v6e16/test2025_predictions.parquet")[KEY + PROBS]
    p = pd.concat([a, b], ignore_index=True)
    for c in KEY:
        p[c] = p[c].astype(str)
    dup = int(p.duplicated(KEY).sum())
    assert dup == 0, f"確率が重複している: {dup} 行"
    return p


def build(old_frame: Path = OLD_FRAME, base: Path = BASE) -> tuple[pd.DataFrame, dict]:
    old = pd.read_parquet(old_frame)
    old["race_id"], old["horse_id"] = old["race_id"].astype(str), old["horse_id"].astype(str)
    old = old[old["year"].between(2022, 2025)].drop(columns=PROBS + ["p_blend"], errors="ignore")
    new = new_probabilities(base)
    m = old.merge(new, on=KEY, how="inner")
    info = {
        "old_rows": int(len(old)),
        "rows": int(len(m)),
        "dropped_no_prob": int(len(old) - len(m)),
        "by_year": {int(k): int(v) for k, v in m.groupby("year").size().items()},
        "prob_na": {c: int(m[c].isna().sum()) for c in PROBS},
    }
    return m, info


def write_rank_tables(rank_dir: Path) -> None:
    """tp70_lib.load_base が読む rank{年}.parquet（確定着順）を、2016 年版の train_table から作る。"""
    from src.scripts.data import rebuild_groups_2016 as rg

    rg.redirect()
    t = pd.read_parquet(rg.BASE / "features/train_table.parquet", columns=KEY + ["date", "finish_position"])
    t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
    t["year"] = pd.to_datetime(t["date"]).dt.year
    t = t.rename(columns={"finish_position": "rank"})
    rank_dir.mkdir(parents=True, exist_ok=True)
    for y in range(2022, 2027):
        t[t["year"] == y][KEY + ["rank"]].to_parquet(rank_dir / f"rank{y}.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank-dir", default=None, help="指定すると rank{年}.parquet も作る（REBUILD_2016 の土台が要る）")
    ap.add_argument("--old-frame", default=str(OLD_FRAME))
    a = ap.parse_args()
    if a.rank_dir:
        write_rank_tables(Path(a.rank_dir))
    m, info = build(Path(a.old_frame))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out, index=False)
    print(info, flush=True)


if __name__ == "__main__":
    main()

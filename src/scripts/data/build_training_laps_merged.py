"""追い切り表の重ね合わせ: 固定の表（tp35/training_laps_fresh.parquet、10/19 作成。2026-05 以降がほぼ無い）に、
手元の調教の記録（data/local/horse_training/）から作った行を重ね、重複を除く。ネットワーク・GCS は使わない（費用 0 円）。

python -m src.scripts.data.build_training_laps_merged              # 既定の出力: tp35/training_laps_merged.parquet（固定の表は上書きしない）
python -m src.scripts.data.build_training_laps_merged --out OUT.parquet

- 馬 × 対象のレースの日の組が手元の記録にあれば、その組は手元の行だけを使う（新しく取った版。時計の無い種別も時計の違いも、組ごと置き換える）。
  手元に無い組は固定の表の行をそのまま残す（固定の表の中の重複にも手を付けない）。
- 手元の記録だけでは 2025 年の被覆が下がる（出走のうち手元に記録ファイルがある割合は、月ごとに 2025-01 の 45% から 2025-12 の 80%）。
  固定の表は捨てない（docs/operations/data_intake_20261124.md）。
- 注意: どちらの行も、多くは発走後に取った記録。発走前の調教ページには 1・2 週前の追い切りがほとんど載らない（同じ docs の 2 章）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config.data_paths import LOCAL, MODELING_DIR
from src.pipeline.features.training_laps import add_lap_features, record_to_rows

FRESH = MODELING_DIR / "training_pipeline_v4/tp35/training_laps_fresh.parquet"
OUT = MODELING_DIR / "training_pipeline_v4/tp35/training_laps_merged.parquet"
LOCAL_DIR = LOCAL / "horse_training"
GROUP = ["horse_id", "race_date"]


def local_laps(root: Path = LOCAL_DIR) -> tuple[pd.DataFrame, int]:
    """手元の記録の全馬分の追い切り（1 調教 1 行）。読めないファイルは数えて飛ばす。"""
    rows: list[dict] = []
    bad = 0
    for f in sorted(root.glob("*/*.json")):
        try:
            rows += record_to_rows(json.loads(f.read_text(encoding="utf-8")))
        except Exception:                             # noqa: BLE001  読めない・形の違う記録は数えて飛ばす
            bad += 1
    if not rows:
        return pd.DataFrame(), bad
    return add_lap_features(pd.DataFrame(rows)), bad


def merge(fresh: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    """固定の表と手元の行を重ねる。馬 × レースの日の組が手元にあれば手元の行だけ、無ければ固定の表の行。列は固定の表に合わせる。"""
    cols = list(fresh.columns)
    fr = fresh.copy()
    loc = local.reindex(columns=cols)
    for d in (fr, loc):
        for c in ("horse_id", "train_date", "race_date"):
            d[c] = d[c].astype(str)
    have = pd.MultiIndex.from_frame(loc[GROUP].drop_duplicates())
    keep = ~pd.MultiIndex.from_frame(fr[GROUP]).isin(have)
    both = pd.concat([fr[keep], loc], ignore_index=True)
    return both.sort_values(["horse_id", "race_date", "train_date"], kind="stable").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    if out.resolve() == FRESH.resolve():
        raise SystemExit("固定の表（training_laps_fresh.parquet）は上書きしない")
    fresh = pd.read_parquet(FRESH)
    local, bad = local_laps()
    m = merge(fresh, local)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out, index=False)
    print(f"固定の表 {len(fresh):,} 本 + 手元 {len(local):,} 本（読めないファイル {bad}）→ 重ねた表 {len(m):,} 本 -> {out}")


if __name__ == "__main__":
    main()

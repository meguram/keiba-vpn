"""めぐコラム用: 2016〜2019 年のペース区分（pace_cls・shape_cls など）を作る。

race_quality.parquet（tp-61 の分類）は 2020 年からしか無い。コラムの集計期間を 2016 年からに広げるため、
2016〜2019 年を tp61_build と同じ手順でラップ表から作り、2020〜2023 年で固定した閾値（thresholds.json）で分類して、
data/local/modeling/shared_cache/race_quality/race_cls_2016_2019.parquet に保存する。
compare_series.read_race_quality() が、race_quality.parquet とこの表をつなぐ。

本番モデルが使う race_quality.parquet・tp61 の出力は書き換えない（別の表に出す）。
確認: 同じ手順で 2020 年以降を作り直すと、既存の race_cls.parquet と pace_cls・shape_cls・bal が完全に一致する（2020〜2026 年 22,830 レース）。

使い方:
    python -m src.research.columns.pace_old_years
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.scripts.data import tp61_analysis as A
from src.scripts.data import tp61_build as B

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/local/modeling/shared_cache/race_quality/race_cls_2016_2019.parquet"
OLD_YEARS = (2016, 2017, 2018, 2019)


def build() -> pd.DataFrame:
    rs = [B.build_year(y)[1] for y in OLD_YEARS]
    old = pd.concat(rs, ignore_index=True)
    cur = pd.read_parquet(A.D / "race.parquet")                   # 2020 年以降（基準の中央値は 2020〜2023 年から取る）
    allr = pd.concat([old, cur], ignore_index=True)
    allr["band"] = allr["distance"].map(A.dist_band)
    allr["k1"] = allr["venue"].astype(str) + "|" + allr["surface"].astype(str) + "|" + allr["distance"].astype(int).astype(str) + "|" + allr["going_group"]
    allr["k2"] = allr["surface"].astype(str) + "|" + allr["distance"].astype(int).astype(str) + "|" + allr["going_group"]
    allr["k3"] = allr["surface"].astype(str) + "|" + allr["band"] + "|" + allr["going_group"]
    allr = A.add_dev(allr)
    th = json.loads((A.D / "thresholds.json").read_text(encoding="utf-8"))   # 閾値は 2020〜2023 年のまま固定
    allr = A.classify(allr, th)
    return allr[allr["year"].isin(OLD_YEARS)]


def main() -> None:
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT, index=False)
    print(f"{OUT}: {len(d):,} レース", d["pace_cls"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()

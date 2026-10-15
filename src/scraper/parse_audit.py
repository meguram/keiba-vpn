"""エクスポート済みの race_result_flat から、パース失敗の兆候（必須項目の欠損）を検査する。

ページ表記の変更でパーサが黙って空の値を返す事故（2026 年の「芝右 外1200m」で距離 0・面が空）を、取得の翌日までに見つけるための検査。
  python -m src.scraper.parse_audit --years 2026        # 欠損があれば終了コード 1
障害レースは race_class も方向も空になるのが通常。race_class だけが空のレースは文字化け等の兆候として数える。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 必須（空・0 があればパース失敗の疑い）。しきい値は「許容するレース数」
TOLERANCE = {"race_class(方向あり)": 2}
REQUIRED = {"surface": 0, "distance": 0, "weather": 0, "track_condition": 0, "start_time": 0, "field_size": 0, "date": 0, "venue": 0}


def audit_frame(df: pd.DataFrame) -> dict:
    """レース単位（1 レース 1 行）に畳んで、項目ごとの欠損レース数を返す。"""
    g = df.groupby("race_id").first()
    out: dict = {"races": int(len(g)), "missing": {}, "samples": {}}
    for col in REQUIRED:
        if col not in g.columns:
            continue
        s = g[col]
        bad = s.isna() | (s == "") | (s == 0)
        n = int(bad.sum())
        if n:
            out["missing"][col] = n
            out["samples"][col] = [str(i) for i in g.index[bad][:5]]
    if "race_class" in g.columns and "direction" in g.columns:
        # 障害レースは race_class も方向も空。race_class だけが空で方向がある平地レースは、出馬表の取得時の文字化け等の兆候
        # （2026 年に文字コードの誤判定で約 1,500 レース）。旧データにも 1 年 1 レース程度の例外がある
        rc = g[(g["race_class"].isna() | (g["race_class"] == "")) & ~(g["direction"].isna() | (g["direction"] == ""))]
        if len(rc):
            out["missing"]["race_class(方向あり)"] = int(len(rc))
            out["samples"]["race_class(方向あり)"] = [str(i) for i in rc.index[:5]]
    return out


def problems(report: dict) -> dict:
    return {c: n for c, n in report["missing"].items() if n > REQUIRED.get(c, TOLERANCE.get(c, 0))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", required=True)
    ap.add_argument("--tables-dir", type=Path, default=Path("data/page_reference/tables"))
    args = ap.parse_args()
    bad_total = 0
    for y in [s.strip() for s in args.years.split(",")]:
        p = args.tables_dir / y / "race_result_flat.parquet"
        if not p.is_file():
            print(f"[{y}] race_result_flat がありません")
            continue
        cols = ["race_id", *REQUIRED, "direction", "race_class"]
        df = pd.read_parquet(p, columns=[c for c in cols if c in _schema_names(p)])
        rep = audit_frame(df)
        bad = problems(rep)
        print(f"[{y}] races={rep['races']} 欠損={rep['missing'] or 'なし'}")
        for c, ids in rep["samples"].items():
            print(f"    {c}: 例 {ids}")
        bad_total += sum(bad.values())
    return 1 if bad_total else 0


def _schema_names(p: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.read_schema(p).names)


if __name__ == "__main__":
    sys.exit(main())

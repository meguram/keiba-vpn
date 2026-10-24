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
    _audit_columns_misread(df, out)
    return out


def _audit_columns_misread(df: pd.DataFrame, out: dict) -> None:
    """列ずれの検査。結果ページに指数列が入った 25 列表を固定位置で読み、odds が追走指数・人気が上がり指数・馬体重が通過順になっていた
    事故（2025-05〜12 の約 2,100R）を、取得の翌日までに見つける。scraper.odds_fix の補正表で直るレースは数えない。
    - odds_sum: レース内の Σ(1/odds) が払戻率から決まる範囲（約 1.0〜1.6、通常 1.26）にない
    - popularity_range: 人気が 1〜頭数の範囲にない
    - weight_range: 馬体重が 300〜700kg にない（着順のあるレースのみ）"""
    if not {"race_id", "odds"} <= set(df.columns):
        return
    try:
        from src.scraper import odds_fix

        covered = odds_fix.fixed_race_ids()
    except Exception:
        covered, odds_fix = set(), None
    d = df[~df["race_id"].astype(str).isin(covered)].copy()
    if d.empty:
        return

    def put(name: str, ids) -> None:
        ids = sorted(set(map(str, ids)))
        if ids:
            out["missing"][name] = len(ids)
            out["samples"][name] = ids[:5]

    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    if odds_fix is not None:
        ok = d.groupby("race_id")["odds"].apply(lambda x: odds_fix.is_plausible_win_odds(x.tolist()))
        put("odds_sum", ok.index[~ok])
    if "popularity" in d.columns and "field_size" in d.columns:
        pop, fs = pd.to_numeric(d["popularity"], errors="coerce"), pd.to_numeric(d["field_size"], errors="coerce")
        put("popularity_range", d["race_id"][(pop > fs) | (pop < 1)])
    if "weight" in d.columns:
        w = pd.to_numeric(d["weight"], errors="coerce")
        fp = pd.to_numeric(d["finish_position"], errors="coerce") if "finish_position" in d.columns else pd.Series(1, index=d.index)
        put("weight_range", d["race_id"][(fp > 0) & ((w < 300) | (w > 700))])


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
        cols = ["race_id", *REQUIRED, "direction", "race_class", "horse_number", "odds", "popularity", "weight", "finish_position"]
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

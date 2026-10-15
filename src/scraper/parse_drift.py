"""エクスポート済みの flat テーブルすべてについて、列ごとの「空の割合」が基準期間から急に増えていないかを検査する。

ページの表記変更・文字コードの誤判定でパーサが黙って空を返す事故は、項目を個別に知らなくても「ある列の空の割合が急増した」ことで見つかる。
  python -m src.scraper.parse_drift --baseline 2020-2025 --recent 2026 [--months 4]
基準（既定 2020〜2025 年）の列ごとの空の割合と、直近 N か月（既定 4）のレース単位の空の割合を比べ、差が --threshold（既定 0.15）を超えた列を出す。
文字化け（U+FFFD を含む文字列）の割合も同じ基準で検査する。問題があれば終了コード 1。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

SKIP_CATEGORIES_SUFFIX = ("_race.parquet",)   # レース単位の表は flat と重複するため見ない
MIN_RACES = 40            # 月ごとのレース数がこれ未満の月は比べない
MIN_BASELINE_ROWS = 500


def _is_empty(col: str, dtype) -> pl.Expr:
    c = pl.col(col)
    if dtype == pl.Utf8:
        return c.is_null() | (c == "")
    if dtype.is_float():
        return c.is_null() | c.is_nan()
    return c.is_null()


def race_level_empty_rates(df: pl.DataFrame, id_col: str = "race_id") -> pl.DataFrame:
    """レースごとに最初の行を取り、列ごとの「空」「文字化け」フラグを返す（行数でなくレース単位で比べる）。"""
    g = df.group_by(id_col, maintain_order=True).agg(pl.all().first())
    exprs = []
    for col, dtype in g.schema.items():
        if col == id_col:
            continue
        exprs.append(_is_empty(col, dtype).cast(pl.Float64).alias(f"E:{col}"))
        if dtype == pl.Utf8:
            exprs.append(pl.col(col).fill_null("").str.contains("�").cast(pl.Float64).alias(f"M:{col}"))
    return g.select(id_col, *([pl.col("date")] if "date" in g.columns else []), *exprs)


def drift(tables_dir: Path, baseline_years: list[int], recent_years: list[int], months: int = 4, threshold: float = 0.15) -> list[dict]:
    findings: list[dict] = []
    cats = sorted({p.name for y in baseline_years + recent_years for p in (tables_dir / str(y)).glob("*_flat.parquet")})
    for name in cats:
        def load(years):
            parts = []
            for y in years:
                p = tables_dir / str(y) / name
                if p.is_file():
                    df = pl.read_parquet(p)
                    if "race_id" in df.columns:
                        parts.append(df)
            return parts
        base_parts, rec_parts = load(baseline_years), load(recent_years)
        if not base_parts or not rec_parts:
            continue
        common = set.intersection(*[set(d.columns) for d in base_parts + rec_parts])
        keep = [c for c in base_parts[0].columns if c in common]
        base = race_level_empty_rates(pl.concat([d.select(keep).cast({c: d.schema[c] for c in keep}, strict=False) for d in base_parts], how="vertical_relaxed"))
        rec = race_level_empty_rates(pl.concat([d.select(keep) for d in rec_parts], how="vertical_relaxed"))
        if base.height < MIN_BASELINE_ROWS or "date" not in rec.columns:
            continue
        rec = rec.with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 7).alias("ym"))
        yms = sorted(rec["ym"].drop_nulls().unique().to_list())[-months:]
        flags = [c for c in base.columns if c[:2] in ("E:", "M:") and c in rec.columns]
        for col in flags:
            b = float(base[col].mean())
            for ym in yms:
                sub = rec.filter(pl.col("ym") == ym)
                if sub.height < MIN_RACES:
                    continue
                r = float(sub[col].mean())
                if r - b > threshold:
                    findings.append({"table": name, "column": col[2:], "kind": "文字化け" if col[:2] == "M:" else "空", "month": ym,
                                     "baseline": round(b, 3), "recent": round(r, 3), "races": sub.height})
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", type=Path, default=Path("data/page_reference/tables"))
    ap.add_argument("--baseline", default="2020-2025")
    ap.add_argument("--recent", default="2026")
    ap.add_argument("--months", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.15)
    args = ap.parse_args()

    def years(s: str) -> list[int]:
        if "-" in s:
            a, b = s.split("-")
            return list(range(int(a), int(b) + 1))
        return [int(x) for x in s.split(",")]

    f = drift(args.tables_dir, years(args.baseline), years(args.recent), args.months, args.threshold)
    for x in sorted(f, key=lambda d: (d["table"], d["column"], d["month"])):
        print(f"{x['table']:34s} {x['column']:28s} {x['kind']}  {x['month']}  基準 {x['baseline']:.3f} → 直近 {x['recent']:.3f}  ({x['races']} レース)")
    print(f"検出 {len(f)} 件")
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())

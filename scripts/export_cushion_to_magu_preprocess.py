#!/usr/bin/env python3
"""
keiba-vpn の data/jra_baba/cushion_values.json を、
magu-keiba-horse-racing-ai 側の前処理ディレクトリ形式に書き出す。

  chuou/data/preprocessed/others/cushion_data/{YYYY}/cushion_records.{csv,parquet}

ローカル JSON に無い列は欠損（NaN / 空文字）で埋める。
2026 は別パイプラインで自動取得している想定で、既定では上書きしない。

使い方:
  python3 scripts/export_cushion_to_magu_preprocess.py
  python3 scripts/export_cushion_to_magu_preprocess.py --skip-years ""
  MAGU_CUSHION_DATA_ROOT=/path/to/cushion_data python3 scripts/export_cushion_to_magu_preprocess.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# cushion_values.json に由来する列 + ライブ取得側で使う列（無ければ欠損）
CANONICAL_COLUMNS: list[tuple[str, str]] = [
    ("year", "Int64"),
    ("venue_code", "string"),
    ("venue_name", "string"),
    ("kai", "Int64"),
    ("pair_index", "Int64"),
    ("weekday", "string"),
    ("cushion_value", "float64"),
    ("turf_moisture_goal", "float64"),
    ("turf_moisture_4corner", "float64"),
    ("dirt_moisture_goal", "float64"),
    ("dirt_moisture_4corner", "float64"),
    ("is_race_day", "boolean"),
    ("race_day", "Int64"),
    ("date", "string"),
    ("course_position", "string"),
    ("measurement_time", "string"),
    ("scraped_at", "string"),
]


def _default_out_root() -> Path:
    env = os.environ.get("MAGU_CUSHION_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    vpn = Path(__file__).resolve().parents[1]
    # myproject/keiba-vpn と同階層に magu-keiba-horse-racing-ai を想定
    return (
        vpn.parent
        / "magu-keiba-horse-racing-ai"
        / "chuou"
        / "data"
        / "preprocessed"
        / "others"
        / "cushion_data"
    )


def _parse_year_list(s: str) -> set[int]:
    s = (s or "").strip()
    if not s:
        return set()
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _normalize_records(raw: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    names = [c for c, _ in CANONICAL_COLUMNS]
    for r in raw:
        row = {k: r.get(k) for k in names}
        rows.append(row)
    df = pd.DataFrame(rows)
    for col, dtype in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
        if dtype == "Int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dtype == "boolean":
            df[col] = df[col].astype("boolean")
        elif dtype == "string":
            def _to_str(v):
                if v is None:
                    return pd.NA
                try:
                    if pd.isna(v):
                        return pd.NA
                except (ValueError, TypeError):
                    pass
                s = str(v).strip()
                return pd.NA if s == "" else s

            df[col] = df[col].map(_to_str).astype("string")
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="cushion_values.json → magu cushion_data 年別出力")
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="入力 JSON（既定: keiba-vpn/data/jra_baba/cushion_values.json）",
    )
    p.add_argument("--out-root", type=Path, default=None, help="cushion_data ディレクトリ")
    p.add_argument(
        "--skip-years",
        type=str,
        default="2026",
        help="書き出さない年（カンマ区切り）。空文字ですべて出力。",
    )
    p.add_argument("--only-years", type=str, default="", help="指定時はその年のみ（カンマ区切り）")
    p.add_argument("--no-parquet", action="store_true", help="CSV のみ（parquet 用 pyarrow なし時）")
    args = p.parse_args()

    vpn = Path(__file__).resolve().parents[1]
    src = args.json or (vpn / "data" / "jra_baba" / "cushion_values.json")
    if not src.is_file():
        print(f"error: 入力がありません: {src}", file=sys.stderr)
        return 1

    out_root = (args.out_root or _default_out_root()).resolve()
    skip_years = _parse_year_list(args.skip_years)
    only_years = _parse_year_list(args.only_years)

    raw = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print("error: JSON は配列である必要があります", file=sys.stderr)
        return 1

    df = _normalize_records(raw)
    if df.empty:
        print("warning: レコード0件")
        return 0

    years = sorted(df["year"].dropna().unique().astype(int).tolist())
    written = 0
    for y in years:
        if only_years and y not in only_years:
            continue
        if y in skip_years:
            print(f"skip year {y} (--skip-years)")
            continue
        sub = df[df["year"] == y].copy()
        ydir = out_root / str(y)
        ydir.mkdir(parents=True, exist_ok=True)
        csv_path = ydir / "cushion_records.csv"
        sub.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"wrote {csv_path} ({len(sub)} rows)")
        if not args.no_parquet:
            try:
                pq = ydir / "cushion_records.parquet"
                sub.to_parquet(pq, index=False)
                print(f"wrote {pq}")
            except Exception as e:
                print(f"note: parquet スキップ ({e}) — CSV のみ利用可")
        written += 1

    meta = {
        "source": str(src),
        "out_root": str(out_root),
        "canonical_columns": [c for c, _ in CANONICAL_COLUMNS],
        "dtypes": {c: t for c, t in CANONICAL_COLUMNS},
        "note": "ローカル JSON に無い列・セルは欠損。measurement_time/scraped_at は主にライブ取得行のみ。",
    }
    meta_path = out_root / "_export_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {meta_path} ({written} year dirs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

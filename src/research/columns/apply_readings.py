"""めぐコラムの記事 JSON にある騎手の読み（reading）を、読みのマスタで付け直す。

記事の騎手一覧（例: jockey-long-distance の jockey_distance_chart）は、画面で「五十音順」に並べ替えるとき reading を使う。
以前は読みが漢字のまま・名の一部だけ・誤読（木幡＝こわた を こはた など）で、50 音順が崩れていた。
マスタ（data/meta/person_reading/jockey_reading.json。src/utils/person_reading.py）から引き直す。

対象: frontend/content/columns/*.json の中で、"name" と "reading" を両方持つ要素（騎手の一覧）。
並び順（rank）は変えない。画面側が reading で並べ替える。読みが引けない名前は一覧にして終了コード 1 にする。

使い方:
    python -m src.research.columns.apply_readings            # 書き換える
    python -m src.research.columns.apply_readings --dry-run  # 変わる箇所だけ出す
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.utils.person_reading import jockey_reading

REPO = Path(__file__).resolve().parents[3]
COLUMNS_DIR = REPO / "frontend/content/columns"


def _walk(o: Any, changes: list[tuple[str, str, str]], missing: list[str]) -> None:
    if isinstance(o, dict):
        if isinstance(o.get("name"), str) and "reading" in o:
            r = jockey_reading(o["name"])
            if r is None:
                missing.append(o["name"])
            elif r != o["reading"]:
                changes.append((o["name"], o["reading"], r))
                o["reading"] = r
            return
        for v in o.values():
            _walk(v, changes, missing)
    elif isinstance(o, list):
        for v in o:
            _walk(v, changes, missing)


def apply(path: Path, dry_run: bool = False) -> tuple[list[tuple[str, str, str]], list[str]]:
    text = path.read_text(encoding="utf-8")
    doc = json.loads(text)
    changes: list[tuple[str, str, str]] = []
    missing: list[str] = []
    _walk(doc, changes, missing)
    if changes and not dry_run:
        out = json.dumps(doc, ensure_ascii=False, indent=2)
        path.write_text(out + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return changes, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    all_missing: list[str] = []
    for path in sorted(COLUMNS_DIR.glob("*.json")):
        changes, missing = apply(path, args.dry_run)
        if changes or missing:
            print(f"{path.name}: 読みを {len(changes)} 件{'変える' if args.dry_run else '変えた'}"
                  + (f"、引けない名前 {len(missing)} 件: {missing}" if missing else ""))
            for name, old, new in changes:
                print(f"  {name}: {old} -> {new}")
        all_missing += missing
    sys.exit(1 if all_missing else 0)


if __name__ == "__main__":
    main()

"""血統予想家の置き場（手元のファイルだけ。GCS を読まない）。

- 欄: data/local/race_analysis/pedigree_persona/<race_id>.json（ミラーの木 data/local/race_analysis の中。束・目録に入る。
  めぐ分析の store は *.json だけを一覧するので、このフォルダは混ざらない）。
- 生の値: data/local/pedigree_persona/<日付>/raw.parquet と meta.json（ミラーしない）。
- 画面に出すかの目印: data/local/meta/pedigree_persona_show.on（ホストに置く。無ければ欄の visible は false で、影だけ）。
"""
from __future__ import annotations

import json
import os
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

JST = timezone(timedelta(hours=9))
REPO = Path(__file__).resolve().parents[3]
SHOW_MARK = REPO / "data/local/meta/pedigree_persona_show.on"


def module_dir() -> Path:
    base = os.environ.get("KEIBA_RACE_ANALYSIS_DIR") or str(REPO / "data/local/race_analysis")
    return Path(base) / "pedigree_persona"


def raw_dir(day: str) -> Path:
    return Path(os.environ.get("KEIBA_PEDIGREE_PERSONA_DIR") or REPO / "data/local/pedigree_persona") / day


def visible() -> bool:
    v = os.environ.get("KEIBA_PEDIGREE_PERSONA_SHOW", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return v in ("1", "true", "yes", "on") or SHOW_MARK.exists()


def read(race_id: str) -> dict[str, Any] | None:
    try:
        return json.loads((module_dir() / f"{race_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write(race_id: str, doc: dict[str, Any]) -> Path:
    out = module_dir() / f"{race_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)
    return out

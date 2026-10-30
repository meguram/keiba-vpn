"""レースごとの分析結果の置き場。分析は事前に計算してここへ保存し、API は読むだけにする（表示のたびに計算しない）。

data/local/race_analysis/<race_id>.json。gcs_mirror の TREES に載っているので、ホスト（または取得役）が保存すると Cloud Run に届く。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]


def analysis_dir() -> Path:
    return Path(os.environ.get("KEIBA_RACE_ANALYSIS_DIR") or REPO / "data/local/race_analysis")


def path_of(race_id: str) -> Path:
    return analysis_dir() / f"{race_id}.json"


def read(race_id: str) -> dict[str, Any] | None:
    try:
        return json.loads(path_of(race_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write(race_id: str, payload: dict[str, Any]) -> None:
    p = path_of(race_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, p)


def prune(keep_days: int, now: float | None = None, keep_year: int | None = None) -> int:
    """古い分析結果を消す（更新時刻が keep_days より前のもの）。keep_year の年のレース（race_id の先頭4桁）は古くても残す。消した件数を返す。"""
    cutoff = (now or time.time()) - keep_days * 86400
    n = 0
    for f in analysis_dir().glob("*.json"):
        if keep_year is not None and f.stem.startswith(str(keep_year)):
            continue
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                n += 1
        except OSError:
            continue
    return n

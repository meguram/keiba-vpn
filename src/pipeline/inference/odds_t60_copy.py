"""T-60 の単勝オッズの、ホストの手元の写し（第 2 段の影のランナーが読む）。

raceday-runner が race_odds を phase=t60 で保存した直後に、単勝オッズと人気だけを
``<推論の作業場所>/odds_t60/<YYYYMMDD>/<race_id>.json`` に書く（一時ファイル → rename）。
影のランナーはこれだけを読み、GCS を読みに行かない（GCP の外から GCS を繰り返し読まない: CLAUDE.md）。

既定はオフ（``KEIBA_ODDS_T60_LOCAL_COPY=1`` か目印 ``data/local/meta/odds_t60_copy.on`` があるときだけ書く。目印は
``scripts/cron/install_candidate_runner_cron.sh`` が置く）。書き込みの失敗は警告だけで、
オッズの保存やその後の取得には一切響かせない（呼び出し側は戻り値を見ない）。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))


ENABLE_MARK = Path(__file__).resolve().parents[3] / "data/local/meta/odds_t60_copy.on"


def enabled() -> bool:
    """KEIBA_ODDS_T60_LOCAL_COPY=1 か、目印 data/local/meta/odds_t60_copy.on があるとき（追跡されるファイルを書き換えずに有効にする）。"""
    if os.environ.get("KEIBA_ODDS_T60_LOCAL_COPY", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return ENABLE_MARK.exists() and os.environ.get("KEIBA_ODDS_T60_LOCAL_COPY", "").strip() != "0"


def base_dir() -> Path:
    """写しの置き場（本体の推論の作業場所の下。影のランナーが別の作業場所でも、ここを読む）。"""
    env = os.environ.get("KEIBA_ODDS_T60_DIR")
    if env:
        return Path(env)
    from src.pipeline.inference.weight_triggered_predict import inference_dir

    return inference_dir() / "odds_t60"


def _day_of(doc: dict[str, Any]) -> str:
    s = str(doc.get("snapshot_at") or "")
    try:
        return datetime.fromisoformat(s).astimezone(_JST).strftime("%Y%m%d")
    except ValueError:
        return datetime.now(_JST).strftime("%Y%m%d")


def payload(race_id: str, doc: dict[str, Any]) -> dict[str, Any]:
    ents = [{"horse_number": int(e.get("horse_number") or 0), "win_odds": e.get("win_odds"), "popularity": e.get("popularity")}
            for e in (doc.get("entries") or []) if int(e.get("horse_number") or 0) > 0]
    return {"race_id": str(race_id), "phase": "t60", "snapshot_at": doc.get("snapshot_at"),
            "written_at": datetime.now(_JST).isoformat(timespec="seconds"), "entries": ents}


def write_copy(race_id: str, doc: dict[str, Any], phase: str | None) -> Path | None:
    """phase=t60 のときだけ写しを書く。失敗しても例外を出さない（None を返す）。"""
    if phase != "t60" or not enabled():
        return None
    try:
        out = base_dir() / _day_of(doc) / f"{race_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.tmp")
        tmp.write_text(json.dumps(payload(race_id, doc), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, out)
        return out
    except Exception as exc:  # noqa: BLE001  写しの失敗で本体の取得を止めない
        logger.warning("T-60 オッズの手元の写しを書けない（取得には影響なし）: %s %s", race_id, exc)
        return None


def read_copy(day: str, race_id: str, root: Path | None = None) -> dict[str, Any] | None:
    """写しを読む（無ければ None）。day は YYYYMMDD。"""
    p = (root or base_dir()) / day / f"{race_id}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

"""tp-55: ハロンごとのラップ予想（ラップ傾向別のタイム＋各傾向の発生確率）を保存・読み出し、予測 API に付ける。

有効化は環境変数 KEIBA_LAP_FORECAST=1（既定は無効＝何もしない）。保存先は data/local/inference/lap_forecast/<日付>/<race_id>.json
（KEIBA_LAP_FORECAST_DIR で変更可）。保存は `python -m src.scripts.data.tp55_backfill`（出馬表が書き出された日の全レース）。

lap_forecast の形:
  version, race_id, date, distance, n_laps, lap_meters（最初の区間が 200m に満たないことがある）, model（A_full=テン1Fあり／A_noten=なし）,
  pace_label（もっとも起こりやすい傾向）, scenarios[{name, prob, laps, sd, first3f, last3f}], expected_laps, band80_lo, band80_hi,
  expected_first3f, expected_last3f, par（コース別の基準ラップ）
laps は『スタートから 1 区間ずつの秒』（最初が『テン1F』）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ENV_FLAG = "KEIBA_LAP_FORECAST"
ENV_DIR = "KEIBA_LAP_FORECAST_DIR"
VERSION = "tp55-1"
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "local" / "inference" / "lap_forecast"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0").strip().lower() in ("1", "true", "yes", "on")


def base_dir() -> Path:
    return Path(os.environ.get(ENV_DIR) or DEFAULT_DIR)


def lap_meters(distance: int) -> list[int]:
    n = -(-int(distance) // 200)
    d0 = int(distance) - 200 * (n - 1)
    return [d0] + [200] * (n - 1)


def path_for(race_id: str, date: str) -> Path:
    return base_dir() / str(date)[:10].replace("/", "-") / f"{race_id}.json"


def save_forecast(doc: dict) -> Path:
    p = path_for(doc["race_id"], doc["date"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return p


def load_forecast(race_id: str, date: str) -> dict | None:
    p = path_for(race_id, date)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def attach_lap_forecast(resp: dict, race_id: str, date: str) -> dict:
    """予測 API の応答に lap_forecast を足す。無効・保存なし・失敗のときは応答をそのまま返す。"""
    if not enabled() or not isinstance(resp, dict) or not date:
        return resp
    doc = load_forecast(race_id, date)
    if not doc:
        return resp
    return {**resp, "lap_forecast": doc}


def make_doc(race_id: str, date: str, distance: int, summary: dict[str, Any], model: str, par: list[float | None]) -> dict:
    return {"version": VERSION, "race_id": race_id, "date": date, "distance": int(distance), "n_laps": summary["n_laps"],
            "lap_meters": lap_meters(distance), "model": model, "pace_label": summary["likely_scenario"], "pace_prob": summary["likely_prob"],
            "scenarios": summary["scenarios"], "expected_laps": summary["expected_laps"], "band80_lo": summary["band80_lo"],
            "band80_hi": summary["band80_hi"], "expected_first3f": summary["expected_first3f"], "expected_last3f": summary["expected_last3f"],
            "par": par}

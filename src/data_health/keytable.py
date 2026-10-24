"""レースのキーテーブル: 「いつの・どの場の・何Rの・何というレース」が、どの race_id か。

race_id は「年＋場＋開催回＋日目＋R」で暦日を含まないため、日付との対応表を別に持つ。
情報源（上ほど優先）: race_lists（開催日ごとの一覧）→ 取得済み JSON の中身（出馬表・結果の date / venue / round / race_name）。
全件検証（validate=full）では各 JSON をどのみち読むので、その副産物として表を育てる。
表は環境ごとのローカル（``<環境>/ledger/race_keys.json``）に保存し、``race_keys.csv`` にも出力する。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
         "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
META_FIELDS = ("date", "venue", "round", "race_name", "grade", "surface", "distance", "field_size", "start_time")
# JSON の中身から日付などを読み取れるカテゴリ
HARVEST_CATEGORIES = ("race_shutuba", "race_result", "race_result_on_time", "race_shutuba_meta", "race_result_meta")


def decode(race_id: str) -> dict[str, Any]:
    """race_id から分かること（年・場・開催回・日目・R）。暦日は含まれない。"""
    if len(race_id) != 12 or not race_id.isdigit():
        return {}
    return {"year": race_id[:4], "venue_code": race_id[4:6], "venue_from_id": VENUE.get(race_id[4:6], race_id[4:6]),
            "kaisai": int(race_id[6:8]), "day": int(race_id[8:10]), "round_from_id": int(race_id[10:12])}


def _ymd(v: Any) -> str | None:
    s = str(v or "").strip()
    m = re.fullmatch(r"(\d{4})[-/]?(\d{2})[-/]?(\d{2})", s)
    return "".join(m.groups()) if m else None


def extract_meta(data: dict[str, Any]) -> dict[str, Any]:
    """取得済み JSON からキーテーブルに載せる項目を取り出す。"""
    out: dict[str, Any] = {}
    d = _ymd(data.get("date"))
    if d:
        out["date"] = d
    for k in ("venue", "race_name", "grade", "surface", "start_time"):
        if data.get(k):
            out[k] = str(data[k])
    for k in ("round", "distance", "field_size"):
        try:
            if data.get(k) not in (None, "", 0):
                out[k] = int(data[k])
        except (TypeError, ValueError):
            pass
    return out


class KeyTable:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.rows: dict[str, dict[str, Any]] = {}
        if path:
            try:
                self.rows = json.loads(path.read_text(encoding="utf-8")).get("races", {})
            except (OSError, ValueError):
                pass

    def update(self, race_id: str, meta: dict[str, Any], source: str) -> None:
        """source の優先: race_lists > data（取得済み JSON）。既に race_lists 由来の値は data で上書きしない。"""
        row = self.rows.setdefault(race_id, {})
        for k, v in meta.items():
            if v in (None, ""):
                continue
            if k not in row or (source == "race_lists" and row.get("_src", {}).get(k) != "race_lists"):
                row[k] = v
                row.setdefault("_src", {})[k] = source

    def update_from_calendar(self, cal_dates: dict[str, list[str]], race_meta: dict[str, dict[str, Any]] | None = None) -> None:
        for date, ids in cal_dates.items():
            for rid in ids:
                self.update(rid, {"date": date, **((race_meta or {}).get(rid, {}))}, "race_lists")

    def get(self, race_id: str) -> dict[str, Any]:
        return self.rows.get(race_id, {})

    def date_of(self, race_id: str) -> str | None:
        return self.rows.get(race_id, {}).get("date")

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"races": self.rows}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def csv_table(universe: dict[str, Any], table: KeyTable, categories: list[str], status: dict[str, dict[str, str]]
              ) -> tuple[list[str], list[list[Any]]]:
    """race_id 1 行 = 1 レース。開催日・場・R・レース名と、カテゴリごとの状態（healthy/invalid/missing/…）。"""
    cols = ["race_id", "date", "venue", "round", "race_name", "grade", "surface", "distance", "date_source", "key_confidence"] + categories
    rows = []
    for rid in sorted(universe):
        race, info, dec = universe[rid], table.get(rid), decode(rid)
        src = (info.get("_src") or {}).get("date", "")
        rows.append([rid, race.date or "", info.get("venue") or dec.get("venue_from_id", ""),
                     info.get("round") or dec.get("round_from_id", ""), info.get("race_name", ""), info.get("grade", ""),
                     info.get("surface", ""), info.get("distance", ""), src, race.confidence]
                    + [status.get(rid, {}).get(c, "") for c in categories])
    return cols, rows


def write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:         # Excel で文字化けしない BOM 付き
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)

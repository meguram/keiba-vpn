"""JRA 馬場情報 PDF（含水率・クッション値）を、テキストの並びから読む補完パーサ。

既存の ``jra_cushion``（表の抽出）が取りこぼす分を埋める用途:
  * 含水率だけが載る PDF（2018〜2020 年 9 月まで。クッション値の公表前）
  * 表の抽出で含水率が空になる開催（2021〜2024 年に 24 日ほど）
  * 日付の割り当て: 曜日の列名（金・土・日…）を、日付範囲の中の同じ曜日に当てる（土曜が無い「金・日・月」にも対応）

画像だけの PDF（2021 年 第6回中京）は、``data/meta/cushion_manual/*.json`` に手で書き起こした値を使う。
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

WEEKDAY_NAMES = ["月", "火", "水", "木", "金", "土", "日"]
_WD = re.compile(r"^([月火水木金土日])曜日$")
_NUM = re.compile(r"^\d{1,2}(?:\.\d+)?$")
_MISSING = {"－", "-", "―", "—", "ー", "×", "−"}
_HEAD_B = re.compile(r"[（(](\d{4})年(\d{1,2})月(\d{1,2})日?[～〜~](?:(\d{1,2})月)?(\d{1,2})日?[）)]")
_HEAD_C = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日から(?:(\d{1,2})月)?(\d{1,2})日の(?:含水率|気象状況)")
_FIELDS = {("turf", "goal"): "turf_moisture_goal", ("turf", "4c"): "turf_moisture_4corner", ("dirt", "goal"): "dirt_moisture_goal", ("dirt", "4c"): "dirt_moisture_4corner"}


def _tokens(text: str) -> list[str]:
    return [t.strip() for t in text.replace("　", " ").split("\n") if t.strip()]


def _date_range(m: re.Match) -> tuple[dt.date, dt.date]:
    y, m1, d1 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    m2 = int(m.group(4)) if m.group(4) else m1
    d2 = int(m.group(5))
    return dt.date(y, m1, d1), dt.date(y, m2, d2)


def _date_for_weekday(start: dt.date, end: dt.date, wd_char: str) -> dt.date | None:
    idx = WEEKDAY_NAMES.index(wd_char)
    d = start
    while d <= end:
        if d.weekday() == idx:
            return d
        d += dt.timedelta(days=1)
    return None


def parse_sections(tokens: list[str]) -> list[dict[str, Any]]:
    """見出し（日付範囲）ごとに、曜日列・クッション値・含水率を読む。"""
    sections: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for tok in tokens:
        m = _HEAD_B.search(tok) or _HEAD_C.search(tok)
        if m:
            start, end = _date_range(m)
            cur = {"start": start, "end": end, "cols_a": [], "cols_b": [], "cushion": [], "stage": "head", "bucket": None, "course": None, "vals": {}}
            sections.append(cur)
            continue
        if cur is None:
            continue
        w = _WD.match(tok)
        if tok == "場所":
            cur["stage"] = "place"
        elif w:
            if cur["stage"] == "place" and cur["vals"]:            # 含水率の表より後ろにクッション値の表が来る並び（2020 年など）
                cur["stage"] = "cushion_after"
            if cur["stage"] == "cushion_after":
                pass
            else:
                (cur["cols_b"] if cur["stage"] == "place" else cur["cols_a"]).append(w.group(1))
            cur["bucket"] = None
        elif "芝コース含水率" in tok:
            cur["course"] = "turf"; cur["bucket"] = None
        elif "ダートコース含水率" in tok:
            cur["course"] = "dirt"; cur["bucket"] = None
        elif tok in ("ゴール前", "４コーナー", "4コーナー", "４コーナ", "4コーナ"):
            if cur["course"]:
                cur["bucket"] = (cur["course"], "goal" if tok == "ゴール前" else "4c")
                cur["vals"].setdefault(cur["bucket"], [])
        elif _NUM.match(tok) or tok in _MISSING:
            v = float(tok) if _NUM.match(tok) else None
            if cur["bucket"] is not None:
                cur["vals"][cur["bucket"]].append(v)
            elif (cur["stage"] == "head" and cur["cols_a"]) or cur["stage"] == "cushion_after":
                cur["cushion"].append(v)
    return sections


def sections_to_records(sections: list[dict[str, Any]], link: dict, *, warnings: list[str] | None = None) -> list[dict[str, Any]]:
    """link: year・venue_code・venue_name・kai。曜日の列名を日付に当てて、1 日 1 行にする。"""
    warnings = warnings if warnings is not None else []
    out: list[dict[str, Any]] = []
    for si, s in enumerate(sections):
        cols = s["cols_b"] or s["cols_a"]
        n = len(cols)
        if n == 0:
            warnings.append(f"{link.get('year')} {link.get('venue_name')} 第{link.get('kai')}回 区間{si}: 曜日の列が読めない")
            continue
        has_cushion = len(s["cushion"]) == n
        for j, wd in enumerate(cols):
            d = _date_for_weekday(s["start"], s["end"], wd)
            if d is None:
                warnings.append(f"{link.get('year')} {link.get('venue_name')} 第{link.get('kai')}回 区間{si}: {wd}曜日が日付範囲に無い")
                continue
            rec: dict[str, Any] = {
                "year": int(link["year"]), "venue_code": link["venue_code"], "venue_name": link["venue_name"], "kai": int(link["kai"]), "pair_index": si,
                "weekday": f"{wd}曜日", "cushion_value": s["cushion"][j] if has_cushion else None,
                "is_race_day": wd != "金", "race_day": 0, "date": d.isoformat(), "course_position": "", "_src": "pdf_text",
            }
            for key, field in _FIELDS.items():
                arr = s["vals"].get(key, [])
                rec[field] = arr[j] if len(arr) == n and j < len(arr) else None
                if key in s["vals"] and len(arr) != n:
                    warnings.append(f"{link.get('year')} {link.get('venue_name')} 第{link.get('kai')}回 区間{si}: {field} の数が列数と違う（{len(arr)} / {n}）")
            out.append(rec)
    return out


def parse_pdf_text(path: Path, link: dict, *, warnings: list[str] | None = None) -> list[dict[str, Any]]:
    """PDF のテキストから 1 日 1 行のレコードを作る。2025 年以降の形式（測定月日の表）は対象外（空を返す）。"""
    import fitz

    doc = fitz.open(str(path))
    text = "\n".join(p.get_text() for p in doc)
    doc.close()
    if "測定月日" in text or "開催日次" in text:
        return []
    return sections_to_records(parse_sections(_tokens(text)), link, warnings=warnings)


def load_manual(manual_dir: Path) -> list[dict[str, Any]]:
    """画像だけの PDF を手で書き起こした値（``data/meta/cushion_manual/*.json``）。"""
    rows: list[dict[str, Any]] = []
    for p in sorted(Path(manual_dir).glob("*.json")):
        for r in json.loads(p.read_text(encoding="utf-8")).get("records", []):
            r = dict(r)
            r.setdefault("_src", "manual_transcription")
            rows.append(r)
    return rows


MOISTURE_FIELDS = ["turf_moisture_goal", "turf_moisture_4corner", "dirt_moisture_goal", "dirt_moisture_4corner"]


def merge_missing(existing: list[dict[str, Any]], incoming: list[dict[str, Any]], tol: float = 0.05, *, overwrite: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """既存の行のうち空の項目を incoming の値で埋め、無い (日付, 競馬場) は追加する。
    overwrite=False（既定）: 既存の値は上書きせず、食い違い（tol 超）は mismatches に記録するだけ。
    overwrite=True: incoming の値（None 以外）で既存を直し、直した内容を corrections に記録する（表の抽出で 4 コーナーが入れ替わっていた分の訂正に使う）。"""
    idx = {(r.get("date"), r.get("venue_code")): i for i, r in enumerate(existing)}
    rows = [dict(r) for r in existing]
    stats: dict[str, Any] = {"added": 0, "filled_fields": 0, "mismatches": [], "agree_fields": 0, "corrected_fields": 0, "corrections": []}
    for r in incoming:
        key = (r.get("date"), r.get("venue_code"))
        if key not in idx:
            rows.append({k: v for k, v in r.items() if not k.startswith("_")})
            idx[key] = len(rows) - 1
            stats["added"] += 1
            continue
        cur = rows[idx[key]]
        for f in ["cushion_value"] + MOISTURE_FIELDS:
            nv = r.get(f)
            if nv is None:
                continue
            ov = cur.get(f)
            if ov is None:
                cur[f] = nv
                stats["filled_fields"] += 1
            elif abs(float(ov) - float(nv)) > tol:
                rec = {"date": key[0], "venue_code": key[1], "field": f, "existing": ov, "pdf_text": nv}
                if overwrite:
                    cur[f] = nv
                    stats["corrected_fields"] += 1
                    stats["corrections"].append(rec)
                else:
                    stats["mismatches"].append(rec)
            else:
                stats["agree_fields"] += 1
    rows.sort(key=lambda x: (x.get("date") or "", x.get("venue_code") or ""))
    return rows, stats

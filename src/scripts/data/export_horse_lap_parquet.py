"""馬別 200m ラップ（backfill_horse_lap の JSONL）を、分析用の Parquet にまとめる（ローカルのみ・取得はしない）。

python -m src.scripts.data.export_horse_lap_parquet
python -m src.scripts.data.export_horse_lap_parquet --src /home/jovyan/acq/horse_lap /home/jovyan/acq/horse_lap23 --out-dir data/local/horse_lap

出力（--out-dir、既定 data/local/horse_lap）:
- horse_lap_sections.parquet : 1 行 = 1 頭 × 1 区間（race_id, horse_id, distance_m, corner, lap_time, position ほか）
- horse_lap_entries.parquet  : 1 行 = 1 頭 1 走（着順・馬番・指数 4 種・区間数・区間タイムの合計・ラップの短評）
- horse_lap_races.parquet    : 1 行 = 1 レース（区間の見出し・頭数・空ページか）
- manifest.json              : 元の JSONL・件数・作成日時

元の JSONL の場所は取得のたびに伸びるので、取得のあとにもう一度実行すれば全体を作り直す（数十秒）。
sections の position・lap_time は、元データの 0（載っていない）を欠損にする。区間ごとの位置は 2023〜2024 年にはほとんど載っていない。
同じ race_id が複数ファイルにあれば、後に読んだ方（--src の後ろ・年の新しいファイル）を採る。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = ["/home/jovyan/acq/horse_lap23", "/home/jovyan/acq/horse_lap"]
DEFAULT_OUT = ROOT / "data" / "local" / "horse_lap"


def load_races(src_dirs: list[Path]) -> tuple[dict[str, dict], list[str]]:
    races: dict[str, dict] = {}
    files: list[str] = []
    for d in src_dirs:
        for f in sorted(d.glob("horse_lap_*.jsonl")):
            files.append(str(f))
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("race_id"):
                        races[str(r["race_id"])] = r
    return races, files


def build(races: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    race_rows, entry_rows, sec_rows = [], [], []
    for rid, r in races.items():
        cols = r.get("section_columns") or []
        entries = r.get("entries") or []
        race_rows.append({
            "race_id": rid,
            "year": int(rid[:4]),
            "n_entries": len(entries),
            "n_finishers": sum(1 for e in entries if (e.get("finish_order") or 0) > 0),
            "expected_finishers": r.get("expected"),
            "n_sections": len(cols),
            "last_distance_m": cols[-1].get("distance_m") if cols else None,
            "section_headers": "|".join(str(c.get("header", "")) for c in cols),
            "empty_ok": bool(r.get("empty_ok", False)),
        })
        for e in entries:
            secs = e.get("sections") or []
            times = [s.get("lap_time") for s in secs if isinstance(s.get("lap_time"), (int, float))]
            entry_rows.append({
                "race_id": rid,
                "year": int(rid[:4]),
                "horse_id": str(e.get("horse_id") or ""),
                "horse_name": e.get("horse_name"),
                "horse_number": e.get("horse_number"),
                "finish_order": e.get("finish_order"),
                "index_total": e.get("index_total"),
                "index_start": e.get("index_start"),
                "index_chase": e.get("index_chase"),
                "index_closing": e.get("index_closing"),
                "n_sections": len(secs),
                "sum_lap_time": round(sum(times), 2) if times else None,
                "lap_comment": e.get("lap_comment"),
            })
            for i, s in enumerate(secs):
                sec_rows.append({
                    "race_id": rid,
                    "horse_id": str(e.get("horse_id") or ""),
                    "horse_number": e.get("horse_number"),
                    "finish_order": e.get("finish_order"),
                    "section_no": i + 1,
                    "distance_m": s.get("distance_m"),
                    "header": s.get("header"),
                    "corner": s.get("corner") or None,
                    "lap_time": s.get("lap_time"),
                    "position": s.get("position"),
                })
    races_df = pd.DataFrame(race_rows).sort_values("race_id", ignore_index=True)
    entries_df = pd.DataFrame(entry_rows).sort_values(["race_id", "horse_number"], ignore_index=True)
    secs_df = pd.DataFrame(sec_rows).sort_values(["race_id", "horse_number", "section_no"], ignore_index=True)
    for df, cols in [(entries_df, ["horse_number", "finish_order", "index_total", "index_start", "index_chase", "index_closing"]),
                     (secs_df, ["horse_number", "finish_order", "distance_m", "position"])]:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")
    secs_df["lap_time"] = pd.to_numeric(secs_df["lap_time"], errors="coerce").astype("float32")
    # 元データでは「載っていない」を 0 で表している（位置は 2023 年のほぼ全部・2024 年の約 87% が 0）。0 は欠損にする
    secs_df.loc[secs_df["position"] == 0, "position"] = pd.NA
    secs_df.loc[secs_df["lap_time"] <= 0, "lap_time"] = float("nan")
    return races_df, entries_df, secs_df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", default=DEFAULT_SRC, help="horse_lap_<年>.jsonl のあるフォルダ（後ろほど優先）")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a = ap.parse_args()

    races, files = load_races([Path(p) for p in a.src])
    if not races:
        print("JSONL が見つかりません:", a.src)
        return 1
    races_df, entries_df, secs_df = build(races)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in [("horse_lap_races", races_df), ("horse_lap_entries", entries_df), ("horse_lap_sections", secs_df)]:
        tmp = out / f"{name}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        tmp.replace(out / f"{name}.parquet")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sources": files,
        "races": len(races_df),
        "entries": len(entries_df),
        "sections": len(secs_df),
        "races_by_year": {str(k): int(v) for k, v in races_df.groupby("year").size().items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
